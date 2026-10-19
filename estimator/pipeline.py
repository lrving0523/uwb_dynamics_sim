"""仅接收已产生的 IMU/UWB 观测，因果运行单点定位与 ESKF。"""

from copy import deepcopy
from pathlib import Path
import time as timer

import numpy as np
import yaml

from estimator.eskf import NominalState, UWBIMUESKF
from estimator.uwb_ls import solve_uwb_position
from simulator.math3d import normalize_quaternion, rotation_matrix, skew


def load_estimation_settings(path, simulation_config):
    """只提取传感器噪声标定值，禁止拷贝真实偏置或仿真初始状态。"""
    with Path(path).open(encoding='utf-8-sig') as stream:
        settings = yaml.safe_load(stream)
    if not isinstance(settings, dict):
        raise ValueError('估计设置必须为映射')
    settings = deepcopy(settings)
    settings['noise'] = {key: float(simulation_config['sensors']['imu'][key])
                         for key in ['acc_std', 'gyro_std', 'acc_bias_rw_std', 'gyro_bias_rw_std']}
    floor = float(settings['measurement_std_floor'])
    if not np.isfinite(floor) or floor <= 0:
        raise ValueError('固定测量标准差下限必须为有限正数')
    settings['uwb_std'] = max(float(simulation_config['sensors']['uwb']['std']), floor)
    validate_settings(settings)
    return settings


def _vector(value, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f'{name} 必须为有限三维向量')
    return array


def _std_vector(value, name):
    values = np.asarray(value, dtype=float)
    if values.shape == ():
        values = np.repeat(values, 3)
    values = _vector(values, name)
    if np.any(values < 0):
        raise ValueError(f'{name} 不可为负')
    return values


def validate_settings(settings):
    """先验和固定噪声只做合法性检查，不根据残差调整参数。"""
    for key in ['gravity', 'uwb_std']:
        if not np.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'{key} 必须为有限正数')
    initial = settings['initial']
    normalize_quaternion(initial['quaternion'])
    for key in ['velocity', 'acc_bias', 'gyro_bias']:
        _vector(initial[key], key)
    for key in ['velocity_std', 'attitude_std', 'acc_bias_std', 'gyro_bias_std']:
        _std_vector(initial[key], key)
    for key, value in settings['noise'].items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f'IMU {key} 必须有限且非负')
    ls = settings['ls']
    if isinstance(ls['max_iterations'], bool) or not isinstance(ls['max_iterations'], int) or ls['max_iterations'] < 1:
        raise ValueError('LS 最大迭代次数必须为正整数')
    if not np.isfinite(ls['tolerance']) or ls['tolerance'] <= 0:
        raise ValueError('LS 收敛阈值必须为有限正数')


def initial_covariance(tag_covariance, settings, lever_arm):
    """天线位置减去旋转杆臂时，保留机体位置与姿态的相关性。"""
    initial = settings['initial']
    covariance = np.zeros((15, 15))
    for start, key in [(3, 'velocity_std'), (6, 'attitude_std'),
                       (9, 'acc_bias_std'), (12, 'gyro_bias_std')]:
        covariance[start:start+3, start:start+3] = np.diag(_std_vector(initial[key], key)**2)
    jacobian = rotation_matrix(initial['quaternion']) @ skew(lever_arm)
    orientation_covariance = covariance[6:9, 6:9]
    covariance[:3, :3] = np.asarray(tag_covariance) + jacobian @ orientation_covariance @ jacobian.T
    covariance[:3, 6:9] = jacobian @ orientation_covariance
    covariance[6:9, :3] = covariance[:3, 6:9].T
    return covariance


def _validate_measurements(measurements):
    if set(measurements) != {'imu', 'uwb'}:
        raise ValueError('在线估计仅接受 imu 与 uwb 观测包；真值必须留在离线评价层')
    result = {}
    for name, widths in [('imu', {'accel': 3, 'gyro': 3}), ('uwb', {})]:
        source = measurements[name]
        times = np.asarray(source['time'], dtype=float)
        if times.ndim != 1 or len(times) < 1 or not np.all(np.isfinite(times)) or times[0] < 0 or np.any(np.diff(times) <= 0):
            raise ValueError(f'{name} 时间戳必须有限、非负、非空且严格递增')
        result[name] = {'time': times}
        for key, width in widths.items():
            values = np.asarray(source[key], dtype=float)
            if values.shape != (len(times), width) or not np.all(np.isfinite(values)):
                raise ValueError(f'{name}.{key} 的观测形状或数值非法')
            result[name][key] = values
    uwb = measurements['uwb']
    anchors = np.asarray(uwb['anchors'], dtype=float)
    ranges = np.asarray(uwb['ranges'], dtype=float)
    if anchors.ndim != 2 or anchors.shape[1:] != (3,) or len(anchors) < 4 or not np.all(np.isfinite(anchors)):
        raise ValueError('UWB 至少需要四个有限三维基站')
    if ranges.shape != (len(result['uwb']['time']), len(anchors)) or not np.all(np.isfinite(ranges)):
        raise ValueError('UWB 距离数组必须与时间、基站数量一致且有限')
    result['uwb'].update(anchors=anchors, ranges=ranges, lever_arm=_vector(uwb['lever_arm'], '天线杆臂'))
    return result


def _append(records, values):
    for name, value in values.items():
        records.setdefault(name, []).append(np.array(value, copy=True))


def _snapshot(filter_, lever_arm, latest_imu_time):
    state = filter_.state
    return {'time': filter_.time, 'position': state.position, 'velocity': state.velocity,
            'quaternion': state.quaternion, 'acc_bias': state.acc_bias, 'gyro_bias': state.gyro_bias,
            'covariance': filter_.P,
            'tag_position': state.position + rotation_matrix(state.quaternion) @ lever_arm,
            'latest_imu_time': latest_imu_time}


def run_estimators(measurements, settings):
    """首帧 LS 初始化后逐事件运行，不重用初始化距离，不访问未来 IMU。"""
    validate_settings(settings)
    values = _validate_measurements(measurements)
    imu, uwb = values['imu'], values['uwb']
    initial_time = float(uwb['time'][0])
    imu_index = int(np.searchsorted(imu['time'], initial_time, side='right'))-1
    if imu_index < 0:
        raise ValueError('首帧 UWB 之前没有已到达的 IMU，不能用未来 IMU 初始化')
    outputs = {'uwb_ls': {}, 'eskf': {}, 'diagnostics': {}, 'timing': {}}
    initial = settings['initial']
    anchors, lever_arm = uwb['anchors'], uwb['lever_arm']
    ls_seconds, event_seconds = [], []
    initialization_start = timer.perf_counter()
    tick = timer.perf_counter()
    solution = solve_uwb_position(anchors, uwb['ranges'][0], settings['uwb_std'], **settings['ls'])
    ls_seconds.append(timer.perf_counter()-tick)
    if not solution['converged']:
        raise ValueError('首帧 UWB LS 未收敛，无法初始化滤波器')
    prior_quaternion = normalize_quaternion(initial['quaternion'])
    state = NominalState(solution['position']-rotation_matrix(prior_quaternion)@lever_arm,
                         np.asarray(initial['velocity']), prior_quaternion,
                         np.asarray(initial['acc_bias']), np.asarray(initial['gyro_bias']))
    filter_ = UWBIMUESKF(state, initial_covariance(solution['covariance'], settings, lever_arm),
                        settings['gravity'], settings['noise'], initial_time=initial_time)
    # 初始化时仅装载当时已经收到的最近样本；该样本只驱动之后的区间。
    filter_.set_imu(initial_time, imu['accel'][imu_index], imu['gyro'][imu_index])
    latest_imu_time = float(imu['time'][imu_index])
    _append(outputs['uwb_ls'], {'time': initial_time, 'tag_position': solution['position'],
            'covariance': solution['covariance'], 'residual': solution['residual'],
            'iterations': solution['iterations'], 'converged': solution['converged']})
    _append(outputs['eskf'], _snapshot(filter_, lever_arm, latest_imu_time))
    initialization_seconds = timer.perf_counter()-initialization_start
    event_seconds.append(0.0)
    imu_index += 1
    uwb_index = 1
    events = np.unique(np.concatenate([imu['time'][imu_index:], uwb['time'][1:]]))
    for current_time in events:
        tick = timer.perf_counter()
        filter_.propagate_to(float(current_time))
        spent = timer.perf_counter()-tick
        if uwb_index < len(uwb['time']) and uwb['time'][uwb_index] == current_time:
            tick = timer.perf_counter()
            solution = solve_uwb_position(anchors, uwb['ranges'][uwb_index], settings['uwb_std'],
                                          initial=solution['position'], **settings['ls'])
            ls_seconds.append(timer.perf_counter()-tick)
            if not solution['converged']:
                raise ValueError(f'时刻 {current_time} 的 UWB LS 未收敛')
            _append(outputs['uwb_ls'], {'time': current_time, 'tag_position': solution['position'],
                    'covariance': solution['covariance'], 'residual': solution['residual'],
                    'iterations': solution['iterations'], 'converged': solution['converged']})
            tick = timer.perf_counter()
            diagnostic = filter_.update_uwb(uwb['ranges'][uwb_index], anchors, lever_arm, settings['uwb_std'])
            spent += timer.perf_counter()-tick
            _append(outputs['diagnostics'], {'time': current_time, **diagnostic})
            uwb_index += 1
        if imu_index < len(imu['time']) and imu['time'][imu_index] == current_time:
            tick = timer.perf_counter()
            filter_.set_imu(float(current_time), imu['accel'][imu_index], imu['gyro'][imu_index])
            spent += timer.perf_counter()-tick
            latest_imu_time = float(imu['time'][imu_index])
            imu_index += 1
        _append(outputs['eskf'], _snapshot(filter_, lever_arm, latest_imu_time))
        event_seconds.append(spent)
    if not outputs['diagnostics']:
        size = len(anchors)
        outputs['diagnostics'] = {'time': np.empty(0), 'innovation': np.empty((0, size)),
                                  'innovation_covariance': np.empty((0, size, size)), 'nis': np.empty(0)}
    for group in outputs.values():
        for key, value in group.items():
            group[key] = np.asarray(value)
    outputs['timing'] = {'ls_update_seconds': np.asarray(ls_seconds),
                          'eskf_event_seconds': np.asarray(event_seconds),
                          'initialization_seconds': np.asarray(initialization_seconds)}
    return outputs
