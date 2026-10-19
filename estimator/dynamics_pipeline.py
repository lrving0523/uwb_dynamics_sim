"""用同一份因果 IMU/UWB/RPM 观测比较物理与训练残余动力学。"""

from collections import deque
from copy import deepcopy
from pathlib import Path
import time as timer

import numpy as np
import yaml

from estimator.eskf import NominalState
from estimator.physics_ekf import PhysicsEKF
from estimator.hybrid_ekf import HybridEKF
from estimator.pipeline import (_append, _snapshot, _validate_measurements,
                                initial_covariance, load_estimation_settings, validate_settings)
from estimator.uwb_ls import solve_uwb_position
from simulator.math3d import normalize_quaternion, rotation_matrix


def load_dynamics_settings(estimation_path, hybrid_path, simulation_config):
    """提取独立模型设置和标定噪声，不拷贝仿真真参数或真实偏置。"""
    settings = load_estimation_settings(estimation_path, simulation_config)
    with Path(hybrid_path).open(encoding='utf-8-sig') as stream:
        hybrid_config = yaml.safe_load(stream)
    settings['noise']['rpm_std'] = float(simulation_config['sensors']['rpm']['std'])
    settings['noise']['model_force_std'] = hybrid_config['filter']['model_force_std']
    settings['sequence_length'] = int(hybrid_config['network']['sequence_length'])
    settings['input_hz'] = float(hybrid_config['network']['input_hz'])
    return settings, deepcopy(hybrid_config['physics'])


def _validate(measurements, settings):
    if set(measurements) != {'imu', 'uwb', 'rpm'}:
        raise ValueError('动力学估计仅接受 imu/uwb/rpm 观测，不能接收真值')
    base_settings = deepcopy(settings)
    base_settings['noise'] = {key: settings['noise'][key] for key in
                              ['acc_std', 'gyro_std', 'acc_bias_rw_std', 'gyro_bias_rw_std']}
    validate_settings(base_settings)
    values = _validate_measurements({key: measurements[key] for key in ['imu', 'uwb']})
    times = np.asarray(measurements['rpm']['time'], dtype=float)
    rpm = np.asarray(measurements['rpm']['rpm'], dtype=float)
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)) or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError('RPM 时间戳必须非空、有限、非负且严格递增')
    if rpm.shape != (len(times), 4) or not np.all(np.isfinite(rpm)):
        raise ValueError('RPM 观测必须为有限 N×4 数组')
    hz = settings['input_hz']
    if not np.isfinite(hz) or hz <= 0 or settings['sequence_length'] < 1:
        raise ValueError('残余模型输入频率与窗口长度必须为正')
    if len(times) > 1 and not np.allclose(np.diff(times), 1/hz, atol=1e-12, rtol=1e-9):
        raise ValueError(f'本阶段混合模型要求 RPM 样本频率等于训练输入频率 {hz:g} Hz')
    values['rpm'] = {'time': times, 'rpm': rpm}
    return values


def _empty_diagnostics(records, width):
    if records:
        return {key: np.asarray(value) for key, value in records.items()}
    return {'time': np.empty(0), 'innovation': np.empty((0, width)),
            'innovation_covariance': np.empty((0, width, width)), 'nis': np.empty(0)}


def run_dynamics_estimators(measurements, settings, physics_model, hybrid_model):
    """先传播过去区间，再接收新陀螺/RPM并作测距和比力更新。

    正式学习对比由实验入口核验训练证据；本接口也允许零残余做算法消融。
    本阶段 TCN 输入采用实际 100 Hz RPM 事件，不把保持的 RPM 伪造成独立样本。
    """
    values = _validate(measurements, settings)
    imu, uwb, rpm = values['imu'], values['uwb'], values['rpm']
    length = settings['sequence_length']
    if hybrid_model.predictor.sequence_length != length or hybrid_model.predictor.input_hz != settings['input_hz']:
        raise ValueError('TCN checkpoint 的输入频率/窗口与估计设置不匹配')
    start = float(uwb['time'][0])
    imu_index = int(np.searchsorted(imu['time'], start, side='right'))-1
    rpm_index = int(np.searchsorted(rpm['time'], start, side='right'))-1
    if imu_index < 0:
        raise ValueError('初始化前没有已到达的 IMU')
    if rpm_index < 0:
        raise ValueError('初始化前没有已到达的 RPM')
    names = ['physics', 'hybrid']
    results = {name: {} for name in names}
    results.update(uwb_diagnostics={name: {} for name in names},
                   accel_diagnostics={name: {} for name in names}, residual={}, timing={})
    window = deque(maxlen=length)
    inference_seconds = []
    hybrid_model.update_residual(np.empty((0, 7)))

    def receive_feature(index):
        stamp = float(rpm['time'][index])
        gyro_index = int(np.searchsorted(imu['time'], stamp, side='right'))-1
        if gyro_index < 0:
            return
        window.append(np.r_[imu['gyro'][gyro_index], rpm['rpm'][index]])
        tick = timer.perf_counter()
        residual = hybrid_model.update_residual(np.asarray(window))
        inference_seconds.append(timer.perf_counter()-tick)
        _append(results['residual'], {'time': stamp, 'force_body': residual, 'ready': len(window) == length})

    for index in range(rpm_index+1):
        receive_feature(index)
    solution = solve_uwb_position(uwb['anchors'], uwb['ranges'][0], settings['uwb_std'], **settings['ls'])
    if not solution['converged']:
        raise ValueError('初始化测距 LS 未收敛')
    initial = settings['initial']
    quaternion = normalize_quaternion(initial['quaternion'])
    lever_arm = uwb['lever_arm']
    covariance = initial_covariance(solution['covariance'], settings, lever_arm)
    filters = {}
    event_seconds = {name: [] for name in names}
    latest_imu_time, latest_rpm_time = float(imu['time'][imu_index]), float(rpm['time'][rpm_index])
    for name, cls, model in [('physics', PhysicsEKF, physics_model), ('hybrid', HybridEKF, hybrid_model)]:
        tick = timer.perf_counter()
        state = NominalState(solution['position']-rotation_matrix(quaternion)@lever_arm,
                             initial['velocity'], quaternion, initial['acc_bias'], initial['gyro_bias'])
        filter_ = cls(state, covariance, model, settings['noise'], initial_time=start)
        filter_.set_gyro(start, imu['gyro'][imu_index])
        filter_.set_rpm(start, rpm['rpm'][rpm_index])
        if latest_imu_time == start:
            diagnostics = filter_.update_accel(imu['accel'][imu_index])
            _append(results['accel_diagnostics'][name], {'time': start, **diagnostics})
        filters[name] = filter_
        snapshot = _snapshot(filter_, lever_arm, latest_imu_time)
        snapshot['latest_rpm_time'] = latest_rpm_time
        _append(results[name], snapshot)
        event_seconds[name].append(timer.perf_counter()-tick)
    imu_index += 1
    rpm_index += 1
    uwb_index = 1
    events = np.unique(np.concatenate([imu['time'][imu_index:], rpm['time'][rpm_index:], uwb['time'][1:]]))
    for current in events:
        spent = {}
        for name, filter_ in filters.items():
            tick = timer.perf_counter()
            filter_.propagate_to(float(current))
            spent[name] = timer.perf_counter()-tick
        new_imu = imu_index < len(imu['time']) and imu['time'][imu_index] == current
        new_rpm = rpm_index < len(rpm['time']) and rpm['time'][rpm_index] == current
        new_uwb = uwb_index < len(uwb['time']) and uwb['time'][uwb_index] == current
        if new_imu:
            for name, filter_ in filters.items():
                tick = timer.perf_counter()
                filter_.set_gyro(float(current), imu['gyro'][imu_index])
                spent[name] += timer.perf_counter()-tick
            latest_imu_time = float(current)
        if new_rpm:
            receive_feature(rpm_index)
            for name, filter_ in filters.items():
                tick = timer.perf_counter()
                filter_.set_rpm(float(current), rpm['rpm'][rpm_index])
                spent[name] += timer.perf_counter()-tick
            latest_rpm_time = float(current)
        for name, filter_ in filters.items():
            tick = timer.perf_counter()
            if new_uwb:
                diagnostic = filter_.update_uwb(uwb['ranges'][uwb_index], uwb['anchors'], lever_arm, settings['uwb_std'])
                _append(results['uwb_diagnostics'][name], {'time': current, **diagnostic})
            if new_imu:
                diagnostic = filter_.update_accel(imu['accel'][imu_index])
                _append(results['accel_diagnostics'][name], {'time': current, **diagnostic})
            spent[name] += timer.perf_counter()-tick
            snapshot = _snapshot(filter_, lever_arm, latest_imu_time)
            snapshot['latest_rpm_time'] = latest_rpm_time
            _append(results[name], snapshot)
            event_seconds[name].append(spent[name])
        imu_index += int(new_imu)
        rpm_index += int(new_rpm)
        uwb_index += int(new_uwb)
    for name in names+['residual']:
        results[name] = {key: np.asarray(value) for key, value in results[name].items()}
    for name in names:
        results['uwb_diagnostics'][name] = _empty_diagnostics(results['uwb_diagnostics'][name], len(uwb['anchors']))
        results['accel_diagnostics'][name] = _empty_diagnostics(results['accel_diagnostics'][name], 3)
    results['timing'] = {f'{name}_event_seconds': np.asarray(event_seconds[name]) for name in names}
    results['timing']['residual_inference_seconds'] = np.asarray(inference_seconds)
    return results
