"""阶段二离线评价：精确关联事件真值，比较天线位置并描述单次诊断。"""

from __future__ import annotations

import numpy as np

from simulator.math3d import normalize_quaternion, quaternion_multiply, rotation_matrix


TIME_MATCH_ATOL = 1e-12
RPE_TIME_ATOL = 1e-9
MAX_COVARIANCE_CONDITION = 1e12


def _time(values, name: str, *, allow_empty=False) -> np.ndarray:
    """拒绝未排序、重复或非有限时钟，避免悄悄重排估计。"""
    time = np.asarray(values, dtype=float)
    if (time.ndim != 1 or (not allow_empty and not len(time))
            or not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0)):
        raise ValueError(f'{name} 时间轴必须有限且严格递增，并符合非空要求')
    return time


def match_timestamps(reference, query, *, reference_name='参考', query_name='查询',
                     atol=TIME_MATCH_ATOL) -> np.ndarray:
    """严格最近邻关联；允许舍入误差，拒绝越界、缺失时刻和重复匹配。

    此函数仅用于离线关联，不插值、不产生新的状态或观测。
    """
    reference = _time(reference, reference_name)
    query = _time(query, query_name, allow_empty=True)
    if not np.isfinite(atol) or atol < 0:
        raise ValueError('时间匹配容差必须有限且非负')
    if not len(query):
        return np.empty(0, dtype=int)
    if np.any(query < reference[0]) or np.any(query > reference[-1]):
        raise ValueError(f'{query_name} 时间超出 {reference_name} 已保存时刻范围')
    right = np.searchsorted(reference, query)
    upper = np.minimum(right, len(reference) - 1)
    lower = np.maximum(right - 1, 0)
    indices = np.where(np.abs(reference[lower] - query) <= np.abs(reference[upper] - query), lower, upper)
    mismatch = np.abs(reference[indices] - query) > atol
    if np.any(mismatch):
        missing = float(query[np.flatnonzero(mismatch)[0]])
        raise ValueError(f'{query_name} 时刻 {missing:.17g} 在 {reference_name} 中没有容差内匹配')
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f'{query_name} 时间分辨率导致多个样本匹配到同一 {reference_name} 时刻')
    return indices


def _array(group: dict, key: str, shape: tuple, name: str, *, finite=True) -> np.ndarray:
    """统一验证采样形状，协方差的数值有效性留给诊断逐帧检查。"""
    if key not in group:
        raise ValueError(f'缺少 {name}.{key}')
    values = np.asarray(group[key], dtype=float)
    if values.shape != shape or (finite and not np.all(np.isfinite(values))):
        raise ValueError(f'{name}.{key} 必须为形状 {shape} 的有限数组')
    return values


def _get_time(group: dict, name: str, *, allow_empty=False) -> np.ndarray:
    if 'time' not in group:
        raise ValueError(f'缺少 {name}.time 时间轴')
    return _time(group['time'], f'{name}.time', allow_empty=allow_empty)


def right_local_rotation_errors(estimated, truth) -> np.ndarray:
    """返回 Log(q_est⁻¹⊗q_true)，单位 rad，使用最短旋转及 Hamilton wxyz。"""
    estimated, truth = np.asarray(estimated, dtype=float), np.asarray(truth, dtype=float)
    if estimated.ndim != 2 or estimated.shape[1:] != (4,) or truth.shape != estimated.shape:
        raise ValueError('姿态必须是形状相同的 (样本数, 4) 四元数数组')
    result = np.empty((len(estimated), 3))
    for index, (nominal, actual) in enumerate(zip(estimated, truth)):
        nominal = normalize_quaternion(nominal)
        conjugate = nominal * np.array([1., -1., -1., -1.])
        relative = quaternion_multiply(conjugate, normalize_quaternion(actual))
        # 半周旋转的轴符号也规范化，使 q 与 -q 的诊断结果保持一致。
        if relative[0] < 0 or (relative[0] == 0 and relative[np.flatnonzero(relative)[0]] < 0):
            relative = -relative
        sine = np.linalg.norm(relative[1:])
        factor = 2. if sine < 1e-12 else 2 * np.arctan2(sine, relative[0]) / sine
        result[index] = factor * relative[1:]
    return result


def _covariance_problem(covariance: np.ndarray) -> str | None:
    """仅在有限、对称、正定且条件数合理时允许求解完整维数 NEES。"""
    if not np.all(np.isfinite(covariance)):
        return 'nonfinite'
    if not np.allclose(covariance, covariance.T, rtol=1e-9, atol=1e-12):
        return 'asymmetric'
    eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2)
    if eigenvalues[0] < 0:
        return 'indefinite'
    if eigenvalues[0] == 0:
        return 'singular'
    if eigenvalues[-1] / eigenvalues[0] > MAX_COVARIANCE_CONDITION:
        return 'ill_conditioned'
    return None


def estimation_errors(dataset: dict, estimates: dict) -> dict:
    """构造图与指标共用的离线误差数组；始终保留原始输入。

    天线误差采用全部共同 UWB 时刻；机体、速度、偏置和 NEES 采用全部
    ESKF 输出事件时刻。NEES 误差为真值减名义状态，姿态块为右局部误差。
    """
    for name in ['truth', 'uwb']:
        if name not in dataset:
            raise ValueError(f'缺少数据组 {name}')
    for name in ['uwb_ls', 'eskf']:
        if name not in estimates:
            raise ValueError(f'缺少估计组 {name}')
    truth, uwb, ls, eskf = dataset['truth'], dataset['uwb'], estimates['uwb_ls'], estimates['eskf']
    truth_time = _get_time(truth, 'truth')
    uwb_time = _get_time(uwb, 'uwb')
    ls_time = _get_time(ls, 'uwb_ls')
    eskf_time = _get_time(eskf, 'eskf')
    if len(ls_time) != len(uwb_time):
        raise ValueError('UWB LS 时间轴必须包含每个 UWB 观测时刻')
    match_timestamps(uwb_time, ls_time, reference_name='uwb', query_name='uwb_ls')
    ls_truth = match_timestamps(truth_time, ls_time, reference_name='truth', query_name='uwb_ls')
    state_truth = match_timestamps(truth_time, eskf_time, reference_name='truth', query_name='eskf')
    common_state = match_timestamps(eskf_time, ls_time, reference_name='eskf', query_name='uwb_ls')
    truth_arrays = {key: _array(truth, key, (len(truth_time), width), 'truth')
                    for key, width in [('position', 3), ('velocity', 3), ('quaternion', 4),
                                       ('imu_acc_bias', 3), ('imu_gyro_bias', 3)]}
    eskf_arrays = {key: _array(eskf, key, (len(eskf_time), width), 'eskf')
                   for key, width in [('position', 3), ('velocity', 3), ('quaternion', 4),
                                      ('acc_bias', 3), ('gyro_bias', 3), ('tag_position', 3)]}
    if 'anchors' not in uwb:
        raise ValueError('缺少 uwb.anchors')
    anchors = np.asarray(uwb['anchors'], dtype=float)
    if anchors.ndim != 2 or anchors.shape[1:] != (3,) or not len(anchors) or not np.all(np.isfinite(anchors)):
        raise ValueError('uwb.anchors 必须为非空有限的 (基站数, 3) 数组')
    lever = _array(uwb, 'lever_arm', (3,), 'uwb')
    ls_tag = _array(ls, 'tag_position', (len(ls_time), 3), 'uwb_ls')
    _array(ls, 'covariance', (len(ls_time), 3, 3), 'uwb_ls', finite=False)
    _array(ls, 'residual', (len(ls_time), len(anchors)), 'uwb_ls')
    converged = _array(ls, 'converged', (len(ls_time),), 'uwb_ls')
    _array(ls, 'iterations', (len(ls_time),), 'uwb_ls')
    covariance = _array(eskf, 'covariance', (len(eskf_time), 15, 15), 'eskf', finite=False)
    true_tag = truth_arrays['position'][ls_truth] + np.asarray([
        rotation_matrix(quaternion) @ lever for quaternion in truth_arrays['quaternion'][ls_truth]
    ])
    rotation_error = right_local_rotation_errors(eskf_arrays['quaternion'], truth_arrays['quaternion'][state_truth])
    body_error = eskf_arrays['position'] - truth_arrays['position'][state_truth]
    velocity_error = eskf_arrays['velocity'] - truth_arrays['velocity'][state_truth]
    acc_error = eskf_arrays['acc_bias'] - truth_arrays['imu_acc_bias'][state_truth]
    gyro_error = eskf_arrays['gyro_bias'] - truth_arrays['imu_gyro_bias'][state_truth]
    state_error = np.column_stack((-body_error, -velocity_error, rotation_error, -acc_error, -gyro_error))
    nees = np.full(len(eskf_time), np.nan)
    exclusions = {kind: 0 for kind in ['nonfinite', 'asymmetric', 'indefinite', 'singular', 'ill_conditioned']}
    for index, (error, sample_covariance) in enumerate(zip(state_error, covariance)):
        problem = _covariance_problem(sample_covariance)
        if problem is None:
            try:
                factor = np.linalg.cholesky((sample_covariance + sample_covariance.T) / 2)
                whitened = np.linalg.solve(factor, error)
                value = float(whitened @ whitened)
                if not np.isfinite(value):
                    problem = 'nonfinite'
                else:
                    nees[index] = value
            except np.linalg.LinAlgError:
                problem = 'singular'
        if problem is not None:
            exclusions[problem] += 1
    diagnostics = estimates.get('diagnostics')
    if diagnostics is None:
        diagnostic_time, nis = np.empty(0), np.empty(0)
    else:
        diagnostic_time = _get_time(diagnostics, 'diagnostics', allow_empty=True)
        diagnostic_indices = match_timestamps(uwb_time, diagnostic_time, reference_name='uwb', query_name='diagnostics')
        if np.any(diagnostic_indices == 0):
            raise ValueError('诊断时间轴不能重复包含用于初始化的首帧 UWB 时刻')
        _array(diagnostics, 'innovation', (len(diagnostic_time), len(anchors)), 'diagnostics', finite=False)
        _array(diagnostics, 'innovation_covariance', (len(diagnostic_time), len(anchors), len(anchors)),
               'diagnostics', finite=False)
        supplied_nis = _array(diagnostics, 'nis', (len(diagnostic_time),), 'diagnostics', finite=False)
        nis = np.where(np.isfinite(supplied_nis) & (supplied_nis >= 0), supplied_nis, np.nan)
    return {
        'common_time': ls_time, 'eskf_time': eskf_time, 'diagnostic_time': diagnostic_time,
        'true_tag_position': true_tag, 'ls_tag_position': ls_tag,
        'eskf_tag_position': eskf_arrays['tag_position'][common_state],
        'ls_tag_error': ls_tag - true_tag,
        'eskf_tag_error': eskf_arrays['tag_position'][common_state] - true_tag,
        'body_position_error': body_error, 'velocity_error': velocity_error,
        'right_local_rotation_error_rad': rotation_error,
        'attitude_error_deg': np.rad2deg(np.linalg.norm(rotation_error, axis=1)),
        'acc_bias_error': acc_error, 'gyro_bias_error': gyro_error,
        'nees15': nees, 'nees15_exclusions': exclusions,
        'nis': nis, 'nis_dof': len(anchors), 'ls_converged': converged,
    }


def descriptive_statistics(values) -> dict:
    """给出有限样本的描述统计；空样本采用 JSON null，不伪造零。"""
    values = np.asarray(values, dtype=float).reshape(-1)
    valid = values[np.isfinite(values)]
    fields = ['mean', 'median', 'p95', 'min', 'max', 'std', 'rmse']
    if not len(valid):
        return {'count': 0, **{key: None for key in fields}}
    return {'count': int(len(valid)), 'mean': float(np.mean(valid)),
            'median': float(np.median(valid)), 'p95': float(np.percentile(valid, 95)),
            'min': float(np.min(valid)), 'max': float(np.max(valid)),
            'std': float(np.std(valid)), 'rmse': float(np.sqrt(np.mean(valid ** 2)))}


def _rmse(errors: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum(errors ** 2, axis=1))))


def _translation_rpe(time: np.ndarray, errors: np.ndarray) -> dict:
    """匹配真实相差 1 s 的端点，计算世界系位移差；不按样本索引近似。"""
    relative_errors = []
    for index, start in enumerate(time):
        target = start + 1.
        right = np.searchsorted(time, target)
        candidates = [candidate for candidate in (right - 1, right) if index < candidate < len(time)]
        if not candidates:
            continue
        endpoint = min(candidates, key=lambda candidate: abs(time[candidate] - target))
        if abs((time[endpoint] - start) - 1.) <= RPE_TIME_ATOL:
            relative_errors.append(np.linalg.norm(errors[endpoint] - errors[index]))
    return descriptive_statistics(relative_errors)


def evaluate_estimates(dataset: dict, estimates: dict) -> dict:
    """返回可严格 JSON 序列化的公平比较报告；仅本离线层访问真值。"""
    errors = estimation_errors(dataset, estimates)
    report = {
        'schema_version': 1,
        'common_samples': int(len(errors['common_time'])),
        'comparison': {
            'quantity': '世界系 ENU 中的 UWB 天线位置',
            'time_axis': '全部共同 UWB 时间戳；等样本权重',
            'time_match_atol_s': TIME_MATCH_ATOL,
            'position_rmse_definition': 'sqrt(mean(sum((估计值-真值)^2, axis=1)))',
            'rpe_definition': '世界系 1 s 位移之差的范数；未利用真实姿态修正 LS 杆臂',
            'rpe_time_atol_s': RPE_TIME_ATOL,
            'truth_access': '仅离线评价；无插值、平滑或时间加权',
        },
    }
    for name, field in [('uwb_ls', 'ls_tag_error'), ('eskf', 'eskf_tag_error')]:
        report[name] = {
            'tag_position_rmse_m': _rmse(errors[field]),
            'tag_ape_m': descriptive_statistics(np.linalg.norm(errors[field], axis=1)),
            'translation_rpe_1s_m': _translation_rpe(errors['common_time'], errors[field]),
        }
    report['uwb_ls']['converged_samples'] = int(np.count_nonzero(errors['ls_converged']))
    report['uwb_ls']['unconverged_samples'] = int(np.count_nonzero(~errors['ls_converged'].astype(bool)))
    report['eskf'].update({
        'state_samples': int(len(errors['eskf_time'])),
        'state_time_axis': '全部 ESKF 输出事件时间戳；等样本权重',
        'body_position_rmse_m': _rmse(errors['body_position_error']),
        'velocity_rmse_mps': _rmse(errors['velocity_error']),
        'attitude_rmse_deg': float(np.sqrt(np.mean(errors['attitude_error_deg'] ** 2))),
        'acc_bias_rmse_mps2': _rmse(errors['acc_bias_error']),
        'gyro_bias_rmse_radps': _rmse(errors['gyro_bias_error']),
    })
    report['diagnostics'] = {
        'interpretation': '单次运行的逐时刻描述统计；不代表统计一致性检验或蒙特卡洛置信区间',
        'nees15': descriptive_statistics(errors['nees15']),
        'nees15_per_dof': descriptive_statistics(errors['nees15'] / 15),
        'nees_dof': 15,
        'nees15_excluded_samples': int(sum(errors['nees15_exclusions'].values())),
        'nees15_exclusions': errors['nees15_exclusions'],
        'nees_covariance_condition_max': MAX_COVARIANCE_CONDITION,
        'nees_error_convention': '[p_true-p_est,v_true-v_est,Log(q_est⁻¹⊗q_true),ba_true-ba_est,bg_true-bg_est]',
        'nis': descriptive_statistics(errors['nis']),
        'nis_per_dof': descriptive_statistics(errors['nis'] / errors['nis_dof']),
        'nis_dof': errors['nis_dof'],
        'nis_excluded_samples': int(np.count_nonzero(~np.isfinite(errors['nis']))),
        'nis_time_axis': 'diagnostics.time；初始化首帧已排除',
    }
    if 'timing' in estimates:
        timing = estimates['timing']
        report['timing'] = {}
        for name, size in [('ls_update_seconds', len(errors['common_time'])),
                           ('eskf_event_seconds', len(errors['eskf_time']))]:
            if name in timing:
                values = _array(timing, name, (size,), 'timing')
                if np.any(values < 0):
                    raise ValueError(f'timing.{name} 不能为负')
                report['timing'][name] = descriptive_statistics(values)
        if 'initialization_seconds' in timing:
            value = np.asarray(timing['initialization_seconds'], dtype=float)
            if value.shape != () or not np.isfinite(value) or value < 0:
                raise ValueError('timing.initialization_seconds 必须为有限非负标量')
            report['timing']['initialization_seconds'] = float(value)
    return report
