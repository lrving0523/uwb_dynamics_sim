"""离线评价固定初始姿态与输入的局部开环动力学预测。"""

from __future__ import annotations

import numpy as np

from evaluation.metrics import TIME_MATCH_ATOL, _array, _get_time, descriptive_statistics
from simulator.math3d import rotation_matrix


_ASSUMPTION = ('局部常输入预测：仅用起点真值位置、速度、姿态初始化，固定起点姿态、测量 RPM、'
               '已产生的最近 TCN 残余力及初始预测加速度；期间不以真值重置状态。'
               '这是短时局部匀加速度预测，不代表完整非线性滚动仿真。')


def _saved_index(time: np.ndarray, query: float) -> int | None:
    """只关联已保存端点，容差内舍入差以外的缺失不插值。"""
    right = int(np.searchsorted(time, query))
    candidates = [index for index in (right - 1, right) if 0 <= index < len(time)]
    if not candidates:
        return None
    index = min(candidates, key=lambda candidate: abs(time[candidate] - query))
    return index if abs(time[index] - query) <= TIME_MATCH_ATOL else None


def model_prediction_arrays(dataset: dict, physics_model, hybrid_force_history: dict,
                            horizon_s: float = .2) -> dict:
    """在相同有效 RPM 起点构造物理/混合预测及误差数组。

    真值仅提供起点条件、起点真实加速度和终点评价；预测不读取中间真值。
    残余力仅在起点执行过去值保持，满窗口前尚无历史时使用零残余。
    返回时间单位 s，力 N，加速度 m/s²，速度增量 m/s，位置 m。
    """
    if not np.isscalar(horizon_s) or not np.isfinite(horizon_s) or horizon_s <= 0:
        raise ValueError('预测时长必须是正的有限秒数')
    horizon_s = float(horizon_s)
    mass = float(physics_model.mass)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError('名义模型质量必须有限且为正')
    for group in ('truth', 'rpm'):
        if group not in dataset:
            raise ValueError(f'缺少数据组 {group}')
    truth, rpm = dataset['truth'], dataset['rpm']
    truth_time, rpm_time = _get_time(truth, 'truth'), _get_time(rpm, 'rpm')
    states = {key: _array(truth, key, (len(truth_time), width), 'truth')
              for key, width in (('position', 3), ('velocity', 3), ('quaternion', 4),
                                 ('acceleration_world', 3))}
    measured_rpm = _array(rpm, 'rpm', (len(rpm_time), 4), 'rpm')
    residual_time = _get_time(hybrid_force_history, 'hybrid_force_history', allow_empty=True)
    residual = _array(hybrid_force_history, 'force_body', (len(residual_time), 3), 'hybrid_force_history')
    excluded = {'initial_truth_unmatched': 0, 'horizon_truth_unmatched': 0, 'horizon_out_of_range': 0}
    starts, ends, rpm_indices = [], [], []
    for index, start in enumerate(rpm_time):
        first = _saved_index(truth_time, float(start))
        if first is None:
            excluded['initial_truth_unmatched'] += 1
            continue
        endpoint = float(start + horizon_s)
        if endpoint > truth_time[-1] + TIME_MATCH_ATOL or endpoint < truth_time[0] - TIME_MATCH_ATOL:
            excluded['horizon_out_of_range'] += 1
            continue
        last = _saved_index(truth_time, endpoint)
        if last is None:
            excluded['horizon_truth_unmatched'] += 1
            continue
        starts.append(first)
        ends.append(last)
        rpm_indices.append(index)
    starts, ends, rpm_indices = (np.asarray(values, dtype=int) for values in (starts, ends, rpm_indices))
    time = rpm_time[rpm_indices]
    initial_position, initial_velocity = states['position'][starts], states['velocity'][starts]
    held_force = np.zeros((len(time), 3))
    residual_index = np.searchsorted(residual_time, time, side='right') - 1
    ready = residual_index >= 0
    held_force[ready] = residual[residual_index[ready]]
    acceleration = np.empty((len(time), 3))
    mixed_acceleration = np.empty_like(acceleration)
    for row, (first, rpm_index) in enumerate(zip(starts, rpm_indices)):
        quaternion, velocity = states['quaternion'][first], states['velocity'][first]
        prediction = np.asarray(physics_model.acceleration(quaternion.copy(), velocity.copy(),
                                                           measured_rpm[rpm_index].copy()), dtype=float)
        if prediction.shape != (3,) or not np.all(np.isfinite(prediction)):
            raise ValueError('物理模型 acceleration 必须返回有限三维世界加速度')
        acceleration[row] = prediction
        mixed_acceleration[row] = prediction + rotation_matrix(quaternion) @ held_force[row] / mass
    actual = {'acceleration': states['acceleration_world'][starts],
              'delta_velocity': states['velocity'][ends] - initial_velocity,
              'position': states['position'][ends]}
    output = {'time': time.copy(), 'endpoint_time': truth_time[ends].copy(),
              'initial_truth_index': starts, 'endpoint_truth_index': ends,
              'residual_force_body': held_force, 'residual_available': ready,
              'truth': actual, 'horizon_s': horizon_s,
              'candidate_samples': int(len(rpm_time)), 'excluded_samples': excluded}
    for method, predicted_acceleration in (('physics', acceleration), ('hybrid', mixed_acceleration)):
        values = {'acceleration': predicted_acceleration,
                  'delta_velocity': predicted_acceleration * horizon_s,
                  'position': initial_position + initial_velocity * horizon_s
                              + .5 * predicted_acceleration * horizon_s ** 2}
        values.update({f'{key}_error': value - actual[key] for key, value in list(values.items())})
        output[method] = values
    return output


def evaluate_model_predictions(dataset: dict, physics_model, hybrid_force_history: dict,
                               horizon_s: float = .2) -> dict:
    """返回可严格 JSON 序列化的局部预测指标，空样本为 null。"""
    arrays = model_prediction_arrays(dataset, physics_model, hybrid_force_history, horizon_s)
    report = {
        'schema_version': 1, 'horizon_s': arrays['horizon_s'],
        'common_samples': int(len(arrays['time'])), 'candidate_samples': arrays['candidate_samples'],
        'excluded_samples': arrays['excluded_samples'],
        'prediction_assumption': _ASSUMPTION,
        'initialization': '仅离线评价使用真值起点；不属于在线滤波性能指标',
        'time_axis': '同一批有效 RPM 测量起点，真值终点精确关联，未插值',
        'time_match_atol_s': TIME_MATCH_ATOL,
        'residual_input': '起点已产生的最近预测保持；无历史为零，不对未来残余插值',
        'residual_available_samples': int(np.count_nonzero(arrays['residual_available'])),
        'rmse_definition': 'sqrt(mean(sum((预测值-真值)^2, axis=1)))；三维误差范数',
        'units': {'time': 's', 'acceleration': 'm/s²', 'delta_velocity': 'm/s', 'position': 'm',
                  'residual_force_body': 'N'},
    }
    for method in ('physics', 'hybrid'):
        metrics = {}
        for quantity, metric_name in (('acceleration', 'acceleration_rmse_mps2'),
                                      ('delta_velocity', 'delta_velocity_rmse_mps'),
                                      ('position', 'position_rmse_m')):
            error = arrays[method][f'{quantity}_error']
            norms = np.linalg.norm(error, axis=1)
            metrics[metric_name] = float(np.sqrt(np.mean(norms ** 2))) if len(norms) else None
            metrics[f'{quantity}_error_norm'] = descriptive_statistics(norms)
            metrics[f'{quantity}_axis_rmse'] = (np.sqrt(np.mean(error ** 2, axis=0)).tolist()
                                                if len(error) else [None, None, None])
        report[method] = metrics
    return report
