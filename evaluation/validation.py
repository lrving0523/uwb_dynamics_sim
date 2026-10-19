"""只评价阶段一物理真值与传感器生成质量，不计算定位算法性能。"""

import numpy as np

from simulator.math3d import rotation_matrix


def summarize_dataset(dataset, config):
    """按时间权重计算跟踪误差，并直接核对力平衡与测量噪声。"""
    truth = dataset['truth']
    times = truth['time']
    rotations = np.asarray([rotation_matrix(q) for q in truth['quaternion']])
    gravity = np.array([0.0, 0.0, -config['drone']['gravity']])
    force_world = np.einsum('nij,nj->ni', rotations,
                    truth['thrust_body']+truth['aero_force_body'])+truth['external_force_world']
    balance = config['drone']['mass']*(truth['acceleration_world']-gravity)-force_world
    error = np.linalg.norm(truth['position']-truth['reference_position'], axis=1)
    report = {
        '说明': '所有误差均为仿真或参考跟踪验证，不代表滤波定位精度。',
        'trajectory': config['trajectory']['type'],
        'disturbance': config['wind']['type'] if config['wind']['enabled'] else 'none',
        'duration_s': config['simulation']['duration'], 'seed': config['simulation']['seed'],
        'tracking_position_rmse_m': float(np.sqrt(np.trapz(error**2, times)/(times[-1]-times[0]))),
        'tracking_position_max_m': float(max(error)),
        'quaternion_norm_max_error': float(max(abs(np.linalg.norm(truth['quaternion'], axis=1)-1))),
        'rotation_orthogonality_max_error': float(np.max(abs(np.transpose(rotations, (0, 2, 1))@rotations-np.eye(3)))),
        'force_balance_max_error_N': float(np.max(abs(balance))),
        'true_rpm_min': float(np.min(truth['true_rpm'])),
        'true_rpm_max': float(np.max(truth['true_rpm'])),
        'samples': {name: len(group['time']) for name, group in dataset.items()},
    }
    # 所有观测时刻均为真值事件；只匹配当前状态，不做时间插值。
    for name in ['imu', 'rpm', 'uwb']:
        group = dataset[name]
        indices = np.searchsorted(times, group['time'])
        indices = np.clip(indices, 0, len(times)-1)
        previous = np.maximum(indices-1, 0)
        indices = np.where(abs(times[previous]-group['time']) < abs(times[indices]-group['time']), previous, indices)
        if not np.allclose(times[indices], group['time'], atol=1e-12, rtol=0):
            raise ValueError('观测时间未对应到当前真值事件')
        if name == 'imu':
            ideal = np.einsum('nji,nj->ni', rotations[indices], truth['acceleration_world'][indices]-gravity)
            residuals = {'imu_acc': group['accel']-ideal-truth['imu_acc_bias'][indices],
                         'imu_gyro': group['gyro']-truth['omega'][indices]-truth['imu_gyro_bias'][indices]}
        elif name == 'rpm':
            residuals = {'rpm': group['rpm']-truth['true_rpm'][indices]}
        else:
            tags = truth['position'][indices]+np.einsum('nij,j->ni', rotations[indices], group['lever_arm'])
            ideal = np.linalg.norm(tags[:, None, :]-group['anchors'], axis=2)
            residuals = {'uwb': group['ranges']-ideal}
        for key, residual in residuals.items():
            report[f'{key}_noise_mean'] = np.mean(residual, axis=0).tolist()
            report[f'{key}_noise_std'] = np.std(residual, axis=0, ddof=1).tolist() if len(residual)>1 else None
    return report
