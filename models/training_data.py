"""整飞行数据划分、100 Hz 因果测量窗口与离线气动力标签。"""

from copy import deepcopy
import hashlib

import numpy as np


def validate_split(split):
    """飞行标识在训练、验证和测试之间必须严格分离。"""
    if set(split) != {'train', 'val', 'test'} or any(not split[name] for name in split):
        raise ValueError('必须提供非空训练、验证、测试整飞行划分')
    flattened = [flight for name in ['train', 'val', 'test'] for flight in split[name]]
    if len(set(flattened)) != len(flattened):
        raise ValueError('训练/验证/测试飞行标识发生重复')


def measurement_sha256(measurements):
    """基于原始测量数值而非 ZIP 时间戳计算可复现数据指纹。"""
    digest = hashlib.sha256()
    for group in ['imu', 'uwb', 'rpm']:
        if group not in measurements:
            continue
        for key in sorted(measurements[group]):
            value = np.ascontiguousarray(measurements[group][key])
            digest.update(f'{group}.{key}|{value.dtype.str}|{value.shape}'.encode('utf-8'))
            digest.update(value.tobytes())
    return digest.hexdigest()


def measurement_fingerprint(measurements):
    """只对网络实际可见的原始测量特征与时钟计算指纹。"""
    times, features = causal_features(measurements)
    digest = hashlib.sha256()
    for values in [np.asarray(times, dtype='<f8'), np.asarray(features, dtype='<f4')]:
        digest.update(str(values.shape).encode('ascii'))
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def causal_features(measurements, input_hz=100):
    """RPM 原始采样时钟上取最近已到达的陀螺；不接收真值标签。"""
    times = np.asarray(measurements['rpm']['time'], dtype=float)
    imu_times = np.asarray(measurements['imu']['time'], dtype=float)
    rpm, gyro = np.asarray(measurements['rpm']['rpm']), np.asarray(measurements['imu']['gyro'])
    if input_hz != 100 or times.ndim != 1 or len(times) == 0 or not np.all(np.isfinite(times)):
        raise ValueError('当前特征时钟需要非空 100 Hz 原始 RPM')
    if not np.allclose(times, times[0]+np.arange(len(times))/input_hz, rtol=0, atol=1e-12) or np.any(np.diff(times) <= 0):
        raise ValueError('不支持非 100 Hz 原始 RPM 时钟')
    if imu_times.ndim != 1 or len(imu_times) == 0 or np.any(np.diff(imu_times) <= 0) or not np.all(np.isfinite(imu_times)):
        raise ValueError('IMU 时间必须有限、非空且严格递增')
    if rpm.shape != (len(times), 4) or gyro.shape != (len(imu_times), 3) or not np.all(np.isfinite(rpm)) or not np.all(np.isfinite(gyro)):
        raise ValueError('RPM 与陀螺测量形状错误或含非有限值')
    indices = np.searchsorted(imu_times, times, side='right')-1
    if np.any(indices < 0):
        raise ValueError('RPM 时刻尚未收到任何陀螺样本')
    return times.copy(), np.column_stack((gyro[indices], rpm)).astype(np.float32)


def flight_windows(dataset, physics_model, sequence_length=10):
    """标签只是真实气动力减名义气动力，不含外力、质量或推力误差。"""
    times, features = causal_features(dataset)
    if sequence_length != 10 or len(times) < sequence_length:
        raise ValueError('每条飞行至少需要 10 个 100 Hz 测量样本')
    truth = dataset['truth']
    truth_times = np.asarray(truth['time'])
    indices = np.searchsorted(truth_times, times)
    if np.any(indices >= len(truth_times)) or not np.allclose(truth_times[indices], times, atol=1e-12, rtol=0):
        raise ValueError('离线标签需要与 RPM 时刻精确对应的真值')
    nominal = np.array([physics_model.aerodynamic_force_body(truth['quaternion'][i], truth['velocity'][i]) for i in indices])
    targets = np.asarray(truth['aero_force_body'])[indices]-nominal
    windows = np.lib.stride_tricks.sliding_window_view(features, sequence_length, axis=0).transpose(0, 2, 1).copy()
    return {'time': times[sequence_length-1:], 'inputs': windows, 'targets': targets[sequence_length-1:].astype(np.float32),
            'truth_indices': indices[sequence_length-1:]}


def flight_config(base_config, seed, index, duration=18., dt=.01):
    """从固定种子生成多幅度/周期/方向轨迹，保持初始偏航与参考一致。"""
    config = deepcopy(base_config)
    rng = np.random.default_rng(seed)
    kinds = ['circle', 'figure_eight', 'variable_speed', 'accel_decel', 'hover']
    yaw = float(rng.uniform(-.5, .5))
    radius = float(rng.uniform(.7, 1.5))
    config['simulation'].update(seed=int(seed), duration=float(duration), dt=float(dt), controller_hz=100.)
    config['trajectory'].update(type=kinds[index % len(kinds)], radius=radius, period=float(rng.uniform(6, 10)),
                                yaw=yaw, distance=2*radius, direction=[float(rng.choice([-1, 1])), float(rng.uniform(-1, 1)), 0.])
    config['drone']['initial_quaternion'] = [float(np.cos(yaw/2)), 0., 0., float(np.sin(yaw/2))]
    config['wind']['enabled'] = False
    config['sensors']['rpm']['hz'] = 100.
    config['sensors']['imu']['hz'] = 200.
    config['output']['csv'] = False
    return config
