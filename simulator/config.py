"""加载统一配置，保留旧版参数并显式校验单位相关约束。"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml


DEFAULTS = {
    'simulation': {'dt': 0.01, 'duration': 12.0, 'seed': 42, 'controller_hz': 100.0},
    'drone': {'mass': 1.5, 'gravity': 9.81, 'inertia': [0.03, 0.03, 0.05],
              'initial_position': [0.0, 0.0, 1.5], 'initial_velocity': [0.0]*3,
              'initial_quaternion': [1.0, 0.0, 0.0, 0.0], 'initial_omega': [0.0]*3},
    'motor': {'arm_length': 0.22, 'kf': 1e-5, 'km': 2e-7,
              'time_constant': 0.035, 'min_rpm': 0.0, 'max_rpm': 10000.0},
    'aerodynamics': {'linear_drag': [0.12, 0.12, 0.16],
                     'quadratic_drag': [0.02, 0.02, 0.03], 'wind_velocity': [0.0]*3},
    'controller': {'position_kp': [4.0, 4.0, 6.0], 'velocity_kd': [3.0, 3.0, 4.0],
                   'position_ki': [0.15, 0.15, 0.3], 'integral_limit': 2.0,
                   'attitude_kp': [2.5, 2.5, 1.0], 'rate_kd': [0.55, 0.55, 0.35],
                   'max_acceleration': 6.0, 'max_tilt_deg': 35.0},
    'trajectory': {'type': 'figure_eight', 'radius': 1.2, 'period': 8.0, 'yaw': 0.0},
    'wind': {'enabled': False, 'type': 'step', 'force': [0.6, -0.3, 0.15],
             'start_time': 4.0, 'frequency': 0.35, 'random_std': [0.3, 0.3, 0.15],
             'random_hz': 10.0},
    'uwb': {'anchors': [[-8, -8, .3], [8, -8, 1.5], [8, 8, 2.4], [-8, 8, 3.2]],
            'lever_arm': [0.08, 0.0, 0.03]},
    'sensors': {},
    'output': {'directory': 'results/phase1', 'formats': ['png', 'pdf', 'svg'], 'csv': True},
}


def _merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path):
    """读取 YAML；平铺传感器参数兼容，嵌套配置具有优先级。"""
    with Path(path).open(encoding='utf-8-sig') as stream:
        supplied = yaml.safe_load(stream)
    if not isinstance(supplied, dict):
        raise ValueError('配置顶层必须为映射')
    config = _merge(DEFAULTS, supplied)
    sensors = config['sensors']
    canonical = {
        'imu': {'hz': sensors.get('imu_hz', 200.0),
                'acc_std': sensors.get('imu_acc_std', 0.05),
                'gyro_std': sensors.get('imu_gyro_std', 0.002),
                'acc_bias': sensors.get('imu_acc_bias', [0.0]*3),
                'gyro_bias': sensors.get('imu_gyro_bias', [0.0]*3),
                'acc_bias_rw_std': sensors.get('imu_acc_bias_rw_std', 0.001),
                'gyro_bias_rw_std': sensors.get('imu_gyro_bias_rw_std', 0.0001)},
        'rpm': {'hz': sensors.get('rpm_hz', 100.0), 'std': sensors.get('rpm_std', 5.0)},
        'uwb': {'hz': sensors.get('uwb_hz', 10.0), 'std': sensors.get('uwb_std', 0.05),
                'anchors': config['uwb']['anchors'], 'lever_arm': config['uwb']['lever_arm']},
    }
    for name, defaults in canonical.items():
        sensors[name] = _merge(defaults, sensors.get(name, {}))
    validate_config(config)
    return config


def validate_config(config):
    """拒绝非有限参数、非法形状和无法提供悬停推力的执行器。"""
    def check_finite(value, path='配置'):
        if isinstance(value, dict):
            for key, item in value.items():
                check_finite(item, f'{path}.{key}')
        elif isinstance(value, (list, tuple)):
            for item in value:
                check_finite(item, path)
        elif isinstance(value, (int, float)) and not np.isfinite(value):
            raise ValueError(f'{path} 必须为有限数值')

    def number(value, name, minimum=0.0, strict=True):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f'{name} 必须为数值')
        if value < minimum or (strict and value == minimum):
            raise ValueError(f'{name} 必须{ "大于" if strict else "不小于"}{minimum}')

    def vector(value, length, name):
        try:
            array = np.asarray(value, dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f'{name} 必须为数值向量') from exc
        if array.shape != (length,) or not np.all(np.isfinite(array)):
            raise ValueError(f'{name} 必须为 {length} 维有限向量')
        return array

    check_finite(config)
    simulation, drone, motor = config['simulation'], config['drone'], config['motor']
    for key in ['dt', 'duration', 'controller_hz']:
        number(simulation[key], f'simulation.{key}')
    seed = simulation['seed']
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('随机种子必须为非负整数')
    for key in ['mass', 'gravity']:
        number(drone[key], f'drone.{key}')
    inertia = vector(drone['inertia'], 3, 'inertia')
    if np.any(inertia <= 0) or 2 * max(inertia) > sum(inertia) + 1e-12:
        raise ValueError('转动惯量必须为正并满足刚体主惯量三角不等式')
    for key in ['initial_position', 'initial_velocity', 'initial_omega']:
        vector(drone[key], 3, key)
    if np.linalg.norm(vector(drone['initial_quaternion'], 4, 'initial_quaternion')) < 1e-12:
        raise ValueError('初始四元数不可为零')
    for key in ['arm_length', 'kf', 'km', 'time_constant', 'max_rpm']:
        number(motor[key], f'motor.{key}')
    number(motor['min_rpm'], 'motor.min_rpm', strict=False)
    if motor['max_rpm'] <= motor['min_rpm']:
        raise ValueError('max_rpm 必须大于 min_rpm')
    hover = np.sqrt(drone['mass'] * drone['gravity'] / (4 * motor['kf'])) * 60 / (2*np.pi)
    if not motor['min_rpm'] <= hover <= motor['max_rpm']:
        raise ValueError('电机转速范围无法实现悬停')
    if 'initial_rpm' in motor:
        initial = vector(motor['initial_rpm'], 4, 'initial_rpm')
        if np.any(initial < motor['min_rpm']) or np.any(initial > motor['max_rpm']):
            raise ValueError('初始 RPM 超出电机范围')
    for name in ['imu', 'uwb', 'rpm']:
        sensor = config['sensors'][name]
        number(sensor['hz'], f'{name}.hz')
        noise_keys = ['acc_std', 'gyro_std', 'acc_bias_rw_std', 'gyro_bias_rw_std'] if name == 'imu' else ['std']
        for key in noise_keys:
            noise = np.asarray(sensor[key], dtype=float)
            if noise.shape != () or np.any(noise < 0):
                raise ValueError(f'{name}.{key} 必须为非负标量')
    for key in ['acc_bias', 'gyro_bias']:
        vector(config['sensors']['imu'][key], 3, key)
    uwb = config['sensors']['uwb']
    anchors = np.asarray(uwb['anchors'], dtype=float)
    if anchors.ndim != 2 or anchors.shape[1:] != (3,) or len(anchors) < 1:
        raise ValueError('UWB 基站必须是非空 N×3 坐标')
    vector(uwb['lever_arm'], 3, 'lever_arm')
    for key in ['linear_drag', 'quadratic_drag']:
        if np.any(vector(config['aerodynamics'][key], 3, key) < 0):
            raise ValueError('气动阻力系数不可为负')
    vector(config['aerodynamics']['wind_velocity'], 3, 'wind_velocity')
    if config['trajectory']['type'] not in ['hover', 'circle', 'figure_eight', 'variable_speed', 'accel_decel']:
        raise ValueError('不支持的参考轨迹类型')
    number(config['trajectory']['period'], 'trajectory.period')
    number(config['trajectory']['radius'], 'trajectory.radius', strict=False)
    wind = config['wind']
    if wind['type'] not in ['none', 'constant', 'step', 'sine', 'random']:
        raise ValueError('不支持的外扰类型')
    vector(wind['force'], 3, 'wind.force')
    if np.any(vector(wind['random_std'], 3, 'wind.random_std') < 0):
        raise ValueError('随机外力标准差不可为负')
    number(wind['random_hz'], 'wind.random_hz')
    number(wind['frequency'], 'wind.frequency', strict=False)
    number(wind['start_time'], 'wind.start_time', strict=False)
    for key in ['position_kp', 'velocity_kd', 'position_ki', 'attitude_kp', 'rate_kd']:
        if np.any(vector(config['controller'][key], 3, key) < 0):
            raise ValueError('控制器增益不可为负')
    for key in ['max_acceleration', 'integral_limit']:
        number(config['controller'][key], f'controller.{key}')
    if not 0 < config['controller']['max_tilt_deg'] < 85:
        raise ValueError('最大倾角必须位于 (0,85) 度')
    if not config['output']['formats'] or any(f not in ['png', 'pdf', 'svg'] for f in config['output']['formats']):
        raise ValueError('绘图格式只支持 png/pdf/svg')
    return config
