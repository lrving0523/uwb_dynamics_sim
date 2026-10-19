"""按控制与观测事件推进动力学，不读取任何未来真值。"""

from copy import deepcopy

import numpy as np

from sensors.imu_sensor import IMUSensor
from sensors.rpm_sensor import RPMSensor
from sensors.uwb_sensor import UWBSensor
from simulator.config import validate_config
from simulator.controller import PositionController
from simulator.drone import Drone
from simulator.trajectory import Trajectory
from simulator.wind_field import WindField


def sample_times(duration, frequency):
    """构造 t_k=k/f；仅保留有效区间内的实际采样时刻。"""
    if not np.isfinite(duration) or duration < 0 or not np.isfinite(frequency) or frequency <= 0:
        raise ValueError('时长必须非负，频率必须为有限正数')
    count = int(np.floor(np.nextafter(duration * frequency, np.inf))) + 1
    times = np.arange(count, dtype=float) / frequency
    return times[times <= duration]


def _event_times(clocks, duration):
    """只合并完全相同的时刻；相邻浮点数也可能分隔外力跳变。"""
    return np.unique(np.concatenate([*clocks.values(), [duration]])).tolist()


def _append(records, values):
    for name, value in values.items():
        records.setdefault(name, []).append(np.array(value, copy=True))


def run_simulation(config):
    """返回四个互相独立的数组字典；测量包仅包含可观测量与安装信息。"""
    validate_config(config)
    sim = config['simulation']
    duration = sim['duration']
    rngs = [np.random.default_rng(child) for child in np.random.SeedSequence(sim['seed']).spawn(4)]
    wind = WindField(config['wind'], rngs[0], duration)
    aero_config = deepcopy(config['aerodynamics'])
    base_air_velocity = np.asarray(aero_config['wind_velocity'], dtype=float)
    aero_config['air_velocity'] = lambda time: base_air_velocity + wind.velocity(time)
    drone = Drone(config['drone'], config['motor'], aero_config)
    controller = PositionController(config['drone'], config['motor'], config['controller'])
    trajectory = Trajectory(config['trajectory'], config['drone']['initial_position'])
    imu = IMUSensor(config['sensors']['imu'], rngs[1])
    rpm = RPMSensor(config['sensors']['rpm'], rngs[2])
    uwb = UWBSensor(config['sensors']['uwb'], rngs[3])
    clocks = {'physics': sample_times(duration, 1/sim['dt']),
              'control': sample_times(duration, sim['controller_hz'])}
    clocks.update({name: sample_times(duration, config['sensors'][name]['hz'])
                   for name in ['imu', 'rpm', 'uwb']})
    # 将非网格上的外扰跳变/随机节点也作为积分边界，避免跨越不连续点。
    if config['wind']['enabled'] and config['wind']['type'] in ['step', 'sine', 'random']:
        boundaries = [config['wind']['start_time']]
        if config['wind']['type'] == 'random':
            boundaries.extend(wind.node_times)
        clocks['disturbance'] = np.unique([value for value in boundaries if 0 <= value <= duration])
    indices = {name: 0 for name in clocks}
    dataset = {name: {} for name in ['truth', 'imu', 'rpm', 'uwb']}
    command = drone.rpm.copy()
    previous_time = 0.0
    previous_control = None
    gravity = np.array([0.0, 0.0, -config['drone']['gravity']])
    for time in _event_times(clocks, duration):
        # 每段只使用已产生的控制命令，最多积分一个配置步长。
        remaining = time - previous_time
        while remaining > 0.0:
            step = min(sim['dt'], remaining)
            drone.step(command, step, wind.force, time=previous_time)
            previous_time += step
            remaining = time - previous_time
        previous_time = time
        state = drone.state
        reference = trajectory.sample(time)
        due = {}
        for name, clock in clocks.items():
            index = indices[name]
            if index < len(clock) and clock[index] == time:
                due[name] = float(clock[index])
                indices[name] += 1
        if 'control' in due:
            elapsed = 0.0 if previous_control is None else time - previous_control
            command = controller.command(state, reference, elapsed)
            previous_control = time
        quantities = drone.quantities(wind.force, time=time)
        if 'imu' in due:
            _append(dataset['imu'], imu.sample(due['imu'], state.quaternion,
                    quantities['acceleration_world'], state.omega, gravity))
        if 'rpm' in due:
            _append(dataset['rpm'], rpm.sample(due['rpm'], drone.rpm))
        if 'uwb' in due:
            _append(dataset['uwb'], uwb.sample(due['uwb'], state.position, state.quaternion))
        # 真值在所有事件时刻额外记录，可在观测时间精确关联，无需插值未来数据。
        _append(dataset['truth'], {
            'time': time, 'position': state.position, 'velocity': state.velocity,
            'quaternion': state.quaternion, 'omega': state.omega,
            'reference_position': reference['position'], 'reference_velocity': reference['velocity'],
            'reference_acceleration': reference['acceleration'],
            'target_rpm': command, 'true_rpm': drone.rpm,
            'imu_acc_bias': imu.acc_bias, 'imu_gyro_bias': imu.gyro_bias,
            **quantities,
        })
    for group in dataset.values():
        for key in group:
            group[key] = np.asarray(group[key])
            if not np.all(np.isfinite(group[key])):
                raise FloatingPointError(f'仿真产生非有限数值：{key}')
    dataset['uwb']['anchors'] = uwb.anchors.copy()
    dataset['uwb']['lever_arm'] = np.asarray(config['sensors']['uwb']['lever_arm'], dtype=float)
    return dataset
