"""世界 ENU、机体 FLU 下的四旋翼六自由度刚体动力学。"""

from dataclasses import dataclass

import numpy as np

from .math3d import normalize_quaternion, quaternion_multiply, quaternion_to_matrix
from .motor_model import MotorModel


def _vector(value, name):
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f'{name}必须是有限三维向量')
    return vector.copy()


@dataclass
class DroneState:
    """位置为 m、速度为 m/s、四元数为 wxyz、机体角速度为 rad/s。"""

    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    omega: np.ndarray

    def __post_init__(self):
        self.position = _vector(self.position, '位置')
        self.velocity = _vector(self.velocity, '速度')
        self.quaternion = normalize_quaternion(self.quaternion)
        self.omega = _vector(self.omega, '角速度')


class Drone:
    """RK4 刚体积分，每个子时刻使用电机解析响应。"""

    def __init__(self, drone_config, motor_config, aero_config=None, state=None):
        self.mass = float(drone_config.get('mass', 1.5))
        self.gravity = float(drone_config.get('gravity', 9.81))
        if not np.isfinite(self.mass) or self.mass <= 0 or not np.isfinite(self.gravity) or self.gravity <= 0:
            raise ValueError('质量和重力加速度必须有限且为正')
        inertia = np.asarray(drone_config.get('inertia', [0.03, 0.03, 0.05]), dtype=float)
        self.inertia = np.diag(inertia) if inertia.shape == (3,) else inertia
        if (self.inertia.shape != (3, 3) or not np.all(np.isfinite(self.inertia))
                or not np.allclose(self.inertia, self.inertia.T)
                or np.min(np.linalg.eigvalsh(self.inertia)) <= 0):
            raise ValueError('惯性矩阵必须是对称正定的三阶矩阵或正三维对角向量')
        self.inverse_inertia = np.linalg.inv(self.inertia)
        self.motor = MotorModel(motor_config)
        hover_rpm = np.sqrt(self.mass * self.gravity / (4 * self.motor.kf)) * 60 / (2 * np.pi)
        if not self.motor.min_rpm <= hover_rpm <= self.motor.max_rpm:
            raise ValueError('给定电机转速范围无法悬停')
        self.rpm = self.motor._rpm(motor_config.get('initial_rpm', np.full(4, hover_rpm)))
        self.state = state if state is not None else DroneState(
            drone_config.get('initial_position', [0, 0, 1.5]),
            drone_config.get('initial_velocity', [0, 0, 0]),
            drone_config.get('initial_quaternion', [1, 0, 0, 0]),
            drone_config.get('initial_omega', [0, 0, 0]),
        )
        if not isinstance(self.state, DroneState):
            raise ValueError('初始状态必须是 DroneState')
        aero = aero_config or {}
        self.linear_drag = np.broadcast_to(np.asarray(aero.get('linear_drag', [0, 0, 0]), dtype=float), (3,)).copy()
        self.quadratic_drag = np.broadcast_to(np.asarray(aero.get('quadratic_drag', [0, 0, 0]), dtype=float), (3,)).copy()
        if not np.all(np.isfinite([self.linear_drag, self.quadratic_drag])) or np.any(self.linear_drag < 0) or np.any(self.quadratic_drag < 0):
            raise ValueError('阻力系数必须有限且非负')
        self.wind_velocity = _vector(aero.get('wind_velocity', [0, 0, 0]), '空气速度')
        self.air_velocity = aero.get('air_velocity')
        if self.air_velocity is not None and not callable(self.air_velocity):
            raise ValueError('动态空气速度必须为可调用函数')

    def _quantities(self, state, rpm, external_force, time):
        rotation = quaternion_to_matrix(state[6:10])
        thrust, torque = self.motor.wrench(rpm)
        air_velocity = self.wind_velocity if self.air_velocity is None else _vector(self.air_velocity(time), '空气速度')
        relative_velocity = rotation.T @ (state[3:6] - air_velocity)
        aero_force = -self.linear_drag * relative_velocity - self.quadratic_drag * np.abs(relative_velocity) * relative_velocity
        external = _vector(external_force(time) if callable(external_force) else external_force, '世界外力')
        acceleration = (rotation @ (thrust + aero_force) + external) / self.mass + np.array([0, 0, -self.gravity])
        return {'acceleration_world': acceleration, 'thrust_body': thrust,
                'torque_body': torque, 'aero_force_body': aero_force,
                'external_force_world': external}

    def _packed(self):
        return np.concatenate((self.state.position, self.state.velocity, self.state.quaternion, self.state.omega))

    def quantities(self, external_force, time=0.0):
        """查询当前真值及分解力；外力与气动力独立保存。"""
        return self._quantities(self._packed(), self.rpm, external_force, time)

    def step(self, target_rpm, dt, external_force, time=0.0):
        """推进 dt 秒；外力跳变的积分端点取左极限，采样仍右连续。"""
        if not np.isfinite(dt) or dt <= 0 or not np.isfinite(time):
            raise ValueError('步长必须有限且为正，时间必须有限')
        target_rpm = self.motor._rpm(target_rpm)
        initial_rpm, initial_state = self.rpm.copy(), self._packed()

        def derivative(state, fraction):
            elapsed = fraction * dt
            query_time = time + elapsed
            if fraction == 1.0:
                query_time = np.nextafter(query_time, time)
            rpm = self.motor.response(initial_rpm, target_rpm, elapsed)
            quantities = self._quantities(state, rpm, external_force, query_time)
            omega = state[10:13]
            angular_acceleration = self.inverse_inertia @ (quantities['torque_body'] - np.cross(omega, self.inertia @ omega))
            quaternion_derivative = 0.5 * quaternion_multiply(state[6:10], np.concatenate(([0.0], omega)))
            return np.concatenate((state[3:6], quantities['acceleration_world'], quaternion_derivative, angular_acceleration))

        k1 = derivative(initial_state, 0.0)
        k2 = derivative(initial_state + dt * k1 / 2, 0.5)
        k3 = derivative(initial_state + dt * k2 / 2, 0.5)
        k4 = derivative(initial_state + dt * k3, 1.0)
        result = initial_state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        self.state = DroneState(result[:3], result[3:6], result[6:10], result[10:13])
        self.rpm = self.motor.response(initial_rpm, target_rpm, dt)
        return self.state
