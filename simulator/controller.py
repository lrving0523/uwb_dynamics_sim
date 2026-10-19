"""位置积分反馈与几何姿态控制，输出目标电机 RPM。"""

import numpy as np

from .math3d import quaternion_to_matrix
from .motor_model import MotorModel


class PositionController:
    """位置参考的前馈加速度、限幅反馈与姿态力矩闭环。"""

    def __init__(self, drone_config, motor_config, controller_config):
        self.mass = float(drone_config.get('mass', 1.5))
        self.gravity = float(drone_config.get('gravity', 9.81))
        inertia = np.asarray(drone_config.get('inertia', [0.03, 0.03, 0.05]), dtype=float)
        self.inertia = np.diag(inertia) if inertia.shape == (3,) else inertia
        self.motor = MotorModel(motor_config)
        config = controller_config
        self.position_kp = np.asarray(config.get('position_kp', [4, 4, 6]), dtype=float)
        self.velocity_kd = np.asarray(config.get('velocity_kd', [3, 3, 4]), dtype=float)
        self.position_ki = np.asarray(config.get('position_ki', [0.15, 0.15, 0.3]), dtype=float)
        self.attitude_kp = np.asarray(config.get('attitude_kp', [2.5, 2.5, 1.0]), dtype=float)
        self.rate_kd = np.asarray(config.get('rate_kd', [0.55, 0.55, 0.35]), dtype=float)
        self.integral_limit = float(config.get('integral_limit', 2.0))
        self.max_acceleration = float(config.get('max_acceleration', 6.0))
        self.max_tilt = np.deg2rad(float(config.get('max_tilt_deg', 35.0)))
        self.integral = np.zeros(3)

    def command(self, state, reference, dt):
        """使用当前状态控制，不读取未来轨迹或外力真值。"""
        if not np.isfinite(dt) or dt < 0:
            raise ValueError('控制周期必须有限且非负')
        position_error = np.asarray(reference['position']) - state.position
        velocity_error = np.asarray(reference['velocity']) - state.velocity
        self.integral = np.clip(self.integral + dt * position_error, -self.integral_limit, self.integral_limit)
        acceleration = (np.asarray(reference['acceleration']) + self.position_kp * position_error
                        + self.velocity_kd * velocity_error + self.position_ki * self.integral)
        magnitude = np.linalg.norm(acceleration)
        if magnitude > self.max_acceleration:
            acceleration = acceleration * (self.max_acceleration / magnitude)
        acceleration = acceleration + np.array([0.0, 0.0, self.gravity])
        acceleration[2] = max(acceleration[2], 0.1 * self.gravity)
        horizontal = np.linalg.norm(acceleration[:2])
        horizontal_limit = acceleration[2] * np.tan(self.max_tilt)
        if horizontal > horizontal_limit:
            acceleration[:2] *= horizontal_limit / horizontal
        desired_z = acceleration / np.linalg.norm(acceleration)
        yaw = float(reference.get('yaw', 0.0))
        heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        desired_y = np.cross(desired_z, heading)
        desired_y /= np.linalg.norm(desired_y)
        desired_rotation = np.column_stack((np.cross(desired_y, desired_z), desired_y, desired_z))
        rotation = quaternion_to_matrix(state.quaternion)
        error_matrix = 0.5 * (desired_rotation.T @ rotation - rotation.T @ desired_rotation)
        attitude_error = np.array([error_matrix[2, 1], error_matrix[0, 2], error_matrix[1, 0]])
        torque = (-self.attitude_kp * attitude_error - self.rate_kd * state.omega
                  + np.cross(state.omega, self.inertia @ state.omega))
        thrust = max(0.0, self.mass * np.dot(acceleration, rotation[:, 2]))
        return self.motor.allocate(thrust, torque)
