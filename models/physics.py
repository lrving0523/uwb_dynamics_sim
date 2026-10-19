"""仅依赖估计状态与测量 RPM 的名义刚体受力模型。"""

import numpy as np

from simulator.math3d import quaternion_to_matrix, skew


def _vector(value, size, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name}必须是有限{size}维向量')
    return result


class PhysicsModel:
    """推力单位 N；姿态为机体 FLU 到世界 ENU 的 Hamilton wxyz。"""

    def __init__(self, config):
        self.mass = float(config.get('mass', 1.5))
        self.gravity = float(config.get('gravity', 9.81))
        self.kf = float(config.get('kf', 1e-5))
        if not np.all(np.isfinite([self.mass, self.gravity, self.kf])) or min(self.mass, self.gravity, self.kf) <= 0:
            raise ValueError('名义质量、重力和推力系数必须有限且为正')
        self.linear_drag = _vector(config.get('linear_drag', [0]*3), 3, '线性阻力').copy()
        self.quadratic_drag = _vector(config.get('quadratic_drag', [0]*3), 3, '二次阻力').copy()
        self.wind_velocity = _vector(config.get('wind_velocity', [0]*3), 3, '名义空气速度').copy()
        if np.any(self.linear_drag < 0) or np.any(self.quadratic_drag < 0):
            raise ValueError('名义阻力系数必须非负')

    def aerodynamic_force_body(self, quaternion, velocity):
        """返回名义气动力，不含推力、重力与世界未知外力。"""
        rotation = quaternion_to_matrix(quaternion)
        relative = rotation.T @ (_vector(velocity, 3, '世界速度')-self.wind_velocity)
        return -self.linear_drag*relative-self.quadratic_drag*np.abs(relative)*relative

    def force_body(self, quaternion, velocity, rpm):
        """四电机转速先换 rad/s，再沿机体正 z 叠加推力。"""
        measured = _vector(rpm, 4, '测量RPM')
        force = self.aerodynamic_force_body(quaternion, velocity)
        force[2] += self.kf*np.sum((measured*(2*np.pi/60))**2)
        return force

    def acceleration(self, quaternion, velocity, rpm):
        """世界加速度包含向下的重力，不把重力重复计入比力。"""
        return quaternion_to_matrix(quaternion) @ self.force_body(quaternion, velocity, rpm)/self.mass + np.array([0., 0., -self.gravity])

    def force_jacobians(self, quaternion, velocity, rpm):
        """返回对世界速度、右局部姿态和测量 RPM 的机体系力雅可比。"""
        rotation = quaternion_to_matrix(quaternion)
        relative = rotation.T @ (_vector(velocity, 3, '世界速度')-self.wind_velocity)
        slope = -np.diag(self.linear_drag+2*self.quadratic_drag*np.abs(relative))
        rpm_jacobian = np.zeros((3, 4))
        rpm_jacobian[2] = 2*self.kf*(2*np.pi/60)**2*_vector(rpm, 4, '测量RPM')
        return slope @ rotation.T, slope @ skew(relative), rpm_jacobian
