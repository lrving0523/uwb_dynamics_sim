"""十字构型电机的平方律分配和一阶转速响应。"""

import numpy as np


class MotorModel:
    """电机顺序为前、左、后、右；输入转速单位为 RPM。"""

    def __init__(self, config):
        self.arm_length = float(config.get('arm_length', 0.22))
        self.kf = float(config.get('kf', 1e-5))
        self.km = float(config.get('km', 2e-7))
        self.time_constant = float(config.get('time_constant', 0.03))
        self.min_rpm = float(config.get('min_rpm', 0.0))
        self.max_rpm = float(config.get('max_rpm', 12000.0))
        positive = [self.arm_length, self.kf, self.km, self.time_constant, self.max_rpm]
        if not np.all(np.isfinite(positive)) or min(positive) <= 0:
            raise ValueError('电机几何、系数、时间常数和最高转速必须有限且为正')
        if not np.isfinite(self.min_rpm) or not 0 <= self.min_rpm < self.max_rpm:
            raise ValueError('最低转速必须非负且低于最高转速')
        length, kf, km = self.arm_length, self.kf, self.km
        self.allocation_matrix = np.array([
            [kf, kf, kf, kf],
            [0, length * kf, 0, -length * kf],
            [-length * kf, 0, length * kf, 0],
            [km, -km, km, -km],
        ])
        self.inverse_allocation = np.linalg.inv(self.allocation_matrix)

    def _rpm(self, rpm):
        rpm = np.asarray(rpm, dtype=float)
        if rpm.shape != (4,) or not np.all(np.isfinite(rpm)):
            raise ValueError('电机转速必须是有限的四维向量')
        return np.clip(rpm, self.min_rpm, self.max_rpm)

    def wrench(self, rpm):
        """先将 RPM 换为 rad/s，再返回机体系推力与反作用力矩。"""
        angular_speed = self._rpm(rpm) * (2 * np.pi / 60)
        wrench = self.allocation_matrix @ np.square(angular_speed)
        return np.array([0.0, 0.0, wrench[0]]), wrench[1:]

    def allocate(self, thrust, torque):
        """反求转速并逐电机饱和；饱和时不保证期望合力矩可实现。"""
        request = np.concatenate(([float(thrust)], np.asarray(torque, dtype=float)))
        if request.shape != (4,) or not np.all(np.isfinite(request)):
            raise ValueError('总推力和力矩必须有限，力矩必须为三维')
        angular_squared = self.inverse_allocation @ request
        minimum = (self.min_rpm * 2 * np.pi / 60) ** 2
        maximum = (self.max_rpm * 2 * np.pi / 60) ** 2
        rpm = np.sqrt(np.clip(angular_squared, minimum, maximum)) * (60 / (2 * np.pi))
        return np.clip(rpm, self.min_rpm, self.max_rpm)

    def response(self, initial, target, dt):
        """固定命令下一阶滞后的解析解；查询不会改变内部状态。"""
        if not np.isfinite(dt) or dt < 0:
            raise ValueError('响应时间必须有限且非负')
        initial, target = self._rpm(initial), self._rpm(target)
        return target + (initial - target) * np.exp(-dt / self.time_constant)
