"""带解析速度和加速度的五类确定性轨迹参考。"""

import numpy as np


class Trajectory:
    """所有轨迹从给定初始位置出发，世界系采用 SI 单位。"""

    def __init__(self, config, initial_position):
        self.kind = config.get('type', 'hover')
        if self.kind not in {'hover', 'circle', 'figure_eight', 'variable_speed', 'accel_decel'}:
            raise ValueError(f'未知轨迹类型：{self.kind}')
        self.initial_position = np.asarray(initial_position, dtype=float).copy()
        self.radius = float(config.get('radius', 1.2))
        self.period = float(config.get('period', 8.0))
        self.yaw = float(config.get('yaw', 0.0))
        self.distance = float(config.get('distance', 2 * self.radius))
        self.direction = np.asarray(config.get('direction', [1, 0, 0]), dtype=float)
        if (self.initial_position.shape != (3,) or self.direction.shape != (3,)
                or not np.all(np.isfinite(self.initial_position)) or not np.all(np.isfinite(self.direction))
                or not np.all(np.isfinite([self.radius, self.period, self.yaw, self.distance]))
                or self.radius < 0 or self.period <= 0 or np.linalg.norm(self.direction) == 0):
            raise ValueError('轨迹参数必须有限，周期为正、半径非负、方向非零')
        self.direction = self.direction / np.linalg.norm(self.direction)

    def sample(self, time):
        """返回位置、速度、加速度以及固定偏航角，分别为 m、m/s、m/s²、rad。"""
        if not np.isfinite(time) or time < 0:
            raise ValueError('轨迹查询时间必须有限且非负')
        offset, velocity, acceleration = np.zeros(3), np.zeros(3), np.zeros(3)
        frequency = 2 * np.pi / self.period
        phase = frequency * time
        if self.kind in {'circle', 'variable_speed'}:
            phase_rate, phase_acceleration = frequency, 0.0
            if self.kind == 'variable_speed':
                phase_rate = frequency * (1 + 0.4 * np.cos(phase))
                phase_acceleration = -0.4 * frequency ** 2 * np.sin(phase)
                phase += 0.4 * np.sin(phase)
            sine, cosine = np.sin(phase), np.cos(phase)
            offset = self.radius * np.array([sine, 1 - cosine, 0])
            first = self.radius * np.array([cosine, sine, 0])
            second = self.radius * np.array([-sine, cosine, 0])
            velocity = first * phase_rate
            acceleration = second * phase_rate ** 2 + first * phase_acceleration
        elif self.kind == 'figure_eight':
            offset = self.radius * np.array([np.sin(phase), 0.5 * np.sin(2 * phase), 0])
            velocity = self.radius * frequency * np.array([np.cos(phase), np.cos(2 * phase), 0])
            acceleration = -self.radius * frequency ** 2 * np.array([np.sin(phase), 2 * np.sin(2 * phase), 0])
        elif self.kind == 'accel_decel':
            progress = min(time / self.period, 1.0)
            offset = self.distance * self.direction * (10 * progress ** 3 - 15 * progress ** 4 + 6 * progress ** 5)
            if time < self.period:
                velocity = self.distance * self.direction / self.period * (30 * progress ** 2 - 60 * progress ** 3 + 30 * progress ** 4)
                acceleration = self.distance * self.direction / self.period ** 2 * (60 * progress - 180 * progress ** 2 + 120 * progress ** 3)
        return {'position': self.initial_position + offset, 'velocity': velocity,
                'acceleration': acceleration, 'yaw': self.yaw}
