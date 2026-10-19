"""世界系外力与空气速度；随机节点预先生成，查询不消耗随机数。"""

import numpy as np


class WindField:
    """支持关闭、恒定、阶跃、正弦及固定时间节点线性插值随机扰动。"""

    def __init__(self, config, rng, duration):
        self.enabled = bool(config.get('enabled', False))
        self.kind = config.get('type', 'none')
        if self.kind not in {'none', 'constant', 'step', 'sine', 'random'}:
            raise ValueError(f'未知扰动类型：{self.kind}')
        self.amplitude = np.asarray(config.get('force', [0, 0, 0]), dtype=float)
        self.velocity_amplitude = np.asarray(config.get('velocity', [0, 0, 0]), dtype=float)
        self.start_time = float(config.get('start_time', 0.0))
        self.frequency = float(config.get('frequency', 0.35))
        self.phase = float(config.get('phase', 0.0))
        self.random_hz = float(config.get('random_hz', 10.0))
        std = np.asarray(config.get('random_std', [0.3, 0.3, 0.15]), dtype=float)
        velocity_std = np.asarray(config.get('velocity_std', [0, 0, 0]), dtype=float)
        vectors = [self.amplitude, self.velocity_amplitude, std, velocity_std]
        if (any(vector.shape != (3,) or not np.all(np.isfinite(vector)) for vector in vectors)
                or np.any(std < 0) or np.any(velocity_std < 0)
                or not np.all(np.isfinite([duration, self.start_time, self.frequency, self.phase, self.random_hz]))
                or duration < 0 or self.start_time < 0 or self.frequency < 0 or self.random_hz <= 0):
            raise ValueError('扰动配置必须有限，幅值为三维、标准差非负、节点频率为正')
        count = max(2, int(np.ceil(max(0.0, duration - self.start_time) * self.random_hz)) + 2)
        self.node_times = self.start_time + np.arange(count) / self.random_hz
        self.force_nodes = np.zeros((count, 3))
        self.velocity_nodes = np.zeros((count, 3))
        if self.enabled and self.kind == 'random':
            self.force_nodes = rng.normal(0, std, size=(count, 3))
            if np.any(velocity_std):
                self.velocity_nodes = rng.normal(0, velocity_std, size=(count, 3))

    def _value(self, time, amplitude, nodes):
        if not np.isfinite(time):
            raise ValueError('扰动查询时间必须有限')
        if not self.enabled or self.kind == 'none':
            return np.zeros(3)
        if self.kind == 'constant':
            return amplitude.copy()
        if time < self.start_time:
            return np.zeros(3)
        if self.kind == 'step':
            return amplitude.copy()
        if self.kind == 'sine':
            return amplitude * np.sin(2 * np.pi * self.frequency * (time - self.start_time) + self.phase)
        return np.array([np.interp(time, self.node_times, nodes[:, axis]) for axis in range(3)])

    def force(self, time):
        """独立未知世界外力，单位 N；随机模式为零均值节点过程。"""
        return self._value(time, self.amplitude, self.force_nodes)

    def velocity(self, time):
        """附加世界空气速度，单位 m/s；缺省始终为零。"""
        return self._value(time, self.velocity_amplitude, self.velocity_nodes)
