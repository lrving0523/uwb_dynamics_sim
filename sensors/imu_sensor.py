"""机体系比力和角速度观测，含测量白噪声与偏置随机游走。"""

import numpy as np

from simulator.math3d import rotation_matrix


def _vector(value, size, name):
    """读取有限向量并隔离调用方数组。"""
    try:
        result = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} 必须为有限数值向量') from error
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} 必须为长度 {size} 的有限向量')
    return result


def _nonnegative_scalar(value, name):
    """读取有限非负标量。"""
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} 必须为有限非负标量') from error
    if result.shape != () or not np.isfinite(result) or result < 0:
        raise ValueError(f'{name} 必须为有限非负标量')
    return float(result)


class IMUSensor:
    """以 m/s² 和 rad/s 返回观测；偏置真值通过独立属性读取。"""

    def __init__(self, config, rng):
        self.acc_std = _nonnegative_scalar(config.get('acc_std', 0.0), '加速度噪声标准差')
        self.gyro_std = _nonnegative_scalar(config.get('gyro_std', 0.0), '角速度噪声标准差')
        self.acc_bias_rw_std = _nonnegative_scalar(
            config.get('acc_bias_rw_std', 0.0), '加速度偏置随机游走强度'
        )
        self.gyro_bias_rw_std = _nonnegative_scalar(
            config.get('gyro_bias_rw_std', 0.0), '角速度偏置随机游走强度'
        )
        self._acc_bias = _vector(config.get('acc_bias', [0.0] * 3), 3, '加速度初始偏置')
        self._gyro_bias = _vector(config.get('gyro_bias', [0.0] * 3), 3, '角速度初始偏置')
        self.rng = rng
        self._last_time = None

    @property
    def acc_bias(self):
        """当前加速度偏置真值副本，单位 m/s²。"""
        return self._acc_bias.copy()

    @property
    def gyro_bias(self):
        """当前角速度偏置真值副本，单位 rad/s。"""
        return self._gyro_bias.copy()

    def sample(self, time, quaternion, acceleration, omega, gravity):
        """世界加速度减去重力后旋转至机体系，使用实际时间间隔演化偏置。"""
        time = _nonnegative_scalar(time, '采样时刻')
        if self._last_time is not None and time <= self._last_time:
            raise ValueError('采样时刻必须严格递增')
        quaternion = _vector(quaternion, 4, '四元数')
        acceleration = _vector(acceleration, 3, '世界加速度')
        omega = _vector(omega, 3, '机体角速度')
        gravity = _vector(gravity, 3, '世界重力加速度')
        rotation = rotation_matrix(quaternion)
        if self._last_time is not None:
            root_dt = np.sqrt(time - self._last_time)
            self._acc_bias += self.rng.normal(0.0, self.acc_bias_rw_std * root_dt, 3)
            self._gyro_bias += self.rng.normal(0.0, self.gyro_bias_rw_std * root_dt, 3)
        accel = rotation.T @ (acceleration - gravity) + self._acc_bias
        gyro = omega + self._gyro_bias
        accel += self.rng.normal(0.0, self.acc_std, 3)
        gyro += self.rng.normal(0.0, self.gyro_std, 3)
        self._last_time = time
        return {'time': time, 'accel': accel, 'gyro': gyro}
