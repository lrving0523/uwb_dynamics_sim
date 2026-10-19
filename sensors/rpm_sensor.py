"""真实电机 RPM 的加性高斯测量模型。"""

import numpy as np


def _nonnegative_scalar(value, name):
    """读取有限非负标量。"""
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f'{name} 必须为有限非负标量') from error
    if result.shape != () or not np.isfinite(result) or result < 0:
        raise ValueError(f'{name} 必须为有限非负标量')
    return float(result)


class RPMSensor:
    """测量当前真实电机转速，保留噪声导致的负观测以免改变统计。"""

    def __init__(self, config, rng):
        self.std = _nonnegative_scalar(config.get('std', 0.0), 'RPM 噪声标准差')
        self.rng = rng
        self._last_time = None

    def sample(self, time, true_rpm):
        """返回四个电机的 RPM 观测，不使用控制器目标转速。"""
        time = _nonnegative_scalar(time, '采样时刻')
        if self._last_time is not None and time <= self._last_time:
            raise ValueError('采样时刻必须严格递增')
        try:
            true_rpm = np.asarray(true_rpm, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError('真实 RPM 必须为长度 4 的有限非负向量') from error
        if (true_rpm.shape != (4,) or not np.all(np.isfinite(true_rpm))
                or np.any(true_rpm < 0)):
            raise ValueError('真实 RPM 必须为长度 4 的有限非负向量')
        rpm = true_rpm + self.rng.normal(0.0, self.std, 4)
        self._last_time = time
        return {'time': time, 'rpm': rpm}
