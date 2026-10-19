"""含机体天线杆臂的原始 UWB 距离观测。"""

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


class UWBSensor:
    """按配置中基站的原始顺序返回距离，单位 m。"""

    def __init__(self, config, rng):
        try:
            self._anchors = np.array(config.get('anchors', []), dtype=float, copy=True)
        except (TypeError, ValueError) as error:
            raise ValueError('UWB 基站必须为有限的 N×3 数组') from error
        if (self._anchors.ndim != 2 or self._anchors.shape[0] == 0
                or self._anchors.shape[1] != 3 or not np.all(np.isfinite(self._anchors))):
            raise ValueError('UWB 基站必须为非空、有限的 N×3 数组')
        self._lever_arm = _vector(config.get('lever_arm', [0.0] * 3), 3, '天线杆臂')
        self.std = _nonnegative_scalar(config.get('std', 0.0), 'UWB 噪声标准差')
        self.rng = rng
        self._last_time = None

    @property
    def anchors(self):
        """基站世界坐标副本，行顺序对应距离观测顺序。"""
        return self._anchors.copy()

    @property
    def lever_arm(self):
        """天线相对机体原点的机体系杆臂副本。"""
        return self._lever_arm.copy()

    def sample(self, time, position, quaternion):
        """先旋转杆臂确定天线世界位置，再叠加独立距离白噪声。"""
        time = _nonnegative_scalar(time, '采样时刻')
        if self._last_time is not None and time <= self._last_time:
            raise ValueError('采样时刻必须严格递增')
        position = _vector(position, 3, '机体世界位置')
        quaternion = _vector(quaternion, 4, '四元数')
        tag_position = position + rotation_matrix(quaternion) @ self._lever_arm
        ranges = np.linalg.norm(tag_position - self._anchors, axis=1)
        ranges += self.rng.normal(0.0, self.std, len(self._anchors))
        self._last_time = time
        return {'time': time, 'ranges': ranges}
