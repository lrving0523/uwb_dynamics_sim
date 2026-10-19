"""SO(3) 旋转向量与右雅可比；Hamilton 四元数顺序为 wxyz。"""

import numpy as np

from simulator.math3d import normalize_quaternion, skew


def _rotvec(vector):
    vector = np.asarray(vector, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError('旋转向量必须是有限的三维向量')
    return vector


def exp_quaternion(rotvec):
    """返回旋转向量的单位四元数，小角度使用连续的级数展开。"""
    vector = _rotvec(rotvec)
    angle_squared = vector @ vector
    if angle_squared < 1e-12:
        scale = 0.5 - angle_squared / 48 + angle_squared ** 2 / 3840
        real = 1 - angle_squared / 8 + angle_squared ** 2 / 384
    else:
        angle = np.sqrt(angle_squared)
        scale, real = np.sin(angle / 2) / angle, np.cos(angle / 2)
    return normalize_quaternion(np.r_[real, vector * scale])


def log_quaternion(quaternion):
    """返回模长不超过 π 的主值旋转向量，等价的 q 与 -q 取相同结果。"""
    quaternion = normalize_quaternion(quaternion)
    if (quaternion[0] < 0 or (quaternion[0] == 0
            and quaternion[np.argmax(np.abs(quaternion))] < 0)):
        quaternion = -quaternion
    sine = np.linalg.norm(quaternion[1:])
    if sine < 1e-8:
        return quaternion[1:] * (2 + sine ** 2 / 3)
    return quaternion[1:] * (2 * np.arctan2(sine, quaternion[0]) / sine)


def right_jacobian(rotvec):
    """满足 Exp(φ+dφ)≈Exp(φ) Exp(Jr(φ)dφ) 的 SO(3) 右雅可比。"""
    vector = _rotvec(rotvec)
    angle_squared = vector @ vector
    cross = skew(vector)
    if angle_squared < 1e-8:
        first = 0.5 - angle_squared / 24 + angle_squared ** 2 / 720
        second = 1 / 6 - angle_squared / 120 + angle_squared ** 2 / 5040
    else:
        angle = np.sqrt(angle_squared)
        first = (1 - np.cos(angle)) / angle_squared
        second = (angle - np.sin(angle)) / (angle_squared * angle)
    return np.eye(3) - first * cross + second * cross @ cross
