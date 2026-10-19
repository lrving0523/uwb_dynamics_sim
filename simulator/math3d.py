"""机体 FLU 到世界 ENU 的 Hamilton 四元数运算，顺序为 wxyz。"""

import numpy as np


def normalize_quaternion(quaternion):
    """归一化有限非零四元数；拒绝无效姿态。"""
    quaternion = np.asarray(quaternion, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError('四元数必须是有限的四维向量')
    norm = np.linalg.norm(quaternion)
    if norm < 1e-15:
        raise ValueError('四元数不能为零')
    return quaternion / norm


def quaternion_multiply(left, right):
    """Hamilton 乘法；不自动归一化，因而也适用于姿态导数。"""
    w1, x1, y1, z1 = np.asarray(left, dtype=float)
    w2, x2, y2, z2 = np.asarray(right, dtype=float)
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quaternion_to_matrix(quaternion):
    """返回将机体向量主动旋转到世界系的旋转矩阵。"""
    w, x, y, z = normalize_quaternion(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def matrix_to_quaternion(matrix):
    """将合法旋转矩阵转成单位四元数，稳定处理接近半周的旋转。"""
    matrix = np.asarray(matrix, dtype=float)
    if (matrix.shape != (3, 3) or not np.all(np.isfinite(matrix))
            or not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-7)):
        raise ValueError('输入必须是正交且行列式为一的旋转矩阵')
    trace = np.trace(matrix)
    if trace > 0:
        scale = 2 * np.sqrt(trace + 1)
        quaternion = np.array([0.25 * scale, (matrix[2, 1] - matrix[1, 2]) / scale,
                               (matrix[0, 2] - matrix[2, 0]) / scale,
                               (matrix[1, 0] - matrix[0, 1]) / scale])
    else:
        index = int(np.argmax(np.diag(matrix)))
        following, last = (index + 1) % 3, (index + 2) % 3
        scale = 2 * np.sqrt(1 + matrix[index, index] - matrix[following, following] - matrix[last, last])
        quaternion = np.zeros(4)
        quaternion[0] = (matrix[last, following] - matrix[following, last]) / scale
        quaternion[index + 1] = 0.25 * scale
        quaternion[following + 1] = (matrix[index, following] + matrix[following, index]) / scale
        quaternion[last + 1] = (matrix[index, last] + matrix[last, index]) / scale
    return normalize_quaternion(quaternion)


def skew(vector):
    """返回叉乘矩阵，使 skew(a) @ b 等于 a × b。"""
    x, y, z = np.asarray(vector, dtype=float)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


rotation_matrix = quaternion_to_matrix
quat_to_matrix = quaternion_to_matrix
quat_multiply = quaternion_multiply
quat_normalize = normalize_quaternion
