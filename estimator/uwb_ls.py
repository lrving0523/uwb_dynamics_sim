"""仅依赖单时刻原始距离的三维天线位置最小二乘基线。"""

import numpy as np


def _positive_scalar(value, name):
    """拒绝布尔量、向量及非有限的正标量参数。"""
    if isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0:
        raise ValueError(f'{name} 必须是有限正标量')
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f'{name} 必须是有限正标量') from error
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f'{name} 必须是有限正标量')
    return value


def _range_model(position, anchors):
    """计算距离及位置雅可比，零距离处明确拒绝线性化。"""
    differences = position - anchors
    distances = np.linalg.norm(differences, axis=1)
    if not np.all(np.isfinite(distances)):
        raise ValueError('测距模型超出有限数值范围')
    # 平方距离初始化的舍入误差不能把实际零距离伪装成可线性化测量。
    scale = max(1.0, np.max(np.abs(anchors)), np.max(np.abs(position)))
    if np.any(distances <= 32.0 * np.finfo(float).eps * scale):
        raise ValueError('天线位置与基站重合，零距离处的测距雅可比未定义')
    return distances, differences / distances[:, None]


def solve_uwb_position(
    anchors, ranges, std=0.05, initial=None, max_iterations=50, tolerance=1e-8
):
    """用固定独立高斯噪声模型求解单时刻的天线位置。

    基站必须为至少四个非共面三维点，测距顺序与基站一致。允许
    高斯噪声造成的个别负距离，目标始终为原始测距残差平方和。
    ``initial`` 只能来自外部先验或已经得到的历史位置，不需要姿态。

    返回的 ``residual`` 定义为测量减预测；``covariance`` 固定为
    ``std**2 * (J.T @ J)**-1``，不使用残差估计或缩放噪声方差。
    ``condition_number`` 为最终 ``J.T @ J`` 的二范数条件数，超过
    1e12 时拒绝报告协方差；位置重合或几何退化也抛出 ValueError。

    ``iterations`` 为执行的高斯牛顿迭代次数；``converged`` 只在
    一阶最优性残差或未缩放的位置修正满足容差时为真。迭代耗尽或
    回溯不能继续下降时返回假，调用者应检查该标志。协方差只是
    返回位置的局部线性化结果，不代表全局最优或迭代成功。
    """
    std = _positive_scalar(std, '测距标准差')
    tolerance = _positive_scalar(tolerance, '收敛容差')
    if (isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, (int, np.integer))
            or max_iterations < 1):
        raise ValueError('最大迭代次数必须是正整数')
    try:
        anchors = np.asarray(anchors, dtype=float)
        ranges = np.asarray(ranges, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError('基站和测距必须是有限实数数组') from error
    if anchors.ndim != 2 or anchors.shape[1:] != (3,) or len(anchors) < 4:
        raise ValueError('基站形状必须为 (A, 3)，且 A 至少为四')
    if ranges.shape != (len(anchors),):
        raise ValueError('测距形状必须为 (A,)，并与基站顺序一致')
    if not np.all(np.isfinite(anchors)) or not np.all(np.isfinite(ranges)):
        raise ValueError('基站和测距必须全部有限')
    if not np.any(ranges > 0.0):
        raise ValueError('全为非正值的测距不能提供有效定位信息')

    # 使用局部坐标减少大世界坐标造成的平方距离差消减误差。
    origin = anchors.mean(axis=0)
    local_anchors = anchors - origin
    if np.linalg.matrix_rank(local_anchors) != 3:
        raise ValueError('基站仿射几何的秩必须为三，不能共面或共线')
    if initial is None:
        squared_anchors = np.sum(local_anchors * local_anchors, axis=1)
        right = squared_anchors[1:] - squared_anchors[0]
        right = right - ranges[1:] ** 2 + ranges[0] ** 2
        position = np.linalg.lstsq(
            2.0 * (local_anchors[1:] - local_anchors[0]), right, rcond=None
        )[0]
    else:
        try:
            initial = np.asarray(initial, dtype=float)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError('初始位置必须是有限三维向量') from error
        if initial.shape != (3,) or not np.all(np.isfinite(initial)):
            raise ValueError('初始位置必须是有限三维向量')
        position = initial - origin
    if not np.all(np.isfinite(position)):
        raise ValueError('初始定位超出有限数值范围')

    converged = False
    iterations = 0
    for iterations in range(1, int(max_iterations) + 1):
        predicted, jacobian = _range_model(position, local_anchors)
        residual = ranges - predicted
        gradient = jacobian.T @ residual
        if np.linalg.norm(gradient, ord=np.inf) <= tolerance * max(
            1.0, np.linalg.norm(residual)
        ):
            converged = True
            break
        step = np.linalg.lstsq(jacobian, residual, rcond=None)[0]
        if np.linalg.norm(step) <= tolerance * (1.0 + np.linalg.norm(position)):
            converged = True
            break

        # 回溯只控制数值步长，不更改固定高斯权重或原始残差。
        cost = float(residual @ residual)
        scale = 1.0
        for _ in range(40):
            candidate = position + scale * step
            candidate_distance = np.linalg.norm(candidate - local_anchors, axis=1)
            candidate_residual = ranges - candidate_distance
            candidate_cost = float(candidate_residual @ candidate_residual)
            if (np.all(np.isfinite(candidate_distance))
                    and np.all(candidate_distance > 0.0)
                    and candidate_cost < cost):
                position = candidate
                break
            scale *= 0.5
        else:
            break

    predicted, jacobian = _range_model(position, local_anchors)
    residual = ranges - predicted
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    if singular_values[-1] <= 0.0:
        raise ValueError('测距雅可比退化，无法计算可靠协方差')
    condition_number = float((singular_values[0] / singular_values[-1]) ** 2)
    if not np.isfinite(condition_number) or condition_number > 1e12:
        raise ValueError('测距雅可比条件数过大，几何病态，无法计算可靠协方差')
    covariance = std ** 2 * np.linalg.inv(jacobian.T @ jacobian)
    covariance = 0.5 * (covariance + covariance.T)
    if not np.all(np.isfinite(covariance)):
        raise ValueError('固定测距方差超出可计算的协方差范围')
    return {
        'position': position + origin,
        'covariance': covariance,
        'residual': residual,
        'iterations': iterations,
        'converged': converged,
        'condition_number': condition_number,
    }
