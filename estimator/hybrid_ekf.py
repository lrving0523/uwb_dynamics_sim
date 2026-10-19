"""使用持有的学习残余力的 Hybrid-EKF，复用相同联合误差滤波算法。"""

from .physics_ekf import PhysicsEKF


class HybridEKF(PhysicsEKF):
    """仅替换动力学模型，不改变物理滤波器的参数与更新规则。"""

    def __init__(self, state, covariance, model, noise, initial_time=0.0):
        from models.hybrid import HybridModel

        if not isinstance(model, HybridModel):
            raise ValueError('HybridEKF 必须使用 HybridModel 模型')
        super().__init__(state, covariance, model, noise, initial_time=initial_time)
