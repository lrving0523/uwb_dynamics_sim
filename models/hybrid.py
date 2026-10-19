"""把已收到的测量历史残余作为给定输入叠加到名义模型。"""

import numpy as np

from simulator.math3d import quaternion_to_matrix


class HybridModel:
    """保持最新残余力；不在状态扰动时重新查询网络。"""

    def __init__(self, physics, predictor):
        self.physics = physics
        self.predictor = predictor
        self.mass, self.gravity = physics.mass, physics.gravity
        self._residual_force = np.zeros(3)

    @property
    def residual_force(self):
        """返回副本，防止调用方意外修改当前条件输入。"""
        return self._residual_force.copy()

    def update_residual(self, window):
        """满因果窗口后更新；前缀不足时严格使用零残余。"""
        values = np.asarray(window, dtype=float)
        if values.ndim != 2 or values.shape[1] != 7 or not np.all(np.isfinite(values)):
            raise ValueError('历史窗口必须为有限 T×7 测量矩阵')
        if len(values) < self.predictor.sequence_length:
            self._residual_force = np.zeros(3)
        else:
            prediction = np.asarray(self.predictor.predict(values[-self.predictor.sequence_length:]), dtype=float)
            if prediction.shape != (3,) or not np.all(np.isfinite(prediction)):
                raise ValueError('残余预测必须是有限三维力')
            self._residual_force = prediction.copy()
        return self.residual_force

    def force_body(self, quaternion, velocity, rpm):
        """混合力为名义力与已持有气动残余之和。"""
        return self.physics.force_body(quaternion, velocity, rpm)+self._residual_force

    def acceleration(self, quaternion, velocity, rpm):
        """将混合机体系力转至世界系并加上重力。"""
        return quaternion_to_matrix(quaternion) @ self.force_body(quaternion, velocity, rpm)/self.mass+np.array([0., 0., -self.gravity])

    def force_jacobians(self, quaternion, velocity, rpm):
        """当前网络预测视作给定条件输入，沿用名义力导数。"""
        return self.physics.force_jacobians(quaternion, velocity, rpm)
