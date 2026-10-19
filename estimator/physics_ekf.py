"""RPM 驱动的物理动力学 ESKF，保留当前输入噪声的联合条件分布。"""

import numpy as np

from simulator.math3d import normalize_quaternion, quaternion_multiply, rotation_matrix, skew
from .eskf import NominalState, _std, _symmetric, _time, _vector, measurement_model
from .rotation import exp_quaternion, right_jacobian


def _model_parameters(model):
    """验证动力学接口及以千克、米每平方秒给出的标量参数。"""
    try:
        mass, gravity = float(model.mass), float(model.gravity)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError('动力学模型必须提供有限标量质量和重力') from error
    if (not np.isfinite(mass) or mass <= 0 or not np.isfinite(gravity) or gravity < 0
            or not callable(getattr(model, 'force_body', None))
            or not callable(getattr(model, 'force_jacobians', None))):
        raise ValueError('动力学模型质量必须为正，重力非负，且提供力与力雅可比接口')
    return mass, np.array([0.0, 0.0, -gravity])


def _force_jacobians(model, quaternion, velocity, rpm):
    """校验模型返回的世界速度、右局部姿态和 RPM 雅可比。"""
    jacobians = model.force_jacobians(quaternion, velocity, rpm)
    if len(jacobians) != 3:
        raise ValueError('力雅可比必须包含速度、姿态和四个 RPM 三组矩阵')
    output = []
    for value, shape in zip(jacobians, ((3, 3), (3, 3), (3, 4))):
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != shape or not np.all(np.isfinite(matrix)):
            raise ValueError('力雅可比维度不符或包含非有限数')
        output.append(matrix)
    return tuple(output)


def _midpoint_terms(state, gyro, rpm, model, dt, input_mean):
    """计算离散中点姿态及含条件力误差均值的机体系力。"""
    omega = gyro - state.gyro_bias - input_mean[:3]
    angle = omega * dt
    half_rotation = rotation_matrix(exp_quaternion(angle / 2))
    quaternion_half = normalize_quaternion(quaternion_multiply(
        state.quaternion, exp_quaternion(angle / 2)))
    rotation_half = rotation_matrix(quaternion_half)
    force = _vector(model.force_body(quaternion_half, state.velocity, rpm), 3, '名义机体系力') + input_mean[3:]
    return angle, half_rotation, quaternion_half, rotation_half, force


def dynamics_propagation_matrices(state, gyro, rpm, model, dt, input_mean=None):
    """返回中点动力学的 15 维 Phi 及六维短期辅助误差映射。

    辅助顺序为测量陀螺噪声、真实机体系力减去模型力；前者通过
    减号作用于姿态，后者通过加号作用于平移。模型偏置及外扰不增广。
    """
    dt = _time(dt)
    gyro, rpm = _vector(gyro, 3, '陀螺仪样本'), _vector(rpm, 4, 'RPM 样本')
    mean = np.zeros(6) if input_mean is None else _vector(input_mean, 6, '辅助条件均值')
    mass, _ = _model_parameters(model)
    angle, half_rotation, quaternion_half, rotation_half, force = _midpoint_terms(
        state, gyro, rpm, model, dt, mean)
    velocity_jacobian, theta_jacobian, _ = _force_jacobians(
        model, quaternion_half, state.velocity, rpm)
    acceleration_velocity = rotation_half @ velocity_jacobian / mass
    acceleration_theta_half = (-rotation_half @ skew(force) + rotation_half @ theta_jacobian) / mass
    acceleration_theta = acceleration_theta_half @ half_rotation.T
    acceleration_bias = acceleration_theta_half @ (-right_jacobian(angle / 2) * (dt / 2))
    phi = np.eye(15)
    phi[:3, 3:6] = np.eye(3) * dt + acceleration_velocity * (dt ** 2 / 2)
    phi[3:6, 3:6] += acceleration_velocity * dt
    phi[:3, 6:9] = acceleration_theta * (dt ** 2 / 2)
    phi[3:6, 6:9] = acceleration_theta * dt
    phi[:3, 12:15] = acceleration_bias * (dt ** 2 / 2)
    phi[3:6, 12:15] = acceleration_bias * dt
    phi[6:9, 6:9] = rotation_matrix(exp_quaternion(-angle))
    phi[6:9, 12:15] = -right_jacobian(angle) * dt
    mapping = np.zeros((15, 6))
    mapping[:9, :3] = phi[:9, 12:15]
    mapping[:3, 3:6] = rotation_half / mass * (dt ** 2 / 2)
    mapping[3:6, 3:6] = rotation_half / mass * dt
    return phi, mapping


def accel_measurement_model(state, rpm, model, input_mean=None):
    """返回比力预测、15 维误差雅可比及六维辅助误差雅可比。"""
    rpm = _vector(rpm, 4, 'RPM 样本')
    mean = np.zeros(6) if input_mean is None else _vector(input_mean, 6, '辅助条件均值')
    mass, _ = _model_parameters(model)
    force = _vector(model.force_body(state.quaternion, state.velocity, rpm), 3, '名义机体系力')
    velocity_jacobian, theta_jacobian, _ = _force_jacobians(
        model, state.quaternion, state.velocity, rpm)
    jacobian = np.zeros((3, 15))
    jacobian[:, 3:6], jacobian[:, 6:9] = velocity_jacobian / mass, theta_jacobian / mass
    jacobian[:, 9:12] = np.eye(3)
    auxiliary_jacobian = np.zeros((3, 6))
    auxiliary_jacobian[:, 3:] = np.eye(3) / mass
    return (force + mean[3:]) / mass + state.acc_bias, jacobian, auxiliary_jacobian


class PhysicsEKF:
    """公开 15 维状态误差、内部六维样本辅助量的动力学滤波器。

    模型力误差由同一随机量同时驱动运动预测与比力观测，只有新 RPM
    到达才替换。陀螺噪声只在新陀螺样本到达时替换。固定力方差是给定
    测量历史后的近似，不精确表达学习模型全部历史输入噪声相关性。
    """

    def __init__(self, state, covariance, model, noise, initial_time=0.0):
        self.state = NominalState(state.position, state.velocity, state.quaternion,
                                  state.acc_bias, state.gyro_bias)
        covariance = np.asarray(covariance, dtype=float)
        if (covariance.shape != (15, 15) or not np.all(np.isfinite(covariance))
                or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12)):
            raise ValueError('初始协方差必须是有限对称的 15×15 矩阵')
        self.P = _symmetric(covariance).copy()
        if np.linalg.eigvalsh(self.P).min() < -1e-12 * max(1, np.linalg.norm(self.P, ord=2)):
            raise ValueError('初始协方差必须半正定')
        self.model = model
        self.mass, self.gravity = _model_parameters(model)
        try:
            self._acc_variance = _std(noise['acc_std'], 3, '加速度计样本标准差') ** 2
            self._gyro_variance = _std(noise['gyro_std'], 3, '陀螺仪样本标准差') ** 2
            self._rpm_variance = _std(noise['rpm_std'], 4, 'RPM 样本标准差') ** 2
            self._force_variance = _std(noise['model_force_std'], 3, '模型力误差标准差') ** 2
            self._bias_variance = np.r_[
                _std(noise['acc_bias_rw_std'], 3, '加速度偏置随机游走强度'),
                _std(noise['gyro_bias_rw_std'], 3, '陀螺偏置随机游走强度')] ** 2
        except (KeyError, TypeError) as error:
            raise ValueError('噪声配置缺少规定的标准差字段') from error
        self.time = _time(initial_time)
        self._last_gyro_time = self._last_rpm_time = None
        self._gyro = self._rpm = None
        self._input_mean = np.zeros(6)
        self._input_covariance = np.zeros((6, 6))
        self._state_input_covariance = np.zeros((15, 6))

    def _validate_sample_time(self, time, previous, label):
        """同类样本严格递增，异步输入只允许在当前时刻切换。"""
        time = _time(time)
        if time != self.time:
            raise ValueError(f'{label}样本只能在滤波器当前时刻接收，不能提前或回填')
        if previous is not None and time <= previous:
            raise ValueError(f'{label}样本时间必须严格递增')
        return time

    def _replace_auxiliary(self, offset, covariance):
        """替换一个独立新样本，同时保留另一辅助量完整边缘分布。"""
        block = slice(offset, offset + 3)
        self._input_mean[block] = 0
        self._input_covariance[block, :] = 0
        self._input_covariance[:, block] = 0
        self._input_covariance[block, block] = covariance
        self._state_input_covariance[:, block] = 0

    def set_gyro(self, time, gyro):
        """接收当前时刻陀螺样本，仅替换三维陀螺辅助噪声。"""
        time = self._validate_sample_time(time, self._last_gyro_time, '陀螺仪')
        gyro = _vector(gyro, 3, '陀螺仪样本')
        self._gyro, self._last_gyro_time = gyro, time
        self._replace_auxiliary(0, np.diag(self._gyro_variance))

    def set_rpm(self, time, rpm):
        """接收当前 RPM，将 RPM 测量方差映射到当前模型力先验。"""
        time = self._validate_sample_time(time, self._last_rpm_time, 'RPM')
        rpm = _vector(rpm, 4, 'RPM 样本')
        _, _, mapping = _force_jacobians(self.model, self.state.quaternion, self.state.velocity, rpm)
        covariance = np.diag(self._force_variance) + (mapping * self._rpm_variance) @ mapping.T
        self._rpm, self._last_rpm_time = rpm, time
        self._replace_auxiliary(3, _symmetric(covariance))

    def _require_inputs(self):
        """传播及比力更新都必须有已到达的陀螺和 RPM。"""
        if self._gyro is None or self._rpm is None:
            raise ValueError('动力学传播及比力更新前必须均收到陀螺与 RPM 样本')

    def _predict_state(self, state, dt):
        """按当前条件输入均值进行一次名义中点积分。"""
        angle, _, _, rotation_half, force = _midpoint_terms(
            state, self._gyro, self._rpm, self.model, dt, self._input_mean)
        acceleration = rotation_half @ force / self.mass + self.gravity
        return NominalState(
            state.position + state.velocity * dt + acceleration * (dt ** 2 / 2),
            state.velocity + acceleration * dt,
            quaternion_multiply(state.quaternion, exp_quaternion(angle)),
            state.acc_bias, state.gyro_bias)

    def _bias_process_covariance(self, dt):
        """三点 Gauss–Legendre 正权求积近似连续偏置随机游走。

        节点状态和剩余传播均使用名义中点积分；随姿态/速度变化的通道
        是求积近似而非精确连续离散化。每一项为 G diag(q) Gᵀ，保证
        所得过程协方差半正定。加速度偏置只影响观测，不驱动运动预测。
        """
        covariance = np.zeros((15, 15))
        if not np.any(self._bias_variance):
            return covariance
        for node, weight in zip((-np.sqrt(3 / 5), 0, np.sqrt(3 / 5)), (5 / 9, 8 / 9, 5 / 9)):
            elapsed = dt * (node + 1) / 2
            intermediate = self._predict_state(self.state, elapsed)
            phi, _ = dynamics_propagation_matrices(
                intermediate, self._gyro, self._rpm, self.model, dt - elapsed, self._input_mean)
            mapping = phi[:, 9:15]
            covariance += (mapping * self._bias_variance) @ mapping.T * (dt * weight / 2)
        return covariance

    def propagate_to(self, time):
        """零阶保持已到达输入；联合传播保留同一样本跨区间相关性。"""
        time = _time(time)
        if time < self.time:
            raise ValueError('不能向过去传播')
        self._require_inputs()
        dt = time - self.time
        if dt == 0:
            return
        phi, mapping = dynamics_propagation_matrices(
            self.state, self._gyro, self._rpm, self.model, dt, self._input_mean)
        cross = phi @ self._state_input_covariance @ mapping.T
        self.P = _symmetric(phi @ self.P @ phi.T + cross + cross.T
                            + mapping @ self._input_covariance @ mapping.T
                            + self._bias_process_covariance(dt))
        self._state_input_covariance = phi @ self._state_input_covariance + mapping @ self._input_covariance
        self.state = self._predict_state(self.state, dt)
        self.time = time

    def _joint_update(self, innovation, jacobian, auxiliary_jacobian, measurement_covariance):
        """完整 21 维 Joseph 条件化，注入均值后重置全部姿态交叉块。"""
        covariance = np.block([[self.P, self._state_input_covariance],
                               [self._state_input_covariance.T, self._input_covariance]])
        observation = np.column_stack((jacobian, auxiliary_jacobian))
        innovation_covariance = _symmetric(observation @ covariance @ observation.T + measurement_covariance)
        gain = np.linalg.solve(innovation_covariance, observation @ covariance).T
        delta = gain @ innovation
        remainder = np.eye(21) - gain @ observation
        posterior = _symmetric(remainder @ covariance @ remainder.T
                               + gain @ measurement_covariance @ gain.T)
        self.state.position += delta[:3]
        self.state.velocity += delta[3:6]
        self.state.quaternion = normalize_quaternion(quaternion_multiply(
            self.state.quaternion, exp_quaternion(delta[6:9])))
        self.state.acc_bias += delta[9:12]
        self.state.gyro_bias += delta[12:15]
        self._input_mean += delta[15:]
        reset = np.eye(21)
        reset[6:9, 6:9] = right_jacobian(delta[6:9])
        posterior = _symmetric(reset @ posterior @ reset.T)
        self.P = posterior[:15, :15].copy()
        self._state_input_covariance = posterior[:15, 15:].copy()
        self._input_covariance = posterior[15:, 15:].copy()
        return dict(innovation=innovation, innovation_covariance=innovation_covariance,
                    nis=float(innovation @ np.linalg.solve(innovation_covariance, innovation)))

    def update_accel(self, accel):
        """加速度计比力只作一次观测；独立量测噪声仅含加速度白噪声。"""
        self._require_inputs()
        accel = _vector(accel, 3, '加速度计比力')
        prediction, jacobian, auxiliary_jacobian = accel_measurement_model(
            self.state, self._rpm, self.model, self._input_mean)
        return self._joint_update(accel - prediction, jacobian, auxiliary_jacobian,
                                  np.diag(self._acc_variance))

    def update_uwb(self, ranges, anchors, lever_arm, std):
        """沿用原始距离观测和杆臂几何，有限负测距作为原始量测保留。"""
        prediction, jacobian = measurement_model(self.state, anchors, lever_arm)
        ranges = _vector(ranges, len(prediction), '原始距离')
        covariance = np.diag(_std(std, len(ranges), '测距标准差', positive=True) ** 2)
        return self._joint_update(ranges - prediction, jacobian,
                                  np.zeros((len(ranges), 6)), covariance)
