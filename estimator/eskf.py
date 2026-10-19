"""右乘局部姿态误差的 15 维 UWB–IMU ESKF，不读取任何仿真真值。"""

from dataclasses import dataclass

import numpy as np

from simulator.math3d import (normalize_quaternion, quaternion_multiply,
                              rotation_matrix, skew)
from .rotation import exp_quaternion, right_jacobian


def _vector(value, size, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (size,) or not np.all(np.isfinite(array)):
        raise ValueError(f'{name}必须是有限的{size}维向量')
    return array.copy()


def _std(value, size, name, positive=False):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        array = np.full(size, float(array))
    array = _vector(array, size, name)
    if np.any(array <= 0 if positive else array < 0):
        raise ValueError(f'{name}必须为正数' if positive else f'{name}不能为负数')
    return array


def _time(value):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError('时间必须是非负有限数')
    return value


def _symmetric(matrix):
    return (matrix + matrix.T) * 0.5


@dataclass
class NominalState:
    """世界系位置/速度、机体到世界姿态，以及机体系加速度计/陀螺偏置。"""

    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    acc_bias: np.ndarray
    gyro_bias: np.ndarray

    def __post_init__(self):
        self.position = _vector(self.position, 3, '位置')
        self.velocity = _vector(self.velocity, 3, '速度')
        self.quaternion = normalize_quaternion(self.quaternion)
        self.acc_bias = _vector(self.acc_bias, 3, '加速度计偏置')
        self.gyro_bias = _vector(self.gyro_bias, 3, '陀螺仪偏置')


def propagation_matrices(state, accel, gyro, dt):
    """返回中点姿态离散积分的精确一阶雅可比 Phi 与样本噪声映射 B。

    误差顺序为 p、v、theta、ba、bg。输入 accel/gyro 已扣除样本噪声
    的条件均值，但尚未扣除名义偏置。B 对应测量中的加性噪声。
    """
    dt = _time(dt)
    force = _vector(accel, 3, '加速度计样本') - state.acc_bias
    omega = _vector(gyro, 3, '陀螺仪样本') - state.gyro_bias
    angle = omega * dt
    half_rotation = rotation_matrix(exp_quaternion(angle * 0.5))
    rotation_half = rotation_matrix(state.quaternion) @ half_rotation
    phi = np.eye(15)
    phi[0:3, 3:6] = np.eye(3) * dt
    phi[3:6, 6:9] = -rotation_half @ skew(force) @ half_rotation.T * dt
    phi[3:6, 9:12] = -rotation_half * dt
    phi[3:6, 12:15] = rotation_half @ skew(force) @ right_jacobian(angle * 0.5) * (dt ** 2 / 2)
    phi[0:3, 6:15] = phi[3:6, 6:15] * (dt / 2)
    phi[6:9, 6:9] = rotation_matrix(exp_quaternion(-angle))
    phi[6:9, 12:15] = -right_jacobian(angle) * dt
    noise_mapping = np.zeros((15, 6))
    noise_mapping[:9] = phi[:9, 9:15]
    return phi, noise_mapping


def measurement_model(state, anchors, lever_arm):
    """返回原始距离与雅可比，杆臂使用机体系坐标，支持单个或多个基站。"""
    anchors = np.asarray(anchors, dtype=float)
    if (anchors.ndim != 2 or anchors.shape[1] != 3 or len(anchors) == 0
            or not np.all(np.isfinite(anchors))):
        raise ValueError('基站必须是非空的有限 N×3 数组')
    lever_arm = _vector(lever_arm, 3, '杆臂')
    rotation = rotation_matrix(state.quaternion)
    displacement = state.position + rotation @ lever_arm - anchors
    ranges = np.linalg.norm(displacement, axis=1)
    if np.any(ranges <= 1e-12):
        raise ValueError('天线与基站重合，距离雅可比未定义')
    directions = displacement / ranges[:, None]
    jacobian = np.zeros((len(anchors), 15))
    jacobian[:, :3] = directions
    jacobian[:, 6:9] = -directions @ rotation @ skew(lever_arm)
    return ranges, jacobian


class UWBIMUESKF:
    """由已到达的零阶保持 IMU 驱动，公开协方差始终为 15×15。

    每个真实 IMU 样本保留一次六维白噪声的条件均值、协方差及其与误差
    状态的交叉协方差。跨多个测距事件传播时，不能把它重复视作独立噪声。
    这些辅助量仅活到下一次 set_imu，不属于持久的运动状态。
    """

    def __init__(self, state, covariance, gravity, noise, initial_time=0.0):
        self.state = NominalState(state.position, state.velocity, state.quaternion,
                                  state.acc_bias, state.gyro_bias)
        covariance = np.asarray(covariance, dtype=float)
        if (covariance.shape != (15, 15) or not np.all(np.isfinite(covariance))
                or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12)):
            raise ValueError('初始协方差必须是有限对称的 15×15 矩阵')
        self.P = _symmetric(covariance).copy()
        if np.linalg.eigvalsh(self.P).min() < -1e-12 * max(1, np.linalg.norm(self.P, ord=2)):
            raise ValueError('初始协方差必须半正定')
        gravity = np.asarray(gravity, dtype=float)
        if gravity.ndim == 0:
            if not np.isfinite(gravity) or gravity < 0:
                raise ValueError('重力大小必须是非负有限数')
            gravity = np.array([0, 0, -float(gravity)])
        self.gravity = _vector(gravity, 3, '世界系重力')
        try:
            self._sample_variance = np.r_[
                _std(noise['acc_std'], 3, '加速度计样本标准差'),
                _std(noise['gyro_std'], 3, '陀螺仪样本标准差')] ** 2
            self._bias_variance = np.r_[
                _std(noise['acc_bias_rw_std'], 3, '加速度偏置随机游走强度'),
                _std(noise['gyro_bias_rw_std'], 3, '陀螺偏置随机游走强度')] ** 2
        except (KeyError, TypeError) as error:
            raise ValueError('噪声配置缺少规定的标准差字段') from error
        self.time = _time(initial_time)
        self._last_sample_time = None
        self._accel = self._gyro = None
        self._input_mean = np.zeros(6)
        self._input_covariance = np.zeros((6, 6))
        self._state_input_covariance = np.zeros((15, 6))

    def set_imu(self, time, accel, gyro):
        """在当前时刻切换未来区间的输入；同一时刻不得重复接收新样本。"""
        time = _time(time)
        if time != self.time:
            raise ValueError('IMU 样本只能在滤波器当前时刻接收，不能回填或提前使用')
        if self._last_sample_time is not None and time <= self._last_sample_time:
            raise ValueError('IMU 样本时间必须严格递增')
        accel, gyro = _vector(accel, 3, '加速度计样本'), _vector(gyro, 3, '陀螺仪样本')
        self._accel, self._gyro = accel, gyro
        self._last_sample_time = time
        self._input_mean = np.zeros(6)
        self._input_covariance = np.diag(self._sample_variance)
        self._state_input_covariance = np.zeros((15, 6))

    def _bias_process_covariance(self, accel, gyro, dt):
        """用三点 Gauss–Legendre 正权求积近似连续偏置随机游走积分。

        每个节点将偏置扰动通过剩余区间的中点离散雅可比输送至终点。
        该近似始终半正定；无转动时，加速度偏置平移通道可精确求积。
        陀螺偏置经比力耦合到平移仍为中点近似；转动时也属于近似，
        不能视作精确连续离散化，较长传播区间需要进一步子步验证。
        """
        covariance = np.zeros((15, 15))
        if not np.any(self._bias_variance):
            return covariance
        nodes = (-np.sqrt(3 / 5), 0, np.sqrt(3 / 5))
        weights = (5 / 9, 8 / 9, 5 / 9)
        omega = gyro - self.state.gyro_bias
        for node, weight in zip(nodes, weights):
            elapsed = dt * (node + 1) / 2
            intermediate = NominalState(
                self.state.position, self.state.velocity,
                quaternion_multiply(self.state.quaternion, exp_quaternion(omega * elapsed)),
                self.state.acc_bias, self.state.gyro_bias)
            phi, _ = propagation_matrices(intermediate, accel, gyro, dt - elapsed)
            mapping = phi[:, 9:15]
            covariance += (mapping * self._bias_variance) @ mapping.T * (dt * weight / 2)
        return covariance

    def propagate_to(self, time):
        """将当前样本保持到目标时刻；相同时刻为空操作，拒绝逆时传播。"""
        time = _time(time)
        if time < self.time:
            raise ValueError('不能向过去传播')
        dt = time - self.time
        if dt == 0:
            return
        if self._accel is None:
            raise ValueError('传播前必须收到当前或过去的 IMU 样本')
        accel = self._accel - self._input_mean[:3]
        gyro = self._gyro - self._input_mean[3:]
        phi, mapping = propagation_matrices(self.state, accel, gyro, dt)
        cross = phi @ self._state_input_covariance @ mapping.T
        self.P = _symmetric(phi @ self.P @ phi.T + cross + cross.T
                            + mapping @ self._input_covariance @ mapping.T
                            + self._bias_process_covariance(accel, gyro, dt))
        self._state_input_covariance = (phi @ self._state_input_covariance
                                         + mapping @ self._input_covariance)
        omega = gyro - self.state.gyro_bias
        rotation_half = rotation_matrix(quaternion_multiply(
            self.state.quaternion, exp_quaternion(omega * dt / 2)))
        acceleration = rotation_half @ (accel - self.state.acc_bias) + self.gravity
        self.state.position += self.state.velocity * dt + acceleration * (dt ** 2 / 2)
        self.state.velocity += acceleration * dt
        self.state.quaternion = normalize_quaternion(quaternion_multiply(
            self.state.quaternion, exp_quaternion(omega * dt)))
        self.time = time

    def update_uwb(self, ranges, anchors, lever_arm, std):
        """联合固定方差测距更新、Joseph 协方差更新和右局部误差注入重置。"""
        prediction, jacobian = measurement_model(self.state, anchors, lever_arm)
        ranges = _vector(ranges, len(prediction), '原始距离')
        measurement_covariance = np.diag(_std(std, len(ranges), '测距标准差', positive=True) ** 2)
        innovation = ranges - prediction
        innovation_covariance = _symmetric(jacobian @ self.P @ jacobian.T + measurement_covariance)
        gain = np.linalg.solve(innovation_covariance, jacobian @ self.P).T
        input_gain = np.linalg.solve(
            innovation_covariance, jacobian @ self._state_input_covariance).T
        delta = gain @ innovation
        remainder = np.eye(15) - gain @ jacobian
        posterior = (remainder @ self.P @ remainder.T
                     + gain @ measurement_covariance @ gain.T)
        self._input_mean += input_gain @ innovation
        self._input_covariance = _symmetric(
            self._input_covariance - input_gain @ innovation_covariance @ input_gain.T)
        self._state_input_covariance = remainder @ self._state_input_covariance
        self.state.position += delta[:3]
        self.state.velocity += delta[3:6]
        self.state.quaternion = normalize_quaternion(quaternion_multiply(
            self.state.quaternion, exp_quaternion(delta[6:9])))
        self.state.acc_bias += delta[9:12]
        self.state.gyro_bias += delta[12:15]
        reset = np.eye(15)
        reset[6:9, 6:9] = right_jacobian(delta[6:9])
        self.P = _symmetric(reset @ posterior @ reset.T)
        self._state_input_covariance = reset @ self._state_input_covariance
        return dict(innovation=innovation, innovation_covariance=innovation_covariance,
                    nis=float(innovation @ np.linalg.solve(innovation_covariance, innovation)))
