"""验证动力学联合误差滤波的物理预测、共享噪声及异步输入语义。"""

from importlib import import_module

import numpy as np
import pytest

from estimator.eskf import NominalState
from estimator.rotation import exp_quaternion, log_quaternion
from simulator.math3d import quaternion_multiply, rotation_matrix, skew


ZERO_NOISE = dict(acc_std=0.0, gyro_std=0.0, acc_bias_rw_std=0.0,
                  gyro_bias_rw_std=0.0, rpm_std=0.0, model_force_std=0.0)
ANCHORS = np.array([[-4, -3, 0], [5, -2, 1], [3, 6, 0], [-3, 4, 5]])


def api(name, module='physics_ekf'):
    """缺少实现时产生明确的功能断言失败。"""
    try:
        result = getattr(import_module('estimator.' + module), name, None)
    except ModuleNotFoundError:
        result = None
    assert result is not None, f'尚未实现 estimator.{module}.{name}'
    return result


class ForceModel:
    """独立解析测试模型，含各向异性阻力，隔离尚在并行开发的模型模块。"""

    def __init__(self, force=(0, 0, 0), mass=2.0, gravity=9.81, drag=None):
        self.force = np.asarray(force, dtype=float)
        self.mass, self.gravity = mass, gravity
        self.drag = np.diag([0.4, 0.2, 0.3] if drag is None else drag)
        self.rpm_mapping = np.array([[0.02, 0, 0, 0], [0, 0.03, 0, 0],
                                     [0, 0, 0.04, 0.05]])

    def force_body(self, quaternion, velocity, rpm):
        return self.force - self.drag @ rotation_matrix(quaternion).T @ velocity + self.rpm_mapping @ rpm

    def force_jacobians(self, quaternion, velocity, rpm):
        rotation = rotation_matrix(quaternion)
        return -self.drag @ rotation.T, -self.drag @ skew(rotation.T @ velocity), self.rpm_mapping


def state(position=(0, 0, 2), velocity=(0, 0, 0), quaternion=(1, 0, 0, 0),
          acc_bias=(0, 0, 0), gyro_bias=(0, 0, 0)):
    return NominalState(position, velocity, quaternion, acc_bias, gyro_bias)


def estimator(initial=None, covariance=None, noise=None, model=None, initial_time=0):
    return api('PhysicsEKF')(
        state() if initial is None else initial,
        np.zeros((15, 15)) if covariance is None else covariance,
        ForceModel(drag=[0, 0, 0]) if model is None else model,
        ZERO_NOISE if noise is None else noise, initial_time=initial_time)


def prepare(filter_, gyro=(0, 0, 0), rpm=(0, 0, 0, 0)):
    filter_.set_gyro(filter_.time, gyro)
    filter_.set_rpm(filter_.time, rpm)
    return filter_


def perturb(nominal, delta):
    return state(nominal.position + delta[:3], nominal.velocity + delta[3:6],
                 quaternion_multiply(nominal.quaternion, exp_quaternion(delta[6:9])),
                 nominal.acc_bias + delta[9:12], nominal.gyro_bias + delta[12:15])


def difference(reference, target):
    inverse = reference.quaternion * np.array([1, -1, -1, -1])
    return np.r_[target.position - reference.position, target.velocity - reference.velocity,
                 log_quaternion(quaternion_multiply(inverse, target.quaternion)),
                 target.acc_bias - reference.acc_bias, target.gyro_bias - reference.gyro_bias]


def joint_covariance(filter_):
    return np.block([[filter_.P, filter_._state_input_covariance],
                     [filter_._state_input_covariance.T, filter_._input_covariance]])


@pytest.mark.parametrize('force,gyro,position,velocity,quaternion', [
    ([0, 0, 19.62], [0, 0, 0], [0, 0, 2], [0, 0, 0], [1, 0, 0, 0]),
    ([0, 0, 0], [0, 0, 0], [0, 0, -2.905], [0, 0, -9.81], [1, 0, 0, 0]),
    ([0, 0, 19.62], [0, 0, 2], [0, 0, 2], [0, 0, 0],
     [np.cos(1), 0, 0, np.sin(1)]),
])
def test_rpm_dynamics_hover_freefall_and_rotation(force, gyro, position, velocity, quaternion):
    filter_ = prepare(estimator(model=ForceModel(force, drag=[0, 0, 0])), gyro)
    for time in np.linspace(0.01, 1, 100):
        filter_.propagate_to(time)
    np.testing.assert_allclose(filter_.state.position, position, atol=2e-12)
    np.testing.assert_allclose(filter_.state.velocity, velocity, atol=2e-12)
    np.testing.assert_allclose(filter_.state.quaternion, quaternion, atol=2e-12)


def test_prediction_uses_midpoint_rotation_and_ignores_acc_bias():
    initial = state(acc_bias=[12, -9, 4], gyro_bias=[0, 0, 0.2])
    filter_ = prepare(estimator(initial, model=ForceModel([2, 0, 0], gravity=0, drag=[0, 0, 0])), [0, 0, 2.2])
    filter_.propagate_to(0.5)
    np.testing.assert_allclose(filter_.state.velocity, [np.cos(0.5) * 0.5, np.sin(0.5) * 0.5, 0])
    np.testing.assert_allclose(filter_.state.position, [np.cos(0.5) * 0.125, np.sin(0.5) * 0.125, 2])


def test_discrete_state_and_auxiliary_jacobians_match_finite_differences():
    initial = state([0.2, -0.3, 2.1], [1.2, -0.4, 0.1], exp_quaternion([0.2, -0.3, 0.4]),
                    [0.2, -0.1, 0.3], [0.03, -0.02, 0.01])
    model = ForceModel([0.5, -0.3, 18])
    gyro, rpm, dt, epsilon = np.array([0.6, -0.4, 0.7]), np.array([1, 2, 3, 4]), 0.13, 1e-6
    mean = np.array([0.02, -0.03, 0.01, 0.3, -0.1, 0.2])

    def predict(nominal, auxiliary):
        filter_ = prepare(estimator(nominal, model=model), gyro, rpm)
        filter_._input_mean = auxiliary.copy()
        filter_.propagate_to(dt)
        return filter_.state

    reference = predict(initial, mean)
    numerical_state = np.column_stack([
        (difference(reference, predict(perturb(initial, epsilon * axis), mean))
         - difference(reference, predict(perturb(initial, -epsilon * axis), mean))) / (2 * epsilon)
        for axis in np.eye(15)])
    numerical_aux = np.column_stack([
        (difference(reference, predict(initial, mean + epsilon * axis))
         - difference(reference, predict(initial, mean - epsilon * axis))) / (2 * epsilon)
        for axis in np.eye(6)])
    phi, mapping = api('dynamics_propagation_matrices')(initial, gyro, rpm, model, dt, mean)
    np.testing.assert_allclose(phi, numerical_state, atol=3e-9)
    np.testing.assert_allclose(mapping, numerical_aux, atol=3e-9)
    np.testing.assert_array_equal(phi[:9, 9:12], np.zeros((9, 3)))


def test_specific_force_state_and_auxiliary_jacobians_match_finite_differences():
    initial = state(velocity=[1.2, -0.4, 0.1], quaternion=exp_quaternion([0.2, -0.3, 0.4]),
                    acc_bias=[0.2, -0.1, 0.3])
    model, rpm = ForceModel([0.5, -0.3, 18]), np.array([1, 2, 3, 4])
    mean, epsilon = np.array([0.1, 0.2, -0.1, 0.4, -0.2, 0.1]), 1e-6
    observation = api('accel_measurement_model')
    prediction, jacobian, auxiliary = observation(initial, rpm, model, mean)
    numerical = np.column_stack([
        (observation(perturb(initial, epsilon * axis), rpm, model, mean)[0]
         - observation(perturb(initial, -epsilon * axis), rpm, model, mean)[0]) / (2 * epsilon)
        for axis in np.eye(15)])
    numerical_aux = np.column_stack([
        (observation(initial, rpm, model, mean + epsilon * axis)[0]
         - observation(initial, rpm, model, mean - epsilon * axis)[0]) / (2 * epsilon)
        for axis in np.eye(6)])
    np.testing.assert_allclose(jacobian, numerical, atol=2e-9)
    np.testing.assert_allclose(auxiliary, numerical_aux, atol=2e-9)
    np.testing.assert_allclose(prediction, model.force_body(initial.quaternion, initial.velocity, rpm) / 2 + mean[3:] / 2 + initial.acc_bias)


def test_shared_force_noise_update_then_prediction_matches_explicit_joint_kalman():
    noise = dict(ZERO_NOISE, model_force_std=2, acc_std=0.5)
    filter_ = prepare(estimator(noise=noise))
    diagnostic = filter_.update_accel([1, -2, 0.5])
    expected_force_mean = np.array([1.6, -3.2, 0.8])
    np.testing.assert_allclose(filter_._input_mean[3:], expected_force_mean)
    np.testing.assert_allclose(diagnostic['innovation_covariance'], np.eye(3) * 1.25)
    assert diagnostic['nis'] == pytest.approx(5.25 / 1.25)
    filter_.propagate_to(0.2)
    mapping = np.zeros((15, 3))
    mapping[:3] = np.eye(3) * 0.01
    mapping[3:6] = np.eye(3) * 0.1
    expected = mapping @ (np.eye(3) * 0.8) @ mapping.T
    np.testing.assert_allclose(filter_.P, expected, atol=1e-15)
    np.testing.assert_allclose(filter_.state.velocity, [0.16, -0.32, -1.882])
    np.testing.assert_allclose(filter_._state_input_covariance[:, 3:], mapping * 0.8)


def test_force_sample_is_shared_across_split_predictions_and_measurement():
    noise = dict(ZERO_NOISE, model_force_std=[1, 2, 3], gyro_std=0.2, acc_std=0.4)
    complete = prepare(estimator(noise=noise))
    split = prepare(estimator(noise=noise))
    complete.propagate_to(0.2)
    split.propagate_to(0.08)
    split.propagate_to(0.2)
    np.testing.assert_allclose(joint_covariance(split), joint_covariance(complete), atol=1e-14)
    complete.update_accel([0.2, -0.1, 0.3])
    split.update_accel([0.2, -0.1, 0.3])
    np.testing.assert_allclose(joint_covariance(split), joint_covariance(complete), atol=1e-14)


@pytest.mark.parametrize('kind', ['gyro', 'rpm'])
def test_asynchronous_sample_replacement_preserves_other_conditional_marginal(kind):
    noise = dict(ZERO_NOISE, model_force_std=2, gyro_std=0.2, rpm_std=0.3, acc_std=0.4)
    filter_ = prepare(estimator(covariance=np.eye(15) * 0.1, noise=noise,
                                model=ForceModel([1, 0.3, 18], drag=[0, 0, 0])), [0.2, -0.1, 0.3])
    filter_.propagate_to(0.1)
    filter_.update_accel([0.8, -0.1, 9.3])
    filter_.update_uwb([5.4, 5.7, 7.2, 6.1], ANCHORS, [0.2, 0.1, 0.05], 0.1)
    offset = 0 if kind == 'gyro' else 3
    replaced, retained = slice(offset, offset + 3), slice(3 - offset, 6 - offset)
    mean, sample, cross, public = filter_._input_mean.copy(), filter_._input_covariance.copy(), filter_._state_input_covariance.copy(), filter_.P.copy()
    if kind == 'gyro':
        filter_.set_gyro(0.1, [0.4, -0.1, 0.2])
        variance = np.eye(3) * 0.04
    else:
        filter_.set_rpm(0.1, [2, 3, 4, 5])
        matrix = filter_.model.force_jacobians(filter_.state.quaternion, filter_.state.velocity, [2, 3, 4, 5])[2]
        variance = np.eye(3) * 4 + matrix @ (np.eye(4) * 0.09) @ matrix.T
    np.testing.assert_array_equal(filter_.P, public)
    np.testing.assert_array_equal(filter_._input_mean[retained], mean[retained])
    np.testing.assert_array_equal(filter_._input_covariance[retained, retained], sample[retained, retained])
    np.testing.assert_array_equal(filter_._state_input_covariance[:, retained], cross[:, retained])
    np.testing.assert_array_equal(filter_._input_mean[replaced], np.zeros(3))
    np.testing.assert_allclose(filter_._input_covariance[replaced, replaced], variance)
    np.testing.assert_array_equal(filter_._input_covariance[replaced, retained], np.zeros((3, 3)))
    np.testing.assert_array_equal(filter_._state_input_covariance[:, replaced], np.zeros((15, 3)))


def test_joint_joseph_updates_and_right_reset_preserve_psd():
    noise = dict(ZERO_NOISE, model_force_std=1.2, gyro_std=0.03, acc_std=0.15,
                 acc_bias_rw_std=0.01, gyro_bias_rw_std=0.002, rpm_std=0.8)
    filter_ = prepare(estimator(covariance=np.eye(15) * 0.3, noise=noise,
                                model=ForceModel([0.5, -0.4, 19.62])), [0.3, -0.2, 0.1])
    for index in range(1, 15):
        time = index * 0.01
        filter_.propagate_to(time)
        filter_.update_accel([0.25, -0.1, 9.7])
        diagnostic = filter_.update_uwb([5.4, 5.7, 7.2, 6.1], ANCHORS, [0.2, 0.1, 0.05], 0.1)
        joint = joint_covariance(filter_)
        assert filter_.P.shape == (15, 15)
        np.testing.assert_allclose(joint, joint.T, atol=1e-14)
        assert np.linalg.eigvalsh(joint).min() > -1e-12
        assert np.isfinite(diagnostic['nis'])
        filter_.set_gyro(time, [0.3, -0.2, 0.1])
        if index % 2 == 0:
            filter_.set_rpm(time, [1, 2, 3, 4])


@pytest.mark.parametrize('kind', ['accel', 'uwb'])
def test_complete_joint_update_matches_gaussian_conditioning_and_numerical_reset(kind):
    initial = state([0.4, -0.3, 2.2], [1.1, -0.2, 0.3], exp_quaternion([0.3, -0.2, 0.4]),
                    [0.1, -0.05, 0.2], [0.03, -0.02, 0.01])
    model = ForceModel([0.5, -0.3, 18])
    filter_ = prepare(estimator(initial, model=model, noise=dict(ZERO_NOISE, acc_std=0.3)))
    rng = np.random.default_rng(719)
    factor = rng.normal(size=(21, 21)) * 0.04
    prior = factor @ factor.T + np.eye(21) * 0.02
    filter_.P = prior[:15, :15].copy()
    filter_._state_input_covariance = prior[:15, 15:].copy()
    filter_._input_covariance = prior[15:, 15:].copy()
    mean = np.array([0.02, -0.01, 0.03, 0.2, -0.1, 0.3])
    filter_._input_mean = mean.copy()
    epsilon = 1e-6

    def prediction(delta):
        nominal = perturb(initial, delta[:15])
        auxiliary = mean + delta[15:]
        if kind == 'accel':
            return model.force_body(nominal.quaternion, nominal.velocity, np.zeros(4)) / 2 + auxiliary[3:] / 2 + nominal.acc_bias
        antenna = nominal.position + rotation_matrix(nominal.quaternion) @ [0.2, 0.1, 0.05]
        return np.linalg.norm(antenna - ANCHORS, axis=1)

    observation = np.column_stack([
        (prediction(epsilon * axis) - prediction(-epsilon * axis)) / (2 * epsilon)
        for axis in np.eye(21)])
    residual = np.array([0.6, -0.4, 0.3] if kind == 'accel' else [0.5, -0.2, 0.3, -0.4])
    measurement_variance = 0.09 if kind == 'accel' else 0.04
    predicted = prediction(np.zeros(21))
    cross = prior @ observation.T
    innovation_covariance = observation @ cross + np.eye(len(residual)) * measurement_variance
    delta = cross @ np.linalg.solve(innovation_covariance, residual)
    conditional = prior - cross @ np.linalg.solve(innovation_covariance, cross.T)
    inverse = exp_quaternion(-delta[6:9])
    reset_theta = np.column_stack([
        (log_quaternion(quaternion_multiply(inverse, exp_quaternion(delta[6:9] + epsilon * axis)))
         - log_quaternion(quaternion_multiply(inverse, exp_quaternion(delta[6:9] - epsilon * axis)))) / (2 * epsilon)
        for axis in np.eye(3)])
    reset = np.eye(21)
    reset[6:9, 6:9] = reset_theta
    if kind == 'accel':
        diagnostic = filter_.update_accel(predicted + residual)
    else:
        diagnostic = filter_.update_uwb(predicted + residual, ANCHORS, [0.2, 0.1, 0.05], 0.2)
    np.testing.assert_allclose(joint_covariance(filter_), reset @ conditional @ reset.T, atol=3e-11)
    np.testing.assert_allclose(difference(initial, filter_.state), delta[:15], atol=3e-10)
    np.testing.assert_allclose(filter_._input_mean, mean + delta[15:], atol=3e-10)
    np.testing.assert_allclose(diagnostic['innovation_covariance'], innovation_covariance, atol=2e-10)


def test_bias_random_walk_has_no_accelerometer_driven_translation_in_freefall():
    noise = dict(ZERO_NOISE, acc_bias_rw_std=[0.1, 0.2, 0.3], gyro_bias_rw_std=[0.02, 0.03, 0.04])
    filter_ = prepare(estimator(noise=noise))
    filter_.propagate_to(0.7)
    expected = np.zeros((15, 15))
    expected[9:12, 9:12] = np.diag(np.array([0.1, 0.2, 0.3]) ** 2 * 0.7)
    variance = np.array([0.02, 0.03, 0.04]) ** 2
    expected[6:9, 6:9] = np.diag(variance * 0.7 ** 3 / 3)
    expected[6:9, 12:15] = expected[12:15, 6:9] = np.diag(-variance * 0.7 ** 2 / 2)
    expected[12:15, 12:15] = np.diag(variance * 0.7)
    np.testing.assert_allclose(filter_.P, expected, atol=1e-15)


@pytest.mark.parametrize('received', ['none', 'gyro', 'rpm'])
def test_propagation_and_accel_update_require_both_arrived_inputs(received):
    filter_ = estimator(noise=dict(ZERO_NOISE, acc_std=0.2))
    if received == 'gyro':
        filter_.set_gyro(0, [0, 0, 0])
    if received == 'rpm':
        filter_.set_rpm(0, [0, 0, 0, 0])
    with pytest.raises(ValueError):
        filter_.propagate_to(0)
    with pytest.raises(ValueError):
        filter_.propagate_to(0.1)
    with pytest.raises(ValueError):
        filter_.update_accel([0, 0, 0])


def test_no_future_input_backward_propagation_or_repeated_input_time():
    filter_ = estimator()
    with pytest.raises(ValueError):
        filter_.set_gyro(0.1, [0, 0, 0])
    with pytest.raises(ValueError):
        filter_.set_rpm(0.1, [0, 0, 0, 0])
    prepare(filter_)
    with pytest.raises(ValueError):
        filter_.set_gyro(0, [0, 0, 0])
    with pytest.raises(ValueError):
        filter_.set_rpm(0, [0, 0, 0, 0])
    filter_.propagate_to(0.2)
    with pytest.raises(ValueError):
        filter_.propagate_to(0.1)
    with pytest.raises(ValueError):
        filter_.set_gyro(0.1, [0, 0, 0])


def test_negative_specific_force_and_range_are_not_clipped():
    filter_ = prepare(estimator(covariance=np.eye(15) * 0.1,
                                noise=dict(ZERO_NOISE, acc_std=0.5)))
    acceleration = filter_.update_accel([-1, -2, -3])
    np.testing.assert_allclose(acceleration['innovation'], [-1, -2, -3])
    diagnostic = filter_.update_uwb([-1], [[0, 0, 0]], [0, 0, 0], 0.1)
    np.testing.assert_allclose(diagnostic['innovation'], [-3])
    with pytest.raises(ValueError):
        filter_.update_uwb([0], [filter_.state.position.copy()], [0, 0, 0], 0.1)


@pytest.mark.parametrize('noise', [dict(ZERO_NOISE, model_force_std=-1),
                                  dict(ZERO_NOISE, rpm_std=[1, 2, 3]),
                                  dict(ZERO_NOISE, gyro_std=np.nan), {}])
def test_invalid_noise_configuration_is_rejected(noise):
    with pytest.raises(ValueError):
        estimator(noise=noise)


def test_invalid_model_interface_and_covariance_are_rejected():
    with pytest.raises(ValueError):
        estimator(model=object())
    with pytest.raises(ValueError):
        estimator(covariance=-np.eye(15))
    with pytest.raises(ValueError):
        estimator(covariance=np.ones((14, 14)))


def test_real_physics_and_hybrid_models_use_the_same_filter_algorithm():
    physics_class = getattr(import_module('models.physics'), 'PhysicsModel', None)
    hybrid_class = getattr(import_module('models.hybrid'), 'HybridModel', None)
    assert physics_class is not None and hybrid_class is not None, '等待真实物理与混合模型接口'
    physics = physics_class(dict(mass=1.5, gravity=9.81, kf=1e-5,
                                 linear_drag=[0, 0, 0], quadratic_drag=[0, 0, 0]))
    hybrid = hybrid_class(physics, lambda window: np.zeros(3))
    noise = dict(ZERO_NOISE, acc_std=0.1, model_force_std=0.4)
    physical_filter = prepare(estimator(model=physics, noise=noise), rpm=[580, 580, 580, 580])
    hybrid_filter = api('HybridEKF', 'hybrid_ekf')(state(), np.zeros((15, 15)), hybrid, noise)
    prepare(hybrid_filter, rpm=[580, 580, 580, 580])
    physical_filter.propagate_to(0.1)
    hybrid_filter.propagate_to(0.1)
    physical_filter.update_accel([0.1, -0.1, 9.7])
    hybrid_filter.update_accel([0.1, -0.1, 9.7])
    np.testing.assert_allclose(difference(physical_filter.state, hybrid_filter.state), np.zeros(15))
    np.testing.assert_allclose(joint_covariance(physical_filter), joint_covariance(hybrid_filter))
    with pytest.raises(ValueError):
        api('HybridEKF', 'hybrid_ekf')(state(), np.eye(15), physics, noise)
