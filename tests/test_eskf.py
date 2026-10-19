"""验证右乘局部误差 ESKF 的离散积分、雅可比与样本噪声相关性。"""

from importlib import import_module

import numpy as np
import pytest

from simulator.math3d import quaternion_multiply, rotation_matrix


ZERO_NOISE = dict(acc_std=0.0, gyro_std=0.0,
                  acc_bias_rw_std=0.0, gyro_bias_rw_std=0.0)
ANCHORS = np.array([[-4, -3, 0], [5, -2, 1], [3, 6, 0], [-3, 4, 5]])


def api(module, name):
    """把缺失实现报告为断言失败，便于先观察红灯。"""
    try:
        result = getattr(import_module('estimator.' + module), name, None)
    except ModuleNotFoundError:
        result = None
    assert result is not None, f'尚未实现 estimator.{module}.{name}'
    return result


def state(position=None, velocity=None, quaternion=None, acc_bias=None, gyro_bias=None):
    return api('eskf', 'NominalState')(
        np.array([0, 0, 2] if position is None else position, dtype=float),
        np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float),
        np.array([1, 0, 0, 0] if quaternion is None else quaternion, dtype=float),
        np.zeros(3) if acc_bias is None else np.asarray(acc_bias, dtype=float),
        np.zeros(3) if gyro_bias is None else np.asarray(gyro_bias, dtype=float))


def filter_(initial=None, covariance=None, gravity=9.81, noise=None, initial_time=0.0):
    return api('eskf', 'UWBIMUESKF')(
        state() if initial is None else initial,
        np.zeros((15, 15)) if covariance is None else covariance,
        gravity, ZERO_NOISE if noise is None else noise, initial_time=initial_time)


def perturb(nominal, delta):
    return state(nominal.position + delta[:3], nominal.velocity + delta[3:6],
                 quaternion_multiply(nominal.quaternion,
                                     api('rotation', 'exp_quaternion')(delta[6:9])),
                 nominal.acc_bias + delta[9:12], nominal.gyro_bias + delta[12:15])


def difference(reference, target):
    inverse = reference.quaternion * np.array([1, -1, -1, -1])
    angle = api('rotation', 'log_quaternion')(
        quaternion_multiply(inverse, target.quaternion))
    return np.r_[target.position - reference.position,
                 target.velocity - reference.velocity, angle,
                 target.acc_bias - reference.acc_bias,
                 target.gyro_bias - reference.gyro_bias]


@pytest.mark.parametrize('vector', [[0, 0, 0], [1e-10, -2e-10, 3e-10],
                                    [0.3, -0.6, 0.7], [np.pi - 1e-8, 0, 0]])
def test_rotation_exponential_logarithm_round_trip(vector):
    quaternion = api('rotation', 'exp_quaternion')(vector)
    assert np.linalg.norm(quaternion) == pytest.approx(1.0, abs=1e-14)
    np.testing.assert_allclose(api('rotation', 'log_quaternion')(quaternion), vector,
                               atol=1e-12)
    np.testing.assert_allclose(api('rotation', 'log_quaternion')(-quaternion), vector,
                               atol=1e-12)


def test_right_jacobian_matches_local_rotation_finite_difference():
    vector, epsilon = np.array([0.4, -0.2, 0.6]), 1e-7
    exp, log = api('rotation', 'exp_quaternion'), api('rotation', 'log_quaternion')
    inverse = exp(vector) * np.array([1, -1, -1, -1])
    numerical = np.column_stack([
        (log(quaternion_multiply(inverse, exp(vector + epsilon * axis)))
         - log(quaternion_multiply(inverse, exp(vector - epsilon * axis)))) / (2 * epsilon)
        for axis in np.eye(3)])
    np.testing.assert_allclose(api('rotation', 'right_jacobian')(vector), numerical,
                               atol=2e-9)


def test_logarithm_selects_identical_half_turn_for_quaternion_double_cover():
    logarithm = api('rotation', 'log_quaternion')
    np.testing.assert_allclose(logarithm([0, 0, 0, 1]), [0, 0, np.pi])
    np.testing.assert_allclose(logarithm([0, 0, 0, -1]), [0, 0, np.pi])


@pytest.mark.parametrize('accel,gyro,position,velocity,quaternion', [
    ([0, 0, 9.81], [0, 0, 0], [0, 0, 2], [0, 0, 0], [1, 0, 0, 0]),
    ([0, 0, 0], [0, 0, 0], [0, 0, -2.905], [0, 0, -9.81], [1, 0, 0, 0]),
    ([0, 0, 9.81], [0, 0, 2], [0, 0, 2], [0, 0, 0],
     [np.cos(1), 0, 0, np.sin(1)]),
])
def test_noise_free_hover_freefall_and_rotation(accel, gyro, position, velocity, quaternion):
    estimator = filter_()
    estimator.set_imu(0.0, accel, gyro)
    for time in np.linspace(0.01, 1.0, 100):
        estimator.propagate_to(time)
    np.testing.assert_allclose(estimator.state.position, position, atol=1e-12)
    np.testing.assert_allclose(estimator.state.velocity, velocity, atol=1e-12)
    np.testing.assert_allclose(estimator.state.quaternion, quaternion, atol=1e-12)
    assert estimator.time == 1.0


def test_propagation_uses_midpoint_attitude_and_bias_compensation():
    estimator = filter_(state(acc_bias=[0.2, -0.1, 0.3], gyro_bias=[0, 0, 0.2]), gravity=0)
    estimator.set_imu(0, [1.2, -0.1, 0.3], [0, 0, 2.2])
    estimator.propagate_to(0.5)
    expected_acceleration = np.array([np.cos(0.5), np.sin(0.5), 0])
    np.testing.assert_allclose(estimator.state.velocity, expected_acceleration * 0.5,
                               atol=1e-14)
    np.testing.assert_allclose(estimator.state.position,
                               [0, 0, 2] + expected_acceleration * 0.125, atol=1e-14)


def test_discrete_propagation_jacobians_match_full_state_finite_difference():
    initial = state([0.4, -0.7, 2.2], [1, -0.3, 0.2],
                    api('rotation', 'exp_quaternion')([0.4, 0.2, -0.3]),
                    [0.1, -0.2, 0.05], [0.03, -0.04, 0.01])
    accel, gyro, dt, epsilon = np.array([0.4, -0.8, 9.5]), np.array([0.7, -0.3, 0.5]), 0.17, 1e-6

    def propagate(nominal, acceleration=accel, angular_rate=gyro):
        estimator = filter_(nominal)
        estimator.set_imu(0, acceleration, angular_rate)
        estimator.propagate_to(dt)
        return estimator.state

    reference = propagate(initial)
    numerical = np.column_stack([
        (difference(reference, propagate(perturb(initial, epsilon * axis)))
         - difference(reference, propagate(perturb(initial, -epsilon * axis)))) / (2 * epsilon)
        for axis in np.eye(15)])
    phi, noise_mapping = api('eskf', 'propagation_matrices')(initial, accel, gyro, dt)
    np.testing.assert_allclose(phi, numerical, atol=8e-10)
    numerical_noise = np.column_stack([
        (difference(reference, propagate(initial, accel - epsilon * axis[:3], gyro - epsilon * axis[3:]))
         - difference(reference, propagate(initial, accel + epsilon * axis[:3], gyro + epsilon * axis[3:]))) / (2 * epsilon)
        for axis in np.eye(6)])
    np.testing.assert_allclose(noise_mapping, numerical_noise, atol=8e-10)


def test_lever_arm_range_jacobian_matches_finite_difference():
    nominal = state([0.4, -0.2, 2.1], quaternion=
                    api('rotation', 'exp_quaternion')([0.2, -0.4, 0.8]))
    lever, epsilon = np.array([0.3, -0.2, 0.1]), 1e-6
    prediction, jacobian = api('eskf', 'measurement_model')(nominal, ANCHORS, lever)
    tag_position = nominal.position + rotation_matrix(nominal.quaternion) @ lever
    np.testing.assert_allclose(prediction, np.linalg.norm(tag_position - ANCHORS, axis=1))
    numerical = np.column_stack([
        (api('eskf', 'measurement_model')(perturb(nominal, epsilon * axis), ANCHORS, lever)[0]
         - api('eskf', 'measurement_model')(perturb(nominal, -epsilon * axis), ANCHORS, lever)[0]) / (2 * epsilon)
        for axis in np.eye(15)])
    np.testing.assert_allclose(jacobian, numerical, atol=8e-10)


def test_update_injects_all_state_blocks_and_resets_rotation_covariance():
    rng = np.random.default_rng(14)
    factor = rng.normal(size=(15, 15))
    covariance = factor @ factor.T * 0.01
    nominal, lever = state(), np.array([0.3, -0.2, 0.1])
    prediction, jacobian = api('eskf', 'measurement_model')(nominal, ANCHORS, lever)
    innovation, measurement_covariance = np.array([0.2, -0.1, 0.3, -0.4]), np.eye(4) * 0.04
    innovation_covariance = jacobian @ covariance @ jacobian.T + measurement_covariance
    gain = np.linalg.solve(innovation_covariance, jacobian @ covariance).T
    delta = gain @ innovation
    remainder = np.eye(15) - gain @ jacobian
    joseph = remainder @ covariance @ remainder.T + gain @ measurement_covariance @ gain.T
    reset = np.eye(15)
    reset[6:9, 6:9] = api('rotation', 'right_jacobian')(delta[6:9])
    estimator = filter_(nominal, covariance)
    diagnostics = estimator.update_uwb(prediction + innovation, ANCHORS, lever, 0.2)
    np.testing.assert_allclose(difference(nominal, estimator.state), delta, atol=1e-14)
    np.testing.assert_allclose(estimator.P, reset @ joseph @ reset.T, atol=1e-14)
    assert set(diagnostics) == {'innovation', 'innovation_covariance', 'nis'}
    np.testing.assert_allclose(diagnostics['innovation'], innovation, atol=1e-14)
    np.testing.assert_allclose(diagnostics['innovation_covariance'], innovation_covariance)
    assert diagnostics['nis'] == pytest.approx(innovation @ np.linalg.solve(innovation_covariance, innovation))


def test_sample_white_noise_covariance_is_unchanged_when_interval_is_split():
    noise = {**ZERO_NOISE, 'acc_std': 0.6}
    whole, split = filter_(noise=noise), filter_(noise=noise)
    for estimator in (whole, split):
        estimator.set_imu(0, [0, 0, 9.81], [0, 0, 0])
    whole.propagate_to(1)
    for time in (0.3, 0.7, 1):
        split.propagate_to(time)
    np.testing.assert_allclose(whole.P, split.P, atol=1e-15)
    np.testing.assert_allclose(whole.P[0:6:3, 0:6:3], [[0.09, 0.18], [0.18, 0.36]], atol=1e-15)


def test_uwb_conditions_held_sample_mean_and_covariance_for_future_segments():
    noise = {**ZERO_NOISE, 'acc_std': 0.6}
    whole, split = filter_(noise=noise), filter_(noise=noise)
    for estimator in (whole, split):
        estimator.set_imu(0, [0, 0, 9.81], [0, 0, 0])
        estimator.propagate_to(0.4)
        estimator.update_uwb([10.4], [[-10, 0, 2]], [0, 0, 0], 0.1)
    whole.propagate_to(1)
    for time in (0.6, 0.8, 1):
        split.propagate_to(time)
    variance = 0.36
    observation_scale = 0.5 * 0.4 ** 2
    innovation_variance = observation_scale ** 2 * variance + 0.1 ** 2
    final_mapping = np.array([0.5, 1.0])
    cross = final_mapping * observation_scale * variance
    expected_mean = cross * 0.4 / innovation_variance
    expected_covariance = np.outer(final_mapping, final_mapping) * variance - np.outer(cross, cross) / innovation_variance
    for estimator in (whole, split):
        np.testing.assert_allclose([estimator.state.position[0], estimator.state.velocity[0]], expected_mean, atol=1e-13)
        np.testing.assert_allclose(estimator.P[0:6:3, 0:6:3], expected_covariance, atol=1e-14)
    np.testing.assert_allclose(whole.P, split.P, atol=1e-14)


def test_new_sample_discards_old_input_noise_conditioning():
    estimator = filter_(noise={**ZERO_NOISE, 'acc_std': 0.6})
    estimator.set_imu(0, [0, 0, 9.81], [0, 0, 0])
    estimator.propagate_to(0.4)
    estimator.update_uwb([10.4], [[-10, 0, 2]], [0, 0, 0], 0.1)
    position, velocity = estimator.state.position.copy(), estimator.state.velocity.copy()
    estimator.set_imu(0.4, [0, 0, 9.81], [0, 0, 0])
    estimator.propagate_to(1)
    np.testing.assert_allclose(estimator.state.position, position + 0.6 * velocity, atol=1e-14)
    np.testing.assert_allclose(estimator.state.velocity, velocity, atol=1e-14)


def test_repeated_joseph_updates_and_bias_random_walk_remain_symmetric_psd():
    noise = dict(acc_std=0.08, gyro_std=0.01, acc_bias_rw_std=0.02, gyro_bias_rw_std=0.002)
    estimator = filter_(covariance=np.eye(15) * 0.1, noise=noise)
    for index in range(30):
        time = index * 0.03
        estimator.set_imu(time, [0.1, -0.2, 9.81], [0.05, -0.03, 0.1])
        estimator.propagate_to(time + 0.015)
        prediction, _ = api('eskf', 'measurement_model')(estimator.state, ANCHORS, [0.2, 0, 0])
        estimator.update_uwb(prediction + [0.01, -0.02, 0.005, 0.015], ANCHORS, [0.2, 0, 0], [0.1] * 4)
        estimator.propagate_to((index + 1) * 0.03)
        np.testing.assert_allclose(estimator.P, estimator.P.T, atol=1e-14)
        assert np.linalg.eigvalsh(estimator.P).min() >= -1e-12
    assert np.linalg.norm(estimator.state.quaternion) == pytest.approx(1, abs=1e-14)


def test_bias_random_walk_integrates_time_and_translation_coupling():
    estimator = filter_(gravity=0, noise={**ZERO_NOISE, 'acc_bias_rw_std': 0.2})
    estimator.set_imu(0, [0, 0, 0], [0, 0, 0])
    estimator.propagate_to(0.5)
    # 对标量加速度偏置布朗运动，解析积分为 q*[T^5/20,T^4/8,T^3/6;...]。
    expected = 0.04 * np.array([[0.5 ** 5 / 20, 0.5 ** 4 / 8, -0.5 ** 3 / 6],
                                [0.5 ** 4 / 8, 0.5 ** 3 / 3, -0.5 ** 2 / 2],
                                [-0.5 ** 3 / 6, -0.5 ** 2 / 2, 0.5]])
    np.testing.assert_allclose(estimator.P[np.ix_([0, 3, 9], [0, 3, 9])], expected, atol=1e-15)


def test_time_contract_rejects_future_sample_and_reverse_propagation():
    estimator = filter_(initial_time=2)
    estimator.propagate_to(2)
    with pytest.raises(ValueError):
        estimator.propagate_to(2.1)
    with pytest.raises(ValueError):
        estimator.set_imu(2.1, [0, 0, 0], [0, 0, 0])
    estimator.set_imu(2, [0, 0, 9.81], [0, 0, 0])
    with pytest.raises(ValueError):
        estimator.set_imu(2, [0, 0, 9.81], [0, 0, 0])
    estimator.propagate_to(2.1)
    with pytest.raises(ValueError):
        estimator.propagate_to(2.0)
    with pytest.raises(ValueError):
        estimator.propagate_to(np.nan)


@pytest.mark.parametrize('covariance', [np.eye(14), np.full((15, 15), np.nan),
                                       np.diag([-0.1] + [1] * 14),
                                       np.eye(15) + np.eye(15, k=1)])
def test_invalid_initial_covariance_is_rejected(covariance):
    with pytest.raises(ValueError):
        filter_(covariance=covariance)


@pytest.mark.parametrize('std', [0, -0.1, np.nan, [0.1, 0.1]])
def test_invalid_range_standard_deviation_is_rejected(std):
    estimator = filter_()
    with pytest.raises(ValueError):
        estimator.update_uwb([4] * 4, ANCHORS, [0, 0, 0], std)


@pytest.mark.parametrize('measured_range', [-0.2, 0.0])
def test_finite_nonpositive_raw_range_uses_the_gaussian_update_without_clipping(measured_range):
    covariance = np.zeros((15, 15))
    covariance[0, 0] = 1
    estimator = filter_(covariance=covariance)
    diagnostics = estimator.update_uwb([measured_range], [[-10, 0, 2]], [0, 0, 0], 0.2)
    innovation = measured_range - 10
    np.testing.assert_allclose(diagnostics['innovation'], [innovation])
    assert estimator.state.position[0] == pytest.approx(innovation / 1.04)
    assert diagnostics['nis'] == pytest.approx(innovation ** 2 / 1.04)


def test_invalid_state_input_and_zero_range_geometry_are_rejected():
    with pytest.raises(ValueError):
        state(quaternion=[0, 0, 0, 0])
    with pytest.raises(ValueError):
        state(position=[np.nan, 0, 0])
    with pytest.raises(ValueError):
        filter_(noise={**ZERO_NOISE, 'acc_std': -1})
    with pytest.raises(ValueError):
        filter_(initial_time=-1)
    with pytest.raises(ValueError):
        api('eskf', 'measurement_model')(state(), [[0, 0, 2]], [0, 0, 0])
    with pytest.raises(ValueError):
        filter_().set_imu(0, [np.inf, 0, 0], [0, 0, 0])


def test_vector_gravity_and_input_arrays_are_copied():
    nominal = state()
    covariance = np.eye(15)
    estimator = filter_(nominal, covariance, gravity=[0, 0, -3])
    nominal.position[0], covariance[0, 0] = 100, 100
    estimator.set_imu(0, [0, 0, 0], [0, 0, 0])
    estimator.propagate_to(1)
    np.testing.assert_allclose(estimator.state.position, [0, 0, 0.5])
    assert estimator.P[0, 0] < 10
