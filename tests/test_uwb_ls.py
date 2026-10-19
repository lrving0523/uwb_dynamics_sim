"""验证原始测距单点定位、固定噪声协方差与失败边界。"""

import numpy as np
import pytest


ANCHORS = np.array([
    [0.0, 0.0, 0.0],
    [6.0, 0.0, 0.0],
    [0.0, 5.0, 0.0],
    [0.0, 0.0, 4.0],
    [6.0, 5.0, 4.0],
    [6.0, 0.0, 4.0],
])
POSITION = np.array([1.3, 2.1, 1.7])
RANGES = np.linalg.norm(POSITION - ANCHORS, axis=1)


def solve(*args, **kwargs):
    """在执行测试时导入待实现的公开入口。"""
    from estimator.uwb_ls import solve_uwb_position

    return solve_uwb_position(*args, **kwargs)


def test_exact_ranges_recover_tag_position_and_result_contract():
    result = solve(ANCHORS, RANGES)

    np.testing.assert_allclose(result['position'], POSITION, atol=1e-10)
    np.testing.assert_allclose(result['residual'], np.zeros(len(ANCHORS)), atol=1e-10)
    assert result['position'].shape == (3,)
    assert result['covariance'].shape == (3, 3)
    assert result['residual'].shape == (len(ANCHORS),)
    assert isinstance(result['iterations'], int)
    assert 0 <= result['iterations'] <= 50
    assert result['converged'] is True
    assert np.isfinite(result['condition_number'])
    assert result['condition_number'] >= 1.0


def test_nonzero_noise_uses_raw_distance_objective_and_measurement_minus_prediction():
    ranges = RANGES + np.array([0.02, -0.03, 0.01, 0.04, -0.02, 0.01])
    result = solve(ANCHORS, ranges, std=0.05)
    differences = result['position'] - ANCHORS
    predicted = np.linalg.norm(differences, axis=1)
    jacobian = differences / predicted[:, None]

    assert result['converged'] is True
    assert np.linalg.norm(result['position'] - POSITION) < 0.08
    np.testing.assert_allclose(result['residual'], ranges - predicted, atol=1e-14)
    np.testing.assert_allclose(jacobian.T @ result['residual'], np.zeros(3), atol=2e-7)


def test_covariance_is_fixed_noise_information_inverse_even_for_exact_ranges():
    result = solve(ANCHORS, RANGES, std=0.2)
    doubled = solve(ANCHORS, RANGES, std=0.4)
    direction = (POSITION - ANCHORS) / RANGES[:, None]
    expected = 0.04 * np.linalg.inv(direction.T @ direction)

    np.testing.assert_allclose(result['covariance'], expected, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(doubled['covariance'], 4 * result['covariance'], rtol=1e-12)
    assert np.all(np.linalg.eigvalsh(result['covariance']) > 0.0)


def test_noisy_ranges_keep_configured_covariance_without_residual_rescaling():
    ranges = RANGES + np.array([0.2, -0.1, 0.15, 0.1, -0.15, -0.2])
    result = solve(ANCHORS, ranges, std=0.05)
    difference = result['position'] - ANCHORS
    direction = difference / np.linalg.norm(difference, axis=1)[:, None]

    np.testing.assert_allclose(
        result['covariance'], 0.0025 * np.linalg.inv(direction.T @ direction), rtol=1e-11
    )


def test_previous_position_warmstart_converges_without_mutating_inputs():
    anchors = ANCHORS.copy()
    ranges = RANGES.copy()
    initial = POSITION + np.array([0.2, -0.15, 0.1])
    saved_initial = initial.copy()
    result = solve(anchors, ranges, initial=initial)

    assert result['converged'] is True
    np.testing.assert_allclose(result['position'], POSITION, atol=1e-8)
    np.testing.assert_array_equal(anchors, ANCHORS)
    np.testing.assert_array_equal(ranges, RANGES)
    np.testing.assert_array_equal(initial, saved_initial)


def test_iteration_limit_does_not_report_false_convergence():
    result = solve(ANCHORS, RANGES, initial=[4.0, 4.0, 3.0], max_iterations=1)

    assert result['iterations'] == 1
    assert result['converged'] is False
    assert np.linalg.norm(result['position'] - POSITION) > 1e-3


def test_small_negative_gaussian_range_is_preserved():
    position = np.array([0.01, 0.02, 0.03])
    ranges = np.linalg.norm(position - ANCHORS, axis=1)
    ranges[0] = -0.005
    result = solve(ANCHORS, ranges, initial=position, max_iterations=100)
    prediction = np.linalg.norm(result['position'] - ANCHORS[0])

    assert result['converged'] is True
    assert np.linalg.norm(result['position'] - position) < 0.06
    assert result['residual'][0] == pytest.approx(-0.005 - prediction)


def test_non_coplanar_four_anchor_case_and_large_coordinate_translation():
    translation = np.array([1e5, -2e5, 3e5])
    result = solve(ANCHORS[:4] + translation, RANGES[:4])

    assert result['converged'] is True
    np.testing.assert_allclose(result['position'], POSITION + translation, atol=1e-8)


@pytest.mark.parametrize('anchors,ranges', [
    (np.zeros((3, 3)), np.ones(3)),
    (np.zeros((4, 2)), np.ones(4)),
    (np.zeros((4, 3, 1)), np.ones(4)),
    ([], []),
    (ANCHORS, RANGES[:-1]),
    (ANCHORS, RANGES[:, None]),
    (ANCHORS, np.full(len(ANCHORS), np.nan)),
    (ANCHORS, np.full(len(ANCHORS), np.inf)),
    (np.where(ANCHORS == 6.0, np.nan, ANCHORS), RANGES),
    (ANCHORS, -RANGES),
    (ANCHORS, np.zeros(len(ANCHORS))),
])
def test_invalid_measurements_are_rejected(anchors, ranges):
    with pytest.raises(ValueError):
        solve(anchors, ranges)


@pytest.mark.parametrize('anchors', [
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]],
    [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
    [[1, 1, 1]] * 4,
])
def test_affine_degenerate_anchor_geometry_is_rejected(anchors):
    ranges = np.linalg.norm(POSITION - np.asarray(anchors), axis=1)
    with pytest.raises(ValueError, match='几何|共面|秩'):
        solve(anchors, ranges)


def test_ill_conditioned_range_jacobian_is_rejected_instead_of_false_covariance():
    far_position = np.array([1e8, 1e8, 1e8])
    far_ranges = np.linalg.norm(far_position - ANCHORS, axis=1)

    with pytest.raises(ValueError, match='病态|条件|雅可比'):
        solve(ANCHORS, far_ranges, initial=far_position)


def test_position_on_anchor_has_no_defined_range_jacobian():
    ranges = np.linalg.norm(ANCHORS[0] - ANCHORS, axis=1)
    with pytest.raises(ValueError, match='重合|零距离|雅可比'):
        solve(ANCHORS, ranges, initial=ANCHORS[0])


def test_linear_initialization_at_anchor_does_not_hide_zero_distance_in_roundoff():
    ranges = np.linalg.norm(ANCHORS[0] - ANCHORS, axis=1)
    with pytest.raises(ValueError, match='重合|零距离|雅可比'):
        solve(ANCHORS, ranges)


@pytest.mark.parametrize('std', [0, -0.1, np.nan, np.inf, [0.05], True])
def test_noise_standard_deviation_must_be_finite_positive_scalar(std):
    with pytest.raises(ValueError):
        solve(ANCHORS, RANGES, std=std)


@pytest.mark.parametrize('initial', [[1, 2], [[1, 2, 3]], [1, 2, np.nan], [1, 2, np.inf]])
def test_invalid_initial_position_is_rejected(initial):
    with pytest.raises(ValueError):
        solve(ANCHORS, RANGES, initial=initial)


@pytest.mark.parametrize('parameters', [
    {'max_iterations': 0}, {'max_iterations': -1}, {'max_iterations': 1.5},
    {'max_iterations': True}, {'tolerance': 0}, {'tolerance': -1},
    {'tolerance': np.nan}, {'tolerance': np.inf}, {'tolerance': [1e-8]},
])
def test_invalid_solver_parameters_are_rejected(parameters):
    with pytest.raises(ValueError):
        solve(ANCHORS, RANGES, **parameters)
