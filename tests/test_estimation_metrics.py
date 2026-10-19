"""检验离线评价的时间匹配、空间误差和诊断统计口径。"""

from copy import deepcopy
import json

import numpy as np
import pytest

from evaluation import metrics


def synthetic_estimation_data():
    """构造包含杆臂与不同采样时钟的解析案例。"""
    time = np.arange(9, dtype=float) / 4
    uwb_time = time[::2]
    position = np.column_stack((time, time ** 2, 1 + 0 * time))
    velocity = np.column_stack((np.ones_like(time), 2 * time, 0 * time))
    quaternion = np.tile([1., 0., 0., 0.], (len(time), 1))
    lever = np.array([0.2, 0.1, 0.05])
    anchors = np.array([[0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, 5.]])
    tag = position + lever
    dataset = {
        'truth': {'time': time.copy(), 'position': position, 'velocity': velocity,
                  'quaternion': quaternion, 'omega': np.zeros((len(time), 3)),
                  'imu_acc_bias': np.zeros((len(time), 3)),
                  'imu_gyro_bias': np.zeros((len(time), 3))},
        'imu': {'time': time.copy(), 'accel': np.zeros((len(time), 3)),
                'gyro': np.zeros((len(time), 3))},
        'uwb': {'time': uwb_time.copy(), 'anchors': anchors, 'lever_arm': lever,
                'ranges': np.linalg.norm(tag[::2, None, :] - anchors[None, :, :], axis=2)},
        'rpm': {'time': uwb_time.copy(), 'rpm': np.ones((len(uwb_time), 4))},
    }
    count, events, updates = len(uwb_time), len(time), len(uwb_time) - 1
    estimates = {
        'uwb_ls': {'time': uwb_time.copy(), 'tag_position': tag[::2] + [3., 4., 0.],
                   'covariance': np.tile(np.eye(3), (count, 1, 1)),
                   'residual': np.zeros((count, len(anchors))),
                   'converged': np.ones(count, dtype=bool), 'iterations': np.ones(count, dtype=int)},
        'eskf': {'time': time.copy(), 'position': position + [-.5, 0, 0],
                 'velocity': velocity + [0, .2, 0], 'quaternion': quaternion.copy(),
                 'acc_bias': np.tile([0, 0, .03], (events, 1)),
                 'gyro_bias': np.tile([.04, 0, 0], (events, 1)),
                 'covariance': np.tile(np.eye(15), (events, 1, 1)),
                 'tag_position': tag + [-.5, 0, 0]},
        'diagnostics': {'time': uwb_time[1:].copy(), 'innovation': np.ones((updates, len(anchors))),
                        'innovation_covariance': np.tile(np.eye(len(anchors)), (updates, 1, 1)),
                        'nis': np.full(updates, float(len(anchors)))},
    }
    return dataset, estimates


def test_known_offsets_use_three_dimensional_rmse_and_common_uwb_times():
    dataset, estimates = synthetic_estimation_data()
    # 非 UWB 时刻的大误差不能影响公平比较的天线误差。
    estimates['eskf']['tag_position'][1::2] += 100
    report = metrics.evaluate_estimates(dataset, estimates)
    assert report['common_samples'] == 5
    assert report['uwb_ls']['tag_position_rmse_m'] == pytest.approx(5.)
    assert report['eskf']['tag_position_rmse_m'] == pytest.approx(.5)
    assert report['eskf']['body_position_rmse_m'] == pytest.approx(.5)
    assert report['eskf']['velocity_rmse_mps'] == pytest.approx(.2)
    assert report['eskf']['acc_bias_rmse_mps2'] == pytest.approx(.03)
    assert report['eskf']['gyro_bias_rmse_radps'] == pytest.approx(.04)
    for key in ['mean', 'median', 'p95', 'max']:
        assert report['uwb_ls']['tag_ape_m'][key] == pytest.approx(5.)
    assert report['eskf']['state_samples'] == 9
    json.dumps(report, allow_nan=False)


def test_attitude_double_cover_and_shortest_rotation_are_equivalent():
    dataset, estimates = synthetic_estimation_data()
    angle = np.deg2rad(60.)
    estimates['eskf']['quaternion'][:] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
    first = metrics.evaluate_estimates(dataset, estimates)
    estimates['eskf']['quaternion'][::2] *= -1
    second = metrics.evaluate_estimates(dataset, estimates)
    assert first['eskf']['attitude_rmse_deg'] == pytest.approx(60.)
    assert second['eskf']['attitude_rmse_deg'] == pytest.approx(60.)
    assert second['diagnostics']['nees15'] == first['diagnostics']['nees15']
    estimates['eskf']['quaternion'][:] = [-1., 0., 0., 0.]
    assert metrics.evaluate_estimates(dataset, estimates)['eskf']['attitude_rmse_deg'] == 0.


def test_translation_rpe_uses_actual_one_second_pairs_and_reports_count():
    dataset, estimates = synthetic_estimation_data()
    estimates['eskf']['tag_position'][:, 0] += .1 * estimates['eskf']['time']
    report = metrics.evaluate_estimates(dataset, estimates)
    rpe = report['eskf']['translation_rpe_1s_m']
    assert rpe['count'] == 3
    assert rpe['rmse'] == pytest.approx(.1)
    assert report['uwb_ls']['translation_rpe_1s_m']['rmse'] == pytest.approx(0.)
    for group in ['truth', 'imu', 'uwb', 'rpm']:
        dataset[group]['time'] *= .1
    for group in ['uwb_ls', 'eskf', 'diagnostics']:
        estimates[group]['time'] *= .1
    unavailable = metrics.evaluate_estimates(dataset, estimates)['eskf']['translation_rpe_1s_m']
    assert unavailable['count'] == 0
    assert unavailable['rmse'] is None


def test_nees_uses_true_minus_nominal_and_right_local_attitude_error():
    dataset, estimates = synthetic_estimation_data()
    angle = .3
    dataset['truth']['quaternion'][:] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
    covariance = np.eye(15)
    covariance[0, 8] = covariance[8, 0] = .4
    estimates['eskf']['covariance'][:] = covariance
    error = np.array([.5, 0, 0, 0, -.2, 0, 0, 0, angle, 0, 0, -.03, -.04, 0, 0])
    expected = error @ np.linalg.solve(covariance, error)
    report = metrics.evaluate_estimates(dataset, estimates)
    assert report['diagnostics']['nees15']['mean'] == pytest.approx(expected)
    assert report['diagnostics']['nees15_per_dof']['mean'] == pytest.approx(expected / 15)
    assert report['diagnostics']['nis']['mean'] == 4.
    assert report['diagnostics']['nis_per_dof']['mean'] == 1.


@pytest.mark.parametrize('kind', ['singular', 'indefinite', 'asymmetric', 'nonfinite', 'ill_conditioned'])
def test_invalid_covariances_are_explicitly_excluded(kind):
    dataset, estimates = synthetic_estimation_data()
    covariance = estimates['eskf']['covariance'][0]
    if kind == 'singular':
        covariance[0, 0] = 0
    elif kind == 'indefinite':
        covariance[0, 0] = -1
    elif kind == 'asymmetric':
        covariance[0, 1] = 1
    elif kind == 'nonfinite':
        covariance[0, 0] = np.nan
    else:
        covariance[0, 0] = 1e-14
    report = metrics.evaluate_estimates(dataset, estimates)
    assert report['diagnostics']['nees15']['count'] == 8
    assert report['diagnostics']['nees15_excluded_samples'] == 1
    assert report['diagnostics']['nees15_exclusions'][kind] == 1
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('change', ['uwb_clock', 'eskf_clock', 'missing_truth', 'out_of_bounds'])
def test_unmatched_or_out_of_bounds_times_fail_explicitly(change):
    dataset, estimates = synthetic_estimation_data()
    if change == 'uwb_clock':
        estimates['uwb_ls']['time'][2] += 1e-7
    elif change == 'eskf_clock':
        estimates['eskf']['time'][2] += 1e-7
    elif change == 'out_of_bounds':
        estimates['eskf']['time'][-1] += 5e-13
    else:
        for field in dataset['truth']:
            dataset['truth'][field] = np.delete(dataset['truth'][field], 1, axis=0)
    with pytest.raises(ValueError, match='时刻|时间'):
        metrics.evaluate_estimates(dataset, estimates)


def test_internal_time_matches_accept_only_roundoff_tolerance():
    dataset, estimates = synthetic_estimation_data()
    estimates['eskf']['time'][2] += 5e-13
    assert metrics.evaluate_estimates(dataset, estimates)['common_samples'] == 5


def test_empty_diagnostics_and_timing_remain_serializable_without_fake_zeros():
    dataset, estimates = synthetic_estimation_data()
    for key in estimates['diagnostics']:
        estimates['diagnostics'][key] = estimates['diagnostics'][key][:0]
    estimates['timing'] = {'ls_update_seconds': np.full(5, .002),
                           'eskf_event_seconds': np.full(9, .003), 'initialization_seconds': .01}
    report = metrics.evaluate_estimates(dataset, estimates)
    assert report['diagnostics']['nis']['count'] == 0
    assert report['diagnostics']['nis']['mean'] is None
    assert report['timing']['ls_update_seconds']['mean'] == pytest.approx(.002)
    assert report['timing']['initialization_seconds'] == .01
    json.dumps(report, allow_nan=False)


def test_metrics_leave_all_inputs_unchanged():
    dataset, estimates = synthetic_estimation_data()
    copies = deepcopy((dataset, estimates))
    metrics.evaluate_estimates(dataset, estimates)
    for source, before in zip((dataset, estimates), copies):
        for group in source:
            for name, value in source[group].items():
                np.testing.assert_array_equal(value, before[group][name])
