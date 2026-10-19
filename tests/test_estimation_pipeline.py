"""检验阶段二初始化、因果事件调度与独立观测接口。"""

from copy import deepcopy

import numpy as np
import pytest

from estimator.pipeline import load_estimation_settings, run_estimators, initial_covariance
from estimator.result_io import load_estimates, save_estimates
from simulator.config import load_config
from simulator.math3d import rotation_matrix, skew
from simulator.runner import run_simulation


def measurements_and_settings():
    config = load_config('config/sim.yaml')
    config['simulation']['duration'] = 0.4
    config['trajectory']['type'] = 'hover'
    data = run_simulation(config)
    measurements = {name: data[name] for name in ['imu', 'uwb']}
    settings = load_estimation_settings('config/estimation.yaml', config)
    return data, measurements, settings


def test_filter_uses_measurements_and_declared_prior_only():
    data, measurements, settings = measurements_and_settings()
    result = run_estimators(measurements, settings)
    assert result['eskf']['covariance'].shape[1:] == (15, 15)
    assert result['eskf']['time'][0] == measurements['uwb']['time'][0]
    assert np.all(result['eskf']['latest_imu_time'] <= result['eskf']['time'])
    assert result['diagnostics']['time'][0] > result['eskf']['time'][0]
    with pytest.raises(ValueError, match='观测'):
        run_estimators(data, settings)
    changed = deepcopy(data)
    for field in changed['truth']:
        changed['truth'][field] = np.full_like(changed['truth'][field], 123456.0)
    second = run_estimators({name: changed[name] for name in ['imu', 'uwb']}, settings)
    for group in ['uwb_ls', 'eskf', 'diagnostics']:
        for key in result[group]:
            np.testing.assert_array_equal(result[group][key], second[group][key])


def test_future_data_does_not_change_existing_estimates():
    _, measurements, settings = measurements_and_settings()
    short = deepcopy(measurements)
    for name in ['imu', 'uwb']:
        mask = short[name]['time'] <= 0.2
        for key in ['time', 'accel', 'gyro', 'ranges']:
            if key in short[name]:
                short[name][key] = short[name][key][mask]
    full, prefix = run_estimators(measurements, settings), run_estimators(short, settings)
    for group in ['uwb_ls', 'eskf', 'diagnostics']:
        size = len(prefix[group]['time'])
        for key in prefix[group]:
            np.testing.assert_array_equal(prefix[group][key], full[group][key][:size])


def test_initial_lever_arm_covariance_keeps_tag_uncertainty():
    _, measurements, settings = measurements_and_settings()
    position_covariance = np.diag([0.01, 0.02, 0.03])
    covariance = initial_covariance(position_covariance, settings, measurements['uwb']['lever_arm'])
    rotation = rotation_matrix(settings['initial']['quaternion'])
    jacobian = np.zeros((3, 15))
    jacobian[:, :3] = np.eye(3)
    jacobian[:, 6:9] = -rotation @ skew(measurements['uwb']['lever_arm'])
    np.testing.assert_allclose(jacobian @ covariance @ jacobian.T, position_covariance, atol=1e-14)
    assert np.linalg.norm(covariance[:3, 6:9]) > 0.0


def test_initialization_does_not_reuse_first_ranges():
    _, measurements, settings = measurements_and_settings()
    result = run_estimators(measurements, settings)
    np.testing.assert_allclose(result['eskf']['tag_position'][0], result['uwb_ls']['tag_position'][0], atol=1e-12)
    assert len(result['diagnostics']['time']) == len(measurements['uwb']['time'])-1
    assert np.min(np.linalg.eigvalsh(result['eskf']['covariance'])) >= -1e-12


def test_no_future_imu_is_used_for_initialization():
    _, measurements, settings = measurements_and_settings()
    measurements['imu']['time'] += 0.001
    with pytest.raises(ValueError, match='IMU'):
        run_estimators(measurements, settings)


def test_noninteger_rate_events_are_preserved_exactly():
    config = load_config('config/sim.yaml')
    config['simulation']['duration'] = 0.35
    config['sensors']['imu']['hz'] = 73.0
    config['sensors']['uwb']['hz'] = 11.0
    data = run_simulation(config)
    settings = load_estimation_settings('config/estimation.yaml', config)
    result = run_estimators({name: data[name] for name in ['imu', 'uwb']}, settings)
    expected = np.unique(np.concatenate([data['imu']['time'], data['uwb']['time']]))
    np.testing.assert_array_equal(result['eskf']['time'], expected)
    np.testing.assert_array_equal(result['uwb_ls']['time'], data['uwb']['time'])


def test_save_estimates_round_trip(tmp_path):
    _, measurements, settings = measurements_and_settings()
    result = run_estimators(measurements, settings)
    save_estimates(result, settings, tmp_path)
    restored = load_estimates(tmp_path)
    for group in result:
        for key in result[group]:
            np.testing.assert_array_equal(result[group][key], restored[group][key])
    assert (tmp_path/'estimation.resolved.yaml').is_file()
    assert (tmp_path/'estimation_manifest.json').is_file()


def test_independent_prior_does_not_copy_simulator_initial_state():
    config = load_config('config/sim.yaml')
    config['drone']['initial_velocity'] = [10.0, -2.0, 3.0]
    config['drone']['initial_quaternion'] = [0.0, 1.0, 0.0, 0.0]
    config['sensors']['imu']['acc_bias'] = [3.0, 2.0, 1.0]
    settings = load_estimation_settings('config/estimation.yaml', config)
    np.testing.assert_array_equal(settings['initial']['velocity'], [0.0]*3)
    np.testing.assert_array_equal(settings['initial']['quaternion'], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(settings['initial']['acc_bias'], [0.0]*3)


def test_delayed_initialization_uses_latest_already_received_imu():
    _, measurements, settings = measurements_and_settings()
    measurements['uwb']['time'] += 0.013
    estimates = run_estimators(measurements, settings)
    assert estimates['eskf']['time'][0] == 0.013
    assert estimates['eskf']['latest_imu_time'][0] == 0.01
    assert np.all(estimates['eskf']['latest_imu_time'] <= estimates['eskf']['time'])


def test_one_uwb_frame_produces_empty_innovation_diagnostics():
    _, measurements, settings = measurements_and_settings()
    measurements['uwb']['time'] = measurements['uwb']['time'][:1]
    measurements['uwb']['ranges'] = measurements['uwb']['ranges'][:1]
    estimates = run_estimators(measurements, settings)
    assert estimates['diagnostics']['innovation'].shape == (0, 8)
    assert estimates['diagnostics']['innovation_covariance'].shape == (0, 8, 8)
    assert estimates['diagnostics']['nis'].shape == (0,)


def test_observation_arrays_are_not_modified():
    _, measurements, settings = measurements_and_settings()
    original = deepcopy(measurements)
    run_estimators(measurements, settings)
    for group in measurements:
        for key in measurements[group]:
            np.testing.assert_array_equal(measurements[group][key], original[group][key])
