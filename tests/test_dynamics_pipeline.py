"""检验动力学增强调度的因果性、统一先验与无补偿消融。"""

from copy import deepcopy

import numpy as np
import pytest

from estimator.dynamics_pipeline import run_dynamics_estimators, load_dynamics_settings
from estimator.dynamics_io import save_dynamics_results, load_dynamics_results
from models.physics import PhysicsModel
from models.hybrid import HybridModel
from simulator.config import load_config
from simulator.runner import run_simulation


class ZeroResidual:
    """仅供算法等价性测试的零残余，不用于正式学习对比。"""

    sequence_length = 10
    input_hz = 100.0
    metadata = {'trained': False}

    def predict(self, window):
        return np.zeros(3)


def case():
    config = load_config('config/sim.yaml')
    config['simulation']['duration'] = 0.22
    config['trajectory']['type'] = 'hover'
    dataset = run_simulation(config)
    settings, model_config = load_dynamics_settings('config/estimation.yaml', 'config/hybrid.yaml', config)
    measurements = {key: dataset[key] for key in ['imu', 'uwb', 'rpm']}
    physics = PhysicsModel(model_config)
    hybrid = HybridModel(PhysicsModel(model_config), ZeroResidual())
    return dataset, measurements, settings, physics, hybrid


def test_zero_residual_is_an_exact_physics_ablation():
    _, measurements, settings, physics, hybrid = case()
    results = run_dynamics_estimators(measurements, settings, physics, hybrid)
    for key in results['physics']:
        np.testing.assert_array_equal(results['physics'][key], results['hybrid'][key])
    np.testing.assert_array_equal(results['residual']['force_body'], 0.0)
    assert not results['residual']['ready'][0]
    assert results['residual']['ready'][9]
    assert results['physics']['covariance'].shape[1:] == (15, 15)


def test_observation_input_has_no_truth_and_is_immutable():
    dataset, measurements, settings, physics, hybrid = case()
    with pytest.raises(ValueError, match='观测'):
        run_dynamics_estimators(dataset, settings, physics, hybrid)
    original = deepcopy(measurements)
    run_dynamics_estimators(measurements, settings, physics, hybrid)
    for group in measurements:
        for key in measurements[group]:
            np.testing.assert_array_equal(measurements[group][key], original[group][key])


def test_future_rpm_and_imu_do_not_change_past_states():
    _, measurements, settings, physics, hybrid = case()
    original = run_dynamics_estimators(measurements, settings, physics, hybrid)
    changed = deepcopy(measurements)
    changed['rpm']['rpm'][changed['rpm']['time'] > 0.15] += 1000.0
    changed['imu']['gyro'][changed['imu']['time'] > 0.15] += 1.0
    second = run_dynamics_estimators(changed, settings, physics, hybrid)
    for method in ['physics', 'hybrid']:
        prefix = original[method]['time'] <= 0.15
        for key in original[method]:
            np.testing.assert_array_equal(original[method][key][prefix], second[method][key][prefix])


def test_no_future_rpm_at_initialization():
    _, measurements, settings, physics, hybrid = case()
    measurements['rpm']['time'] += 0.001
    with pytest.raises(ValueError, match='RPM'):
        run_dynamics_estimators(measurements, settings, physics, hybrid)


def test_slow_gyro_history_uses_only_available_samples():
    config = load_config('config/sim.yaml')
    config['simulation']['duration'] = 0.23
    config['sensors']['imu']['hz'] = 73.0
    config['sensors']['uwb']['hz'] = 11.0
    data = run_simulation(config)
    settings, model_config = load_dynamics_settings('config/estimation.yaml', 'config/hybrid.yaml', config)
    result = run_dynamics_estimators({key: data[key] for key in ['imu', 'uwb', 'rpm']}, settings,
                    PhysicsModel(model_config), HybridModel(PhysicsModel(model_config), ZeroResidual()))
    for method in ['physics', 'hybrid']:
        assert np.all(result[method]['latest_imu_time'] <= result[method]['time'])
        assert np.all(result[method]['latest_rpm_time'] <= result[method]['time'])
    assert np.min(np.linalg.eigvalsh(result['physics']['covariance'])) >= -1e-11


def test_hybrid_rejects_unmatched_rpm_feature_rate():
    _, measurements, settings, physics, hybrid = case()
    measurements['rpm']['time'] *= 1.1
    with pytest.raises(ValueError, match='100|频率'):
        run_dynamics_estimators(measurements, settings, physics, hybrid)


def test_dynamics_result_round_trip(tmp_path):
    _, measurements, settings, physics, hybrid = case()
    results = run_dynamics_estimators(measurements, settings, physics, hybrid)
    save_dynamics_results(results, settings, tmp_path)
    restored = load_dynamics_results(tmp_path)
    for method in ['physics', 'hybrid', 'residual', 'timing']:
        for key in results[method]:
            np.testing.assert_array_equal(results[method][key], restored[method][key])
    for group in ['uwb_diagnostics', 'accel_diagnostics']:
        for method in ['physics', 'hybrid']:
            for key in results[group][method]:
                np.testing.assert_array_equal(results[group][method][key], restored[group][method][key])
