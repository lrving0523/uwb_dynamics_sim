"""验证配置、异步时钟和可复现数据生成。"""

from copy import deepcopy
import json

import numpy as np
import pytest

from simulator.config import load_config, validate_config
from simulator.dataset import load_dataset, save_dataset
from simulator.runner import run_simulation, sample_times


def short_config():
    config = load_config('config/sim.yaml')
    config['simulation']['duration'] = 0.31
    config['simulation']['dt'] = 0.01
    config['simulation']['controller_hz'] = 83.0
    config['sensors']['imu']['hz'] = 137.0
    config['sensors']['rpm']['hz'] = 73.0
    config['sensors']['uwb']['hz'] = 11.0
    config['trajectory']['type'] = 'hover'
    return config


def test_arbitrary_frequencies_and_no_fabricated_endpoint():
    config = short_config()
    dataset = run_simulation(config)
    for name, hz in [('imu', 137.0), ('rpm', 73.0), ('uwb', 11.0)]:
        expected = np.arange(int(np.floor(0.31 * hz)) + 1) / hz
        np.testing.assert_allclose(dataset[name]['time'], expected, atol=1e-14)
        assert np.all(np.diff(dataset[name]['time']) > 0)
        assert dataset[name]['time'][-1] < 0.31
    assert dataset['truth']['time'][-1] == 0.31
    assert dataset['truth']['position'].shape[1:] == (3,)
    for time in sample_times(0.31, 100):
        assert np.any(np.isclose(dataset['truth']['time'], time, atol=1e-14, rtol=0))


def test_seed_reproduces_every_numeric_array():
    config = short_config()
    first, second = run_simulation(config), run_simulation(config)
    for group in first:
        for key in first[group]:
            np.testing.assert_array_equal(first[group][key], second[group][key])
    changed = deepcopy(config)
    changed['simulation']['seed'] += 1
    assert not np.array_equal(first['imu']['accel'], run_simulation(changed)['imu']['accel'])


def test_sensor_noise_streams_are_independent():
    config = short_config()
    changed = deepcopy(config)
    changed['sensors']['uwb']['std'] = 100.0
    first, second = run_simulation(config), run_simulation(changed)
    for name in ['imu', 'rpm', 'truth']:
        for key in first[name]:
            np.testing.assert_array_equal(first[name][key], second[name][key])


def test_save_reload_and_measurement_truth_separation(tmp_path):
    config = short_config()
    dataset = run_simulation(config)
    save_dataset(dataset, config, tmp_path)
    restored = load_dataset(tmp_path)
    for group in dataset:
        for key in dataset[group]:
            np.testing.assert_array_equal(dataset[group][key], restored[group][key])
    assert set(restored['imu']) == {'time', 'accel', 'gyro'}
    assert set(restored['rpm']) == {'time', 'rpm'}
    assert set(restored['uwb']) == {'time', 'ranges', 'anchors', 'lever_arm'}
    assert 'imu_acc_bias' in restored['truth']
    assert (tmp_path / 'config.resolved.yaml').is_file()
    assert (tmp_path / 'truth.csv').is_file()
    manifest = json.loads((tmp_path / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['seed'] == config['simulation']['seed']
    assert 'truth.npz' in manifest['sha256']
    reloaded_config = load_config(tmp_path / 'config.resolved.yaml')
    replay = run_simulation(reloaded_config)
    np.testing.assert_array_equal(dataset['imu']['accel'], replay['imu']['accel'])


@pytest.mark.parametrize('path,value', [
    (('simulation', 'dt'), 0),
    (('simulation', 'seed'), -1),
    (('drone', 'mass'), -1),
    (('motor', 'time_constant'), 0),
    (('motor', 'max_rpm'), 10),
    (('sensors', 'imu', 'hz'), float('nan')),
    (('sensors', 'uwb', 'std'), -1),
    (('drone', 'initial_quaternion'), [0, 0, 0, 0]),
])
def test_invalid_configuration_is_rejected(path, value):
    config = short_config()
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_clock_zero_duration_and_fractional_end():
    np.testing.assert_array_equal(sample_times(0.0, 73), [0.0])
    np.testing.assert_allclose(sample_times(0.3, 10), [0, 0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        sample_times(1.0, 0)


def test_zero_noise_samples_match_current_physical_truth():
    from simulator.math3d import rotation_matrix

    config = short_config()
    imu = config['sensors']['imu']
    for key in ['acc_std', 'gyro_std', 'acc_bias_rw_std', 'gyro_bias_rw_std']:
        imu[key] = 0.0
    imu['acc_bias'] = [0.0]*3
    imu['gyro_bias'] = [0.0]*3
    config['sensors']['rpm']['std'] = 0.0
    config['sensors']['uwb']['std'] = 0.0
    data = run_simulation(config)
    truth = data['truth']
    gravity = np.array([0.0, 0.0, -config['drone']['gravity']])
    for name in ['imu', 'rpm', 'uwb']:
        indices = np.argmin(abs(data[name]['time'][:, None]-truth['time'][None, :]), axis=1)
        np.testing.assert_allclose(truth['time'][indices], data[name]['time'], atol=1e-14)
        if name == 'imu':
            specific = [rotation_matrix(truth['quaternion'][i]).T @
                        (truth['acceleration_world'][i]-gravity) for i in indices]
            np.testing.assert_allclose(data[name]['accel'], specific, atol=1e-12)
            np.testing.assert_allclose(data[name]['gyro'], truth['omega'][indices], atol=1e-12)
        elif name == 'rpm':
            np.testing.assert_array_equal(data[name]['rpm'], truth['true_rpm'][indices])
        else:
            tag = [truth['position'][i] + rotation_matrix(truth['quaternion'][i]) @
                   data['uwb']['lever_arm'] for i in indices]
            ranges = np.linalg.norm(np.asarray(tag)[:, None, :] - data['uwb']['anchors'], axis=2)
            np.testing.assert_allclose(data[name]['ranges'], ranges, atol=1e-12)


def test_off_grid_disturbance_is_an_exact_integration_boundary():
    config = short_config()
    onset = 0.123456
    config['wind'].update(enabled=True, type='step', start_time=onset, force=[1.0, 0.0, 0.0])
    truth = run_simulation(config)['truth']
    index = np.flatnonzero(truth['time'] == onset)
    assert len(index) == 1
    np.testing.assert_allclose(truth['velocity'][index[0]], 0.0, atol=1e-12)
    np.testing.assert_array_equal(truth['external_force_world'][index[0]], [1.0, 0.0, 0.0])


def test_nearby_step_boundary_preserves_full_force_impulse():
    config = short_config()
    config['simulation'].update(duration=0.11, controller_hz=100.0)
    config['sensors']['imu']['hz'] = 200.0
    config['sensors']['rpm']['hz'] = 100.0
    config['sensors']['uwb']['hz'] = 10.0
    config['aerodynamics']['linear_drag'] = [0.0]*3
    config['aerodynamics']['quadratic_drag'] = [0.0]*3
    for key in ['position_kp', 'velocity_kd', 'position_ki']:
        config['controller'][key] = [0.0]*3
    onset = np.nextafter(0.1, np.inf)
    config['wind'].update(enabled=True, type='step', start_time=float(onset), force=[6.0, 0.0, 0.0])
    truth = run_simulation(config)['truth']
    expected = 6.0/config['drone']['mass']*(0.11-onset)
    assert truth['velocity'][-1, 0] == pytest.approx(expected, abs=1e-13)
    assert np.any(truth['time'] == onset)


def test_clock_never_samples_after_a_just_short_endpoint():
    duration = np.nextafter(0.1, 0.0)
    times = sample_times(duration, 10.0)
    np.testing.assert_array_equal(times, [0.0])
    assert np.all(times <= duration)
