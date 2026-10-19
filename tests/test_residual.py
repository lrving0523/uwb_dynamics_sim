"""真实因果网络、整飞行划分与离线气动力标签验证。"""

from copy import deepcopy
import importlib

import numpy as np
import pytest
import torch

from models import residual


def test_tcn_is_causal_and_accepts_batch_time_features():
    torch.manual_seed(3)
    net = residual.ResidualTCN(channels=8, layers=3, kernel_size=3)
    inputs = torch.randn(2, 10, 7)
    changed = inputs.clone(); changed[:, 6:] += 100
    first, second = net.forward_sequence(inputs), net.forward_sequence(changed)
    assert first.shape == (2, 10, 3)
    torch.testing.assert_close(first[:, :6], second[:, :6])
    torch.testing.assert_close(net(inputs), first[:, -1])
    assert any(isinstance(layer, torch.nn.Conv1d) for layer in net.modules())


def test_training_fits_only_training_normalization_and_roundtrips(tmp_path):
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(80, 10, 7)).astype('float32')
    train_y = (.2*train_x[:, -1, :3]).astype('float32')
    val_x = rng.normal(size=(20, 10, 7)).astype('float32')+4
    val_y = .2*val_x[:, -1, :3]
    predictor, history = residual.train_residual_model(
        train_x, train_y, val_x, val_y,
        {'channels': 8, 'layers': 2, 'kernel_size': 3, 'sequence_length': 10, 'input_hz': 100},
        {'epochs': 12, 'batch_size': 40, 'learning_rate': .01, 'seed': 5, 'threads': 1},
        {'data_split': {'train': ['a'], 'val': ['b'], 'test': ['c']}})
    np.testing.assert_allclose(predictor.input_mean.numpy(), train_x.mean(axis=(0, 1)), atol=1e-6)
    assert history[-1]['train_loss'] < history[0]['train_loss']
    assert predictor.metadata['trained'] is True
    path = tmp_path/'network.pt'; predictor.save(path)
    checkpoint = torch.load(path, weights_only=True)
    assert checkpoint['metadata']['data_split']['test'] == ['c']
    restored = residual.ResidualPredictor.load(path)
    np.testing.assert_array_equal(predictor.predict(train_x[0]), restored.predict(train_x[0]))
    assert restored.sequence_length == 10 and restored.input_hz == 100


def test_offline_labels_exclude_external_force_mass_and_thrust_error():
    data_module = importlib.import_module('models.training_data')
    from models.physics import PhysicsModel
    times = np.arange(12)/100
    dataset = {'rpm': {'time': times, 'rpm': np.ones((12, 4))*5000},
               'imu': {'time': times, 'gyro': np.ones((12, 3))},
               'truth': {'time': times, 'quaternion': np.tile([1, 0, 0, 0], (12, 1)),
                         'velocity': np.tile([2, 0, 0], (12, 1)),
                         'aero_force_body': np.tile([-.7, 0, 0], (12, 1)),
                         'external_force_world': np.ones((12, 3))*100,
                         'acceleration_world': np.ones((12, 3))*100,
                         'true_rpm': np.ones((12, 4))*100}}
    model = PhysicsModel({'mass': 50, 'kf': .0003, 'linear_drag': [.1, 0, 0]})
    result = data_module.flight_windows(dataset, model, 10)
    np.testing.assert_allclose(result['targets'], np.tile([-.5, 0, 0], (3, 1)))
    assert result['inputs'].shape == (3, 10, 7)
    changed = deepcopy(dataset)
    changed['truth']['external_force_world'] *= -7
    np.testing.assert_array_equal(data_module.flight_windows(changed, model, 10)['targets'], result['targets'])


def test_causal_features_never_use_future_gyro_and_ignore_truth():
    data_module = importlib.import_module('models.training_data')
    times = np.arange(12)/100
    measurements = {'rpm': {'time': times, 'rpm': np.ones((12, 4))*5000},
                    'imu': {'time': np.r_[0, .005, .015, .025, 1.0], 'gyro': np.repeat(np.arange(5)[:, None], 3, axis=1)}}
    _, features = data_module.causal_features(measurements)
    np.testing.assert_array_equal(features[:4, 0], [0, 1, 2, 3])
    changed = deepcopy(measurements); changed['imu']['gyro'][-1] = 100000
    np.testing.assert_array_equal(features, data_module.causal_features(changed)[1])
    measurements['rpm']['time'][5] += .001
    with pytest.raises(ValueError):
        data_module.causal_features(measurements)


def test_data_split_rejects_whole_flight_overlap():
    data_module = importlib.import_module('models.training_data')
    with pytest.raises(ValueError):
        data_module.validate_split({'train': ['a', 'b'], 'val': ['b'], 'test': ['c']})


def test_training_entrypoint_saves_real_data_checkpoint_and_independent_test(tmp_path):
    from experiments.train_residual import run_training
    config = {'physics': {}, 'network': {'channels': 4, 'layers': 1, 'kernel_size': 3},
              'training': {'epochs': 2, 'batch_size': 16, 'seed': 7, 'threads': 1},
              'data': {'simulation_config': 'config/sim.yaml', 'duration': .15, 'dt': .01,
                       'split_seeds': {'train': [10], 'val': [20], 'test': [30]}}}
    report = run_training(config, tmp_path)
    assert report['trained'] is True
    assert report['test_force_rmse_3d_N'] >= 0
    assert (tmp_path/'residual_tcn.pt').is_file()
    restored = residual.ResidualPredictor.load(tmp_path/'residual_tcn.pt')
    assert restored.metadata['nominal_physics']['linear_drag'] == [0., 0., 0.]
    assert restored.metadata['nominal_physics']['mass'] == 1.5
    assert (tmp_path/'flights'/'train_10_circle'/'truth.npz').is_file()
    with np.load(tmp_path/'test_predictions.npz', allow_pickle=False) as predictions:
        assert predictions['target_force_body'].shape == (7, 3)
        np.testing.assert_array_equal(predictions['physics_residual_force_body'], 0)
        assert set(predictions['flight_id']) == {0}


def test_measurement_fingerprint_tracks_only_network_visible_history():
    from models.training_data import measurement_fingerprint
    times = np.arange(12)/100
    measurements = {'rpm': {'time': times, 'rpm': np.ones((12, 4))*5000},
                    'imu': {'time': times, 'gyro': np.zeros((12, 3)), 'accel': np.ones((12, 3))},
                    'uwb': {'ranges': np.ones((2, 4))}}
    baseline = measurement_fingerprint(measurements)
    measurements['imu']['accel'] *= 2
    measurements['uwb']['ranges'] *= 3
    assert measurement_fingerprint(measurements) == baseline
    measurements['imu']['gyro'][0] = .1
    assert measurement_fingerprint(measurements) != baseline
