"""检验局部常输入预测的真实端点、因果残余输入及离线误差口径。"""

from copy import deepcopy
import importlib
import importlib.util
import json

import numpy as np
import pytest


def prediction_module():
    """缺少模块时给出明确失败，避免以导入异常代替行为验证。"""
    assert importlib.util.find_spec('evaluation.model_prediction') is not None, '缺少离线模型预测评价模块'
    return importlib.import_module('evaluation.model_prediction')


class ConstantPhysics:
    """为解析匀加速度案例提供与正式模型相同的接口。"""

    mass = 2.
    gravity = 9.81

    def __init__(self, acceleration=(1., -.5, .2)):
        self.world_acceleration = np.asarray(acceleration, dtype=float)

    def acceleration(self, quaternion, velocity, rpm):
        return self.world_acceleration.copy()


def constant_acceleration_data(acceleration=(1., -.5, .2)):
    """保存精确解析真值和测量 RPM；不依赖训练或仿真。"""
    time = np.arange(11, dtype=float) / 10
    acceleration = np.asarray(acceleration, dtype=float)
    return {
        'truth': {'time': time.copy(), 'position': .5 * time[:, None] ** 2 * acceleration,
                  'velocity': time[:, None] * acceleration,
                  'quaternion': np.tile([1., 0., 0., 0.], (len(time), 1)),
                  'acceleration_world': np.tile(acceleration, (len(time), 1))},
        'rpm': {'time': time.copy(), 'rpm': np.zeros((len(time), 4))},
    }


def test_exact_constant_acceleration_has_zero_error_for_both_models():
    module = prediction_module()
    dataset = constant_acceleration_data()
    before = deepcopy(dataset)
    report = module.evaluate_model_predictions(dataset, ConstantPhysics(), {'time': [], 'force_body': np.empty((0, 3))})
    assert report['common_samples'] == 9
    assert report['excluded_samples']['horizon_out_of_range'] == 2
    assert '局部常输入' in report['prediction_assumption']
    for method in ('physics', 'hybrid'):
        for metric in ('acceleration_rmse_mps2', 'delta_velocity_rmse_mps', 'position_rmse_m'):
            assert report[method][metric] == pytest.approx(0., abs=1e-14)
    json.dumps(report, allow_nan=False)
    for group in dataset:
        for key in dataset[group]:
            np.testing.assert_array_equal(dataset[group][key], before[group][key])


def test_current_body_residual_improves_all_three_prediction_quantities():
    module = prediction_module()
    dataset = constant_acceleration_data((1., 2., 3.))
    history = {'time': [0.], 'force_body': [[2., 4., 6.]]}
    report = module.evaluate_model_predictions(dataset, ConstantPhysics((0., 0., 0.)), history)
    assert report['physics']['acceleration_rmse_mps2'] == pytest.approx(np.sqrt(14))
    assert report['physics']['delta_velocity_rmse_mps'] == pytest.approx(.2 * np.sqrt(14))
    assert report['physics']['position_rmse_m'] == pytest.approx(.02 * np.sqrt(14))
    for metric in ('acceleration_rmse_mps2', 'delta_velocity_rmse_mps', 'position_rmse_m'):
        assert report['hybrid'][metric] == pytest.approx(0., abs=1e-14)


def test_residual_rotation_uses_initial_body_to_world_attitude():
    module = prediction_module()
    dataset = constant_acceleration_data((0., 1., 0.))
    dataset['truth']['quaternion'][:] = [np.sqrt(.5), 0., 0., np.sqrt(.5)]
    report = module.evaluate_model_predictions(dataset, ConstantPhysics((0., 0., 0.)),
                                                {'time': [0.], 'force_body': [[2., 0., 0.]]})
    assert report['hybrid']['acceleration_rmse_mps2'] == pytest.approx(0., abs=1e-14)


def test_residual_history_is_previous_sample_hold_and_never_uses_future_values():
    module = prediction_module()
    dataset = constant_acceleration_data()
    history = {'time': np.array([.2, .8]), 'force_body': np.array([[2., 0., 0.], [100., 0., 0.]])}
    first = module.model_prediction_arrays(dataset, ConstantPhysics(), history)
    history['force_body'][1] = [-1000., 500., 30.]
    second = module.model_prediction_arrays(dataset, ConstantPhysics(), history)
    past = first['time'] < .8
    np.testing.assert_array_equal(first['hybrid']['acceleration'][past], second['hybrid']['acceleration'][past])
    np.testing.assert_array_equal(first['residual_force_body'][:2], np.zeros((2, 3)))
    np.testing.assert_array_equal(first['residual_force_body'][2:8], np.tile([2., 0., 0.], (6, 1)))


def test_missing_saved_endpoints_are_excluded_instead_of_interpolated():
    module = prediction_module()
    dataset = constant_acceleration_data()
    for key in dataset['truth']:
        dataset['truth'][key] = np.delete(dataset['truth'][key], 4, axis=0)
    report = module.evaluate_model_predictions(dataset, ConstantPhysics(), {'time': [], 'force_body': np.empty((0, 3))})
    assert report['common_samples'] == 7
    assert report['excluded_samples'] == {'initial_truth_unmatched': 1, 'horizon_truth_unmatched': 1,
                                           'horizon_out_of_range': 2}


def test_prediction_does_not_reset_to_intermediate_truth_states():
    module = prediction_module()
    dataset = constant_acceleration_data()
    history = {'time': [], 'force_body': np.empty((0, 3))}
    before = module.model_prediction_arrays(dataset, ConstantPhysics(), history)
    dataset['truth']['velocity'][1] += 100.
    dataset['truth']['position'][1] += 200.
    dataset['truth']['quaternion'][1] = [0., 0., 0., 1.]
    after = module.model_prediction_arrays(dataset, ConstantPhysics(), history)
    for field in ('position', 'delta_velocity', 'acceleration'):
        np.testing.assert_array_equal(before['physics'][field][0], after['physics'][field][0])


def test_no_valid_horizon_reports_null_errors_instead_of_zero():
    module = prediction_module()
    report = module.evaluate_model_predictions(constant_acceleration_data(), ConstantPhysics(),
                                                {'time': [], 'force_body': np.empty((0, 3))}, horizon_s=2.)
    assert report['common_samples'] == 0
    assert report['physics']['position_rmse_m'] is None
    assert report['hybrid']['acceleration_rmse_mps2'] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize('change', ['horizon', 'truth_shape', 'history_time', 'mass'])
def test_invalid_prediction_inputs_are_rejected(change):
    module = prediction_module()
    dataset, model = constant_acceleration_data(), ConstantPhysics()
    history, horizon = {'time': [0.], 'force_body': [[0., 0., 0.]]}, .2
    if change == 'horizon':
        horizon = 0.
    elif change == 'truth_shape':
        dataset['truth']['acceleration_world'] = np.zeros((2, 3))
    elif change == 'history_time':
        history = {'time': [.2, .1], 'force_body': np.zeros((2, 3))}
    else:
        model.mass = 0.
    with pytest.raises(ValueError):
        module.evaluate_model_predictions(dataset, model, history, horizon_s=horizon)
