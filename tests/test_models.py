"""名义动力学的单位、坐标与混合条件雅可比验证。"""

import numpy as np
import pytest

from models import physics, hybrid
from simulator.math3d import quaternion_multiply, quaternion_to_matrix


def test_hover_rpm_uses_radians_per_second_and_body_up():
    model = physics.PhysicsModel({})
    rpm = np.full(4, np.sqrt(1.5 * 9.81 / (4e-5)) * 60 / (2 * np.pi))
    np.testing.assert_allclose(model.force_body([1, 0, 0, 0], np.zeros(3), rpm), [0, 0, 14.715])
    np.testing.assert_allclose(model.acceleration([1, 0, 0, 0], np.zeros(3), rpm), 0, atol=1e-14)


def test_drag_is_opposite_body_relative_air_speed():
    model = physics.PhysicsModel({'linear_drag': [1, 2, 3], 'quadratic_drag': [.1, .2, .3],
                                 'wind_velocity': [1, 0, 0]})
    q = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
    np.testing.assert_allclose(model.force_body(q, [1, 2, -1], [0]*4), [-2.4, 0, 3.3], atol=1e-14)


def test_force_jacobians_match_independent_central_differences():
    model = physics.PhysicsModel({'linear_drag': [.1, .2, .3], 'quadratic_drag': [.04, .03, .02],
                                 'wind_velocity': [.4, -.3, .1]})
    q = np.array([.8, .1, .3, -.2]); q /= np.linalg.norm(q)
    v, rpm = np.array([1.3, -.8, .5]), np.array([4000., 5000., 6000., 5500.])
    jv, jt, jr = model.force_jacobians(q, v, rpm)
    eps = 1e-5
    def perturb_q(index, sign):
        axis = np.eye(3)[index]
        return quaternion_multiply(q, np.r_[np.cos(eps/2), sign*np.sin(eps/2)*axis])
    fdv = np.column_stack([(model.force_body(q, v+eps*d, rpm)-model.force_body(q, v-eps*d, rpm))/(2*eps) for d in np.eye(3)])
    fdt = np.column_stack([(model.force_body(perturb_q(i, 1), v, rpm)-model.force_body(perturb_q(i, -1), v, rpm))/(2*eps) for i in range(3)])
    fdr = np.column_stack([(model.force_body(q, v, rpm+eps*d)-model.force_body(q, v, rpm-eps*d))/(2*eps) for d in np.eye(4)])
    np.testing.assert_allclose(jv, fdv, atol=1e-9)
    np.testing.assert_allclose(jt, fdt, atol=1e-9)
    np.testing.assert_allclose(jr, fdr, atol=1e-9)


@pytest.mark.parametrize('config', [{'mass': 0}, {'kf': -1}, {'gravity': np.nan}, {'linear_drag': [-1, 0, 0]}])
def test_invalid_physical_parameters_are_rejected(config):
    with pytest.raises(ValueError):
        physics.PhysicsModel(config)


def test_hybrid_holds_measured_window_prediction_as_given_input():
    class Predictor:
        sequence_length = 10
        input_hz = 100
        def predict(self, window):
            return window[-1, :3].copy()
    nominal = physics.PhysicsModel({})
    model = hybrid.HybridModel(nominal, Predictor())
    window = np.tile([.1, -.2, .3, 5000, 5000, 5000, 5000], (10, 1))
    model.update_residual(window[:9])
    np.testing.assert_array_equal(model.residual_force, 0)
    model.update_residual(window)
    q, v, rpm = [1, 0, 0, 0], np.zeros(3), np.full(4, 5000)
    np.testing.assert_allclose(model.force_body(q, v, rpm)-nominal.force_body(q, v, rpm), [.1, -.2, .3])
    np.testing.assert_allclose(model.acceleration(q, v, rpm)-nominal.acceleration(q, v, rpm), np.array([.1, -.2, .3])/1.5)
    for actual, expected in zip(model.force_jacobians(q, v, rpm), nominal.force_jacobians(q, v, rpm)):
        np.testing.assert_array_equal(actual, expected)
    exposed = model.residual_force
    exposed[:] = 100
    np.testing.assert_allclose(model.residual_force, [.1, -.2, .3])
