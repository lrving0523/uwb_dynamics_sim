"""用可解析运动验证刚体、执行器、控制器与扰动约定。"""

from importlib import import_module

import numpy as np
import pytest


DRONE = {'mass': 1.5, 'gravity': 9.81, 'inertia': [0.03, 0.03, 0.05],
         'initial_position': [0.0, 0.0, 1.5]}
MOTOR = {'arm_length': 0.22, 'kf': 1e-5, 'km': 2e-7,
         'time_constant': 0.03, 'min_rpm': 0.0, 'max_rpm': 12000.0}


def api(module, name):
    """将空模块显式报告为缺失功能，便于先观察测试失败。"""
    result = getattr(import_module('simulator.' + module), name, None)
    assert result is not None, f'尚未实现 {module}.{name}'
    return result


def test_motor_rpm_conversion_and_axis_order():
    motor = api('motor_model', 'MotorModel')(MOTOR)
    force, torque = motor.wrench([60.0 / (2 * np.pi), 0, 0, 0])
    np.testing.assert_allclose(force, [0, 0, 1e-5], atol=1e-15)
    np.testing.assert_allclose(torque, [0, -2.2e-6, 2e-7], atol=1e-15)


def test_motor_allocation_round_trip_and_saturation():
    motor = api('motor_model', 'MotorModel')(MOTOR)
    force, torque = motor.wrench([4500, 4700, 4300, 4600])
    np.testing.assert_allclose(motor.allocate(force[2], torque), [4500, 4700, 4300, 4600])
    rpm = motor.allocate(1e8, [1e5, -1e5, 1e5])
    assert np.all((rpm >= 0) & (rpm <= MOTOR['max_rpm']))


def test_motor_lag_is_analytic_and_stateless():
    motor = api('motor_model', 'MotorModel')(MOTOR)
    result = motor.response(np.zeros(4), np.full(4, 6000), 0.03)
    np.testing.assert_allclose(result, 6000 * (1 - np.exp(-1)))
    np.testing.assert_array_equal(motor.response(np.zeros(4), np.full(4, 6000), 0), np.zeros(4))
    np.testing.assert_array_equal(result, motor.response(np.zeros(4), np.full(4, 6000), 0.03))


def test_quaternion_hamilton_world_rotation_and_round_trip():
    maths = import_module('simulator.math3d')
    quaternion_to_matrix = api('math3d', 'quaternion_to_matrix')
    q = np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)])
    rotation = quaternion_to_matrix(q)
    np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-14)
    recovered = maths.matrix_to_quaternion(rotation)
    assert abs(np.dot(q, recovered)) == pytest.approx(1.0)
    np.testing.assert_allclose(maths.quaternion_multiply(q, [q[0], 0, 0, -q[3]]), [1, 0, 0, 0], atol=1e-14)


def test_hover_equilibrium_and_initial_state_defaults():
    drone = api('drone', 'Drone')(DRONE, MOTOR, {})
    initial_rpm = drone.rpm.copy()
    for index in range(200):
        drone.step(initial_rpm, 0.01, lambda t: np.zeros(3), index * 0.01)
    np.testing.assert_allclose(drone.state.position, [0, 0, 1.5], atol=1e-12)
    np.testing.assert_allclose(drone.state.velocity, [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(drone.quantities(lambda t: np.zeros(3))['acceleration_world'], [0, 0, 0], atol=1e-12)


def test_freefall_matches_analytic_position_and_velocity():
    drone = api('drone', 'Drone')({**DRONE, 'initial_position': [0, 0, 10]}, {**MOTOR, 'initial_rpm': [0, 0, 0, 0]}, {})
    for index in range(100):
        drone.step(np.zeros(4), 0.01, lambda t: np.zeros(3), index * 0.01)
    np.testing.assert_allclose(drone.state.position, [0, 0, 5.095], atol=1e-11)
    np.testing.assert_allclose(drone.state.velocity, [0, 0, -9.81], atol=1e-11)


def test_constant_body_rotation_preserves_unit_quaternion():
    state = api('drone', 'DroneState')(np.zeros(3), np.zeros(3), np.array([1, 0, 0, 0]), np.array([0, 0, 2.0]))
    drone = api('drone', 'Drone')(DRONE, {**MOTOR, 'initial_rpm': [0, 0, 0, 0]}, {}, state=state)
    for index in range(100):
        drone.step(np.zeros(4), 0.01, lambda t: np.zeros(3), index * 0.01)
    np.testing.assert_allclose(drone.state.quaternion, [np.cos(1), 0, 0, np.sin(1)], atol=1e-9)
    assert np.linalg.norm(drone.state.quaternion) == pytest.approx(1.0, abs=1e-14)


def test_air_drag_uses_relative_body_velocity_and_force_stays_separate():
    state = api('drone', 'DroneState')(np.zeros(3), np.array([2, 0, 0]), np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)]), np.zeros(3))
    drone = api('drone', 'Drone')(DRONE, MOTOR, {'linear_drag': [1, 2, 3], 'quadratic_drag': [0, 0, 0], 'wind_velocity': [1, 0, 0]}, state=state)
    quantities = drone.quantities(lambda t: np.array([0, 3, 0]))
    np.testing.assert_allclose(quantities['aero_force_body'], [0, 2, 0], atol=1e-14)
    np.testing.assert_allclose(quantities['external_force_world'], [0, 3, 0])
    np.testing.assert_allclose(quantities['acceleration_world'], [-4 / 3, 2, 0], atol=1e-13)


def test_external_step_uses_left_limit_for_integration_endpoint():
    drone = api('drone', 'Drone')(DRONE, MOTOR, {})
    drone.step(drone.rpm, 0.01, lambda t: np.array([6.0 if t >= 0.01 else 0, 0, 0]), 0.0)
    assert drone.state.velocity[0] == pytest.approx(0.0, abs=1e-15)
    assert drone.quantities(lambda t: np.array([6.0 if t >= 0.01 else 0, 0, 0]), 0.01)['external_force_world'][0] == 6.0


@pytest.mark.parametrize('kind', ['hover', 'circle', 'figure_eight', 'variable_speed', 'accel_decel'])
def test_trajectories_start_at_initial_position_and_have_consistent_derivatives(kind):
    trajectory = api('trajectory', 'Trajectory')({'type': kind, 'radius': 1.2, 'period': 8.0, 'yaw': 0.25}, np.array([2, -1, 3]))
    np.testing.assert_allclose(trajectory.sample(0)['position'], [2, -1, 3], atol=1e-14)
    time, epsilon = 1.3, 1e-4
    left, center, right = trajectory.sample(time - epsilon), trajectory.sample(time), trajectory.sample(time + epsilon)
    np.testing.assert_allclose((right['position'] - left['position']) / (2 * epsilon), center['velocity'], atol=1e-6)
    np.testing.assert_allclose((right['velocity'] - left['velocity']) / (2 * epsilon), center['acceleration'], atol=1e-6)
    assert center['yaw'] == 0.25


@pytest.mark.parametrize('kind', ['none', 'constant', 'step', 'sine', 'random'])
def test_wind_modes_are_finite_repeatable_and_query_order_independent(kind):
    config = {'enabled': True, 'type': kind, 'force': [0.6, -0.3, 0.15], 'start_time': 0.5,
              'frequency': 0.35, 'random_std': [0.3, 0.3, 0.15], 'random_hz': 10.0}
    wind_type = api('wind_field', 'WindField')
    wind = wind_type(config, np.random.default_rng(12), 3.0)
    expected = wind.force(1.17)
    wind.force(2.8)
    np.testing.assert_array_equal(wind.force(1.17), expected)
    np.testing.assert_array_equal(wind_type(config, np.random.default_rng(12), 3.0).force(1.17), expected)
    assert np.all(np.isfinite(expected))
    np.testing.assert_array_equal(wind_type({**config, 'enabled': False}, np.random.default_rng(12), 3.0).force(1), np.zeros(3))
    if kind in {'none', 'step', 'sine'}:
        np.testing.assert_array_equal(wind.force(0.0), np.zeros(3))
    if kind in {'constant', 'step'}:
        np.testing.assert_allclose(wind.force(1.0), [0.6, -0.3, 0.15])


def test_geometric_controller_recovers_position_under_constant_force():
    drone = api('drone', 'Drone')(DRONE, MOTOR, {'linear_drag': [0.12, 0.12, 0.16]})
    controller = api('controller', 'PositionController')(DRONE, MOTOR, {})
    reference = {'position': np.array([1.0, -0.5, 2.0]), 'velocity': np.zeros(3), 'acceleration': np.zeros(3), 'yaw': 0.0}
    for index in range(1000):
        target = controller.command(drone.state, reference, 0.01)
        drone.step(target, 0.01, lambda t: np.array([0.3, -0.1, 0]), index * 0.01)
    assert np.linalg.norm(drone.state.position - reference['position']) < 0.09
    assert np.linalg.norm(drone.state.velocity) < 0.03


def test_controller_initial_zero_interval_does_not_accumulate_integral():
    drone = api('drone', 'Drone')(DRONE, MOTOR, {})
    controller = api('controller', 'PositionController')(DRONE, MOTOR, {})
    reference = {'position': np.array([1.0, 0.0, 1.5]), 'velocity': np.zeros(3),
                 'acceleration': np.zeros(3), 'yaw': 0.0}
    rpm = controller.command(drone.state, reference, 0.0)
    assert np.all(np.isfinite(rpm))
    np.testing.assert_array_equal(controller.integral, np.zeros(3))
    with pytest.raises(ValueError):
        controller.command(drone.state, reference, -0.01)


def test_rk4_couples_motor_lag_at_substages():
    drone = api('drone', 'Drone')(DRONE, {**MOTOR, 'initial_rpm': [0, 0, 0, 0]}, {})
    target = np.full(4, 6000.0)
    for index in range(100):
        drone.step(target, 0.001, lambda t: np.zeros(3), index * 0.001)
    duration, tau = 0.1, 0.03
    integrated_response_squared = duration - 2 * tau * (1 - np.exp(-duration / tau)) + tau / 2 * (1 - np.exp(-2 * duration / tau))
    expected_velocity = 4e-5 * (200 * np.pi) ** 2 / 1.5 * integrated_response_squared - 9.81 * duration
    assert drone.state.velocity[2] == pytest.approx(expected_velocity, abs=1e-8)


@pytest.mark.parametrize('module,name,args', [
    ('motor_model', 'MotorModel', ({**MOTOR, 'time_constant': 0},)),
    ('drone', 'Drone', ({**DRONE, 'mass': -1}, MOTOR, {})),
    ('trajectory', 'Trajectory', ({'type': 'invalid'}, np.zeros(3))),
])
def test_invalid_local_inputs_raise(module, name, args):
    with pytest.raises(ValueError):
        api(module, name)(*args)
