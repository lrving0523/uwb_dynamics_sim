"""验证传感器坐标约定、噪声统计和严格采样接口。"""

import numpy as np
import pytest

from sensors import imu_sensor, rpm_sensor, uwb_sensor


IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
GRAVITY = np.array([0.0, 0.0, -9.81])


def imu_config(**overrides):
    config = {
        'acc_std': 0.0,
        'gyro_std': 0.0,
        'acc_bias': [0.0, 0.0, 0.0],
        'gyro_bias': [0.0, 0.0, 0.0],
        'acc_bias_rw_std': 0.0,
        'gyro_bias_rw_std': 0.0,
    }
    config.update(overrides)
    return config


def uwb_config(**overrides):
    config = {
        'anchors': [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        'lever_arm': [0.0, 0.0, 0.0],
        'std': 0.0,
    }
    config.update(overrides)
    return config


@pytest.mark.parametrize('acceleration,expected', [
    ([0.0, 0.0, 0.0], [0.0, 0.0, 9.81]),
    ([0.0, 0.0, -9.81], [0.0, 0.0, 0.0]),
    ([1.0, 2.0, 3.0], [1.0, 2.0, 12.81]),
])
def test_imu_measures_specific_force(acceleration, expected):
    sensor = imu_sensor.IMUSensor(imu_config(), np.random.default_rng(2))
    measurement = sensor.sample(0, IDENTITY, acceleration, [0.2, -0.3, 0.4], GRAVITY)
    assert set(measurement) == {'time', 'accel', 'gyro'}
    assert isinstance(measurement['time'], float)
    np.testing.assert_allclose(measurement['accel'], expected, atol=1e-14)
    np.testing.assert_array_equal(measurement['gyro'], [0.2, -0.3, 0.4])


def test_imu_rotates_world_specific_force_into_body_frame():
    sensor = imu_sensor.IMUSensor(imu_config(), np.random.default_rng(3))
    quaternion = [np.sqrt(0.5), 0.0, np.sqrt(0.5), 0.0]
    measurement = sensor.sample(0.0, quaternion, [0, 0, 0], [0, 0, 0], GRAVITY)
    np.testing.assert_allclose(measurement['accel'], [-9.81, 0.0, 0.0], atol=1e-13)


def test_imu_first_sample_preserves_bias_and_hides_truth():
    config = imu_config(
        acc_bias=[0.1, -0.2, 0.3], gyro_bias=[0.01, 0.02, -0.03],
        acc_bias_rw_std=1.0, gyro_bias_rw_std=1.0,
    )
    sensor = imu_sensor.IMUSensor(config, np.random.default_rng(4))
    measurement = sensor.sample(0.0, IDENTITY, [0, 0, 0], [0, 0, 0], GRAVITY)
    assert set(measurement) == {'time', 'accel', 'gyro'}
    np.testing.assert_array_equal(sensor.acc_bias, [0.1, -0.2, 0.3])
    np.testing.assert_array_equal(sensor.gyro_bias, [0.01, 0.02, -0.03])
    np.testing.assert_allclose(measurement['accel'], [0.1, -0.2, 10.11])
    np.testing.assert_array_equal(measurement['gyro'], [0.01, 0.02, -0.03])
    config['acc_bias'][0] = 100.0
    assert sensor.acc_bias[0] == 0.1


def test_imu_random_walk_scales_with_actual_elapsed_time():
    config = imu_config(acc_bias_rw_std=0.3, gyro_bias_rw_std=0.02)
    short = imu_sensor.IMUSensor(config, np.random.default_rng(5))
    long = imu_sensor.IMUSensor(config, np.random.default_rng(5))
    for sensor in (short, long):
        sensor.sample(0, IDENTITY, GRAVITY, [0, 0, 0], GRAVITY)
    short.sample(0.25, IDENTITY, GRAVITY, [0, 0, 0], GRAVITY)
    long.sample(1.0, IDENTITY, GRAVITY, [0, 0, 0], GRAVITY)
    np.testing.assert_allclose(long.acc_bias, 2 * short.acc_bias, atol=1e-15)
    np.testing.assert_allclose(long.gyro_bias, 2 * short.gyro_bias, atol=1e-15)


def test_imu_random_walk_has_correct_increment_variance():
    sensor = imu_sensor.IMUSensor(
        imu_config(acc_bias_rw_std=0.3, gyro_bias_rw_std=0.02),
        np.random.default_rng(15),
    )
    sensor.sample(0, IDENTITY, GRAVITY, [0, 0, 0], GRAVITY)
    increments = []
    for index in range(1, 4001):
        previous = np.r_[sensor.acc_bias, sensor.gyro_bias]
        sensor.sample(index * 0.04, IDENTITY, GRAVITY, [0, 0, 0], GRAVITY)
        increments.append(np.r_[sensor.acc_bias, sensor.gyro_bias] - previous)
    increments = np.asarray(increments)
    expected_std = np.array([0.06] * 3 + [0.004] * 3)
    # 均值容差取五倍标准误差，避免把正常的有限样本波动判为模型错误。
    np.testing.assert_array_less(
        np.abs(increments.mean(axis=0)), 5 * expected_std / np.sqrt(len(increments))
    )
    np.testing.assert_allclose(increments.std(axis=0), expected_std, rtol=0.05)


def test_imu_measurement_noise_has_configured_mean_and_spread():
    sensor = imu_sensor.IMUSensor(
        imu_config(acc_std=0.4, gyro_std=0.03), np.random.default_rng(6)
    )
    samples = [sensor.sample(index * 0.01, IDENTITY, GRAVITY, [0, 0, 0], GRAVITY)
               for index in range(6000)]
    accel = np.stack([sample['accel'] for sample in samples])
    gyro = np.stack([sample['gyro'] for sample in samples])
    np.testing.assert_allclose(accel.mean(axis=0), 0.0, atol=0.02)
    np.testing.assert_allclose(gyro.mean(axis=0), 0.0, atol=0.002)
    np.testing.assert_allclose(accel.std(axis=0), 0.4, rtol=0.04)
    np.testing.assert_allclose(gyro.std(axis=0), 0.03, rtol=0.04)


def test_uwb_rotates_lever_arm_and_preserves_anchor_order():
    config = uwb_config(lever_arm=[1, 0, 0])
    sensor = uwb_sensor.UWBSensor(config, np.random.default_rng(7))
    quaternion = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    measurement = sensor.sample(0, [0, 2, 0], quaternion)
    assert set(measurement) == {'time', 'ranges'}
    assert isinstance(measurement['time'], float)
    np.testing.assert_allclose(measurement['ranges'], [3.0, 5.0], atol=1e-14)
    np.testing.assert_array_equal(sensor.anchors, [[0, 0, 0], [4, 0, 0]])
    config['anchors'][0][0] = 100.0
    assert sensor.anchors[0, 0] == 0.0


def test_uwb_range_noise_has_configured_spread():
    sensor = uwb_sensor.UWBSensor(uwb_config(std=0.2), np.random.default_rng(8))
    ranges = np.stack([sensor.sample(index, [0, 3, 0], IDENTITY)['ranges']
                       for index in range(6000)])
    residuals = ranges - np.array([3.0, 5.0])
    np.testing.assert_allclose(residuals.mean(axis=0), 0.0, atol=0.01)
    np.testing.assert_allclose(residuals.std(axis=0), 0.2, rtol=0.04)


def test_rpm_uses_current_motor_rpm_without_clipping():
    exact = rpm_sensor.RPMSensor({'std': 0}, np.random.default_rng(9))
    measurement = exact.sample(0, [10, 20, 30, 40])
    assert set(measurement) == {'time', 'rpm'}
    assert isinstance(measurement['time'], float)
    np.testing.assert_array_equal(measurement['rpm'], [10, 20, 30, 40])
    noisy = rpm_sensor.RPMSensor({'std': 5}, np.random.default_rng(10))
    rpm = np.stack([noisy.sample(index, np.zeros(4))['rpm'] for index in range(6000)])
    assert np.any(rpm < 0)
    np.testing.assert_allclose(rpm.mean(axis=0), 0.0, atol=0.2)
    np.testing.assert_allclose(rpm.std(axis=0), 5.0, rtol=0.04)


@pytest.mark.parametrize('name', ['imu', 'uwb', 'rpm'])
def test_sensor_seed_is_reproducible(name):
    def make_sensor():
        generator = np.random.default_rng(11)
        if name == 'imu':
            sensor = imu_sensor.IMUSensor(imu_config(acc_std=0.1, gyro_std=0.1), generator)
            return sensor, lambda time: sensor.sample(time, IDENTITY, [0, 0, 0], [0, 0, 0], GRAVITY)
        if name == 'uwb':
            sensor = uwb_sensor.UWBSensor(uwb_config(std=0.1), generator)
            return sensor, lambda time: sensor.sample(time, [1, 1, 1], IDENTITY)
        sensor = rpm_sensor.RPMSensor({'std': 5.0}, generator)
        return sensor, lambda time: sensor.sample(time, [100, 200, 300, 400])

    first, second = make_sensor()[1], make_sensor()[1]
    for time in (0.0, 0.02, 0.07):
        left, right = first(time), second(time)
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])


@pytest.mark.parametrize('name', ['imu', 'uwb', 'rpm'])
@pytest.mark.parametrize('bad_time', [-0.1, np.inf, np.nan])
def test_invalid_sensor_time_is_rejected(name, bad_time):
    if name == 'imu':
        sensor = imu_sensor.IMUSensor(imu_config(), np.random.default_rng(12))
        sample = lambda time: sensor.sample(time, IDENTITY, [0, 0, 0], [0, 0, 0], GRAVITY)
    elif name == 'uwb':
        sensor = uwb_sensor.UWBSensor(uwb_config(), np.random.default_rng(12))
        sample = lambda time: sensor.sample(time, [0, 0, 0], IDENTITY)
    else:
        sensor = rpm_sensor.RPMSensor({'std': 0}, np.random.default_rng(12))
        sample = lambda time: sensor.sample(time, [0, 0, 0, 0])
    with pytest.raises(ValueError):
        sample(bad_time)
    sample(0.0)
    with pytest.raises(ValueError):
        sample(0.0)


@pytest.mark.parametrize('key,value', [
    ('acc_std', -1), ('gyro_std', np.inf), ('acc_bias_rw_std', -1),
    ('gyro_bias_rw_std', np.nan), ('acc_bias', [0, 0]),
    ('gyro_bias', [0, 0, np.nan]),
])
def test_invalid_imu_config_is_rejected(key, value):
    with pytest.raises(ValueError):
        imu_sensor.IMUSensor(imu_config(**{key: value}), np.random.default_rng(13))


@pytest.mark.parametrize('config', [
    uwb_config(anchors=[]), uwb_config(anchors=[[0, 0]]),
    uwb_config(anchors=[[0, 0, np.nan]]), uwb_config(lever_arm=[0, 0]),
    uwb_config(std=-1), uwb_config(std=np.inf),
])
def test_invalid_uwb_config_is_rejected(config):
    with pytest.raises(ValueError):
        uwb_sensor.UWBSensor(config, np.random.default_rng(14))


@pytest.mark.parametrize('std', [-1, np.inf, np.nan])
def test_invalid_rpm_noise_is_rejected(std):
    with pytest.raises(ValueError):
        rpm_sensor.RPMSensor({'std': std}, np.random.default_rng(16))


@pytest.mark.parametrize('index,bad', [
    (0, [0, 0, 0, 0]), (0, [1, 0, 0]), (0, [np.nan, 0, 0, 0]),
    (1, [0, 0]), (1, [0, 0, np.inf]), (2, [0, 0]), (3, [0, 0, np.nan]),
])
def test_imu_rejects_invalid_state_without_advancing_clock(index, bad):
    sensor = imu_sensor.IMUSensor(imu_config(), np.random.default_rng(17))
    arguments = [IDENTITY, [0, 0, 0], [0, 0, 0], GRAVITY]
    arguments[index] = bad
    with pytest.raises(ValueError):
        sensor.sample(0.0, *arguments)
    sensor.sample(0.0, IDENTITY, [0, 0, 0], [0, 0, 0], GRAVITY)


@pytest.mark.parametrize('position,quaternion', [
    ([0, 0], IDENTITY), ([0, 0, np.inf], IDENTITY),
    ([0, 0, 0], [0, 0, 0, 0]), ([0, 0, 0], [1, 0, 0]),
])
def test_uwb_rejects_invalid_state(position, quaternion):
    sensor = uwb_sensor.UWBSensor(uwb_config(), np.random.default_rng(18))
    with pytest.raises(ValueError):
        sensor.sample(0.0, position, quaternion)


@pytest.mark.parametrize('rpm', [[1, 2, 3], [1, 2, 3, np.nan], [1, 2, 3, -1]])
def test_rpm_rejects_invalid_true_motor_state(rpm):
    sensor = rpm_sensor.RPMSensor({'std': 0.0}, np.random.default_rng(19))
    with pytest.raises(ValueError):
        sensor.sample(0.0, rpm)
