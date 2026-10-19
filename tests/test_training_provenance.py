"""正式对比拒绝未训练模型、数据重用和不兼容气动标签定义。"""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.run_stage3 import validate_trained_checkpoint
from models.training_data import measurement_fingerprint


def inputs():
    times = np.arange(11)/100.0
    return {'imu': {'time': times, 'gyro': np.zeros((11, 3))},
            'rpm': {'time': times, 'rpm': np.full((11, 4), 6000.0)}}


def metadata():
    return {'trained': True, 'data_split': {'train': ['tr'], 'val': ['va'], 'test': ['te']},
            'measurement_sha256': {'train': {'tr': 'a'*64}, 'val': {'va': 'b'*64}},
            'nominal_physics': {'linear_drag': [0.0]*3, 'quadratic_drag': [0.0]*3, 'wind_velocity': [0.0]*3},
            'flight_records': [{'split': 'train', 'seed': 100}, {'split': 'val', 'seed': 200}]}


def test_untrained_checkpoint_is_rejected():
    info = metadata()
    info['trained'] = False
    with pytest.raises(ValueError, match='训练'):
        validate_trained_checkpoint(SimpleNamespace(metadata=info), inputs())


def test_train_or_validation_feature_reuse_is_rejected():
    info = metadata()
    info['measurement_sha256']['val']['va'] = measurement_fingerprint(inputs())
    with pytest.raises(ValueError, match='重合'):
        validate_trained_checkpoint(SimpleNamespace(metadata=info), inputs())


def test_overlapping_flight_ids_are_rejected():
    info = metadata()
    info['data_split']['val'] = ['tr']
    with pytest.raises(ValueError, match='重复|划分'):
        validate_trained_checkpoint(SimpleNamespace(metadata=info), inputs())


def test_changed_nominal_drag_cannot_double_compensate_old_residual():
    info = metadata()
    changed = deepcopy(info['nominal_physics'])
    changed['linear_drag'] = [0.12]*3
    with pytest.raises(ValueError, match='气动'):
        validate_trained_checkpoint(SimpleNamespace(metadata=info), inputs(), model_config=changed)


def test_truncated_training_flight_still_rejects_reused_seed():
    info = metadata()
    # 截断后整段指纹已改变，但正式实验仍不得重用训练飞行的随机种子。
    with pytest.raises(ValueError, match='种子'):
        validate_trained_checkpoint(SimpleNamespace(metadata=info), inputs(), calibration={'simulation': {'seed': 100}})


def test_held_out_seed_and_matching_aerodynamic_definition_are_accepted():
    info = metadata()
    result = validate_trained_checkpoint(SimpleNamespace(metadata=info), inputs(),
                model_config=info['nominal_physics'], calibration={'simulation': {'seed': 42}})
    assert result == measurement_fingerprint(inputs())
