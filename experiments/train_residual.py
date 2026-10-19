"""生成整飞行划分数据，真实训练 TCN，并只在结束后评估独立测试。"""

import os

# 避免 Anaconda MKL 与 PyTorch 各自加载的 OpenMP 运行时发生冲突。
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import yaml

from models.physics import PhysicsModel
from models.residual import train_residual_model
from models.training_data import flight_config, flight_windows, measurement_fingerprint, validate_split
from simulator.config import load_config
from simulator.dataset import save_dataset
from simulator.math3d import quaternion_to_matrix
from simulator.runner import run_simulation


def _json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def _source_hashes():
    """保存模型、训练入口和全部仿真/传感器源文件的内容散列。"""
    root = Path(__file__).resolve().parents[1]
    paths = [*sorted((root/'simulator').glob('*.py')), *sorted((root/'sensors').glob('*.py')),
             root/'models/physics.py', root/'models/residual.py', root/'models/hybrid.py',
             root/'models/training_data.py', Path(__file__).resolve()]
    return {str(path.relative_to(root)).replace('\\', '/'): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run_training(config, output):
    """完整生成/保存数据后训练；验证选择最佳轮数，测试不参与优化。"""
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    data_config, network_config, training_config = config['data'], config['network'], config['training']
    physics = PhysicsModel(config['physics'])
    base = load_config(data_config.get('simulation_config', 'config/sim.yaml'))
    split_seeds = data_config['split_seeds']
    if set(split_seeds) != {'train', 'val', 'test'}:
        raise ValueError('种子必须按 train/val/test 整次飞行分组')
    flattened_seeds = [seed for name in split_seeds for seed in split_seeds[name]]
    if len(flattened_seeds) != len(set(flattened_seeds)):
        raise ValueError('训练、验证和测试不能共享随机种子')
    (directory/'config.resolved.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    splits, hashes, flight_records = {}, {}, []
    collected = {name: [] for name in ['train', 'val', 'test']}
    test_flights = []
    for split in ['train', 'val', 'test']:
        splits[split], hashes[split] = [], {}
        for index, seed in enumerate(split_seeds[split]):
            settings = flight_config(base, seed, index, data_config.get('duration', 18), data_config.get('dt', .01))
            flight_id = f"{split}_{seed}_{settings['trajectory']['type']}"
            dataset = run_simulation(settings)
            save_dataset(dataset, settings, directory/'flights'/flight_id)
            windows = flight_windows(dataset, physics, network_config.get('sequence_length', 10))
            collected[split].append(windows)
            splits[split].append(flight_id)
            hashes[split][flight_id] = measurement_fingerprint(dataset)
            flight_records.append({'flight_id': flight_id, 'split': split, 'seed': int(seed),
                                   'trajectory': settings['trajectory'], 'window_count': len(windows['time']),
                                   'measurement_sha256': hashes[split][flight_id],
                                   'data_directory': str((directory/'flights'/flight_id).resolve())})
            if split == 'test':
                test_flights.append((flight_id, dataset, windows))
            print(f"已保存 {flight_id}：{len(windows['time'])} 个因果窗口", flush=True)
    validate_split(splits)
    arrays = {split: (np.concatenate([flight['inputs'] for flight in collected[split]]),
                      np.concatenate([flight['targets'] for flight in collected[split]])) for split in collected}
    sources = _source_hashes()
    nominal_physics = {'mass': physics.mass, 'gravity': physics.gravity, 'kf': physics.kf,
                       'linear_drag': physics.linear_drag.tolist(), 'quadratic_drag': physics.quadratic_drag.tolist(),
                       'wind_velocity': physics.wind_velocity.tolist()}
    metadata = {'data_split': splits, 'measurement_sha256': hashes, 'source_sha256': sources,
                'nominal_physics': nominal_physics,
                'config': config, 'flight_records': flight_records,
                'label_definition': '真实机体系气动力减名义气动力；不含世界外力、质量和推力误差',
                'limitations': '0.1秒陀螺/RPM历史通常不能唯一确定平移速度和气动力；仅为给定历史的经验条件预测。'}
    start = perf_counter()
    predictor, history = train_residual_model(*arrays['train'], *arrays['val'], network_config, training_config, metadata)
    elapsed_training = perf_counter()-start
    predictor.save(directory/'residual_tcn.pt')
    predictions = predictor.predict_batch(arrays['test'][0])
    targets = arrays['test'][1].astype(float)
    error = predictions-targets
    zeros = np.zeros_like(targets)
    acceleration_errors = {'physics': [], 'hybrid': []}
    per_flight, time_parts, id_parts = [], [], []
    offset = 0
    for index, (flight_id, dataset, windows) in enumerate(test_flights):
        count = len(windows['time'])
        predicted = predictions[offset:offset+count]
        target = targets[offset:offset+count]
        selected = windows['truth_indices']
        truth = dataset['truth']
        rpm = dataset['rpm']['rpm'][9:]
        nominal_acceleration = np.array([physics.acceleration(truth['quaternion'][i], truth['velocity'][i], r) for i, r in zip(selected, rpm)])
        residual_acceleration = np.array([quaternion_to_matrix(truth['quaternion'][i]) @ force/physics.mass for i, force in zip(selected, predicted)])
        true_acceleration = truth['acceleration_world'][selected]
        for name, result in [('physics', nominal_acceleration), ('hybrid', nominal_acceleration+residual_acceleration)]:
            acceleration_errors[name].append(result-true_acceleration)
        per_flight.append({'flight_id': flight_id, 'count': count,
                           'force_rmse_3d_N': float(np.sqrt(np.mean(np.sum((predicted-target)**2, axis=1)))),
                           'physics_force_rmse_3d_N': float(np.sqrt(np.mean(np.sum(target**2, axis=1))))})
        time_parts.append(windows['time'])
        id_parts.append(np.full(count, index, dtype=np.int64))
        offset += count
    np.savez_compressed(directory/'test_predictions.npz', time=np.concatenate(time_parts),
                        target_force_body=targets, predicted_force_body=predictions,
                        physics_residual_force_body=zeros, flight_id=np.concatenate(id_parts),
                        flight_names=np.asarray(splits['test'], dtype=str))
    timing = predictor.benchmark(arrays['test'][0][0])
    report = {'trained': True, 'seed': int(training_config.get('seed', 42)),
              'epochs': int(training_config.get('epochs', 60)), 'best_epoch': predictor.metadata['best_epoch'],
              'best_validation_force_rmse_3d_N': predictor.metadata['best_validation_force_rmse_3d_N'],
              'test_force_rmse_3d_N': float(np.sqrt(np.mean(np.sum(error**2, axis=1)))),
              'test_force_rmse_axes_N': np.sqrt(np.mean(error**2, axis=0)).tolist(),
              'physics_test_force_rmse_3d_N': float(np.sqrt(np.mean(np.sum(targets**2, axis=1)))),
              'test_acceleration_rmse_3d_m_s2': {name: float(np.sqrt(np.mean(np.sum(np.concatenate(values)**2, axis=1)))) for name, values in acceleration_errors.items()},
              'training_seconds': elapsed_training, 'inference': timing,
              'data_split': splits, 'measurement_sha256': hashes, 'source_sha256': sources,
              'nominal_physics': nominal_physics,
              'window_count': {split: len(values[0]) for split, values in arrays.items()},
              'per_test_flight': per_flight, 'torch_version': str(torch.__version__),
              'input_layout': 'B×T×7；T=10；gyro3+rpm4；100Hz',
              'label_definition': metadata['label_definition'], 'limitations': metadata['limitations'],
              'checkpoint_sha256': hashlib.sha256((directory/'residual_tcn.pt').read_bytes()).hexdigest()}
    _json(directory/'training_history.json', {'epoch': [row['epoch'] for row in history],
                                            'train_loss': [row['train_loss'] for row in history],
                                            'val_loss': [row['val_force_rmse_3d_N']**2 for row in history],
                                            'objective_loss': [row['objective_loss'] for row in history],
                                            'loss_unit': 'N²', 'loss_definition': '三轴力误差平方和的样本均值',
                                            'objective_loss_unit': '训练集逐轴标准化后的均方误差', 'records': history})
    _json(directory/'training_report.json', report)
    _json(directory/'source_manifest.json', {'source_sha256': sources, 'flights': flight_records,
                                           'config_sha256': hashlib.sha256((directory/'config.resolved.yaml').read_bytes()).hexdigest()})
    print(f"训练完成：测试残余三维RMSE={report['test_force_rmse_3d_N']:.6f} N，零残余={report['physics_test_force_rmse_3d_N']:.6f} N", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description='真实训练三维气动残余 TCN 并保存独立测试证据')
    parser.add_argument('--config', default='config/hybrid.yaml')
    parser.add_argument('--output', default='results/residual_training')
    arguments = parser.parse_args()
    with Path(arguments.config).open(encoding='utf-8-sig') as stream:
        config = yaml.safe_load(stream)
    run_training(config, arguments.output)


if __name__ == '__main__':
    main()
