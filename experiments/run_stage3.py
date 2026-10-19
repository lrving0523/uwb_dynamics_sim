"""用实际训练的残余模型运行同数据四种定位基线并离线评价。"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

# 本机 Conda MKL 与 Torch 使用不同 OpenMP DLL；在数值库导入前选择顺序 MKL。
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

import numpy as np
import yaml

from estimator.dynamics_pipeline import load_dynamics_settings, run_dynamics_estimators
from estimator.dynamics_io import save_dynamics_results
from estimator.pipeline import load_estimation_settings, run_estimators
from estimator.result_io import save_estimates
from evaluation.dynamics_plots import plot_dynamics_comparison
from evaluation.metrics import evaluate_estimates
from evaluation.model_prediction import evaluate_model_predictions
from models.physics import PhysicsModel
from models.hybrid import HybridModel
from models.residual import ResidualPredictor
from models.training_data import measurement_fingerprint, validate_split
from simulator.config import load_config
from simulator.dataset import load_dataset


def load_observations(directory):
    """不加载真值即可获得所有在线模型的输入。"""
    observations = {}
    for name in ['imu', 'uwb', 'rpm']:
        with np.load(Path(directory)/f'{name}.npz', allow_pickle=False) as archive:
            observations[name] = {key: archive[key].copy() for key in archive.files}
    return observations


def validate_trained_checkpoint(predictor, measurements, model_config=None, calibration=None):
    """正式学习实验拒绝未训练网络与训练/验证输入重用。"""
    metadata = predictor.metadata
    if metadata.get('trained') is not True:
        raise ValueError('正式 Hybrid 对比必须使用实际训练完成的 checkpoint')
    if not metadata.get('data_split', {}).get('train') or not metadata.get('data_split', {}).get('val'):
        raise ValueError('checkpoint 缺少独立训练/验证飞行划分证据')
    validate_split(metadata['data_split'])
    if model_config is not None:
        nominal = metadata.get('nominal_physics', metadata.get('config', {}).get('physics'))
        if not isinstance(nominal, dict):
            raise ValueError('checkpoint 缺少名义气动标签定义，不能确认补偿兼容性')
        for key in ['linear_drag', 'quadratic_drag', 'wind_velocity']:
            if key not in nominal or key not in model_config or not np.array_equal(nominal[key], model_config[key]):
                raise ValueError('当前名义气动模型与残余训练标签定义不同，需要按该气动项重新训练')
    if calibration is not None:
        records = metadata.get('flight_records', [])
        if not records:
            raise ValueError('checkpoint 缺少飞行种子记录，不能核验整次飞行划分')
        used_seeds = {item['seed'] for item in records if item['split'] in ['train', 'val']}
        if calibration['simulation']['seed'] in used_seeds:
            raise ValueError('正式独立实验不能重用训练/验证飞行种子，即使只截取其中一段')
    fingerprint = measurement_fingerprint(measurements)
    sources = metadata.get('measurement_sha256', {})
    for split in ['train', 'val']:
        if not sources.get(split):
            raise ValueError('checkpoint 缺少训练/验证输入指纹，不能核验数据独立性')
        if set(sources[split]) != set(metadata['data_split'][split]):
            raise ValueError('checkpoint 飞行划分与输入指纹标识不一致')
        if fingerprint in sources[split].values():
            raise ValueError(f'当前输入与 {split} 飞行重合，不能作为正式独立对比')
    return fingerprint


def _method_report(dataset, baseline, dynamics, name):
    wrapped = {'uwb_ls': baseline['uwb_ls'], 'eskf': dynamics[name],
               'diagnostics': dynamics['uwb_diagnostics'][name],
               'timing': {'ls_update_seconds': baseline['timing']['ls_update_seconds'],
                           'eskf_event_seconds': dynamics['timing'][f'{name}_event_seconds'],
                           'initialization_seconds': np.asarray(0.0)}}
    report = evaluate_estimates(dataset, wrapped)
    report['eskf']['state_time_axis'] = '该动力学滤波器全部输出事件时间戳；等样本权重'
    return report


def run_dataset(dataset_directory, estimation_path, hybrid_config_path, checkpoint, output_directory,
                formats=('png', 'pdf', 'svg'), plots=True):
    """先完成估计并保存，再读取真值，所有正式残余预测均来自 checkpoint。"""
    source, output = Path(dataset_directory), Path(output_directory)
    calibration = load_config(source/'config.resolved.yaml')
    baseline_settings = load_estimation_settings(estimation_path, calibration)
    settings, model_config = load_dynamics_settings(estimation_path, hybrid_config_path, calibration)
    observations = load_observations(source)
    predictor = ResidualPredictor.load(checkpoint)
    fingerprint = validate_trained_checkpoint(predictor, observations, model_config, calibration)
    physics = PhysicsModel(model_config)
    hybrid = HybridModel(PhysicsModel(model_config), predictor)
    started = time.perf_counter()
    baseline = run_estimators({key: observations[key] for key in ['imu', 'uwb']}, baseline_settings)
    baseline_seconds = time.perf_counter()-started
    started = time.perf_counter()
    dynamics = run_dynamics_estimators(observations, settings, physics, hybrid)
    dynamics_seconds = time.perf_counter()-started
    save_estimates(baseline, baseline_settings, output/'baseline')
    save_dynamics_results(dynamics, settings, output)
    (output/'nominal_physics.yaml').write_text(yaml.safe_dump(model_config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    # 在线路径在此结束，真值仅用于下面的独立评价和可视化。
    dataset = load_dataset(source)
    baseline_report = evaluate_estimates(dataset, baseline)
    physics_report = _method_report(dataset, baseline, dynamics, 'physics')
    hybrid_report = _method_report(dataset, baseline, dynamics, 'hybrid')
    report = {
        'comparison': baseline_report['comparison'], 'common_samples': baseline_report['common_samples'],
        'uwb_ls': baseline_report['uwb_ls'], 'eskf': baseline_report['eskf'],
        'physics': physics_report['eskf'], 'hybrid': hybrid_report['eskf'],
        'diagnostics': {'eskf': baseline_report['diagnostics'], 'physics': physics_report['diagnostics'],
                        'hybrid': hybrid_report['diagnostics']},
        'model_prediction': evaluate_model_predictions(dataset, PhysicsModel(model_config), dynamics['residual'], horizon_s=0.2),
        '运行信息': {'输入目录': str(source.resolve()), '输出目录': str(output.resolve()),
                      '随机种子': calibration['simulation']['seed'], '当前输入指纹': fingerprint,
                      'checkpoint': str(Path(checkpoint).resolve()),
                      'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                      '基线估计耗时_s': baseline_seconds, '两种动力学方法及推理总耗时_s': dynamics_seconds,
                      '正式残余模型': '实际训练TCN；没有使用解析理想残余或真实外力',
                      '固定模型力标准差_N': settings['noise']['model_force_std'],
                      '近似说明': '当前力误差在预测/比力更新中共享；TCN全部历史输入噪声相关性采用固定条件误差近似。',
                      '诊断说明': 'NIS/NEES仅单次描述，不作蒙特卡洛一致性结论。'},
    }
    figures = plot_dynamics_comparison(dataset, baseline, dynamics, output/'figures', formats=formats) if plots else []
    report['运行信息']['图文件数'] = len(figures)
    (output/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    manifest_path = output/'dynamics_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest.update(checkpoint_sha256=report['运行信息']['checkpoint_sha256'], measurement_fingerprint=fingerprint,
                     training_metadata=predictor.metadata, nominal_physics=model_config,
                     source_directory=str(source.resolve()))
    manifest['source_sha256'] = {name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                                 for name in ['imu.npz', 'uwb.npz', 'rpm.npz', 'config.resolved.yaml']}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description='阶段三物理/训练混合动力学与基础定位比较')
    parser.add_argument('--dataset', type=Path, default=Path('results/phase1'))
    parser.add_argument('--estimation', type=Path, default=Path('config/estimation.yaml'))
    parser.add_argument('--config', type=Path, default=Path('config/hybrid.yaml'))
    parser.add_argument('--checkpoint', type=Path, default=Path('results/residual_training/residual_tcn.pt'))
    parser.add_argument('--output', type=Path, default=Path('results/phase3'))
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], default=['png', 'pdf', 'svg'])
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    try:
        report = run_dataset(args.dataset, args.estimation, args.config, args.checkpoint, args.output, args.formats, not args.no_plots)
        compact = {name: report[name] for name in ['common_samples', 'uwb_ls', 'eskf', 'physics', 'hybrid', 'model_prediction', '运行信息']}
        print(json.dumps(compact, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, FloatingPointError, np.linalg.LinAlgError) as error:
        parser.exit(2, f'阶段三运行失败：{error}\n')


if __name__ == '__main__':
    main()
