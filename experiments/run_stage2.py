"""从保存的原始观测运行阶段二基线，完成后再读取真值评价。"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from estimator.pipeline import load_estimation_settings, run_estimators
from estimator.result_io import save_estimates
from evaluation.estimation_plots import plot_estimation
from evaluation.metrics import evaluate_estimates
from simulator.config import load_config
from simulator.dataset import load_dataset


def load_measurements(directory):
    """在线路径仅从两个观测文件读取数值数组。"""
    observations = {}
    for name in ['imu', 'uwb']:
        with np.load(Path(directory)/f'{name}.npz', allow_pickle=False) as archive:
            observations[name] = {key: archive[key].copy() for key in archive.files}
    return observations


def run_dataset(dataset_directory, settings_path, output_directory, formats=('png', 'pdf', 'svg'), plots=True):
    """先冻结估计结果，再在独立离线步骤中读取真值。"""
    dataset_directory, output_directory = Path(dataset_directory), Path(output_directory)
    calibration = load_config(dataset_directory/'config.resolved.yaml')
    settings = load_estimation_settings(settings_path, calibration)
    observations = load_measurements(dataset_directory)
    started = time.perf_counter()
    estimates = run_estimators(observations, settings)
    filtering_seconds = time.perf_counter()-started
    save_estimates(estimates, settings, output_directory)
    manifest_path = output_directory/'estimation_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    sources = ['imu.npz', 'uwb.npz', 'config.resolved.yaml']
    manifest['source_observations'] = {name: hashlib.sha256((dataset_directory/name).read_bytes()).hexdigest()
                                        for name in sources}
    manifest['source_directory'] = str(dataset_directory.resolve())
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    # 真值在上面两种估计均完成后读取，只参与指标和图形。
    dataset = load_dataset(dataset_directory)
    report = evaluate_estimates(dataset, estimates)
    report['运行信息'] = {'输入数据目录': str(dataset_directory.resolve()),
                         '输出目录': str(output_directory.resolve()),
                         '随机种子': calibration['simulation']['seed'],
                         '估计总耗时_s': filtering_seconds,
                         '初始化先验': settings['initial'],
                         '首帧处理': '首帧UWB仅做LS初始化，后续原始距离才做测量更新。',
                         '一致性声明': '单次NEES/NIS仅作描述，没有进行蒙特卡洛一致性检验。'}
    figures = plot_estimation(dataset, estimates, output_directory/'figures', formats=formats) if plots else []
    report['运行信息']['图文件数'] = len(figures)
    (output_directory/'metrics.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description='阶段二 UWB 单点定位与15维 UWB–IMU ESKF')
    parser.add_argument('--dataset', type=Path, default=Path('results/phase1'), help='阶段一保存的数据目录')
    parser.add_argument('--settings', type=Path, default=Path('config/estimation.yaml'), help='独立估计先验')
    parser.add_argument('--output', type=Path, default=Path('results/phase2'), help='估计结果与图目录')
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], default=['png', 'pdf', 'svg'])
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    try:
        report = run_dataset(args.dataset, args.settings, args.output, args.formats, not args.no_plots)
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, OSError, FloatingPointError, np.linalg.LinAlgError) as exc:
        parser.exit(2, f'阶段二运行失败：{exc}\n')


if __name__ == '__main__':
    main()
