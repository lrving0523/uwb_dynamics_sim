"""阶段一命令行入口：生成并保存六自由度仿真与多频传感器数据。"""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from evaluation.plots import plot_dataset
from evaluation.validation import summarize_dataset
from simulator.config import load_config, validate_config
from simulator.dataset import save_dataset
from simulator.runner import run_simulation


def main():
    parser = argparse.ArgumentParser(description='无人机六自由度与 UWB/IMU/RPM 数据生成')
    parser.add_argument('--config', type=Path, default=Path(__file__).parent/'config'/'sim.yaml', help='统一 YAML 配置')
    parser.add_argument('--output', type=Path, help='覆盖输出目录')
    parser.add_argument('--duration', type=float, help='覆盖仿真时长，单位秒')
    parser.add_argument('--seed', type=int, help='覆盖随机种子')
    parser.add_argument('--trajectory', choices=['hover', 'circle', 'figure_eight', 'variable_speed', 'accel_decel'], help='覆盖参考轨迹')
    parser.add_argument('--disturbance', choices=['none', 'constant', 'step', 'sine', 'random'], help='覆盖外扰类型并启用')
    parser.add_argument('--no-plots', action='store_true', help='仅保存原始数据')
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.output is not None:
            config['output']['directory'] = str(args.output)
        for name in ['duration', 'seed']:
            if getattr(args, name) is not None:
                config['simulation'][name] = getattr(args, name)
        if args.trajectory:
            config['trajectory']['type'] = args.trajectory
        if args.disturbance:
            config['wind'].update(enabled=args.disturbance != 'none', type=args.disturbance)
        validate_config(config)
        start = time.perf_counter()
        dataset = run_simulation(config)
        simulation_seconds = time.perf_counter()-start
        directory = save_dataset(dataset, config, config['output']['directory'])
        validation = summarize_dataset(dataset, config)
        (directory/'physical_validation.json').write_text(
            json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
        figures = [] if args.no_plots else plot_dataset(dataset, directory/'figures', formats=config['output']['formats'])
        truth = dataset['truth']
        error = np.linalg.norm(truth['position']-truth['reference_position'], axis=1)
        # 非均匀真值时间轴采用时间积分权重，不按事件数人为改变误差。
        tracking_rmse = float(np.sqrt(np.trapz(error**2, truth['time'])/config['simulation']['duration']))
        summary = {'说明': '参考跟踪误差，不是定位估计误差；本轮未运行滤波器。',
                   '轨迹': config['trajectory']['type'], '随机种子': config['simulation']['seed'],
                   '时长_s': config['simulation']['duration'], '跟踪位置RMSE_m': tracking_rmse,
                   '最大跟踪误差_m': float(np.max(error)),
                   '四元数最大范数偏差': float(np.max(np.abs(np.linalg.norm(truth['quaternion'], axis=1)-1))),
                   '仿真耗时_s': simulation_seconds,
                   '样本数': {name: len(group['time']) for name, group in dataset.items()},
                   '图文件数': len(figures), '输出目录': str(directory.resolve())}
        (directory/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    except (ValueError, OSError, FloatingPointError) as exc:
        parser.exit(2, f'仿真失败：{exc}\n')


if __name__ == '__main__':
    main()
