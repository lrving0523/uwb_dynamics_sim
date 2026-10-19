"""批量运行阶段一轨迹与外扰验证，并保留每个实验的数据和图。"""

import argparse
from copy import deepcopy
import json
from pathlib import Path

from evaluation.plots import plot_dataset
from evaluation.validation import summarize_dataset
from simulator.config import load_config
from simulator.dataset import save_dataset
from simulator.runner import run_simulation


def main():
    parser = argparse.ArgumentParser(description='阶段一轨迹与外扰验证')
    parser.add_argument('--config', type=Path, default=Path('config/sim.yaml'))
    parser.add_argument('--output', type=Path, default=Path('results/stage1_validation'))
    parser.add_argument('--duration', type=float, default=8.0)
    args = parser.parse_args()
    base = load_config(args.config)
    base['simulation']['duration'] = args.duration
    cases = []
    for mode in ['hover', 'circle', 'figure_eight', 'variable_speed', 'accel_decel']:
        config = deepcopy(base)
        config['trajectory']['type'] = mode
        config['wind']['enabled'] = False
        cases.append((f'trajectory_{mode}', config))
    for mode in ['constant', 'step', 'sine', 'random']:
        config = deepcopy(base)
        config['trajectory']['type'] = 'hover'
        config['wind'].update(enabled=True, type=mode, start_time=min(2.0, args.duration/3))
        cases.append((f'disturbance_{mode}', config))
    reports = {}
    for name, config in cases:
        directory = args.output/name
        config['output']['directory'] = str(directory)
        config['output']['formats'] = ['png']
        dataset = run_simulation(config)
        save_dataset(dataset, config, directory)
        plot_dataset(dataset, directory/'figures', formats=['png'])
        reports[name] = summarize_dataset(dataset, config)
        print(f"{name}：跟踪 RMSE {reports[name]['tracking_position_rmse_m']:.6f} m", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'validation_report.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding='utf-8')
    if any(report['force_balance_max_error_N'] > 1e-10 or report['quaternion_norm_max_error'] > 1e-12
           for report in reports.values()):
        raise RuntimeError('物理一致性检查未通过，请查看报告')


if __name__ == '__main__':
    main()
