"""在阶段一已保存的悬停、动态及外扰数据上复验同一套估计参数。"""

import argparse
import json
from pathlib import Path

from experiments.run_stage2 import run_dataset


def main():
    parser = argparse.ArgumentParser(description='阶段二多轨迹基线验证')
    parser.add_argument('--datasets-root', type=Path, default=Path('results/stage1_validation'))
    parser.add_argument('--settings', type=Path, default=Path('config/estimation.yaml'))
    parser.add_argument('--output', type=Path, default=Path('results/stage2_validation'))
    args = parser.parse_args()
    cases = ['trajectory_hover', 'trajectory_circle', 'trajectory_figure_eight',
             'trajectory_variable_speed', 'disturbance_step']
    reports = {}
    for name in cases:
        reports[name] = run_dataset(args.datasets_root/name, args.settings, args.output/name, formats=['png'])
        ls = reports[name]['uwb_ls']['tag_position_rmse_m']
        eskf = reports[name]['eskf']['tag_position_rmse_m']
        print(f'{name}：天线位置 RMSE，UWB LS={ls:.6f} m，ESKF={eskf:.6f} m', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'validation_report.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
