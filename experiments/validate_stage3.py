"""在 checkpoint 声明的全部独立测试飞行上运行四种定位方法。"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')

from experiments.run_stage3 import run_dataset
from models.residual import ResidualPredictor


def main():
    parser = argparse.ArgumentParser(description='阶段三全部独立测试飞行验证')
    parser.add_argument('--checkpoint', type=Path, default=Path('results/residual_training/residual_tcn.pt'))
    parser.add_argument('--config', type=Path, default=Path('config/hybrid.yaml'))
    parser.add_argument('--estimation', type=Path, default=Path('config/estimation.yaml'))
    parser.add_argument('--output', type=Path, default=Path('results/stage3_validation'))
    args = parser.parse_args()
    predictor = ResidualPredictor.load(args.checkpoint)
    records = [record for record in predictor.metadata['flight_records'] if record['split'] == 'test']
    if set(record['flight_id'] for record in records) != set(predictor.metadata['data_split']['test']):
        raise ValueError('独立测试飞行记录与 checkpoint 划分不一致')
    reports = {}
    for record in records:
        name = record['flight_id']
        report = run_dataset(record['data_directory'], args.estimation, args.config,
                              args.checkpoint, args.output/name, formats=['png'])
        reports[name] = report
        values = {method: report[method]['tag_position_rmse_m'] for method in ['uwb_ls', 'eskf', 'physics', 'hybrid']}
        print(f'{name}：天线位置RMSE_m {json.dumps(values, ensure_ascii=False)}', flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'validation_report.json').write_text(json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
