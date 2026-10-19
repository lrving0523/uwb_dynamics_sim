"""读取已有训练结果并重绘；运行：python -m experiments.plot_training。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from evaluation.dynamics_plots import plot_model_training


def main(argv=None):
    """只读取训练记录与独立测试数组，不启动训练或加载模型权重。"""
    parser = argparse.ArgumentParser(description='使用已有训练记录和独立测试预测重新导出训练图。')
    parser.add_argument('--training', type=Path, default=Path('results/residual_training'),
                        help='训练结果目录，包含 training_history.json 与 test_predictions.npz')
    parser.add_argument('--output', type=Path,
                        help='图片输出目录；默认使用训练结果目录下的 figures')
    parser.add_argument('--formats', nargs='+', choices=('png', 'pdf', 'svg'),
                        default=['png', 'pdf', 'svg'], help='导出格式，默认同时生成 png pdf svg')
    args = parser.parse_args(argv)
    history = json.loads((args.training / 'training_history.json').read_text(encoding='utf-8'))
    with np.load(args.training / 'test_predictions.npz', allow_pickle=False) as archive:
        predictions = {key: archive[key].copy() for key in archive.files}
    output = args.output if args.output is not None else args.training / 'figures'
    paths = plot_model_training(history, predictions, output, formats=args.formats)
    print(f'已重导出 {len(paths)} 个训练图文件：')
    for path in paths:
        print(path.resolve())


if __name__ == '__main__':
    main()
