"""保存动力学增强方法的运动状态与两类观测创新。"""

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml


def save_dynamics_results(results, settings, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ['physics', 'hybrid', 'residual', 'timing']:
        target = directory/f'{name}.npz'
        np.savez_compressed(target, **results[name])
        files.append(target)
    for group in ['uwb_diagnostics', 'accel_diagnostics']:
        for name in ['physics', 'hybrid']:
            target = directory/f'{group}_{name}.npz'
            np.savez_compressed(target, **results[group][name])
            files.append(target)
    target = directory/'dynamics_settings.yaml'
    target.write_text(yaml.safe_dump(settings, sort_keys=False, allow_unicode=True), encoding='utf-8')
    files.append(target)
    manifest = {'schema_version': 1, 'description': '阶段三动力学增强；公开误差15维，力误差辅助量不是外扰估计。',
                'frames': '世界ENU、机体FLU、Hamilton wxyz，右局部旋转误差',
                'noise_assumption': '共享当前模型力误差；TCN测量历史条件误差用固定协方差近似，不声称完整历史噪声独立性。',
                'sha256': {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files}}
    (directory/'dynamics_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def load_dynamics_results(directory):
    directory = Path(directory)
    results = {}

    def read(name):
        with np.load(directory/f'{name}.npz', allow_pickle=False) as archive:
            return {key: archive[key].copy() for key in archive.files}

    for name in ['physics', 'hybrid', 'residual', 'timing']:
        results[name] = read(name)
    for group in ['uwb_diagnostics', 'accel_diagnostics']:
        results[group] = {name: read(f'{group}_{name}') for name in ['physics', 'hybrid']}
    return results
