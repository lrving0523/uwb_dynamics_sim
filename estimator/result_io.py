"""独立保存估计结果、固定参数及其校验清单。"""

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import yaml


GROUPS = ('uwb_ls', 'eskf', 'diagnostics', 'timing')


def save_estimates(estimates, settings, directory):
    """估计和运行耗时分包；耗时不属于逐数组复现的科学量。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for group in GROUPS:
        target = directory/f'{group}.npz'
        np.savez_compressed(target, **estimates[group])
        files.append(target)
    settings_path = directory/'estimation.resolved.yaml'
    settings_path.write_text(yaml.safe_dump(settings, allow_unicode=True, sort_keys=False), encoding='utf-8')
    files.append(settings_path)
    manifest = {'schema_version': 1, 'description': 'UWB 单点定位与15维ESKF；误差顺序p,v,theta,ba,bg。',
                'position_reference': 'uwb_ls.tag_position 为天线位置；eskf.position 为机体IMU中心，tag_position 为估计天线位置。',
                'units': {'time': 's', 'position': 'm', 'velocity': 'm/s',
                           'quaternion': 'Hamilton wxyz，机体到世界', 'attitude_error': '右局部旋转，rad',
                           'acc_bias': 'm/s²', 'gyro_bias': 'rad/s', 'timing': 's'},
                'timing_note': '初始化耗时包含首帧LS；事件耗时不包含独立LS基线求解。耗时不保证逐数组复现。',
                'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pyyaml': yaml.__version__},
                'fields': {group: {name: list(np.shape(value)) for name, value in estimates[group].items()}
                            for group in GROUPS},
                'sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files}}
    (directory/'estimation_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


def load_estimates(directory):
    """加载纯数值数组，禁止 pickle。"""
    result = {}
    for group in GROUPS:
        with np.load(Path(directory)/f'{group}.npz', allow_pickle=False) as archive:
            result[group] = {name: archive[name].copy() for name in archive.files}
    return result
