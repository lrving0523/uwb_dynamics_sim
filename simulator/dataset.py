"""独立保存真值与测量，记录配置、单位和文件校验和。"""

import csv
import hashlib
import json
from pathlib import Path
import platform

import matplotlib
import numpy as np
import yaml


GROUPS = ('truth', 'imu', 'uwb', 'rpm')


def _write_csv(data, path):
    fields = {name: np.asarray(values) for name, values in data.items()
              if name not in ('anchors', 'lever_arm')}
    size = len(fields['time'])
    flat = {name: values.reshape(size, -1) for name, values in fields.items()}
    headers = [name if values.shape[1] == 1 else f'{name}_{i}'
               for name, values in flat.items() for i in range(values.shape[1])]
    matrix = np.concatenate(list(flat.values()), axis=1)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(matrix)


def save_dataset(dataset, config, output_dir):
    """NPZ 保存全部精度；CSV 便于检查，元数据不混入测量矩阵。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    for name in GROUPS:
        np.savez_compressed(directory / f'{name}.npz', **dataset[name])
        if config['output']['csv']:
            _write_csv(dataset[name], directory / f'{name}.csv')
    config_path = directory / 'config.resolved.yaml'
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')
    files = [directory / f'{name}.npz' for name in GROUPS] + [config_path]
    if config['output']['csv']:
        files += [directory / f'{name}.csv' for name in GROUPS]
    manifest = {
        'schema_version': 1, 'seed': config['simulation']['seed'],
        'description': '阶段一仿真；truth 为离线真值，其余文件为独立时间轴测量。',
        'frames': {'world': 'ENU，z 向上', 'body': 'FLU，推力沿 +z',
                   'quaternion': 'Hamilton wxyz，机体到世界'},
        'units': {'time': 's', 'position': 'm', 'velocity': 'm/s',
                  'acceleration_world': 'm/s²', 'accel': 'm/s²（机体比力）',
                  'gyro': 'rad/s', 'omega': 'rad/s', 'ranges': 'm',
                  'rpm': 'rev/min', 'target_rpm': 'rev/min', 'true_rpm': 'rev/min',
                  'force': 'N', 'torque': 'N·m', 'quaternion': '无量纲'},
        'noise_convention': '白噪声 std 为每次采样标准差；偏置随机游走 std 按 sqrt(秒) 缩放。',
        'sampling': 't_k=k/f，包含 t=0，不制造末尾观测；truth 记录积分网格、所有事件及终止状态，故可为非均匀时间轴。',
        'counts': {name: len(dataset[name]['time']) for name in GROUPS},
        'fields': {name: {key: list(np.shape(value)) for key, value in dataset[name].items()}
                   for name in GROUPS},
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pyyaml': yaml.__version__, 'matplotlib': matplotlib.__version__},
        'sha256': {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in files},
    }
    (directory / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return directory


def load_dataset(directory):
    """不反序列化 pickle；返回与在线生成器相同的数组字典。"""
    result = {}
    for name in GROUPS:
        with np.load(Path(directory) / f'{name}.npz', allow_pickle=False) as archive:
            result[name] = {key: archive[key].copy() for key in archive.files}
    return result
