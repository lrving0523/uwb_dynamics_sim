"""导出阶段二离线比较图，保留事件采样、诊断缺口及来源清单。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

from evaluation.metrics import estimation_errors, evaluate_estimates
from evaluation.plots import _axis, _COLORS, _components, _figure, _font_name


def _tag_comparison(errors: dict):
    """仅用相同 UWB 时刻的天线位置，不将 LS 转成机体定位。"""
    figure = _figure((11.5, 8.5))
    space = figure.add_subplot(2, 2, 1, projection='3d')
    curves = [
        ('true_tag_position', '天线真值', _COLORS[5], ':', None),
        ('ls_tag_position', 'UWB LS 天线', _COLORS[1], '--', 'o'),
        ('eskf_tag_position', 'ESKF 天线', _COLORS[0], '-', '.'),
    ]
    for field, label, color, style, marker in curves:
        space.plot(*errors[field].T, color=color, linestyle=style, marker=marker,
                   markersize=2.2, linewidth=.9, label=label)
    positions = np.concatenate([errors[field] for field, *_ in curves])
    lower, upper = positions.min(axis=0), positions.max(axis=0)
    span = max(float(np.max(upper - lower)), .1)
    center = (lower + upper) / 2
    for setter, midpoint in zip((space.set_xlim, space.set_ylim, space.set_zlim), center):
        setter(midpoint - span * .55, midpoint + span * .55)
    space.set_box_aspect((1, 1, 1))
    space.set(xlabel='东向 x (m)', ylabel='北向 y (m)', zlabel='天向 z (m)',
              title='(a) 天线轨迹：世界 ENU，空间等比例')
    space.legend(loc='upper left')
    for index, component in enumerate(('x', 'y', 'z')):
        axis = figure.add_subplot(2, 2, index + 2)
        for field, label, color, style, marker in [
            ('ls_tag_error', 'UWB LS', _COLORS[1], '--', 'o'),
            ('eskf_tag_error', 'ESKF', _COLORS[0], '-', '.'),
        ]:
            axis.plot(errors['common_time'], errors[field][:, index], color=color,
                      linestyle=style, marker=marker, markersize=2.2, linewidth=.9, label=label)
        _axis(axis, f'天线位置误差 {component} (m)', f'({chr(98 + index)}) {component} 分量：估计值减真值')
        axis.legend()
    figure.suptitle('公平比较：同一 UWB 时间戳的天线位置，未平滑或重采样')
    return figure


def _state_errors(errors: dict):
    """机体状态图采用 ESKF 自身输出事件时间，姿态使用最短旋转角。"""
    figure = _figure((12., 10.))
    axes = figure.subplots(3, 2, sharex=True).ravel()
    panels = [
        ('body_position_error', '位置误差 (m)', '(a) 机体位置：世界 ENU，估计值减真值'),
        ('velocity_error', '速度误差 (m/s)', '(b) 机体速度：世界 ENU，估计值减真值'),
        ('attitude_error_deg', '最短旋转角 (°)', '(c) 姿态误差：q 与 −q 等价'),
        ('right_local_rotation_error_rad', '局部旋转误差 (rad)', '(d) 右局部旋转向量：名义逆姿态乘真值'),
        ('acc_bias_error', '加速度计偏置误差 (m/s²)', '(e) 加速度计偏置：机体 FLU，估计值减真值'),
        ('gyro_bias_error', '陀螺仪偏置误差 (rad/s)', '(f) 陀螺仪偏置：机体 FLU，估计值减真值'),
    ]
    for axis, (field, ylabel, title) in zip(axes, panels):
        values = errors[field]
        if values.ndim == 2:
            _components(axis, errors['eskf_time'], values)
        else:
            axis.plot(errors['eskf_time'], values, color=_COLORS[3], linestyle='-.',
                      marker='.', markersize=1.6, linewidth=.85, label='最短旋转角')
            axis.legend()
        _axis(axis, ylabel, title)
    figure.suptitle('ESKF 状态误差：全部输出事件时刻，未平滑或重采样')
    return figure


def _diagnostics(errors: dict):
    """只显示现有的有效诊断；NaN 形成断点，空输入显式说明。"""
    figure = _figure((11., 7.))
    axes = figure.subplots(2, 1, sharex=True)
    for axis, time, values, label, title, color, style in [
        (axes[0], errors['diagnostic_time'], errors['nis'] / errors['nis_dof'],
         f'NIS / {errors["nis_dof"]}', '(a) 测距创新诊断：首帧初始化已排除', _COLORS[1], '--'),
        (axes[1], errors['eskf_time'], errors['nees15'] / 15,
         'NEES / 15', '(b) 完整状态误差诊断：仅有限、正定且条件数合理的协方差', _COLORS[0], '-'),
    ]:
        valid = np.isfinite(values)
        if np.any(valid):
            axis.plot(time, values, color=color, linestyle=style, marker='.', markersize=2.,
                      linewidth=.85, label=f'{label}，有效样本 {np.count_nonzero(valid)}')
            axis.legend()
        else:
            axis.text(.5, .5, '无可用诊断样本；未绘制数值', transform=axis.transAxes,
                      ha='center', va='center', color=_COLORS[5])
        if len(values) and np.count_nonzero(~valid):
            axis.text(.99, .98, f'无效样本 {np.count_nonzero(~valid)}，保留为断点',
                      transform=axis.transAxes, ha='right', va='top', fontsize=9)
        _axis(axis, f'{label}（无量纲）', title)
    figure.suptitle('单次运行 NIS / NEES 描述；不构成统计一致性检验或蒙特卡洛置信区间')
    return figure


def _source_manifest(dataset: dict, estimates: dict) -> list[dict]:
    """为输入数组记录字段、形状及内容摘要，便于追溯同一批数据。"""
    sources = []
    for prefix, source in [('dataset', dataset), ('estimates', estimates)]:
        for group_name, group in source.items():
            if not isinstance(group, dict):
                continue
            for field, value in group.items():
                values = np.asarray(value)
                if values.dtype.hasobject:
                    continue
                digest = hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
                sources.append({'source': f'{prefix}.{group_name}.{field}',
                                'shape': list(values.shape), 'dtype': str(values.dtype), 'sha256': digest})
    return sources


def plot_estimation(dataset: dict, estimates: dict, output_dir: Path,
                    formats=('png', 'pdf', 'svg')) -> list[Path]:
    """导出三组估计图和中文来源清单，返回图像文件路径。

    PNG 使用 300 dpi，PDF 嵌入 TrueType 字体，SVG 保留可编辑文字。
    不平滑、不插值；相邻原始采样用线段连接，诊断缺失保留为断点。
    """
    if isinstance(formats, str):
        raise ValueError('formats 必须是 png、pdf、svg 构成的非空格式序列')
    formats = tuple(dict.fromkeys(formats))
    if not formats or any(item not in ('png', 'pdf', 'svg') for item in formats):
        raise ValueError('仅支持非空的 png、pdf、svg 格式序列')
    errors = estimation_errors(dataset, estimates)
    report = evaluate_estimates(dataset, estimates)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font = _font_name()
    styles = {
        'font.family': 'sans-serif', 'font.sans-serif': [font], 'font.size': 10,
        'axes.titlesize': 10, 'axes.labelsize': 10, 'legend.fontsize': 9,
        'figure.titlesize': 12, 'axes.unicode_minus': False, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white',
        'savefig.transparent': False, 'path.simplify': False,
    }
    builders = [('tag_comparison', _tag_comparison), ('eskf_state_errors', _state_errors),
                ('estimation_diagnostics', _diagnostics)]
    outputs = []
    with mpl.rc_context(styles):
        for name, builder in builders:
            figure = builder(errors)
            try:
                for file_format in formats:
                    destination = output_dir / f'{name}.{file_format}'
                    metadata = {'Title': figure._suptitle.get_text(), 'Creator': 'uwb_dynamics_sim'}
                    description = '离线真值匹配；无平滑、无插值；单次运行描述，不作统计一致性结论'
                    if file_format == 'pdf':
                        metadata.update(Subject=description, CreationDate=None, ModDate=None)
                    elif file_format == 'svg':
                        metadata.update(Description=description, Date=None)
                    else:
                        metadata.update(Description=description)
                    figure.savefig(destination, format=file_format, dpi=300, metadata=metadata)
                    outputs.append(destination)
            finally:
                figure.clear()
    manifest = {
        'schema_version': 1,
        'description': '阶段二离线估计图；科研通用草图，未声明符合特定期刊规范',
        'common_samples': report['common_samples'],
        'formats': list(formats), 'outputs': [path.name for path in outputs],
        'processing': {
            'smoothing': False, 'interpolation': False, 'time_weighting': False,
            'transformations': '在已保存真值事件时刻构造误差；绘制原始误差样本，无误差带',
            'missing_values': '无效 NEES/NIS 为 NaN 断点；空诊断文字说明，未替换为零',
            'tag_comparison': '仅共同 UWB 时刻的天线位置；未使用真值姿态移除 LS 杆臂',
        },
        'time_axes': {'tag_comparison': 'uwb_ls.time；匹配 dataset.uwb.time、eskf.time 与 truth.time',
                      'eskf_state_errors': 'eskf.time；匹配 truth.time',
                      'estimation_diagnostics': ['diagnostics.time（NIS）', 'eskf.time（NEES）']},
        'units': {'position': 'm', 'velocity': 'm/s', 'attitude': '°', 'rotation_vector': 'rad',
                  'acc_bias': 'm/s²', 'gyro_bias': 'rad/s', 'time': 's', 'NIS_NEES': '无量纲'},
        'diagnostics': {
            'nis_samples': report['diagnostics']['nis']['count'],
            'nees15_samples': report['diagnostics']['nees15']['count'],
            'nees15_exclusions': report['diagnostics']['nees15_exclusions'],
            'interpretation': report['diagnostics']['interpretation'],
        },
        'coordinate_frames': {'world': 'ENU（东、北、上）', 'body': 'FLU（前、左、上）',
                              'quaternion': 'Hamilton wxyz，机体到世界'},
        'fonts': {'family': font, 'pdf': '嵌入 TrueType 字体', 'svg': '保留 text；依赖阅读环境字体'},
        'export': {'dpi': 300, 'background': '不透明白色', 'pdf_fonttype': 42, 'svg_text': True},
        'versions': {'matplotlib': mpl.__version__, 'numpy': np.__version__},
        'source': '调用方提供的阶段一数据与阶段二估计数组；输入内容保持不变',
        'source_arrays': _source_manifest(dataset, estimates),
        'figure_descriptions': {
            'tag_comparison': '世界 ENU 等比例三维天线轨迹及三个位置分量误差，LS 与 ESKF 用颜色和线型共同区分',
            'eskf_state_errors': '全部 ESKF 输出时刻的机体位置、速度、姿态与偏置误差；姿态使用最短角及右局部旋转向量',
            'estimation_diagnostics': '各自真实时钟下自由度归一的 NIS 和 NEES；无蒙特卡洛区间或通过检验结论',
        },
    }
    (output_dir / 'estimation_figure_manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )
    return outputs
