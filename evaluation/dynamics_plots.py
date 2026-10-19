"""导出阶段三训练证据与多方法动力学估计图，保留原始采样和来源。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl
import numpy as np

from evaluation.metrics import _array, _get_time, estimation_errors
from evaluation.plots import _axis, _COLORS, _figure, _font_name


_METHODS = {
    'uwb_ls': ('UWB LS', _COLORS[1], '--', 'o'),
    'eskf': ('ESKF', _COLORS[0], '-', '.'),
    'physics': ('Physics-EKF（物理）', _COLORS[2], '-.', '^'),
    'hybrid': ('Hybrid-EKF（已训练 TCN）', _COLORS[3], ':', 's'),
}


def _formats(formats) -> tuple:
    """在任何导出之前拒绝模糊或不支持的格式。"""
    if isinstance(formats, str):
        raise ValueError('formats 必须是 png、pdf、svg 构成的非空格式序列')
    formats = tuple(dict.fromkeys(formats))
    if not formats or any(item not in ('png', 'pdf', 'svg') for item in formats):
        raise ValueError('仅支持非空的 png、pdf、svg 格式序列')
    return formats


def _sources(groups: dict) -> list[dict]:
    """递归记录输入数组摘要，包含嵌套的动力学诊断及训练样本。"""
    records = []

    def visit(prefix, value):
        if isinstance(value, dict) or hasattr(value, 'items'):
            for key, item in value.items():
                visit(f'{prefix}.{key}' if prefix else str(key), item)
            return
        values = np.asarray(value)
        if values.dtype.hasobject:
            return
        records.append({'source': prefix, 'shape': list(values.shape), 'dtype': str(values.dtype),
                        'sha256': hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()})

    visit('', groups)
    return records


def _export(builders, output_dir: Path, formats: tuple, manifest: dict, manifest_name: str) -> list[Path]:
    """统一采用实际中文字体、白底、PDF 嵌入及 SVG 可编辑文字。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font = _font_name()
    styles = {
        'font.family': 'sans-serif', 'font.sans-serif': [font], 'font.size': 10,
        'axes.titlesize': 10, 'axes.labelsize': 10, 'legend.fontsize': 8,
        'figure.titlesize': 12, 'axes.unicode_minus': False, 'pdf.fonttype': 42,
        'ps.fonttype': 42, 'svg.fonttype': 'none', 'savefig.facecolor': 'white',
        'savefig.transparent': False, 'path.simplify': False,
    }
    outputs = []
    with mpl.rc_context(styles):
        for name, build in builders:
            figure = build()
            try:
                for file_format in formats:
                    destination = output_dir / f'{name}.{file_format}'
                    description = '原始采样；未平滑、未插值、无误差带；离线单次实验描述'
                    metadata = {'Title': figure._suptitle.get_text(), 'Creator': 'uwb_dynamics_sim'}
                    if file_format == 'pdf':
                        metadata.update(Subject=description, CreationDate=None, ModDate=None)
                    elif file_format == 'svg':
                        metadata.update(Description=description, Date=None)
                    else:
                        metadata['Description'] = description
                    figure.savefig(destination, format=file_format, dpi=300, metadata=metadata)
                    outputs.append(destination)
            finally:
                figure.clear()
    manifest.update({
        'schema_version': 1, 'formats': list(formats), 'outputs': [path.name for path in outputs],
        'coordinate_frames': {'world': 'ENU（东、北、上）', 'body': 'FLU（前、左、上）',
                              'quaternion': 'Hamilton wxyz，机体到世界'},
        'fonts': {'family': font, 'pdf': '嵌入 TrueType 字体', 'svg': '保留 text；依赖阅读环境字体'},
        'export': {'dpi': 300, 'background': '不透明白色', 'pdf_fonttype': 42, 'svg_text': True},
        'versions': {'matplotlib': mpl.__version__, 'numpy': np.__version__},
    })
    (output_dir / manifest_name).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8'
    )
    return outputs


def _training_arrays(history: dict, predictions: dict) -> tuple[dict, dict, list[tuple[int, int]]]:
    """训练损失与独立测试预测分开验证，飞行内部时钟必须递增。"""
    epoch = np.asarray(history.get('epoch', []), dtype=float)
    if (epoch.ndim != 1 or not len(epoch) or not np.all(np.isfinite(epoch))
            or np.any(np.diff(epoch) <= 0)):
        raise ValueError('history.epoch 必须非空、有限且严格递增')
    losses = {'epoch': epoch}
    for field in ('train_loss', 'val_loss'):
        losses[field] = _array(history, field, epoch.shape, 'history')
        if np.any(losses[field] < 0):
            raise ValueError('训练与验证损失必须非负')
    losses['loss_unit'] = str(history.get('loss_unit', '未声明单位；见训练配置'))
    time = np.asarray(predictions.get('time', []), dtype=float)
    flight = np.asarray(predictions.get('flight_id', []))
    if time.ndim != 1 or not len(time) or not np.all(np.isfinite(time)):
        raise ValueError('test_predictions.time 必须为非空有限一维时间轴')
    if flight.shape != time.shape or not np.issubdtype(flight.dtype, np.integer):
        raise ValueError('test_predictions.flight_id 必须为与时间轴等长的整数数组')
    arrays = {'time': time, 'flight_id': flight}
    unique_flights = np.unique(flight)
    if 'flight_names' in predictions:
        names = np.asarray(predictions['flight_names'])
        if names.shape != unique_flights.shape or names.dtype.kind not in ('U', 'S'):
            raise ValueError('test_predictions.flight_names 必须按排序后的飞行编号提供等长名称数组')
        arrays['flight_names'] = {int(identifier): str(name) for identifier, name in zip(unique_flights, names)}
    else:
        arrays['flight_names'] = {}
    for key in ('target_force_body', 'predicted_force_body'):
        arrays[key] = _array(predictions, key, (len(time), 3), 'test_predictions')
    arrays['physics_residual_force_body'] = (
        _array(predictions, 'physics_residual_force_body', (len(time), 3), 'test_predictions')
        if 'physics_residual_force_body' in predictions else np.zeros((len(time), 3)))
    boundaries = np.concatenate(([0], np.flatnonzero(np.diff(flight) != 0) + 1, [len(time)]))
    segments = list(zip(boundaries[:-1].tolist(), boundaries[1:].tolist()))
    for first, last in segments:
        if np.any(np.diff(time[first:last]) <= 0):
            raise ValueError('同一连续飞行段内测试时间必须严格递增')
    return losses, arrays, segments


def _loss_plot(losses: dict):
    figure = _figure((10., 5.5))
    axis = figure.subplots()
    for field, label, color, style, marker in [
        ('train_loss', '训练集损失', _COLORS[0], '-', 'o'),
        ('val_loss', '验证集损失', _COLORS[1], '--', 's'),
    ]:
        axis.plot(losses['epoch'], losses[field], color=color, linestyle=style, marker=marker,
                  markersize=3., linewidth=1., label=label)
    _axis(axis, f'损失 ({losses["loss_unit"]})', '逐轮真实训练记录；验证集用于选择模型')
    axis.set_xlabel('训练轮次')
    axis.legend()
    figure.suptitle('TCN 训练与验证损失：独立测试集未参与模型选择')
    return figure


def _test_force_plot(arrays: dict, segments: list):
    """每列一种独立飞行，每行一种机体力分量，原始数据不跨飞行相连。"""
    flights = np.unique(arrays['flight_id'])
    figure = _figure((max(6., 4.6 * len(flights)), 8.8))
    axes = figure.subplots(3, len(flights), sharex='col', sharey='row', squeeze=False)
    curves = [('target_force_body', '真残余力', _COLORS[5], '-'),
              ('physics_residual_force_body', '物理残余基线', _COLORS[1], '--'),
              ('predicted_force_body', '已训练 TCN 预测', _COLORS[0], '-.')]
    for component in range(3):
        for column, flight in enumerate(flights):
            axis = axes[component, column]
            for first, last in segments:
                if arrays['flight_id'][first] != flight:
                    continue
                for field, label, color, style in curves:
                    axis.plot(arrays['time'][first:last], arrays[field][first:last, component],
                              color=color, linestyle=style, linewidth=.9, label=label)
            name = arrays['flight_names'].get(int(flight), '')
            flight_title = f'飞行 {int(flight)}' + (f'：{name}' if name else '')
            panel = component * len(flights) + column + 1
            title = f'({panel}) {flight_title}' if component == 0 else f'({panel}) 机体 FLU {"xyz"[component]} 分量'
            _axis(axis, f'残余力 {"xyz"[component]} (N)' if column == 0 else '', title)
            if component < 2:
                axis.set_xlabel('')
    # constrained 布局为图外图例预留空间，避免遮住曲线或导出时裁切。
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles[:3], labels[:3], loc='outside lower center', ncol=3, frameon=False)
    figure.suptitle('独立测试飞行的气动残余力：每列一次飞行，同一分量采用相同纵轴尺度\n'
                   '保留全部原始样本；未平滑、未下采样')
    return figure


def plot_model_training(history: dict, test_predictions: dict, output_dir: Path,
                        formats=('png', 'pdf', 'svg')) -> list[Path]:
    """导出真实损失及测试三轴残余图；不跨飞行边界连接样本。

    history 接受 epoch、train_loss、val_loss，两条损失须使用同一单位，
    通过可选 loss_unit 明确声明。测试数组接受
    time、target_force_body、predicted_force_body、整数 flight_id；可选
    physics_residual_force_body，缺省为零残余预测基线；flight_names 可选，
    按排序后的独立飞行编号提供每次飞行的名称，用于分面标题。
    """
    formats = _formats(formats)
    losses, arrays, segments = _training_arrays(history, test_predictions)
    manifest = {
        'description': '阶段三 TCN 训练记录与独立测试残余力；科研通用草图，期刊要求未指定',
        'test_samples': len(arrays['time']), 'test_flights': np.unique(arrays['flight_id']).tolist(),
        'processing': {'smoothing': False, 'interpolation': False, 'aggregation': False,
                       'uncertainty': '未添加误差带；不将重叠历史窗口当作独立重复',
                       'flight_boundaries': '每列独立飞行，飞行编号改变时断线，保持各自原始时间',
                       'test_layout': '三行力分量乘独立飞行列数；同一行纵轴尺度相同；单一全图图例',
                       'downsampling': False, 'test_markers': False},
        'units': {'time': 's', 'force_body': 'N', 'epoch': '轮次', 'loss': losses['loss_unit']},
        'time_axes': {'training_loss': 'history.epoch', 'residual_test_prediction': 'test_predictions.time'},
        'source_arrays': _sources({'history': history, 'test_predictions': test_predictions}),
        'baseline': '显式提供的物理残余预测；未提供时为零残余预测',
        'figure_descriptions': {
            'training_loss': '训练与验证损失使用颜色、线型及标记区分，未绘制测试集选择曲线',
            'residual_test_prediction': '按独立飞行分列的三轴机体气动残余真值、物理基线和已训练网络预测；全图三个方法图例',
        },
    }
    return _export([('training_loss', lambda: _loss_plot(losses)),
                    ('residual_test_prediction', lambda: _test_force_plot(arrays, segments))],
                   output_dir, formats, manifest, 'training_figure_manifest.json')


def _comparison_errors(dataset, baseline, dynamics):
    """复用阶段二同一杆臂与时间关联口径，模型标识只在此适配接口。"""
    errors = {'eskf': estimation_errors(dataset, baseline)}
    for method in ('physics', 'hybrid'):
        if method not in dynamics:
            raise ValueError(f'缺少动力学状态 {method}')
        wrapped = {'uwb_ls': baseline['uwb_ls'], 'eskf': dynamics[method]}
        diagnostics = dynamics.get('uwb_diagnostics', {}).get(method)
        if diagnostics is not None:
            wrapped['diagnostics'] = diagnostics
        errors[method] = estimation_errors(dataset, wrapped)
        if not np.array_equal(errors[method]['common_time'], errors['eskf']['common_time']):
            raise ValueError('各方法必须采用相同的共同 UWB 时间轴')
    return errors


def _tag_plot(errors):
    figure = _figure((12., 9.))
    space = figure.add_subplot(2, 2, 1, projection='3d')
    common = errors['eskf']
    truth = common['true_tag_position']
    space.plot(*truth.T, color=_COLORS[5], linestyle='-', linewidth=1., label='天线真值')
    positions = [truth]
    for method, (label, color, style, marker) in _METHODS.items():
        position = common['ls_tag_position'] if method == 'uwb_ls' else errors[method]['eskf_tag_position']
        positions.append(position)
        space.plot(*position.T, color=color, linestyle=style, marker=marker, markersize=2.,
                   linewidth=.85, label=label)
    positions = np.concatenate(positions)
    lower, upper = positions.min(axis=0), positions.max(axis=0)
    span, center = max(float(np.max(upper - lower)), .1), (lower + upper) / 2
    for setter, midpoint in zip((space.set_xlim, space.set_ylim, space.set_zlim), center):
        setter(midpoint - .55 * span, midpoint + .55 * span)
    space.set_box_aspect((1, 1, 1))
    space.set(xlabel='东向 x (m)', ylabel='北向 y (m)', zlabel='天向 z (m)', title='(a) 天线轨迹：世界 ENU，等比例')
    space.legend(loc='upper left')
    for component in range(3):
        axis = figure.add_subplot(2, 2, component + 2)
        for method, (label, color, style, marker) in _METHODS.items():
            error = common['ls_tag_error'] if method == 'uwb_ls' else errors[method]['eskf_tag_error']
            axis.plot(common['common_time'], error[:, component], color=color, linestyle=style,
                      marker=marker, markersize=2., linewidth=.9, label=label)
        _axis(axis, f'天线位置误差 {"xyz"[component]} (m)', f'({chr(98 + component)}) 估计值减真值')
        axis.legend()
    figure.suptitle('四方法公平比较：同一 UWB 天线时间轴，未平滑或重采样')
    return figure


def _state_plot(errors):
    figure = _figure((12., 8.))
    axes = figure.subplots(2, 2).ravel()
    fields = [('body_position_error', '机体位置误差范数 (m)', '(a) 世界 ENU 机体位置'),
              ('velocity_error', '速度误差范数 (m/s)', '(b) 世界 ENU 速度'),
              ('attitude_error_deg', '最短旋转角 (°)', '(c) 姿态：q 与 −q 等价'),
              ('acc_bias_error', '加速度计偏置误差范数 (m/s²)', '(d) 机体 FLU 加速度计偏置')]
    for axis, (field, ylabel, title) in zip(axes, fields):
        maximum = 0.
        for method in ('eskf', 'physics', 'hybrid'):
            label, color, style, marker = _METHODS[method]
            error = errors[method][field]
            values = np.linalg.norm(error, axis=1) if error.ndim == 2 else error
            maximum = max(maximum, float(np.max(values)))
            axis.plot(errors[method]['eskf_time'], values, color=color, linestyle=style,
                      marker=marker, markersize=1.6, linewidth=.9, label=label)
        _axis(axis, ylabel, title)
        # 非负误差范数保留零基线，避免放大常量的浮点末位变化。
        axis.set_ylim(0., max(1e-3, 1.05 * maximum))
        axis.legend()
    figure.suptitle('三种滤波器的状态误差：各自全部输出事件时刻，原始样本无误差带')
    return figure


def plot_dynamics_comparison(dataset: dict, baseline_estimates: dict, dynamics_results: dict,
                             output_dir: Path, formats=('png', 'pdf', 'svg')) -> list[Path]:
    """导出共同 UWB 天线比较与三种滤波状态误差，不绘制未知扰力估计。"""
    formats = _formats(formats)
    errors = _comparison_errors(dataset, baseline_estimates, dynamics_results)
    manifest = {
        'description': '阶段三多方法离线比较；科研通用草图，未声明特定期刊规范',
        'common_samples': len(errors['eskf']['common_time']),
        'methods': {method: details[0] for method, details in _METHODS.items()},
        'processing': {'smoothing': False, 'interpolation': False, 'aggregation': False,
                       'tag_comparison': '全部共同 UWB 时刻的天线位置；LS 保留天线杆臂口径',
                       'state_errors': '各滤波器自己的全部输出事件，精确关联已保存真值',
                       'state_error_limits': '非负误差范数纵轴从零开始，完整包含各方法最大值',
                       'uncertainty': '无误差带；单次运行描述，不作统计一致性结论'},
        'units': {'time': 's', 'position': 'm', 'velocity': 'm/s', 'attitude': '°', 'acc_bias': 'm/s²'},
        'time_axes': {'dynamics_tag_comparison': 'uwb_ls.time；共同 UWB 时刻',
                      'dynamics_state_errors': ['eskf.time', 'physics.time', 'hybrid.time']},
        'source_arrays': _sources({'dataset': dataset, 'baseline_estimates': baseline_estimates,
                                   'dynamics_results': dynamics_results}),
        'figure_descriptions': {
            'dynamics_tag_comparison': '等比例天线轨迹与三轴误差；UWB LS、ESKF、Physics-EKF 和已训练 Hybrid-EKF',
            'dynamics_state_errors': '三个滤波器的机体位置、速度、姿态和加速度计偏置误差范数',
        },
    }
    return _export([('dynamics_tag_comparison', lambda: _tag_plot(errors)),
                    ('dynamics_state_errors', lambda: _state_plot(errors))],
                   output_dir, formats, manifest, 'dynamics_figure_manifest.json')
