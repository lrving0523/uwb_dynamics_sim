"""检验训练与动力学比较图的公平时钟、飞行断点及导出来源。"""

from copy import deepcopy
import importlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import pytest

from test_estimation_metrics import synthetic_estimation_data


def plotting_module():
    assert importlib.util.find_spec('evaluation.dynamics_plots') is not None, '缺少阶段三比较制图模块'
    return importlib.import_module('evaluation.dynamics_plots')


def training_data():
    """每次独立飞行保留自身时间轴，时间可在飞行边界重新开始。"""
    history = {'epoch': [1, 2, 3], 'train_loss': [.9, .5, .2], 'val_loss': [1., .7, .4], 'loss_unit': 'N²'}
    target = np.arange(18, dtype=float).reshape(6, 3) / 10
    prediction = {'time': np.array([0., .1, .2, 0., .1, .2]), 'target_force_body': target,
                  'predicted_force_body': target + .1,
                  'physics_residual_force_body': np.zeros((6, 3)), 'flight_id': np.array([0, 0, 0, 1, 1, 1])}
    return history, prediction


def dynamics_data():
    dataset, baseline = synthetic_estimation_data()
    physics, hybrid = deepcopy(baseline['eskf']), deepcopy(baseline['eskf'])
    physics['tag_position'] += [0., .3, 0.]
    hybrid['tag_position'] += [0., 0., .2]
    return dataset, baseline, {'physics': physics, 'hybrid': hybrid,
                                'uwb_diagnostics': {'physics': deepcopy(baseline['diagnostics']),
                                                    'hybrid': deepcopy(baseline['diagnostics'])}}


def assert_exports(outputs, directory, manifest_name, expected_stems):
    """检查实际文件的 DPI、字体和可编辑文字，而非只检查存在。"""
    assert len(outputs) == len(expected_stems) * 3
    assert {path.stem for path in outputs} == expected_stems
    for path in outputs:
        assert isinstance(path, Path) and path.stat().st_size > 1000
        if path.suffix == '.png':
            with Image.open(path) as raster:
                assert raster.info['dpi'][0] == pytest.approx(300, abs=.1)
                assert raster.convert('RGB').getpixel((0, 0)) == (255, 255, 255)
        elif path.suffix == '.pdf':
            assert b'/FontFile2' in path.read_bytes()
        else:
            assert '<text' in path.read_text(encoding='utf-8')
    manifest = json.loads((directory / manifest_name).read_text(encoding='utf-8'))
    assert manifest['processing']['smoothing'] is False
    assert manifest['processing']['interpolation'] is False
    assert set(manifest['outputs']) == {path.name for path in outputs}
    assert manifest['source_arrays']
    return manifest


@pytest.mark.filterwarnings('error:Glyph .* missing from font')
def test_training_export_has_loss_and_independent_test_force_with_provenance(tmp_path):
    module = plotting_module()
    history, prediction = training_data()
    before = deepcopy((history, prediction))
    outputs = module.plot_model_training(history, prediction, tmp_path)
    manifest = assert_exports(outputs, tmp_path, 'training_figure_manifest.json',
                              {'training_loss', 'residual_test_prediction'})
    assert manifest['units']['force_body'] == 'N'
    assert manifest['units']['loss'] == 'N²'
    assert manifest['test_flights'] == [0, 1]
    for source, original in zip((history, prediction), before):
        for key in source:
            np.testing.assert_array_equal(source[key], original[key])


def test_training_test_flights_use_separate_facets_and_one_shared_legend(tmp_path, monkeypatch):
    module = plotting_module()
    history, prediction = training_data()
    prediction['flight_names'] = np.array(['circle', 'figure_eight'])
    captured = {}
    original = Figure.savefig

    def capture(figure, filename, *args, **kwargs):
        if Path(filename).stem == 'residual_test_prediction':
            captured['lines'] = [[(line.get_label(), np.asarray(line.get_xdata()).copy(),
                                  np.asarray(line.get_ydata()).copy()) for line in axis.lines]
                                 for axis in figure.axes]
            captured['markers'] = [[line.get_marker() for line in axis.lines] for axis in figure.axes]
            captured['styles'] = [[(line.get_color(), line.get_linestyle()) for line in axis.lines]
                                   for axis in figure.axes]
            captured['titles'] = [axis.get_title() for axis in figure.axes]
            captured['limits'] = [axis.get_ylim() for axis in figure.axes]
            captured['legend_labels'] = [[text.get_text() for text in legend.get_texts()]
                                         for legend in figure.legends]
            captured['axis_legends'] = [axis.get_legend() for axis in figure.axes]
        return original(figure, filename, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', capture)
    module.plot_model_training(history, prediction, tmp_path, formats=('svg',))
    assert len(captured['lines']) == 6
    for panel, axis in enumerate(captured['lines']):
        assert len(axis) == 3
        for _, time, _ in axis:
            np.testing.assert_array_equal(time, [0., .1, .2])
        assert any('TCN' in label for label, _, _ in axis)
        assert any('物理' in label for label, _, _ in axis)
        # 每个分面只使用对应飞行和对应分量；全部原始点保持不变。
        component, flight = divmod(panel, 2)
        np.testing.assert_array_equal(axis[0][2], prediction['target_force_body'][flight * 3:(flight + 1) * 3, component])
    # 取消密集采样点标记，方法仍有不同颜色与线型；图例只占全图底部。
    for markers in captured['markers']:
        assert all(marker in (None, 'None', '') for marker in markers)
    for styles in captured['styles']:
        assert len({color for color, _ in styles}) == 3
        assert len({style for _, style in styles}) == 3
    assert len(captured['legend_labels']) == 1 and len(captured['legend_labels'][0]) == 3
    assert all(legend is None for legend in captured['axis_legends'])
    assert 'circle' in captured['titles'][0] and 'figure_eight' in captured['titles'][1]
    for row in range(3):
        assert captured['limits'][2 * row] == captured['limits'][2 * row + 1]


@pytest.mark.filterwarnings('error:Glyph .* missing from font')
def test_dynamics_exports_show_four_methods_at_the_same_uwb_times(tmp_path, monkeypatch):
    module = plotting_module()
    dataset, baseline, dynamics = dynamics_data()
    captured = {}
    original = Figure.savefig

    def capture(figure, filename, *args, **kwargs):
        captured[Path(filename).stem] = [[(line.get_label(), np.asarray(line.get_xdata()).copy(),
                                         np.asarray(line.get_ydata()).copy()) for line in axis.lines]
                                        for axis in figure.axes]
        if Path(filename).stem == 'dynamics_state_errors':
            captured['state_limits'] = [axis.get_ylim() for axis in figure.axes]
        return original(figure, filename, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', capture)
    outputs = module.plot_dynamics_comparison(dataset, baseline, dynamics, tmp_path)
    manifest = assert_exports(outputs, tmp_path, 'dynamics_figure_manifest.json',
                              {'dynamics_tag_comparison', 'dynamics_state_errors'})
    assert manifest['common_samples'] == len(dataset['uwb']['time'])
    assert set(manifest['methods']) == {'uwb_ls', 'eskf', 'physics', 'hybrid'}
    for axis in captured['dynamics_tag_comparison'][1:]:
        assert len(axis) == 4
        for _, time, _ in axis:
            np.testing.assert_array_equal(time, dataset['uwb']['time'])
        labels = ' '.join(label for label, _, _ in axis)
        assert all(label in labels for label in ('UWB LS', 'ESKF', 'Physics-EKF', 'Hybrid-EKF'))
        physics = next(values for label, _, values in axis if 'Physics-EKF' in label)
        hybrid = next(values for label, _, values in axis if 'Hybrid-EKF' in label)
        if not np.array_equal(physics, hybrid):
            captured['different'] = True
    assert captured['different']
    for axis in captured['dynamics_state_errors']:
        assert any('Physics-EKF' in label for label, _, _ in axis)
        assert any('Hybrid-EKF' in label for label, _, _ in axis)
    for (lower, upper), axis in zip(captured['state_limits'], captured['dynamics_state_errors']):
        # 范数从零开始，不将浮点末位差异放大成变化或裁掉真实样本。
        assert lower == 0.
        assert upper >= max(float(np.max(values)) for _, _, values in axis)


@pytest.mark.parametrize('kind', ['format', 'training_shape', 'flight_time', 'dynamics_time'])
def test_invalid_plot_inputs_fail_before_creating_outputs(tmp_path, kind):
    module = plotting_module()
    if kind == 'dynamics_time':
        dataset, baseline, dynamics = dynamics_data()
        dynamics['physics']['time'][1] += .01
        with pytest.raises(ValueError):
            module.plot_dynamics_comparison(dataset, baseline, dynamics, tmp_path)
    else:
        history, prediction = training_data()
        formats = ('png',)
        if kind == 'format':
            formats = 'png'
        elif kind == 'training_shape':
            history['train_loss'] = [1.]
        else:
            prediction['time'][2] = .05
        with pytest.raises(ValueError):
            module.plot_model_training(history, prediction, tmp_path, formats=formats)
    assert list(tmp_path.iterdir()) == []
