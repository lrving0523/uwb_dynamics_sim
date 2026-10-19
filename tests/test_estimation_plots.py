"""检验阶段二图形保留真实采样时间、诊断缺失与输入数据。"""

from copy import deepcopy
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import pytest

from evaluation import estimation_plots
from test_estimation_metrics import synthetic_estimation_data


@pytest.mark.filterwarnings('error:Glyph .* missing from font')
def test_exports_three_figure_groups_with_editable_text_and_provenance(tmp_path):
    dataset, estimates = synthetic_estimation_data()
    before = deepcopy((dataset, estimates))
    outputs = estimation_plots.plot_estimation(dataset, estimates, tmp_path)
    assert len(outputs) == 9
    assert {path.stem for path in outputs} == {'tag_comparison', 'eskf_state_errors', 'estimation_diagnostics'}
    for path in outputs:
        assert isinstance(path, Path) and path.stat().st_size > 1000
        if path.suffix == '.png':
            with Image.open(path) as raster:
                assert raster.info['dpi'][0] == pytest.approx(300, abs=.1)
        elif path.suffix == '.pdf':
            assert b'/FontFile2' in path.read_bytes()
        else:
            assert '<text' in path.read_text(encoding='utf-8')
    manifest = json.loads((tmp_path / 'estimation_figure_manifest.json').read_text(encoding='utf-8'))
    assert manifest['common_samples'] == 5
    assert manifest['processing']['smoothing'] is False
    assert manifest['processing']['interpolation'] is False
    assert manifest['diagnostics']['nis_samples'] == 4
    assert manifest['diagnostics']['nees15_samples'] == 9
    assert set(manifest['outputs']) == {path.name for path in outputs}
    assert '单次运行' in manifest['diagnostics']['interpretation']
    for source, copy in zip((dataset, estimates), before):
        for group in source:
            for name, value in source[group].items():
                np.testing.assert_array_equal(value, copy[group][name])


def test_tag_lines_use_only_common_uwb_times_and_diagnostics_use_own_clock(tmp_path, monkeypatch):
    dataset, estimates = synthetic_estimation_data()
    captured = {}
    original = Figure.savefig

    def capture(figure, filename, *args, **kwargs):
        captured[Path(filename).stem] = [
            [(line.get_label(), np.asarray(line.get_xdata()).copy(), np.asarray(line.get_ydata()).copy())
             for line in axis.lines] for axis in figure.axes]
        return original(figure, filename, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', capture)
    estimation_plots.plot_estimation(dataset, estimates, tmp_path, formats=('png',))
    # 第一面板为等比例空间轨迹，其余面板均保留共同 UWB 时间轴。
    for axis in captured['tag_comparison'][1:]:
        for _, time, _ in axis:
            np.testing.assert_array_equal(time, dataset['uwb']['time'])
    nis_line = captured['estimation_diagnostics'][0][0]
    np.testing.assert_array_equal(nis_line[1], estimates['diagnostics']['time'])
    np.testing.assert_array_equal(nis_line[2], estimates['diagnostics']['nis'] / 4)


def test_missing_diagnostics_are_labeled_and_never_drawn_as_zero(tmp_path, monkeypatch):
    dataset, estimates = synthetic_estimation_data()
    for key in estimates['diagnostics']:
        estimates['diagnostics'][key] = estimates['diagnostics'][key][:0]
    content = {}
    original = Figure.savefig

    def capture(figure, filename, *args, **kwargs):
        if Path(filename).stem == 'estimation_diagnostics':
            content['nis_lines'] = len(figure.axes[0].lines)
            content['labels'] = ' '.join(text.get_text() for text in figure.axes[0].texts)
        return original(figure, filename, *args, **kwargs)

    monkeypatch.setattr(Figure, 'savefig', capture)
    estimation_plots.plot_estimation(dataset, estimates, tmp_path, formats=('svg',))
    assert content['nis_lines'] == 0
    assert '无' in content['labels']
    manifest = json.loads((tmp_path / 'estimation_figure_manifest.json').read_text(encoding='utf-8'))
    assert manifest['diagnostics']['nis_samples'] == 0


@pytest.mark.parametrize('formats', [(), ('jpeg',), 'png'])
def test_invalid_formats_fail_before_writing_files(tmp_path, formats):
    dataset, estimates = synthetic_estimation_data()
    with pytest.raises(ValueError):
        estimation_plots.plot_estimation(dataset, estimates, tmp_path, formats=formats)
    assert list(tmp_path.iterdir()) == []


def test_unmatched_truth_fails_before_export(tmp_path):
    dataset, estimates = synthetic_estimation_data()
    estimates['eskf']['time'][1] += .01
    with pytest.raises(ValueError, match='时刻|时间'):
        estimation_plots.plot_estimation(dataset, estimates, tmp_path)
    assert list(tmp_path.iterdir()) == []
