"""验证原始仿真数据的图形导出与采样完整性。"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pytest

from evaluation import plots


@pytest.fixture
def raw_dataset():
    """使用长度不同的时钟，暴露错误复用真值时间轴的问题。"""
    t = np.linspace(0.0, 1.0, 7)
    position = np.column_stack((t, t ** 2, 1.0 + 0.2 * t))
    imu_t = np.array([0.0, 0.17, 0.34, 0.51, 0.68])
    uwb_t = np.array([0.0, 0.31, 0.62])
    rpm_t = np.array([0.0, 0.23, 0.46, 0.69])
    accel = np.column_stack((imu_t, -imu_t, 9.81 + imu_t))
    accel[2, 0] = np.nan
    return {
        "truth": {
            "time": t,
            "position": position,
            "velocity": np.column_stack((np.ones_like(t), 2 * t, 0.2 * np.ones_like(t))),
            "quaternion": np.tile([1.0, 0.0, 0.0, 0.0], (len(t), 1)),
            "omega": np.zeros((len(t), 3)),
            "acceleration_world": np.tile([0.0, 2.0, 0.0], (len(t), 1)),
            "reference_position": position + [0.1, 0.0, 0.0],
            "target_rpm": 1100 + t[:, None] * np.array([[10, 20, 30, 40]]),
            "true_rpm": 1080 + t[:, None] * np.array([[11, 21, 31, 41]]),
            "aero_force_body": np.column_stack((-t, 0.5 * t, -0.1 * t)),
            "external_force_world": np.column_stack((2 * t, -t, 0.3 * t)),
            "thrust_body": np.tile([0.0, 0.0, 14.7], (len(t), 1)),
        },
        "imu": {"time": imu_t, "accel": accel, "gyro": np.column_stack((imu_t, 2 * imu_t, -imu_t))},
        "uwb": {
            "time": uwb_t,
            "ranges": np.array([[2.0, 3.0], [2.2, np.nan], [2.4, 3.4]]),
            "anchors": np.array([[0.0, 0.0, 0.0], [3.0, 2.0, 1.0]]),
        },
        "rpm": {"time": rpm_t, "rpm": 1070 + rpm_t[:, None] * np.array([[10, 20, 30, 40]])},
    }


def test_exports_five_groups_and_provenance_in_all_formats(raw_dataset, tmp_path):
    outputs = plots.plot_dataset(raw_dataset, tmp_path)
    assert len(outputs) == 15
    assert {path.stem for path in outputs} == {"truth_trajectory", "imu", "uwb", "rpm", "forces"}
    assert {path.suffix for path in outputs} == {".png", ".pdf", ".svg"}
    for path in outputs:
        assert isinstance(path, Path)
        assert path.is_file() and path.stat().st_size > 1000
        if path.suffix == ".png":
            assert path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
        elif path.suffix == ".pdf":
            assert path.read_bytes().startswith(b"%PDF")
        else:
            assert "<text" in path.read_text(encoding="utf-8")
    assert "时间" in (tmp_path / "imu.svg").read_text(encoding="utf-8")
    manifest = json.loads((tmp_path / "figure_manifest.json").read_text(encoding="utf-8"))
    assert manifest["samples"] == {"truth": 7, "imu": 5, "uwb": 3, "rpm": 4}
    assert manifest["formats"] == ["png", "pdf", "svg"]
    assert manifest["processing"]["smoothing"] is False
    assert manifest["processing"]["interpolation"] is False
    assert manifest["units"]["forces"] == "N"
    assert set(manifest["outputs"]) == {path.name for path in outputs}
    assert plt.get_fignums() == []


def test_raw_samples_keep_independent_clocks_and_nan_gaps(raw_dataset, tmp_path, monkeypatch):
    saved_lines = {}
    original_savefig = Figure.savefig

    def inspect_and_save(figure, filename, *args, **kwargs):
        saved_lines[Path(filename).stem] = [
            [(line.get_label(), np.array(line.get_xdata()), np.array(line.get_ydata()))
             for line in axis.lines]
            for axis in figure.axes
        ]
        return original_savefig(figure, filename, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", inspect_and_save)
    before = {group: {name: array.copy() for name, array in values.items()}
              for group, values in raw_dataset.items()}
    plots.plot_dataset(raw_dataset, tmp_path, formats=("png",))
    imu_line = saved_lines["imu"][0][0]
    np.testing.assert_array_equal(imu_line[1], raw_dataset["imu"]["time"])
    np.testing.assert_array_equal(imu_line[2], raw_dataset["imu"]["accel"][:, 0])
    uwb_line = saved_lines["uwb"][0][1]
    np.testing.assert_array_equal(uwb_line[1], raw_dataset["uwb"]["time"])
    np.testing.assert_array_equal(uwb_line[2], raw_dataset["uwb"]["ranges"][:, 1])
    for motor, axis_lines in enumerate(saved_lines["rpm"]):
        np.testing.assert_array_equal(axis_lines[0][1], raw_dataset["truth"]["time"])
        np.testing.assert_array_equal(axis_lines[0][2], raw_dataset["truth"]["target_rpm"][:, motor])
        np.testing.assert_array_equal(axis_lines[1][2], raw_dataset["truth"]["true_rpm"][:, motor])
        np.testing.assert_array_equal(axis_lines[2][1], raw_dataset["rpm"]["time"])
        np.testing.assert_array_equal(axis_lines[2][2], raw_dataset["rpm"]["rpm"][:, motor])
    for row, field in enumerate(("aero_force_body", "external_force_world")):
        for component in range(3):
            np.testing.assert_array_equal(saved_lines["forces"][row][component][2],
                                          raw_dataset["truth"][field][:, component])
    for group in before:
        for field in before[group]:
            np.testing.assert_array_equal(raw_dataset[group][field], before[group][field])


@pytest.mark.parametrize("formats", [("jpeg",), (), ("../png",)])
def test_rejects_unsupported_or_empty_formats_before_export(raw_dataset, tmp_path, formats):
    with pytest.raises(ValueError):
        plots.plot_dataset(raw_dataset, tmp_path, formats=formats)
    assert list(tmp_path.iterdir()) == []


def test_rejects_misaligned_sensor_samples_before_export(raw_dataset, tmp_path):
    raw_dataset["imu"]["accel"] = np.ones((4, 3))
    with pytest.raises(ValueError, match="imu.*accel"):
        plots.plot_dataset(raw_dataset, tmp_path, formats=("png",))
    assert list(tmp_path.iterdir()) == []
