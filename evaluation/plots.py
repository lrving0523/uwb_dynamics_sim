"""以独立采样时钟导出中文原始数据图，不进行滤波或估计。"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
from matplotlib import font_manager
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import numpy as np


_COLORS = ("#0072B2", "#D55E00", "#00866A", "#7A5195", "#8C564B", "#5D6D7E", "#B33A3A", "#006B70")
_STYLES = ("-", "--", "-.", ":")
_COMPONENTS = ("x", "y", "z")


def _font_name() -> str:
    """只优先选择实际已安装的中文字体，避免不存在字体的反复回退。"""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC",
                      "WenQuanYi Micro Hei", "DengXian", "Arial Unicode MS"):
        if candidate in available:
            return candidate
    return "DejaVu Sans"


def _validated_arrays(dataset: dict) -> dict:
    """在生成任何文件前验证形状；保留缺失数值的 NaN 断点。"""
    specifications = {
        "truth": {"position": 3, "reference_position": 3, "target_rpm": 4,
                  "true_rpm": 4, "aero_force_body": 3, "external_force_world": 3},
        "imu": {"accel": 3, "gyro": 3},
        "uwb": {},
        "rpm": {"rpm": 4},
    }
    arrays = {}
    for group, fields in specifications.items():
        if group not in dataset or "time" not in dataset[group]:
            raise ValueError(f"缺少 {group}.time 时间轴")
        values = dataset[group]
        time = np.asarray(values["time"], dtype=float)
        if (time.ndim != 1 or not len(time) or not np.all(np.isfinite(time))
                or np.any(np.diff(time) <= 0)):
            raise ValueError(f"{group}.time 必须为非空、有限且严格递增的一维时间轴")
        arrays[group] = {"time": time}
        for field, width in fields.items():
            if field not in values:
                raise ValueError(f"缺少 {group}.{field}")
            array = np.asarray(values[field], dtype=float)
            if array.shape != (len(time), width) or np.any(np.isinf(array)):
                raise ValueError(f"{group}.{field} 必须为形状 ({len(time)}, {width}) 且不含无穷值的数组")
            arrays[group][field] = array
    uwb = dataset["uwb"]
    if "anchors" not in uwb or "ranges" not in uwb:
        raise ValueError("缺少 uwb.anchors 或 uwb.ranges")
    anchors = np.asarray(uwb["anchors"], dtype=float)
    ranges = np.asarray(uwb["ranges"], dtype=float)
    if (anchors.ndim != 2 or anchors.shape[1] != 3 or not len(anchors)
            or not np.all(np.isfinite(anchors))):
        raise ValueError("uwb.anchors 必须为非空、有限的 (基站数, 3) 坐标数组")
    if ranges.shape != (len(arrays["uwb"]["time"]), len(anchors)) or np.any(np.isinf(ranges)):
        raise ValueError("uwb.ranges 的行数必须匹配测距时间轴，列数必须匹配基站数，且不含无穷值")
    arrays["uwb"].update(anchors=anchors, ranges=ranges)
    return arrays


def _figure(size: tuple[float, float]) -> Figure:
    figure = Figure(figsize=size, layout="constrained", facecolor="white")
    FigureCanvasAgg(figure)
    return figure


def _axis(axis, ylabel: str, title: str = "") -> None:
    axis.set(xlabel="时间 (s)", ylabel=ylabel, title=title)
    axis.grid(True, color="#D9D9D9", linewidth=0.5)
    axis.margins(x=0.02)


def _components(axis, time: np.ndarray, values: np.ndarray) -> None:
    for index, component in enumerate(_COMPONENTS):
        axis.plot(time, values[:, index], color=_COLORS[index], linestyle=_STYLES[index],
                  marker=".", markersize=1.6, linewidth=0.85, label=component)
    axis.legend(ncol=3)


def _trajectory(truth: dict) -> Figure:
    figure = _figure((11.5, 8.5))
    space = figure.add_subplot(2, 2, 1, projection="3d")
    position, reference = truth["position"], truth["reference_position"]
    space.plot(*position.T, color=_COLORS[0], linewidth=1.2, label="真实轨迹")
    space.plot(*reference.T, color=_COLORS[1], linestyle="--", linewidth=1.0, label="参考轨迹")
    space.set(xlabel="东向 x (m)", ylabel="北向 y (m)", zlabel="天向 z (m)", title="(a) 世界系 ENU 三维轨迹")
    finite = np.concatenate((position, reference), axis=0)
    finite = finite[np.all(np.isfinite(finite), axis=1)]
    if len(finite):
        lower, upper = finite.min(axis=0), finite.max(axis=0)
        span = max(float(np.max(upper - lower)), 0.1)
        center = (lower + upper) / 2
        for setter, midpoint in zip((space.set_xlim, space.set_ylim, space.set_zlim), center):
            setter(midpoint - span * 0.55, midpoint + span * 0.55)
        space.set_box_aspect((1, 1, 1))
    space.legend(loc="upper left")
    for index, (component, direction) in enumerate(zip(_COMPONENTS, ("东向", "北向", "天向"))):
        axis = figure.add_subplot(2, 2, index + 2)
        axis.plot(truth["time"], position[:, index], color=_COLORS[0], linewidth=1.0,
                  marker=".", markersize=1.6, label="真实位置")
        axis.plot(truth["time"], reference[:, index], color=_COLORS[1], linestyle="--",
                  linewidth=1.0, label="参考位置")
        _axis(axis, f"{direction}位置 {component} (m)", f"({chr(98 + index)}) {component} 分量")
        axis.legend()
    figure.suptitle("真实轨迹与参考轨迹：空间等比例视图及位置时序")
    return figure


def _imu(imu: dict) -> Figure:
    figure = _figure((11.0, 7.5))
    axes = figure.subplots(2, 1, sharex=True)
    for axis, field, label, title in zip(
        axes, ("accel", "gyro"), ("比力 (m/s²)", "角速度 (rad/s)"),
        ("(a) 加速度计比力，机体系 B（FLU）", "(b) 陀螺仪角速度，机体系 B（FLU）"),
    ):
        _components(axis, imu["time"], imu[field])
        _axis(axis, label, title)
    figure.suptitle("IMU 原始测量：采用 IMU 自身时间戳，NaN 保留为断点")
    return figure


def _uwb(uwb: dict) -> Figure:
    figure = _figure((11.0, 6.0))
    axis = figure.subplots()
    for index in range(uwb["ranges"].shape[1]):
        axis.plot(uwb["time"], uwb["ranges"][:, index], color=_COLORS[index % len(_COLORS)],
                  linestyle=_STYLES[index % len(_STYLES)], marker=("o", "s", "^", "D")[index % 4],
                  markersize=2.4, linewidth=0.8, label=f"基站 {index + 1}")
    _axis(axis, "原始测距 (m)", "基站编号按 anchors 坐标数组的行顺序排列")
    axis.legend(ncol=min(4, len(uwb["anchors"])))
    figure.suptitle("UWB 原始测距：采用 UWB 自身时间戳，NaN 保留为断点")
    return figure


def _rpm(truth: dict, rpm: dict) -> Figure:
    figure = _figure((11.5, 8.0))
    axes = figure.subplots(2, 2, sharex=True, sharey=True).ravel()
    for motor, (axis, location) in enumerate(zip(axes, ("前", "左", "后", "右"))):
        axis.plot(truth["time"], truth["target_rpm"][:, motor], color=_COLORS[3],
                  linestyle="--", linewidth=1.0, label="目标转速")
        axis.plot(truth["time"], truth["true_rpm"][:, motor], color=_COLORS[0],
                  linewidth=1.0, label="真实转速")
        axis.plot(rpm["time"], rpm["rpm"][:, motor], color=_COLORS[1], linestyle="none",
                  marker=".", markersize=2.3, label="测量转速")
        _axis(axis, "转速 (r/min)", f"电机 {motor + 1}（{location}）")
        axis.legend()
    figure.suptitle("电机转速：目标与真值采用真值时钟，测量采用 RPM 自身时钟")
    return figure


def _forces(truth: dict) -> Figure:
    figure = _figure((11.0, 7.5))
    axes = figure.subplots(2, 1, sharex=True, sharey=True)
    for axis, field, title in zip(
        axes, ("aero_force_body", "external_force_world"),
        ("(a) 气动力：机体系 B（FLU：前、左、上）", "(b) 未知外力：世界系 W（ENU：东、北、上）"),
    ):
        _components(axis, truth["time"], truth[field])
        _axis(axis, "力 (N)", title)
    figure.suptitle("力的分解真值：不同坐标系分别展示，均采用真值时间戳")
    return figure


def plot_dataset(dataset: dict, output_dir: Path, formats=("png", "pdf", "svg")) -> list[Path]:
    """导出五组原始数据图和来源清单，返回图像文件路径。

    线段只连接输入中的相邻有效样本；不平滑、不重采样、不添加误差带。
    NaN 在原始位置保留为断点，输入数组始终保持不变。
    """
    if isinstance(formats, str):
        raise ValueError("formats 必须是由 png、pdf、svg 构成的非空序列")
    formats = tuple(dict.fromkeys(formats))
    if not formats or any(item not in ("png", "pdf", "svg") for item in formats):
        raise ValueError("仅支持非空的 png、pdf、svg 格式序列")
    arrays = _validated_arrays(dataset)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    font = _font_name()
    outputs = []
    styles = {
        "font.family": "sans-serif", "font.sans-serif": [font], "font.size": 10,
        "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 9,
        "figure.titlesize": 12, "axes.unicode_minus": False, "pdf.fonttype": 42,
        "ps.fonttype": 42, "svg.fonttype": "none", "savefig.facecolor": "white",
        "savefig.transparent": False, "path.simplify": False,
    }
    builders = (
        ("truth_trajectory", lambda: _trajectory(arrays["truth"])),
        ("imu", lambda: _imu(arrays["imu"])),
        ("uwb", lambda: _uwb(arrays["uwb"])),
        ("rpm", lambda: _rpm(arrays["truth"], arrays["rpm"])),
        ("forces", lambda: _forces(arrays["truth"])),
    )
    with mpl.rc_context(styles):
        for name, build in builders:
            figure = build()
            try:
                for file_format in formats:
                    destination = output_dir / f"{name}.{file_format}"
                    metadata = {"Title": figure._suptitle.get_text(), "Creator": "uwb_dynamics_sim"}
                    if file_format == "pdf":
                        metadata.update(Subject="原始采样；无平滑、无重采样、无误差带", CreationDate=None, ModDate=None)
                    elif file_format == "svg":
                        metadata.update(Description="原始采样；无平滑、无重采样、无误差带", Date=None)
                    else:
                        metadata.update(Description="原始采样；无平滑、无重采样、无误差带")
                    figure.savefig(destination, format=file_format, dpi=300, metadata=metadata)
                    outputs.append(destination)
            finally:
                figure.clear()
    manifest = {
        "schema_version": 1,
        "description": "阶段一仿真原始数据图；未执行估计、滤波或误差统计。",
        "samples": {group: len(values["time"]) for group, values in arrays.items()},
        "formats": list(formats),
        "outputs": [path.name for path in outputs],
        "units": {"time": "s", "position": "m", "accel": "m/s²", "gyro": "rad/s",
                  "ranges": "m", "rpm": "r/min", "forces": "N"},
        "processing": {"smoothing": False, "interpolation": False, "aggregation": False,
                       "missing_values": "NaN 保留为断点；不删除样本", "uncertainty": "未添加误差带"},
        "coordinate_frames": {"world": "ENU（东、北、上）", "body": "FLU（前、左、上）"},
        "time_axes": {"truth_trajectory": "truth.time", "imu": "imu.time", "uwb": "uwb.time",
                      "rpm": ["truth.time", "rpm.time"], "forces": "truth.time"},
        "anchors_world_m": arrays["uwb"]["anchors"].tolist(),
        "fonts": {"family": font, "pdf": "嵌入 TrueType 字体", "svg": "保留 text；依赖阅读环境字体"},
        "export": {"dpi": 300, "background": "不透明白色", "svg_text": True, "pdf_fonttype": 42},
        "versions": {"matplotlib": mpl.__version__, "numpy": np.__version__},
        "source": "调用方提供的 truth、imu、uwb、rpm 数组；不修改输入数组",
    }
    (output_dir / "figure_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return outputs
