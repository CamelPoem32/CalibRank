"""Child-process trajectory animation support for calibration experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


def save_calibration_animation_payload(
    payload_path: str | Path,
    *,
    reference_timestamps_s: Sequence[float],
    reference_poses_se3: np.ndarray,
    baseline_timestamps_s: Sequence[float],
    baseline_poses_se3: np.ndarray,
    injected_timestamps_s: Sequence[float],
    injected_poses_se3: np.ndarray,
    baseline_tau_I_s: Sequence[float],
    injected_tau_I_s: Sequence[float],
    calibration_timestamps_s: Sequence[float],
    target_tau_I_delta_s: float = 0.0,
    baseline_bias_g_radps: Sequence[Sequence[float]] | None = None,
    injected_bias_g_radps: Sequence[Sequence[float]] | None = None,
    target_bias_g_delta_radps: Sequence[float] | None = None,
    baseline_tau_L_s: Sequence[float] | None = None,
    injected_tau_L_s: Sequence[float] | None = None,
    target_tau_L_delta_s: float = 0.0,
    baseline_T_B_I: Sequence[np.ndarray] | None = None,
    injected_T_B_I: Sequence[np.ndarray] | None = None,
    baseline_T_B_L: Sequence[np.ndarray] | None = None,
    injected_T_B_L: Sequence[np.ndarray] | None = None,
    baseline_T_B_I_ln: Sequence[Sequence[float]] | None = None,
    injected_T_B_I_ln: Sequence[Sequence[float]] | None = None,
    baseline_T_B_L_ln: Sequence[Sequence[float]] | None = None,
    injected_T_B_L_ln: Sequence[Sequence[float]] | None = None,
    target_T_B_I_delta_ln: Sequence[float] | None = None,
    target_T_B_L_delta_ln: Sequence[float] | None = None,
    target_T_L_I_delta_ln: Sequence[float] | None = None,
) -> Path:
    """Write a compact numeric animation payload for a renderer subprocess.

    Args:
        payload_path: Output .npz path.
        reference_timestamps_s: Reference LiDAR trajectory timestamps, shape (N,).
        reference_poses_se3: Reference LiDAR trajectory poses, shape (N, 4, 4).
        baseline_timestamps_s: Baseline estimated trajectory timestamps.
        baseline_poses_se3: Baseline estimated poses.
        injected_timestamps_s: Injected estimated trajectory timestamps.
        injected_poses_se3: Injected estimated poses.
        baseline_tau_I_s: Baseline window-wise IMU time-offset estimates.
        injected_tau_I_s: Injected window-wise IMU time-offset estimates.
        calibration_timestamps_s: Window midpoint timestamps for calibration estimates, shape (K,).
        target_tau_I_delta_s: Known injected IMU time-offset delta.
        baseline_bias_g_radps: Baseline gyroscope-bias estimates, shape (K, 3).
        injected_bias_g_radps: Injected-run gyroscope-bias estimates, shape (K, 3).
        target_bias_g_delta_radps: Known injected gyroscope-bias delta, shape (3,).
        baseline_tau_L_s: Baseline LiDAR time-offset estimates, shape (K,).
        injected_tau_L_s: Injected-run LiDAR time-offset estimates, shape (K,).
        target_tau_L_delta_s: Known injected LiDAR time-offset delta.
        baseline_T_B_I: Baseline body-from-IMU estimates, shape (K, 4, 4).
        injected_T_B_I: Injected-run body-from-IMU estimates, shape (K, 4, 4).
        baseline_T_B_L: Baseline body-from-LiDAR estimates, shape (K, 4, 4).
        injected_T_B_L: Injected-run body-from-LiDAR estimates, shape (K, 4, 4).
        baseline_T_B_I_ln: Optional precomputed Ln(T_B_I) vectors, shape (K, 6) in [rot, trans] order.
        injected_T_B_I_ln: Optional precomputed injected Ln(T_B_I) vectors.
        baseline_T_B_L_ln: Optional precomputed Ln(T_B_L) vectors.
        injected_T_B_L_ln: Optional precomputed injected Ln(T_B_L) vectors.
        target_T_B_I_delta_ln: Known injected-minus-baseline T_B_I target in Ln coordinates, shape (6,).
        target_T_B_L_delta_ln: Known injected-minus-baseline T_B_L target.
        target_T_L_I_delta_ln: Known injected-minus-baseline T_L_I target, where T_L_I = inv(T_B_L) @ T_B_I.

    Returns:
        Resolved payload path.
    """
    path = Path(payload_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)

    calibration_timestamps = np.asarray(calibration_timestamps_s, dtype=float).reshape(-1)
    n_calibration = calibration_timestamps.size

    baseline_T_B_I_array = _as_optional_se3_series(baseline_T_B_I, n_calibration)
    injected_T_B_I_array = _as_optional_se3_series(injected_T_B_I, n_calibration)
    baseline_T_B_L_array = _as_optional_se3_series(baseline_T_B_L, n_calibration)
    injected_T_B_L_array = _as_optional_se3_series(injected_T_B_L, n_calibration)

    baseline_T_L_I_array = _compute_T_L_I_series(baseline_T_B_I_array, baseline_T_B_L_array)
    injected_T_L_I_array = _compute_T_L_I_series(injected_T_B_I_array, injected_T_B_L_array)

    np.savez_compressed(
        path,
        reference_timestamps_s=np.asarray(reference_timestamps_s, dtype=float),
        reference_positions_m=np.asarray(reference_poses_se3, dtype=float)[:, :3, 3],
        baseline_timestamps_s=np.asarray(baseline_timestamps_s, dtype=float),
        baseline_positions_m=np.asarray(baseline_poses_se3, dtype=float)[:, :3, 3],
        injected_timestamps_s=np.asarray(injected_timestamps_s, dtype=float),
        injected_positions_m=np.asarray(injected_poses_se3, dtype=float)[:, :3, 3],
        calibration_timestamps_s=calibration_timestamps,
        baseline_tau_I_s=np.asarray(baseline_tau_I_s, dtype=float),
        injected_tau_I_s=np.asarray(injected_tau_I_s, dtype=float),
        target_tau_I_delta_s=np.asarray([float(target_tau_I_delta_s)], dtype=float),
        baseline_tau_L_s=_as_optional_scalar_series(baseline_tau_L_s, n_calibration),
        injected_tau_L_s=_as_optional_scalar_series(injected_tau_L_s, n_calibration),
        target_tau_L_delta_s=np.asarray([float(target_tau_L_delta_s)], dtype=float),
        baseline_bias_g_radps=_as_optional_vector_series(baseline_bias_g_radps, n_calibration, 3),
        injected_bias_g_radps=_as_optional_vector_series(injected_bias_g_radps, n_calibration, 3),
        target_bias_g_delta_radps=_as_target_vector(target_bias_g_delta_radps, 3),
        baseline_T_B_I_ln=_optional_or_computed_ln(baseline_T_B_I_ln, baseline_T_B_I_array, n_calibration),
        injected_T_B_I_ln=_optional_or_computed_ln(injected_T_B_I_ln, injected_T_B_I_array, n_calibration),
        baseline_T_B_L_ln=_optional_or_computed_ln(baseline_T_B_L_ln, baseline_T_B_L_array, n_calibration),
        injected_T_B_L_ln=_optional_or_computed_ln(injected_T_B_L_ln, injected_T_B_L_array, n_calibration),
        baseline_T_L_I_ln=_se3_ln_series(baseline_T_L_I_array),
        injected_T_L_I_ln=_se3_ln_series(injected_T_L_I_array),
        target_T_B_I_delta_ln=_as_target_vector(target_T_B_I_delta_ln, 6),
        target_T_B_L_delta_ln=_as_target_vector(target_T_B_L_delta_ln, 6, default=np.zeros(6)),
        target_T_L_I_delta_ln=_as_target_vector(target_T_L_I_delta_ln, 6),
    )
    return path.resolve()


def render_calibration_animation(
    payload_path: str | Path,
    output_path: str | Path,
    *,
    backend: str = "mp4",
    fps: int = 12,
    dpi: int = 130,
    every_nth_frame: int = 1,
    max_rendered_frames: int | None = None,
    codec: str = "libx264",
    fallback_html: bool = True,
) -> dict[str, str | int]:
    """Render a calibration trajectory animation from a saved payload.

    Args:
        payload_path: Input .npz payload path.
        output_path: Requested output .mp4 or .html path.
        backend: "mp4" or "html".
        fps: Animation frames per second.
        dpi: Figure DPI for rendered frames.
        every_nth_frame: Frame decimation factor.
        max_rendered_frames: Optional cap for smoke tests and quick previews.
        codec: ffmpeg codec used for MP4 output.
        fallback_html: If true, write referenced-frame HTML when MP4 rendering fails or when backend="html".

    Returns:
        Dictionary with backend, output_path, and frame_count.
    """
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    payload = _load_payload(payload_path)
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    frame_times = _frame_times(payload)
    frame_times = frame_times[:: max(int(every_nth_frame), 1)]
    if max_rendered_frames is not None:
        frame_times = frame_times[: int(max_rendered_frames)]
    if frame_times.size == 0:
        raise ValueError("Animation payload does not contain any frame times")

    frames_dir = output.with_suffix("").parent / f"{output.with_suffix('').name}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_files = _render_frame_sequence(payload, frame_times, frames_dir, dpi=dpi)

    if backend == "html":
        html_path = output if output.suffix.lower() == ".html" else output.with_suffix(".html")
        write_referenced_frame_html(
            html_path,
            frame_files=frame_files,
            fps=fps,
            title="Calibration injection trajectory animation",
        )
        return {"backend": "html", "output_path": str(html_path), "frame_count": len(frame_files)}

    try:
        fig, ax = _make_animation_figure()
        lines = _initialize_animation_axes(ax, payload)
        writer = FFMpegWriter(fps=fps, codec=codec)
        with writer.saving(fig, str(output), dpi=dpi):
            for time_s in frame_times:
                _update_animation_lines(lines, payload, float(time_s))
                writer.grab_frame()
        plt.close(fig)
        return {"backend": "mp4", "output_path": str(output), "frame_count": len(frame_times)}
    except Exception:
        plt.close("all")
        if not fallback_html:
            raise
        html_path = output.with_suffix(".html")
        write_referenced_frame_html(
            html_path,
            frame_files=frame_files,
            fps=fps,
            title="Calibration injection trajectory animation",
        )
        return {"backend": "html", "output_path": str(html_path), "frame_count": len(frame_files)}


def write_referenced_frame_html(
    html_path: str | Path,
    *,
    frame_files: Sequence[str | Path],
    fps: int,
    title: str,
) -> Path:
    """Write an HTML animation shell that references external PNG frames.

    Args:
        html_path: Output HTML path.
        frame_files: Ordered PNG frame files. Paths are written relative to the HTML file where possible.
        fps: Playback frame rate.
        title: Page title.

    Returns:
        Resolved HTML path.
    """
    html = Path(html_path).expanduser().resolve()
    html.parent.mkdir(parents=True, exist_ok=True)
    relative_frames = [Path(frame).resolve().relative_to(html.parent) for frame in frame_files]
    frame_list = ",\n".join(f'    "{frame.as_posix()}"' for frame in relative_frames)
    delay_ms = max(int(round(1000.0 / max(int(fps), 1))), 1)
    html.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: sans-serif; margin: 24px; background: #f8f8f8; color: #222; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ccc; background: white; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <img id="frame" src="" alt="calibration animation frame">
  <script>
  const frames = [
{frame_list}
  ];
  let index = 0;
  const frame = document.getElementById("frame");
  function tick() {{
    frame.src = frames[index];
    index = (index + 1) % frames.length;
  }}
  tick();
  setInterval(tick, {delay_ms});
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return html


def _load_payload(payload_path: str | Path) -> Mapping[str, np.ndarray]:
    """Load the numeric animation payload into a dictionary."""
    with np.load(Path(payload_path).expanduser(), allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _frame_times(payload: Mapping[str, np.ndarray]) -> np.ndarray:
    """Choose animation frame times from all available trajectory timestamps."""
    times = np.unique(
        np.concatenate(
            [
                payload["reference_timestamps_s"],
                payload["baseline_timestamps_s"],
                payload["injected_timestamps_s"],
            ]
        )
    )
    return times[np.isfinite(times)]


def _render_frame_sequence(payload: Mapping[str, np.ndarray], frame_times: np.ndarray, frames_dir: Path, *, dpi: int) -> list[Path]:
    """Render individual PNG frames used by MP4 fallback HTML."""
    import matplotlib.pyplot as plt

    frame_files = []
    for index, time_s in enumerate(frame_times):
        fig, ax = _make_animation_figure()
        lines = _initialize_animation_axes(ax, payload)
        _update_animation_lines(lines, payload, float(time_s))
        frame_path = frames_dir / f"frame_{index:05d}.png"
        fig.savefig(frame_path, dpi=dpi)
        plt.close(fig)
        frame_files.append(frame_path)
    return frame_files


def _make_animation_figure():
    """Create a wide figure with room for the calibration text panel."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 7.5))
    fig.subplots_adjust(left=0.06, right=0.56, top=0.9, bottom=0.1)
    return fig, ax


def _initialize_animation_axes(axis, payload: Mapping[str, np.ndarray]) -> dict[str, object]:
    """Create animation artists and consistent axis limits."""
    reference = payload["reference_positions_m"]
    baseline = payload["baseline_positions_m"]
    injected = payload["injected_positions_m"]
    positions = np.vstack([reference, baseline, injected])
    finite = positions[np.all(np.isfinite(positions), axis=1)]
    if finite.size == 0:
        finite = np.zeros((1, 3))
    span = np.maximum(np.ptp(finite[:, :2], axis=0), 1.0)
    center = 0.5 * (np.min(finite[:, :2], axis=0) + np.max(finite[:, :2], axis=0))
    radius = 0.6 * float(np.max(span))

    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m]")
    axis.grid(True, alpha=0.3)

    reference_line, = axis.plot([], [], color="black", linewidth=1.5, label="reference LiDAR trajectory")
    baseline_line, = axis.plot([], [], color="C0", linewidth=1.5, label="baseline estimate")
    injected_line, = axis.plot([], [], color="C3", linewidth=1.5, label="injected estimate")
    title = axis.set_title("")
    calibration_text = axis.figure.text(
        0.59,
        0.88,
        "",
        ha="left",
        va="top",
        family="monospace",
        fontsize=8.2,
        linespacing=1.12,
        bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.96, "boxstyle": "round,pad=0.45"},
    )
    axis.legend(loc="best")
    return {
        "axis": axis,
        "title": title,
        "calibration_text": calibration_text,
        "reference_line": reference_line,
        "baseline_line": baseline_line,
        "injected_line": injected_line,
    }


def _update_animation_lines(lines: Mapping[str, object], payload: Mapping[str, np.ndarray], time_s: float) -> None:
    """Update trajectory artists and the calibration-state text panel."""
    for prefix, line_name in (("reference", "reference_line"), ("baseline", "baseline_line"), ("injected", "injected_line")):
        timestamps = payload[f"{prefix}_timestamps_s"]
        positions = payload[f"{prefix}_positions_m"]
        mask = timestamps <= time_s
        if not np.any(mask):
            mask[0] = True
        line = lines[line_name]
        line.set_data(positions[mask, 0], positions[mask, 1])

    lines["title"].set_text(f"Calibration injection trajectory, t = {time_s:.2f} s")
    lines["calibration_text"].set_text(_format_calibration_panel(payload, time_s))


def _format_calibration_panel(payload: Mapping[str, np.ndarray], time_s: float) -> str:
    """Build the frame-local text block with estimated and target calibration."""
    calibration_times = payload["calibration_timestamps_s"]
    index = _latest_index(calibration_times, time_s)
    latest_time = calibration_times[index] if calibration_times.size else np.nan

    bias_delta = _vector_delta(payload, "bias_g_radps", index, 3)
    tau_I_delta = _scalar_delta(payload, "tau_I_s", index)
    tau_L_delta = _scalar_delta(payload, "tau_L_s", index)
    T_B_I_delta = _vector_delta(payload, "T_B_I_ln", index, 6)
    T_B_L_delta = _vector_delta(payload, "T_B_L_ln", index, 6)
    T_L_I_delta = _vector_delta(payload, "T_L_I_ln", index, 6)

    lines = [
        f"window t <= {latest_time:.3f} s",
        "",
        "bias_g delta [rad/s]",
        f"  est    {_format_vector(bias_delta)}",
        f"  true target {_format_vector(payload['target_bias_g_delta_radps'])}",
        "",
        "time offsets delta [s]",
        f"  tau_I est {tau_I_delta: .6f}  true target {float(payload['target_tau_I_delta_s'][0]): .6f}",
        f"  tau_L est {tau_L_delta: .6f}  true target {float(payload['target_tau_L_delta_s'][0]): .6f}",
        "",
        "T_B_I Ln injected estimate",
        f"  {_format_vector(_row(payload['injected_T_B_I_ln'], index, 6))}",
        "T_B_I delta Ln",
        f"  est    {_format_vector(T_B_I_delta)}",
        f"  true target {_format_vector(payload['target_T_B_I_delta_ln'])}",
        "",
        "T_B_L Ln injected estimate",
        f"  {_format_vector(_row(payload['injected_T_B_L_ln'], index, 6))}",
        "T_B_L delta Ln",
        f"  est    {_format_vector(T_B_L_delta)}",
        f"  true target {_format_vector(payload['target_T_B_L_delta_ln'])}",
        "",
        "T_L_I Ln injected estimate",
        f"  {_format_vector(_row(payload['injected_T_L_I_ln'], index, 6))}",
        "T_L_I delta Ln",
        f"  est    {_format_vector(T_L_I_delta)}",
        f"  true target {_format_vector(payload['target_T_L_I_delta_ln'])}",
    ]
    return "\n".join(lines)


def _latest_index(timestamps: np.ndarray, time_s: float) -> int:
    """Return the latest calibration row available at time_s."""
    if timestamps.size == 0:
        return 0
    index = int(np.searchsorted(timestamps, time_s, side="right")) - 1
    return min(max(index, 0), len(timestamps) - 1)


def _scalar_delta(payload: Mapping[str, np.ndarray], suffix: str, index: int) -> float:
    """Return injected-minus-baseline scalar estimate for one row."""
    baseline = payload[f"baseline_{suffix}"]
    injected = payload[f"injected_{suffix}"]
    if baseline.size == 0 or injected.size == 0:
        return float("nan")
    return float(injected[index] - baseline[index])


def _vector_delta(payload: Mapping[str, np.ndarray], suffix: str, index: int, width: int) -> np.ndarray:
    """Return injected-minus-baseline vector estimate for one row."""
    baseline = _row(payload[f"baseline_{suffix}"], index, width)
    injected = _row(payload[f"injected_{suffix}"], index, width)
    return injected - baseline


def _row(values: np.ndarray, index: int, width: int) -> np.ndarray:
    """Return one vector row, or NaNs when the array is unavailable."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.full(width, np.nan, dtype=float)
    return values[min(max(index, 0), len(values) - 1)].reshape(width)


def _format_vector(values: Sequence[float]) -> str:
    """Format a numeric vector compactly for the animation panel."""
    values = np.asarray(values, dtype=float).reshape(-1)
    return "[" + " ".join("   nan" if not np.isfinite(v) else f"{v: .4f}" for v in values) + "]"


def _as_optional_scalar_series(values: Sequence[float] | None, length: int) -> np.ndarray:
    """Convert an optional scalar series to shape (length,)."""
    if values is None:
        return np.full(length, np.nan, dtype=float)
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape != (length,):
        raise ValueError(f"scalar calibration series must have shape ({length},)")
    return array


def _as_optional_vector_series(values: Sequence[Sequence[float]] | None, length: int, width: int) -> np.ndarray:
    """Convert an optional vector series to shape (length, width)."""
    if values is None:
        return np.full((length, width), np.nan, dtype=float)
    array = np.asarray(values, dtype=float)
    if array.shape != (length, width):
        raise ValueError(f"vector calibration series must have shape ({length}, {width})")
    return array


def _as_optional_se3_series(values: Sequence[np.ndarray] | None, length: int) -> np.ndarray:
    """Convert an optional SE(3) series to shape (length, 4, 4)."""
    if values is None:
        return np.full((length, 4, 4), np.nan, dtype=float)
    array = np.asarray(values, dtype=float)
    if array.shape != (length, 4, 4):
        raise ValueError(f"SE(3) calibration series must have shape ({length}, 4, 4)")
    return array


def _as_target_vector(values: Sequence[float] | None, width: int, *, default: np.ndarray | None = None) -> np.ndarray:
    """Convert an optional target vector to shape (width,)."""
    if values is None:
        if default is not None:
            return np.asarray(default, dtype=float).reshape(width)
        return np.full(width, np.nan, dtype=float)
    return np.asarray(values, dtype=float).reshape(width)


def _optional_or_computed_ln(values: Sequence[Sequence[float]] | None, se3s: np.ndarray, length: int) -> np.ndarray:
    """Use provided Lie vectors or compute them from SE(3) matrices."""
    if values is not None:
        return _as_optional_vector_series(values, length, 6)
    return _se3_ln_series(se3s)


def _se3_ln_series(se3s: np.ndarray) -> np.ndarray:
    """Map an SE(3) series to mrob.SE3.Ln vectors, preserving NaNs."""
    vectors = np.full((len(se3s), 6), np.nan, dtype=float)
    finite_rows = [index for index, se3 in enumerate(se3s) if np.all(np.isfinite(se3))]
    if not finite_rows:
        return vectors

    import mrob

    for index in finite_rows:
        vectors[index] = mrob.SE3(se3s[index]).Ln()
    return vectors


def _compute_T_L_I_series(T_B_I: np.ndarray, T_B_L: np.ndarray) -> np.ndarray:
    """Compute T_L_I = inv(T_B_L) @ T_B_I for every calibration row."""
    T_L_I = np.full_like(T_B_I, np.nan)
    for index, (body_from_imu, body_from_lidar) in enumerate(zip(T_B_I, T_B_L)):
        if np.all(np.isfinite(body_from_imu)) and np.all(np.isfinite(body_from_lidar)):
            T_L_I[index] = np.linalg.inv(body_from_lidar) @ body_from_imu
    return T_L_I
