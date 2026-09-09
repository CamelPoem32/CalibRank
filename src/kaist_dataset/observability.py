"""Prepare KAIST measurements and run the notebook-11 observability workflow."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np

from calib_observability.backend import estimate_poses_dummy
from calib_observability.factor_observability import SUPPORTED_CALIBRATION_VARIABLES
from calib_observability.lie_se3 import se3_exp, se3_log
from calib_observability.simulation import reframe_dataset_to_fixed_extrinsic
from calib_observability.types import AccelerometerOptions, JacobianOptions
from calib_observability.visualization.quasi_realtime_rover import (
    save_quasi_realtime_rover_animation_mp4_subprocess,
)
from calib_observability.workflows import (
    DiscretePoseTrajectory,
    build_dataset_from_imported_sensor_streams,
    plot_local_crlb_accuracy,
    plot_observability_over_time,
    plot_rover_dataset_overview,
    run_rolling_observability_analysis,
)
from kaist_dataset import load_imus
from kaist_dataset.data import import_true_trajectory
from kaist_dataset.run_kaist_multi_sensor_pipeline import (
    T_B_I_INITIAL_FOR_DATASET,
    T_B_LL_INITIAL,
    T_B_RL_INITIAL,
    jsonable,
    kaist_dataset_layout,
    load_successful_lidar_map_poses,
    resolve_processed_data_directory,
)
from transform import se3_to_relative_se3


@dataclass(frozen=True)
class KaistObservabilityConfig:
    """CLI-independent configuration; times are seconds and counts are limits."""

    dataset_root: Path
    lidar: str = "left"
    processed_data_dir: Path | None = None
    output_dir: Path | None = None
    start_time: float = 0.0
    end_time: float | None = None
    imu_frequency_hz: float = 100.0
    window_size: float = 5.0
    step_size: float = 1.0
    max_imu_samples: int = 1000000
    max_lidar_poses: int = 50000
    max_ground_truth_poses: int = 50000
    max_analysis_windows: int = 1000
    max_rendered_frames: int = 1000
    trajectory_samples: int = 700
    mp4_fps: float = 10.0
    mp4_dpi: int = 100
    gyro_noise_std: float = 0.01
    accel_noise_std: float = 0.1
    lidar_rotation_noise_std: float = 0.05
    lidar_translation_noise_std: float = 0.05
    simple_accel_noise_std: float = 1e-6
    gravity_z: float = -9.81
    use_sparse: bool = False
    verbose: int = 0
    n_processes: int = 1

    def __post_init__(self) -> None:
        """Validate limits and physical settings before loading any files."""
        if self.lidar not in ("left", "right"):
            raise ValueError("lidar must be left or right")
        if self.verbose not in (0, 1, 2):
            raise ValueError("verbose must be 0, 1 or 2")
        if not isinstance(self.n_processes, int) or isinstance(self.n_processes, bool) or self.n_processes < 1:
            raise ValueError("n_processes must be a positive integer")
        counts = ("max_imu_samples", "max_lidar_poses", "max_ground_truth_poses",
                  "max_analysis_windows", "max_rendered_frames", "trajectory_samples", "mp4_dpi")
        for name in counts:
            value = getattr(self, name)
            minimum = 3 if name == "max_imu_samples" else 2
            if not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        positive = ("imu_frequency_hz", "window_size", "step_size", "mp4_fps",
                    "gyro_noise_std", "accel_noise_std", "lidar_rotation_noise_std",
                    "lidar_translation_noise_std", "simple_accel_noise_std")
        for name in positive:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.gravity_z):
            raise ValueError("gravity_z must be finite")
        if not np.isfinite(self.start_time) or self.start_time < 0:
            raise ValueError("start_time must be finite and nonnegative")
        if self.end_time is not None and (
            not np.isfinite(self.end_time) or self.end_time <= self.start_time
        ):
            raise ValueError("end_time must be finite and greater than start_time")


@dataclass
class PreparedObservability:
    """Dataset, reference trajectory in the displayed frame, and run metadata."""

    dataset: object
    reference_trajectory: DiscretePoseTrajectory
    metadata: dict


def uniform_sample_indices(count: int, maximum: int) -> np.ndarray:
    """Retain ordered samples across an entire stream, including both endpoints.

    Args:
        count: Number of available samples.
        maximum: Maximum retained samples, at least two.

    Returns:
        Unique increasing integer indices, shape `(min(count, maximum),)`.
    """
    if count < 2 or maximum < 2:
        raise ValueError("At least two available and retained samples are required")
    return np.rint(np.linspace(0, count - 1, min(count, maximum))).astype(int)


def _crop_indices(times, start, end, maximum, *, bracket=False):
    """Select a bounded interval with optional interpolation support.

    Args:
        times: Increasing timestamps in seconds, shape (N,).
        start: Requested interval start in seconds.
        end: Requested interval end in seconds.
        maximum: Maximum number of selected knots.
        bracket: Include one available knot on each side before subsampling.

    Returns:
        Increasing integer indices into times.
    """
    first = int(np.searchsorted(times, start, side="left"))
    stop = int(np.searchsorted(times, end, side="right"))
    if bracket:
        first = max(0, first - 1)
        stop = min(len(times), stop + 1)
    return first + uniform_sample_indices(stop - first, maximum)


def prepare_observability_inputs(imu, lidar_times, lidar_poses, truth_times, truth_poses, config):
    """Convert loaded KAIST streams into a bounded observability dataset.

    Args:
        imu: IMUData with timestamps in seconds and paired rad/s, m/s^2 channels.
        lidar_times: Absolute scan timestamps, shape `(L,)`.
        lidar_poses: Successful map poses T_W_L, shape `(L, 4, 4)`.
        truth_times: Absolute ground-truth timestamps, shape `(G,)`.
        truth_poses: Reference body poses T_W_B, shape `(G, 4, 4)`.
        config: KaistObservabilityConfig specifying cropping, budgets and weights.

    Returns:
        PreparedObservability with relative-time data, reference and metadata.

    Raises:
        ValueError: Streams have no overlap or insufficient retained support.
    """
    imu_times = np.asarray(imu.timestamps_s, dtype=float)
    lidar_times = np.asarray(lidar_times, dtype=float)
    truth_times = np.asarray(truth_times, dtype=float)
    for name, times in (("IMU", imu_times), ("LiDAR", lidar_times), ("truth", truth_times)):
        if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
            raise ValueError(f"{name} requires finite strictly increasing timestamps")

    # Crop on a common absolute clock; CLI offsets are relative to its start.
    overlap_start = max(imu_times[0], lidar_times[0], truth_times[0])
    overlap_end = min(imu_times[-1], lidar_times[-1], truth_times[-1])
    start = overlap_start + config.start_time
    end = overlap_end if config.end_time is None else min(overlap_end, overlap_start + config.end_time)
    if end <= start:
        raise ValueError("No common IMU/LiDAR/ground-truth interval remains")
    li = _crop_indices(lidar_times, start, end, config.max_lidar_poses)
    start, end = float(lidar_times[li[0]]), float(lidar_times[li[-1]])
    ii = _crop_indices(imu_times, start, end, config.max_imu_samples, bracket=True)
    gi = _crop_indices(truth_times, start, end, config.max_ground_truth_poses, bracket=True)

    # One origin preserves synchronization and avoids epoch-scale derivatives.
    origin = start
    lt = lidar_times[li] - origin
    it = imu_times[ii] - origin
    gt = truth_times[gi] - origin
    local_from_world = np.eye(4)
    local_from_world[:3, 3] = -np.asarray(lidar_poses)[li[0], :3, 3]
    local_lidar = local_from_world @ np.asarray(lidar_poses)[li]
    local_truth_body = local_from_world @ np.asarray(truth_poses)[gi]

    # Rounded KAIST calibration matrices are normalized via the same Lie-map
    # used by the imported adapter, so body conversion and reframing agree.
    T_B_L = se3_exp(se3_log(T_B_LL_INITIAL if config.lidar == "left" else T_B_RL_INITIAL))
    T_B_I = se3_exp(se3_log(T_B_I_INITIAL_FOR_DATASET))
    relative_poses = se3_to_relative_se3(local_lidar)
    raw_dataset = build_dataset_from_imported_sensor_streams(
        imu_timestamps=it,
        gyroscope=np.asarray(imu.gyro_radps)[ii],
        accelerometer=np.asarray(imu.accel_mps2)[ii],
        lidar_scan_timestamps=lt,
        lidar_relative_poses=relative_poses,
        T_B_I_initial_tangent=se3_log(T_B_I),
        T_B_L_initial_tangent=se3_log(T_B_L),
        gyro_noise_std=config.gyro_noise_std,
        accel_noise_std=config.accel_noise_std,
        lidar_pose_noise_std=np.array([config.lidar_rotation_noise_std] * 3 +
                                     [config.lidar_translation_noise_std] * 3),
        gravity_world=np.array([0.0, 0.0, config.gravity_z]),
    )
    body_trajectory = DiscretePoseTrajectory(lt, local_lidar @ np.linalg.inv(T_B_L))
    dataset = reframe_dataset_to_fixed_extrinsic(
        replace(raw_dataset, trajectory=body_trajectory), "T_B_L",
    )
    reference = DiscretePoseTrajectory(
        gt, local_truth_body @ T_B_L, mode="kaist_ground_truth_B_equals_L",
    )

    # Report actual adapter-retained counts, not merely the requested budgets.
    original = {"imu": len(imu_times), "lidar_poses": len(lidar_times), "ground_truth_poses": len(truth_times)}
    retained = {"imu": len(dataset.imu.sensor_timestamps), "lidar_poses": len(lt), "ground_truth_poses": len(gt)}
    effective = {}
    for name, times in (("imu", dataset.imu.sensor_timestamps), ("lidar", lt), ("ground_truth", gt)):
        effective[name] = float((len(times) - 1) / (times[-1] - times[0]))
    metadata = {
        "original_counts": original, "retained_counts": retained,
        "effective_rates_hz": effective,
        "common_overlap_absolute_s": [float(overlap_start), float(overlap_end)],
        "selected_interval_absolute_s": [start, end],
        "timestamp_origin_s": origin, "T_local_world": local_from_world,
        "T_vehicle_lidar": T_B_L, "T_vehicle_imu": T_B_I,
        "T_display_imu": dataset.T_B_I_true, "display_body_frame": f"vlp_{config.lidar}",
        "calibration_values_are_linearization_assumptions": True,
        "sampling": "Uniform retained indices over the selected overlap; bracketing IMU/truth support is retained.",
        "sample_caps_applied": {
            "imu": int(np.count_nonzero((imu_times >= start) & (imu_times <= end))) > config.max_imu_samples,
            "lidar": int(np.count_nonzero((lidar_times >= start) & (lidar_times <= end))) > config.max_lidar_poses,
            "ground_truth": int(np.count_nonzero((truth_times >= start) & (truth_times <= end))) > config.max_ground_truth_poses,
        },
    }
    return PreparedObservability(dataset, reference, metadata)


def load_observability_inputs(config: KaistObservabilityConfig) -> PreparedObservability:
    """Load one Urban directory using the existing KAIST ingestion helpers.

    Args:
        config: Validated paths, LiDAR choice, sample budgets and analysis settings.

    Returns:
        PreparedObservability including resolved source paths in its metadata.
    """
    root = Path(config.dataset_root).expanduser().resolve()
    name, sensor_root, truth_path = kaist_dataset_layout(root)
    processed = resolve_processed_data_directory(root, name, config.processed_data_dir, "poses", config.lidar)
    lidar_path = processed / f"lidar_map_poses_vlp_{config.lidar}.csv"

    # Load only measurements and precomputed poses; raw point clouds are unused.
    streams = load_imus(sensor_root, target_frequency_hz=config.imu_frequency_hz, max_files=1)
    if not streams:
        raise ValueError(f"No IMU streams found under {sensor_root}")
    imu_name, imu = next(iter(streams.items()))
    lt, lp = load_successful_lidar_map_poses(lidar_path)
    gt, gp = import_true_trajectory(truth_path)
    prepared = prepare_observability_inputs(imu, lt, lp, gt, gp, config)
    prepared.metadata["inputs"] = {
        "dataset_root": str(root), "processed_data_dir": str(processed),
        "lidar_csv": str(lidar_path), "ground_truth_csv": str(truth_path),
        "imu_name": imu_name, "imu_csv": imu.metadata.get("source_path"),
    }
    return prepared


def _save_comparison(prepared, output, samples):
    """Save bounded trajectory comparison in the dashboard frame.

    Args:
        prepared: PreparedObservability containing both trajectories.
        output: Existing results directory.
        samples: Number of uniformly spaced display samples.

    Returns:
        Path to the saved PNG.
    """
    dataset = prepared.dataset
    times = np.linspace(dataset.start_time, dataset.end_time, samples)
    positions = np.vstack([dataset.trajectory.position_at(t) for t in times])
    reference = np.vstack([prepared.reference_trajectory.position_at(t) for t in times])
    fig, axis = plt.subplots(figsize=(10, 8))
    axis.plot(positions[:, 0], positions[:, 1], label="LiDAR-derived trajectory")
    axis.plot(reference[:, 0], reference[:, 1], "--", label="KAIST ground truth")
    axis.set(xlabel="x [m]", ylabel="y [m]", title="KAIST trajectory comparison")
    axis.set_aspect("equal", adjustable="datalim")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    path = output / "trajectory_comparison.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _save_diagnostic_csv(series, path):
    """Save per-window observability and uncertainty values.

    Args:
        series: Nonempty ObservabilityVisualizationSeries.
        path: Destination CSV path.

    Returns:
        Path to the saved CSV; unbounded diagnostics retain inf/NaN values.
    """
    rows = []
    for index, snapshot in enumerate(series.snapshots):
        row = {"time_s": snapshot.current_time, "window_start_s": snapshot.window_start,
               "valid": snapshot.is_valid, "status": snapshot.status}
        for variable in SUPPORTED_CALIBRATION_VARIABLES:
            for label, values in (("rank", series.ranks), ("condition", series.condition_numbers),
                                  ("worst_std", series.worst_std_bounds)):
                row[f"{variable}_{label}"] = values[variable][index]
        rows.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_observability(config: KaistObservabilityConfig) -> Path:
    """Analyze one KAIST dataset and export a bounded subprocess MP4 dashboard.

    Args:
        config: Validated input paths, sampling budgets, window and export settings.

    Returns:
        Absolute results directory containing MP4, plots, CSV and JSON metadata.

    Raises:
        RuntimeError: FFmpeg is unavailable or no analysis windows are valid.
    """
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("FFmpeg is required for MP4 export. Install it before running this command.")
    root = Path(config.dataset_root).expanduser().resolve()
    output = config.output_dir
    output = root / "outputs" / "calib_observability" / config.lidar if output is None else Path(output).expanduser()
    if not output.is_absolute():
        output = root / output
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    if config.verbose >= 1:
        print(f"Loading KAIST {config.lidar} LiDAR and IMU...", flush=True)
    prepared = load_observability_inputs(config)
    for name, applied in prepared.metadata["sample_caps_applied"].items():
        if applied and config.verbose >= 1:
            print(f"{name}: sample cap reduced effective rate to {prepared.metadata['effective_rates_hz'][name]:.3f} Hz", flush=True)
    if config.verbose >= 2:
        print(json.dumps(jsonable(prepared.metadata), indent=2), flush=True)
    dataset = prepared.dataset
    provider = estimate_poses_dummy(dataset)
    lidar_rate = prepared.metadata["effective_rates_hz"]["lidar"]
    options = AccelerometerOptions(
        mode="simple", factor_rate_hz=lidar_rate, support_half_width_seconds=0.2,
        gravity_norm_tolerance_m_s2=0.75, low_dynamic_gyro_threshold_rad_s=0.35,
        require_low_dynamic_gate=True, measurement_std_m_s2=config.simple_accel_noise_std,
        save_factor_terms=True,
    )

    # Analyze once; the same series drives PNG, CSV and MP4 exports.
    if config.verbose >= 1:
        print(f"Computing at most {config.max_analysis_windows} analysis windows...", flush=True)
    series = run_rolling_observability_analysis(
        dataset, provider, window_duration=config.window_size, window_step=config.step_size,
        max_analysis_windows=config.max_analysis_windows, fixed_extrinsic="T_B_L",
        verbose=config.verbose, n_processes=config.n_processes,
        accelerometer_options=options, jacobian_options=JacobianOptions(method="analytic"),
        use_sparse=config.use_sparse, lidar_rate_hz=lidar_rate,
        tau_target_std_seconds=1.0 / lidar_rate, normalization="physical_then_column",
        max_display_rows=300, max_display_cols=40,
    )
    valid = sum(snapshot.is_valid for snapshot in series.snapshots)
    if not valid:
        raise RuntimeError("No valid observability windows; increase the window size or retained sensor counts.")
    if config.verbose >= 1:
        print(f"Computed {len(series.snapshots)} windows ({valid} valid). Saving results...", flush=True)
    artifacts = {
        "overview": plot_rover_dataset_overview(dataset, output, trajectory_samples=config.trajectory_samples),
        "observability": plot_observability_over_time(series, output),
        "accuracy": plot_local_crlb_accuracy(series, output),
        "trajectory_comparison": _save_comparison(prepared, output, config.trajectory_samples),
        "diagnostics_csv": _save_diagnostic_csv(series, output / "observability_windows.csv"),
    }

    # The subprocess receives capped lightweight snapshots and array-backed
    # trajectories. It produces MP4 only and never embeds an HTML animation.
    artifacts["animation"] = save_quasi_realtime_rover_animation_mp4_subprocess(
        dataset, series.snapshots, output / "observability_dashboard.mp4",
        display_variables=SUPPORTED_CALIBRATION_VARIABLES,
        reference_trajectory=prepared.reference_trajectory,
        trajectory_samples=config.trajectory_samples,
        max_rendered_frames=config.max_rendered_frames,
        mp4_fps=config.mp4_fps, mp4_dpi=config.mp4_dpi,
        verbose=config.verbose,
        interval_ms=max(1, round(1000 / config.mp4_fps)), figsize=(17, 10),
    )
    summary = {
        **prepared.metadata,
        "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "output_dir": str(output), "analysis_window_count": len(series.snapshots),
        "valid_window_count": valid, "analysis_times_s": series.times,
        "rendered_frame_count": min(len(series.snapshots), config.max_rendered_frames),
        "artifacts": artifacts,
    }

    # Paths are converted explicitly; no full dataset or factor graph is copied.
    def paths_to_strings(value):
        """Convert nested artifact paths to strings for JSON serialization.

        Args:
            value: Path, dictionary, or scalar artifact metadata.

        Returns:
            The value with any nested Paths represented as strings.
        """
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: paths_to_strings(item) for key, item in value.items()}
        return value

    (output / "run_summary.json").write_text(
        json.dumps(jsonable(paths_to_strings(summary)), indent=2), encoding="utf-8",
    )
    plt.close("all")
    if config.verbose >= 1:
        print(f"Saved results: {output}", flush=True)
    return output
