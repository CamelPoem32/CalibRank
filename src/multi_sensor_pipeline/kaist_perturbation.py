"""Reusable KAIST calibration-perturbation experiment helpers.

The project transform convention is ``T_A_B`` maps points from frame ``B`` to
frame ``A``. Absolute LiDAR map-pose observations in notebook 21 are sensor
poses ``T_W_L``. To inject an artificial LiDAR calibration ``T_B_L(t)`` while
keeping the same underlying body trajectory, this module first computes
``T_W_B = T_W_L_reference @ inv(T_B_L_reference)`` and then emits perturbed
measurements ``T_W_L_perturbed = T_W_B @ T_B_L_truth(t)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import warnings

import mrob
import numpy as np
import pandas as pd

from calib_observability.sensor_emulation import (
    IMUCalibrationEmulationResult,
    SE3CalibrationLaw,
    ScalarCalibrationLaw,
    emulate_time_varying_imu_calibration,
    evaluate_truth_at_window_midpoints,
    warp_sensor_timestamps,
)
from kaist_dataset.lidar_map import load_lidar_map_pose_csv

from .rolling_graph import RollingGraph, SolverConfig, TrajectoryConfig
from .sensors import Sensor
from .streams import ComplexAccelStream, GyroStream, LidarPoseStream, SimpleAccelStream
from .variables import VariableConfig, VariableKey, VariableType


@dataclass(frozen=True)
class LidarPoseObservations:
    """Absolute LiDAR map-pose measurements loaded from one CSV."""

    sensor_id: str
    path: Path
    timestamps_s: np.ndarray
    poses_T_W_L: np.ndarray
    dataframe: pd.DataFrame


@dataclass(frozen=True)
class PerturbedLidarPoseObservations:
    """Absolute LiDAR map-pose measurements after calibration injection."""

    sensor_id: str
    timestamps_s: np.ndarray
    poses_T_W_L: np.ndarray
    reference_timestamps_s: np.ndarray
    T_B_L_truth: np.ndarray
    tau_L_truth: np.ndarray
    T_B_L_reference: np.ndarray
    tau_L_reference: float


@dataclass(frozen=True)
class KaistPerturbationLaws:
    """Calibration laws used to synthesize perturbed KAIST streams."""

    T_B_I_law: SE3CalibrationLaw
    tau_I_law: ScalarCalibrationLaw
    T_B_LL_law: SE3CalibrationLaw
    tau_LL_law: ScalarCalibrationLaw
    T_B_RL_law: SE3CalibrationLaw | None = None
    tau_RL_law: ScalarCalibrationLaw | None = None


@dataclass(frozen=True)
class PerturbedKaistStreams:
    """Pipeline streams and perturbed arrays for one KAIST experiment."""

    streams: list[Any]
    sensors: list[Sensor]
    emulated_imu: IMUCalibrationEmulationResult
    lidar_left: PerturbedLidarPoseObservations
    lidar_right: PerturbedLidarPoseObservations | None
    pose_timestamps_s: np.ndarray
    initial_body_poses_T_W_B: np.ndarray


@dataclass
class KaistPerturbedGraphRun:
    """Result of one fixed/free rolling graph run."""

    label: str
    fixed_calibration: bool
    graph: RollingGraph
    results: list[Any]
    estimated_timestamps_s: np.ndarray
    estimated_poses_T_W_B: np.ndarray
    window_size_s: float
    step_size_s: float


def build_kaist_perturbation_laws(
    *,
    t_start_s: float,
    t_end_s: float,
    T_B_I_reference: np.ndarray,
    tau_I_reference: float,
    T_B_LL_reference: np.ndarray,
    tau_LL_reference: float = 0.0,
    T_B_RL_reference: np.ndarray | None = None,
    tau_RL_reference: float = 0.0,
    T_B_I_mode: str = "linear",
    T_B_I_end_delta: Sequence[float] | None = None,
    tau_I_mode: str = "linear",
    tau_I_end: float | None = None,
    T_B_L_mode: str = "linear",
    T_B_LL_end_delta: Sequence[float] | None = None,
    T_B_RL_end_delta: Sequence[float] | None = None,
    tau_L_mode: str = "linear",
    tau_LL_end: float | None = None,
    tau_RL_end: float | None = None,
) -> KaistPerturbationLaws:
    """Build notebook-15-style calibration laws for KAIST IMU and LiDAR streams."""

    t_start = float(t_start_s)
    t_end = float(t_end_s)
    if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
        raise ValueError("t_start_s and t_end_s must be finite with t_end_s > t_start_s")

    T_B_I_delta = _default_delta(T_B_I_end_delta, [np.deg2rad(3.0), np.deg2rad(-2.0), np.deg2rad(5.0), 0.05, -0.02, 0.03])
    T_B_LL_delta = _default_delta(T_B_LL_end_delta, [np.deg2rad(2.0), np.deg2rad(-1.0), np.deg2rad(1.5), 0.10, 0.00, 0.03])
    T_B_RL_delta = _default_delta(T_B_RL_end_delta, [np.deg2rad(-2.0), np.deg2rad(1.0), np.deg2rad(-1.5), -0.10, 0.00, 0.03])

    T_B_I_law = _make_se3_law(T_B_I_mode, t_start, t_end, T_B_I_reference, T_B_I_delta)
    tau_I_law = _make_scalar_law(tau_I_mode, t_start, t_end, float(tau_I_reference), float(tau_I_reference if tau_I_end is None else tau_I_end))
    T_B_LL_law = _make_se3_law(T_B_L_mode, t_start, t_end, T_B_LL_reference, T_B_LL_delta)
    tau_LL_law = _make_scalar_law(tau_L_mode, t_start, t_end, float(tau_LL_reference), float(tau_LL_reference if tau_LL_end is None else tau_LL_end))

    T_B_RL_law = None
    tau_RL_law = None
    if T_B_RL_reference is not None:
        T_B_RL_law = _make_se3_law(T_B_L_mode, t_start, t_end, T_B_RL_reference, T_B_RL_delta)
        tau_RL_law = _make_scalar_law(tau_L_mode, t_start, t_end, float(tau_RL_reference), float(tau_RL_reference if tau_RL_end is None else tau_RL_end))

    return KaistPerturbationLaws(T_B_I_law=T_B_I_law, tau_I_law=tau_I_law, T_B_LL_law=T_B_LL_law, tau_LL_law=tau_LL_law, T_B_RL_law=T_B_RL_law, tau_RL_law=tau_RL_law)


def load_kaist_lidar_map_pose_observations(
    left_csv_path: str | Path,
    right_csv_path: str | Path | None = None,
    *,
    left_sensor_id: str = "lidar_0",
    right_sensor_id: str = "lidar_1",
    stride: int = 1,
    max_poses: int | None = None,
    require_left: bool = True,
    require_right: bool = False,
) -> tuple[LidarPoseObservations | None, LidarPoseObservations | None]:
    """Load left/right notebook-21 LiDAR map-pose CSVs and keep successful finite rows."""

    left = _load_one_lidar_pose_csv(left_csv_path, left_sensor_id, stride=stride, max_poses=max_poses, required=require_left)
    right = None if right_csv_path is None else _load_one_lidar_pose_csv(right_csv_path, right_sensor_id, stride=stride, max_poses=max_poses, required=require_right)
    return left, right


def perturb_lidar_sensor_poses(
    timestamps_s: Sequence[float],
    poses_T_W_L_reference: Sequence[np.ndarray],
    *,
    T_B_L_reference: np.ndarray,
    T_B_L_law: SE3CalibrationLaw,
    tau_L_reference: float,
    tau_L_law: ScalarCalibrationLaw,
    sensor_id: str = "lidar_0",
) -> PerturbedLidarPoseObservations:
    """Perturb absolute sensor poses while preserving the underlying body trajectory.

    With ``T_A_B`` convention:
    ``T_W_B = T_W_L_reference @ inv(T_B_L_reference)`` and
    ``T_W_L_perturbed = T_W_B @ T_B_L_truth(t)``.
    """

    timestamps = _as_timestamps(timestamps_s, "timestamps_s")
    poses = _as_pose_stack(poses_T_W_L_reference, "poses_T_W_L_reference")
    if poses.shape[0] != timestamps.size:
        raise ValueError("poses_T_W_L_reference must contain one pose per timestamp")
    T_ref = _as_pose(T_B_L_reference, "T_B_L_reference")
    reference_timestamps, perturbed_timestamps, tau_truth = warp_sensor_timestamps(timestamps, float(tau_L_reference), tau_L_law)
    T_truth = _as_pose_stack(T_B_L_law(reference_timestamps), "T_B_L_law(reference_timestamps)")
    T_W_B = poses @ np.linalg.inv(T_ref)
    poses_perturbed = T_W_B @ T_truth
    return PerturbedLidarPoseObservations(sensor_id=str(sensor_id), timestamps_s=perturbed_timestamps, poses_T_W_L=poses_perturbed, reference_timestamps_s=reference_timestamps, T_B_L_truth=T_truth, tau_L_truth=tau_truth, T_B_L_reference=T_ref, tau_L_reference=float(tau_L_reference))


def build_perturbed_kaist_streams(
    *,
    imu_timestamps_s: Sequence[float],
    gyroscope_radps: np.ndarray,
    accelerometer_mps2: np.ndarray,
    lidar_left: LidarPoseObservations,
    laws: KaistPerturbationLaws,
    T_B_I_reference: np.ndarray,
    tau_I_reference: float,
    T_B_LL_reference: np.ndarray,
    lidar_right: LidarPoseObservations | None = None,
    T_B_RL_reference: np.ndarray | None = None,
    tau_LL_reference: float = 0.0,
    tau_RL_reference: float = 0.0,
    imu_sensor_id: str = "imu_0",
    accel_mode: str = "simple",
    imu_samples_per_factor: int | None = None,
    lidar_samples_per_factor: int | None = None,
    imu_time_offset_margin: float = 5.0,
    lidar_time_offset_margin: float = 5.0,
    gyro_information: Any = 1.0,
    accel_information: Any = 1.0,
    lidar_pose_information: Any = 1.0,
    lidar_factor_stride: int = 1,
    gravity_world: Sequence[float] = (0.0, 0.0, 9.81),
    bias_reference: Sequence[float] | None = None,
    include_lever_arm_correction: bool = False,
    angular_acceleration_smoothing_window: int | None = None,
) -> PerturbedKaistStreams:
    """Build perturbed IMU and absolute LiDAR pose streams for ``RollingGraph``."""

    emulated_imu = emulate_time_varying_imu_calibration(
        sensor_timestamps=imu_timestamps_s,
        gyroscope=gyroscope_radps,
        accelerometer=accelerometer_mps2,
        T_B_I_reference=T_B_I_reference,
        tau_I_reference=tau_I_reference,
        T_B_I_law=laws.T_B_I_law,
        tau_I_law=laws.tau_I_law,
        bias_reference=np.zeros(3) if bias_reference is None else bias_reference,
        include_lever_arm_correction=include_lever_arm_correction,
        angular_acceleration_smoothing_window=angular_acceleration_smoothing_window,
    )

    perturbed_left = perturb_lidar_sensor_poses(lidar_left.timestamps_s, lidar_left.poses_T_W_L, T_B_L_reference=T_B_LL_reference, T_B_L_law=laws.T_B_LL_law, tau_L_reference=tau_LL_reference, tau_L_law=laws.tau_LL_law, sensor_id=lidar_left.sensor_id)
    perturbed_right = None
    if lidar_right is not None:
        if T_B_RL_reference is None or laws.T_B_RL_law is None or laws.tau_RL_law is None:
            raise ValueError("Right LiDAR observations require T_B_RL_reference, T_B_RL_law, and tau_RL_law")
        perturbed_right = perturb_lidar_sensor_poses(lidar_right.timestamps_s, lidar_right.poses_T_W_L, T_B_L_reference=T_B_RL_reference, T_B_L_law=laws.T_B_RL_law, tau_L_reference=tau_RL_reference, tau_L_law=laws.tau_RL_law, sensor_id=lidar_right.sensor_id)

    imu_sensor = Sensor(imu_sensor_id, kind="imu")
    left_sensor = Sensor(lidar_left.sensor_id, kind="lidar")
    sensors = [imu_sensor, left_sensor]
    streams: list[Any] = [
        GyroStream(imu_sensor, emulated_imu.sensor_timestamps, emulated_imu.gyroscope, samples_per_factor=imu_samples_per_factor, time_offset_margin=imu_time_offset_margin, information=gyro_information),
    ]
    if str(accel_mode).lower() == "complex":
        streams.append(ComplexAccelStream(sensor=imu_sensor, timestamps=emulated_imu.sensor_timestamps, acceleration=emulated_imu.accelerometer, angular_velocity=emulated_imu.gyroscope, samples_per_factor=imu_samples_per_factor, time_offset_margin=imu_time_offset_margin, gravity_world=gravity_world, information=accel_information))
    else:
        streams.append(SimpleAccelStream(sensor=imu_sensor, timestamps=emulated_imu.sensor_timestamps, acceleration=emulated_imu.accelerometer, samples_per_factor=imu_samples_per_factor, time_offset_margin=imu_time_offset_margin, gravity_world=gravity_world, information=accel_information))
    streams.append(LidarPoseStream(left_sensor, perturbed_left.timestamps_s, perturbed_left.poses_T_W_L, samples_per_factor=lidar_samples_per_factor, time_offset_margin=lidar_time_offset_margin, information=lidar_pose_information, factor_stride=lidar_factor_stride, stream_name=f"{lidar_left.sensor_id}.perturbed_map_pose"))

    if perturbed_right is not None:
        right_sensor = Sensor(perturbed_right.sensor_id, kind="lidar")
        sensors.append(right_sensor)
        streams.append(LidarPoseStream(right_sensor, perturbed_right.timestamps_s, perturbed_right.poses_T_W_L, samples_per_factor=lidar_samples_per_factor, time_offset_margin=lidar_time_offset_margin, information=lidar_pose_information, factor_stride=lidar_factor_stride, stream_name=f"{perturbed_right.sensor_id}.perturbed_map_pose"))

    pose_timestamps = _overlap_pose_timestamps(perturbed_left.timestamps_s, emulated_imu.sensor_timestamps, None if perturbed_right is None else perturbed_right.timestamps_s)
    initial_poses = initial_body_poses_from_lidar_sensor_poses(pose_timestamps, perturbed_left.timestamps_s, perturbed_left.poses_T_W_L, T_B_LL_reference)
    return PerturbedKaistStreams(streams=streams, sensors=sensors, emulated_imu=emulated_imu, lidar_left=perturbed_left, lidar_right=perturbed_right, pose_timestamps_s=pose_timestamps, initial_body_poses_T_W_B=initial_poses)


def build_kaist_variable_configs(
    fixed: bool,
    *,
    imu_sensor_id: str = "imu_0",
    left_lidar_sensor_id: str = "lidar_0",
    right_lidar_sensor_id: str | None = None,
    T_B_I_initial: np.ndarray | None = None,
    tau_I_initial: float = 0.0,
    gyro_bias_initial: Sequence[float] = (0.0, 0.0, 0.0),
    T_B_LL_initial: np.ndarray | None = None,
    tau_LL_initial: float = 0.0,
    T_B_RL_initial: np.ndarray | None = None,
    tau_RL_initial: float = 0.0,
    prior_source: str = "optimized",
    T_B_I_prior_information: Any = None,
    tau_I_prior_information: Any = None,
    gyro_bias_prior_information: Any = None,
    T_B_L_prior_information: Any = None,
    tau_L_prior_information: Any = None,
    initial_source="optimized",
) -> dict[VariableKey, VariableConfig]:
    """Create fixed or free calibration configs for KAIST IMU and LiDAR streams.

    If ``initial_source`` or ``prior_source`` is ``"numerical"``, the numerical source is applied only to IMU variables. LiDAR calibration variables keep optimized/configured sources because the current numerical estimator uses LiDAR poses as reference motion but estimates ``T_B_I``, ``tau_I``, and gyro bias only.
    """

    imu_initial_source = "constant" if fixed else initial_source
    lidar_initial_source = "constant" if fixed else _non_numerical_lidar_source(initial_source)
    lidar_prior_source = _non_numerical_lidar_source(prior_source)
    configs: dict[VariableKey, VariableConfig] = {
        VariableKey(imu_sensor_id, VariableType.EXTRINSIC): _variable_config(imu_initial_source, np.eye(4) if T_B_I_initial is None else T_B_I_initial, fixed, prior_source, T_B_I_prior_information),
        VariableKey(imu_sensor_id, VariableType.TIME_OFFSET): _variable_config(imu_initial_source, float(tau_I_initial), fixed, prior_source, tau_I_prior_information),
        VariableKey(imu_sensor_id, VariableType.GYRO_BIAS): _variable_config(imu_initial_source, gyro_bias_initial, fixed, prior_source, gyro_bias_prior_information),
        VariableKey(left_lidar_sensor_id, VariableType.EXTRINSIC): _variable_config(lidar_initial_source, np.eye(4) if T_B_LL_initial is None else T_B_LL_initial, fixed, lidar_prior_source, T_B_L_prior_information),
        VariableKey(left_lidar_sensor_id, VariableType.TIME_OFFSET): _variable_config(lidar_initial_source, float(tau_LL_initial), fixed, lidar_prior_source, tau_L_prior_information),
    }
    if right_lidar_sensor_id is not None:
        configs[VariableKey(right_lidar_sensor_id, VariableType.EXTRINSIC)] = _variable_config(lidar_initial_source, np.eye(4) if T_B_RL_initial is None else T_B_RL_initial, fixed, lidar_prior_source, T_B_L_prior_information)
        configs[VariableKey(right_lidar_sensor_id, VariableType.TIME_OFFSET)] = _variable_config(lidar_initial_source, float(tau_RL_initial), fixed, lidar_prior_source, tau_L_prior_information)
    return configs


def run_kaist_perturbed_calibration_graph(
    *,
    label: str,
    fixed_calibration: bool,
    perturbed_streams: PerturbedKaistStreams,
    variable_configs: Mapping[Any, VariableConfig],
    window_size: float,
    solver_config: SolverConfig | None = None,
    trajectory_config: TrajectoryConfig | None = None,
    step_size: float | None = None,
    window_scale: float = 100.0,
    step_fraction: float = 0.5,
    clear_previous: bool = True,
    verbose: int = 0,
    debug: bool = False,
) -> KaistPerturbedGraphRun:
    """Run one KAIST rolling graph with the notebook-19 ``WINDOW_SIZE * 100`` convention."""

    rolling_window_size = float(window_size) * float(window_scale)
    rolling_step_size = rolling_window_size * float(step_fraction) if step_size is None else float(step_size)
    graph = RollingGraph(streams=perturbed_streams.streams, sensors=perturbed_streams.sensors, variable_configs=variable_configs, solver_config=SolverConfig() if solver_config is None else solver_config, trajectory_config=TrajectoryConfig(anchor_first_pose=True, anchor_first_pose_each_window=False, anchor_last_pose=False, anchor_all_poses=False, use_imu_gyr=False) if trajectory_config is None else trajectory_config, debug=debug)
    results = graph.generate_filter_iterative(window_size=rolling_window_size, step_size=rolling_step_size, pose_timestamps=perturbed_streams.pose_timestamps_s, states=perturbed_streams.initial_body_poses_T_W_B, first_pose=perturbed_streams.initial_body_poses_T_W_B[0], clear_previous=clear_previous, verbose=verbose)
    estimated_timestamps, estimated_poses = graph.rolling_state.rolling_trajectory
    return KaistPerturbedGraphRun(label=str(label), fixed_calibration=bool(fixed_calibration), graph=graph, results=list(results), estimated_timestamps_s=np.asarray(estimated_timestamps, dtype=float), estimated_poses_T_W_B=np.asarray(estimated_poses, dtype=float), window_size_s=rolling_window_size, step_size_s=rolling_step_size)


def summarize_perturbed_calibration_results(runs: Sequence[KaistPerturbedGraphRun], laws: KaistPerturbationLaws | None = None) -> pd.DataFrame:
    """Return one diagnostics row per rolling window for fixed/free runs."""

    rows: list[dict[str, Any]] = []
    for run in runs:
        for result in run.results:
            midpoint = 0.5 * (float(result.window_start) + float(result.window_end))
            row: dict[str, Any] = {"run": run.label, "fixed_calibration": run.fixed_calibration, "window": int(result.window_index), "start": float(result.window_start), "end": float(result.window_end), "midpoint": midpoint, "chi2_before": float(result.chi2_before), "chi2_after": float(result.chi2_after)}
            for key, value in result.calibration_values.items():
                label = key.label.replace(":", "_")
                if key.variable_type == VariableType.EXTRINSIC:
                    row[f"{label}_ln"] = mrob.SE3(np.asarray(value, dtype=float)).Ln()
                elif key.variable_type == VariableType.TIME_OFFSET:
                    row[label] = float(value)
                elif key.variable_type == VariableType.GYRO_BIAS:
                    row[label] = np.asarray(value, dtype=float).reshape(3)
            rows.append(row)
    return pd.DataFrame(rows)


def plot_fixed_free_trajectories(runs: Sequence[KaistPerturbedGraphRun], *, reference_timestamps: Sequence[float] | None = None, reference_poses: Sequence[np.ndarray] | None = None, initial_timestamps: Sequence[float] | None = None, initial_poses: Sequence[np.ndarray] | None = None):
    """Plot fixed/free trajectory estimates against optional reference and initialization."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    if reference_poses is not None:
        ref = np.asarray(reference_poses, dtype=float)
        axes[0].plot(ref[:, 0, 3], ref[:, 1, 3], color="black", linewidth=1.5, label="reference")
        if reference_timestamps is not None:
            axes[1].plot(reference_timestamps, ref[:, 2, 3], color="black", linewidth=1.5, label="reference")
    if initial_poses is not None:
        init = np.asarray(initial_poses, dtype=float)
        axes[0].plot(init[:, 0, 3], init[:, 1, 3], linestyle="--", alpha=0.75, label="initial")
        if initial_timestamps is not None:
            axes[1].plot(initial_timestamps, init[:, 2, 3], linestyle="--", alpha=0.75, label="initial")
    for run in runs:
        poses = np.asarray(run.estimated_poses_T_W_B, dtype=float)
        times = np.asarray(run.estimated_timestamps_s, dtype=float)
        if poses.size == 0:
            continue
        axes[0].plot(poses[:, 0, 3], poses[:, 1, 3], label=run.label)
        axes[1].plot(times, poses[:, 2, 3], label=run.label)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x_W [m]")
    axes[0].set_ylabel("y_W [m]")
    axes[0].set_title("XY trajectory")
    axes[1].set_xlabel("timestamp [s]")
    axes[1].set_ylabel("z_W [m]")
    axes[1].set_title("Height")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    return fig, axes


def plot_fixed_free_chi2(runs: Sequence[KaistPerturbedGraphRun]):
    """Plot rolling chi-square before/after optimization for fixed/free runs."""

    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for run in runs:
        if not run.results:
            continue
        midpoints = np.asarray([0.5 * (result.window_start + result.window_end) for result in run.results], dtype=float)
        axis.plot(midpoints, [result.chi2_before for result in run.results], linestyle="--", marker=".", label=f"{run.label} before")
        axis.plot(midpoints, [result.chi2_after for result in run.results], marker="o", label=f"{run.label} after")
    axis.set_xlabel("window midpoint [s]")
    axis.set_ylabel("chi2")
    axis.set_title("Rolling graph objective")
    axis.grid(True, alpha=0.25)
    axis.legend()
    return fig, axis


def plot_calibration_truth_tracking(runs: Sequence[KaistPerturbedGraphRun], laws: KaistPerturbationLaws, *, imu_sensor_id: str = "imu_0", left_lidar_sensor_id: str = "lidar_0"):
    """Plot selected time-offset estimates against injected truth laws."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True, constrained_layout=True)
    for run in runs:
        if not run.results:
            continue
        midpoints, _, _ = evaluate_truth_at_window_midpoints(run.results, laws.T_B_I_law, laws.tau_I_law)
        tau_I_est = _calibration_series(run.results, VariableKey(imu_sensor_id, VariableType.TIME_OFFSET))
        tau_L_est = _calibration_series(run.results, VariableKey(left_lidar_sensor_id, VariableType.TIME_OFFSET))
        axes[0].plot(midpoints, tau_I_est, marker="o", label=f"{run.label} tau_I")
        axes[1].plot(midpoints, tau_L_est, marker="o", label=f"{run.label} tau_L")
    if runs and runs[0].results:
        midpoints = np.asarray([0.5 * (result.window_start + result.window_end) for result in runs[0].results], dtype=float)
        axes[0].plot(midpoints, laws.tau_I_law(midpoints), color="black", linestyle="--", label="tau_I truth")
        axes[1].plot(midpoints, laws.tau_LL_law(midpoints), color="black", linestyle="--", label="tau_L left truth")
    axes[0].set_ylabel("tau_I [s]")
    axes[1].set_ylabel("tau_L [s]")
    axes[1].set_xlabel("window midpoint [s]")
    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend()
    return fig, axes


def initial_body_poses_from_lidar_sensor_poses(query_timestamps_s: Sequence[float], lidar_timestamps_s: Sequence[float], poses_T_W_L: Sequence[np.ndarray], T_B_L_initial: np.ndarray) -> np.ndarray:
    """Interpolate LiDAR sensor poses and convert them to body-pose initial states."""

    query_timestamps = _as_timestamps(query_timestamps_s, "query_timestamps_s")
    lidar_timestamps = _as_timestamps(lidar_timestamps_s, "lidar_timestamps_s")
    lidar_poses = _as_pose_stack(poses_T_W_L, "poses_T_W_L")
    if lidar_poses.shape[0] != lidar_timestamps.size:
        raise ValueError("poses_T_W_L must contain one pose per lidar timestamp")
    T_L_B = np.linalg.inv(_as_pose(T_B_L_initial, "T_B_L_initial"))
    return np.stack([_interpolate_pose(lidar_timestamps, lidar_poses, timestamp) @ T_L_B for timestamp in query_timestamps], axis=0)


def _load_one_lidar_pose_csv(path: str | Path, sensor_id: str, *, stride: int, max_poses: int | None, required: bool) -> LidarPoseObservations | None:
    csv_path = Path(path).expanduser()
    if not csv_path.is_absolute():
        csv_path = Path.cwd() / csv_path
    if not csv_path.is_file():
        if required:
            raise FileNotFoundError(f"LiDAR map pose CSV does not exist: {csv_path}")
        warnings.warn(f"LiDAR map pose CSV not found: {csv_path}; skipping {sensor_id}", RuntimeWarning, stacklevel=2)
        return None
    timestamps, poses, dataframe = load_lidar_map_pose_csv(csv_path)
    success = dataframe["success"].to_numpy(dtype=bool) if "success" in dataframe else np.ones(len(dataframe), dtype=bool)
    finite = np.all(np.isfinite(poses[:, :3, :]), axis=(1, 2)) if len(poses) else np.zeros(0, dtype=bool)
    indices = np.flatnonzero(success & finite)[:: max(int(stride), 1)]
    if max_poses is not None:
        indices = indices[: int(max_poses)]
    return LidarPoseObservations(sensor_id=str(sensor_id), path=csv_path, timestamps_s=timestamps[indices], poses_T_W_L=poses[indices], dataframe=dataframe.iloc[indices].reset_index(drop=True))


def _make_se3_law(mode: str, t_start: float, t_end: float, reference: np.ndarray, end_delta: np.ndarray) -> SE3CalibrationLaw:
    mode = str(mode).lower()
    if mode == "constant":
        return SE3CalibrationLaw("constant", reference=reference@mrob.SE3(end_delta).T())
    if mode == "linear":
        return SE3CalibrationLaw("linear", t_start=t_start, t_end=t_end, reference=reference, end_delta_xi=end_delta)
    if mode == "sinusoidal":
        return SE3CalibrationLaw("sinusoidal", t_start=t_start, t_end=t_end, reference=reference, amplitude_xi=end_delta, cycles=1.0)
    if mode == "piecewise":
        control_times = np.linspace(t_start, t_end, 5)
        control_tangents = np.vstack([alpha * end_delta for alpha in np.linspace(0.0, 1.0, 5)])
        return SE3CalibrationLaw("piecewise", reference=reference, control_times=control_times, control_tangents=control_tangents)
    raise ValueError(f"Unsupported SE(3) law mode: {mode}")


def _make_scalar_law(mode: str, t_start: float, t_end: float, start: float, end: float) -> ScalarCalibrationLaw:
    mode = str(mode).lower()
    if mode == "constant":
        return ScalarCalibrationLaw("constant", value=end)
    if mode == "linear":
        return ScalarCalibrationLaw("linear", t_start=t_start, t_end=t_end, start=start, end=end)
    if mode == "sinusoidal":
        return ScalarCalibrationLaw("sinusoidal", t_start=t_start, t_end=t_end, value=start, amplitude=end - start, cycles=1.0)
    if mode == "piecewise":
        return ScalarCalibrationLaw("piecewise", control_times=np.linspace(t_start, t_end, 5), control_values=np.linspace(start, end, 5))
    raise ValueError(f"Unsupported scalar law mode: {mode}")


def _non_numerical_lidar_source(source: Any) -> Any:
    source_text = str(getattr(source, "value", source)).strip().lower()
    return "optimized" if source_text == "numerical" else source


def _variable_config(initial_source: str, initial_value: Any, fixed: bool, prior_source: str, prior_information: Any) -> VariableConfig:
    if prior_information is None or fixed:
        return VariableConfig(initial_source=initial_source, initial_value=initial_value, fixed=fixed).normalized()
    return VariableConfig(initial_source=initial_source, initial_value=initial_value, fixed=fixed, prior_source=prior_source, prior_value=initial_value, prior_information=prior_information).normalized()


def _overlap_pose_timestamps(left_timestamps: np.ndarray, imu_timestamps: np.ndarray, right_timestamps: np.ndarray | None = None) -> np.ndarray:
    start = max(float(left_timestamps[0]), float(imu_timestamps[0]))
    end = min(float(left_timestamps[-1]), float(imu_timestamps[-1]))
    if right_timestamps is not None and right_timestamps.size:
        start = max(start, float(right_timestamps[0]))
        end = min(end, float(right_timestamps[-1]))
    mask = (left_timestamps >= start) & (left_timestamps <= end)
    timestamps = left_timestamps[mask]
    if timestamps.size < 2:
        raise ValueError("Fewer than two left LiDAR pose timestamps overlap IMU/right-LiDAR support")
    return timestamps.copy()


def _interpolate_pose(timestamps: np.ndarray, poses: np.ndarray, query_time: float) -> np.ndarray:
    if query_time <= timestamps[0]:
        return poses[0].copy()
    if query_time >= timestamps[-1]:
        return poses[-1].copy()
    upper = int(np.searchsorted(timestamps, query_time, side="left"))
    if np.isclose(timestamps[upper], query_time):
        return poses[upper].copy()
    lower = upper - 1
    alpha = (float(query_time) - timestamps[lower]) / (timestamps[upper] - timestamps[lower])
    T_lower = mrob.SE3(poses[lower])
    T_upper = mrob.SE3(poses[upper])
    return np.asarray(T_lower.mul(mrob.SE3(alpha * T_lower.inv().mul(T_upper).Ln())).T(), dtype=float)


def _calibration_series(results: Sequence[Any], key: VariableKey) -> np.ndarray:
    values = []
    for result in results:
        value = result.calibration_value(key)
        values.append(np.nan if value is None else float(value))
    return np.asarray(values, dtype=float)


def _as_timestamps(values: Sequence[float], name: str) -> np.ndarray:
    timestamps = np.asarray(values, dtype=float).reshape(-1)
    if timestamps.ndim != 1 or timestamps.size == 0 or not np.all(np.isfinite(timestamps)):
        raise ValueError(f"{name} must be a non-empty finite one-dimensional array")
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return timestamps.copy()


def _as_pose(value: Any, name: str) -> np.ndarray:
    pose = np.asarray(value, dtype=float)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    return pose.copy()


def _as_pose_stack(values: Any, name: str) -> np.ndarray:
    poses = np.asarray(values, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape (N, 4, 4)")
    if not np.all(np.isfinite(poses)):
        raise ValueError(f"{name} must contain finite transforms")
    return poses.copy()


def _default_delta(value: Sequence[float] | None, default: Sequence[float]) -> np.ndarray:
    delta = np.asarray(default if value is None else value, dtype=float).reshape(6)
    if not np.all(np.isfinite(delta)):
        raise ValueError("Calibration perturbation deltas must be finite")
    return delta


__all__ = [
    "KaistPerturbationLaws",
    "KaistPerturbedGraphRun",
    "LidarPoseObservations",
    "PerturbedKaistStreams",
    "PerturbedLidarPoseObservations",
    "build_kaist_perturbation_laws",
    "load_kaist_lidar_map_pose_observations",
    "perturb_lidar_sensor_poses",
    "build_perturbed_kaist_streams",
    "build_kaist_variable_configs",
    "run_kaist_perturbed_calibration_graph",
    "summarize_perturbed_calibration_results",
    "plot_fixed_free_trajectories",
    "plot_fixed_free_chi2",
    "plot_calibration_truth_tracking",
    "initial_body_poses_from_lidar_sensor_poses",
]
