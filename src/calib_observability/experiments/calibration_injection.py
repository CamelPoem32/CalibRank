"""Utilities for controlled real-data calibration injection experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import mrob
import numpy as np

from src.factor_graph_calibration import CalibrationWindowResult, FactorGraphCalibration
from src.new_college_dataset.data import IMUData, LidarData
from src.transform import relative_se3_to_se3


@dataclass(frozen=True)
class CalibrationInjectionConfig:
    """Describe deterministic perturbations injected into one real IMU stream.

    Args:
        gyro_bias_delta_radps: Additive gyroscope bias in rad/s, shape ``(3,)``.
        tau_I_delta_s: Positive IMU reference-clock offset to inject. The
            sensor timestamps are shifted by ``-tau_I_delta_s`` so the factor
            graph should recover a positive ``tau_I`` delta.
        R_I0_I1: Optional virtual-frame rotation. It maps vectors from the
            injected IMU frame ``I1`` into the original IMU frame ``I0``.
        metadata: Free-form experiment notes stored in copied IMU metadata.
    """

    gyro_bias_delta_radps: Sequence[float] = (0.0, 0.0, 0.0)
    tau_I_delta_s: float = 0.0
    R_I0_I1: Sequence[Sequence[float]] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bias = np.asarray(self.gyro_bias_delta_radps, dtype=float).reshape(3)
        if not np.all(np.isfinite(bias)):
            raise ValueError("gyro_bias_delta_radps must contain finite values")

        tau = float(self.tau_I_delta_s)
        if not np.isfinite(tau):
            raise ValueError("tau_I_delta_s must be finite")

        rotation = None
        if self.R_I0_I1 is not None:
            rotation = np.asarray(self.R_I0_I1, dtype=float)
            if rotation.shape != (3, 3):
                raise ValueError("R_I0_I1 must have shape (3, 3)")
            if not np.all(np.isfinite(rotation)):
                raise ValueError("R_I0_I1 must contain finite values")
            if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
                raise ValueError("R_I0_I1 must be an orthonormal rotation matrix")
            if np.linalg.det(rotation) < 0.0:
                raise ValueError("R_I0_I1 must be a proper rotation")

        object.__setattr__(self, "gyro_bias_delta_radps", bias)
        object.__setattr__(self, "tau_I_delta_s", tau)
        object.__setattr__(self, "R_I0_I1", rotation)


@dataclass(frozen=True)
class FactorGraphExperimentResult:
    """Store one calibration run and its rolling trajectory.

    Args:
        label: Short run label, for example ``"baseline"`` or ``"injected"``.
        graph: Solved factor-graph object. Its ``rolling_results`` and
            ``rolling_trajectory`` properties remain available for inspection.
        rolling_results: Window-level calibration results returned by
            ``generate_filter_iterative``.
        trajectory_timestamps_s: Stitched rolling trajectory timestamps, shape
            ``(N,)``.
        trajectory_poses_se3: Stitched rolling trajectory poses, shape
            ``(N, 4, 4)``.
    """

    label: str
    graph: FactorGraphCalibration
    rolling_results: list[CalibrationWindowResult]
    trajectory_timestamps_s: np.ndarray
    trajectory_poses_se3: np.ndarray


def copy_imu_data(imu_data: IMUData, *, name: str | None = None, metadata: Mapping[str, Any] | None = None) -> IMUData:
    """Return an independent copy of an ``IMUData`` object.

    Args:
        imu_data: Source IMU stream.
        name: Optional replacement stream name.
        metadata: Metadata entries merged into the copied stream metadata.

    Returns:
        Copied ``IMUData`` with independent NumPy arrays.
    """
    copied_metadata = dict(imu_data.metadata)
    if metadata is not None:
        copied_metadata.update(dict(metadata))

    return IMUData(
        timestamps_s=imu_data.timestamps_s.copy(),
        accel_mps2=imu_data.accel_mps2.copy(),
        gyro_radps=imu_data.gyro_radps.copy(),
        name=name or imu_data.name,
        frame_id=imu_data.frame_id,
        frequency_hz=imu_data.frequency_hz,
        metadata=copied_metadata,
    )


def apply_gyroscope_bias_delta(imu_data: IMUData, gyro_bias_delta_radps: Sequence[float]) -> IMUData:
    """Add a constant gyroscope bias to an IMU stream.

    Args:
        imu_data: Source IMU stream.
        gyro_bias_delta_radps: Constant angular-rate delta in rad/s, shape
            ``(3,)``.

    Returns:
        New ``IMUData`` whose gyroscope samples are ``gyro + delta``.
    """
    delta = np.asarray(gyro_bias_delta_radps, dtype=float).reshape(3)
    if not np.all(np.isfinite(delta)):
        raise ValueError("gyro_bias_delta_radps must contain finite values")

    return IMUData(
        timestamps_s=imu_data.timestamps_s.copy(),
        accel_mps2=imu_data.accel_mps2.copy(),
        gyro_radps=imu_data.gyro_radps + delta[None, :],
        name=f"{imu_data.name}_gyro_bias_injected",
        frame_id=imu_data.frame_id,
        frequency_hz=imu_data.frequency_hz,
        metadata={**imu_data.metadata, "gyro_bias_delta_radps": delta.copy()},
    )


def apply_imu_time_offset_delta(imu_data: IMUData, tau_I_delta_s: float) -> IMUData:
    """Inject an IMU clock offset using the project timestamp convention.

    Args:
        imu_data: Source IMU stream.
        tau_I_delta_s: Desired positive reference-clock offset in seconds.

    Returns:
        New ``IMUData`` with ``timestamps_s - tau_I_delta_s``. With the project
        convention ``t_reference = t_sensor + tau_I``, the injected stream
        satisfies ``t_injected + tau_I_delta_s == t_original``.
    """
    tau = float(tau_I_delta_s)
    if not np.isfinite(tau):
        raise ValueError("tau_I_delta_s must be finite")

    shifted_timestamps = imu_data.timestamps_s - tau
    if shifted_timestamps.size > 1 and np.any(np.diff(shifted_timestamps) <= 0.0):
        raise ValueError("time-offset injection produced non-increasing timestamps")

    return IMUData(
        timestamps_s=shifted_timestamps,
        accel_mps2=imu_data.accel_mps2.copy(),
        gyro_radps=imu_data.gyro_radps.copy(),
        name=f"{imu_data.name}_tau_injected",
        frame_id=imu_data.frame_id,
        frequency_hz=imu_data.frequency_hz,
        metadata={**imu_data.metadata, "tau_I_delta_s": tau},
    )


def apply_virtual_imu_frame_rotation(imu_data: IMUData, R_I0_I1: Sequence[Sequence[float]]) -> IMUData:
    """Express IMU vector measurements in a deterministically rotated frame.

    Args:
        imu_data: Source stream whose vectors are expressed in frame ``I0``.
        R_I0_I1: Rotation mapping coordinates from injected frame ``I1`` into
            original frame ``I0``.

    Returns:
        New ``IMUData`` with row-vector measurements transformed as
        ``v_I1 = v_I0 @ R_I0_I1``.
    """
    rotation = np.asarray(R_I0_I1, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError("R_I0_I1 must have shape (3, 3)")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8):
        raise ValueError("R_I0_I1 must be orthonormal")

    # Measurements are stored as row vectors. Since R_I0_I1 maps I1 into I0,
    # multiplying rows by R_I0_I1 expresses the same physical vectors in I1.
    accel_rotated = imu_data.accel_mps2 @ rotation
    gyro_rotated = imu_data.gyro_radps @ rotation

    return IMUData(
        timestamps_s=imu_data.timestamps_s.copy(),
        accel_mps2=accel_rotated,
        gyro_radps=gyro_rotated,
        name=f"{imu_data.name}_frame_injected",
        frame_id=imu_data.frame_id,
        frequency_hz=imu_data.frequency_hz,
        metadata={**imu_data.metadata, "R_I0_I1": rotation.copy()},
    )


def build_injected_imu_data(imu_data: IMUData, config: CalibrationInjectionConfig) -> IMUData:
    """Build a modified IMU stream with all configured perturbations applied.

    Args:
        imu_data: Baseline real IMU stream.
        config: Deterministic injection parameters.

    Returns:
        New ``IMUData`` containing the injected measurements. The input stream
        is never modified.
    """
    injected = copy_imu_data(
        imu_data,
        name=f"{imu_data.name}_injected",
        metadata={"injection_config": dict(config.metadata)},
    )

    # Apply frame rotation before additive gyroscope bias so the bias is in the
    # final injected IMU coordinate frame.
    if config.R_I0_I1 is not None:
        injected = apply_virtual_imu_frame_rotation(injected, config.R_I0_I1)

    if np.linalg.norm(config.gyro_bias_delta_radps) > 0.0:
        injected = apply_gyroscope_bias_delta(injected, config.gyro_bias_delta_radps)

    if config.tau_I_delta_s != 0.0:
        injected = apply_imu_time_offset_delta(injected, config.tau_I_delta_s)

    return copy_imu_data(
        injected,
        name=f"{imu_data.name}_injected",
        metadata={
            "gyro_bias_delta_radps": config.gyro_bias_delta_radps.copy(),
            "tau_I_delta_s": config.tau_I_delta_s,
            "R_I0_I1": None if config.R_I0_I1 is None else config.R_I0_I1.copy(),
            **dict(config.metadata),
        },
    )


def specific_force_from_accelerometer(accel_mps2: Sequence[Sequence[float]], *, sign: float = 1.0) -> np.ndarray:
    """Convert loaded accelerometer samples to factor-graph specific force.

    Args:
        accel_mps2: Loaded accelerometer samples, shape ``(N, 3)``.
        sign: Explicit multiplier. Use ``1.0`` when the data already follows
            the factor convention and ``-1.0`` for sensors with opposite sign.

    Returns:
        Specific-force samples for ``FactorGraphCalibration``, shape ``(N, 3)``.
    """
    accel = np.asarray(accel_mps2, dtype=float)
    if accel.ndim != 2 or accel.shape[1] != 3:
        raise ValueError("accel_mps2 must have shape (N, 3)")
    if not np.all(np.isfinite(accel)):
        raise ValueError("accel_mps2 must contain finite values")

    return float(sign) * accel.copy()


def accumulate_lidar_reference_poses(lidar_data: LidarData, *, first_pose: Any | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate relative LiDAR odometry into a reference trajectory.

    Args:
        lidar_data: LiDAR relative-pose stream.
        first_pose: Optional first absolute pose. Defaults to identity.

    Returns:
        Tuple ``(scan_timestamps_s, reference_poses_se3)`` with shapes ``(N+1,)``
        and ``(N+1, 4, 4)``.
    """
    scan_timestamps = lidar_data.scan_timestamps_s
    if scan_timestamps is None:
        if lidar_data.timestamps_s.size == 0:
            raise ValueError("Cannot infer scan timestamps from empty LiDAR data")
        dt = float(np.median(np.diff(lidar_data.timestamps_s))) if lidar_data.timestamps_s.size > 1 else 0.1
        scan_timestamps = np.concatenate(
            (
                [lidar_data.timestamps_s[0] - 0.5 * dt],
                lidar_data.timestamps_s + 0.5 * dt,
            )
        )

    poses = relative_se3_to_se3(lidar_data.relative_poses_se3, first_se3=first_pose)
    return np.asarray(scan_timestamps, dtype=float).copy(), poses


def prepare_factor_graph_streams(
    imu_data: IMUData,
    lidar_data: LidarData,
    *,
    accelerometer_sign: float = 1.0,
    align_sensor_start_times: bool = True,
    imu_time_origin_s: float | None = None,
    lidar_time_origin_s: float | None = None,
) -> dict[str, np.ndarray]:
    """Prepare real IMU and LiDAR arrays for ``FactorGraphCalibration``.

    Args:
        imu_data: IMU stream used for gyroscope and accelerometer factors.
        lidar_data: Relative LiDAR odometry stream.
        accelerometer_sign: Multiplier applied before passing accelerometer
            samples as specific force.
        align_sensor_start_times: If true, subtract each stream's first
            timestamp independently. This is useful with the current New College
            loaders because IMU timestamps are relative while LiDAR filenames
            may be absolute-like.
        imu_time_origin_s: Optional IMU origin to subtract. Use the baseline
            origin for both baseline and injected runs so a time-offset
            injection is not cancelled by re-zeroing.
        lidar_time_origin_s: Optional LiDAR origin to subtract.

    Returns:
        Dictionary containing timestamps and measurement arrays for one factor
        graph run.
    """
    lidar_scan_timestamps, lidar_reference_poses = accumulate_lidar_reference_poses(lidar_data)
    imu_timestamps = imu_data.timestamps_s.copy()

    # The current New College loaders do not retain a shared absolute origin
    # for IMU and LiDAR, so the notebook can start-align streams explicitly.
    # if align_sensor_start_times:
    #     imu_origin = float(imu_timestamps[0]) if imu_time_origin_s is None else float(imu_time_origin_s)
    #     lidar_origin = float(lidar_scan_timestamps[0]) if lidar_time_origin_s is None else float(lidar_time_origin_s)
    #     imu_timestamps = imu_timestamps - imu_origin
    #     lidar_scan_timestamps = lidar_scan_timestamps - lidar_origin
    # else:
    #     common_origin = min(float(imu_timestamps[0]), float(lidar_scan_timestamps[0]))
    #     imu_timestamps = imu_timestamps - common_origin
    #     lidar_scan_timestamps = lidar_scan_timestamps - common_origin

    return {
        "pose_timestamps": lidar_scan_timestamps.copy(),
        "initial_poses": lidar_reference_poses.copy(),
        "imu_timestamps": imu_timestamps,
        "angular_velocity_imu": imu_data.gyro_radps.copy(),
        "specific_force_imu": specific_force_from_accelerometer(
            imu_data.accel_mps2,
            sign=accelerometer_sign,
        ),
        "lidar_timestamps": lidar_scan_timestamps.copy(),
        "lidar_odometry_poses": lidar_reference_poses.copy(),
    }


def validate_factor_graph_streams(streams: Mapping[str, np.ndarray]) -> None:
    """Validate the array shapes and monotonic timestamps passed to the graph.

    Args:
        streams: Dictionary returned by ``prepare_factor_graph_streams``.

    Returns:
        None. Raises ``ValueError`` when a required field is inconsistent.
    """
    pose_timestamps = np.asarray(streams["pose_timestamps"], dtype=float).reshape(-1)
    initial_poses = np.asarray(streams["initial_poses"], dtype=float)
    imu_timestamps = np.asarray(streams["imu_timestamps"], dtype=float).reshape(-1)
    gyro = np.asarray(streams["angular_velocity_imu"], dtype=float)
    specific_force = np.asarray(streams["specific_force_imu"], dtype=float)
    lidar_timestamps = np.asarray(streams["lidar_timestamps"], dtype=float).reshape(-1)
    lidar_poses = np.asarray(streams["lidar_odometry_poses"], dtype=float)

    for name, timestamps in (("pose_timestamps", pose_timestamps), ("imu_timestamps", imu_timestamps), ("lidar_timestamps", lidar_timestamps)):
        if timestamps.size < 2:
            raise ValueError(f"{name} must contain at least two timestamps")
        if not np.all(np.isfinite(timestamps)) or np.any(np.diff(timestamps) <= 0.0):
            raise ValueError(f"{name} must be finite and strictly increasing")

    if initial_poses.shape != (pose_timestamps.size, 4, 4):
        raise ValueError("initial_poses must have shape (len(pose_timestamps), 4, 4)")
    if lidar_poses.shape != (lidar_timestamps.size, 4, 4):
        raise ValueError("lidar_odometry_poses must have shape (len(lidar_timestamps), 4, 4)")
    if gyro.shape != (imu_timestamps.size, 3):
        raise ValueError("angular_velocity_imu must have shape (len(imu_timestamps), 3)")
    if specific_force.shape != (imu_timestamps.size, 3):
        raise ValueError("specific_force_imu must have shape (len(imu_timestamps), 3)")


def run_factor_graph_calibration_experiment(
    *,
    label: str,
    factor_graph_kwargs: Mapping[str, Any],
    window_size: float,
    step_size: float,
    pose_timestamps: Sequence[float],
    initial_poses: Sequence[Any],
    imu_timestamps: Sequence[float],
    angular_velocity_imu: np.ndarray,
    specific_force_imu: np.ndarray,
    lidar_timestamps: Sequence[float],
    lidar_odometry_poses: Sequence[Any],
    T_B_I_initial: Any,
    T_B_L_initial: Any,
    bias_initial: Sequence[float],
    tau_I_initial: float,
    tau_L_initial: float,
    verbose: int = 0,
    gravity_world=(0, 0, -9.81)
) -> FactorGraphExperimentResult:
    """Run one rolling factor-graph calibration experiment.

    Args:
        label: Human-readable run label.
        factor_graph_kwargs: Constructor arguments for ``FactorGraphCalibration``.
        window_size: Rolling-window duration in seconds.
        step_size: Rolling-window stride in seconds.
        pose_timestamps: Pose-node timestamps, shape ``(N,)``.
        initial_poses: Initial pose states, shape ``(N, 4, 4)``.
        imu_timestamps: IMU sensor timestamps, shape ``(M,)``.
        angular_velocity_imu: Gyroscope samples in rad/s, shape ``(M, 3)``.
        specific_force_imu: Specific-force samples in m/s², shape ``(M, 3)``.
        lidar_timestamps: LiDAR odometry timestamps, shape ``(K,)``.
        lidar_odometry_poses: Absolute/reference LiDAR poses, shape ``(K, 4, 4)``.
        T_B_I_initial: Initial body-from-IMU transform.
        T_B_L_initial: Initial body-from-LiDAR transform.
        bias_initial: Initial gyroscope bias, shape ``(3,)``.
        tau_I_initial: Initial IMU clock offset in seconds.
        tau_L_initial: Initial LiDAR clock offset in seconds.
        verbose: Factor graph verbosity.

    Returns:
        ``FactorGraphExperimentResult`` with rolling window results and stitched
        trajectory.
    """
    streams = {
        "pose_timestamps": np.asarray(pose_timestamps, dtype=float),
        "initial_poses": np.asarray(initial_poses, dtype=float),
        "imu_timestamps": np.asarray(imu_timestamps, dtype=float),
        "angular_velocity_imu": np.asarray(angular_velocity_imu, dtype=float),
        "specific_force_imu": np.asarray(specific_force_imu, dtype=float),
        "lidar_timestamps": np.asarray(lidar_timestamps, dtype=float),
        "lidar_odometry_poses": np.asarray(lidar_odometry_poses, dtype=float),
    }
    validate_factor_graph_streams(streams)

    filter_graph = FactorGraphCalibration(**dict(factor_graph_kwargs))
    rolling_results = filter_graph.generate_filter_iterative(
        window_size=window_size,
        step_size=step_size,
        pose_timestamps=pose_timestamps,
        states=initial_poses,
        imu_timestamps=imu_timestamps,
        angular_velocity_imu=angular_velocity_imu,
        specific_force_imu=specific_force_imu,
        lidar_timestamps=lidar_timestamps,
        lidar_odometry_poses=lidar_odometry_poses,
        T_B_I_initial=T_B_I_initial,
        T_B_L_initial=T_B_L_initial,
        bias_initial=bias_initial,
        tau_I_initial=tau_I_initial,
        tau_L_initial=tau_L_initial,
        verbose=verbose,
    )
    trajectory_timestamps, trajectory_poses = filter_graph.rolling_trajectory

    return FactorGraphExperimentResult(
        label=label,
        graph=filter_graph,
        rolling_results=list(rolling_results),
        trajectory_timestamps_s=np.asarray(trajectory_timestamps, dtype=float),
        trajectory_poses_se3=np.asarray(trajectory_poses, dtype=float),
    )


def align_trajectories_by_timestamp(
    reference_timestamps_s: Sequence[float],
    reference_poses_se3: Sequence[Any],
    estimated_timestamps_s: Sequence[float],
    estimated_poses_se3: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate two trajectories onto their common estimated timestamps.

    Args:
        reference_timestamps_s: Reference trajectory timestamps, shape ``(N,)``.
        reference_poses_se3: Reference poses, shape ``(N, 4, 4)``.
        estimated_timestamps_s: Estimated trajectory timestamps, shape ``(M,)``.
        estimated_poses_se3: Estimated poses, shape ``(M, 4, 4)``.

    Returns:
        Tuple ``(query_timestamps_s, reference_aligned, estimated_aligned)``.
    """
    ref_t = np.asarray(reference_timestamps_s, dtype=float).reshape(-1)
    est_t = np.asarray(estimated_timestamps_s, dtype=float).reshape(-1)
    ref_poses = np.asarray(reference_poses_se3, dtype=float)
    est_poses = np.asarray(estimated_poses_se3, dtype=float)

    start = max(float(ref_t[0]), float(est_t[0]))
    end = min(float(ref_t[-1]), float(est_t[-1]))
    if end < start:
        raise ValueError("Trajectories do not overlap in time")

    query_t = est_t[(est_t >= start) & (est_t <= end)]
    if query_t.size == 0:
        query_t = np.array([0.5 * (start + end)], dtype=float)

    reference_aligned = np.stack([_interpolate_se3(ref_t, ref_poses, t) for t in query_t], axis=0)
    estimated_aligned = np.stack([_interpolate_se3(est_t, est_poses, t) for t in query_t], axis=0)
    return query_t, reference_aligned, estimated_aligned


def _interpolate_se3(timestamps_s: np.ndarray, poses_se3: np.ndarray, query_time_s: float) -> np.ndarray:
    """Interpolate an SE(3) pose using the relative Lie-algebra increment."""
    if query_time_s <= timestamps_s[0]:
        return poses_se3[0].copy()
    if query_time_s >= timestamps_s[-1]:
        return poses_se3[-1].copy()

    upper = int(np.searchsorted(timestamps_s, query_time_s, side="left"))
    if np.isclose(timestamps_s[upper], query_time_s):
        return poses_se3[upper].copy()

    lower = upper - 1
    alpha = (query_time_s - timestamps_s[lower]) / (timestamps_s[upper] - timestamps_s[lower])
    T_lower = mrob.SE3(poses_se3[lower])
    T_upper = mrob.SE3(poses_se3[upper])
    relative_xi = T_lower.inv().mul(T_upper).Ln()
    return np.asarray(T_lower.mul(mrob.SE3(alpha * relative_xi)).T(), dtype=float)
