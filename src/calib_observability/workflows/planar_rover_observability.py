'''High-level observability workflows for simulated and imported rover data.

Simulation and real-data ingestion are intentionally separated. Simulated
measurements may be generated with synthetic noise by the simulation package.
Imported measurements are copied exactly as supplied: this module never adds
random perturbations to real IMU or LiDAR samples.

For imported streams, timestamps remain sensor-clock measurements. Initial
temporal offsets define only the current mapping to a shared reference clock,

    reference_time = sensor_time + tau_initial,

which is needed by the existing observability assembly. The inherited
``*_true`` calibration field names in ``CalibrationSimulationDataset`` are
legacy lower-level API names; for imported data they contain linearization
values, not known ground truth.
'''

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..backend import estimate_poses_dummy
from ..diagnostics import (
    DEFAULT_PRACTICAL_RANK_POLICY,
    PracticalRankPolicy,
    coordinate_metadata_for_variable,
)
from ..factor_observability import (
    SUPPORTED_CALIBRATION_VARIABLES,
    NormalizationMode,
)
from ..lie_se3 import se3_exp, se3_inverse, se3_log
from ..scaling import ParameterScales
from ..simulation import (
    PlanarRoverConfig,
    reframe_dataset_to_fixed_extrinsic,
    simulate_planar_rover,
)
from ..simulation.dataset import CalibrationSimulationDataset
from ..types import (
    AccelerometerOptions,
    FixedExtrinsic,
    JacobianOptions,
    validate_accelerometer_mode,
)
from ..visualization.quasi_realtime_rover import (
    ObservabilityVisualizationSeries,
    build_observability_visualization_series,
    save_quasi_realtime_rover_animation,
    save_quasi_realtime_rover_animation_mp4_subprocess,
)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from new_college_dataset.data import import_true_trajectory


DatasetFactory = Callable[[PlanarRoverConfig, str], object]
PoseProviderFactory = Callable[[object], object]

DEFAULT_GRAVITY_WORLD = np.array([0.0, 0.0, -9.81], dtype=float)
VARIABLE_DISPLAY_LABELS = {
    "T_B_I": "O_T_B_I",
    "T_B_L": "O_T_B_L",
    "b_g": "O_b_g",
    "tau_I": "O_tau_I",
    "tau_L": "O_tau_L",
}


##################################################
# Workflow result and imported-measurement containers
##################################################
@dataclass
class RoverWorkflowArtifacts:
    '''Store paths and in-memory results produced by one workflow run.

    Attributes:
        dataset: Simulation or imported-data dataset used by the analysis.
        pose_provider: Continuous pose provider used to assemble factors.
        series: Rolling observability and local-accuracy diagnostics.
        overview_paths: Saved dataset-overview figures.
        observability_paths: Saved observability figures.
        accuracy_paths: Saved local-accuracy figures.
        animation_path: Optional HTML dashboard animation.
    '''

    dataset: object
    pose_provider: object
    series: ObservabilityVisualizationSeries
    overview_paths: dict[str, Path] = field(default_factory=dict)
    observability_paths: dict[str, Path] = field(default_factory=dict)
    accuracy_paths: dict[str, Path] = field(default_factory=dict)
    animation_path: Path | None = None


@dataclass(frozen=True)
class ImportedImuMeasurements:
    '''Store measured IMU samples without inventing ground-truth samples.

    ``sensor_timestamps`` are the imported sensor-clock timestamps shifted by a
    common constant origin. ``reference_timestamps`` are the same samples mapped
    through the current temporal-offset linearization,

        reference_timestamps = sensor_timestamps + tau_I_initial.

    The gyroscope and accelerometer arrays are copied from the imported data.
    Covariances are weighting models used during whitening; they do not modify
    the measurements.

    Attributes:
        sensor_timestamps: IMU sensor-clock timestamps with shape `(N,)`.
        reference_timestamps: Current common-clock timestamps with shape `(N,)`.
        gyroscope: Measured angular velocity with shape `(N, 3)`.
        accelerometer: Measured specific force with shape `(N, 3)`.
        gyro_covariance: Gyroscope measurement covariance with shape `(3, 3)`.
        accel_covariance: Accelerometer measurement covariance with shape `(3, 3)`.
        gravity_world: Gravity vector used by accelerometer factors.
        accelerometer_convention: Text identifier of the sample convention.
    '''

    sensor_timestamps: NDArray[np.float64]
    reference_timestamps: NDArray[np.float64]
    gyroscope: NDArray[np.float64]
    accelerometer: NDArray[np.float64]
    gyro_covariance: NDArray[np.float64]
    accel_covariance: NDArray[np.float64]
    gravity_world: NDArray[np.float64]
    accelerometer_convention: str = (
        "specific_force_imu_frame_R_IW_times_a_minus_g"
    )


@dataclass(frozen=True)
class ImportedLidarMeasurements:
    '''Store measured LiDAR odometry without creating synthetic observations.

    ``measurements`` contains the imported relative transforms exactly as
    supplied, except for the explicit optional inversion requested by
    ``invert_lidar_measurements``. Covariances are factor weights only.

    Attributes:
        sensor_timestamps: LiDAR scan timestamps on the sensor clock, shape `(M + 1,)`.
        reference_timestamps: Current common-clock scan timestamps, shape `(M + 1,)`.
        measurements: Consecutive relative-pose measurements, shape `(M, 4, 4)`.
        covariances: Per-measurement covariance matrices, shape `(M, 6, 6)`.
        relative_start_times: Interval starts on the LiDAR sensor clock.
        relative_end_times: Interval ends on the LiDAR sensor clock.
    '''

    sensor_timestamps: NDArray[np.float64]
    reference_timestamps: NDArray[np.float64]
    measurements: NDArray[np.float64]
    covariances: NDArray[np.float64]
    relative_start_times: NDArray[np.float64]
    relative_end_times: NDArray[np.float64]


@dataclass
class ImportedCalibrationDataset(CalibrationSimulationDataset):
    '''Distinguish imported measurements from a synthetic simulation dataset.

    The lower-level assembly currently reads the inherited fields
    ``T_B_I_true``, ``T_B_L_true``, ``tau_I_true``, ``tau_L_true``, and
    ``gyro_bias_true``. In this imported-data subclass those fields contain the
    current linearization values supplied by the caller. They are not asserted
    to be ground truth.

    ``imu_rotation_residual_std`` is a factor-weighting parameter for the
    integrated gyroscope rotation residual. It is not synthetic sample noise.

    Attributes:
        imu_rotation_residual_std: Standard deviation used to whiten each
            three-dimensional gyroscope propagation residual.
    '''

    imu_rotation_residual_std: float = 0.01

    def __post_init__(self) -> None:
        '''Validate imported-data-specific factor weighting.'''
        residual_std = float(self.imu_rotation_residual_std)
        if not np.isfinite(residual_std) or residual_std <= 0.0:
            raise ValueError(
                "imu_rotation_residual_std must be finite and positive"
            )
        self.imu_rotation_residual_std = residual_std

    def window_jacobians(
        self,
        start: float,
        end: float,
        pose_provider: object,
        *,
        include_imu: bool = True,
        include_lidar: bool = True,
        include_priors: bool = False,
        include_smoothness: bool = False,
        use_sparse: bool = True,
        parameter_scaling: object | None = None,
        pose_node_rate_hz: float = 5.0,
        imu_rotation_noise_std: float | None = None,
        finite_difference_epsilon: float = 1e-7,
        jacobian_options: JacobianOptions | None = None,
        fixed_extrinsic: FixedExtrinsic = "T_B_L",
        practical_rank_policy: PracticalRankPolicy = (
            DEFAULT_PRACTICAL_RANK_POLICY
        ),
        accelerometer_options: AccelerometerOptions | None = None,
    ) -> tuple[object, list[NDArray[np.float64]], dict[str, int]]:
        '''Assemble a window using imported-data residual weighting.

        The simulation-era lower-level API names the override
        ``imu_rotation_noise_std``. Here it means the standard deviation of the
        already-formed gyroscope rotation residual. When omitted, the dataset
        value supplied during import is used. No measurement samples are
        perturbed.
        '''
        residual_std = (
            self.imu_rotation_residual_std
            if imu_rotation_noise_std is None
            else float(imu_rotation_noise_std)
        )
        if not np.isfinite(residual_std) or residual_std <= 0.0:
            raise ValueError(
                "imu_rotation_noise_std must be finite and positive"
            )

        # This adapter is intentionally nontrivial: it injects the uncertainty
        # model attached to the imported dataset into the simulation-era
        # assembly API, which otherwise silently uses its hard-coded default.
        return super().window_jacobians(
            start,
            end,
            pose_provider,
            include_imu=include_imu,
            include_lidar=include_lidar,
            include_priors=include_priors,
            include_smoothness=include_smoothness,
            use_sparse=use_sparse,
            parameter_scaling=parameter_scaling,
            pose_node_rate_hz=pose_node_rate_hz,
            imu_rotation_noise_std=residual_std,
            finite_difference_epsilon=finite_difference_epsilon,
            jacobian_options=jacobian_options,
            fixed_extrinsic=fixed_extrinsic,
            practical_rank_policy=practical_rank_policy,
            accelerometer_options=accelerometer_options,
        )


##################################################
# Discrete trajectory reconstructed from LiDAR odometry
##################################################
class DiscretePoseTrajectory:
    '''Interpolate an SE(3) trajectory reconstructed from discrete pose knots.

    Consecutive LiDAR odometry measurements are accumulated into pose knots.
    Between knots, the trajectory follows the single-segment geodesic

        T(t) = T_i Exp(alpha Log(T_i^{-1} T_{i+1})).

    The resulting trajectory is an odometry-derived reference used to evaluate
    factors. It is not labeled or treated as ground truth.
    '''

    def __init__(
        self,
        timestamps: NDArray[np.float64],
        poses: NDArray[np.float64],
        *,
        mode: str = "imported_lidar_odometry",
    ) -> None:
        '''Initialize discrete pose knots.

        Args:
            timestamps: Strictly increasing knot times with shape `(N,)`.
            poses: Pose knots with shape `(N, 4, 4)`.
            mode: Descriptive trajectory mode.

        Raises:
            ValueError: The timestamps or pose array is invalid.
        '''
        self.timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
        self.poses = np.asarray(poses, dtype=float)
        self.mode = str(mode)

        if self.poses.shape != (self.timestamps.size, 4, 4):
            raise ValueError(
                "timestamps must have shape (N,) and poses must have "
                "shape (N, 4, 4)"
            )
        if not np.all(np.isfinite(self.timestamps)) or not np.all(
            np.isfinite(self.poses)
        ):
            raise ValueError("trajectory timestamps and poses must be finite")
        if self.timestamps.size < 2 or np.any(
            np.diff(self.timestamps) <= 0.0
        ):
            raise ValueError(
                "trajectory timestamps must be strictly increasing"
            )

    @property
    def start_time(self) -> float:
        '''Return the first trajectory time.'''
        return float(self.timestamps[0])

    @property
    def end_time(self) -> float:
        '''Return the last trajectory time.'''
        return float(self.timestamps[-1])

    def pose_at(self, time_seconds: float) -> NDArray[np.float64]:
        '''Interpolate the pose at one time.

        Args:
            time_seconds: Query time in the trajectory reference clock.

        Returns:
            Interpolated SE(3) pose with shape `(4, 4)`.
        '''
        query_time = float(
            np.clip(time_seconds, self.start_time, self.end_time)
        )
        right_index = int(
            np.searchsorted(self.timestamps, query_time, side="right")
        )

        if right_index <= 0:
            return self.poses[0].copy()
        if right_index >= self.timestamps.size:
            return self.poses[-1].copy()

        left_index = right_index - 1
        left_time = float(self.timestamps[left_index])
        right_time = float(self.timestamps[right_index])
        interpolation_fraction = (
            query_time - left_time
        ) / (right_time - left_time)

        # Interpolate along the SE(3) geodesic between adjacent odometry knots.
        relative_motion = (
            se3_inverse(self.poses[left_index])
            @ self.poses[right_index]
        )
        interpolation_increment = se3_exp(
            interpolation_fraction * se3_log(relative_motion)
        )
        return self.poses[left_index] @ interpolation_increment

    def poses_at(
        self,
        query_times: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        '''Interpolate poses at multiple times.

        Args:
            query_times: Query times with shape `(M,)`.

        Returns:
            Interpolated poses with shape `(M, 4, 4)`.
        '''
        times = np.asarray(query_times, dtype=float).reshape(-1)
        if times.size == 0:
            return np.zeros((0, 4, 4), dtype=float)

        return np.stack(
            [self.pose_at(float(time_seconds)) for time_seconds in times],
            axis=0,
        )

    def position_at(self, time_seconds: float) -> NDArray[np.float64]:
        '''Return the interpolated world position.'''
        return self.pose_at(time_seconds)[:3, 3].copy()

    def euler_at(self, time_seconds: float) -> NDArray[np.float64]:
        '''Return interpolated ZYX Euler angles `[roll, pitch, yaw]`.'''
        rotation = self.pose_at(time_seconds)[:3, :3]

        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        pitch = float(
            np.arctan2(
                -rotation[2, 0],
                np.hypot(rotation[2, 1], rotation[2, 2]),
            )
        )
        roll = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
        return np.array([roll, pitch, yaw], dtype=float)

    def yaw_at(self, time_seconds: float) -> float:
        '''Return the interpolated yaw angle in radians.'''
        return float(self.euler_at(time_seconds)[2])

    def sample(
        self,
        num: int = 400,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        '''Sample times, positions, and Euler angles for plotting.

        Args:
            num: Number of uniformly spaced trajectory samples.

        Returns:
            Tuple `(times, positions, eulers)`.
        '''
        sample_times = np.linspace(
            self.start_time,
            self.end_time,
            int(num),
        )
        positions = np.vstack(
            [
                self.position_at(float(time_seconds))
                for time_seconds in sample_times
            ]
        )
        eulers = np.vstack(
            [
                self.euler_at(float(time_seconds))
                for time_seconds in sample_times
            ]
        )
        return sample_times, positions, eulers


##################################################
# Imported measurement validation and uncertainty
##################################################
def _as_strictly_increasing_timestamps(
    values: ArrayLike,
    name: str,
) -> NDArray[np.float64]:
    '''Validate and return a strictly increasing timestamp vector.'''
    timestamps = np.asarray(values, dtype=float).reshape(-1)

    if timestamps.size < 2 or not np.all(np.isfinite(timestamps)):
        raise ValueError(
            f"{name} must contain at least two finite timestamps"
        )
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")

    return timestamps


def _as_measurement_covariance(
    covariance: ArrayLike | None,
    fallback_std: ArrayLike,
    dimension: int,
    name: str,
) -> NDArray[np.float64]:
    '''Build and validate one measurement covariance matrix.

    An explicit covariance takes precedence. Otherwise ``fallback_std`` may be
    a positive scalar or a positive standard-deviation vector.
    '''
    if covariance is None:
        standard_deviation = np.asarray(fallback_std, dtype=float)
        if standard_deviation.shape == ():
            standard_deviation = np.full(
                dimension,
                float(standard_deviation),
            )
        if standard_deviation.shape != (dimension,):
            raise ValueError(
                f"{name}_std must be scalar or shape ({dimension},)"
            )
        if (
            not np.all(np.isfinite(standard_deviation))
            or np.any(standard_deviation <= 0.0)
        ):
            raise ValueError(
                f"{name}_std must contain finite positive values"
            )
        covariance_matrix = np.diag(standard_deviation**2)
    else:
        covariance_matrix = np.asarray(covariance, dtype=float)

    if covariance_matrix.shape != (dimension, dimension):
        raise ValueError(
            f"{name} covariance must have shape "
            f"({dimension}, {dimension})"
        )
    if not np.all(np.isfinite(covariance_matrix)):
        raise ValueError(f"{name} covariance must be finite")
    if not np.allclose(
        covariance_matrix,
        covariance_matrix.T,
        atol=1e-12,
    ):
        raise ValueError(f"{name} covariance must be symmetric")

    try:
        np.linalg.cholesky(covariance_matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            f"{name} covariance must be positive definite"
        ) from exc

    return covariance_matrix


def _as_lidar_covariances(
    covariances: ArrayLike | None,
    fallback_std: ArrayLike,
    number_of_measurements: int,
) -> NDArray[np.float64]:
    '''Build per-measurement LiDAR covariance matrices.'''
    if covariances is None:
        covariance = _as_measurement_covariance(
            None,
            fallback_std,
            6,
            "lidar_pose",
        )
        return np.repeat(
            covariance[None, :, :],
            number_of_measurements,
            axis=0,
        )

    covariance_array = np.asarray(covariances, dtype=float)
    if covariance_array.shape == (6, 6):
        covariance_array = np.repeat(
            covariance_array[None, :, :],
            number_of_measurements,
            axis=0,
        )
    if covariance_array.shape != (number_of_measurements, 6, 6):
        raise ValueError(
            "lidar_covariances must have shape (6, 6) or (M, 6, 6)"
        )

    validated = np.empty_like(covariance_array)
    for measurement_index, covariance in enumerate(covariance_array):
        validated[measurement_index] = _as_measurement_covariance(
            covariance,
            np.ones(6),
            6,
            f"lidar_pose[{measurement_index}]",
        )
    return validated


def _initial_transform(
    tangent: ArrayLike | None,
    name: str,
) -> NDArray[np.float64]:
    '''Convert an optional rotation-first tangent into an SE(3) transform.'''
    if tangent is None:
        tangent_vector = np.zeros(6, dtype=float)
    else:
        tangent_vector = np.asarray(tangent, dtype=float).reshape(6)
        if not np.all(np.isfinite(tangent_vector)):
            raise ValueError(f"{name} must be finite")

    return se3_exp(tangent_vector)


##################################################
# Bridge from imported measurements to observability assembly
##################################################
def build_dataset_from_imported_sensor_streams(
    *,
    imu_timestamps: NDArray[np.float64],
    gyroscope: NDArray[np.float64],
    accelerometer: NDArray[np.float64],
    lidar_scan_timestamps: NDArray[np.float64],
    lidar_relative_poses: NDArray[np.float64],
    T_B_I_initial_tangent: NDArray[np.float64] | None = None,
    T_B_L_initial_tangent: NDArray[np.float64] | None = None,
    gyro_bias_initial: NDArray[np.float64] | None = None,
    tau_I_initial: float = 0.0,
    tau_L_initial: float = 0.0,
    gyro_noise_std: float = 0.01,
    accel_noise_std: float = 0.1,
    lidar_pose_noise_std: NDArray[np.float64] | float = 0.05,
    gravity_world: NDArray[np.float64] | None = None,
    invert_lidar_measurements: bool = False,
    start_time: float | None = None,
    end_time: float | None = None,
    gyro_covariance: NDArray[np.float64] | None = None,
    accel_covariance: NDArray[np.float64] | None = None,
    lidar_covariances: NDArray[np.float64] | None = None,
    imu_rotation_residual_std: float | None = None,
    true_poses_path = None,
) -> ImportedCalibrationDataset:
    '''Build the observability dataset API from measured IMU and LiDAR streams.

    The imported measurements are never regenerated and no random noise is
    added. ``gyro_noise_std``, ``accel_noise_std``, and
    ``lidar_pose_noise_std`` are retained as backward-compatible names for
    covariance fallbacks only. Prefer passing explicit covariance matrices when
    sensor calibration or estimator covariances are available.

    Input timestamps are interpreted as sensor-clock timestamps. The current
    temporal-offset linearization maps each stream to a common reference clock,

        t_reference = t_sensor + tau_initial.

    ``start_time`` and ``end_time`` are interpreted in this common reference
    clock. All stored times are shifted by one constant origin for numerical
    stability while preserving the relation above.

    Args:
        imu_timestamps: Measured IMU sensor timestamps, shape `(N,)`.
        gyroscope: Measured angular velocity, shape `(N, 3)`.
        accelerometer: Measured specific force, shape `(N, 3)`.
        lidar_scan_timestamps: Measured LiDAR sensor timestamps, shape `(M + 1,)`.
        lidar_relative_poses: Consecutive measured transforms, shape `(M, 4, 4)`.
        T_B_I_initial_tangent: Initial body-from-IMU calibration tangent.
        T_B_L_initial_tangent: Initial body-from-LiDAR calibration tangent.
        gyro_bias_initial: Initial gyroscope bias linearization.
        tau_I_initial: Current IMU sensor-to-reference clock offset.
        tau_L_initial: Current LiDAR sensor-to-reference clock offset.
        gyro_noise_std: Legacy covariance fallback standard deviation. No noise
            is added to gyroscope samples.
        accel_noise_std: Legacy covariance fallback standard deviation. No noise
            is added to accelerometer samples.
        lidar_pose_noise_std: Legacy covariance fallback standard deviation. No
            noise is added to LiDAR measurements.
        gravity_world: Gravity vector used by accelerometer factors.
        invert_lidar_measurements: Whether to invert every imported relative pose
            to correct the external odometry convention.
        start_time: Optional analysis start in the common reference clock.
        end_time: Optional analysis end in the common reference clock.
        gyro_covariance: Optional explicit gyroscope covariance, shape `(3, 3)`.
        accel_covariance: Optional explicit accelerometer covariance, shape `(3, 3)`.
        lidar_covariances: Optional LiDAR covariance, shape `(6, 6)` or `(M, 6, 6)`.
        imu_rotation_residual_std: Optional standard deviation of the
            integrated gyroscope rotation residual. When omitted, the
            legacy ``gyro_noise_std`` value is used as the factor weight.

    Returns:
        A ``CalibrationSimulationDataset`` compatible with the existing lower
        observability assembly. Its legacy ``*_true`` fields store the supplied
        initial linearization values, not known truth.

    Raises:
        ValueError: Shapes, timestamps, covariances, or the requested overlap are
            invalid.
    '''
    imu_sensor_times = _as_strictly_increasing_timestamps(
        imu_timestamps,
        "imu_timestamps",
    )
    lidar_sensor_times = _as_strictly_increasing_timestamps(
        lidar_scan_timestamps,
        "lidar_scan_timestamps",
    )
    gyro_samples = np.asarray(gyroscope, dtype=float)
    accel_samples = np.asarray(accelerometer, dtype=float)
    relative_poses = np.asarray(lidar_relative_poses, dtype=float)

    if gyro_samples.shape != (imu_sensor_times.size, 3):
        raise ValueError(
            "gyroscope must have shape (len(imu_timestamps), 3)"
        )
    if accel_samples.shape != (imu_sensor_times.size, 3):
        raise ValueError(
            "accelerometer must have shape (len(imu_timestamps), 3)"
        )
    if relative_poses.shape != (lidar_sensor_times.size - 1, 4, 4):
        raise ValueError(
            "lidar_relative_poses must have shape "
            "(len(lidar_scan_timestamps) - 1, 4, 4)"
        )
    if (
        not np.all(np.isfinite(gyro_samples))
        or not np.all(np.isfinite(accel_samples))
        or not np.all(np.isfinite(relative_poses))
    ):
        raise ValueError("imported sensor measurements must be finite")

    tau_I_linearization = float(tau_I_initial)
    tau_L_linearization = float(tau_L_initial)
    if not np.all(
        np.isfinite([tau_I_linearization, tau_L_linearization])
    ):
        raise ValueError("initial temporal offsets must be finite")

    # Map measured sensor clocks into the current common reference clock.
    imu_reference_times = imu_sensor_times + tau_I_linearization
    lidar_reference_times = lidar_sensor_times + tau_L_linearization

    available_start = max(
        float(imu_reference_times[0]),
        float(lidar_reference_times[0]),
    )
    available_end = min(
        float(imu_reference_times[-1]),
        float(lidar_reference_times[-1]),
    )
    requested_start = (
        available_start
        if start_time is None
        else max(float(start_time), available_start)
    )
    requested_end = (
        available_end
        if end_time is None
        else min(float(end_time), available_end)
    )

    if (
        not np.all(np.isfinite([requested_start, requested_end]))
        or requested_end <= requested_start
    ):
        raise ValueError(
            "imported streams do not have a positive overlapping time range"
        )

    # Keep complete LiDAR intervals whose endpoint scans lie in the requested range.
    first_scan_index = int(
        np.searchsorted(
            lidar_reference_times,
            requested_start,
            side="left",
        )
    )
    last_scan_index = (
        int(
            np.searchsorted(
                lidar_reference_times,
                requested_end,
                side="right",
            )
        )
        - 1
    )
    if last_scan_index <= first_scan_index:
        raise ValueError(
            "imported streams do not contain a complete LiDAR interval "
            "in the requested range"
        )

    kept_lidar_sensor_times = lidar_sensor_times[
        first_scan_index : last_scan_index + 1
    ]
    kept_lidar_reference_times = lidar_reference_times[
        first_scan_index : last_scan_index + 1
    ]
    kept_relative_poses = relative_poses[
        first_scan_index:last_scan_index
    ].copy()

    if invert_lidar_measurements:
        kept_relative_poses = np.stack(
            [
                se3_inverse(relative_pose)
                for relative_pose in kept_relative_poses
            ],
            axis=0,
        )

    # Select measured IMU samples by their current common-clock timestamps.
    imu_mask = (
        (imu_reference_times >= kept_lidar_reference_times[0])
        & (imu_reference_times <= kept_lidar_reference_times[-1])
    )
    if np.count_nonzero(imu_mask) < 2:
        raise ValueError(
            "imported streams do not contain enough IMU samples "
            "inside the selected LiDAR range"
        )

    # Shift both clocks by the same reference-time origin. This preserves
    # reference_time = sensor_time + tau_initial exactly.
    # reference_time_origin = float(kept_lidar_reference_times[0])
    # shifted_lidar_reference_times = (
    #     kept_lidar_reference_times - reference_time_origin
    # )
    # shifted_lidar_sensor_times = (
    #     kept_lidar_sensor_times - reference_time_origin
    # )
    # shifted_imu_reference_times = (
    #     imu_reference_times[imu_mask] - reference_time_origin
    # )
    # shifted_imu_sensor_times = (
    #     imu_sensor_times[imu_mask] - reference_time_origin
    # )
    imu_reference_times = imu_reference_times[imu_mask]
    imu_sensor_times = imu_sensor_times[imu_mask]

    # Accumulate measured LiDAR increments into an odometry reference trajectory.
    cumulative_poses = [np.eye(4, dtype=float)]
    for relative_pose in kept_relative_poses:
        cumulative_poses.append(
            cumulative_poses[-1] @ relative_pose
        )
    if true_poses_path is None:
        # From lidar
        trajectory = DiscretePoseTrajectory(
            kept_lidar_reference_times,
            np.stack(cumulative_poses, axis=0),
        )
    else:
        true_timestamps, true_trajectory = import_true_trajectory(true_poses_path)
        trajectory = DiscretePoseTrajectory(
            true_timestamps,
            true_trajectory,
        )

    # Build uncertainty models for whitening only. Samples are copied unchanged.
    gyro_covariance_matrix = _as_measurement_covariance(
        gyro_covariance,
        gyro_noise_std,
        3,
        "gyro",
    )
    accel_covariance_matrix = _as_measurement_covariance(
        accel_covariance,
        accel_noise_std,
        3,
        "accel",
    )
    all_lidar_covariances = _as_lidar_covariances(
        lidar_covariances,
        lidar_pose_noise_std,
        relative_poses.shape[0],
    )
    kept_lidar_covariances = all_lidar_covariances[
        first_scan_index:last_scan_index
    ].copy()

    gravity = (
        DEFAULT_GRAVITY_WORLD.copy()
        if gravity_world is None
        else np.asarray(gravity_world, dtype=float).reshape(3)
    )
    if not np.all(np.isfinite(gravity)):
        raise ValueError("gravity_world must be finite")

    imu = ImportedImuMeasurements(
        sensor_timestamps=imu_sensor_times,
        reference_timestamps=imu_reference_times,
        gyroscope=gyro_samples[imu_mask].copy(),
        accelerometer=accel_samples[imu_mask].copy(),
        gyro_covariance=gyro_covariance_matrix,
        accel_covariance=accel_covariance_matrix,
        gravity_world=gravity,
    )
    lidar = ImportedLidarMeasurements(
        sensor_timestamps=kept_lidar_sensor_times,
        reference_timestamps=kept_lidar_reference_times,
        measurements=kept_relative_poses,
        covariances=kept_lidar_covariances,
        relative_start_times=lidar_sensor_times[:-1],
        relative_end_times=lidar_sensor_times[1:],
    )

    gyro_bias = (
        np.zeros(3, dtype=float)
        if gyro_bias_initial is None
        else np.asarray(gyro_bias_initial, dtype=float).reshape(3)
    )
    if not np.all(np.isfinite(gyro_bias)):
        raise ValueError("gyro_bias_initial must be finite")

    residual_std = (
        float(gyro_noise_std)
        if imu_rotation_residual_std is None
        else float(imu_rotation_residual_std)
    )
    if not np.isfinite(residual_std) or residual_std <= 0.0:
        raise ValueError(
            "imu_rotation_residual_std must be finite and positive"
        )

    # The lower-level assembly still uses simulation-era field names. For an
    # imported dataset these values are current calibration linearization points.
    return ImportedCalibrationDataset(
        trajectory=trajectory,
        imu=imu,
        lidar=lidar,
        T_B_L_true=_initial_transform(
            T_B_L_initial_tangent,
            "T_B_L_initial_tangent",
        ),
        T_B_I_true=_initial_transform(
            T_B_I_initial_tangent,
            "T_B_I_initial_tangent",
        ),
        tau_I_true=tau_I_linearization,
        tau_L_true=tau_L_linearization,
        gyro_bias_true=gyro_bias,
        imu_rotation_residual_std=residual_std,
    )


##################################################
# Dataset construction and pose-provider selection
##################################################
def build_planar_rover_dataset(
    rover_config: PlanarRoverConfig,
    *,
    trajectory_mode: str | None = None,
    fixed_extrinsic: FixedExtrinsic = "T_B_L",
    dataset_factory: DatasetFactory | None = None,
    pose_provider_factory: PoseProviderFactory | None = None,
) -> tuple[object, object]:
    '''Create a simulation or imported dataset and its pose provider.

    Args:
        rover_config: Simulation configuration or loader configuration carrier.
        trajectory_mode: Optional trajectory mode passed to the source factory.
        fixed_extrinsic: Body-frame convention used by downstream assembly.
        dataset_factory: Optional real-data loader. When omitted, simulation is
            generated explicitly.
        pose_provider_factory: Optional pose-provider constructor.

    Returns:
        Tuple `(dataset, pose_provider)`.
    '''
    selected_mode = (
        rover_config.mode
        if trajectory_mode is None
        else trajectory_mode
    )

    if dataset_factory is None:
        source_dataset = simulate_planar_rover(
            rover_config,
            mode=selected_mode,
        )
    else:
        source_dataset = dataset_factory(
            rover_config,
            selected_mode,
        )

    dataset = reframe_dataset_to_fixed_extrinsic(
        source_dataset,
        fixed_extrinsic,
    )

    if pose_provider_factory is None:
        pose_provider = estimate_poses_dummy(dataset)
    else:
        pose_provider = pose_provider_factory(dataset)

    return dataset, pose_provider


##################################################
# Dataset overview plots
##################################################
def plot_rover_dataset_overview(
    dataset: object,
    output_dir: str | Path,
    *,
    trajectory_samples: int = 800,
) -> dict[str, Path]:
    '''Plot trajectory geometry and the available sensor measurements.

    The function accepts both simulated sensor containers and the imported
    measurement containers defined in this module.

    Args:
        dataset: Dataset exposing `trajectory`, `imu`, and `lidar`.
        output_dir: Directory for generated figures.
        trajectory_samples: Number of trajectory samples used for the overview.

    Returns:
        Mapping from figure purpose to saved path.
    '''
    import matplotlib.pyplot as plt

    output_directory = _ensure_output_dir(output_dir)
    trajectory = getattr(dataset, "trajectory")
    imu = getattr(dataset, "imu")
    lidar = getattr(dataset, "lidar")

    sample_times, positions, eulers = trajectory.sample(
        int(trajectory_samples)
    )

    # Plot the odometry or simulated path together with LiDAR event locations.
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        color="0.35",
        lw=2.0,
        label="trajectory reference",
    )
    ax.scatter(
        positions[0, 0],
        positions[0, 1],
        marker="o",
        color="tab:green",
        label="start",
    )
    ax.scatter(
        positions[-1, 0],
        positions[-1, 1],
        marker="x",
        color="tab:red",
        label="end",
    )

    lidar_times = _measurement_reference_timestamps(lidar)
    if lidar_times.size:
        lidar_positions = np.vstack(
            [
                np.asarray(
                    trajectory.position_at(float(time_seconds)),
                    dtype=float,
                )
                for time_seconds in lidar_times
            ]
        )
        ax.scatter(
            lidar_positions[:, 0],
            lidar_positions[:, 1],
            s=12,
            alpha=0.5,
            label="LiDAR samples",
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Rover trajectory and LiDAR sample locations")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    trajectory_path = (
        output_directory / "dataset_trajectory_overview.png"
    )
    fig.savefig(trajectory_path, dpi=160)
    plt.close(fig)

    # Plot signal norms to remain readable for both simulation and real logs.
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(9.0, 7.0),
        sharex=False,
    )
    axes[0].plot(sample_times, positions[:, 0], label="x")
    axes[0].plot(sample_times, positions[:, 1], label="y")
    axes[0].plot(sample_times, eulers[:, 2], label="yaw")
    axes[0].set_ylabel("pose")
    axes[0].set_title("Trajectory coordinates")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(ncol=3)

    imu_times = _measurement_reference_timestamps(imu)
    gyroscope = np.asarray(
        getattr(imu, "gyroscope"),
        dtype=float,
    )
    accelerometer = np.asarray(
        getattr(imu, "accelerometer"),
        dtype=float,
    )

    axes[1].plot(
        imu_times,
        np.linalg.norm(gyroscope, axis=1),
        color="tab:blue",
    )
    axes[1].set_ylabel("|gyro| [rad/s]")
    axes[1].set_title("Measured gyroscope stream")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(
        imu_times,
        np.linalg.norm(accelerometer, axis=1),
        color="tab:orange",
    )
    axes[2].set_xlabel("reference time [s]")
    axes[2].set_ylabel("|accel| [m/s^2]")
    axes[2].set_title("Measured accelerometer specific-force stream")
    axes[2].grid(True, alpha=0.3)
    fig.tight_layout()

    sensor_path = output_directory / "dataset_sensor_overview.png"
    fig.savefig(sensor_path, dpi=160)
    plt.close(fig)

    return {
        "trajectory": trajectory_path,
        "sensors": sensor_path,
    }


##################################################
# Rolling observability alias
##################################################
# Keep the notebook-facing name without an empty forwarding function.
# The partial also preserves this module's historical default variable set.
run_rolling_observability_analysis = partial(
    build_observability_visualization_series,
    display_variables=SUPPORTED_CALIBRATION_VARIABLES,
)


##################################################
# Rolling observability plots
##################################################
def plot_observability_over_time(
    series: ObservabilityVisualizationSeries,
    output_dir: str | Path,
    *,
    display_variables: tuple[str, ...] = SUPPORTED_CALIBRATION_VARIABLES,
) -> dict[str, Path]:
    '''Plot practical ranks, condition numbers, and factor counts over time.

    Args:
        series: Canonical rolling observability series.
        output_dir: Directory for generated figures.
        display_variables: Calibration variables shown in the plots.

    Returns:
        Mapping from diagnostic name to saved figure path.
    '''
    import matplotlib.pyplot as plt

    output_directory = _ensure_output_dir(output_dir)
    paths: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for variable_name in display_variables:
        values = np.asarray(
            series.ranks.get(
                variable_name,
                np.full(series.times.shape, np.nan),
            ),
            dtype=float,
        )
        ax.plot(
            series.times,
            values,
            marker="o",
            ms=3,
            label=_display_label(variable_name),
        )

    ax.plot(
        series.times,
        series.C_X_L_rank,
        marker="s",
        ms=3,
        linestyle="--",
        label="C_X_L",
    )
    if np.any(np.isfinite(series.C_X_I_accel_rank)):
        ax.plot(
            series.times,
            series.C_X_I_accel_rank,
            marker="^",
            ms=3,
            linestyle="--",
            label="C_X_I accel",
        )

    ax.set_xlabel("current time [s]")
    ax.set_ylabel("practical rank")
    ax.set_title("Rolling observability ranks")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=3)
    fig.tight_layout()

    paths["ranks"] = (
        output_directory / "rolling_observability_ranks.png"
    )
    fig.savefig(paths["ranks"], dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    for variable_name in _condition_plot_variables(
        display_variables
    ):
        values = np.asarray(
            series.condition_numbers.get(
                variable_name,
                np.full(series.times.shape, np.nan),
            ),
            dtype=float,
        )
        finite_values = np.where(
            np.isfinite(values) & (values > 0.0),
            values,
            np.nan,
        )
        ax.semilogy(
            series.times,
            finite_values,
            marker="o",
            ms=3,
            label=_display_label(variable_name),
        )

    ax.set_xlabel("current time [s]")
    ax.set_ylabel("practical condition number")
    ax.set_title("Rolling per-variable condition numbers")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()

    paths["condition_numbers"] = (
        output_directory / "rolling_condition_numbers.png"
    )
    fig.savefig(paths["condition_numbers"], dpi=160)
    plt.close(fig)

    imu_counts = np.asarray(
        [
            snapshot.counts.get("imu", 0)
            for snapshot in series.snapshots
        ],
        dtype=float,
    )
    lidar_counts = np.asarray(
        [
            snapshot.counts.get("lidar", 0)
            for snapshot in series.snapshots
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.plot(
        series.times,
        imu_counts,
        marker="o",
        ms=3,
        label="gyro factors",
    )
    ax.plot(
        series.times,
        lidar_counts,
        marker="o",
        ms=3,
        label="LiDAR factors",
    )
    ax.plot(
        series.times,
        series.accelerometer_factor_count,
        marker="o",
        ms=3,
        label="accelerometer factors",
    )
    ax.set_xlabel("current time [s]")
    ax.set_ylabel("factor count")
    ax.set_title("Rolling factor counts")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    paths["factor_counts"] = (
        output_directory / "rolling_factor_counts.png"
    )
    fig.savefig(paths["factor_counts"], dpi=160)
    plt.close(fig)

    return paths


def plot_local_crlb_accuracy(
    series: ObservabilityVisualizationSeries,
    output_dir: str | Path,
    *,
    display_variables: tuple[str, ...] = SUPPORTED_CALIBRATION_VARIABLES,
) -> dict[str, Path]:
    '''Plot local CRLB-like standard-deviation bounds.

    Args:
        series: Canonical rolling observability series.
        output_dir: Directory for generated figures.
        display_variables: Calibration variables shown in the plots.

    Returns:
        Mapping from diagnostic name to saved figure path.
    '''
    import matplotlib.pyplot as plt

    output_directory = _ensure_output_dir(output_dir)
    paths: dict[str, Path] = {}

    for variable_name in display_variables:
        coordinate_bounds = series.coordinate_std_bounds.get(
            variable_name
        )
        bounded_mask = series.coordinate_bounded_mask.get(
            variable_name
        )

        if (
            coordinate_bounds is None
            or coordinate_bounds.ndim != 2
            or coordinate_bounds.shape[1] <= 1
        ):
            continue

        labels, units = coordinate_metadata_for_variable(
            variable_name,
            coordinate_bounds.shape[1],
        )

        fig, ax = plt.subplots(figsize=(9.5, 4.2))
        for column_index, label in enumerate(labels):
            values = np.asarray(
                coordinate_bounds[:, column_index],
                dtype=float,
            )
            if bounded_mask is not None:
                values = np.where(
                    bounded_mask[:, column_index],
                    values,
                    np.nan,
                )

            finite_values = np.where(
                np.isfinite(values) & (values > 0.0),
                values,
                np.nan,
            )
            ax.semilogy(
                series.times,
                finite_values,
                marker="o",
                ms=3,
                label=f"{label} [{units[column_index]}]",
            )

        ax.set_xlabel("current time [s]")
        ax.set_ylabel("local CRLB-like std bound")
        ax.set_title(
            f"{_display_label(variable_name)} coordinate bounds"
        )
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(ncol=2)
        fig.tight_layout()

        path = output_directory / (
            f"{_filename_token(variable_name)}"
            "_coordinate_crlb_bounds.png"
        )
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths[f"{variable_name}_coordinate_bounds"] = path

    tau_variables = [
        variable_name
        for variable_name in display_variables
        if variable_name in {"tau_I", "tau_L"}
    ]
    if tau_variables:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(9.5, 6.0),
            sharex=True,
        )

        for variable_name in tau_variables:
            seconds = np.asarray(
                series.tau_std_bounds.get(
                    variable_name,
                    np.full(series.times.shape, np.nan),
                ),
                dtype=float,
            )
            frames = np.asarray(
                series.tau_std_bounds_lidar_frames.get(
                    variable_name,
                    np.full(series.times.shape, np.nan),
                ),
                dtype=float,
            )

            axes[0].semilogy(
                series.times,
                np.where(
                    np.isfinite(seconds) & (seconds > 0.0),
                    seconds,
                    np.nan,
                ),
                marker="o",
                ms=3,
                label=_display_label(variable_name),
            )
            axes[1].semilogy(
                series.times,
                np.where(
                    np.isfinite(frames) & (frames > 0.0),
                    frames,
                    np.nan,
                ),
                marker="o",
                ms=3,
                label=_display_label(variable_name),
            )

        axes[0].set_ylabel("std bound [s]")
        axes[0].set_title("Timing local CRLB-like bounds")
        axes[1].set_xlabel("current time [s]")
        axes[1].set_ylabel("std bound [LiDAR frames]")

        for axis in axes:
            axis.grid(True, which="both", alpha=0.3)
            axis.legend()

        fig.tight_layout()
        paths["tau_bounds"] = (
            output_directory / "tau_local_crlb_bounds.png"
        )
        fig.savefig(paths["tau_bounds"], dpi=160)
        plt.close(fig)

    return paths


##################################################
# Simple-accelerometer dashboard
##################################################
def save_simple_accelerometer_dashboard(
    dataset: object,
    pose_provider: object,
    output_html: str | Path,
    *,
    window_duration: float,
    window_step: float,
    accelerometer_options: AccelerometerOptions,
    fixed_extrinsic: FixedExtrinsic = "T_B_L",
    practical_rank_policy: PracticalRankPolicy = DEFAULT_PRACTICAL_RANK_POLICY,
    parameter_scales: ParameterScales = ParameterScales(),
    tau_target_std_seconds: float = 0.2,
    jacobian_options: JacobianOptions | None = None,
    use_sparse: bool = False,
    display_variables: tuple[str, ...] = SUPPORTED_CALIBRATION_VARIABLES,
    normalization: NormalizationMode = "physical_then_column",
    max_display_rows: int = 80,
    max_display_cols: int = 40,
    lidar_rate_hz: float | None = None,
    coordinate_null_fraction_tolerance: float = 1e-6,
    trajectory_samples: int = 700,
    interval_ms: int = 300,
    output_mp4: str | Path | None = None,
    mp4_fps: float | None = None,
    mp4_dpi: int = 160,
    max_rendered_frames: int | None = None,
    html_dpi: int = 80,
    html_frame_format: str = "jpeg",
    embed_limit_mb: float = 1000.0,
    figsize=(17, 10),
    downsample = 1,
    standalone_html: bool = True,
    standalone_html_max_frames: int = 2000,
    save_html=True,
) -> tuple[ObservabilityVisualizationSeries, Path]:
    '''Run simple-accelerometer diagnostics and save the live dashboard.

    Every snapshot is rendered by default. Set ``max_rendered_frames`` only when an explicitly downsampled preview is desired.

    Args:
        dataset: Dataset used by rolling observability analysis.
        pose_provider: Continuous pose provider.
        output_html: Destination HTML animation path.
        window_duration: Rolling-window duration in seconds.
        window_step: Time step between dashboard snapshots.
        accelerometer_options: Accelerometer configuration with mode ``simple``.
        fixed_extrinsic: Body-frame convention.
        practical_rank_policy: Practical-rank threshold policy.
        parameter_scales: Physical parameter scales.
        tau_target_std_seconds: Target timing standard deviation.
        jacobian_options: Analytic or numerical Jacobian configuration.
        use_sparse: Whether sparse assembly and projection are used.
        display_variables: Calibration variables shown by the dashboard.
        normalization: Target-observability normalization mode.
        max_display_rows: Maximum displayed matrix rows.
        max_display_cols: Maximum displayed matrix columns.
        lidar_rate_hz: Optional LiDAR rate for timing-frame conversion.
        coordinate_null_fraction_tolerance: Coordinate boundedness threshold.
        trajectory_samples: Number of path samples in the animation.
        interval_ms: Animation playback interval in milliseconds.
        output_mp4: Optional companion MP4 path.
        mp4_fps: Optional MP4 frame rate.
        mp4_dpi: MP4 rendering resolution.
        max_rendered_frames: Optional frame limit. ``None`` renders every snapshot.
        html_dpi: HTML frame rasterization resolution.
        html_frame_format: Embedded HTML frame format.
        embed_limit_mb: Maximum embedded HTML size in megabytes.

    Returns:
        Tuple ``(simple_series, animation_path)``.

    Raises:
        ValueError: If the accelerometer mode is not ``simple``.
    '''
    accelerometer_mode = validate_accelerometer_mode(accelerometer_options.mode)
    if accelerometer_mode != "simple":
        raise ValueError("save_simple_accelerometer_dashboard expects AccelerometerOptions(mode='simple')")

    simple_series = build_observability_visualization_series(
        dataset, pose_provider, window_duration=window_duration, window_step=window_step, fixed_extrinsic=fixed_extrinsic,
        practical_rank_policy=practical_rank_policy, parameter_scales=parameter_scales, tau_target_std_seconds=tau_target_std_seconds,
        jacobian_options=jacobian_options, accelerometer_options=accelerometer_options, use_sparse=use_sparse, display_variables=display_variables,
        normalization=normalization, max_display_rows=max_display_rows, max_display_cols=max_display_cols, lidar_rate_hz=lidar_rate_hz,
        coordinate_null_fraction_tolerance=coordinate_null_fraction_tolerance, show_local_accuracy_summary=True,
    )

    rendered_snapshots = simple_series.snapshots[::downsample]
    output_path = Path(output_html)
    requested_mp4_path = Path(output_mp4) if output_mp4 is not None else None
    save_html_effective = bool(save_html)

    # Notebook 11 passes an .mp4 path as the primary animation path. Treat that
    # as an MP4 request and avoid writing HTML bytes to an MP4-named file.
    if output_path.suffix.lower() == ".mp4":
        if requested_mp4_path is None:
            requested_mp4_path = output_path
        save_html_effective = False
        html_output_path = output_path.with_suffix(".html")
    else:
        html_output_path = output_path

    if save_html_effective:
        animation_path = save_quasi_realtime_rover_animation(
            dataset, rendered_snapshots, html_output_path, display_variables=display_variables, trajectory_samples=trajectory_samples, interval_ms=interval_ms,
            show_local_accuracy_summary=True, output_mp4=None, mp4_fps=mp4_fps, mp4_dpi=mp4_dpi, max_rendered_frames=max_rendered_frames,
            html_dpi=html_dpi, html_frame_format=html_frame_format, embed_limit_mb=embed_limit_mb, figsize=figsize, standalone_html=standalone_html, standalone_html_max_frames=standalone_html_max_frames,
            save_html=True,)
    else:
        animation_path = requested_mp4_path if requested_mp4_path is not None else html_output_path

    if requested_mp4_path is not None:
        mp4_path = save_quasi_realtime_rover_animation_mp4_subprocess(
            dataset, rendered_snapshots, requested_mp4_path, display_variables=display_variables, trajectory_samples=trajectory_samples, interval_ms=interval_ms,
            show_local_accuracy_summary=True, mp4_fps=mp4_fps, mp4_dpi=mp4_dpi, max_rendered_frames=max_rendered_frames,
            html_dpi=html_dpi, html_frame_format=html_frame_format, embed_limit_mb=embed_limit_mb, figsize=figsize, standalone_html=standalone_html, standalone_html_max_frames=standalone_html_max_frames,
        )
        if not save_html_effective or output_path.suffix.lower() == ".mp4":
            animation_path = mp4_path

    return simple_series, animation_path


##################################################
# Internal plotting helpers
##################################################
def _measurement_reference_timestamps(
    sensor_data: object,
) -> NDArray[np.float64]:
    '''Return timestamps on the analysis reference clock.

    Imported containers expose ``reference_timestamps``. Simulation containers
    expose ``true_times``. The final fallback is the raw sensor clock.
    '''
    for attribute_name in (
        "reference_timestamps",
        "true_times",
        "sensor_timestamps",
    ):
        if hasattr(sensor_data, attribute_name):
            timestamps = np.asarray(
                getattr(sensor_data, attribute_name),
                dtype=float,
            ).reshape(-1)
            if not np.all(np.isfinite(timestamps)):
                raise ValueError(
                    f"{attribute_name} contains non-finite timestamps"
                )
            return timestamps

    raise AttributeError(
        f"{type(sensor_data).__name__} does not expose timestamps"
    )


def _display_label(variable_name: str) -> str:
    '''Return the compact display label for one calibration variable.'''
    return VARIABLE_DISPLAY_LABELS.get(
        variable_name,
        f"O_{variable_name}",
    )


def _condition_plot_variables(
    display_variables: tuple[str, ...],
) -> tuple[str, ...]:
    '''Exclude scalar timing variables from condition-number plots.'''
    return tuple(
        variable_name
        for variable_name in display_variables
        if variable_name not in {"tau_I", "tau_L"}
    )


def _ensure_output_dir(output_dir: str | Path) -> Path:
    '''Create and return an output directory.'''
    output_directory = Path(output_dir).expanduser()
    output_directory.mkdir(parents=True, exist_ok=True)
    return output_directory


def _filename_token(value: str) -> str:
    '''Convert a plot label into a stable filename token.'''
    return (
        value.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("{", "")
        .replace("}", "")
    )