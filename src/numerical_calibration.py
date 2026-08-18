'''Numerical coarse IMU calibration from LiDAR angular velocity and IMU gyro streams.

Frame convention:
    source = IMU angular velocity expressed in I.
    reference = body angular velocity expressed in B.
    spatial calibration solves ``omega_B ~= R_B_I @ omega_I``.

Temporal convention:
    IMU query time = body/reference time + tau_I.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial.transform import Rotation

try:
    import transform
except ImportError:  # pragma: no cover - supports importing this module as src.numerical_calibration.
    from src import transform

BiasMode = Literal['provided', 'stationary']


@dataclass(frozen=True)
class NumericalCalibrationConfig:
    '''Configuration for TwistnSync/SciPy numerical coarse calibration.

    Args:
        margin_s: Extra sensor support around each calibration window.
        resample: Whether TwistnSync should internally resample streams.
        resample_step_s: Common synchronization step. ``None`` uses median sample spacing.
        min_samples: Minimum synchronized vector pairs required.
        max_abs_tau_s: Absolute tau search bound for the fallback.
        tau_grid_step_s: Tau-grid step for fallback. ``None`` uses half the sync step.
        prefer_twistnsync: Use finite TwistnSync adapter result when available.
        stationary_interval: Optional interval used by stationary bias mode.
    '''

    margin_s: float = 2.0
    resample: bool = True
    resample_step_s: float | None = None
    min_samples: int = 20
    max_abs_tau_s: float = 1.0
    tau_grid_step_s: float | None = None
    prefer_twistnsync: bool = True
    stationary_interval: tuple[float, float] | None = None


@dataclass(frozen=True)
class NumericalCalibrationResult:
    '''Result and diagnostics of one numerical coarse-calibration window.

    Args:
        tau_I: IMU temporal offset in seconds using ``IMU query time = body time + tau_I``.
        R_B_I: Rotation mapping IMU angular velocity in I to body angular velocity in B.
        T_B_I: SE(3) prior with estimated rotation and supplied/previous translation.
        bias_g_used: Gyroscope bias subtracted before synchronization/alignment.
        temporal_delay_raw: Raw TwistnSync delay when available.
        spatial_rssd: SciPy ``align_vectors`` RSSD diagnostic.
        source_timestamps: IMU timestamps used by the estimator.
        reference_timestamps: LiDAR/body timestamps used by the estimator.
        synchronized_timestamps: Common body/reference timestamps.
        source_angvels: Bias-corrected IMU angular velocities before synchronization.
        reference_angvels: Body angular velocities before synchronization.
        source_angvels_synchronized: IMU vectors at ``t + tau_I``.
        reference_angvels_synchronized: Body vectors at ``t``.
        source_angvels_aligned: ``R_B_I @ source_angvels_synchronized``.
        residuals: ``reference_angvels_synchronized - source_angvels_aligned``.
        residual_rmse: Component residual RMSE, shape ``(3,)``.
        residual_vector_rmse: RMSE of vector residual norms.
        residual_vector_median: Median vector residual norm.
        excitation_singular_values: SVD values of centered synchronized IMU vectors.
        excitation_ratios: ``(s2/s1, s3/s1)``.
        success: Whether this result may be used as a prior.
        message: Short status/failure message.
        diagnostics: Extra TwistnSync/fallback diagnostics.
    '''

    tau_I: float | None
    R_B_I: NDArray[np.float64] | None
    T_B_I: NDArray[np.float64] | None
    bias_g_used: NDArray[np.float64] | None
    temporal_delay_raw: float | None
    spatial_rssd: float | None
    source_timestamps: NDArray[np.float64]
    reference_timestamps: NDArray[np.float64]
    synchronized_timestamps: NDArray[np.float64]
    source_angvels: NDArray[np.float64]
    reference_angvels: NDArray[np.float64]
    source_angvels_synchronized: NDArray[np.float64]
    reference_angvels_synchronized: NDArray[np.float64]
    source_angvels_aligned: NDArray[np.float64]
    residuals: NDArray[np.float64]
    residual_rmse: NDArray[np.float64]
    residual_vector_rmse: float | None
    residual_vector_median: float | None
    excitation_singular_values: NDArray[np.float64]
    excitation_ratios: tuple[float, float]
    success: bool
    message: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _failure_result(message: str, *, diagnostics: dict[str, Any] | None = None) -> NumericalCalibrationResult:
    '''Create a structured unsuccessful numerical-calibration result.

    Args:
        message: Human-readable failure reason.
        diagnostics: Optional implementation diagnostics.

    Returns:
        ``NumericalCalibrationResult`` with empty arrays and ``success=False``.
    '''

    # Return shape-compatible empty arrays so notebooks can inspect failed windows safely.
    empty_t = np.empty(0, dtype=float)
    empty_v = np.empty((0, 3), dtype=float)
    return NumericalCalibrationResult(
        tau_I=None,
        R_B_I=None,
        T_B_I=None,
        bias_g_used=None,
        temporal_delay_raw=None,
        spatial_rssd=None,
        source_timestamps=empty_t.copy(),
        reference_timestamps=empty_t.copy(),
        synchronized_timestamps=empty_t.copy(),
        source_angvels=empty_v.copy(),
        reference_angvels=empty_v.copy(),
        source_angvels_synchronized=empty_v.copy(),
        reference_angvels_synchronized=empty_v.copy(),
        source_angvels_aligned=empty_v.copy(),
        residuals=empty_v.copy(),
        residual_rmse=np.full(3, np.nan),
        residual_vector_rmse=None,
        residual_vector_median=None,
        excitation_singular_values=np.full(3, np.nan),
        excitation_ratios=(np.nan, np.nan),
        success=False,
        message=str(message),
        diagnostics={} if diagnostics is None else dict(diagnostics),
    )


def _as_timestamps(timestamps: ArrayLike, name: str) -> NDArray[np.float64]:
    '''Validate strictly increasing one-dimensional timestamps.

    Args:
        timestamps: Timestamp sequence with shape ``(N,)``.
        name: Name used in validation errors.

    Returns:
        Validated timestamp copy.
    '''

    # Normalize and copy first; the module never mutates caller-owned arrays.
    values = np.asarray(timestamps, dtype=float).reshape(-1)
    if values.size < 2:
        raise ValueError(f'{name} must contain at least two timestamps')
    if not np.all(np.isfinite(values)):
        raise ValueError(f'{name} must contain only finite values')
    if np.any(np.diff(values) <= 0.0):
        raise ValueError(f'{name} must be strictly increasing')
    return values.copy()


def _as_vector_stream(values: ArrayLike, timestamps: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    '''Validate an ``Nx3`` vector stream paired with timestamps.

    Args:
        values: Vector samples with shape ``(N, 3)``.
        timestamps: Matching timestamp array with shape ``(N,)``.
        name: Name used in validation errors.

    Returns:
        Validated vector-stream copy.
    '''

    # Keep shape validation close to timestamp validation for readable failures.
    array = np.asarray(values, dtype=float)
    if array.shape != (timestamps.size, 3):
        raise ValueError(f'{name} must have shape ({timestamps.size}, 3), got {array.shape}')
    if not np.all(np.isfinite(array)):
        raise ValueError(f'{name} must contain only finite values')
    return array.copy()


def _as_pose_stack(values: ArrayLike, name: str) -> NDArray[np.float64]:
    '''Validate an ``Nx4x4`` pose stack.

    Args:
        values: Pose matrices with shape ``(N, 4, 4)``.
        name: Name used in validation errors.

    Returns:
        Validated pose stack copy.
    '''

    # Notebook pipelines pass accumulated LiDAR odometry poses, one pose per timestamp.
    poses = np.asarray(values, dtype=float)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f'{name} must have shape (N, 4, 4)')
    if not np.all(np.isfinite(poses)):
        raise ValueError(f'{name} must contain only finite values')
    return poses.copy()


def _interp_vectors(timestamps: NDArray[np.float64], values: NDArray[np.float64], query_timestamps: NDArray[np.float64]) -> NDArray[np.float64]:
    '''Interpolate a vector stream component-wise.

    Args:
        timestamps: Strictly increasing sample timestamps, shape ``(N,)``.
        values: Vector samples, shape ``(N, 3)``.
        query_timestamps: Query timestamps, shape ``(M,)``.

    Returns:
        Interpolated vectors with shape ``(M, 3)``.
    '''

    # ``np.interp`` is scalar, so interpolate each component explicitly.
    return np.column_stack([np.interp(query_timestamps, timestamps, values[:, axis]) for axis in range(3)])


def _window_mask(timestamps: NDArray[np.float64], start: float, end: float) -> NDArray[np.bool_]:
    '''Return an inclusive mask for one time window.

    Args:
        timestamps: Timestamp array with shape ``(N,)``.
        start: Window start time.
        end: Window end time.

    Returns:
        Boolean mask with shape ``(N,)``.
    '''

    # Inclusive boundaries keep exact edge samples available for interpolation.
    return (timestamps >= float(start)) & (timestamps <= float(end))

def estimate_angular_velocity_bias(
    angular_velocity_imu: ArrayLike,
    imu_timestamps: ArrayLike | None = None,
    *,
    mode: BiasMode = 'provided',
    provided_bias: ArrayLike | None = None,
    stationary_interval: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    '''Estimate or select the constant gyroscope bias used by coarse calibration.

    Args:
        angular_velocity_imu: IMU angular velocity samples, shape ``(N, 3)``.
        imu_timestamps: IMU timestamps, required for ``mode='stationary'``.
        mode: ``'provided'`` to use ``provided_bias`` or ``'stationary'`` to average a known still interval.
        provided_bias: Supplied bias estimate with shape ``(3,)``.
        stationary_interval: Inclusive ``(start, end)`` interval used by the stationary estimator.

    Returns:
        Bias estimate with shape ``(3,)``.
    '''

    # Provided mode is used by notebook 16 after estimating one real stationary-start bias.
    if mode == 'provided':
        if provided_bias is None:
            return np.zeros(3, dtype=float)
        bias = np.asarray(provided_bias, dtype=float).reshape(3)
        if not np.all(np.isfinite(bias)):
            raise ValueError('provided_bias must contain only finite values')
        return bias.copy()

    # Stationary mode averages a specified still interval and does not assume every window is still.
    if mode == 'stationary':
        if imu_timestamps is None or stationary_interval is None:
            raise ValueError('imu_timestamps and stationary_interval are required for stationary bias mode')
        timestamps = _as_timestamps(imu_timestamps, 'imu_timestamps')
        values = _as_vector_stream(angular_velocity_imu, timestamps, 'angular_velocity_imu')
        mask = _window_mask(timestamps, stationary_interval[0], stationary_interval[1])
        if np.count_nonzero(mask) == 0:
            raise ValueError('stationary_interval contains no IMU samples')
        return np.mean(values[mask], axis=0)

    raise ValueError("mode must be 'provided' or 'stationary'")


def derive_body_angular_velocity_from_lidar(
    lidar_timestamps: ArrayLike,
    lidar_odometry_poses: ArrayLike,
    *,
    T_B_L: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    '''Derive local/body angular velocity from accumulated LiDAR odometry poses.

    Args:
        lidar_timestamps: One timestamp per accumulated LiDAR pose, shape ``(N,)``.
        lidar_odometry_poses: Accumulated ``T_O_L(t)`` pose stack, shape ``(N, 4, 4)``.
        T_B_L: Optional body-from-LiDAR transform. If provided, rotate ``omega_L`` into B with ``R_B_L``.

    Returns:
        Tuple ``(midpoint_timestamps, omega_B)`` with shapes ``(N-1,)`` and ``(N-1, 3)``.
    '''

    # Use existing transform convention: delta_R = R_k.T @ R_{k+1}, i.e. local/body angular velocity.
    timestamps = _as_timestamps(lidar_timestamps, 'lidar_timestamps')
    poses = _as_pose_stack(lidar_odometry_poses, 'lidar_odometry_poses')
    if poses.shape[0] != timestamps.size:
        raise ValueError('lidar_odometry_poses must have one pose per lidar timestamp')
    omega_L = transform.se3_to_angvels(poses, timestamps)
    midpoint_timestamps = 0.5 * (timestamps[:-1] + timestamps[1:])

    # If B is not the LiDAR frame, rotate local LiDAR vectors by the fixed body-from-LiDAR rotation.
    if T_B_L is None:
        return midpoint_timestamps, omega_L
    T_B_L_matrix = np.asarray(T_B_L, dtype=float)
    if T_B_L_matrix.shape != (4, 4) or not np.all(np.isfinite(T_B_L_matrix)):
        raise ValueError('T_B_L must have shape (4, 4) and finite values')
    omega_B = (T_B_L_matrix[:3, :3] @ omega_L.T).T
    return midpoint_timestamps, omega_B


def _estimate_tau_by_norm_correlation(
    source_timestamps: NDArray[np.float64],
    source_angvels: NDArray[np.float64],
    reference_timestamps: NDArray[np.float64],
    reference_angvels: NDArray[np.float64],
    *,
    max_abs_tau_s: float,
    step_s: float,
) -> tuple[float, dict[str, Any]]:
    '''Estimate tau by correlating angular-speed norms over a tau grid.

    Args:
        source_timestamps: IMU sensor timestamps, shape ``(N,)``.
        source_angvels: Bias-corrected IMU angular velocities, shape ``(N, 3)``.
        reference_timestamps: Body/reference timestamps, shape ``(M,)``.
        reference_angvels: Body angular velocities, shape ``(M, 3)``.
        max_abs_tau_s: Absolute tau search bound in seconds.
        step_s: Positive tau grid spacing in seconds.

    Returns:
        Tuple ``(tau_I, diagnostics)`` in the factor-graph convention.
    '''

    # Factor convention: body/reference time t queries the IMU at source timestamp s = t + tau_I.
    max_abs_tau_s = float(max_abs_tau_s)
    step_s = float(step_s)
    if max_abs_tau_s < 0.0 or step_s <= 0.0:
        raise ValueError('max_abs_tau_s must be nonnegative and step_s must be positive')
    source_norm = np.linalg.norm(source_angvels, axis=1)
    reference_norm = np.linalg.norm(reference_angvels, axis=1)
    tau_grid = np.arange(-max_abs_tau_s, max_abs_tau_s + 0.5 * step_s, step_s, dtype=float)
    scores = np.full(tau_grid.shape, -np.inf, dtype=float)

    # Score each candidate using normalized dot product of zero-mean norm signals over valid overlap.
    for index, tau in enumerate(tau_grid):
        physical_start = max(reference_timestamps[0], source_timestamps[0] - tau)
        physical_end = min(reference_timestamps[-1], source_timestamps[-1] - tau)
        if physical_end <= physical_start:
            continue
        queries = reference_timestamps[(reference_timestamps >= physical_start) & (reference_timestamps <= physical_end)]
        if queries.size < 3:
            continue
        source_values = np.interp(queries + tau, source_timestamps, source_norm)
        reference_values = np.interp(queries, reference_timestamps, reference_norm)
        source_centered = source_values - np.mean(source_values)
        reference_centered = reference_values - np.mean(reference_values)
        denominator = np.linalg.norm(source_centered) * np.linalg.norm(reference_centered)
        if denominator > 0.0:
            scores[index] = float(np.dot(source_centered, reference_centered) / denominator)

    if not np.any(np.isfinite(scores)):
        raise ValueError('no finite temporal-correlation score')
    best_index = int(np.nanargmax(scores))
    return float(tau_grid[best_index]), {
        'fallback_tau_grid': tau_grid,
        'fallback_scores': scores,
        'fallback_best_score': float(scores[best_index]),
    }


def estimate_temporal_offset_twistnsync(
    source_timestamps: ArrayLike,
    source_angvels: ArrayLike,
    reference_timestamps: ArrayLike,
    reference_angvels: ArrayLike,
    *,
    resample: bool = True,
    resample_step_s: float | None = None,
    max_abs_tau_s: float = 1.0,
    tau_grid_step_s: float | None = None,
    prefer_twistnsync: bool = True,
) -> tuple[float, float | None, dict[str, Any]]:
    '''Estimate IMU temporal offset ``tau_I`` from source/reference angular velocities.

    The returned offset follows the factor-graph convention: ``IMU query time = body/reference time + tau_I``.

    Args:
        source_timestamps: IMU sensor timestamps, shape ``(N,)``.
        source_angvels: Bias-corrected IMU angular velocities in I, shape ``(N, 3)``.
        reference_timestamps: Body/reference timestamps, shape ``(M,)``.
        reference_angvels: Body angular velocities in B, shape ``(M, 3)``.
        resample: Passed to ``twistnsync.TimeSync`` when available.
        resample_step_s: Optional resampling/search step in seconds.
        max_abs_tau_s: Absolute tau bound for fallback.
        tau_grid_step_s: Optional tau-grid step for fallback.
        prefer_twistnsync: Whether a finite TwistnSync candidate should override fallback.

    Returns:
        Tuple ``(tau_I, temporal_delay_raw, diagnostics)``.
    '''

    # Validate ordinary arrays; the old MocapData/smartphone containers are deliberately not accepted.
    src_t = _as_timestamps(source_timestamps, 'source_timestamps')
    ref_t = _as_timestamps(reference_timestamps, 'reference_timestamps')
    src_w = _as_vector_stream(source_angvels, src_t, 'source_angvels')
    ref_w = _as_vector_stream(reference_angvels, ref_t, 'reference_angvels')
    if resample_step_s is None:
        resample_step_s = float(min(np.median(np.diff(src_t)), np.median(np.diff(ref_t))))
    fallback_step_s = float(tau_grid_step_s if tau_grid_step_s is not None else max(0.5 * resample_step_s, 1e-4))

    # Always compute a deterministic fallback diagnostic in the exact factor convention.
    fallback_tau, diagnostics = _estimate_tau_by_norm_correlation(
        src_t, src_w, ref_t, ref_w, max_abs_tau_s=max_abs_tau_s, step_s=fallback_step_s
    )
    diagnostics['temporal_method'] = 'norm_correlation_fallback'
    temporal_delay_raw = None
    tau_from_twistnsync = None

    # Adapt TwistnSync's raw delay to tau_I. This is the documented sign point: tau = source_start - reference_start - raw_delay.
    try:
        import twistnsync as tns
        time_sync = tns.TimeSync(src_w, ref_w, src_t, ref_t, bool(resample))
        time_sync.resample(step=None if resample_step_s is None else float(resample_step_s))
        time_sync.obtain_delay()
        temporal_delay_raw = float(time_sync.time_delay)
        tau_from_twistnsync = float(src_t[0] - ref_t[0] - temporal_delay_raw)
        diagnostics.update({
            'twistnsync_available': True,
            'twistnsync_tau_candidate': tau_from_twistnsync,
            'twistnsync_time_delay': temporal_delay_raw,
        })
    except Exception as exc:  # pragma: no cover - exact TwistnSync failures are environment-dependent.
        diagnostics.update({'twistnsync_available': False, 'twistnsync_error': repr(exc)})

    # Use TwistnSync only when it is finite and inside the configured search range; otherwise keep fallback.
    if prefer_twistnsync and tau_from_twistnsync is not None and np.isfinite(tau_from_twistnsync) and abs(tau_from_twistnsync) <= max_abs_tau_s:
        diagnostics['temporal_method'] = 'twistnsync'
        return tau_from_twistnsync, temporal_delay_raw, diagnostics
    return fallback_tau, temporal_delay_raw, diagnostics

def synchronize_angular_velocity_signals(
    source_timestamps: ArrayLike,
    source_angvels: ArrayLike,
    reference_timestamps: ArrayLike,
    reference_angvels: ArrayLike,
    tau_I: float,
    *,
    window_start: float | None = None,
    window_end: float | None = None,
    resample_step_s: float | None = None,
    min_samples: int = 20,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    '''Synchronize source/reference angular velocities using the estimated tau.

    Args:
        source_timestamps: IMU sensor timestamps, shape ``(N,)``.
        source_angvels: Bias-corrected IMU angular velocities in I, shape ``(N, 3)``.
        reference_timestamps: Body/reference timestamps, shape ``(M,)``.
        reference_angvels: Body angular velocities in B, shape ``(M, 3)``.
        tau_I: Temporal offset in the convention ``s_imu = t_body + tau_I``.
        window_start: Optional physical/body synchronization start.
        window_end: Optional physical/body synchronization end.
        resample_step_s: Optional common body-time step.
        min_samples: Minimum number of synchronized samples.

    Returns:
        Tuple ``(body_times, source_sync, reference_sync)``.
    '''

    # Determine physical overlap where reference(t) and source(t + tau_I) are both supported.
    src_t = _as_timestamps(source_timestamps, 'source_timestamps')
    ref_t = _as_timestamps(reference_timestamps, 'reference_timestamps')
    src_w = _as_vector_stream(source_angvels, src_t, 'source_angvels')
    ref_w = _as_vector_stream(reference_angvels, ref_t, 'reference_angvels')
    tau_I = float(tau_I)
    if not np.isfinite(tau_I):
        raise ValueError('tau_I must be finite')
    start = max(ref_t[0], src_t[0] - tau_I)
    end = min(ref_t[-1], src_t[-1] - tau_I)
    if window_start is not None:
        start = max(start, float(window_start))
    if window_end is not None:
        end = min(end, float(window_end))
    if end <= start:
        raise ValueError('no temporal overlap remains after applying tau_I')

    # Build an explicit common body-time grid; by default use the denser stream's median spacing.
    if resample_step_s is None:
        resample_step_s = float(min(np.median(np.diff(src_t)), np.median(np.diff(ref_t))))
    if resample_step_s <= 0.0 or not np.isfinite(resample_step_s):
        raise ValueError('resample_step_s must be positive and finite')
    count = int(np.floor((end - start) / resample_step_s)) + 1
    body_times = start + np.arange(count, dtype=float) * float(resample_step_s)
    body_times = body_times[body_times <= end]
    if body_times.size < min_samples:
        raise ValueError(f'too few synchronized samples: {body_times.size} < {min_samples}')

    # Source query uses the factor convention directly: IMU timestamp = body time + tau_I.
    source_sync = _interp_vectors(src_t, src_w, body_times + tau_I)
    reference_sync = _interp_vectors(ref_t, ref_w, body_times)
    return body_times, source_sync, reference_sync


def estimate_rotation_align_vectors(
    source_angvels: ArrayLike,
    reference_angvels: ArrayLike,
) -> tuple[NDArray[np.float64], float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float, float]:
    '''Estimate ``R_B_I`` with SciPy ``align_vectors``.

    Args:
        source_angvels: IMU angular velocities in I, shape ``(N, 3)``.
        reference_angvels: Body angular velocities in B, shape ``(N, 3)``.

    Returns:
        Tuple ``(R_B_I, rssd, aligned, residuals, component_rmse, vector_rmse, vector_median)``.
    '''

    # Source/reference order mirrors the old helper: align_vectors(reference, source) returns source-to-reference rotation.
    source = np.asarray(source_angvels, dtype=float)
    reference = np.asarray(reference_angvels, dtype=float)
    if source.shape != reference.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError('source_angvels and reference_angvels must both have shape (N, 3)')
    if source.shape[0] < 2:
        raise ValueError('at least two vector pairs are required for align_vectors')
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(reference)):
        raise ValueError('angular velocity streams must contain only finite values')

    # SciPy returns R such that reference ~= R @ source, exactly R_B_I for this problem.
    rotation, rssd = Rotation.align_vectors(reference, source)
    R_B_I = rotation.as_matrix()
    if not np.all(np.isfinite(R_B_I)) or np.linalg.det(R_B_I) <= 0.0:
        raise ValueError('SciPy returned an invalid rotation')
    aligned = (R_B_I @ source.T).T
    residuals = reference - aligned
    residual_rmse = np.sqrt(np.mean(residuals**2, axis=0))
    residual_norms = np.linalg.norm(residuals, axis=1)
    vector_rmse = float(np.sqrt(np.mean(residual_norms**2)))
    vector_median = float(np.median(residual_norms))
    return R_B_I, float(rssd), aligned, residuals, residual_rmse, vector_rmse, vector_median


def _excitation_diagnostics(source_angvels_synchronized: NDArray[np.float64]) -> tuple[NDArray[np.float64], tuple[float, float]]:
    '''Compute singular-value excitation diagnostics for synchronized IMU vectors.

    Args:
        source_angvels_synchronized: Synchronized IMU angular velocities, shape ``(N, 3)``.

    Returns:
        Tuple ``(singular_values, (s2/s1, s3/s1))``.
    '''

    # Center first so the SVD describes rotational excitation instead of a constant offset.
    centered = source_angvels_synchronized - np.mean(source_angvels_synchronized, axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    padded = np.zeros(3, dtype=float)
    padded[: min(3, singular_values.size)] = singular_values[:3]
    ratios = (float(padded[1] / padded[0]), float(padded[2] / padded[0])) if padded[0] > 0.0 else (np.nan, np.nan)
    return padded, ratios


def estimate_imu_calibration_numerical(
    *,
    window_start: float,
    window_end: float,
    imu_timestamps: ArrayLike,
    angular_velocity_imu: ArrayLike,
    lidar_timestamps: ArrayLike,
    lidar_odometry_poses: ArrayLike,
    T_B_L: ArrayLike | None = None,
    bias_g: ArrayLike | None = None,
    bias_mode: BiasMode = 'provided',
    stationary_interval: tuple[float, float] | None = None,
    T_B_I_translation: ArrayLike | None = None,
    T_B_I_previous: ArrayLike | None = None,
    config: NumericalCalibrationConfig | None = None,
) -> NumericalCalibrationResult:
    '''Estimate numerical coarse ``tau_I`` and ``R_B_I`` for one calibration window.

    Args:
        window_start: Body/reference-time window start.
        window_end: Body/reference-time window end.
        imu_timestamps: IMU sensor timestamps, shape ``(N,)``.
        angular_velocity_imu: IMU angular velocities in I, shape ``(N, 3)``.
        lidar_timestamps: Accumulated LiDAR odometry timestamps, shape ``(M,)``.
        lidar_odometry_poses: Accumulated LiDAR poses ``T_O_L(t)``, shape ``(M, 4, 4)``.
        T_B_L: Known body-from-LiDAR transform; ``None`` means ``B == L``.
        bias_g: Supplied bias when ``bias_mode='provided'``.
        bias_mode: ``'provided'`` or ``'stationary'``.
        stationary_interval: Stationary interval for stationary bias mode.
        T_B_I_translation: Translation to preserve in the constructed prior.
        T_B_I_previous: Previous transform whose translation is preserved when translation is omitted.
        config: Numerical calibration configuration.

    Returns:
        ``NumericalCalibrationResult``. ``success=False`` means callers should fall back to previous calibration.
    '''

    config = NumericalCalibrationConfig() if config is None else config
    try:
        if float(window_end) <= float(window_start):
            return _failure_result('window_end must be greater than window_start')

        # Prepare measurement-only inputs. This high-level estimator has no truth arguments by design.
        imu_t = _as_timestamps(imu_timestamps, 'imu_timestamps')
        imu_w_raw = _as_vector_stream(angular_velocity_imu, imu_t, 'angular_velocity_imu')
        bias_used = estimate_angular_velocity_bias(
            imu_w_raw,
            imu_t,
            mode=bias_mode,
            provided_bias=bias_g,
            stationary_interval=stationary_interval if stationary_interval is not None else config.stationary_interval,
        )
        imu_w = imu_w_raw - bias_used[None, :]
        reference_t_all, reference_w_all = derive_body_angular_velocity_from_lidar(lidar_timestamps, lidar_odometry_poses, T_B_L=T_B_L)

        # Select support with margin; source timestamps are sensor-clock times because tau is still unknown.
        margin = float(config.margin_s)
        source_mask = _window_mask(imu_t, float(window_start) - margin, float(window_end) + margin)
        reference_mask = _window_mask(reference_t_all, float(window_start) - margin, float(window_end) + margin)
        if np.count_nonzero(source_mask) < config.min_samples:
            return _failure_result('too few IMU samples in numerical calibration window')
        if np.count_nonzero(reference_mask) < config.min_samples:
            return _failure_result('too few LiDAR/body angular velocity samples in numerical calibration window')
        source_t = imu_t[source_mask]
        source_w = imu_w[source_mask]
        reference_t = reference_t_all[reference_mask]
        reference_w = reference_w_all[reference_mask]

        # Estimate tau, synchronize at body/reference times, and solve source-to-reference rotation.
        tau_I, temporal_delay_raw, temporal_diagnostics = estimate_temporal_offset_twistnsync(
            source_t,
            source_w,
            reference_t,
            reference_w,
            resample=config.resample,
            resample_step_s=config.resample_step_s,
            max_abs_tau_s=config.max_abs_tau_s,
            tau_grid_step_s=config.tau_grid_step_s,
            prefer_twistnsync=config.prefer_twistnsync,
        )
        synchronized_t, source_sync, reference_sync = synchronize_angular_velocity_signals(
            source_t,
            source_w,
            reference_t,
            reference_w,
            tau_I,
            window_start=window_start,
            window_end=window_end,
            resample_step_s=config.resample_step_s,
            min_samples=config.min_samples,
        )
        R_B_I, spatial_rssd, aligned, residuals, residual_rmse, vector_rmse, vector_median = estimate_rotation_align_vectors(source_sync, reference_sync)
        excitation_singular_values, excitation_ratios = _excitation_diagnostics(source_sync)

        # Construct T_B_I prior by replacing only rotation. Translation comes from estimator state, not artificial truth.
        if T_B_I_translation is not None:
            translation = np.asarray(T_B_I_translation, dtype=float).reshape(3)
        elif T_B_I_previous is not None:
            previous = np.asarray(T_B_I_previous, dtype=float)
            if previous.shape != (4, 4):
                raise ValueError('T_B_I_previous must have shape (4, 4)')
            translation = previous[:3, 3].copy()
        else:
            translation = np.zeros(3, dtype=float)
        if not np.all(np.isfinite(translation)):
            raise ValueError('T_B_I translation must contain only finite values')
        T_B_I = np.eye(4, dtype=float)
        T_B_I[:3, :3] = R_B_I
        T_B_I[:3, 3] = translation

        return NumericalCalibrationResult(
            tau_I=float(tau_I),
            R_B_I=R_B_I,
            T_B_I=T_B_I,
            bias_g_used=bias_used,
            temporal_delay_raw=temporal_delay_raw,
            spatial_rssd=spatial_rssd,
            source_timestamps=source_t,
            reference_timestamps=reference_t,
            synchronized_timestamps=synchronized_t,
            source_angvels=source_w,
            reference_angvels=reference_w,
            source_angvels_synchronized=source_sync,
            reference_angvels_synchronized=reference_sync,
            source_angvels_aligned=aligned,
            residuals=residuals,
            residual_rmse=residual_rmse,
            residual_vector_rmse=vector_rmse,
            residual_vector_median=vector_median,
            excitation_singular_values=excitation_singular_values,
            excitation_ratios=excitation_ratios,
            success=True,
            message='ok',
            diagnostics=temporal_diagnostics,
        )
    except Exception as exc:
        return _failure_result(str(exc), diagnostics={'exception_type': type(exc).__name__})
