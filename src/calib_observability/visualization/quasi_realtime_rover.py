'''Quasi-realtime rover observability visualization helpers.

The module separates the workflow into two layers. Rolling-window helpers
compute diagnostics as data becomes available, while plotting helpers turn the
stored snapshots into notebook series, static figures, and animations.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import animation
import gc


from ..assembly import JacobianBundle
from ..factor_observability import (
    SUPPORTED_CALIBRATION_VARIABLES,
    NormalizationMode,
    build_accelerometer_motion_sensitivity,
    build_imu_gyro_motion_sensitivity,
    build_lidar_motion_only_matrix_dense,
    effective_target_observability_from_bundle_dense,
    effective_target_observability_from_bundle_sparse,
)
from ..observability import effective_observability_dense, effective_observability_sparse_lsmr
from ..diagnostics import DEFAULT_PRACTICAL_RANK_POLICY, PracticalRankPolicy, coordinate_metadata_for_variable, validate_stored_rank_against_matrix
from ..scaling import ParameterScales
from ..types import AccelerometerOptions, FixedExtrinsic, JacobianOptions

##################################################
# Display conventions
##################################################



VARIABLE_DISPLAY_LABELS = {
    "T_B_I": "O_T_B_I",
    "T_B_L": "O_T_B_L",
    "b_g": "O_b_g",
    "tau_I": "O_tau_I",
    "tau_L": "O_tau_L",
}
VARIABLE_MAX_RANKS = {
    "T_B_I": 6,
    "T_B_L": 6,
    "b_g": 3,
    "tau_I": 1,
    "tau_L": 1,
}
J_C_DISPLAY_COLUMN_ORDER = ("T_B_I", "b_g", "tau_I", "tau_L")
J_C_FACTOR_FAMILY_ORDER = (
    ("lidar", "LiDAR"),
    ("gyro", "gyro"),
    ("accelerometer", "accel"),
)


##################################################
# Runtime configuration and snapshots
##################################################


@dataclass(frozen=True)
class QuasiRealtimeConfig:
    '''Configure rolling-window quasi-realtime diagnostics.

    Attributes:
        window_length (float): Duration of the active analysis window in seconds.
        frame_step (float): Time increment between consecutive snapshots.
        use_sparse (bool): Whether to assemble and project sparse Jacobians.
        relative_rank_threshold (float): Relative threshold retained for legacy
            rank displays.
        normalization (NormalizationMode): Target-matrix normalization mode.
        display_variables (tuple[str, ...]): Calibration variables shown in the
            dashboard.
        max_display_rows (int): Maximum rows retained in heatmap arrays.
        max_display_cols (int): Maximum columns retained in heatmap arrays.
        jacobian_options (JacobianOptions | None): Factor Jacobian configuration.
        fixed_extrinsic (FixedExtrinsic): Body-frame convention used by the dataset.
        practical_rank_policy (PracticalRankPolicy): Canonical practical-rank
            thresholds.
        parameter_scales (ParameterScales): Physical coordinate scales supplied to
            dataset assembly.
        tau_target_std_seconds (float): Target local timing standard deviation.
        lidar_rate_hz (float | None): Optional LiDAR rate used to express timing
            bounds in frames.
        coordinate_null_fraction_tolerance (float): Maximum null-space fraction for
            declaring a coordinate fully bounded.
        show_local_accuracy_summary (bool): Whether dashboard text includes local
            CRLB-like coordinate summaries.
        accelerometer_options (AccelerometerOptions | None): Optional accelerometer
            factor configuration.
        normalize_J_C_factor_blocks_for_display (bool): Whether the displayed
            copy of ``J_C`` is independently normalized per LiDAR, gyro, and
            accelerometer row family. This is visualization-only and never
            changes physical Jacobians or diagnostics.
    '''

    window_length: float = 6.0
    frame_step: float = 1.0
    use_sparse: bool = False
    relative_rank_threshold: float = 1e-5
    normalization: NormalizationMode = "physical_then_column"
    display_variables: tuple[str, ...] = ("T_B_I", "b_g", "tau_I", "tau_L")
    max_display_rows: int = 80
    max_display_cols: int = 40
    jacobian_options: JacobianOptions | None = None
    fixed_extrinsic: FixedExtrinsic = "T_B_L"
    practical_rank_policy: PracticalRankPolicy = DEFAULT_PRACTICAL_RANK_POLICY
    parameter_scales: ParameterScales = ParameterScales()
    tau_target_std_seconds: float = 0.2
    lidar_rate_hz: float | None = None
    coordinate_null_fraction_tolerance: float = 1e-6
    show_local_accuracy_summary: bool = True
    accelerometer_options: AccelerometerOptions | None = None
    normalize_J_C_factor_blocks_for_display: bool = True


@dataclass(frozen=True)
class MatrixDisplayLayout:
    '''Store semantic separators and labels for one displayed matrix.

    Boundary values are counts in displayed coordinates. A row boundary ``b`` is
    drawn at imshow coordinate ``y = b - 0.5``; a column boundary is drawn at
    ``x = b - 0.5``. Empty tuples mean ordinary heatmap display without semantic
    separators. Display rows may be downsampled independently inside each
    semantic block.
    '''

    row_boundaries: tuple[int, ...] = ()
    row_centers: tuple[float, ...] = ()
    row_labels: tuple[str, ...] = ()
    column_boundaries: tuple[int, ...] = ()
    column_centers: tuple[float, ...] = ()
    column_labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class MatrixDisplayResult:
    '''Store a display-only matrix together with semantic layout metadata.

    ``block_scales`` records any display-only normalization scales. Numerical
    observability, information, rank, singular-value, and local-accuracy
    calculations continue to use the physical source matrices.
    '''

    matrix: NDArray[np.float64]
    layout: MatrixDisplayLayout = field(default_factory=MatrixDisplayLayout)
    block_scales: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class MotionMatrixFrame:
    '''Store the motion-sensitivity heatmap selected for one animation frame.'''

    matrix: NDArray[np.float64]
    title: str
    layout: MatrixDisplayLayout = field(default_factory=MatrixDisplayLayout)


@dataclass
class QuasiRealtimeSnapshot:
    '''Store diagnostics for one current time and active window.

    Attributes:
        current_time (float): Current quasi-realtime frame time.
        window_start (float): Start of the active prefix or rolling window.
        window_end (float): End of the active window.
        is_valid (bool): Whether at least one target result is available.
        status (str): Human-readable snapshot state.
        counts (dict[str, int]): Factor counts returned by dataset assembly.
        machine_ranks (dict[str, float]): Machine-rank values by variable.
        effective_ranks (dict[str, float]): Practical-rank values by variable.
        condition_numbers (dict[str, float]): Practical condition numbers by
            variable.
        J_C_display (NDArray[np.float64]): Downsampled calibration Jacobian.
        C_X_L_display (NDArray[np.float64]): Downsampled LiDAR motion sensitivity.
        C_X_I_gyro_display (NDArray[np.float64]): Downsampled gyro sensitivity.
        target_O_display (dict[str, NDArray[np.float64]]): Downsampled projected
            target matrices.
        bundle (JacobianBundle | None): Full Jacobian bundle for this window.
        motion_lidar (Any | None): LiDAR motion-only diagnostics.
        motion_imu (Any | None): Gyroscope motion-sensitivity diagnostics.
        motion_accelerometer (Any | None): Accelerometer sensitivity diagnostics.
        target_results (dict[str, Any]): Per-variable observability results.
        trace_information (dict[str, float]): Physical information traces.
        worst_std_bounds (dict[str, float]): Worst retained standard-deviation
            bounds.
        tau_std_bounds (dict[str, float]): Scalar timing bounds in seconds.
        tau_std_bounds_lidar_frames (dict[str, float]): Timing bounds in LiDAR
            frames.
        tau_target_ratio (dict[str, float]): Timing bound divided by its target.
        tau_meets_target (dict[str, bool]): Timing target decisions.
        local_accuracy_by_variable (dict[str, Any]): Local CRLB-like diagnostics.
        accelerometer_mode (str): Active accelerometer factor mode.
        accelerometer_factor_count (int): Accepted accelerometer factor count.
        accelerometer_candidate_count (int): Candidate accelerometer sample count.
        accelerometer_gate_mask (NDArray[np.bool_]): Candidate acceptance mask.
        C_X_I_accel_display (NDArray[np.float64]): Downsampled accelerometer
            sensitivity matrix.
        accelerometer_factor_terms (tuple[Any, ...]): Optional saved factor terms.
        accelerometer_jacobian_check_diagnostics (tuple[Any, ...]): Stored
            accelerometer Jacobian checks.
    '''

    current_time: float
    window_start: float
    window_end: float
    is_valid: bool
    status: str
    counts: dict[str, int]
    machine_ranks: dict[str, float]
    effective_ranks: dict[str, float]
    condition_numbers: dict[str, float]
    J_C_display: NDArray[np.float64]
    C_X_L_display: NDArray[np.float64]
    C_X_I_gyro_display: NDArray[np.float64]
    J_C_display_layout: MatrixDisplayLayout = field(default_factory=MatrixDisplayLayout)
    C_X_L_display_layout: MatrixDisplayLayout = field(default_factory=MatrixDisplayLayout)
    C_X_I_gyro_display_layout: MatrixDisplayLayout = field(default_factory=MatrixDisplayLayout)
    C_X_I_accel_display_layout: MatrixDisplayLayout = field(default_factory=MatrixDisplayLayout)
    J_C_display_block_scales: dict[str, float] = field(default_factory=dict)
    target_O_display: dict[str, NDArray[np.float64]] = field(default_factory=dict)
    bundle: JacobianBundle | None = None
    motion_lidar: Any | None = None
    motion_imu: Any | None = None
    motion_accelerometer: Any | None = None
    target_results: dict[str, Any] = field(default_factory=dict)
    trace_information: dict[str, float] = field(default_factory=dict)
    worst_std_bounds: dict[str, float] = field(default_factory=dict)
    tau_std_bounds: dict[str, float] = field(default_factory=dict)
    tau_std_bounds_lidar_frames: dict[str, float] = field(default_factory=dict)
    tau_target_ratio: dict[str, float] = field(default_factory=dict)
    tau_meets_target: dict[str, bool] = field(default_factory=dict)
    local_accuracy_by_variable: dict[str, Any] = field(default_factory=dict)
    accelerometer_mode: str = "disabled"
    accelerometer_factor_count: int = 0
    accelerometer_candidate_count: int = 0
    accelerometer_gate_mask: NDArray[np.bool_] = field(default_factory=lambda: np.zeros(0, dtype=bool))
    C_X_I_accel_display: NDArray[np.float64] = field(default_factory=lambda: np.zeros((1, 1), dtype=float))
    accelerometer_factor_terms: tuple[Any, ...] = ()
    accelerometer_jacobian_check_diagnostics: tuple[Any, ...] = ()



##################################################
# Rolling-window construction
##################################################

def rolling_window_bounds(
    current_time: float,
    dataset_start_time: float,
    window_length: float,
) -> tuple[float, float]:
    '''Compute prefix-then-rolling bounds for one frame.

    Args:
        current_time (float): Current frame time.
        dataset_start_time (float): First available dataset time.
        window_length (float): Maximum active-window duration.

    Returns:
        tuple[float, float]: Window start and current-time window end.

    Raises:
        ValueError: If a time is non-finite or `window_length` is not positive.
    '''

    current = float(current_time)
    dataset_start = float(dataset_start_time)
    length = float(window_length)
    if not np.isfinite([current, dataset_start, length]).all() or length <= 0.0:
        raise ValueError("current_time, dataset_start_time, and positive window_length must be finite")
    window_start = max(dataset_start, current - length)
    return window_start, current


def matrix_for_display(
    matrix: ArrayLike | sparse.spmatrix | None,
    *,
    max_rows: int = 80,
    max_cols: int = 40,
) -> NDArray[np.float64]:
    '''Convert a matrix into a finite downsampled heatmap array.

    Args:
        matrix (ArrayLike | sparse.spmatrix | None): Dense or sparse source matrix.
            `None` and empty matrices produce a `(1, 1)` zero placeholder.
        max_rows (int): Maximum number of displayed rows.
        max_cols (int): Maximum number of displayed columns.

    Returns:
        NDArray[np.float64]: Finite two-dimensional display matrix.

    Raises:
        ValueError: If `matrix` is not two-dimensional or a display limit is not
            positive.

    Notes:
        Downsampling affects visualization only. Numerical diagnostics continue to
        use the original matrix.
    '''

    if matrix is None:
        return np.zeros((1, 1), dtype=float)
    dense_matrix = matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix, dtype=float)
    if dense_matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    if dense_matrix.size == 0:
        return np.zeros((1, 1), dtype=float)

    # Downsample only for display. The diagnostics still use the original
    # matrices, while the dashboard remains readable as the window grows.
    row_count, column_count = dense_matrix.shape
    row_indices = _display_indices(row_count, max_rows)
    column_indices = _display_indices(column_count, max_cols)
    display_matrix = dense_matrix[np.ix_(row_indices, column_indices)]
    return np.nan_to_num(display_matrix, nan=0.0, posinf=0.0, neginf=0.0)


def compute_quasi_realtime_snapshots(
    dataset: object,
    pose_provider: object,
    config: QuasiRealtimeConfig | None = None,
) -> list[QuasiRealtimeSnapshot]:
    '''Compute growing-prefix and rolling-window snapshots.

    Args:
        dataset (object): Dataset exposing `start_time`, `end_time`, and
            `window_jacobians`.
        pose_provider (object): Continuous provider used by dataset assembly.
        config (QuasiRealtimeConfig | None): Runtime configuration. Defaults are
            used when omitted.

    Returns:
        list[QuasiRealtimeSnapshot]: Snapshots ordered by current time.

    Raises:
        ValueError: If `frame_step` is not positive.
    '''

    runtime_config = QuasiRealtimeConfig() if config is None else config
    if runtime_config.frame_step <= 0.0:
        raise ValueError("frame_step must be positive")
    dataset_start_time = float(getattr(dataset, "start_time"))
    dataset_end_time = float(getattr(dataset, "end_time"))
    frame_times = np.arange(
        dataset_start_time,
        dataset_end_time + 0.5 * runtime_config.frame_step,
        runtime_config.frame_step,
    )
    if frame_times.size == 0 or frame_times[-1] < dataset_end_time:
        frame_times = np.r_[frame_times, dataset_end_time]

    snapshots: list[QuasiRealtimeSnapshot] = []
    for current_time in frame_times:
        window_start, window_end = rolling_window_bounds(
            float(current_time),
            dataset_start_time,
            runtime_config.window_length,
        )
        snapshots.append(
            build_window_snapshot(
                dataset,
                pose_provider,
                current_time=float(current_time),
                window_start=window_start,
                window_end=window_end,
                config=runtime_config,
            )
        )
    return snapshots


def build_window_snapshot(
    dataset: object,
    pose_provider: object,
    *,
    current_time: float,
    window_start: float,
    window_end: float,
    config: QuasiRealtimeConfig,
) -> QuasiRealtimeSnapshot:
    '''Assemble one window and extract dashboard diagnostics.

    Args:
        dataset (object): Dataset exposing `window_jacobians`.
        pose_provider (object): Pose and twist provider used during assembly.
        current_time (float): Current frame time.
        window_start (float): Start of the active analysis window.
        window_end (float): End of the active analysis window.
        config (QuasiRealtimeConfig): Runtime diagnostic configuration.

    Returns:
        QuasiRealtimeSnapshot: Valid diagnostics or a structured waiting snapshot.

    Notes:
        Expected data-availability and numerical failures are converted into
        invalid snapshots so animations can continue while a window accumulates
        enough factors.
    '''

    empty_ranks = _nan_by_variable(config.display_variables)
    empty_condition_numbers = _nan_by_variable(config.display_variables)
    if window_end <= window_start:
        return _empty_snapshot(
            current_time,
            window_start,
            window_end,
            "waiting for a nonzero analysis interval",
            config,
            empty_ranks,
            empty_ranks,
            empty_condition_numbers,
        )

    try:
        # Assemble all factors available inside the active prefix/rolling
        # window. The dataset keeps IMU and LiDAR attached to shared pose nodes.
        bundle, body_motions, counts = dataset.window_jacobians(
            window_start,
            window_end,
            pose_provider,
            use_sparse=config.use_sparse,
            jacobian_options=config.jacobian_options,
            fixed_extrinsic=config.fixed_extrinsic,
            practical_rank_policy=config.practical_rank_policy,
            parameter_scaling=config.parameter_scales,
            accelerometer_options=config.accelerometer_options,
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return _empty_snapshot(
            current_time,
            window_start,
            window_end,
            f"waiting for enough factors: {exc}",
            config,
            empty_ranks,
            empty_ranks,
            empty_condition_numbers,
        )

    has_factor_rows = bundle.J_C.shape[0] > 0 and sum(counts.values()) > 0
    if not has_factor_rows:
        return _empty_snapshot(
            current_time,
            window_start,
            window_end,
            "waiting for enough factors",
            config,
            empty_ranks,
            empty_ranks,
            empty_condition_numbers,
            counts=counts,
            bundle=bundle,
        )

    # Compute motion-only factor diagnostics. These are not joint observability,
    # but they are useful live indicators of what each sensor family contributes.
    motion_lidar = None
    if counts.get("lidar", 0) > 0:
        motion_lidar = build_lidar_motion_only_matrix_dense(
            body_motions,
            relative_rank_threshold=config.relative_rank_threshold,
            practical_rank_policy=config.practical_rank_policy,
        )

    motion_imu = None
    if counts.get("imu", 0) > 0 and "T_B_I" in bundle.calibration_column_slices:
        motion_imu = build_imu_gyro_motion_sensitivity(
            bundle,
            relative_rank_threshold=config.relative_rank_threshold,
            practical_rank_policy=config.practical_rank_policy,
        )

    motion_accelerometer = None
    if counts.get("accelerometer_factor_count", 0) > 0 and "T_B_I" in bundle.calibration_column_slices:
        motion_accelerometer = build_accelerometer_motion_sensitivity(
            bundle,
            practical_rank_policy=config.practical_rank_policy,
        )

    machine_ranks = _nan_by_variable(config.display_variables)
    effective_ranks = _nan_by_variable(config.display_variables)
    condition_numbers = _nan_by_variable(config.display_variables)
    target_results: dict[str, Any] = {}
    target_O_display: dict[str, NDArray[np.float64]] = {}
    trace_information = _nan_by_variable(config.display_variables)
    worst_std_bounds = _nan_by_variable(config.display_variables)
    tau_std_bounds = _nan_by_variable(config.display_variables)
    tau_std_bounds_lidar_frames = _nan_by_variable(config.display_variables)
    tau_target_ratio = _nan_by_variable(config.display_variables)
    tau_meets_target = {variable_name: False for variable_name in config.display_variables}
    local_accuracy_by_variable: dict[str, Any] = {}

    # Project each target variable against trajectory poses and the remaining
    # calibration variables, then keep compact matrices for visual inspection.
    for variable_name in config.display_variables:
        if variable_name not in bundle.calibration_column_slices:
            continue
        try:
            result = (
                effective_target_observability_from_bundle_sparse(
                    bundle,
                    variable_name,
                    normalization=config.normalization,
                    relative_rank_threshold=config.relative_rank_threshold,
                    practical_rank_policy=config.practical_rank_policy,
                    tau_target_std_seconds=config.tau_target_std_seconds,
                    lidar_rate_hz=config.lidar_rate_hz,
                    coordinate_null_fraction_tolerance=config.coordinate_null_fraction_tolerance,
                )
                if config.use_sparse
                else effective_target_observability_from_bundle_dense(
                    bundle,
                    variable_name,
                    normalization=config.normalization,
                    relative_rank_threshold=config.relative_rank_threshold,
                    practical_rank_policy=config.practical_rank_policy,
                    tau_target_std_seconds=config.tau_target_std_seconds,
                    lidar_rate_hz=config.lidar_rate_hz,
                    coordinate_null_fraction_tolerance=config.coordinate_null_fraction_tolerance,
                )
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            continue
        target_results[variable_name] = result
        local_accuracy_by_variable[variable_name] = result.local_accuracy_diagnostics
        machine_ranks[variable_name] = float(result.machine_rank_O_X)
        effective_ranks[variable_name] = float(result.effective_rank_O_X)
        condition_numbers[variable_name] = float(result.practical_rank_diagnostics.practical_condition_number)
        trace_information[variable_name] = float(result.physical_information_diagnostics.trace_information)
        worst_std_bounds[variable_name] = float(result.physical_information_diagnostics.worst_retained_std_bound)
        accuracy = result.local_accuracy_diagnostics
        if accuracy.scalar_std_bound is not None:
            tau_std_bounds[variable_name] = float(accuracy.scalar_std_bound)
            tau_std_bounds_lidar_frames[variable_name] = float(accuracy.scalar_std_bound_lidar_frames) if accuracy.scalar_std_bound_lidar_frames is not None else np.nan
            tau_target_ratio[variable_name] = float(accuracy.target_ratio) if accuracy.target_ratio is not None else np.nan
            tau_meets_target[variable_name] = bool(accuracy.meets_target) if accuracy.meets_target is not None else False
        validate_stored_rank_against_matrix(result.O_X_physical, result.practical_rank_diagnostics, config.practical_rank_policy)
        target_O_display[variable_name] = matrix_for_display(
            result.O_X_physical,
            max_rows=config.max_display_rows,
            max_cols=config.max_display_cols,
        )

    J_C_display_result = _semantic_j_c_display(
        bundle.J_C,
        bundle.calibration_column_slices,
        _factor_family_row_slices(bundle),
        max_rows=config.max_display_rows,
        max_cols=config.max_display_cols,
        normalize_factor_blocks=config.normalize_J_C_factor_blocks_for_display,
    )
    C_X_L_display_result = _c_x_display_result(
        None if motion_lidar is None else motion_lidar.C_X_L_column_normalized,
        "lidar",
        max_rows=config.max_display_rows,
        max_cols=config.max_display_cols,
    )
    C_X_I_gyro_display_result = _c_x_display_result(
        None if motion_imu is None else motion_imu.rotation_only_normalized,
        "gyro",
        max_rows=config.max_display_rows,
        max_cols=config.max_display_cols,
    )
    C_X_I_accel_display_result = _c_x_display_result(
        None if motion_accelerometer is None else motion_accelerometer.C_X_I_accel_physical,
        "accelerometer",
        max_rows=config.max_display_rows,
        max_cols=config.max_display_cols,
    )
    return QuasiRealtimeSnapshot(
        current_time=float(current_time),
        window_start=float(window_start),
        window_end=float(window_end),
        is_valid=bool(target_results),
        status="ok" if target_results else "waiting for observable target blocks",
        counts=dict(counts),
        machine_ranks=machine_ranks,
        effective_ranks=effective_ranks,
        condition_numbers=condition_numbers,
        J_C_display=J_C_display_result.matrix,
        C_X_L_display=C_X_L_display_result.matrix,
        C_X_I_gyro_display=C_X_I_gyro_display_result.matrix,
        J_C_display_layout=J_C_display_result.layout,
        C_X_L_display_layout=C_X_L_display_result.layout,
        C_X_I_gyro_display_layout=C_X_I_gyro_display_result.layout,
        C_X_I_accel_display_layout=C_X_I_accel_display_result.layout,
        J_C_display_block_scales=J_C_display_result.block_scales,
        target_O_display=target_O_display,
        bundle=bundle,
        motion_lidar=motion_lidar,
        motion_imu=motion_imu,
        motion_accelerometer=motion_accelerometer,
        target_results=target_results,
        trace_information=trace_information,
        worst_std_bounds=worst_std_bounds,
        tau_std_bounds=tau_std_bounds,
        tau_std_bounds_lidar_frames=tau_std_bounds_lidar_frames,
        tau_target_ratio=tau_target_ratio,
        tau_meets_target=tau_meets_target,
        local_accuracy_by_variable=local_accuracy_by_variable,
        accelerometer_mode=str(bundle.metadata.get("accelerometer_mode", "disabled")),
        accelerometer_factor_count=int(bundle.metadata.get("accelerometer_factor_count", 0)),
        accelerometer_candidate_count=int(bundle.metadata.get("accelerometer_candidate_count", 0)),
        accelerometer_gate_mask=np.asarray(bundle.metadata.get("accelerometer_gate_mask", np.zeros(0, dtype=bool)), dtype=bool),
        C_X_I_accel_display=C_X_I_accel_display_result.matrix,
        accelerometer_factor_terms=tuple(bundle.metadata.get("accelerometer_factor_terms", ())),
        accelerometer_jacobian_check_diagnostics=tuple(bundle.metadata.get("accelerometer_jacobian_check_results", ())),
    )



##################################################
# Time-series aggregation
##################################################

def dashboard_series(
    snapshots: list[QuasiRealtimeSnapshot],
    display_variables: tuple[str, ...] = ("T_B_I", "b_g", "tau_I", "tau_L"),
) -> dict[str, object]:
    '''Align snapshot diagnostics into time-series arrays.

    Args:
        snapshots (list[QuasiRealtimeSnapshot]): Ordered snapshot sequence.
        display_variables (tuple[str, ...]): Variables included in output mappings.

    Returns:
        dict[str, object]: Times, ranks, conditions, information, timing bounds,
            coordinate diagnostics, and accelerometer series.
    '''

    times = np.asarray([snapshot.current_time for snapshot in snapshots], dtype=float)
    effective_ranks = {
        variable_name: np.asarray(
            [snapshot.effective_ranks.get(variable_name, np.nan) for snapshot in snapshots],
            dtype=float,
        )
        for variable_name in display_variables
    }
    machine_ranks = {
        variable_name: np.asarray(
            [snapshot.machine_ranks.get(variable_name, np.nan) for snapshot in snapshots],
            dtype=float,
        )
        for variable_name in display_variables
    }
    condition_numbers = {
        variable_name: np.asarray(
            [snapshot.condition_numbers.get(variable_name, np.nan) for snapshot in snapshots],
            dtype=float,
        )
        for variable_name in display_variables
    }
    return {
        "times": times,
        "effective_ranks": effective_ranks,
        "machine_ranks": machine_ranks,
        "condition_numbers": condition_numbers,
        "trace_information": {
            variable_name: np.asarray([snapshot.trace_information.get(variable_name, np.nan) for snapshot in snapshots], dtype=float)
            for variable_name in display_variables
        },
        "worst_std_bounds": {
            variable_name: np.asarray([snapshot.worst_std_bounds.get(variable_name, np.nan) for snapshot in snapshots], dtype=float)
            for variable_name in display_variables
        },
        "tau_std_bounds": {
            variable_name: np.asarray([snapshot.tau_std_bounds.get(variable_name, np.nan) for snapshot in snapshots], dtype=float)
            for variable_name in display_variables
        },
        "tau_std_bounds_lidar_frames": {
            variable_name: np.asarray([snapshot.tau_std_bounds_lidar_frames.get(variable_name, np.nan) for snapshot in snapshots], dtype=float)
            for variable_name in display_variables
        },
        "tau_target_ratio": {
            variable_name: np.asarray([snapshot.tau_target_ratio.get(variable_name, np.nan) for snapshot in snapshots], dtype=float)
            for variable_name in display_variables
        },
        "coordinate_std_bounds": {
            variable_name: _coordinate_accuracy_array(snapshots, variable_name, "coordinate_std_bounds")
            for variable_name in display_variables
        },
        "coordinate_bounded_mask": {
            variable_name: _coordinate_accuracy_array(snapshots, variable_name, "coordinate_is_fully_bounded", fill_value=False, dtype=bool)
            for variable_name in display_variables
        },
        "coordinate_null_fraction": {
            variable_name: _coordinate_accuracy_array(snapshots, variable_name, "coordinate_null_fraction")
            for variable_name in display_variables
        },
        "coordinate_observable_fraction": {
            variable_name: _coordinate_accuracy_array(snapshots, variable_name, "coordinate_observable_fraction")
            for variable_name in display_variables
        },
        "retained_mode_std_bounds": {
            variable_name: [
                snapshot.local_accuracy_by_variable[variable_name].retained_mode_std_bounds if variable_name in snapshot.local_accuracy_by_variable else np.asarray([], dtype=float)
                for snapshot in snapshots
            ]
            for variable_name in display_variables
        },
        "worst_retained_mode_std_bound": {
            variable_name: np.asarray(
                [
                    snapshot.local_accuracy_by_variable[variable_name].worst_retained_mode_std_bound if variable_name in snapshot.local_accuracy_by_variable else np.nan
                    for snapshot in snapshots
                ],
                dtype=float,
            )
            for variable_name in display_variables
        },
        "covariance_kind": {
            variable_name: [
                snapshot.local_accuracy_by_variable[variable_name].covariance_kind if variable_name in snapshot.local_accuracy_by_variable else "missing" for snapshot in snapshots
            ]
            for variable_name in display_variables
        },
        "C_X_I_accel_rank": np.asarray([
            snapshot.motion_accelerometer.practical_rank if snapshot.motion_accelerometer is not None else np.nan
            for snapshot in snapshots
        ], dtype=float),
        "accelerometer_factor_count": np.asarray([snapshot.accelerometer_factor_count for snapshot in snapshots], dtype=int),
    }



@dataclass(frozen=True)

##################################################
# Canonical visualization series
##################################################

class ObservabilityVisualizationSeries:
    '''Store canonical arrays shared by visualization notebooks.

    Attributes:
        times (NDArray[np.float64]): Snapshot times.
        snapshots (list[QuasiRealtimeSnapshot]): Source snapshots.
        ranks (dict[str, NDArray[np.float64]]): Practical ranks by variable.
        condition_numbers (dict[str, NDArray[np.float64]]): Practical conditions.
        trace_information (dict[str, NDArray[np.float64]]): Information traces.
        worst_std_bounds (dict[str, NDArray[np.float64]]): Worst retained bounds.
        tau_std_bounds (dict[str, NDArray[np.float64]]): Timing bounds in seconds.
        tau_std_bounds_lidar_frames (dict[str, NDArray[np.float64]]): Timing bounds
            in LiDAR frames.
        tau_target_ratio (dict[str, NDArray[np.float64]]): Timing target ratios.
        tau_meets_target (dict[str, NDArray[np.bool_]]): Timing target decisions.
        coordinate_std_bounds (dict[str, NDArray[np.float64]]): Per-coordinate
            standard-deviation bounds.
        coordinate_bounded_mask (dict[str, NDArray[np.bool_]]): Fully bounded
            coordinate masks.
        coordinate_null_fraction (dict[str, NDArray[np.float64]]): Coordinate null
            fractions.
        coordinate_observable_fraction (dict[str, NDArray[np.float64]]): Coordinate
            observable fractions.
        retained_mode_std_bounds (dict[str, list[NDArray[np.float64]]]): Retained
            singular-mode bounds per snapshot.
        worst_retained_mode_std_bound (dict[str, NDArray[np.float64]]): Worst mode
            bounds by variable.
        covariance_kind (dict[str, list[str]]): Covariance interpretation labels.
        C_X_L_rank (NDArray[np.float64]): LiDAR motion-only practical rank.
        C_X_I_accel_rank (NDArray[np.float64]): Accelerometer sensitivity rank.
        accelerometer_factor_count (NDArray[np.int_]): Accepted factor counts.
        matrix_gate_passed (dict[str, NDArray[np.bool_]]): Absolute matrix-gate
            decisions by variable.
    '''

    times: NDArray[np.float64]
    snapshots: list[QuasiRealtimeSnapshot]
    ranks: dict[str, NDArray[np.float64]]
    condition_numbers: dict[str, NDArray[np.float64]]
    trace_information: dict[str, NDArray[np.float64]]
    worst_std_bounds: dict[str, NDArray[np.float64]]
    tau_std_bounds: dict[str, NDArray[np.float64]]
    tau_std_bounds_lidar_frames: dict[str, NDArray[np.float64]]
    tau_target_ratio: dict[str, NDArray[np.float64]]
    tau_meets_target: dict[str, NDArray[np.bool_]]
    coordinate_std_bounds: dict[str, NDArray[np.float64]]
    coordinate_bounded_mask: dict[str, NDArray[np.bool_]]
    coordinate_null_fraction: dict[str, NDArray[np.float64]]
    coordinate_observable_fraction: dict[str, NDArray[np.float64]]
    retained_mode_std_bounds: dict[str, list[NDArray[np.float64]]]
    worst_retained_mode_std_bound: dict[str, NDArray[np.float64]]
    covariance_kind: dict[str, list[str]]
    C_X_L_rank: NDArray[np.float64]
    C_X_I_accel_rank: NDArray[np.float64]
    accelerometer_factor_count: NDArray[np.int_]
    matrix_gate_passed: dict[str, NDArray[np.bool_]]


def build_observability_visualization_series(
    dataset: object,
    pose_provider: object,
    *,
    window_duration: float,
    window_step: float,
    fixed_extrinsic: FixedExtrinsic = "T_B_L",
    practical_rank_policy: PracticalRankPolicy = DEFAULT_PRACTICAL_RANK_POLICY,
    parameter_scales: ParameterScales = ParameterScales(),
    tau_target_std_seconds: float = 0.2,
    jacobian_options: JacobianOptions | None = None,
    accelerometer_options: AccelerometerOptions | None = None,
    use_sparse: bool = False,
    display_variables: tuple[str, ...] = ("T_B_I", "b_g", "tau_I", "tau_L"),
    normalization: NormalizationMode = "physical_then_column",
    max_display_rows: int = 80,
    max_display_cols: int = 40,
    lidar_rate_hz: float | None = None,
    coordinate_null_fraction_tolerance: float = 1e-6,
    show_local_accuracy_summary: bool = True,
) -> ObservabilityVisualizationSeries:
    '''Build canonical visualization arrays for notebooks 04 and 07.

    Args:
        dataset (object): Simulation dataset.
        pose_provider (object): Continuous pose and twist provider.
        window_duration (float): Rolling-window duration in seconds.
        window_step (float): Time step between snapshots.
        fixed_extrinsic (FixedExtrinsic): Fixed body-frame convention.
        practical_rank_policy (PracticalRankPolicy): Canonical rank thresholds.
        parameter_scales (ParameterScales): Physical parameter scales.
        tau_target_std_seconds (float): Scalar timing accuracy target.
        jacobian_options (JacobianOptions | None): Jacobian construction options.
        accelerometer_options (AccelerometerOptions | None): Accelerometer factor
            configuration.
        use_sparse (bool): Whether to use sparse assembly and projection.
        display_variables (tuple[str, ...]): Variables included in the series.
        normalization (NormalizationMode): Projected-matrix normalization mode.
        max_display_rows (int): Maximum heatmap rows.
        max_display_cols (int): Maximum heatmap columns.
        lidar_rate_hz (float | None): Optional rate for frame-unit timing bounds.
        coordinate_null_fraction_tolerance (float): Bounded-coordinate tolerance.
        show_local_accuracy_summary (bool): Dashboard local-accuracy flag.

    Returns:
        ObservabilityVisualizationSeries: Canonical snapshots and aligned arrays.

    Notes:
        Rank and condition values are read from stored practical-rank diagnostics
        for the same physical projected matrices represented by each snapshot.
    '''

    config = QuasiRealtimeConfig(
        window_length=window_duration,
        frame_step=window_step,
        use_sparse=use_sparse,
        normalization=normalization,
        display_variables=display_variables,
        jacobian_options=jacobian_options,
        fixed_extrinsic=fixed_extrinsic,
        practical_rank_policy=practical_rank_policy,
        parameter_scales=parameter_scales,
        tau_target_std_seconds=tau_target_std_seconds,
        accelerometer_options=accelerometer_options,
        max_display_rows=max_display_rows,
        max_display_cols=max_display_cols,
        lidar_rate_hz=lidar_rate_hz,
        coordinate_null_fraction_tolerance=coordinate_null_fraction_tolerance,
        show_local_accuracy_summary=show_local_accuracy_summary,
    )
    snapshots = compute_quasi_realtime_snapshots(dataset, pose_provider, config)
    series = dashboard_series(snapshots, display_variables)
    C_X_L_rank = np.asarray([
        snapshot.motion_lidar.practical_rank if snapshot.motion_lidar is not None else np.nan
        for snapshot in snapshots
    ], dtype=float)
    matrix_gate_passed = {
        variable_name: np.asarray([
            bool(snapshot.target_results[variable_name].practical_rank_diagnostics.matrix_passed_absolute_gate)
            if variable_name in snapshot.target_results else False
            for snapshot in snapshots
        ], dtype=bool)
        for variable_name in display_variables
    }
    tau_meets_target = {
        variable_name: np.asarray([snapshot.tau_meets_target.get(variable_name, False) for snapshot in snapshots], dtype=bool)
        for variable_name in display_variables
    }
    return ObservabilityVisualizationSeries(
        times=series["times"],
        snapshots=snapshots,
        ranks=series["effective_ranks"],
        condition_numbers=series["condition_numbers"],
        trace_information=series["trace_information"],
        worst_std_bounds=series["worst_std_bounds"],
        tau_std_bounds=series["tau_std_bounds"],
        tau_std_bounds_lidar_frames=series["tau_std_bounds_lidar_frames"],
        tau_target_ratio=series["tau_target_ratio"],
        tau_meets_target=tau_meets_target,
        coordinate_std_bounds=series["coordinate_std_bounds"],
        coordinate_bounded_mask=series["coordinate_bounded_mask"],
        coordinate_null_fraction=series["coordinate_null_fraction"],
        coordinate_observable_fraction=series["coordinate_observable_fraction"],
        retained_mode_std_bounds=series["retained_mode_std_bounds"],
        worst_retained_mode_std_bound=series["worst_retained_mode_std_bound"],
        covariance_kind=series["covariance_kind"],
        C_X_L_rank=C_X_L_rank,
        C_X_I_accel_rank=series["C_X_I_accel_rank"],
        accelerometer_factor_count=series["accelerometer_factor_count"],
        matrix_gate_passed=matrix_gate_passed,
    )


##################################################
# Figure and animation exports
##################################################

def _animation_frame_indices(number_of_snapshots: int, max_rendered_frames: int | None) -> NDArray[np.int64]:
    '''Choose snapshot indices used for animation rendering.

    Args:
        number_of_snapshots: Total number of available snapshots.
        max_rendered_frames: Optional maximum number of uniformly distributed frames. ``None`` renders every snapshot.

    Returns:
        Sorted snapshot indices. The first and final snapshots are retained when subsampling a sequence with at least two snapshots.

    Raises:
        ValueError: If ``number_of_snapshots`` is negative or ``max_rendered_frames`` is not positive.
    '''
    if number_of_snapshots < 0:
        raise ValueError("number_of_snapshots must be nonnegative")
    if max_rendered_frames is None or number_of_snapshots <= max_rendered_frames:
        return np.arange(number_of_snapshots, dtype=np.int64)
    if max_rendered_frames <= 0:
        raise ValueError("max_rendered_frames must be positive")
    if max_rendered_frames == 1:
        return np.asarray([number_of_snapshots - 1], dtype=np.int64)

    frame_indices = np.linspace(0, number_of_snapshots - 1, int(max_rendered_frames))
    return np.unique(np.rint(frame_indices).astype(np.int64))


def _pad_animation_matrix(matrix: NDArray[np.float64], target_shape: tuple[int, int]) -> NDArray[np.float64]:
    '''Pad one display matrix to a fixed animation shape.

    Args:
        matrix: Two-dimensional display matrix.
        target_shape: Fixed ``(rows, columns)`` shape used by the image artist.

    Returns:
        Zero-padded matrix with ``target_shape``.

    Raises:
        ValueError: If the matrix is not two-dimensional or exceeds the target shape.
    '''
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("animation heatmap matrices must be two-dimensional")

    target_rows, target_columns = target_shape
    if values.shape[0] > target_rows or values.shape[1] > target_columns:
        raise ValueError("animation heatmap matrix exceeds its target shape")

    padded = np.zeros(target_shape, dtype=float)
    padded[: values.shape[0], : values.shape[1]] = values
    return padded


def _fixed_positive_log_limits(value_arrays: list[NDArray[np.float64]]) -> tuple[float, float]:
    '''Compute stable positive limits for a logarithmic animation axis.

    Args:
        value_arrays: Arrays whose finite positive values share one logarithmic axis.

    Returns:
        Positive lower and upper limits with modest visual padding.
    '''
    finite_parts = [values[np.isfinite(values) & (values > 0.0)] for values in value_arrays]
    finite_parts = [values for values in finite_parts if values.size > 0]
    if not finite_parts:
        return 1e-1, 1e1

    all_values = np.concatenate(finite_parts)
    lower = float(np.min(all_values))
    upper = float(np.max(all_values))
    if np.isclose(lower, upper):
        return lower / 1.5, upper * 1.5
    return lower / 1.15, upper * 1.15


def save_quasi_realtime_rover_animation(
    dataset: object,
    snapshots: list[QuasiRealtimeSnapshot],
    output_html: str | Path,
    *,
    display_variables: tuple[str, ...] = ("T_B_I", "b_g", "tau_I", "tau_L"),
    trajectory_samples: int = 600,
    interval_ms: int = 250,
    figsize=(17, 10),
    show_local_accuracy_summary: bool = True,
    output_mp4: str | Path | None = None,
    mp4_fps: float | None = None,
    mp4_dpi: int = 160,
    max_rendered_frames: int | None = None,
    html_dpi: int = 80,
    html_frame_format: str = "jpeg",
    embed_limit_mb: float = 1000.0,
    standalone_html: bool = False,
    standalone_html_max_frames: int = 300,
    save_html=True,
) -> Path:
    '''Save a standalone HTML animation and optionally an MP4 companion.

    Every snapshot is rendered by default. Frame subsampling is applied only when ``max_rendered_frames`` is explicitly set below the number of snapshots.

    Args:
        dataset: Dataset containing a queryable trajectory.
        snapshots: Complete ordered snapshot sequence.
        output_html: Destination standalone HTML file.
        display_variables: Variables shown in dashboard text and condition plots.
        trajectory_samples: Number of samples used to draw the complete path.
        interval_ms: Playback delay between rendered frames in milliseconds. This controls playback speed, not rendering cost per frame.
        figsize: Matplotlib figure size in inches.
        show_local_accuracy_summary: Whether dashboard text includes coordinate-accuracy details.
        output_mp4: Optional MP4 destination. Requesting both HTML and MP4 renders the frame sequence twice.
        mp4_fps: MP4 frame rate. Defaults to ``1000 / interval_ms``.
        mp4_dpi: MP4 rendering resolution.
        max_rendered_frames: Optional maximum number of uniformly selected frames. ``None`` renders every snapshot.
        html_dpi: Resolution used while rasterizing HTML frames.
        html_frame_format: Embedded frame format accepted by Matplotlib, normally ``"jpeg"`` for speed or ``"png"`` for sharper text.
        embed_limit_mb: Maximum total size of embedded HTML frames in megabytes.

    Returns:
        Saved HTML path.

    Raises:
        ValueError: If snapshots are empty or an animation option is invalid.
        RuntimeError: If MP4 export is requested without an FFmpeg writer.
    '''

    ##################################################
    # Validate rendering options
    ##################################################

    if not snapshots:
        raise ValueError("snapshots must be nonempty")
    if trajectory_samples <= 0:
        raise ValueError("trajectory_samples must be positive")
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    if html_dpi <= 0:
        raise ValueError("html_dpi must be positive")
    if mp4_dpi <= 0:
        raise ValueError("mp4_dpi must be positive")
    if embed_limit_mb <= 0.0:
        raise ValueError("embed_limit_mb must be positive")
    if html_frame_format not in {"png", "jpeg", "tiff", "svg"}:
        raise ValueError("html_frame_format must be 'png', 'jpeg', 'tiff', or 'svg'")

    output_path = Path(output_html)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ##################################################
    # Select frames only when an explicit limit is requested
    ##################################################

    frame_indices = _animation_frame_indices(len(snapshots), max_rendered_frames)
    rendered_snapshots = [snapshots[int(snapshot_index)] for snapshot_index in frame_indices]

    trajectory = getattr(dataset, "trajectory")
    sample_times, sampled_positions, _ = trajectory.sample(int(trajectory_samples))
    positions_xy = sampled_positions[:, :2]
    frame_times = np.asarray([snapshot.current_time for snapshot in rendered_snapshots], dtype=float)
    condition_variables = _condition_plot_variables(display_variables)
    condition_series = dashboard_series(rendered_snapshots, display_variables)["condition_numbers"]

    ##################################################
    # Precompute values that do not need to be rebuilt per frame
    ##################################################

    rover_positions = np.vstack([trajectory.position_at(float(current_time))[:2] for current_time in frame_times])
    rover_yaws = np.asarray([trajectory.yaw_at(float(current_time)) for current_time in frame_times], dtype=float)
    traversed_stop_indices = np.searchsorted(sample_times, frame_times + 1e-12, side="right")
    window_start_indices = np.asarray([np.searchsorted(sample_times, snapshot.window_start - 1e-12, side="left") for snapshot in rendered_snapshots], dtype=int)
    rank_texts = [_rank_dashboard_text(snapshot, display_variables, show_local_accuracy_summary=show_local_accuracy_summary) for snapshot in rendered_snapshots]

    condition_values = {}
    for variable_name in condition_variables:
        values = np.asarray(condition_series[variable_name], dtype=float)
        condition_values[variable_name] = np.where(np.isfinite(values) & (values > 0.0), values, np.nan)

    motion_frames = [_motion_matrix_for_frame(snapshot) for snapshot in rendered_snapshots]
    jacobian_shape = (max(snapshot.J_C_display.shape[0] for snapshot in rendered_snapshots), max(snapshot.J_C_display.shape[1] for snapshot in rendered_snapshots))
    motion_shape = (max(frame.matrix.shape[0] for frame in motion_frames), max(frame.matrix.shape[1] for frame in motion_frames))
    jacobian_matrices = [_pad_animation_matrix(snapshot.J_C_display, jacobian_shape) for snapshot in rendered_snapshots]
    motion_matrices = [_pad_animation_matrix(frame.matrix, motion_shape) for frame in motion_frames]
    motion_titles = [frame.title for frame in motion_frames]
    motion_layouts = [frame.layout for frame in motion_frames]

    ##################################################
    # Construct static artists once
    ##################################################

    fig = plt.figure(figsize=figsize, dpi=int(html_dpi))
    grid = fig.add_gridspec(2, 4, width_ratios=[1.35, 1.35, 1.0, 1.0], height_ratios=[1.0, 1.0])
    trajectory_axis = fig.add_subplot(grid[:, :2])
    rank_axis = fig.add_subplot(grid[0, 2])
    condition_axis = fig.add_subplot(grid[0, 3])
    jacobian_axis = fig.add_subplot(grid[1, 2])
    motion_axis = fig.add_subplot(grid[1, 3])

    trajectory_axis.plot(positions_xy[:, 0], positions_xy[:, 1], color="0.78", linewidth=2.0, label="full path")
    traversed_line, = trajectory_axis.plot([], [], color="#1f77b4", linewidth=2.5, label="traversed")
    window_line, = trajectory_axis.plot([], [], color="#d62728", linewidth=4.0, alpha=0.35, label="active window")
    rover_marker, = trajectory_axis.plot([], [], marker="o", color="#111111", markersize=8)
    heading_arrow = trajectory_axis.quiver([0.0], [0.0], [1.0], [0.0], angles="xy", scale_units="xy", scale=1.0)
    trajectory_axis.set_aspect("equal", adjustable="box")
    trajectory_axis.set_title("Quasi-realtime rover trajectory")
    trajectory_axis.set_xlabel("x [m]")
    trajectory_axis.set_ylabel("y [m]")
    trajectory_axis.grid(True, alpha=0.3)
    trajectory_axis.legend(loc="upper right")
    _pad_axis_limits(trajectory_axis, positions_xy)

    rank_axis.axis("off")
    rank_text = rank_axis.text(0.0, 1.0, "", va="top", ha="left", family="monospace", fontsize=9)

    condition_lines = {}
    for variable_name in condition_variables:
        line, = condition_axis.semilogy([], [], marker="o", markersize=3, linewidth=1.4, label=_display_label(variable_name))
        condition_lines[variable_name] = line

    condition_axis.set_title("Per-variable condition")
    condition_axis.set_xlabel("time [s]")
    condition_axis.set_ylabel("effective cond")
    condition_axis.grid(True, which="both", alpha=0.3)
    condition_axis.legend(fontsize=7)
    condition_axis.set_xlim((frame_times[0] - 0.5, frame_times[0] + 0.5) if frame_times.size == 1 else (frame_times[0], frame_times[-1]))
    condition_axis.set_ylim(*_fixed_positive_log_limits(list(condition_values.values())))

    J_C_image = jacobian_axis.imshow(np.zeros(jacobian_shape, dtype=float), aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    jacobian_axis.set_title("J_C")
    jacobian_axis.set_xlabel("calibration column")
    jacobian_axis.set_ylabel("display row")
    jacobian_horizontal_lines, jacobian_vertical_lines = _create_separator_lines(jacobian_axis, 2, 3)

    motion_image = motion_axis.imshow(np.zeros(motion_shape, dtype=float), aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    motion_title = motion_axis.set_title("C_X_L / C_X_I gyro")
    motion_axis.set_xlabel("sensitivity column")
    motion_axis.set_ylabel("display row")
    motion_horizontal_lines, motion_vertical_lines = _create_separator_lines(motion_axis, 0, 3)
    figure_title = fig.suptitle("")

    ##################################################
    # Update only changing artist data
    ##################################################

    def update(frame_index: int) -> tuple[object, ...]:
        '''Update all artists for one rendered animation frame.'''
        snapshot = rendered_snapshots[frame_index]
        current_position = rover_positions[frame_index]
        current_yaw = rover_yaws[frame_index]
        traversed_stop = int(traversed_stop_indices[frame_index])
        window_start = int(window_start_indices[frame_index])

        traversed_line.set_data(positions_xy[:traversed_stop, 0], positions_xy[:traversed_stop, 1])
        window_line.set_data(positions_xy[window_start:traversed_stop, 0], positions_xy[window_start:traversed_stop, 1])
        rover_marker.set_data([current_position[0]], [current_position[1]])
        heading_arrow.set_offsets([current_position])
        heading_arrow.set_UVC([np.cos(current_yaw)], [np.sin(current_yaw)])
        rank_text.set_text(rank_texts[frame_index])

        for variable_name, line in condition_lines.items():
            line.set_data(frame_times[: frame_index + 1], condition_values[variable_name][: frame_index + 1])

        jacobian_matrix = jacobian_matrices[frame_index]
        J_C_image.set_data(jacobian_matrix)
        _update_heatmap_limits(
            J_C_image,
            jacobian_matrix,
            fixed_limit=1.0 if snapshot.J_C_display_block_scales else None,
        )
        _update_separator_lines(jacobian_horizontal_lines, jacobian_vertical_lines, snapshot.J_C_display_layout)
        _update_layout_ticks(jacobian_axis, snapshot.J_C_display_layout, x_fontsize=6, y_fontsize=6)
        jacobian_axis.set_title(_j_c_display_title(snapshot))

        motion_matrix = motion_matrices[frame_index]
        motion_image.set_data(motion_matrix)
        _update_heatmap_limits(motion_image, motion_matrix)
        _update_separator_lines(motion_horizontal_lines, motion_vertical_lines, motion_layouts[frame_index])
        _update_layout_ticks(motion_axis, motion_layouts[frame_index], x_fontsize=6, y_fontsize=6)
        motion_title.set_text(motion_titles[frame_index])
        figure_title.set_text(f"t = {snapshot.current_time:.2f} s, window = [{snapshot.window_start:.2f}, {snapshot.window_end:.2f}] s")

        return (
            traversed_line,
            window_line,
            rover_marker,
            heading_arrow,
            rank_text,
            *condition_lines.values(),
            J_C_image,
            motion_image,
            motion_title,
            figure_title,
            *jacobian_horizontal_lines,
            *jacobian_vertical_lines,
            *motion_horizontal_lines,
            *motion_vertical_lines,
        )

    figure_animation = animation.FuncAnimation(fig, update, frames=len(rendered_snapshots), interval=interval_ms, blit=False, cache_frame_data=False)

        ##################################################
    # Render HTML and optional MP4
    ##################################################

    frames_per_second = (
        float(mp4_fps)
        if mp4_fps is not None
        else 1000.0 / float(interval_ms)
    )
    frames_per_second = max(frames_per_second, 1e-6)

    try:
        ##################################################
        # MP4 export
        ##################################################

        if output_mp4 is not None:
            if not animation.writers.is_available("ffmpeg"):
                raise RuntimeError(
                    "Matplotlib FFmpeg writer is not available. "
                    "Install ffmpeg or leave output_mp4=None."
                )

            mp4_path = Path(output_mp4)
            mp4_path.parent.mkdir(parents=True, exist_ok=True)

            print(
                f"Starting MP4 rendering: "
                f"{len(rendered_snapshots)} frames",
                flush=True,
            )

            writer = animation.FFMpegWriter(
                fps=frames_per_second,
                metadata={"artist": "calib_observability"},
            )
            figure_animation.save(
                str(mp4_path),
                writer=writer,
                dpi=int(mp4_dpi),
            )

            print(f"Saved MP4: {mp4_path}", flush=True)
            gc.collect()

        ##################################################
        # HTML export
        ##################################################

        if save_html:
            if standalone_html:
                # JSHTML stores every encoded frame in memory and then builds one
                # enormous Python string. Refuse unsafe full-sequence rendering
                # rather than allowing the operating system to kill the kernel.
                if len(rendered_snapshots) > standalone_html_max_frames:
                    raise ValueError(
                        "Standalone JSHTML was requested for "
                        f"{len(rendered_snapshots)} frames, but the configured "
                        f"safe limit is {standalone_html_max_frames}. "
                        "Set max_rendered_frames to a smaller value or use "
                        "standalone_html=False."
                    )

                print(
                    f"Starting standalone HTML rendering: "
                    f"{len(rendered_snapshots)} frames",
                    flush=True,
                )

                with mpl.rc_context(
                    {
                        "animation.embed_limit": float(embed_limit_mb),
                        "animation.frame_format": html_frame_format,
                    }
                ):
                    html = figure_animation.to_jshtml(
                        fps=1000.0 / float(interval_ms),
                        default_mode="loop",
                    )

                output_path.write_text(html, encoding="utf-8")

                # Explicitly release the potentially large base64 HTML string
                # before leaving the function.
                del html
                gc.collect()

            else:
                # This writes one HTML file plus a sibling directory named
                # '<html_stem>_frames'. Frames are streamed to disk rather than
                # accumulated as base64 strings in kernel memory.
                print(
                    f"Starting external-frame HTML rendering: "
                    f"{len(rendered_snapshots)} frames",
                    flush=True,
                )

                with mpl.rc_context(
                    {
                        "animation.frame_format": html_frame_format,
                    }
                ):
                    html_writer = animation.HTMLWriter(
                        fps=1000.0 / float(interval_ms),
                        embed_frames=False,
                        default_mode="loop",
                    )

                    figure_animation.save(
                        str(output_path),
                        writer=html_writer,
                        dpi=int(html_dpi),
                    )

                frame_directory = output_path.with_name(
                    output_path.stem + "_frames"
                )

                print(f"Saved HTML: {output_path}", flush=True)
                print(
                    f"Saved external frames: {frame_directory}",
                    flush=True,
                )

    finally:
        # Always release Matplotlib and animation references, including when
        # FFmpeg or HTML rendering raises an exception.
        plt.close(fig)
        del figure_animation
        gc.collect()

    return output_path


def save_weak_calibration_directions(
    snapshot: QuasiRealtimeSnapshot,
    output_path: str | Path,
    *,
    max_vectors: int = 6,
) -> Path:
    '''Save weak right-singular calibration directions.

    Args:
        snapshot (QuasiRealtimeSnapshot): Snapshot containing a Jacobian bundle.
        output_path (str | Path): Destination image path.
        max_vectors (int): Maximum number of weakest directions to display.

    Returns:
        Path: Saved figure path.

    Raises:
        ValueError: If the snapshot has no Jacobian bundle.
    '''
    if snapshot.bundle is None:
        raise ValueError("snapshot must contain a Jacobian bundle")
    bundle = snapshot.bundle
    if sparse.issparse(bundle.J_T) or sparse.issparse(bundle.J_C):
        projection = effective_observability_sparse_lsmr(bundle.J_T, bundle.J_C)  # type: ignore[arg-type]
        O_C = projection.O_C.toarray()
    else:
        O_C = effective_observability_dense(bundle.J_T, bundle.J_C)
    _, _, Vt = np.linalg.svd(O_C, full_matrices=False)
    weakest = Vt[-min(max_vectors, Vt.shape[0]) :]
    labels = list(bundle.metadata.get("calibration_labels", []))
    if len(labels) != O_C.shape[1]:
        labels = [f"c_{column_index}" for column_index in range(O_C.shape[1])]

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    max_abs = float(np.max(np.abs(weakest))) if weakest.size else 1.0
    image = ax.imshow(weakest, aspect="auto", cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(weakest.shape[0]), [f"v-{index}" for index in range(weakest.shape[0], 0, -1)])
    ax.set_title("Weak calibration directions")
    fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def save_factor_sensitivity_figures(
    snapshot: QuasiRealtimeSnapshot,
    output_dir: str | Path,
    *,
    display_variables: tuple[str, ...] = ("T_B_I", "b_g", "tau_I", "tau_L"),
) -> list[Path]:
    '''Save static factor-sensitivity heatmaps.

    Args:
        snapshot (QuasiRealtimeSnapshot): Snapshot providing display matrices.
        output_dir (str | Path): Destination directory.
        display_variables (tuple[str, ...]): Target matrices included when present.

    Returns:
        list[Path]: Saved figure paths in creation order.
    '''

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    matrices: list[tuple[str, NDArray[np.float64], MatrixDisplayLayout, float | None]] = [
        (_j_c_display_title(snapshot), snapshot.J_C_display, snapshot.J_C_display_layout, 1.0 if snapshot.J_C_display_block_scales else None),
        ("C_X_L", snapshot.C_X_L_display, snapshot.C_X_L_display_layout, None),
        ("C_X_I_gyro", snapshot.C_X_I_gyro_display, snapshot.C_X_I_gyro_display_layout, None),
        ("C_X_I_accel", snapshot.C_X_I_accel_display, snapshot.C_X_I_accel_display_layout, None),
    ]
    for variable_name in display_variables:
        if variable_name in snapshot.target_O_display:
            matrices.append((_display_label(variable_name), snapshot.target_O_display[variable_name], MatrixDisplayLayout(), None))

    saved_paths: list[Path] = []
    for title, matrix, layout, fixed_limit in matrices:
        output_path = output_directory / f"{_filename_token(title)}_heatmap.png"
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        max_abs = fixed_limit if fixed_limit is not None else _finite_symmetric_limit(matrix)
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
        ax.set_title(title)
        ax.set_xlabel("column")
        ax.set_ylabel("display row")
        horizontal_lines, vertical_lines = _create_separator_lines(
            ax,
            len(layout.row_boundaries),
            len(layout.column_boundaries),
        )
        _update_separator_lines(horizontal_lines, vertical_lines, layout)
        _update_layout_ticks(ax, layout, x_fontsize=7, y_fontsize=7, clear_missing=False)
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        saved_paths.append(output_path)
    return saved_paths


def latest_valid_snapshot(snapshots: list[QuasiRealtimeSnapshot]) -> QuasiRealtimeSnapshot:
    '''Return the last snapshot containing target diagnostics.

    Args:
        snapshots (list[QuasiRealtimeSnapshot]): Snapshot sequence.

    Returns:
        QuasiRealtimeSnapshot: Latest snapshot with `is_valid` set.

    Raises:
        ValueError: If the sequence contains no valid snapshot.
    '''

    for snapshot in reversed(snapshots):
        if snapshot.is_valid:
            return snapshot
    raise ValueError("no valid snapshots were produced")



##################################################
# Internal display and formatting helpers
##################################################

def _display_indices(count: int, maximum_count: int) -> NDArray[np.int64]:
    '''Choose approximately uniform indices for a display axis.

    Args:
        count (int): Number of source entries.
        maximum_count (int): Maximum entries to retain.

    Returns:
        NDArray[np.int64]: Sorted unique source indices.

    Raises:
        ValueError: If `maximum_count` is not positive.
    '''
    if maximum_count <= 0:
        raise ValueError("maximum display size must be positive")
    if count <= maximum_count:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, maximum_count).round().astype(np.int64))


def _nan_by_variable(display_variables: tuple[str, ...]) -> dict[str, float]:
    '''Create a variable-to-NaN diagnostic mapping.

    Args:
        display_variables (tuple[str, ...]): Variable names to initialize.

    Returns:
        dict[str, float]: Mapping with `np.nan` for every variable.
    '''
    return {variable_name: np.nan for variable_name in display_variables}


def _empty_snapshot(
    current_time: float,
    window_start: float,
    window_end: float,
    status: str,
    config: QuasiRealtimeConfig,
    machine_ranks: dict[str, float],
    effective_ranks: dict[str, float],
    condition_numbers: dict[str, float],
    *,
    counts: dict[str, int] | None = None,
    bundle: JacobianBundle | None = None,
) -> QuasiRealtimeSnapshot:
    '''Build a structured invalid snapshot.

    Args:
        current_time (float): Current frame time.
        window_start (float): Active-window start.
        window_end (float): Active-window end.
        status (str): Human-readable waiting or failure state.
        config (QuasiRealtimeConfig): Display configuration.
        machine_ranks (dict[str, float]): Initial machine-rank mapping.
        effective_ranks (dict[str, float]): Initial practical-rank mapping.
        condition_numbers (dict[str, float]): Initial condition mapping.
        counts (dict[str, int] | None): Optional factor counts.
        bundle (JacobianBundle | None): Optional partially assembled bundle.

    Returns:
        QuasiRealtimeSnapshot: Invalid snapshot with placeholder display matrices.
    '''
    return QuasiRealtimeSnapshot(
        current_time=float(current_time),
        window_start=float(window_start),
        window_end=float(window_end),
        is_valid=False,
        status=status,
        counts={} if counts is None else dict(counts),
        machine_ranks=dict(machine_ranks),
        effective_ranks=dict(effective_ranks),
        condition_numbers=dict(condition_numbers),
        J_C_display=matrix_for_display(None, max_rows=config.max_display_rows, max_cols=config.max_display_cols),
        C_X_L_display=matrix_for_display(None, max_rows=config.max_display_rows, max_cols=config.max_display_cols),
        C_X_I_gyro_display=matrix_for_display(None, max_rows=config.max_display_rows, max_cols=config.max_display_cols),
        bundle=bundle,
    )


def _display_label(variable_name: str) -> str:
    '''Return the dashboard label for a calibration variable.

    Args:
        variable_name (str): Calibration variable name.

    Returns:
        str: Registered label or an `O_`-prefixed fallback.
    '''
    return VARIABLE_DISPLAY_LABELS.get(variable_name, f"O_{variable_name}")


def _rank_dashboard_text(
    snapshot: QuasiRealtimeSnapshot,
    display_variables: tuple[str, ...],
    *,
    show_local_accuracy_summary: bool = True,
) -> str:
    '''Format the snapshot text dashboard.

    Args:
        snapshot (QuasiRealtimeSnapshot): Snapshot to summarize.
        display_variables (tuple[str, ...]): Variables included in the summary.
        show_local_accuracy_summary (bool): Whether to append coordinate bounds.

    Returns:
        str: Multiline dashboard text.
    '''
    accel_rank = np.nan
    if snapshot.motion_accelerometer is not None:
        accel_rank = float(snapshot.motion_accelerometer.practical_rank)
    accel_rank_text = "nan" if np.isnan(accel_rank) else f"{accel_rank:.0f}"
    accel_max_text = "6" if snapshot.motion_accelerometer is not None else "nan"
    lines = [
        f"status: {snapshot.status}",
        f"time: {snapshot.current_time:.2f} s",
        f"window: {snapshot.window_start:.2f} - {snapshot.window_end:.2f} s",
        f"factors: imu={snapshot.counts.get('imu', 0)}, lidar={snapshot.counts.get('lidar', 0)}, accel={snapshot.accelerometer_factor_count}",
        f"accel mode: {snapshot.accelerometer_mode}",
        f"C_X_I accel: {accel_rank_text} / {accel_max_text}",
        "",
        "ranks (practical/max):",
    ]
    for variable_name in display_variables:
        effective_rank = snapshot.effective_ranks.get(variable_name, np.nan)
        condition_number = snapshot.condition_numbers.get(variable_name, np.nan)
        maximum_rank = _maximum_rank_for_variable(snapshot, variable_name)
        rank_text = "nan" if np.isnan(effective_rank) else f"{effective_rank:.0f}"
        maximum_text = "nan" if maximum_rank is None else f"{maximum_rank:d}"
        cond_text = "nan" if not np.isfinite(condition_number) else f"{condition_number:.2e}"
        if variable_name in {"tau_I", "tau_L"}:
            tau_std = snapshot.tau_std_bounds.get(variable_name, np.nan)
            tau_text = "inf" if np.isinf(tau_std) else ("nan" if np.isnan(tau_std) else f"{tau_std:.2e}s")
            lines.append(f"{_display_label(variable_name)}: tau std={tau_text}")
        else:
            lines.append(f"{_display_label(variable_name)}: {rank_text} / {maximum_text}, cond={cond_text}")
    if show_local_accuracy_summary:
        lines.extend(_local_accuracy_dashboard_lines(snapshot, display_variables))
    return "\n".join(lines)


def _coordinate_accuracy_array(
    snapshots: list[QuasiRealtimeSnapshot],
    variable_name: str,
    field_name: str,
    *,
    fill_value: float | bool = np.nan,
    dtype: object = float,
) -> NDArray:
    '''Stack one coordinate-accuracy field across snapshots.

    Args:
        snapshots (list[QuasiRealtimeSnapshot]): Ordered snapshots.
        variable_name (str): Target calibration variable.
        field_name (str): Attribute read from local accuracy diagnostics.
        fill_value (float | bool): Value used when diagnostics are missing.
        dtype (object): NumPy output dtype.

    Returns:
        NDArray: Array with shape `(num_snapshots, variable_dimension)`.
    '''
    dimension = _accuracy_dimension(snapshots, variable_name)
    rows = np.full((len(snapshots), dimension), fill_value, dtype=dtype)
    for snapshot_index, snapshot in enumerate(snapshots):
        accuracy = snapshot.local_accuracy_by_variable.get(variable_name)
        if accuracy is None:
            continue
        values = np.asarray(getattr(accuracy, field_name), dtype=dtype).reshape(-1)
        rows[snapshot_index, : min(dimension, values.size)] = values[:dimension]
    return rows


def _accuracy_dimension(snapshots: list[QuasiRealtimeSnapshot], variable_name: str) -> int:
    '''Infer the coordinate dimension for one variable.

    Args:
        snapshots (list[QuasiRealtimeSnapshot]): Snapshots to inspect.
        variable_name (str): Calibration variable name.

    Returns:
        int: Number of coordinate labels for the variable.
    '''
    for snapshot in snapshots:
        accuracy = snapshot.local_accuracy_by_variable.get(variable_name)
        if accuracy is not None:
            return len(accuracy.coordinate_labels)
    labels, _ = coordinate_metadata_for_variable(variable_name, VARIABLE_MAX_RANKS.get(variable_name, 1))
    return len(labels)


def _local_accuracy_dashboard_lines(snapshot: QuasiRealtimeSnapshot, display_variables: tuple[str, ...]) -> list[str]:
    '''Format local CRLB-like dashboard lines.

    Args:
        snapshot (QuasiRealtimeSnapshot): Snapshot containing local diagnostics.
        display_variables (tuple[str, ...]): Variables included in the summary.

    Returns:
        list[str]: Formatted lines appended to the rank dashboard.
    '''
    lines = ["", "Local CRLB-like bounds:"]
    for variable_name in display_variables:
        accuracy = snapshot.local_accuracy_by_variable.get(variable_name)
        if accuracy is None:
            continue
        lines.append(f"  {_display_label(variable_name)} rank={accuracy.practical_rank}/{accuracy.maximum_rank}, nullity={accuracy.nullity}, {accuracy.covariance_kind}")
        if variable_name in {"tau_I", "tau_L"}:
            seconds = _format_bound(accuracy.scalar_std_bound, "s")
            frames = _format_bound(accuracy.scalar_std_bound_lidar_frames, "frames")
            ratio = "nan" if accuracy.target_ratio is None or not np.isfinite(accuracy.target_ratio) else f"{accuracy.target_ratio:.2f}x"
            status = "pass" if accuracy.meets_target else "fail"
            lines.append(f"    {seconds} = {frames}, target {ratio}, {status}")
            continue
        for label, unit, bound, bounded in zip(accuracy.coordinate_labels, accuracy.coordinate_units, accuracy.coordinate_std_bounds, accuracy.coordinate_is_fully_bounded):
            text = _format_bound(float(bound), unit) if bool(bounded) else "unbounded"
            if unit == "rad" and bool(bounded) and np.isfinite(bound):
                text += f" ({np.degrees(float(bound)):.2e} deg)"
            lines.append(f"    {label}: {text}")
        lines.append(f"    worst mode: {_format_bound(accuracy.worst_retained_mode_std_bound, 'native')}")
        if accuracy.retained_mode_std_bounds.size:
            for mode_index in range(min(2, accuracy.retained_mode_std_bounds.size)):
                coeff = _dominant_coefficients(accuracy, mode_index)
                lines.append(f"    mode {mode_index + 1}: {accuracy.retained_mode_kinds[mode_index]}, {coeff}")
    return lines


def _dominant_coefficients(accuracy: Any, mode_index: int, maximum_terms: int = 3) -> str:
    '''Format dominant coefficients of one retained mode.

    Args:
        accuracy (Any): Local accuracy diagnostics object.
        mode_index (int): Retained mode column index.
        maximum_terms (int): Maximum coefficients to report.

    Returns:
        str: Comma-separated signed coordinate coefficients.
    '''
    direction = np.asarray(accuracy.retained_mode_directions)[:, mode_index]
    order = np.argsort(np.abs(direction))[::-1][:maximum_terms]
    return ", ".join(f"{accuracy.coordinate_labels[index]}={direction[index]:+.2f}" for index in order)


def _format_bound(value: float | None, unit: str) -> str:
    '''Format a scalar bound with its unit.

    Args:
        value (float | None): Bound to format.
        unit (str): Unit label or `native`.

    Returns:
        str: Compact finite, infinite, or missing-value representation.
    '''
    if value is None or np.isnan(float(value)):
        return "nan"
    if np.isinf(float(value)):
        return "inf" if unit == "native" else f"inf {unit}"
    return f"{float(value):.2e}" if unit == "native" else f"{float(value):.2e} {unit}"


def _maximum_rank_for_variable(snapshot: QuasiRealtimeSnapshot, variable_name: str) -> int | None:
    '''Return the displayed maximum rank for one variable.

    Args:
        snapshot (QuasiRealtimeSnapshot): Snapshot providing active target labels.
        variable_name (str): Calibration variable name.

    Returns:
        int | None: Active or registered maximum rank, if known.
    '''
    result = snapshot.target_results.get(variable_name)
    if result is not None:
        return len(result.target_labels)
    if variable_name in VARIABLE_MAX_RANKS:
        return VARIABLE_MAX_RANKS[variable_name]
    return None


def _condition_plot_variables(display_variables: tuple[str, ...]) -> tuple[str, ...]:
    '''Select multidimensional variables for condition plots.

    Args:
        display_variables (tuple[str, ...]): Candidate calibration variables.

    Returns:
        tuple[str, ...]: Variables excluding scalar timing offsets.

    Notes:
        A nonzero scalar column has condition number one, so timing variables are
        better represented by local standard-deviation bounds.
    '''

    return tuple(variable_name for variable_name in display_variables if variable_name not in {"tau_I", "tau_L"})



def _semantic_j_c_display(
    matrix: ArrayLike | sparse.spmatrix | None,
    calibration_column_slices: dict[str, slice],
    factor_family_row_slices: dict[str, tuple[slice, ...]],
    *,
    max_rows: int,
    max_cols: int,
    normalize_factor_blocks: bool,
) -> MatrixDisplayResult:
    '''Build a semantic display copy of ``J_C`` without mutating the physical matrix.

    Rows are split by true factor-family row metadata, downsampled independently
    inside LiDAR, gyro, and accelerometer blocks, then concatenated in semantic
    display order. Columns are selected from ``calibration_column_slices`` in the
    canonical ``T_B_I | b_g | tau_I | tau_L`` order. Optional normalization is
    applied independently to each displayed sensor-family row block only.
    '''
    if matrix is None:
        return MatrixDisplayResult(np.zeros((1, 1), dtype=float))
    if max_rows <= 0 or max_cols <= 0:
        raise ValueError("maximum display sizes must be positive")

    source_shape = matrix.shape if sparse.issparse(matrix) else np.asarray(matrix).shape
    if len(source_shape) != 2:
        raise ValueError("matrix must be two-dimensional")
    if source_shape[0] == 0 or source_shape[1] == 0:
        return MatrixDisplayResult(np.zeros((1, 1), dtype=float))

    column_blocks = _semantic_column_blocks(calibration_column_slices)
    if not column_blocks:
        return MatrixDisplayResult(np.zeros((1, 1), dtype=float))
    column_counts = [indices.size for _, _, indices in column_blocks]
    displayed_column_counts = _allocate_block_display_counts(column_counts, max_cols)
    column_indices_parts = []
    column_lengths = []
    column_labels = []
    for (_variable_name, label, source_indices), display_count in zip(column_blocks, displayed_column_counts):
        if display_count <= 0:
            continue
        local_indices = _display_indices(source_indices.size, display_count)
        column_indices_parts.append(source_indices[local_indices])
        column_lengths.append(int(display_count))
        column_labels.append(label)
    column_indices = np.concatenate(column_indices_parts).astype(np.int64)

    row_blocks = _semantic_row_blocks(factor_family_row_slices)
    if not row_blocks:
        display = _matrix_take(matrix, np.arange(source_shape[0], dtype=np.int64), column_indices)
        row_indices = _display_indices(display.shape[0], max_rows)
        display = np.asarray(display[row_indices, :], dtype=float)
        return MatrixDisplayResult(np.nan_to_num(display, nan=0.0, posinf=0.0, neginf=0.0), _column_only_layout(column_lengths, column_labels))

    row_counts = [indices.size for _, _, indices in row_blocks]
    displayed_row_counts = _allocate_block_display_counts(row_counts, max_rows)
    matrix_parts = []
    row_lengths = []
    row_labels = []
    block_scales: dict[str, float] = {}
    for (family_name, label, source_indices), display_count in zip(row_blocks, displayed_row_counts):
        if display_count <= 0:
            continue
        local_indices = _display_indices(source_indices.size, display_count)
        row_indices = source_indices[local_indices]
        block = _matrix_take(matrix, row_indices, column_indices)
        if normalize_factor_blocks:
            scale = _finite_nonzero_max_abs(block)
            block_scales[family_name] = scale
            block = block / scale
        matrix_parts.append(block)
        row_lengths.append(int(block.shape[0]))
        row_labels.append(label)

    display_matrix = np.vstack(matrix_parts) if matrix_parts else np.zeros((1, len(column_indices)), dtype=float)
    layout = _matrix_layout(row_lengths, row_labels, column_lengths, column_labels)
    if not normalize_factor_blocks:
        display_matrix = np.nan_to_num(display_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    return MatrixDisplayResult(np.asarray(display_matrix, dtype=float), layout, block_scales if normalize_factor_blocks else {})


def _semantic_column_blocks(calibration_column_slices: dict[str, slice]) -> list[tuple[str, str, NDArray[np.int64]]]:
    blocks = []
    for variable_name in J_C_DISPLAY_COLUMN_ORDER:
        column_slice = calibration_column_slices.get(variable_name)
        if column_slice is None:
            continue
        start = 0 if column_slice.start is None else int(column_slice.start)
        stop = start if column_slice.stop is None else int(column_slice.stop)
        if stop > start:
            blocks.append((variable_name, variable_name, np.arange(start, stop, dtype=np.int64)))
    return blocks


def _factor_family_row_slices(bundle: JacobianBundle) -> dict[str, tuple[slice, ...]]:
    '''Return true source-row slices for LiDAR, gyro, and accelerometer factors.

    The function first honors explicit bundle metadata when present. Otherwise it
    derives families from residual block names retained in ``bundle.row_slices``;
    the simulation assembly names measurement factors ``lidar_*``, ``imu_*``,
    and ``accel_*`` and priors are intentionally ignored.
    '''
    metadata_slices = bundle.metadata.get("factor_family_row_slices") if bundle.metadata else None
    if isinstance(metadata_slices, dict):
        return {str(family): _coerce_row_slices(value) for family, value in metadata_slices.items()}

    families: dict[str, list[slice]] = {"lidar": [], "gyro": [], "accelerometer": []}
    for residual_name, row_slice in bundle.row_slices.items():
        family = _family_from_residual_name(residual_name)
        if family is not None:
            families[family].append(row_slice)
    return {family: tuple(slices) for family, slices in families.items() if slices}


def _coerce_row_slices(value: object) -> tuple[slice, ...]:
    if isinstance(value, slice):
        return (value,)
    coerced = []
    for item in value if isinstance(value, (list, tuple)) else ():
        if isinstance(item, slice):
            coerced.append(item)
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            coerced.append(slice(int(item[0]), int(item[1])))
    return tuple(coerced)


def _family_from_residual_name(residual_name: str) -> str | None:
    if residual_name.startswith("lidar_"):
        return "lidar"
    if residual_name.startswith("imu_") or residual_name.startswith("gyro_"):
        return "gyro"
    if residual_name.startswith("accel_") or residual_name.startswith("accelerometer_"):
        return "accelerometer"
    return None


def _semantic_row_blocks(factor_family_row_slices: dict[str, tuple[slice, ...]]) -> list[tuple[str, str, NDArray[np.int64]]]:
    blocks = []
    for family_name, label in J_C_FACTOR_FAMILY_ORDER:
        row_indices = _row_indices_from_slices(factor_family_row_slices.get(family_name, ()))
        if row_indices.size:
            blocks.append((family_name, label, row_indices))
    return blocks


def _row_indices_from_slices(row_slices: tuple[slice, ...]) -> NDArray[np.int64]:
    parts = []
    for row_slice in row_slices:
        start = 0 if row_slice.start is None else int(row_slice.start)
        stop = start if row_slice.stop is None else int(row_slice.stop)
        if stop > start:
            parts.append(np.arange(start, stop, dtype=np.int64))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)


def _allocate_block_display_counts(source_counts: list[int], max_count: int) -> list[int]:
    if max_count <= 0:
        raise ValueError("maximum display size must be positive")
    positive_indices = [index for index, count in enumerate(source_counts) if count > 0]
    if not positive_indices:
        return [0 for _ in source_counts]
    if sum(source_counts) <= max_count:
        return list(source_counts)
    if max_count < len(positive_indices):
        raise ValueError("max_display limit is smaller than the number of nonempty semantic blocks")

    allocation = [0 for _ in source_counts]
    for index in positive_indices:
        allocation[index] = 1
    remaining = max_count - len(positive_indices)
    capacities = [max(source_counts[index] - allocation[index], 0) for index in range(len(source_counts))]
    total_positive = float(sum(source_counts[index] for index in positive_indices))
    raw_extras = [remaining * source_counts[index] / total_positive if index in positive_indices else 0.0 for index in range(len(source_counts))]
    for index in positive_indices:
        extra = min(int(np.floor(raw_extras[index])), capacities[index], remaining)
        allocation[index] += extra
        remaining -= extra
    while remaining > 0:
        candidates = [index for index in positive_indices if allocation[index] < source_counts[index]]
        if not candidates:
            break
        candidates.sort(key=lambda index: (raw_extras[index] - np.floor(raw_extras[index]), source_counts[index]), reverse=True)
        allocation[candidates[0]] += 1
        remaining -= 1
    return allocation


def _matrix_take(matrix: ArrayLike | sparse.spmatrix, row_indices: NDArray[np.int64], column_indices: NDArray[np.int64]) -> NDArray[np.float64]:
    if sparse.issparse(matrix):
        return matrix[row_indices, :][:, column_indices].toarray().astype(float)
    dense_matrix = np.asarray(matrix, dtype=float)
    return dense_matrix[np.ix_(row_indices, column_indices)]


def _matrix_layout(
    row_lengths: list[int],
    row_labels: list[str],
    column_lengths: list[int],
    column_labels: list[str],
) -> MatrixDisplayLayout:
    row_boundaries, row_centers = _boundaries_and_centers(row_lengths)
    column_boundaries, column_centers = _boundaries_and_centers(column_lengths)
    return MatrixDisplayLayout(
        row_boundaries=row_boundaries,
        row_centers=row_centers,
        row_labels=tuple(row_labels),
        column_boundaries=column_boundaries,
        column_centers=column_centers,
        column_labels=tuple(column_labels),
    )


def _column_only_layout(column_lengths: list[int], column_labels: list[str]) -> MatrixDisplayLayout:
    column_boundaries, column_centers = _boundaries_and_centers(column_lengths)
    return MatrixDisplayLayout(
        column_boundaries=column_boundaries,
        column_centers=column_centers,
        column_labels=tuple(column_labels),
    )


def _boundaries_and_centers(lengths: list[int]) -> tuple[tuple[int, ...], tuple[float, ...]]:
    boundaries = []
    centers = []
    start = 0
    nonzero_lengths = [int(length) for length in lengths if int(length) > 0]
    for index, length in enumerate(nonzero_lengths):
        centers.append(start + 0.5 * (length - 1))
        start += length
        if index < len(nonzero_lengths) - 1:
            boundaries.append(start)
    return tuple(boundaries), tuple(centers)


def _finite_nonzero_max_abs(matrix: NDArray[np.float64]) -> float:
    values = np.asarray(matrix, dtype=float)
    finite_values = values[np.isfinite(values) & (values != 0.0)]
    if finite_values.size == 0:
        return 1.0
    return float(np.max(np.abs(finite_values)))


def _c_x_display_result(
    matrix: ArrayLike | sparse.spmatrix | None,
    matrix_kind: str,
    *,
    max_rows: int,
    max_cols: int,
) -> MatrixDisplayResult:
    '''Build a display matrix and semantic column layout for one C_X panel.

    C_X columns describe target coordinates, so they use rotation/translation
    layouts rather than the full calibration-vector layout used by ``J_C``.
    '''
    display_matrix = matrix_for_display(matrix, max_rows=max_rows, max_cols=max_cols)
    layout = _c_x_layout(display_matrix, matrix_kind)
    return MatrixDisplayResult(display_matrix, layout)


def _c_x_layout(matrix: NDArray[np.float64], matrix_kind: str) -> MatrixDisplayLayout:
    values = np.asarray(matrix)
    column_count = int(values.shape[1]) if values.ndim == 2 else 0
    if matrix_kind in {"lidar", "accelerometer"} and column_count == 6:
        return _column_only_layout([3, 3], ["rotation", "translation"])
    if matrix_kind == "gyro" and column_count == 3:
        return _column_only_layout([3], ["rotation"])
    return MatrixDisplayLayout()


def _finite_symmetric_limit(matrix: NDArray[np.float64]) -> float:
    values = np.asarray(matrix, dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 1e-12
    return max(float(np.max(np.abs(finite_values))), 1e-12)


def _update_heatmap_limits(image: object, matrix: NDArray[np.float64], *, fixed_limit: float | None = None) -> None:
    max_abs = float(fixed_limit) if fixed_limit is not None else _finite_symmetric_limit(matrix)
    image.set_clim(-max_abs, max_abs)


def _create_separator_lines(axis: object, horizontal_count: int, vertical_count: int) -> tuple[tuple[object, ...], tuple[object, ...]]:
    horizontal_lines = tuple(axis.axhline(0.0, color="black", linewidth=0.6, alpha=0.9, zorder=5, visible=False) for _ in range(horizontal_count))
    vertical_lines = tuple(axis.axvline(0.0, color="black", linewidth=0.6, alpha=0.9, zorder=5, visible=False) for _ in range(vertical_count))
    return horizontal_lines, vertical_lines


def _update_separator_lines(
    horizontal_lines: tuple[object, ...],
    vertical_lines: tuple[object, ...],
    layout: MatrixDisplayLayout,
) -> None:
    '''Move reusable separator artists to displayed block boundaries.'''
    for line, boundary in zip(horizontal_lines, layout.row_boundaries):
        y = float(boundary) - 0.5
        line.set_ydata([y, y])
        line.set_visible(True)
    for line in horizontal_lines[len(layout.row_boundaries) :]:
        line.set_visible(False)
    for line, boundary in zip(vertical_lines, layout.column_boundaries):
        x = float(boundary) - 0.5
        line.set_xdata([x, x])
        line.set_visible(True)
    for line in vertical_lines[len(layout.column_boundaries) :]:
        line.set_visible(False)


def _update_layout_ticks(
    axis: object,
    layout: MatrixDisplayLayout,
    *,
    x_fontsize: int,
    y_fontsize: int,
    clear_missing: bool = True,
) -> None:
    if layout.column_labels:
        axis.set_xticks(layout.column_centers)
        axis.set_xticklabels(layout.column_labels, rotation=0, fontsize=x_fontsize)
    elif clear_missing:
        axis.set_xticks([])
    if layout.row_labels:
        axis.set_yticks(layout.row_centers)
        axis.set_yticklabels(layout.row_labels, fontsize=y_fontsize)
    else:
        axis.set_yticks([])


def _j_c_display_title(snapshot: QuasiRealtimeSnapshot) -> str:
    return "J_C, factor-family normalized" if snapshot.J_C_display_block_scales else "J_C"

def _motion_matrix_for_frame(snapshot: QuasiRealtimeSnapshot) -> MotionMatrixFrame:
    '''Choose the factor-sensitivity matrix shown in one frame.

    Args:
        snapshot (QuasiRealtimeSnapshot): Current animation snapshot.

    Returns:
        MotionMatrixFrame: Display matrix, title, and semantic column layout.
    '''
    if snapshot.motion_accelerometer is not None:
        return MotionMatrixFrame(snapshot.C_X_I_accel_display, "C_X_I accel", snapshot.C_X_I_accel_display_layout)
    if snapshot.motion_lidar is not None:
        return MotionMatrixFrame(snapshot.C_X_L_display, "C_X_L", snapshot.C_X_L_display_layout)
    if snapshot.motion_imu is not None:
        return MotionMatrixFrame(snapshot.C_X_I_gyro_display, "C_X_I gyro", snapshot.C_X_I_gyro_display_layout)
    return MotionMatrixFrame(np.zeros((1, 1), dtype=float), "C_X waiting", MatrixDisplayLayout())


def _update_heatmap(image: object, axis: object, matrix: NDArray[np.float64], title: str) -> None:
    '''Update an image artist for a possibly new matrix shape.

    Args:
        image (object): Matplotlib image artist.
        axis (object): Matplotlib axis containing the image.
        matrix (NDArray[np.float64]): New display matrix.
        title (str): Axis title.

    Returns:
        None
    '''
    image.set_data(matrix)
    _update_heatmap_limits(image, matrix)
    # `imshow` keeps the extent from the matrix used during initialization.
    # Animation frames can change matrix shape, so refresh the extent together
    # with the data; otherwise the image can collapse into a tiny corner.
    image.set_extent((-0.5, matrix.shape[1] - 0.5, matrix.shape[0] - 0.5, -0.5))
    axis.set_title(title)
    axis.set_xlim(-0.5, matrix.shape[1] - 0.5)
    axis.set_ylim(matrix.shape[0] - 0.5, -0.5)


def _pad_axis_limits(axis: object, positions_xy: NDArray[np.float64]) -> None:
    '''Pad trajectory-axis limits around sampled positions.

    Args:
        axis (object): Matplotlib trajectory axis.
        positions_xy (NDArray[np.float64]): Sampled planar positions, shape `(N, 2)`.

    Returns:
        None
    '''
    minimum = np.min(positions_xy, axis=0)
    maximum = np.max(positions_xy, axis=0)
    span = np.maximum(maximum - minimum, 1.0)
    padding = 0.08 * span
    axis.set_xlim(minimum[0] - padding[0], maximum[0] + padding[0])
    axis.set_ylim(minimum[1] - padding[1], maximum[1] + padding[1])


def _filename_token(value: str) -> str:
    '''Convert a display title into a filesystem-safe token.

    Args:
        value (str): Display title.

    Returns:
        str: Token with spaces and path separators replaced.
    '''
    return value.replace(" ", "_").replace("/", "_").replace("{", "").replace("}", "")