'''Calibration-law and IMU measurement emulation helpers.

The emulation routines in this module create artificial time-varying IMU
calibration experiments from an already measured IMU stream. Ground-truth
trajectory poses are intentionally not used to synthesize gyroscope or
accelerometer values. Instead, real samples are re-expressed through configured
calibration laws and their timestamps are warped according to the temporal
offset convention used by ``FactorGyroCalibProp``.

The default model is quasi-static: ``T_B_I(t)`` is interpreted as a slowly
varying calibration parameter. Angular velocity, acceleration, ``r_dot`` and
``r_ddot`` generated only by physical motion of the mounting transform itself
are intentionally omitted because the current factor graph estimates a constant
calibration inside each rolling window.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .conventions import as_matrix, as_vector
from .lie_se3 import se3_exp, se3_inverse, se3_log


LawMode = Literal["constant", "linear", "sinusoidal", "piecewise", "piecewise_constant", "piecewise-constant", "callable"]


def _as_timestamps(timestamps: ArrayLike, name: str = "timestamps") -> NDArray[np.float64]:
    '''Validate a one-dimensional timestamp array.

    Args:
        timestamps: Timestamp array with shape ``(N,)``.
        name: Argument name used in validation errors.

    Returns:
        Validated floating-point timestamps.

    Raises:
        ValueError: If the timestamps are not one-dimensional and finite.
    '''

    values = np.asarray(timestamps, dtype=float).reshape(-1)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    return values.copy()


def _require_strictly_increasing(timestamps: NDArray[np.float64], name: str) -> None:
    '''Require strictly increasing timestamps.

    Args:
        timestamps: Timestamp array with shape ``(N,)``.
        name: Argument name used in validation errors.

    Raises:
        ValueError: If fewer than two timestamps are supplied or if any adjacent
            timestamp difference is non-positive.
    '''

    if timestamps.size < 2:
        raise ValueError(f"{name} must contain at least two timestamps")
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing")


def _as_vector_stream(values: ArrayLike, timestamps: NDArray[np.float64], name: str) -> NDArray[np.float64]:
    '''Validate a timestamped 3D vector stream.

    Args:
        values: Vector samples with shape ``(N, 3)``.
        timestamps: Matching timestamps with shape ``(N,)``.
        name: Argument name used in validation errors.

    Returns:
        Validated copy of ``values``.

    Raises:
        ValueError: If shape, length or finite-value validation fails.
    '''

    array = np.asarray(values, dtype=float)
    if array.shape != (timestamps.size, 3):
        raise ValueError(f"{name} must have shape ({timestamps.size}, 3), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()


def _normalized_alpha(timestamps: NDArray[np.float64], t_start: float, t_end: float) -> NDArray[np.float64]:
    '''Map timestamps to a clipped ``[0, 1]`` interpolation parameter.

    Args:
        timestamps: Query timestamps with shape ``(N,)``.
        t_start: Time where ``alpha == 0``.
        t_end: Time where ``alpha == 1``.

    Returns:
        Clipped interpolation parameters with shape ``(N,)``.

    Raises:
        ValueError: If ``t_end`` is not greater than ``t_start``.
    '''

    t_start = float(t_start)
    t_end = float(t_end)
    if not np.isfinite(t_start) or not np.isfinite(t_end) or t_end <= t_start:
        raise ValueError("law time interval must satisfy finite t_end > t_start")
    return np.clip((timestamps - t_start) / (t_end - t_start), 0.0, 1.0)


def _as_se3_stack(values: ArrayLike, name: str) -> NDArray[np.float64]:
    '''Validate a stack of SE(3) matrices.

    Args:
        values: Array with shape ``(N, 4, 4)``.
        name: Argument name used in validation errors.

    Returns:
        Validated transform stack.

    Raises:
        ValueError: If the array has a wrong shape or contains non-finite
            entries.
    '''

    array = np.asarray(values, dtype=float)
    if array.ndim != 3 or array.shape[1:] != (4, 4):
        raise ValueError(f"{name} must have shape (N, 4, 4)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array.copy()



def _piecewise_constant_indices(control_times: NDArray[np.float64], timestamps: NDArray[np.float64]) -> NDArray[np.int64]:
    '''Return zero-order-hold control indices for query timestamps.

    Args:
        control_times: Strictly increasing control timestamps with shape ``(K,)``.
        timestamps: Query timestamps with shape ``(N,)``.

    Returns:
        Control indices with shape ``(N,)``. Queries before the first control use
        index 0, queries between controls use the previous control, and queries
        after the final control use the final index.
    '''

    # Find the control point immediately to the left of each query. ``side='right'``
    # makes exact control-time queries switch to the new control value.
    indices = np.searchsorted(control_times, timestamps, side="right") - 1

    # Clamp outside the control support instead of extrapolating or rejecting;
    # this gives clear zero-order-hold endpoint behavior.
    return np.clip(indices, 0, control_times.size - 1).astype(np.int64)

def interpolate_se3_lie(T_start: ArrayLike, T_end: ArrayLike, alpha: ArrayLike) -> NDArray[np.float64]:
    '''Interpolate between two SE(3) transforms on the Lie group.

    Args:
        T_start: Start transform with shape ``(4, 4)``.
        T_end: End transform with shape ``(4, 4)``.
        alpha: Interpolation parameter(s). Values are clipped to ``[0, 1]``.

    Returns:
        Interpolated transform stack with shape ``(N, 4, 4)``.
    '''

    # Validate the two endpoint transforms and the interpolation parameters.
    start = as_matrix(T_start, (4, 4), "T_start")
    end = as_matrix(T_end, (4, 4), "T_end")
    a = np.asarray(alpha, dtype=float).reshape(-1)
    if not np.all(np.isfinite(a)):
        raise ValueError("alpha must contain only finite values")

    # Interpolate through the relative transform tangent so every output remains
    # on SE(3) and follows the project rotation-first convention.
    delta = se3_log(se3_inverse(start) @ end)
    return np.stack([start @ se3_exp(float(np.clip(value, 0.0, 1.0)) * delta) for value in a], axis=0)


@dataclass(frozen=True)
class ScalarCalibrationLaw:
    '''Configurable scalar calibration law.

    Args:
        mode: One of ``constant``, ``linear``, ``sinusoidal``, ``piecewise`` or
            ``callable``. ``piecewise`` is zero-order hold, not linear interpolation.
        t_start: Start time for normalized linear/sinusoidal laws.
        t_end: End time for normalized linear/sinusoidal laws.
        value: Constant value or sinusoidal offset.
        start: Linear start value.
        end: Linear end value.
        amplitude: Sinusoidal amplitude.
        period: Sinusoidal period in seconds. If omitted, ``cycles`` over
            ``[t_start, t_end]`` is used.
        phase: Sinusoidal phase in radians.
        cycles: Number of sinusoidal cycles over ``[t_start, t_end]`` when
            ``period`` is omitted.
        control_times: Piecewise-constant control timestamps.
        control_values: Piecewise-constant scalar values.
        function: Arbitrary callable for ``mode='callable'``.
    '''

    mode: LawMode
    t_start: float = 0.0
    t_end: float = 1.0
    value: float | None = None
    start: float | None = None
    end: float | None = None
    amplitude: float | None = None
    period: float | None = None
    phase: float = 0.0
    cycles: float = 1.0
    control_times: ArrayLike | None = None
    control_values: ArrayLike | None = None
    function: Callable[[NDArray[np.float64]], ArrayLike] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        mode = str(self.mode).lower().replace("_", "-")
        if mode == "piecewise-constant":
            mode = "piecewise"
        if mode not in {"constant", "linear", "sinusoidal", "piecewise", "callable"}:
            raise ValueError("Unsupported scalar law mode")
        object.__setattr__(self, "mode", mode)

    def evaluate(self, timestamps: ArrayLike) -> NDArray[np.float64]:
        '''Evaluate the scalar law.

        Args:
            timestamps: Query timestamps with shape ``(N,)``.

        Returns:
            Scalar values with shape ``(N,)``.
        '''

        # Normalize and validate query times once; every branch below returns one
        # scalar value for each query timestamp.
        t = _as_timestamps(timestamps)

        # Callable mode delegates the law to user code but keeps strict output
        # shape validation so downstream emulation receives predictable arrays.
        if self.mode == "callable":
            if self.function is None:
                raise ValueError("function is required for callable scalar law")
            values = np.asarray(self.function(t), dtype=float).reshape(-1)
            if values.shape != t.shape:
                raise ValueError("callable scalar law must return shape (N,)")
            return values

        # Constant mode broadcasts one configured scalar over all query times.
        if self.mode == "constant":
            if self.value is None:
                raise ValueError("value is required for constant scalar law")
            return np.full(t.shape, float(self.value), dtype=float)

        # Linear mode uses clipped normalized time, so values stay at the start
        # before t_start and at the end after t_end.
        if self.mode == "linear":
            if self.start is None or self.end is None:
                raise ValueError("start and end are required for linear scalar law")
            alpha = _normalized_alpha(t, self.t_start, self.t_end)
            return (1.0 - alpha) * float(self.start) + alpha * float(self.end)

        # Sinusoidal mode oscillates around ``value``. Either a physical period
        # or a number of cycles across [t_start, t_end] may define the phase.
        if self.mode == "sinusoidal":
            offset = 0.0 if self.value is None else float(self.value)
            amplitude = 0.0 if self.amplitude is None else float(self.amplitude)
            if self.period is None:
                alpha = _normalized_alpha(t, self.t_start, self.t_end)
                angle = 2.0 * np.pi * float(self.cycles) * alpha + float(self.phase)
            else:
                period = float(self.period)
                if period <= 0.0:
                    raise ValueError("period must be positive")
                angle = 2.0 * np.pi * (t - float(self.t_start)) / period + float(self.phase)
            return offset + amplitude * np.sin(angle)

        # Piecewise mode is intentionally zero-order hold. This is useful for
        # experiments that switch calibration values by window or by manually
        # chosen control regions without inventing intermediate drift.
        control_times = _as_timestamps(self.control_times, "control_times")
        control_values = np.asarray(self.control_values, dtype=float).reshape(-1)
        _require_strictly_increasing(control_times, "control_times")
        if control_values.shape != control_times.shape:
            raise ValueError("control_values must have shape matching control_times")
        if not np.all(np.isfinite(control_values)):
            raise ValueError("control_values must contain only finite values")
        return control_values[_piecewise_constant_indices(control_times, t)]

    __call__ = evaluate


@dataclass(frozen=True)
class SE3CalibrationLaw:
    '''Configurable SE(3) calibration law.

    Args:
        mode: One of ``constant``, ``linear``, ``sinusoidal``, ``piecewise`` or
            ``callable``. ``piecewise`` is zero-order hold, not linear interpolation.
        t_start: Start time for normalized linear/sinusoidal laws.
        t_end: End time for normalized linear/sinusoidal laws.
        reference: Reference transform for constant or sinusoidal laws.
        start: Linear start transform. Defaults to ``reference``.
        end: Linear end transform.
        end_delta_xi: Optional end perturbation used as
            ``end = start @ Exp(end_delta_xi)``.
        amplitude_xi: Sinusoidal tangent amplitude around ``reference``.
        offset_xi: Constant tangent offset for sinusoidal laws.
        period: Sinusoidal period in seconds. If omitted, ``cycles`` over
            ``[t_start, t_end]`` is used.
        phase: Sinusoidal phase in radians.
        cycles: Number of sinusoidal cycles over ``[t_start, t_end]`` when
            ``period`` is omitted.
        control_times: Piecewise-constant control timestamps.
        control_poses: Piecewise-constant control transforms.
        control_tangents: Optional piecewise-constant control tangents around ``reference``.
        function: Arbitrary callable for ``mode='callable'``.
    '''

    mode: LawMode
    t_start: float = 0.0
    t_end: float = 1.0
    reference: ArrayLike | None = None
    start: ArrayLike | None = None
    end: ArrayLike | None = None
    end_delta_xi: ArrayLike | None = None
    amplitude_xi: ArrayLike | None = None
    offset_xi: ArrayLike | None = None
    period: float | None = None
    phase: float = 0.0
    cycles: float = 1.0
    control_times: ArrayLike | None = None
    control_poses: ArrayLike | None = None
    control_tangents: ArrayLike | None = None
    function: Callable[[NDArray[np.float64]], ArrayLike] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        mode = str(self.mode).lower().replace("_", "-")
        if mode == "piecewise-constant":
            mode = "piecewise"
        if mode not in {"constant", "linear", "sinusoidal", "piecewise", "callable"}:
            raise ValueError("Unsupported SE(3) law mode")
        object.__setattr__(self, "mode", mode)

    def _reference_matrix(self) -> NDArray[np.float64]:
        if self.reference is None:
            return np.eye(4)
        return as_matrix(self.reference, (4, 4), "reference")

    def evaluate(self, timestamps: ArrayLike) -> NDArray[np.float64]:
        '''Evaluate the SE(3) law.

        Args:
            timestamps: Query timestamps with shape ``(N,)``.

        Returns:
            Transform stack with shape ``(N, 4, 4)``.
        '''

        # Normalize query times first; each branch returns an (N, 4, 4) stack.
        t = _as_timestamps(timestamps)

        # Callable mode is the escape hatch for custom laws, still with strict
        # SE(3)-stack validation before the result leaves this function.
        if self.mode == "callable":
            if self.function is None:
                raise ValueError("function is required for callable SE(3) law")
            return _as_se3_stack(self.function(t), "callable SE(3) law result")

        # Most built-in modes are expressed relative to a reference transform.
        reference = self._reference_matrix()

        # Constant mode repeats the same valid SE(3) transform for every query.
        if self.mode == "constant":
            return np.repeat(reference[None, :, :], t.size, axis=0)

        # Linear mode interpolates on the Lie group, never elementwise in the
        # matrix entries. ``end_delta_xi`` is rotation-first [omega, v].
        if self.mode == "linear":
            start = reference if self.start is None else as_matrix(self.start, (4, 4), "start")
            if self.end is None:
                if self.end_delta_xi is None:
                    raise ValueError("end or end_delta_xi is required for linear SE(3) law")
                end = start @ se3_exp(as_vector(self.end_delta_xi, 6, "end_delta_xi"))
            else:
                end = as_matrix(self.end, (4, 4), "end")
            alpha = _normalized_alpha(t, self.t_start, self.t_end)
            return interpolate_se3_lie(start, end, alpha)

        # Sinusoidal mode creates a tangent perturbation around the reference,
        # then maps each tangent back to SE(3) with Exp.
        if self.mode == "sinusoidal":
            amplitude = np.zeros(6) if self.amplitude_xi is None else as_vector(self.amplitude_xi, 6, "amplitude_xi")
            offset = np.zeros(6) if self.offset_xi is None else as_vector(self.offset_xi, 6, "offset_xi")
            if self.period is None:
                alpha = _normalized_alpha(t, self.t_start, self.t_end)
                angle = 2.0 * np.pi * float(self.cycles) * alpha + float(self.phase)
            else:
                period = float(self.period)
                if period <= 0.0:
                    raise ValueError("period must be positive")
                angle = 2.0 * np.pi * (t - float(self.t_start)) / period + float(self.phase)
            tangents = offset[None, :] + np.sin(angle)[:, None] * amplitude[None, :]
            return np.stack([reference @ se3_exp(xi) for xi in tangents], axis=0)

        # Piecewise mode accepts either absolute control poses or tangent
        # controls around the reference, then applies zero-order hold.
        control_times = _as_timestamps(self.control_times, "control_times")
        _require_strictly_increasing(control_times, "control_times")
        if self.control_poses is not None:
            control_poses = _as_se3_stack(self.control_poses, "control_poses")
        elif self.control_tangents is not None:
            tangents = np.asarray(self.control_tangents, dtype=float)
            if tangents.shape != (control_times.size, 6):
                raise ValueError("control_tangents must have shape (len(control_times), 6)")
            control_poses = np.stack([reference @ se3_exp(xi) for xi in tangents], axis=0)
        else:
            raise ValueError("control_poses or control_tangents is required for piecewise SE(3) law")
        if control_poses.shape[0] != control_times.size:
            raise ValueError("control_poses must have one transform per control time")

        # Piecewise mode is zero-order hold: choose the most recent control
        # transform and keep it until the next control timestamp. There is no
        # Lie interpolation inside a segment in this mode.
        control_indices = _piecewise_constant_indices(control_times, t)
        return control_poses[control_indices].copy()

    __call__ = evaluate


@dataclass(frozen=True)
class IMUCalibrationEmulationResult:
    '''Output of time-varying IMU calibration emulation.

    Args:
        reference_timestamps: Physical/reference times ``s_old - tau_ref``.
        sensor_timestamps: Emulated sensor timestamps ``t_ref + tau_new(t_ref)``.
        gyroscope: Emulated gyroscope samples in the artificial IMU frame,
            shape ``(N, 3)``.
        accelerometer: Emulated accelerometer samples in the artificial IMU
            frame, shape ``(N, 3)``.
        T_B_I_truth: Artificial truth evaluated at ``reference_timestamps``,
            shape ``(N, 4, 4)``.
        tau_I_truth: Artificial temporal truth evaluated at
            ``reference_timestamps``, shape ``(N,)``.
        T_B_I_reference: Fixed reference calibration used to interpret the
            original stream.
        tau_I_reference: Fixed reference temporal offset used to interpret the
            original stream.
        bias_reference: Bias subtracted/re-added during gyroscope
            re-expression.
        include_lever_arm_correction: Whether accelerometer lever-arm
            correction was applied.
    '''

    reference_timestamps: NDArray[np.float64]
    sensor_timestamps: NDArray[np.float64]
    gyroscope: NDArray[np.float64]
    accelerometer: NDArray[np.float64]
    T_B_I_truth: NDArray[np.float64]
    tau_I_truth: NDArray[np.float64]
    T_B_I_reference: NDArray[np.float64]
    tau_I_reference: float
    bias_reference: NDArray[np.float64]
    include_lever_arm_correction: bool


def finite_difference_vector_stream(
    timestamps: ArrayLike,
    values: ArrayLike,
    *,
    smoothing_window: int | None = None,
) -> NDArray[np.float64]:
    '''Differentiate a timestamped 3D vector stream.

    Args:
        timestamps: Strictly increasing timestamps with shape ``(N,)``.
        values: Vector stream with shape ``(N, 3)``.
        smoothing_window: Optional centered moving-average window applied before
            differentiation. Values ``None`` and ``1`` disable smoothing.

    Returns:
        Numerical derivative with shape ``(N, 3)``.
    '''

    # Validate and optionally smooth the stream before differentiating each
    # component against the actual nonuniform timestamps.
    t = _as_timestamps(timestamps)
    _require_strictly_increasing(t, "timestamps")
    x = _as_vector_stream(values, t, "values")

    if smoothing_window is not None and int(smoothing_window) > 1:
        window = int(smoothing_window)
        kernel = np.ones(window, dtype=float) / float(window)
        padded = np.pad(x, ((window // 2, window - 1 - window // 2), (0, 0)), mode="edge")
        x = np.column_stack([np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(3)])

    return np.column_stack([np.gradient(x[:, axis], t, edge_order=1) for axis in range(3)])


def warp_sensor_timestamps(
    sensor_timestamps: ArrayLike,
    tau_reference: float,
    tau_law: ScalarCalibrationLaw | Callable[[ArrayLike], ArrayLike],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    '''Warp sensor timestamps according to the factor-graph tau convention.

    For an original sample timestamp ``s_old`` and fixed reference offset
    ``tau_reference``, the physical/reference time is
    ``t_ref = s_old - tau_reference``. The emulated timestamp is then
    ``s_new = t_ref + tau_new(t_ref)``.

    Args:
        sensor_timestamps: Original sensor timestamps with shape ``(N,)``.
        tau_reference: Original/reference temporal offset.
        tau_law: Scalar law evaluated at reference times.

    Returns:
        Tuple ``(reference_timestamps, emulated_sensor_timestamps, tau_truth)``.

    Raises:
        ValueError: If the warped timestamps are not strictly increasing.
    '''

    # Convert original sensor times to physical/reference times using the same
    # sign convention as the MROB IMU factors.
    sensor_times = _as_timestamps(sensor_timestamps, "sensor_timestamps")
    _require_strictly_increasing(sensor_times, "sensor_timestamps")
    reference_times = sensor_times - float(tau_reference)
    tau_truth = np.asarray(tau_law(reference_times), dtype=float).reshape(-1)
    if tau_truth.shape != reference_times.shape:
        raise ValueError("tau_law must return shape matching sensor_timestamps")
    if not np.all(np.isfinite(tau_truth)):
        raise ValueError("tau_law returned non-finite values")
    # Add the artificial time offset at each physical sample. Reject invalid
    # laws instead of sorting, clipping, or silently changing the sample order.
    emulated_times = reference_times + tau_truth
    _require_strictly_increasing(emulated_times, "emulated sensor timestamps")
    return reference_times, emulated_times, tau_truth


def lever_arm_acceleration(
    omega_body: ArrayLike,
    alpha_body: ArrayLike,
    lever_arm_body: ArrayLike,
) -> NDArray[np.float64]:
    '''Compute rigid-body lever-arm acceleration in the body frame.

    Args:
        omega_body: Angular velocity samples in body coordinates, shape
            ``(N, 3)``.
        alpha_body: Angular acceleration samples in body coordinates, shape
            ``(N, 3)``.
        lever_arm_body: Lever arm in body coordinates, shape ``(3,)`` or
            ``(N, 3)``.

    Returns:
        Lever-arm acceleration samples with shape ``(N, 3)``.
    '''

    # Broadcast a fixed lever arm when needed, then evaluate alpha x r and the
    # centripetal omega x (omega x r) term sample by sample.
    omega = np.asarray(omega_body, dtype=float)
    alpha = np.asarray(alpha_body, dtype=float)
    if omega.ndim != 2 or omega.shape[1] != 3:
        raise ValueError("omega_body must have shape (N, 3)")
    if alpha.shape != omega.shape:
        raise ValueError("alpha_body must have shape matching omega_body")
    lever = np.asarray(lever_arm_body, dtype=float)
    if lever.shape == (3,):
        lever = np.repeat(lever[None, :], omega.shape[0], axis=0)
    if lever.shape != omega.shape:
        raise ValueError("lever_arm_body must have shape (3,) or (N, 3)")
    return np.cross(alpha, lever) + np.cross(omega, np.cross(omega, lever))


def emulate_time_varying_imu_calibration(
    *,
    sensor_timestamps: ArrayLike,
    gyroscope: ArrayLike,
    accelerometer: ArrayLike,
    T_B_I_reference: ArrayLike,
    tau_I_reference: float,
    T_B_I_law: SE3CalibrationLaw | Callable[[ArrayLike], ArrayLike],
    tau_I_law: ScalarCalibrationLaw | Callable[[ArrayLike], ArrayLike],
    bias_reference: ArrayLike | None = None,
    include_lever_arm_correction: bool = False,
    angular_acceleration_smoothing_window: int | None = None,
) -> IMUCalibrationEmulationResult:
    '''Emulate real IMU samples under time-varying ``T_B_I`` and ``tau_I``.

    Args:
        sensor_timestamps: Original real IMU timestamps ``s_old``.
        gyroscope: Original real gyroscope samples in IMU coordinates,
            shape ``(N, 3)``.
        accelerometer: Original real accelerometer samples in IMU coordinates,
            shape ``(N, 3)``.
        T_B_I_reference: Fixed reference body-from-IMU transform that
            approximately produced the original stream.
        tau_I_reference: Fixed reference temporal offset.
        T_B_I_law: Artificial body-from-IMU transform law evaluated at physical
            reference times ``s_old - tau_I_reference``.
        tau_I_law: Artificial temporal-offset law evaluated at physical
            reference times.
        bias_reference: Optional gyroscope bias in IMU coordinates. If omitted,
            zero bias is used.
        include_lever_arm_correction: Whether to remove the reference lever-arm
            acceleration and add the artificial lever-arm acceleration using
            real gyroscope-derived body angular motion.
        angular_acceleration_smoothing_window: Optional smoothing window for
            the gyroscope-derived angular acceleration.

    Returns:
        ``IMUCalibrationEmulationResult`` containing warped timestamps,
        re-expressed samples and artificial truth values.
    '''

    # Validate and copy every input stream first. The emulation must never mutate
    # real measured arrays that the notebook may inspect later.
    sensor_times = _as_timestamps(sensor_timestamps, "sensor_timestamps")
    _require_strictly_increasing(sensor_times, "sensor_timestamps")
    gyro_real = _as_vector_stream(gyroscope, sensor_times, "gyroscope")
    accel_real = _as_vector_stream(accelerometer, sensor_times, "accelerometer")
    T_ref = as_matrix(T_B_I_reference, (4, 4), "T_B_I_reference")
    bias = np.zeros(3) if bias_reference is None else as_vector(bias_reference, 3, "bias_reference")

    # Warp timestamps and evaluate the artificial spatial law at the physical
    # sample times. No trajectory lookup happens in this step.
    reference_times, emulated_sensor_times, tau_truth = warp_sensor_timestamps(
        sensor_times,
        tau_I_reference,
        tau_I_law,
    )
    T_truth = _as_se3_stack(T_B_I_law(reference_times), "T_B_I_law(reference_times)")
    if T_truth.shape[0] != sensor_times.size:
        raise ValueError("T_B_I_law must return one transform per sensor timestamp")

    C_ref = T_ref[:3, :3]
    C_new = T_truth[:, :3, :3]

    # Re-express the measured angular motion in the body frame, then in the
    # artificial IMU frame. The bias is subtracted and re-added to keep the
    # experiment focused on calibration drift rather than changing bias truth.
    omega_body = (C_ref @ (gyro_real - bias[None, :]).T).T
    gyro_emulated = np.einsum("nij,nj->ni", np.transpose(C_new, (0, 2, 1)), omega_body) + bias[None, :]

    # Re-express the measured specific force. The orientation-only path exactly
    # preserves the original stream when C_new == C_ref.
    accel_body_ref_sensor = (C_ref @ accel_real.T).T

    # Optional lever-arm correction removes the old rigid-body lever term and
    # adds the new one, estimated only from the real gyroscope stream.
    if include_lever_arm_correction:
        alpha_body = finite_difference_vector_stream(
            reference_times,
            omega_body,
            smoothing_window=angular_acceleration_smoothing_window,
        )
        r_ref = T_ref[:3, 3]
        r_new = T_truth[:, :3, 3]
        accel_body_origin = accel_body_ref_sensor - lever_arm_acceleration(omega_body, alpha_body, r_ref)
        accel_body_new_sensor = accel_body_origin + lever_arm_acceleration(omega_body, alpha_body, r_new)
    else:
        accel_body_new_sensor = accel_body_ref_sensor

    # Rotate the body-frame specific force into each artificial IMU frame and
    # package all truth arrays needed by diagnostics and rolling-window plots.
    accel_emulated = np.einsum("nij,nj->ni", np.transpose(C_new, (0, 2, 1)), accel_body_new_sensor)

    return IMUCalibrationEmulationResult(
        reference_timestamps=reference_times,
        sensor_timestamps=emulated_sensor_times,
        gyroscope=gyro_emulated,
        accelerometer=accel_emulated,
        T_B_I_truth=T_truth,
        tau_I_truth=tau_truth,
        T_B_I_reference=T_ref.copy(),
        tau_I_reference=float(tau_I_reference),
        bias_reference=bias.copy(),
        include_lever_arm_correction=bool(include_lever_arm_correction),
    )


def evaluate_truth_at_window_midpoints(
    results: Sequence[object],
    T_B_I_law: SE3CalibrationLaw | Callable[[ArrayLike], ArrayLike],
    tau_I_law: ScalarCalibrationLaw | Callable[[ArrayLike], ArrayLike],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    '''Evaluate artificial truth at rolling-window midpoints.

    Args:
        results: Rolling ``CalibrationWindowResult`` sequence.
        T_B_I_law: Artificial spatial calibration law.
        tau_I_law: Artificial temporal calibration law.

    Returns:
        Tuple ``(window_midpoints, T_B_I_truth, tau_I_truth)`` where shapes are
        ``(N,)``, ``(N, 4, 4)`` and ``(N,)``.
    '''

    # Rolling graphs estimate one constant calibration per window, so compare it
    # to the artificial law at the window midpoint.
    midpoints = np.asarray([0.5 * (result.window_start + result.window_end) for result in results], dtype=float)
    return midpoints, _as_se3_stack(T_B_I_law(midpoints), "T_B_I_law(midpoints)"), np.asarray(tau_I_law(midpoints), dtype=float).reshape(-1)


def calibration_tracking_summary(
    results: Sequence[object],
    *,
    T_B_I_truth: ArrayLike,
    tau_I_truth: ArrayLike,
    bias_truth: ArrayLike,
    T_B_L_truth: ArrayLike,
    tau_L_truth: float,
) -> list[dict[str, object]]:
    '''Build rolling-window calibration tracking summary rows.

    Args:
        results: Rolling ``CalibrationWindowResult`` sequence.
        T_B_I_truth: Artificial body-from-IMU truth at window midpoints,
            shape ``(N, 4, 4)``.
        tau_I_truth: Artificial temporal truth at window midpoints, shape
            ``(N,)``.
        bias_truth: Bias reference with shape ``(3,)``.
        T_B_L_truth: Body-from-LiDAR truth/reference with shape ``(4, 4)``.
        tau_L_truth: LiDAR temporal truth/reference.

    Returns:
        List of dictionaries suitable for a pandas DataFrame.
    '''

    import utils

    # Validate truth arrays once, then build table rows with estimates and
    # errors against the artificial time-varying truth.
    result_list = list(results)
    T_I_truth = _as_se3_stack(T_B_I_truth, "T_B_I_truth")
    tau_truth = np.asarray(tau_I_truth, dtype=float).reshape(-1)
    bias_ref = as_vector(bias_truth, 3, "bias_truth")
    T_L_truth = as_matrix(T_B_L_truth, (4, 4), "T_B_L_truth")
    if T_I_truth.shape[0] != len(result_list) or tau_truth.shape != (len(result_list),):
        raise ValueError("truth arrays must match number of results")

    rows: list[dict[str, object]] = []
    for index, result in enumerate(result_list):
        rows.append(
            {
                "window": result.window_index,
                "start": result.window_start,
                "end": result.window_end,
                "midpoint": 0.5 * (result.window_start + result.window_end),
                "poses": result.pose_timestamps.size,
                "chi2_before": result.chi2_before,
                "chi2_after": result.chi2_after,
                "tau_I_estimate": result.tau_I,
                "tau_I_truth": float(tau_truth[index]),
                "tau_I_error": result.tau_I - float(tau_truth[index]),
                "T_B_I_rotation_error": float(utils.calculate_se3_rotation_distances(result.T_B_I, T_I_truth[index])),
                "T_B_I_translation_error": float(utils.calculate_se3_translation_distances(result.T_B_I, T_I_truth[index])),
                "bias_estimate": result.bias_g,
                "bias_error": result.bias_g - bias_ref,
                "T_B_L_error": float(utils.calculate_se3_distances(result.T_B_L, T_L_truth)),
                "tau_L_estimate": result.tau_L,
                "tau_L_error": result.tau_L - float(tau_L_truth),
            }
        )
    return rows
