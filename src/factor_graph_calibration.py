from __future__ import annotations

import copy
import gc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import mrob
import numpy as np


def subsample_indices(indices: Iterable[int], final_length: Optional[int]) -> np.ndarray:
    """
    Select almost equally spaced values from a sequence of integer indices.

    The first and last input indices are always preserved. If final_length is
    None or is not smaller than the input length, all indices are returned.
    """
    indices = np.asarray(list(indices), dtype=np.int64).reshape(-1)

    if len(indices) == 0:
        return indices.copy()

    if final_length is None or final_length >= len(indices):
        return indices.copy()

    if final_length < 2:
        raise ValueError("final_length must be at least 2 when subsampling measurement support")

    positions = np.rint(np.linspace(0, len(indices) - 1, int(final_length))).astype(np.int64)
    positions[0] = 0
    positions[-1] = len(indices) - 1

    if len(np.unique(positions)) != len(positions):
        raise RuntimeError("Failed to generate unique equally spaced subsampling positions")

    return indices[positions]


def select_time_support_indices(timestamps: Sequence[float], support_start: float, support_end: float, final_length: Optional[int]) -> np.ndarray:
    """
    Return subsampled indices whose first and last timestamps bracket a time interval.

    One sample before support_start and one sample after support_end are included
    whenever they exist. This is important because the C++ factors interpolate
    measurements at shifted query times.
    """
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)

    if len(timestamps) < 2:
        raise ValueError("At least two timestamped measurements are required")

    if not np.all(np.isfinite(timestamps)):
        raise ValueError("Measurement timestamps must be finite")

    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Measurement timestamps must be strictly increasing")

    if support_end <= support_start:
        raise ValueError("support_end must be greater than support_start")

    if support_start < timestamps[0] or support_end > timestamps[-1]:
        raise IndexError(f"Requested support [{support_start}, {support_end}] lies outside measurement timestamps [{timestamps[0]}, {timestamps[-1]}]")

    i_start = max(int(np.searchsorted(timestamps, support_start, side="right")) - 1, 0)
    i_end = min(int(np.searchsorted(timestamps, support_end, side="left")) + 1, len(timestamps))

    indices = np.arange(i_start, i_end, dtype=np.int64)

    if len(indices) < 2:
        raise ValueError("Selected measurement support contains fewer than two samples")

    return subsample_indices(indices, final_length)


def _as_vector3(values: Sequence[float], name: str) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)

    if values.shape != (3,):
        raise ValueError(f"{name} must contain exactly three values")

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")

    return values.copy()


def _as_pose_matrix(pose: Any) -> np.ndarray:
    if isinstance(pose, mrob.SE3):
        return np.asarray(pose.T(), dtype=float).copy()

    pose = np.asarray(pose, dtype=float)

    if pose.shape == (6,):
        return np.asarray(mrob.SE3(pose).T(), dtype=float)

    if pose.shape != (4, 4):
        raise ValueError("A pose must be an mrob.SE3 object, a six-dimensional tangent vector, or a 4x4 transformation matrix")

    if not np.all(np.isfinite(pose)):
        raise ValueError("Pose matrix must be finite")

    if not mrob.isSE3(pose):
        raise ValueError("Pose matrix must be a valid SE(3) transformation")

    return pose.copy()


def _as_mrob_se3(pose: Any) -> mrob.SE3:
    if isinstance(pose, mrob.SE3):
        return mrob.SE3(pose)

    pose = np.asarray(pose, dtype=float)

    if pose.shape == (6,):
        return mrob.SE3(pose)

    return mrob.SE3(_as_pose_matrix(pose))


def _interpolate_vector(timestamps: Sequence[float], values: np.ndarray, query_time: float) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)

    if values.shape != (len(timestamps), 3):
        raise ValueError("Interpolated vector measurements must have shape (N, 3)")

    if query_time < timestamps[0] or query_time > timestamps[-1]:
        raise IndexError(f"Query time {query_time} lies outside measurement support [{timestamps[0]}, {timestamps[-1]}]")

    i_upper = int(np.searchsorted(timestamps, query_time, side="left"))

    if i_upper == 0:
        return values[0].copy()

    if i_upper >= len(timestamps):
        return values[-1].copy()

    if timestamps[i_upper] == query_time:
        return values[i_upper].copy()

    i_lower = i_upper - 1
    alpha = (query_time - timestamps[i_lower]) / (timestamps[i_upper] - timestamps[i_lower])
    return (1.0 - alpha) * values[i_lower] + alpha * values[i_upper]

def _interpolate_pose(timestamps: Sequence[float], poses: Sequence[Any], query_time: float) -> np.ndarray:
    """
    Interpolate an SE(3) pose along the relative Lie-algebra increment.
    """
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)

    if query_time < timestamps[0] or query_time > timestamps[-1]:
        raise IndexError(f"Query time {query_time} lies outside pose support [{timestamps[0]}, {timestamps[-1]}]")

    i_upper = int(np.searchsorted(timestamps, query_time, side="left"))

    if i_upper == 0:
        return _as_pose_matrix(poses[0])

    if i_upper >= len(timestamps):
        return _as_pose_matrix(poses[-1])

    if timestamps[i_upper] == query_time:
        return _as_pose_matrix(poses[i_upper])

    i_lower = i_upper - 1
    alpha = (query_time - timestamps[i_lower]) / (timestamps[i_upper] - timestamps[i_lower])

    T_left = _as_mrob_se3(poses[i_lower])
    T_right = _as_mrob_se3(poses[i_upper])
    relative_xi = T_left.inv().mul(T_right).Ln()

    return np.asarray(T_left.mul(mrob.SE3(alpha * relative_xi)).T(), dtype=float)

def _integrate_vector_interval(timestamps: Sequence[float], values: np.ndarray, start_time: float, end_time: float) -> np.ndarray:
    """
    Integrate piecewise-linearly interpolated three-dimensional measurements.
    """
    timestamps = np.asarray(timestamps, dtype=float).reshape(-1)
    values = np.asarray(values, dtype=float)

    if values.shape != (len(timestamps), 3):
        raise ValueError("Integrated vector measurements must have shape (N, 3)")

    if start_time < timestamps[0] or end_time > timestamps[-1]:
        raise IndexError(f"Integration interval [{start_time}, {end_time}] lies outside measurement support [{timestamps[0]}, {timestamps[-1]}]")

    interior_times = timestamps[(timestamps > start_time) & (timestamps < end_time)]
    integration_times = np.concatenate(([start_time], interior_times, [end_time]))
    integration_values = np.vstack([_interpolate_vector(timestamps, values, time) for time in integration_times])
    dt = np.diff(integration_times)

    return np.sum(0.5 * dt[:, None] * (integration_values[:-1] + integration_values[1:]), axis=0)


def _information_matrix(value: Any, dimension: int, factor_index: int = 0) -> np.ndarray:
    """
    Convert a scalar, one matrix, or a sequence of matrices to one information matrix.
    """
    if value is None:
        return np.eye(dimension)

    value = np.asarray(value, dtype=float)

    if value.ndim == 0:
        matrix = np.eye(dimension) * float(value)
    elif value.shape == (dimension, dimension):
        matrix = value
    elif value.ndim == 3 and value.shape[1:] == (dimension, dimension):
        matrix = value[min(factor_index, len(value) - 1)]
    else:
        raise ValueError(f"Information must be a scalar, a ({dimension}, {dimension}) matrix, or an (N, {dimension}, {dimension}) array")

    if not np.all(np.isfinite(matrix)):
        raise ValueError("Information matrix must be finite")

    if not np.allclose(matrix, matrix.T, atol=1e-10):
        raise ValueError("Information matrix must be symmetric")

    return matrix.copy()


@dataclass
class CalibrationWindowResult:
    window_index: int
    window_start: float
    window_end: float
    pose_timestamps: np.ndarray
    trajectory_poses: np.ndarray
    T_B_I: Optional[np.ndarray]
    T_B_L: Optional[np.ndarray]
    bias_g: Optional[np.ndarray]
    tau_I: Optional[float]
    tau_L: Optional[float]
    chi2_before: float
    chi2_after: float
    factor_counts: Dict[str, int] = field(default_factory=dict)


class FactorGraphCalibration:
    """
    Python wrapper around the MROB LiDAR-IMU calibration factors.

    A new MROB graph is built for every batch or rolling window. Rolling-window
    continuity is preserved by using solved poses and calibration variables from
    the preceding window as initial states in the next overlapping window.
    """

    def __init__(
        self,
        imu_samples_per_factor: Optional[int] = 64,
        lidar_samples_per_factor: Optional[int] = 32,
        imu_time_offset_margin: float = 0.25,
        lidar_time_offset_margin: float = 0.25,
        gravity_world: Sequence[float] = (0.0, 0.0, -9.81),
        gyro_information: Any = 1.0,
        accel_information: Any = 1.0,
        lidar_information: Any = 1.0,
        bias_regularization_information: Any = None,
        tau_I_regularization_information: Any = None,
        tau_L_regularization_information: Any = None,
        T_B_I_regularization_information: Any = None,
        T_B_L_regularization_information: Any = None,
        include_gyro_factors: bool = True,
        include_accel_factors: bool = True,
        include_lidar_factors: bool = True,
        T_B_I_anchor: bool = False,
        T_B_L_anchor: bool = False,
        bias_anchor: bool = False,
        tau_I_anchor: bool = False,
        tau_L_anchor: bool = False,
        anchor_first_pose: bool = True,
        anchor_first_pose_each_window: bool = False,
        anchor_last_pose: bool = False,
        anchor_all_poses: bool = False,
        accel_norm_tolerance: Optional[float] = None,
        accel_gyro_threshold: Optional[float] = None,
        gyro_factor_stride: int = 1,
        accel_factor_stride: int = 1,
        lidar_factor_stride: int = 1,
        method: str = "LM",
        maxIters: int = 30,
        lambdaParam: float = 1e-5,
        solutionTolerance: float = 1e-6,
        solver_verbose: bool = False,
        scheduler: Optional[Sequence[Tuple[float, int]]] = None,
    ):
        self._init_filter_parameters(
            imu_samples_per_factor=imu_samples_per_factor,
            lidar_samples_per_factor=lidar_samples_per_factor,
            imu_time_offset_margin=imu_time_offset_margin,
            lidar_time_offset_margin=lidar_time_offset_margin,
            gravity_world=gravity_world,
            gyro_information=gyro_information,
            accel_information=accel_information,
            lidar_information=lidar_information,
            bias_regularization_information=bias_regularization_information,
            tau_I_regularization_information=tau_I_regularization_information,
            tau_L_regularization_information=tau_L_regularization_information,
            T_B_I_regularization_information=T_B_I_regularization_information,
            T_B_L_regularization_information=T_B_L_regularization_information,
            include_gyro_factors=include_gyro_factors,
            include_accel_factors=include_accel_factors,
            include_lidar_factors=include_lidar_factors,
            T_B_I_anchor=T_B_I_anchor,
            T_B_L_anchor=T_B_L_anchor,
            bias_anchor=bias_anchor,
            tau_I_anchor=tau_I_anchor,
            tau_L_anchor=tau_L_anchor,
            anchor_first_pose=anchor_first_pose,
            anchor_first_pose_each_window=anchor_first_pose_each_window,
            anchor_last_pose=anchor_last_pose,
            anchor_all_poses=anchor_all_poses,
            accel_norm_tolerance=accel_norm_tolerance,
            accel_gyro_threshold=accel_gyro_threshold,
            gyro_factor_stride=gyro_factor_stride,
            accel_factor_stride=accel_factor_stride,
            lidar_factor_stride=lidar_factor_stride,
            method=method,
            maxIters=maxIters,
            lambdaParam=lambdaParam,
            solutionTolerance=solutionTolerance,
            solver_verbose=solver_verbose,
            scheduler=scheduler,
        )
        self.reset(clear_rolling_state=True)

    def _init_filter_parameters(
        self,
        imu_samples_per_factor,
        lidar_samples_per_factor,
        imu_time_offset_margin,
        lidar_time_offset_margin,
        gravity_world,
        gyro_information,
        accel_information,
        lidar_information,
        bias_regularization_information,
        tau_I_regularization_information,
        tau_L_regularization_information,
        T_B_I_regularization_information,
        T_B_L_regularization_information,
        include_gyro_factors,
        include_accel_factors,
        include_lidar_factors,
        T_B_I_anchor,
        T_B_L_anchor,
        bias_anchor,
        tau_I_anchor,
        tau_L_anchor,
        anchor_first_pose,
        anchor_first_pose_each_window,
        anchor_last_pose,
        anchor_all_poses,
        accel_norm_tolerance,
        accel_gyro_threshold,
        gyro_factor_stride,
        accel_factor_stride,
        lidar_factor_stride,
        method,
        maxIters,
        lambdaParam,
        solutionTolerance,
        solver_verbose,
        scheduler,
    ):
        """Initialize factor selection, graph structure, regularization, and solver parameters."""
        self._imu_samples_per_factor = imu_samples_per_factor
        self._lidar_samples_per_factor = lidar_samples_per_factor
        self._imu_time_offset_margin = float(imu_time_offset_margin)
        self._lidar_time_offset_margin = float(lidar_time_offset_margin)

        self._gravity_world = _as_vector3(gravity_world, "gravity_world")
        self._gyro_information = copy.deepcopy(gyro_information)
        self._accel_information = copy.deepcopy(accel_information)
        self._lidar_information = copy.deepcopy(lidar_information)

        self._bias_regularization_information = copy.deepcopy(bias_regularization_information)
        self._tau_I_regularization_information = copy.deepcopy(tau_I_regularization_information)
        self._tau_L_regularization_information = copy.deepcopy(tau_L_regularization_information)
        self._T_B_I_regularization_information = copy.deepcopy(T_B_I_regularization_information)
        self._T_B_L_regularization_information = copy.deepcopy(T_B_L_regularization_information)

        self._include_gyro_factors = bool(include_gyro_factors)
        self._include_accel_factors = bool(include_accel_factors)
        self._include_lidar_factors = bool(include_lidar_factors)

        self._T_B_I_anchor = bool(T_B_I_anchor)
        self._T_B_L_anchor = bool(T_B_L_anchor)
        self._bias_anchor = bool(bias_anchor)
        self._tau_I_anchor = bool(tau_I_anchor)
        self._tau_L_anchor = bool(tau_L_anchor)

        self._anchor_first_pose = bool(anchor_first_pose)
        self._anchor_first_pose_each_window = bool(anchor_first_pose_each_window)
        self._anchor_last_pose = bool(anchor_last_pose)
        self._anchor_all_poses = bool(anchor_all_poses)

        self._accel_norm_tolerance = accel_norm_tolerance
        self._accel_gyro_threshold = accel_gyro_threshold

        self._gyro_factor_stride = int(gyro_factor_stride)
        self._accel_factor_stride = int(accel_factor_stride)
        self._lidar_factor_stride = int(lidar_factor_stride)

        self._method = str(method).upper()
        self._maxIters = int(maxIters)
        self._lambdaParam = float(lambdaParam)
        self._solutionTolerance = float(solutionTolerance)
        self._solver_verbose = bool(solver_verbose)
        self._scheduler = None if scheduler is None else [(float(value), int(iterations)) for value, iterations in scheduler]

        if self._imu_samples_per_factor is not None and self._imu_samples_per_factor < 2:
            raise ValueError("imu_samples_per_factor must be at least 2 or None")

        if self._lidar_samples_per_factor is not None and self._lidar_samples_per_factor < 2:
            raise ValueError("lidar_samples_per_factor must be at least 2 or None")

        if self._imu_time_offset_margin < 0 or self._lidar_time_offset_margin < 0:
            raise ValueError("Measurement support margins must be nonnegative")

        if min(self._gyro_factor_stride, self._accel_factor_stride, self._lidar_factor_stride) < 1:
            raise ValueError("Factor strides must be positive integers")

    @property
    def filter_object(self):
        return self._filter_object

    @property
    def nodes(self):
        return self._nodes

    @property
    def nodes_pose(self):
        return self._nodes_pose

    @property
    def node_T_B_I(self):
        return self._node_T_B_I

    @property
    def node_T_B_L(self):
        return self._node_T_B_L

    @property
    def node_bias_g(self):
        return self._node_bias_g

    @property
    def node_tau_I(self):
        return self._node_tau_I

    @property
    def node_tau_L(self):
        return self._node_tau_L

    @property
    def states(self):
        if self._states_cache is None:
            self._states_cache = self._filter_object.get_estimated_state()
        return self._states_cache

    @property
    def states_init(self):
        return self._states_init

    @property
    def trajectory_poses(self):
        if self._trajectory_poses_cache is None:
            self._trajectory_poses_cache = np.array([np.asarray(self.states[node_id], dtype=float) for node_id in self.nodes_pose])
        return self._trajectory_poses_cache

    @property
    def T_B_I(self):
        if self.node_T_B_I is None:
            return None
        return np.asarray(self.states[self.node_T_B_I], dtype=float)

    @property
    def T_B_L(self):
        if self.node_T_B_L is None:
            return None
        return np.asarray(self.states[self.node_T_B_L], dtype=float)

    @property
    def bias_g(self):
        if self.node_bias_g is None:
            return None
        return np.asarray(self.states[self.node_bias_g], dtype=float).reshape(3)

    @property
    def tau_I(self):
        if self.node_tau_I is None:
            return None
        return float(np.asarray(self.states[self.node_tau_I], dtype=float).reshape(-1)[0])

    @property
    def tau_L(self):
        if self.node_tau_L is None:
            return None
        return float(np.asarray(self.states[self.node_tau_L], dtype=float).reshape(-1)[0])

    @property
    def chi2(self):
        self._chi2 = float(self._filter_object.chi2())
        return self._chi2

    @property
    def chi2_prev(self):
        return self._chi2_prev

    @property
    def factor_counts(self):
        return self._factor_counts.copy()

    @property
    def factor_metadata(self):
        return copy.deepcopy(self._factor_metadata)

    @property
    def rolling_results(self):
        return list(self._rolling_results)

    @property
    def rolling_trajectory(self):
        if len(self._rolling_output_pose_cache) == 0:
            return np.empty(0), np.empty((0, 4, 4))

        keys = sorted(self._rolling_output_pose_cache)
        timestamps = np.array([self._rolling_output_pose_times[key] for key in keys])
        poses = np.array([self._rolling_output_pose_cache[key] for key in keys])
        return timestamps, poses

    @property
    def imu_samples_per_factor(self):
        return self._imu_samples_per_factor

    @imu_samples_per_factor.setter
    def imu_samples_per_factor(self, value):
        if value is not None and value < 2:
            raise ValueError("imu_samples_per_factor must be at least 2 or None")
        self._imu_samples_per_factor = value

    @property
    def lidar_samples_per_factor(self):
        return self._lidar_samples_per_factor

    @lidar_samples_per_factor.setter
    def lidar_samples_per_factor(self, value):
        if value is not None and value < 2:
            raise ValueError("lidar_samples_per_factor must be at least 2 or None")
        self._lidar_samples_per_factor = value

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        self._method = str(value).upper()

    @property
    def maxIters(self):
        return self._maxIters

    @maxIters.setter
    def maxIters(self, value):
        self._maxIters = int(value)

    @property
    def lambdaParam(self):
        return self._lambdaParam

    @lambdaParam.setter
    def lambdaParam(self, value):
        self._lambdaParam = float(value)

    @property
    def solutionTolerance(self):
        return self._solutionTolerance

    @solutionTolerance.setter
    def solutionTolerance(self, value):
        self._solutionTolerance = float(value)

    @property
    def solver_verbose(self):
        return self._solver_verbose

    @solver_verbose.setter
    def solver_verbose(self, value):
        self._solver_verbose = bool(value)

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, value):
        self._scheduler = None if value is None else [(float(lmbda), int(iterations)) for lmbda, iterations in value]

    @property
    def T_B_I_anchor(self):
        return self._T_B_I_anchor

    @T_B_I_anchor.setter
    def T_B_I_anchor(self, value):
        self._T_B_I_anchor = bool(value)

    @property
    def T_B_L_anchor(self):
        return self._T_B_L_anchor

    @T_B_L_anchor.setter
    def T_B_L_anchor(self, value):
        self._T_B_L_anchor = bool(value)

    @property
    def bias_anchor(self):
        return self._bias_anchor

    @bias_anchor.setter
    def bias_anchor(self, value):
        self._bias_anchor = bool(value)

    @property
    def tau_I_anchor(self):
        return self._tau_I_anchor

    @tau_I_anchor.setter
    def tau_I_anchor(self, value):
        self._tau_I_anchor = bool(value)

    @property
    def tau_L_anchor(self):
        return self._tau_L_anchor

    @tau_L_anchor.setter
    def tau_L_anchor(self, value):
        self._tau_L_anchor = bool(value)

    def _invalidate_state_cache(self):
        self._states_cache = None
        self._trajectory_poses_cache = None

    def reset(self, clear_rolling_state: bool = False):
        # self._filter_object = mrob.FGraph()
        self._nodes = []
        self._nodes_pose = []
        self._node_T_B_I = None
        self._node_T_B_L = None
        self._node_bias_g = None
        self._node_tau_I = None
        self._node_tau_L = None

        self._factor_ids = {"gyro": [], "accel": [], "lidar": [], "bias_prior": [], "tau_I_prior": [], "tau_L_prior": [], "T_B_I_prior": [], "T_B_L_prior": []}
        self._factor_counts = {name: 0 for name in self._factor_ids}
        self._factor_metadata = {"gyro": [], "accel": [], "lidar": []}

        self._pose_timestamps = np.empty(0)
        self._initial_poses = np.empty((0, 4, 4))
        self._imu_timestamps = None
        self._angular_velocity_imu = None
        self._specific_force_imu = None
        self._lidar_timestamps = None
        self._lidar_odometry_poses = None

        self._states_cache = None
        self._trajectory_poses_cache = None
        self._states_init = None
        self._chi2 = 0.0
        self._chi2_prev = 0.0
        self._last_window_result = None

        if clear_rolling_state:
            self.clear_rolling_state()
            
        # print("reset: releasing old C++ graph", flush=True)

        old_filter_object = getattr(self, "_filter_object", None)
        self._filter_object = None

        if old_filter_object is not None:
            # print(f"Old filter object is not none {old_filter_object}")
            del old_filter_object
            gc.collect()

        # print("reset: old C++ graph released", flush=True)

        self._filter_object = mrob.FGraph()

        # print("reset: new C++ graph created", flush=True)

    def clear_rolling_state(self):
        self._rolling_pose_cache = {}
        self._rolling_calibration_state = {}
        self._rolling_results = []
        self._rolling_output_pose_cache = {}
        self._rolling_output_pose_times = {}

    def _node_mode(self, anchor: bool):
        return mrob.NODE_ANCHOR if anchor else mrob.NODE_STANDARD

    def add_pose_node(self, pose: Any, anchor: bool = False) -> int:
        node_id = self._filter_object.add_node_pose_3d(_as_mrob_se3(pose), mode=self._node_mode(anchor))
        self._nodes.append(node_id)
        self._invalidate_state_cache()
        return node_id

    def add_vector_node(self, values: Sequence[float], anchor: bool = False) -> int:
        node_id = self._filter_object.add_node_landmark_3d(_as_vector3(values, "values"), mode=self._node_mode(anchor))
        self._nodes.append(node_id)
        self._invalidate_state_cache()
        return node_id

    def add_scalar_node(self, value: float, anchor: bool = False) -> int:
        node_id = self._filter_object.add_node_scalar(float(value), mode=self._node_mode(anchor))
        self._nodes.append(node_id)
        self._invalidate_state_cache()
        return node_id

    def add_gyro_calibration_factor(self, pose_time_origin: float, pose_time_target: float, timestamps: np.ndarray, angular_velocity_imu: np.ndarray, node_origin: int, node_target: int, information: np.ndarray) -> int:
        factor_id = self._filter_object.add_factor_gyro_calib_prop(pose_time_origin, pose_time_target, timestamps, angular_velocity_imu, node_origin, node_target, self.node_T_B_I, self.node_bias_g, self.node_tau_I, information)
        self._factor_ids["gyro"].append(factor_id)
        self._factor_counts["gyro"] += 1
        return factor_id

    def add_accel_calibration_factor(self, pose_time: float, timestamps: np.ndarray, specific_force_imu: np.ndarray, node_pose: int, information: np.ndarray) -> int:
        factor_id = self._filter_object.add_factor_accel_gravity_calib(pose_time, timestamps, specific_force_imu, self._gravity_world, node_pose, self.node_T_B_I, self.node_tau_I, information)
        self._factor_ids["accel"].append(factor_id)
        self._factor_counts["accel"] += 1
        return factor_id

    def add_lidar_calibration_factor(self, pose_time_origin: float, pose_time_target: float, timestamps: np.ndarray, lidar_odometry_poses: Sequence[Any], node_origin: int, node_target: int, information: np.ndarray) -> int:
        factor_id = self._filter_object.add_factor_lidar_calib_odometry(pose_time_origin, pose_time_target, timestamps, lidar_odometry_poses, node_origin, node_target, self.node_T_B_L, self.node_tau_L, information)
        self._factor_ids["lidar"].append(factor_id)
        self._factor_counts["lidar"] += 1
        return factor_id

    def add_bias_regularization_factor(self, target: Sequence[float] = (0.0, 0.0, 0.0), information: Any = None) -> Optional[int]:
        if self.node_bias_g is None or information is None or self.bias_anchor:
            return None

        factor_id = self._filter_object.add_factor_1_landmark_3d(_as_vector3(target, "bias regularization target"), self.node_bias_g, _information_matrix(information, 3))
        self._factor_ids["bias_prior"].append(factor_id)
        self._factor_counts["bias_prior"] += 1
        return factor_id

    def add_tau_regularization_factor(self, node_id: Optional[int], target: float = 0.0, information: Any = None, family: str = "tau_I_prior") -> Optional[int]:
        if node_id is None or information is None:
            return None

        factor_id = self._filter_object.add_factor_1_scalar_obs(float(target), node_id, _information_matrix(information, 1))
        
        self._factor_ids[family].append(factor_id)
        self._factor_counts[family] += 1
        return factor_id

    def add_pose_regularization_factor(self, node_id: Optional[int], target: Any, information: Any, family: str) -> Optional[int]:
        if node_id is None or information is None:
            return None

        factor_id = self._filter_object.add_factor_1pose_3d(_as_mrob_se3(target), node_id, _information_matrix(information, 6))
        self._factor_ids[family].append(factor_id)
        self._factor_counts[family] += 1
        return factor_id

    def _validate_problem_data(
        self,
        pose_timestamps,
        imu_timestamps,
        angular_velocity_imu,
        specific_force_imu,
        lidar_timestamps,
        lidar_odometry_poses,
    ):
        pose_timestamps = np.asarray(pose_timestamps, dtype=float).reshape(-1)

        if len(pose_timestamps) == 0:
            raise ValueError("At least one trajectory pose is required")

        if np.any(np.diff(pose_timestamps) <= 0):
            raise ValueError("pose_timestamps must be strictly increasing")

        if imu_timestamps is not None:
            imu_timestamps = np.asarray(imu_timestamps, dtype=float).reshape(-1)
            if len(imu_timestamps) < 2 or np.any(np.diff(imu_timestamps) <= 0):
                raise ValueError("imu_timestamps must contain at least two strictly increasing values")
        elif self._include_gyro_factors or self._include_accel_factors:
            raise ValueError("imu_timestamps are required when gyro or accelerometer factors are enabled")

        if angular_velocity_imu is not None:
            angular_velocity_imu = np.asarray(angular_velocity_imu, dtype=float)
            if imu_timestamps is None or angular_velocity_imu.shape != (len(imu_timestamps), 3):
                raise ValueError("angular_velocity_imu must have shape (len(imu_timestamps), 3)")
        elif self._include_gyro_factors:
            raise ValueError("angular_velocity_imu is required when gyro factors are enabled")

        if specific_force_imu is not None:
            specific_force_imu = np.asarray(specific_force_imu, dtype=float)
            if imu_timestamps is None or specific_force_imu.shape != (len(imu_timestamps), 3):
                raise ValueError("specific_force_imu must have shape (len(imu_timestamps), 3)")
        elif self._include_accel_factors:
            raise ValueError("specific_force_imu is required when accelerometer factors are enabled")

        if self._include_lidar_factors:
            if lidar_timestamps is None or lidar_odometry_poses is None:
                raise ValueError("lidar_timestamps and lidar_odometry_poses are required when LiDAR factors are enabled")

            lidar_timestamps = np.asarray(lidar_timestamps, dtype=float).reshape(-1)

            if len(lidar_timestamps) < 2 or np.any(np.diff(lidar_timestamps) <= 0):
                raise ValueError("lidar_timestamps must contain at least two strictly increasing values")

            if len(lidar_odometry_poses) != len(lidar_timestamps):
                raise ValueError("lidar_odometry_poses must have the same length as lidar_timestamps")

        return pose_timestamps, imu_timestamps, angular_velocity_imu, specific_force_imu, lidar_timestamps

    def _initialize_trajectory_poses(
        self,
        pose_timestamps: Sequence[float],
        states: Optional[Sequence[Any]],
        first_pose: Any,
        imu_timestamps: Optional[Sequence[float]],
        angular_velocity_imu: Optional[np.ndarray],
        T_B_I_initial: Any,
        bias_initial: Sequence[float],
        tau_I_initial: float,
        lidar_timestamps: Optional[Sequence[float]] = None,
        lidar_odometry_poses: Optional[Sequence[Any]] = None,
        T_B_L_initial: Any = None,
        tau_L_initial: float = 0.0,
    ) -> np.ndarray:
        """
        Use supplied poses first, then initialize missing rotations from IMU and translations from LiDAR.
        """
        pose_timestamps = np.asarray(pose_timestamps, dtype=float).reshape(-1)
        supplied_states = [] if states is None else list(states)
        initial_poses = [_as_pose_matrix(state) for state in supplied_states[:len(pose_timestamps)]]

        if len(initial_poses) == 0:
            initial_poses.append(_as_pose_matrix(first_pose))

        if len(initial_poses) == len(pose_timestamps):
            return np.asarray(initial_poses)

        if imu_timestamps is None or angular_velocity_imu is None:
            raise ValueError("imu_timestamps and angular_velocity_imu are required to initialize unavailable poses")

        imu_timestamps = np.asarray(imu_timestamps, dtype=float).reshape(-1)
        angular_velocity_imu = np.asarray(angular_velocity_imu, dtype=float)
        bias_initial = _as_vector3(bias_initial, "bias_initial")
        T_B_I_initial = _as_pose_matrix(T_B_I_initial)

        C = T_B_I_initial[:3, :3]
        corrected_angular_velocity = angular_velocity_imu - bias_initial[None, :]

        use_lidar_translation = lidar_timestamps is not None and lidar_odometry_poses is not None

        if use_lidar_translation:
            lidar_timestamps = np.asarray(lidar_timestamps, dtype=float).reshape(-1)
            T_B_L_initial = np.eye(4) if T_B_L_initial is None else _as_pose_matrix(T_B_L_initial)
            T_L_B_initial = np.linalg.inv(T_B_L_initial)

        for pose_index in range(len(initial_poses), len(pose_timestamps)):
            previous_pose = initial_poses[-1]
            next_pose = previous_pose.copy()

            # Initialize rotation using the existing IMU propagation.
            imu_time_origin = float(pose_timestamps[pose_index - 1] + tau_I_initial)
            imu_time_target = float(pose_timestamps[pose_index] + tau_I_initial)
            phi_I = _integrate_vector_interval(imu_timestamps, corrected_angular_velocity, imu_time_origin, imu_time_target)
            delta_rotation_body = C @ mrob.SO3(phi_I).R() @ C.T
            next_pose[:3, :3] = previous_pose[:3, :3] @ delta_rotation_body

            # Initialize only translation from the relative LiDAR odometry motion.
            if use_lidar_translation:
                lidar_time_origin = float(pose_timestamps[pose_index - 1] + tau_L_initial)
                lidar_time_target = float(pose_timestamps[pose_index] + tau_L_initial)

                if lidar_timestamps[0] <= lidar_time_origin and lidar_time_target <= lidar_timestamps[-1]:
                    T_O_L_origin = _interpolate_pose(lidar_timestamps, lidar_odometry_poses, lidar_time_origin)
                    T_O_L_target = _interpolate_pose(lidar_timestamps, lidar_odometry_poses, lidar_time_target)

                    relative_lidar_pose = np.linalg.inv(T_O_L_origin) @ T_O_L_target
                    relative_body_pose = T_B_L_initial @ relative_lidar_pose @ T_L_B_initial

                    next_pose[:3, 3] = previous_pose[:3, 3] + previous_pose[:3, :3] @ relative_body_pose[:3, 3]

            initial_poses.append(next_pose)

        return np.asarray(initial_poses)

    def _pose_anchor(self, pose_index: int, number_poses: int, is_rolling_window: bool = False) -> bool:
        """
        Select anchored trajectory poses for a batch graph or one rolling window.
        """
        if self._anchor_all_poses:
            return True

        anchor_first_pose = self._anchor_first_pose_each_window if is_rolling_window else self._anchor_first_pose

        if anchor_first_pose and pose_index == 0:
            return True

        if self._anchor_last_pose and pose_index == number_poses - 1:
            return True

        return False

    def _accel_measurement_accepted(self, pose_time: float, tau_I_initial: float) -> Tuple[bool, str]:
        query_time = pose_time + tau_I_initial
        specific_force = _interpolate_vector(self._imu_timestamps, self._specific_force_imu, query_time)

        if self._accel_norm_tolerance is not None:
            magnitude_error = abs(np.linalg.norm(specific_force) - np.linalg.norm(self._gravity_world))
            if magnitude_error > self._accel_norm_tolerance:
                return False, f"gravity magnitude error {magnitude_error}"

        if self._accel_gyro_threshold is not None and self._angular_velocity_imu is not None:
            angular_velocity = _interpolate_vector(self._imu_timestamps, self._angular_velocity_imu, query_time)
            if np.linalg.norm(angular_velocity) > self._accel_gyro_threshold:
                return False, f"angular velocity norm {np.linalg.norm(angular_velocity)}"

        return True, ""

    def build_problem(
        self,
        pose_timestamps: Sequence[float],
        states: Optional[Sequence[Any]] = None,
        first_pose: Any = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        imu_timestamps: Optional[Sequence[float]] = None,
        angular_velocity_imu: Optional[np.ndarray] = None,
        specific_force_imu: Optional[np.ndarray] = None,
        lidar_timestamps: Optional[Sequence[float]] = None,
        lidar_odometry_poses: Optional[Sequence[Any]] = None,
        T_B_I_initial: Any = None,
        T_B_L_initial: Any = None,
        bias_initial: Sequence[float] = (0.0, 0.0, 0.0),
        tau_I_initial: float = 0.0,
        tau_L_initial: float = 0.0,
        bias_regularization_target: Sequence[float] = (0.0, 0.0, 0.0),
        tau_I_regularization_target: float = 0.0,
        tau_L_regularization_target: float = 0.0,
        is_rolling_window: bool = False,
    ):
        """
        Build one complete batch or rolling-window calibration problem.
        """
        self.reset(clear_rolling_state=False)

        # Validate and cache the trajectory and sensor data used by this graph.
        validated = self._validate_problem_data(pose_timestamps, imu_timestamps, angular_velocity_imu, specific_force_imu, lidar_timestamps, lidar_odometry_poses)
        self._pose_timestamps, self._imu_timestamps, self._angular_velocity_imu, self._specific_force_imu, self._lidar_timestamps = validated
        self._lidar_odometry_poses = None if lidar_odometry_poses is None else list(lidar_odometry_poses)

        T_B_I_initial = np.eye(4) if T_B_I_initial is None else _as_pose_matrix(T_B_I_initial)
        T_B_L_initial = np.eye(4) if T_B_L_initial is None else _as_pose_matrix(T_B_L_initial)
        bias_initial = _as_vector3(bias_initial, "bias_initial")
        tau_I_initial = float(tau_I_initial)
        tau_L_initial = float(tau_L_initial)
        self._initial_poses = self._initialize_trajectory_poses(pose_timestamps=self._pose_timestamps, states=states, first_pose=first_pose, imu_timestamps=self._imu_timestamps,
            angular_velocity_imu=self._angular_velocity_imu, T_B_I_initial=T_B_I_initial, bias_initial=bias_initial, tau_I_initial=tau_I_initial,
            lidar_timestamps=self._lidar_timestamps, lidar_odometry_poses=self._lidar_odometry_poses, T_B_L_initial=T_B_L_initial, tau_L_initial=tau_L_initial)

        # Add trajectory nodes and preserve their logical order separately from MROB node IDs.
        for pose_index, pose in enumerate(self._initial_poses):
            node_id = self.add_pose_node(pose, anchor=self._pose_anchor(pose_index, len(self._initial_poses), is_rolling_window))
            self._nodes_pose.append(node_id)

        # Add only the calibration nodes required by the enabled factor families.
        if self._include_gyro_factors or self._include_accel_factors:
            self._node_T_B_I = self.add_pose_node(T_B_I_initial, anchor=self._T_B_I_anchor)
            self._node_tau_I = self.add_scalar_node(tau_I_initial, anchor=self._tau_I_anchor)

        if self._include_gyro_factors:
            self._node_bias_g = self.add_vector_node(bias_initial, anchor=self._bias_anchor)

        if self._include_lidar_factors:
            self._node_T_B_L = self.add_pose_node(T_B_L_initial, anchor=self._T_B_L_anchor)
            self._node_tau_L = self.add_scalar_node(tau_L_initial, anchor=self._tau_L_anchor)

        # Add optional priors. Bias regularization is available in the current bindings. Soft scalar priors require a scalar-prior binding or callback.
        self.add_bias_regularization_factor(target=bias_regularization_target, information=self._bias_regularization_information)
        if not self._tau_I_anchor:
            self.add_tau_regularization_factor(self.node_tau_I, target=tau_I_regularization_target, information=self._tau_I_regularization_information, family="tau_I_prior")
        if not self._tau_L_anchor:
            self.add_tau_regularization_factor(self.node_tau_L, target=tau_L_regularization_target, information=self._tau_L_regularization_information, family="tau_L_prior")
        if not self._T_B_I_anchor:
            self.add_pose_regularization_factor(self.node_T_B_I, T_B_I_initial, self._T_B_I_regularization_information, "T_B_I_prior")
        if not self._T_B_L_anchor:
            self.add_pose_regularization_factor(self.node_T_B_L, T_B_L_initial, self._T_B_L_regularization_information, "T_B_L_prior")

        # Add gyroscope propagation factors between selected consecutive trajectory poses.
        if self._include_gyro_factors:
            for factor_index, pose_index in enumerate(range(0, len(self.nodes_pose) - 1, self._gyro_factor_stride)):
                target_index = pose_index + self._gyro_factor_stride
                if target_index >= len(self.nodes_pose):
                    break

                pose_time_origin = float(self._pose_timestamps[pose_index])
                pose_time_target = float(self._pose_timestamps[target_index])
                support_indices = select_time_support_indices(self._imu_timestamps, pose_time_origin - self._imu_time_offset_margin, pose_time_target + self._imu_time_offset_margin, self._imu_samples_per_factor)
                timestamps = self._imu_timestamps[support_indices]
                measurements = self._angular_velocity_imu[support_indices]
                information = _information_matrix(self._gyro_information, 3, factor_index)

                factor_id = self.add_gyro_calibration_factor(pose_time_origin, pose_time_target, timestamps, measurements, self.nodes_pose[pose_index], self.nodes_pose[target_index], information)
                self._factor_metadata["gyro"].append({"factor_id": factor_id, "pose_indices": (pose_index, target_index), "measurement_indices": support_indices})

        # Add low-dynamic accelerometer factors to individual trajectory poses.
        if self._include_accel_factors:
            for factor_index, pose_index in enumerate(range(0, len(self.nodes_pose), self._accel_factor_stride)):
                pose_time = float(self._pose_timestamps[pose_index])
                accepted, rejection_reason = self._accel_measurement_accepted(pose_time, tau_I_initial)

                if not accepted:
                    self._factor_metadata["accel"].append({"factor_id": None, "pose_index": pose_index, "accepted": False, "reason": rejection_reason})
                    continue

                support_indices = select_time_support_indices(self._imu_timestamps, pose_time - self._imu_time_offset_margin, pose_time + self._imu_time_offset_margin, self._imu_samples_per_factor)
                timestamps = self._imu_timestamps[support_indices]
                measurements = self._specific_force_imu[support_indices]
                information = _information_matrix(self._accel_information, 3, factor_index)

                factor_id = self.add_accel_calibration_factor(pose_time, timestamps, measurements, self.nodes_pose[pose_index], information)
                self._factor_metadata["accel"].append({"factor_id": factor_id, "pose_index": pose_index, "accepted": True, "measurement_indices": support_indices})

        # Add LiDAR relative-pose factors between selected consecutive trajectory poses.
        if self._include_lidar_factors:
            for factor_index, pose_index in enumerate(range(0, len(self.nodes_pose) - 1, self._lidar_factor_stride)):
                target_index = pose_index + self._lidar_factor_stride
                if target_index >= len(self.nodes_pose):
                    break

                pose_time_origin = float(self._pose_timestamps[pose_index])
                pose_time_target = float(self._pose_timestamps[target_index])
                support_indices = select_time_support_indices(self._lidar_timestamps, pose_time_origin - self._lidar_time_offset_margin, pose_time_target + self._lidar_time_offset_margin, self._lidar_samples_per_factor)
                timestamps = self._lidar_timestamps[support_indices]
                measurements = [self._lidar_odometry_poses[index] for index in support_indices]
                information = _information_matrix(self._lidar_information, 6, factor_index)

                factor_id = self.add_lidar_calibration_factor(pose_time_origin, pose_time_target, timestamps, measurements, self.nodes_pose[pose_index], self.nodes_pose[target_index], information)
                self._factor_metadata["lidar"].append({"factor_id": factor_id, "pose_indices": (pose_index, target_index), "measurement_indices": support_indices})

        # Cache the initialized graph state only after every node and factor has been added.
        self._invalidate_state_cache()
        self._states_init = [np.asarray(value, dtype=float).copy() for value in self.states]
        return self

    def solve_problem(
        self,
        solver_verbose: Optional[bool] = None,
        solutionTolerance: Optional[float] = None,
        maxIters: Optional[int] = None,
        lambdaParam: Optional[float] = None,
        scheduler: Optional[Sequence[Tuple[float, int]]] = None,
        method: Optional[str] = None,
    ):
        solver_verbose = self.solver_verbose if solver_verbose is None else bool(solver_verbose)
        solutionTolerance = self.solutionTolerance if solutionTolerance is None else float(solutionTolerance)
        maxIters = self.maxIters if maxIters is None else int(maxIters)
        lambdaParam = self.lambdaParam if lambdaParam is None else float(lambdaParam)
        method = self.method if method is None else str(method).upper()
        scheduler = self.scheduler if scheduler is None else [(float(value), int(iterations)) for value, iterations in scheduler]

        self._chi2_prev = self.chi2
        self._states_init = [np.asarray(value, dtype=float).copy() for value in self.states]

        mrob_method = mrob.GN if method == "GN" else mrob.LM
        solve_scheduler = [(lambdaParam, maxIters)] if scheduler is None else scheduler

        for current_lambda, current_iterations in solve_scheduler:
            self._filter_object.solve(method=mrob_method, verbose=solver_verbose, solutionTolerance=solutionTolerance, maxIters=current_iterations, lambdaParam=current_lambda)

        self._invalidate_state_cache()
        self._chi2 = self.chi2
        gc.collect()
        return self.states

    def _create_window_result(self, window_index: int, window_start: float, window_end: float, chi2_before: float) -> CalibrationWindowResult:
        return CalibrationWindowResult(
            window_index=window_index,
            window_start=float(window_start),
            window_end=float(window_end),
            pose_timestamps=self._pose_timestamps.copy(),
            trajectory_poses=self.trajectory_poses.copy(),
            T_B_I=None if self.T_B_I is None else self.T_B_I.copy(),
            T_B_L=None if self.T_B_L is None else self.T_B_L.copy(),
            bias_g=None if self.bias_g is None else self.bias_g.copy(),
            tau_I=self.tau_I,
            tau_L=self.tau_L,
            chi2_before=float(chi2_before),
            chi2_after=float(self.chi2),
            factor_counts=self.factor_counts,
        )

    def generate_filter(self, verbose: int = 0, **build_kwargs) -> CalibrationWindowResult:
        self.build_problem(**build_kwargs)

        if verbose > 0:
            print("RAW:")
            self.print_problem(complete=verbose > 1)

        chi2_before = self.chi2
        self.solve_problem()

        if verbose > 0:
            print("FILTERED:")
            self.print_problem(complete=verbose > 1)
            self.print_update()

        result = self._create_window_result(0, self._pose_timestamps[0], self._pose_timestamps[-1], chi2_before)
        self._last_window_result = result
        return result

    def _time_key(self, timestamp: float) -> float:
        return round(float(timestamp), 9)

    def _rolling_state_prefix(self, pose_timestamps: Sequence[float]) -> List[np.ndarray]:
        """
        Return the consecutive solved-state prefix available from the previous window.
        """
        rolling_states = []

        for timestamp in pose_timestamps:
            key = self._time_key(timestamp)

            if key not in self._rolling_pose_cache:
                break

            rolling_states.append(self._rolling_pose_cache[key].copy())

        return rolling_states

    def _initialize_window_poses(
        self,
        pose_timestamps: np.ndarray,
        window_pose_indices: np.ndarray,
        states: Optional[Sequence[Any]],
        first_pose: Any,
        imu_timestamps: Optional[Sequence[float]],
        angular_velocity_imu: Optional[np.ndarray],
        T_B_I_initial: Any,
        bias_initial: Sequence[float],
        tau_I_initial: float,
        lidar_timestamps: Optional[Sequence[float]],
        lidar_odometry_poses: Optional[Sequence[Any]],
        T_B_L_initial: Any,
        tau_L_initial: float,
    ) -> np.ndarray:
        """
        Initialize a window from rolling states, supplied global states, or propagated sensor motion.
        """
        window_pose_timestamps = pose_timestamps[window_pose_indices]
        rolling_states = self._rolling_state_prefix(window_pose_timestamps)

        if len(rolling_states) > 0:
            window_states = rolling_states
        else:
            supplied_states = [] if states is None else list(states)
            window_states = [supplied_states[index] for index in window_pose_indices if index < len(supplied_states)]

            # The first supplied state may precede the window. Propagate it to the
            # first window timestamp before initializing the remainder of the window.
            if len(window_states) == 0:
                first_window_index = int(window_pose_indices[0])
                prefix_timestamps = pose_timestamps[:first_window_index + 1]
                prefix_poses = self._initialize_trajectory_poses(
                    pose_timestamps=prefix_timestamps,
                    states=supplied_states,
                    first_pose=first_pose,
                    imu_timestamps=imu_timestamps,
                    angular_velocity_imu=angular_velocity_imu,
                    T_B_I_initial=T_B_I_initial,
                    bias_initial=bias_initial,
                    tau_I_initial=tau_I_initial,
                    lidar_timestamps=lidar_timestamps,
                    lidar_odometry_poses=lidar_odometry_poses,
                    T_B_L_initial=T_B_L_initial,
                    tau_L_initial=tau_L_initial,
                )
                window_states = [prefix_poses[-1]]

        return self._initialize_trajectory_poses(
            pose_timestamps=window_pose_timestamps,
            states=window_states,
            first_pose=window_states[0],
            imu_timestamps=imu_timestamps,
            angular_velocity_imu=angular_velocity_imu,
            T_B_I_initial=T_B_I_initial,
            bias_initial=bias_initial,
            tau_I_initial=tau_I_initial,
            lidar_timestamps=lidar_timestamps,
            lidar_odometry_poses=lidar_odometry_poses,
            T_B_L_initial=T_B_L_initial,
            tau_L_initial=tau_L_initial,
        )

    def _rolling_calibration_initial(self, name: str, default: Any):
        if name not in self._rolling_calibration_state:
            return default

        value = self._rolling_calibration_state[name]
        return value.copy() if isinstance(value, np.ndarray) else value

    def _store_rolling_solution(self, result: CalibrationWindowResult):
        for timestamp, pose in zip(result.pose_timestamps, result.trajectory_poses):
            self._rolling_pose_cache[self._time_key(timestamp)] = pose.copy()

        if result.T_B_I is not None:
            self._rolling_calibration_state["T_B_I"] = result.T_B_I.copy()
        if result.T_B_L is not None:
            self._rolling_calibration_state["T_B_L"] = result.T_B_L.copy()
        if result.bias_g is not None:
            self._rolling_calibration_state["bias_g"] = result.bias_g.copy()
        if result.tau_I is not None:
            self._rolling_calibration_state["tau_I"] = float(result.tau_I)
        if result.tau_L is not None:
            self._rolling_calibration_state["tau_L"] = float(result.tau_L)

    def _commit_rolling_output(self, result: CalibrationWindowResult, commit_end: float, include_end: bool):
        for timestamp, pose in zip(result.pose_timestamps, result.trajectory_poses):
            should_commit = timestamp <= commit_end if include_end else timestamp < commit_end
            if should_commit:
                key = self._time_key(timestamp)
                self._rolling_output_pose_cache[key] = pose.copy()
                self._rolling_output_pose_times[key] = float(timestamp)

    def generate_filter_window(
        self,
        window_index: int,
        window_start: float,
        window_end: float,
        pose_timestamps: Sequence[float],
        states: Optional[Sequence[Any]] = None,
        first_pose: Any = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        imu_timestamps: Optional[Sequence[float]] = None,
        angular_velocity_imu: Optional[np.ndarray] = None,
        specific_force_imu: Optional[np.ndarray] = None,
        lidar_timestamps: Optional[Sequence[float]] = None,
        lidar_odometry_poses: Optional[Sequence[Any]] = None,
        T_B_I_initial: Any = None,
        T_B_L_initial: Any = None,
        bias_initial: Sequence[float] = (0.0, 0.0, 0.0),
        tau_I_initial: float = 0.0,
        tau_L_initial: float = 0.0,
        verbose: int = 0,
    ) -> CalibrationWindowResult:
        """
        Build and solve one rolling window while warm-starting all overlapping states.
        """
        pose_timestamps = np.asarray(pose_timestamps, dtype=float).reshape(-1)
        window_pose_indices = np.flatnonzero((pose_timestamps >= window_start) & (pose_timestamps <= window_end))

        if len(window_pose_indices) == 0:
            raise ValueError(f"No trajectory poses lie inside rolling window [{window_start}, {window_end}]")

        T_B_I_initial = self._rolling_calibration_initial("T_B_I", np.eye(4) if T_B_I_initial is None else T_B_I_initial)
        T_B_L_initial = self._rolling_calibration_initial("T_B_L", np.eye(4) if T_B_L_initial is None else T_B_L_initial)
        bias_initial = self._rolling_calibration_initial("bias_g", _as_vector3(bias_initial, "bias_initial"))
        tau_I_initial = self._rolling_calibration_initial("tau_I", float(tau_I_initial))
        tau_L_initial = self._rolling_calibration_initial("tau_L", float(tau_L_initial))

        # Initialize the window from a consecutive solved prefix when available.
        # Otherwise propagate supplied global states up to the first window pose.
        window_pose_timestamps = pose_timestamps[window_pose_indices]
        window_initial_poses = self._initialize_window_poses(
            pose_timestamps=pose_timestamps,
            window_pose_indices=window_pose_indices,
            states=states,
            first_pose=first_pose,
            imu_timestamps=imu_timestamps,
            angular_velocity_imu=angular_velocity_imu,
            T_B_I_initial=T_B_I_initial,
            bias_initial=bias_initial,
            tau_I_initial=tau_I_initial,
            lidar_timestamps=lidar_timestamps,
            lidar_odometry_poses=lidar_odometry_poses,
            T_B_L_initial=T_B_L_initial,
            tau_L_initial=tau_L_initial,
        )

        # Build the current window from global sensor streams so factors retain their complete temporal support.
        self.build_problem(
            pose_timestamps=window_pose_timestamps,
            states=window_initial_poses,
            first_pose=first_pose,
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
            is_rolling_window=True,
        )

        if verbose > 0:
            print(f"WINDOW {window_index}, RAW [{window_start}, {window_end}]:")
            self.print_problem(complete=verbose > 1)

        # Solve the current window and cache its solution for the next shifted window.
        chi2_before = self.chi2
        self.solve_problem()
        result = self._create_window_result(window_index, window_start, window_end, chi2_before)
        self._store_rolling_solution(result)
        self._rolling_results.append(result)
        self._last_window_result = result

        if verbose > 0:
            print(f"WINDOW {window_index}, FILTERED [{window_start}, {window_end}]:")
            self.print_problem(complete=verbose > 1)
            self.print_update()

        return result

    def generate_filter_iterative(
        self,
        window_size: float,
        step_size: float,
        pose_timestamps: Sequence[float],
        states: Optional[Sequence[Any]] = None,
        first_pose: Any = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        imu_timestamps: Optional[Sequence[float]] = None,
        angular_velocity_imu: Optional[np.ndarray] = None,
        specific_force_imu: Optional[np.ndarray] = None,
        lidar_timestamps: Optional[Sequence[float]] = None,
        lidar_odometry_poses: Optional[Sequence[Any]] = None,
        T_B_I_initial: Any = None,
        T_B_L_initial: Any = None,
        bias_initial: Sequence[float] = (0.0, 0.0, 0.0),
        tau_I_initial: float = 0.0,
        tau_L_initial: float = 0.0,
        clear_previous: bool = True,
        verbose: int = 0,
    ) -> List[CalibrationWindowResult]:
        """
        Solve overlapping calibration windows and carry solved states into each next window.
        """
        if window_size <= 0 or step_size <= 0:
            raise ValueError("window_size and step_size must be positive")

        if step_size > window_size:
            raise ValueError("step_size should not exceed window_size because rolling windows would not overlap")

        pose_timestamps = np.asarray(pose_timestamps, dtype=float).reshape(-1)

        if len(pose_timestamps) == 0:
            raise ValueError("pose_timestamps must not be empty")

        if clear_previous:
            self.clear_rolling_state()

        # Initialize the complete fallback trajectory once. Rolling solutions still override overlapping poses in every window.

        # Generate window boundaries with enough initial sensor support for temporal offsets.
        trajectory_start = float(pose_timestamps[0])
        trajectory_end = float(pose_timestamps[-1])

        safe_start_candidates = [trajectory_start]
        safe_end_candidates = [trajectory_end]

        if self._include_gyro_factors or self._include_accel_factors:
            if imu_timestamps is None:
                raise ValueError("imu_timestamps are required to determine the first valid rolling window")
            safe_start_candidates.append(float(np.asarray(imu_timestamps)[0]) + self._imu_time_offset_margin)
            safe_end_candidates.append(float(np.asarray(imu_timestamps)[-1]) - self._imu_time_offset_margin)

        if self._include_lidar_factors:
            if lidar_timestamps is None:
                raise ValueError("lidar_timestamps are required to determine the first valid rolling window")
            safe_start_candidates.append(float(np.asarray(lidar_timestamps)[0]) + self._lidar_time_offset_margin)
            safe_end_candidates.append(float(np.asarray(lidar_timestamps)[-1]) - self._lidar_time_offset_margin)

        required_start = max(safe_start_candidates)
        required_end = min(safe_end_candidates)

        if required_end <= required_start:
            raise ValueError(f"No valid rolling interval remains inside sensor support [{required_start}, {required_end}]")

        window_starts = []
        current_start = required_start

        while current_start < required_end:
            window_starts.append(current_start)
            current_start += step_size

        # Build, solve, warm-start, and commit the non-overlapping output segment of every window.
        for window_index, window_start in enumerate(window_starts):
            window_end = min(window_start + window_size, required_end)
            window_pose_indices = np.flatnonzero((pose_timestamps >= window_start) & (pose_timestamps <= window_end))

            if len(window_pose_indices) == 0:
                continue

            result = self.generate_filter_window(
                window_index=window_index,
                window_start=window_start,
                window_end=window_end,
                pose_timestamps=pose_timestamps,
                states=states,
                first_pose=first_pose,
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

            is_last_window = window_end >= required_end
            commit_end = required_end if is_last_window else window_start + step_size
            self._commit_rolling_output(result, commit_end=commit_end, include_end=is_last_window)

            if is_last_window:
                break

        return self.rolling_results

    def print_problem(self, complete: bool = False, pose_count: int = 20):
        print(f"Chi2 error = {self.chi2}")
        print(f"Nodes = {self._filter_object.number_nodes()}, factors = {self._filter_object.number_factors()}")
        print(f"Factor counts = {self.factor_counts}")

        if len(self.nodes_pose) > 0:
            pose_step = int(len(self.nodes_pose) / pose_count)
            for pose_index in range(0, len(self.nodes_pose), max(pose_step, 1)):
                print(f"pose[{pose_index}] at t={self._pose_timestamps[pose_index]}: {mrob.SE3(self.trajectory_poses[pose_index]).Ln()}")

        if self.T_B_I is not None:
            print(f"T_B_I:\n{mrob.SE3(self.T_B_I).Ln()}")

        if self.bias_g is not None:
            print(f"bias_g = {self.bias_g}")

        if self.tau_I is not None:
            print(f"tau_I = {self.tau_I}")

        if self.T_B_L is not None:
            print(f"T_B_L:\n{mrob.SE3(self.T_B_L).Ln()}")

        if self.tau_L is not None:
            print(f"tau_L = {self.tau_L}")

        # if complete:
        #     self._filter_object.print(True)

    def print_update(self, pose_count: int = 20):
        if self.states_init is None:
            print("No initial graph state is cached")
            return

        print(f"Chi2 error decreased by {self.chi2_prev - self.chi2}, from {self.chi2_prev} to {self.chi2}")

        pose_step = int(len(self.nodes_pose) / pose_count)
        for pose_index in range(0, len(self.nodes_pose), max(pose_step, 1)):
            node_id = self.nodes_pose[pose_index]
            initial_pose = np.asarray(self.states_init[node_id], dtype=float)
            current_pose = np.asarray(self.states[node_id], dtype=float)
            update = mrob.SE3(initial_pose).inv().mul(mrob.SE3(current_pose)).Ln()
            print(f"pose[{pose_index}] update = {update}")

        if self.node_T_B_I is not None:
            initial = np.asarray(self.states_init[self.node_T_B_I], dtype=float)
            update = mrob.SE3(initial).inv().mul(mrob.SE3(self.T_B_I)).Ln()
            print(f"T_B_I update = {update}")

        if self.node_bias_g is not None:
            initial = np.asarray(self.states_init[self.node_bias_g], dtype=float).reshape(3)
            print(f"bias_g update = {self.bias_g - initial}")

        if self.node_tau_I is not None:
            initial = float(np.asarray(self.states_init[self.node_tau_I], dtype=float).reshape(-1)[0])
            print(f"tau_I update = {self.tau_I - initial}")

        if self.node_T_B_L is not None:
            initial = np.asarray(self.states_init[self.node_T_B_L], dtype=float)
            update = mrob.SE3(initial).inv().mul(mrob.SE3(self.T_B_L)).Ln()
            print(f"T_B_L update = {update}")

        if self.node_tau_L is not None:
            initial = float(np.asarray(self.states_init[self.node_tau_L], dtype=float).reshape(-1)[0])
            print(f"tau_L update = {self.tau_L - initial}")

    def print_rolling_results(self):
        for result in self.rolling_results:
            print(
                f"window={result.window_index}, interval=[{result.window_start}, {result.window_end}], "
                f"poses={len(result.pose_timestamps)}, chi2={result.chi2_before} -> {result.chi2_after}, "
                f"bias={result.bias_g}, tau_I={result.tau_I}, tau_L={result.tau_L}"
            )