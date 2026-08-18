"""Rolling factor-graph orchestration for modular multi-sensor calibration.

This file owns graph construction order, source resolution for calibration values, solver execution, result extraction, and rolling-window state updates. It deliberately does not own gyro, accelerometer, LiDAR, or radar measurement mathematics; streams add their own factors through a small ``StreamContext`` interface.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import mrob
import numpy as np

try:
    import data_processing
    from numerical_calibration import NumericalCalibrationConfig, estimate_imu_calibration_numerical
except ImportError:  # pragma: no cover
    from src import data_processing
    from src.numerical_calibration import NumericalCalibrationConfig, estimate_imu_calibration_numerical

from .metadata import CalibrationVariableMetadata, GraphMetadata, PriorMetadata, StreamMetadata, TrajectoryNodeMetadata
from .rolling_state import RollingState
from .sensors import Sensor, ensure_sensor
from .streams import MeasurementStream, StreamContext
from .variables import (
    ValueSource,
    VariableConfig,
    VariableKey,
    VariableNode,
    VariableRequirement,
    VariableType,
    add_variable_node,
    add_variable_prior,
    coerce_variable_value,
    copy_variable_value,
    default_value_for,
    normalize_value_source,
    normalize_variable_config_map,
)


@dataclass
class SolverConfig:
    """mrob solver settings kept at graph level."""

    method: str = "LM"
    maxIters: int = 30
    lambdaParam: float = 1e-5
    solutionTolerance: float = 1e-6
    solver_verbose: bool = False
    scheduler: Sequence[tuple[float, int]] | None = None

    def normalized(self) -> "SolverConfig":
        scheduler = None if self.scheduler is None else [(float(value), int(iterations)) for value, iterations in self.scheduler]
        return SolverConfig(method=str(self.method).upper(), maxIters=int(self.maxIters), lambdaParam=float(self.lambdaParam), solutionTolerance=float(self.solutionTolerance), solver_verbose=bool(self.solver_verbose), scheduler=scheduler)


@dataclass
class TrajectoryConfig:
    """Trajectory-node anchoring and initialization policy."""

    anchor_first_pose: bool = True
    anchor_first_pose_each_window: bool = False
    anchor_last_pose: bool = False
    anchor_all_poses: bool = False
    use_imu_gyr: bool = False


@dataclass
class ResolvedValue:
    """One resolved initial or prior value, including requested and effective sources."""

    value: Any
    requested_source: ValueSource
    effective_source: ValueSource


@dataclass
class WindowResult:
    """Result of one batch graph or rolling window."""

    window_index: int
    window_start: float
    window_end: float
    pose_timestamps: np.ndarray
    trajectory_poses: np.ndarray
    calibration_values: dict[VariableKey, Any]
    chi2_before: float
    chi2_after: float
    factor_counts: dict[str, int]
    metadata: GraphMetadata
    numerical_results: dict[str, Any] = field(default_factory=dict)

    def calibration_value(self, key: VariableKey | tuple[str, str]) -> Any | None:
        """Return a copy of one calibration value from this result."""

        key = key if isinstance(key, VariableKey) else VariableKey(key[0], key[1])
        if key not in self.calibration_values:
            return None
        return copy_variable_value(self.calibration_values[key])


class _CalibrationValueResolver:
    """Resolve initial and prior values while caching numerical calibration per IMU sensor."""

    def __init__(self, rolling_graph: "RollingGraph", *, window_start: float, window_end: float):
        self.rolling_graph = rolling_graph
        self.window_start = float(window_start)
        self.window_end = float(window_end)
        self.numerical_results: dict[str, Any] = {}

    def resolve_initial(self, key: VariableKey, config: VariableConfig) -> ResolvedValue:
        return self._resolve(key, normalize_value_source(config.initial_source), config.initial_value, role="initial")

    def resolve_prior(self, key: VariableKey, config: VariableConfig) -> ResolvedValue:
        source = normalize_value_source(config.prior_source, allow_none=True)
        if source is None:
            raise ValueError(f"{key.label} has no prior_source")
        return self._resolve(key, source, config.prior_value, role="prior")

    def _resolve(self, key: VariableKey, source: ValueSource, configured_value: Any, *, role: str) -> ResolvedValue:
        # Resolve numerical values once per IMU sensor and reuse them for both initial values and soft priors.
        if source == ValueSource.NUMERICAL:
            result = self._numerical_result_for_imu(key.sensor_id)
            value = self._value_from_numerical_result(key, result)
            if result.success and value is not None:
                return ResolvedValue(coerce_variable_value(key, value, role=role), requested_source=source, effective_source=ValueSource.NUMERICAL)
            fallback_value, fallback_source = self._fallback_value(key, configured_value, role=role)
            return ResolvedValue(coerce_variable_value(key, fallback_value, role=role), requested_source=source, effective_source=fallback_source)

        value, effective_source = self._non_numerical_value(key, source, configured_value, role=role)
        return ResolvedValue(coerce_variable_value(key, value, role=role), requested_source=source, effective_source=effective_source)

    def _fallback_value(self, key: VariableKey, configured_value: Any, *, role: str) -> tuple[Any, ValueSource]:
        # Numerical estimation can fail from insufficient excitation; fall back explicitly rather than pretending the failed numerical result is a prior.
        optimized = self.rolling_graph.rolling_state.get_calibration(key)
        if optimized is not None:
            return optimized, ValueSource.OPTIMIZED
        if configured_value is not None:
            return configured_value, ValueSource.CONSTANT
        return default_value_for(key.variable_type), ValueSource.DEFAULT

    def _non_numerical_value(self, key: VariableKey, source: ValueSource, configured_value: Any, *, role: str) -> tuple[Any, ValueSource]:
        if source == ValueSource.CONSTANT:
            return configured_value, ValueSource.CONSTANT
        if source == ValueSource.OPTIMIZED:
            optimized = self.rolling_graph.rolling_state.get_calibration(key)
            if optimized is not None:
                return optimized, ValueSource.OPTIMIZED
            if configured_value is not None:
                return configured_value, ValueSource.CONSTANT
            return default_value_for(key.variable_type), ValueSource.DEFAULT
        if source == ValueSource.DEFAULT:
            return default_value_for(key.variable_type), ValueSource.DEFAULT
        raise ValueError(f"{key.label} {role} cannot be resolved from source {source}")

    def _numerical_result_for_imu(self, imu_sensor_id: str) -> Any:
        if imu_sensor_id in self.numerical_results:
            return self.numerical_results[imu_sensor_id]

        imu_data = self.rolling_graph._numerical_imu_data(imu_sensor_id)
        lidar_data = self.rolling_graph._numerical_lidar_data()
        if imu_data is None:
            raise ValueError(f"numerical calibration requested for {imu_sensor_id}, but no GyroStream for that sensor is available")
        if lidar_data is None:
            raise ValueError(f"numerical calibration requested for {imu_sensor_id}, but no LidarOdometryStream is available")

        # Numerical IMU calibration needs current estimates for the LiDAR extrinsic, gyro bias, and prior IMU translation. These are resolved without using numerical again to avoid source cycles.
        lidar_sensor_id = lidar_data["sensor_id"]
        T_B_L_key = VariableKey(lidar_sensor_id, VariableType.EXTRINSIC)
        bias_key = VariableKey(imu_sensor_id, VariableType.GYRO_BIAS)
        T_B_I_key = VariableKey(imu_sensor_id, VariableType.EXTRINSIC)
        T_B_L_config = self.rolling_graph._config_for(T_B_L_key)
        bias_config = self.rolling_graph._config_for(bias_key)
        T_B_I_config = self.rolling_graph._config_for(T_B_I_key)
        T_B_L, _ = self._non_numerical_value(T_B_L_key, ValueSource.OPTIMIZED, T_B_L_config.initial_value, role="numerical T_B_L")
        bias_g, _ = self._non_numerical_value(bias_key, ValueSource.OPTIMIZED, bias_config.initial_value, role="numerical bias_g")
        T_B_I_previous, _ = self._non_numerical_value(T_B_I_key, ValueSource.OPTIMIZED, T_B_I_config.initial_value, role="numerical T_B_I_previous")

        result = estimate_imu_calibration_numerical(
            window_start=self.window_start,
            window_end=self.window_end,
            imu_timestamps=imu_data["imu_timestamps"],
            angular_velocity_imu=imu_data["angular_velocity_imu"],
            lidar_timestamps=lidar_data["lidar_timestamps"],
            lidar_odometry_poses=lidar_data["lidar_odometry_poses"],
            T_B_L=coerce_variable_value(T_B_L_key, T_B_L, role="numerical T_B_L"),
            bias_g=coerce_variable_value(bias_key, bias_g, role="numerical bias_g"),
            bias_mode="provided",
            T_B_I_previous=coerce_variable_value(T_B_I_key, T_B_I_previous, role="numerical T_B_I_previous"),
            config=self.rolling_graph.numerical_calibration_config,
        )
        self.numerical_results[imu_sensor_id] = result
        return result

    @staticmethod
    def _value_from_numerical_result(key: VariableKey, result: Any) -> Any | None:
        if not result.success:
            return None
        if key.variable_type == VariableType.EXTRINSIC:
            return result.T_B_I
        if key.variable_type == VariableType.TIME_OFFSET:
            return result.tau_I
        if key.variable_type == VariableType.GYRO_BIAS:
            return result.bias_g_used
        raise AssertionError(f"Unhandled variable type {key.variable_type}")


class RollingGraph:
    """Build, solve, and roll modular mrob calibration graphs from measurement streams."""

    def __init__(
        self,
        *,
        streams: Sequence[MeasurementStream],
        sensors: Sequence[Sensor | str] | None = None,
        variable_configs: Mapping[Any, VariableConfig | Mapping[str, Any]] | None = None,
        solver_config: SolverConfig | None = None,
        trajectory_config: TrajectoryConfig | None = None,
        rolling_state: RollingState | None = None,
        numerical_calibration_config: NumericalCalibrationConfig | None = None,
    ):
        self.streams = list(streams)
        self._sensors_explicitly_configured = sensors is not None
        self.sensors = self._normalize_sensors(sensors)
        self._validate_stream_sensors()
        self.variable_configs = normalize_variable_config_map(variable_configs)
        self.solver_config = (SolverConfig() if solver_config is None else solver_config).normalized()
        self.trajectory_config = TrajectoryConfig() if trajectory_config is None else trajectory_config
        self.rolling_state = RollingState() if rolling_state is None else rolling_state
        self.numerical_calibration_config = NumericalCalibrationConfig() if numerical_calibration_config is None else numerical_calibration_config
        self._last_result: WindowResult | None = None
        self._reset_graph()

    def _normalize_sensors(self, sensors: Sequence[Sensor | str] | None) -> dict[str, Sensor]:
        normalized: dict[str, Sensor] = {}
        for sensor in [] if sensors is None else sensors:
            sensor = ensure_sensor(sensor)
            if sensor.sensor_id in normalized:
                raise ValueError(f"Duplicate sensor id {sensor.sensor_id!r}")
            normalized[sensor.sensor_id] = sensor
        return normalized

    def _validate_stream_sensors(self) -> None:
        seen_stream_names: set[str] = set()
        for stream in self.streams:
            if not hasattr(stream, "sensor"):
                raise ValueError(f"Stream {stream!r} does not expose a physical sensor")
            sensor = ensure_sensor(stream.sensor)
            stream.sensor = sensor
            if self._sensors_explicitly_configured and sensor.sensor_id not in self.sensors:
                raise ValueError(f"Stream {stream.stream_name!r} references unknown sensor {sensor.sensor_id!r}")
            self.sensors.setdefault(sensor.sensor_id, sensor)
            if stream.stream_name in seen_stream_names:
                raise ValueError(f"Duplicate stream_name {stream.stream_name!r}")
            seen_stream_names.add(stream.stream_name)

    def _reset_graph(self) -> None:
        old_graph = getattr(self, "_filter_object", None)
        self._filter_object = None
        if old_graph is not None:
            del old_graph
            gc.collect()
        self._filter_object = mrob.FGraph()
        self._nodes: list[int] = []
        self._pose_nodes: list[int] = []
        self._variable_nodes: dict[VariableKey, int] = {}
        self._variable_node_records: dict[VariableKey, VariableNode] = {}
        self._metadata = GraphMetadata()
        self._states_cache = None
        self._trajectory_poses_cache = None
        self._states_init = None
        self._chi2_prev = 0.0
        self._chi2 = 0.0

    @property
    def filter_object(self) -> mrob.FGraph:
        return self._filter_object

    @property
    def nodes(self) -> list[int]:
        return list(self._nodes)

    @property
    def nodes_pose(self) -> list[int]:
        return list(self._pose_nodes)

    @property
    def variable_nodes(self) -> dict[VariableKey, int]:
        return dict(self._variable_nodes)

    @property
    def metadata(self) -> GraphMetadata:
        return self._metadata

    @property
    def states(self) -> list[Any]:
        if self._states_cache is None:
            self._states_cache = self._filter_object.get_estimated_state()
        return self._states_cache

    @property
    def states_init(self) -> list[Any] | None:
        return self._states_init

    @property
    def trajectory_poses(self) -> np.ndarray:
        if self._trajectory_poses_cache is None:
            self._trajectory_poses_cache = np.array([np.asarray(self.states[node_id], dtype=float) for node_id in self._pose_nodes])
        return self._trajectory_poses_cache

    @property
    def calibration_values(self) -> dict[VariableKey, Any]:
        return self._extract_calibration_values()

    @property
    def factor_counts(self) -> dict[str, int]:
        return self._metadata.factor_counts

    @property
    def chi2(self) -> float:
        self._chi2 = float(self._filter_object.chi2())
        return self._chi2

    @property
    def last_result(self) -> WindowResult | None:
        return self._last_result

    def node_for(self, key: VariableKey | tuple[str, str]) -> int | None:
        """Return the mrob node id for one calibration variable, if the current graph contains it."""

        key = key if isinstance(key, VariableKey) else VariableKey(key[0], key[1])
        return self._variable_nodes.get(key)

    def clear_rolling_state(self) -> None:
        """Drop all rolling warm-start and committed-output state."""

        self.rolling_state.clear()

    def _invalidate_state_cache(self) -> None:
        self._states_cache = None
        self._trajectory_poses_cache = None

    def _config_for(self, key: VariableKey) -> VariableConfig:
        return self.variable_configs.get(key, VariableConfig()).normalized()

    def _collect_requirements(self, *, record_metadata: bool) -> list[VariableRequirement]:
        requirements: list[VariableRequirement] = []
        for stream in self.streams:
            stream_requirements = tuple(stream.required_variables())
            requirements.extend(stream_requirements)
            if record_metadata:
                self._metadata.streams.append(StreamMetadata(stream_name=stream.stream_name, stream_type=stream.stream_type, sensor_id=stream.sensor.sensor_id, required_variables=tuple(requirement.key for requirement in stream_requirements)))
        return requirements

    def _unique_required_keys(self, requirements: Sequence[VariableRequirement]) -> list[VariableKey]:
        return sorted({requirement.key for requirement in requirements}, key=lambda key: key.label)

    def _validate_pose_timestamps(self, pose_timestamps: Sequence[float]) -> np.ndarray:
        pose_timestamps = np.asarray(pose_timestamps, dtype=float).reshape(-1)
        if pose_timestamps.size == 0:
            raise ValueError("pose_timestamps must contain at least one trajectory pose")
        if not np.all(np.isfinite(pose_timestamps)):
            raise ValueError("pose_timestamps must contain only finite values")
        if np.any(np.diff(pose_timestamps) <= 0.0):
            raise ValueError("pose_timestamps must be strictly increasing")
        return pose_timestamps.copy()

    def _resolve_initial_values(self, requirements: Sequence[VariableRequirement], resolver: _CalibrationValueResolver) -> dict[VariableKey, ResolvedValue]:
        initial_values: dict[VariableKey, ResolvedValue] = {}
        for key in self._unique_required_keys(requirements):
            config = self._config_for(key)
            config.validate(key)
            initial_values[key] = resolver.resolve_initial(key, config)
        return initial_values

    def _trajectory_imu_data(self) -> dict[str, Any] | None:
        for stream in self.streams:
            data = stream.trajectory_imu_data()
            if data is not None:
                return data
        return None

    def _trajectory_lidar_data(self) -> dict[str, Any] | None:
        for stream in self.streams:
            data = stream.trajectory_lidar_data()
            if data is not None:
                return data
        return None

    def _numerical_imu_data(self, sensor_id: str) -> dict[str, Any] | None:
        for stream in self.streams:
            data = stream.numerical_imu_data(sensor_id)
            if data is not None:
                return data
        return None

    def _numerical_lidar_data(self) -> dict[str, Any] | None:
        for stream in self.streams:
            data = stream.numerical_lidar_data()
            if data is not None:
                return data
        return None

    def _initial_value_for_key(self, initial_values: Mapping[VariableKey, ResolvedValue], sensor_id: str, variable_type: VariableType) -> Any:
        key = VariableKey(sensor_id, variable_type)
        if key in initial_values:
            return initial_values[key].value
        return default_value_for(variable_type)

    def _initialize_trajectory_poses_from_values(
        self,
        pose_timestamps: np.ndarray,
        *,
        states: Sequence[Any] | None,
        first_pose: Any,
        initial_values: Mapping[VariableKey, ResolvedValue],
    ) -> np.ndarray:
        imu_data = self._trajectory_imu_data()
        lidar_data = self._trajectory_lidar_data()

        imu_sensor_id = None if imu_data is None else imu_data["sensor_id"]
        lidar_sensor_id = None if lidar_data is None else lidar_data["sensor_id"]
        T_B_I_initial = np.eye(4) if imu_sensor_id is None else self._initial_value_for_key(initial_values, imu_sensor_id, VariableType.EXTRINSIC)
        bias_initial = np.zeros(3) if imu_sensor_id is None else self._initial_value_for_key(initial_values, imu_sensor_id, VariableType.GYRO_BIAS)
        tau_I_initial = 0.0 if imu_sensor_id is None else self._initial_value_for_key(initial_values, imu_sensor_id, VariableType.TIME_OFFSET)
        T_B_L_initial = np.eye(4) if lidar_sensor_id is None else self._initial_value_for_key(initial_values, lidar_sensor_id, VariableType.EXTRINSIC)
        tau_L_initial = 0.0 if lidar_sensor_id is None else self._initial_value_for_key(initial_values, lidar_sensor_id, VariableType.TIME_OFFSET)

        # Reuse the existing trajectory initializer so current LiDAR/IMU propagation conventions remain unchanged.
        return data_processing._initialize_trajectory_poses(
            pose_timestamps=pose_timestamps,
            states=states,
            first_pose=first_pose,
            imu_timestamps=None if imu_data is None else imu_data["imu_timestamps"],
            angular_velocity_imu=None if imu_data is None else imu_data["angular_velocity_imu"],
            T_B_I_initial=T_B_I_initial,
            bias_initial=bias_initial,
            tau_I_initial=float(tau_I_initial),
            lidar_timestamps=None if lidar_data is None else lidar_data["lidar_timestamps"],
            lidar_odometry_poses=None if lidar_data is None else lidar_data["lidar_odometry_poses"],
            T_B_L_initial=T_B_L_initial,
            tau_L_initial=float(tau_L_initial),
            use_imu_gyr=self.trajectory_config.use_imu_gyr,
        )

    def _pose_anchor(self, pose_index: int, number_poses: int, *, is_rolling_window: bool) -> bool:
        if self.trajectory_config.anchor_all_poses:
            return True
        anchor_first_pose = self.trajectory_config.anchor_first_pose_each_window if is_rolling_window else self.trajectory_config.anchor_first_pose
        if anchor_first_pose and pose_index == 0:
            return True
        if self.trajectory_config.anchor_last_pose and pose_index == number_poses - 1:
            return True
        return False

    def _create_trajectory_nodes(self, pose_timestamps: np.ndarray, initial_poses: np.ndarray, *, is_rolling_window: bool) -> None:
        # Trajectory nodes are graph-owned variables, separate from sensor calibration variables.
        for pose_index, pose in enumerate(initial_poses):
            fixed = self._pose_anchor(pose_index, len(initial_poses), is_rolling_window=is_rolling_window)
            node_id = self._filter_object.add_node_pose_3d(data_processing._as_mrob_se3(pose), mode=mrob.NODE_ANCHOR if fixed else mrob.NODE_STANDARD)
            self._nodes.append(node_id)
            self._pose_nodes.append(node_id)
            self._metadata.trajectory_nodes.append(TrajectoryNodeMetadata(pose_index=pose_index, timestamp=float(pose_timestamps[pose_index]), node_id=node_id, fixed=fixed))
        self._invalidate_state_cache()

    def _create_calibration_nodes(self, initial_values: Mapping[VariableKey, ResolvedValue]) -> None:
        # Resolve the calibration variables shared by all streams in this window.
        for key, resolved in sorted(initial_values.items(), key=lambda item: item[0].label):
            config = self._config_for(key)
            node_id = add_variable_node(self._filter_object, key, resolved.value, fixed=config.fixed)
            self._nodes.append(node_id)
            self._variable_nodes[key] = node_id
            self._variable_node_records[key] = VariableNode(key=key, node_id=node_id, initial_value=copy_variable_value(resolved.value), fixed=config.fixed, initial_source=resolved.effective_source)
            self._metadata.calibration_variables.append(CalibrationVariableMetadata(key=key, node_id=node_id, fixed=config.fixed, initial_source=resolved.effective_source, initial_value=copy_variable_value(resolved.value)))
        self._invalidate_state_cache()

    def _add_variable_priors(self, initial_values: Mapping[VariableKey, ResolvedValue], resolver: _CalibrationValueResolver) -> None:
        # Add optional soft priors independently from node initialization. Fixed nodes are anchors, not priors, so their prior metadata is recorded as skipped.
        for key in sorted(initial_values, key=lambda item: item.label):
            config = self._config_for(key)
            if config.prior_source is None:
                continue
            node_id = self._variable_nodes[key]
            prior = resolver.resolve_prior(key, config)
            if config.fixed:
                self._metadata.priors.append(PriorMetadata(factor_id=None, key=key, node_id=node_id, prior_source=prior.effective_source, prior_value=copy_variable_value(prior.value), fixed_node=True, added=False, information=config.prior_information))
                continue
            factor_id = add_variable_prior(self._filter_object, key, node_id, prior.value, config.prior_information)
            self._metadata.priors.append(PriorMetadata(factor_id=factor_id, key=key, node_id=node_id, prior_source=prior.effective_source, prior_value=copy_variable_value(prior.value), fixed_node=False, added=True, information=config.prior_information))
        self._invalidate_state_cache()

    def _add_stream_factors(self, pose_timestamps: np.ndarray) -> None:
        # Hand factor construction to streams; RollingGraph only supplies nodes and records metadata.
        context = StreamContext(graph=self._filter_object, pose_timestamps=pose_timestamps, pose_nodes=self._pose_nodes, variable_nodes=self._variable_nodes, metadata=self._metadata)
        for stream in self.streams:
            stream.add_factors(context)
        self._invalidate_state_cache()

    def build_problem(
        self,
        *,
        pose_timestamps: Sequence[float],
        states: Sequence[Any] | None = None,
        first_pose: Any = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        window_index: int = 0,
        window_start: float | None = None,
        window_end: float | None = None,
        is_rolling_window: bool = False,
        _resolver: _CalibrationValueResolver | None = None,
        _initial_values: Mapping[VariableKey, ResolvedValue] | None = None,
    ) -> "RollingGraph":
        """Build one complete mrob graph for a batch or rolling window."""

        self._reset_graph()
        pose_timestamps = self._validate_pose_timestamps(pose_timestamps)
        window_start = float(pose_timestamps[0] if window_start is None else window_start)
        window_end = float(pose_timestamps[-1] if window_end is None else window_end)
        if window_end < window_start:
            raise ValueError("window_end must not be smaller than window_start")
        self._metadata.window = {"window_index": int(window_index), "window_start": window_start, "window_end": window_end, "is_rolling_window": bool(is_rolling_window)}

        # Collect stream variable requirements before creating any calibration node, so shared variables are created exactly once.
        requirements = self._collect_requirements(record_metadata=True)
        resolver = _CalibrationValueResolver(self, window_start=window_start, window_end=window_end) if _resolver is None else _resolver
        initial_values = self._resolve_initial_values(requirements, resolver) if _initial_values is None else dict(_initial_values)

        # Create trajectory pose nodes using existing propagation logic and the resolved calibration initial values.
        initial_poses = self._initialize_trajectory_poses_from_values(pose_timestamps, states=states, first_pose=first_pose, initial_values=initial_values)
        self._create_trajectory_nodes(pose_timestamps, initial_poses, is_rolling_window=is_rolling_window)

        # Create shared calibration nodes, add optional soft priors, and then let each stream add its own factors.
        self._create_calibration_nodes(initial_values)
        self._add_variable_priors(initial_values, resolver)
        self._add_stream_factors(pose_timestamps)
        self._metadata.numerical_results.update(resolver.numerical_results)

        self._states_init = [np.asarray(value, dtype=float).copy() for value in self.states]
        return self

    def solve_problem(
        self,
        *,
        solver_verbose: bool | None = None,
        solutionTolerance: float | None = None,
        maxIters: int | None = None,
        lambdaParam: float | None = None,
        scheduler: Sequence[tuple[float, int]] | None = None,
        method: str | None = None,
    ) -> list[Any]:
        """Run the configured mrob solver without hard-coding LM or GN."""

        solver_verbose = self.solver_config.solver_verbose if solver_verbose is None else bool(solver_verbose)
        solutionTolerance = self.solver_config.solutionTolerance if solutionTolerance is None else float(solutionTolerance)
        maxIters = self.solver_config.maxIters if maxIters is None else int(maxIters)
        lambdaParam = self.solver_config.lambdaParam if lambdaParam is None else float(lambdaParam)
        method = self.solver_config.method if method is None else str(method).upper()
        scheduler = self.solver_config.scheduler if scheduler is None else [(float(value), int(iterations)) for value, iterations in scheduler]

        self._chi2_prev = self.chi2
        self._states_init = [np.asarray(value, dtype=float).copy() for value in self.states]
        mrob_method = mrob.GN if method == "GN" else mrob.LM
        solve_scheduler = [(lambdaParam, maxIters)] if scheduler is None else scheduler

        for current_lambda, current_iterations in solve_scheduler:
            if int(current_iterations) <= 0:
                continue
            self._filter_object.solve(method=mrob_method, verbose=solver_verbose, solutionTolerance=solutionTolerance, maxIters=int(current_iterations), lambdaParam=float(current_lambda))

        self._invalidate_state_cache()
        self._chi2 = self.chi2
        gc.collect()
        return self.states

    def _extract_calibration_values(self) -> dict[VariableKey, Any]:
        values: dict[VariableKey, Any] = {}
        for key, node_id in self._variable_nodes.items():
            state_value = np.asarray(self.states[node_id], dtype=float)
            if key.variable_type == VariableType.TIME_OFFSET:
                values[key] = float(state_value.reshape(-1)[0])
            elif key.variable_type == VariableType.GYRO_BIAS:
                values[key] = state_value.reshape(3).copy()
            else:
                values[key] = state_value.copy()
        return values

    def _create_result(self, window_index: int, window_start: float, window_end: float, chi2_before: float) -> WindowResult:
        return WindowResult(
            window_index=int(window_index),
            window_start=float(window_start),
            window_end=float(window_end),
            pose_timestamps=np.array([node.timestamp for node in self._metadata.trajectory_nodes], dtype=float),
            trajectory_poses=self.trajectory_poses.copy(),
            calibration_values=self._extract_calibration_values(),
            chi2_before=float(chi2_before),
            chi2_after=float(self.chi2),
            factor_counts=self.factor_counts,
            metadata=self._metadata,
            numerical_results=dict(self._metadata.numerical_results),
        )

    def generate_filter(self, *, pose_timestamps: Sequence[float], states: Sequence[Any] | None = None, first_pose: Any = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), verbose: int = 0) -> WindowResult:
        """Build and solve one batch calibration graph."""

        self.build_problem(pose_timestamps=pose_timestamps, states=states, first_pose=first_pose, window_index=0, is_rolling_window=False)
        if verbose > 0:
            self.print_problem()
        chi2_before = self.chi2
        self.solve_problem()
        result = self._create_result(0, float(self._metadata.window["window_start"]), float(self._metadata.window["window_end"]), chi2_before)
        self._last_result = result
        return result

    def _window_state_seed(
        self,
        *,
        pose_timestamps: np.ndarray,
        window_pose_indices: np.ndarray,
        states: Sequence[Any] | None,
        first_pose: Any,
        initial_values: Mapping[VariableKey, ResolvedValue],
    ) -> list[np.ndarray]:
        window_pose_timestamps = pose_timestamps[window_pose_indices]
        rolling_states = self.rolling_state.pose_prefix(window_pose_timestamps)
        if len(rolling_states) > 0:
            return rolling_states

        supplied_states = [] if states is None else list(states)
        window_states = [supplied_states[index] for index in window_pose_indices if index < len(supplied_states)]
        if len(window_states) > 0:
            return [data_processing._as_pose_matrix(state) for state in window_states]

        # If the first supplied/global state precedes this window, propagate to the first window timestamp before initializing the window itself.
        first_window_index = int(window_pose_indices[0])
        prefix_timestamps = pose_timestamps[: first_window_index + 1]
        prefix_poses = self._initialize_trajectory_poses_from_values(prefix_timestamps, states=supplied_states, first_pose=first_pose, initial_values=initial_values)
        return [prefix_poses[-1]]

    def generate_filter_window(
        self,
        *,
        window_index: int,
        window_start: float,
        window_end: float,
        pose_timestamps: Sequence[float],
        states: Sequence[Any] | None = None,
        first_pose: Any = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        verbose: int = 0,
    ) -> WindowResult:
        """Build and solve one rolling window while warm-starting overlapping trajectory and calibration variables."""

        pose_timestamps = self._validate_pose_timestamps(pose_timestamps)
        window_pose_indices = np.flatnonzero((pose_timestamps >= float(window_start)) & (pose_timestamps <= float(window_end)))
        if len(window_pose_indices) == 0:
            raise ValueError(f"No trajectory poses lie inside rolling window [{window_start}, {window_end}]")

        # Resolve calibration initial values once here because trajectory warm-starting can need them before the graph is built.
        requirements = self._collect_requirements(record_metadata=False)
        resolver = _CalibrationValueResolver(self, window_start=float(window_start), window_end=float(window_end))
        initial_values = self._resolve_initial_values(requirements, resolver)
        window_pose_timestamps = pose_timestamps[window_pose_indices]
        window_states = self._window_state_seed(pose_timestamps=pose_timestamps, window_pose_indices=window_pose_indices, states=states, first_pose=first_pose, initial_values=initial_values)
        first_window_pose = window_states[0] if len(window_states) > 0 else first_pose

        self.build_problem(
            pose_timestamps=window_pose_timestamps,
            states=window_states,
            first_pose=first_window_pose,
            window_index=window_index,
            window_start=window_start,
            window_end=window_end,
            is_rolling_window=True,
            _resolver=resolver,
            _initial_values=initial_values,
        )
        if verbose > 0:
            print(f"WINDOW {window_index}, RAW [{window_start}, {window_end}]:")
            self.print_problem()

        chi2_before = self.chi2
        self.solve_problem()
        result = self._create_result(window_index, window_start, window_end, chi2_before)
        self.rolling_state.store_window_solution(result.pose_timestamps, result.trajectory_poses, result.calibration_values, result=result, metadata={"window_index": int(window_index), "window_start": float(window_start), "window_end": float(window_end)})
        self._last_result = result

        if verbose > 0:
            print(f"WINDOW {window_index}, FILTERED [{window_start}, {window_end}]:")
            self.print_problem()
        return result

    def _rolling_support_interval(self, pose_timestamps: np.ndarray) -> tuple[float, float]:
        safe_starts = [float(pose_timestamps[0])]
        safe_ends = [float(pose_timestamps[-1])]
        for stream in self.streams:
            interval = stream.valid_time_interval()
            if interval is None:
                continue
            safe_starts.append(float(interval[0]))
            safe_ends.append(float(interval[1]))
        required_start = max(safe_starts)
        required_end = min(safe_ends)
        if required_end <= required_start:
            raise ValueError(f"No valid rolling interval remains inside sensor support [{required_start}, {required_end}]")
        return required_start, required_end

    def generate_filter_iterative(
        self,
        *,
        window_size: float,
        step_size: float,
        pose_timestamps: Sequence[float],
        states: Sequence[Any] | None = None,
        first_pose: Any = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        clear_previous: bool = True,
        verbose: int = 0,
    ) -> list[WindowResult]:
        """Solve overlapping rolling windows and carry solved state into each next window."""

        if float(window_size) <= 0.0 or float(step_size) <= 0.0:
            raise ValueError("window_size and step_size must be positive")
        if float(step_size) > float(window_size):
            raise ValueError("step_size should not exceed window_size because rolling windows would not overlap")
        pose_timestamps = self._validate_pose_timestamps(pose_timestamps)
        if clear_previous:
            self.clear_rolling_state()

        required_start, required_end = self._rolling_support_interval(pose_timestamps)
        window_starts: list[float] = []
        current_start = required_start
        while current_start < required_end:
            window_starts.append(float(current_start))
            current_start += float(step_size)

        for window_index, current_window_start in enumerate(window_starts):
            current_window_end = min(current_window_start + float(window_size), required_end)
            window_pose_indices = np.flatnonzero((pose_timestamps >= current_window_start) & (pose_timestamps <= current_window_end))
            if len(window_pose_indices) == 0:
                continue

            result = self.generate_filter_window(window_index=window_index, window_start=current_window_start, window_end=current_window_end, pose_timestamps=pose_timestamps, states=states, first_pose=first_pose, verbose=verbose)
            is_last_window = current_window_end >= required_end
            commit_end = required_end if is_last_window else current_window_start + float(step_size)
            self.rolling_state.commit_output_segment(result.pose_timestamps, result.trajectory_poses, commit_end=commit_end, include_end=is_last_window)
            if is_last_window:
                break
        return list(self.rolling_state.window_results)

    def print_problem(self) -> None:
        """Print a compact graph summary for notebook diagnostics."""

        print(f"Chi2 error = {self.chi2}")
        print(f"Nodes = {self._filter_object.number_nodes()}, factors = {self._filter_object.number_factors()}")
        print(f"Factor counts = {self.factor_counts}")
        for variable in self._metadata.calibration_variables:
            print(f"{variable.key.label}: node={variable.node_id}, fixed={variable.fixed}, initial_source={variable.initial_source.value}")


