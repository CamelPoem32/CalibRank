from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

MROB_ROOT = Path("/home/camel/Skoltech/Mobile_Robotics_Lab/mrob")
if (MROB_ROOT / "mrobpy").exists():
    sys.path.insert(0, str(MROB_ROOT / "mrobpy"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from multi_sensor_pipeline import (
    GyroStream,
    LidarOdometryStream,
    RollingGraph,
    Sensor,
    SimpleAccelStream,
    SolverConfig,
    TrajectoryConfig,
    ValueSource,
    VariableConfig,
    VariableKey,
    VariableType,
)
from numerical_calibration import NumericalCalibrationResult


def _pose(x: float = 0.0) -> np.ndarray:
    pose = np.eye(4)
    pose[0, 3] = float(x)
    return pose


def _poses(timestamps) -> np.ndarray:
    return np.asarray([_pose(timestamp) for timestamp in timestamps])


def _imu_measurements():
    timestamps = np.linspace(-0.5, 1.5, 41)
    gyroscope = np.tile([0.01, -0.02, 0.03], (timestamps.size, 1))
    accelerometer = np.tile([0.0, 0.0, -9.81], (timestamps.size, 1))
    return timestamps, gyroscope, accelerometer


def _lidar_measurements():
    timestamps = np.linspace(-0.5, 1.5, 41)
    return timestamps, _poses(timestamps)


def _base_streams():
    imu = Sensor("imu_0", kind="imu")
    lidar = Sensor("lidar_0", kind="lidar")
    imu_t, gyro, accel = _imu_measurements()
    lidar_t, lidar_poses = _lidar_measurements()
    return [
        GyroStream(imu, imu_t, gyro, samples_per_factor=None, time_offset_margin=0.2),
        SimpleAccelStream(imu, imu_t, accel, samples_per_factor=None, time_offset_margin=0.2),
        LidarOdometryStream(lidar, lidar_t, lidar_poses, samples_per_factor=None, time_offset_margin=0.2),
    ]


def _successful_result(T_B_I, tau_I):
    return NumericalCalibrationResult(
        tau_I=tau_I,
        R_B_I=T_B_I[:3, :3].copy(),
        T_B_I=T_B_I.copy(),
        bias_g_used=np.zeros(3),
        temporal_delay_raw=0.0,
        spatial_rssd=0.0,
        source_timestamps=np.array([0.0, 1.0]),
        reference_timestamps=np.array([0.0, 1.0]),
        synchronized_timestamps=np.array([0.0, 1.0]),
        source_angvels=np.zeros((2, 3)),
        reference_angvels=np.zeros((2, 3)),
        source_angvels_synchronized=np.zeros((2, 3)),
        reference_angvels_synchronized=np.zeros((2, 3)),
        source_angvels_aligned=np.zeros((2, 3)),
        residuals=np.zeros((2, 3)),
        residual_rmse=np.zeros(3),
        residual_vector_rmse=0.0,
        residual_vector_median=0.0,
        excitation_singular_values=np.ones(3),
        excitation_ratios=(1.0, 1.0),
        success=True,
        message="ok",
    )


def test_gyro_accel_and_lidar_streams_create_factors_and_share_imu_nodes():
    pose_timestamps = np.array([0.0, 0.5, 1.0])
    graph = RollingGraph(streams=_base_streams(), solver_config=SolverConfig(maxIters=0))

    graph.build_problem(pose_timestamps=pose_timestamps, states=_poses(pose_timestamps))

    imu_extrinsic = VariableKey("imu_0", VariableType.EXTRINSIC)
    imu_tau = VariableKey("imu_0", VariableType.TIME_OFFSET)
    assert graph.factor_counts["gyro_calib_prop"] == 2
    assert graph.factor_counts["accel_gravity_calib"] == 3
    assert graph.factor_counts["lidar_calib_odometry"] == 2
    assert len([variable for variable in graph.metadata.calibration_variables if variable.key == imu_extrinsic]) == 1
    assert graph.node_for(imu_extrinsic) in graph.metadata.factors[0].node_ids
    assert graph.node_for(imu_extrinsic) in graph.metadata.factors[2].node_ids
    assert graph.node_for(imu_tau) in graph.metadata.factors[0].node_ids
    assert graph.node_for(imu_tau) in graph.metadata.factors[2].node_ids


def test_two_imus_receive_different_calibration_nodes():
    pose_timestamps = np.array([0.0, 0.5, 1.0])
    imu_t, gyro, accel = _imu_measurements()
    streams = [
        GyroStream("imu_0", imu_t, gyro, samples_per_factor=None, time_offset_margin=0.2),
        SimpleAccelStream("imu_0", imu_t, accel, samples_per_factor=None, time_offset_margin=0.2),
        GyroStream("imu_1", imu_t, gyro, samples_per_factor=None, time_offset_margin=0.2),
        SimpleAccelStream("imu_1", imu_t, accel, samples_per_factor=None, time_offset_margin=0.2),
    ]
    graph = RollingGraph(streams=streams, solver_config=SolverConfig(maxIters=0))

    graph.build_problem(pose_timestamps=pose_timestamps, states=_poses(pose_timestamps))

    assert graph.node_for(("imu_0", "extrinsic")) != graph.node_for(("imu_1", "extrinsic"))
    assert graph.node_for(("imu_0", "time_offset")) != graph.node_for(("imu_1", "time_offset"))


def test_initial_value_and_prior_value_are_independent_sources():
    pose_timestamps = np.array([0.0, 0.5, 1.0])
    imu_t, gyro, _ = _imu_measurements()
    key = VariableKey("imu_0", VariableType.EXTRINSIC)
    T_initial = _pose(1.0)
    T_prior = _pose(2.0)
    graph = RollingGraph(
        streams=[GyroStream("imu_0", imu_t, gyro, samples_per_factor=None, time_offset_margin=0.2)],
        variable_configs={key: VariableConfig(initial_source="constant", initial_value=T_initial, prior_source="constant", prior_value=T_prior, prior_information=1.0)},
        solver_config=SolverConfig(maxIters=0),
    )

    graph.build_problem(pose_timestamps=pose_timestamps, states=_poses(pose_timestamps))

    node_id = graph.node_for(key)
    assert np.allclose(graph.states_init[node_id], T_initial)
    assert np.allclose(graph.metadata.priors[0].prior_value, T_prior)
    assert graph.metadata.priors[0].added is True


def test_numerical_calibration_can_initialize_without_adding_prior():
    pose_timestamps = np.array([0.0, 0.5, 1.0])
    key = VariableKey("imu_0", VariableType.EXTRINSIC)
    T_numeric = _pose(3.0)
    graph = RollingGraph(
        streams=_base_streams(),
        variable_configs={key: VariableConfig(initial_source="numerical")},
        solver_config=SolverConfig(maxIters=0),
    )

    with patch("multi_sensor_pipeline.rolling_graph.estimate_imu_calibration_numerical", return_value=_successful_result(T_numeric, 0.12)) as estimate:
        graph.build_problem(pose_timestamps=pose_timestamps, states=_poses(pose_timestamps))

    assert estimate.call_count == 1
    assert np.allclose(graph.states_init[graph.node_for(key)], T_numeric)
    assert graph.metadata.priors == []


def test_numerical_calibration_can_be_prior_without_changing_initial_value():
    pose_timestamps = np.array([0.0, 0.5, 1.0])
    key = VariableKey("imu_0", VariableType.EXTRINSIC)
    T_initial = _pose(1.0)
    T_numeric_prior = _pose(4.0)
    graph = RollingGraph(
        streams=_base_streams(),
        variable_configs={key: VariableConfig(initial_source="constant", initial_value=T_initial, prior_source="numerical", prior_information=1.0)},
        solver_config=SolverConfig(maxIters=0),
    )

    with patch("multi_sensor_pipeline.rolling_graph.estimate_imu_calibration_numerical", return_value=_successful_result(T_numeric_prior, -0.04)) as estimate:
        graph.build_problem(pose_timestamps=pose_timestamps, states=_poses(pose_timestamps))

    assert estimate.call_count == 1
    assert np.allclose(graph.states_init[graph.node_for(key)], T_initial)
    assert np.allclose(graph.metadata.priors[0].prior_value, T_numeric_prior)
    assert graph.metadata.priors[0].prior_source == ValueSource.NUMERICAL


def test_numerical_calibration_is_reused_for_initial_and_prior():
    pose_timestamps = np.array([0.0, 0.5, 1.0])
    key = VariableKey("imu_0", VariableType.EXTRINSIC)
    T_numeric = _pose(5.0)
    graph = RollingGraph(
        streams=_base_streams(),
        variable_configs={key: VariableConfig(initial_source="numerical", prior_source="numerical", prior_information=1.0)},
        solver_config=SolverConfig(maxIters=0),
    )

    with patch("multi_sensor_pipeline.rolling_graph.estimate_imu_calibration_numerical", return_value=_successful_result(T_numeric, 0.22)) as estimate:
        graph.build_problem(pose_timestamps=pose_timestamps, states=_poses(pose_timestamps))

    assert estimate.call_count == 1
    assert np.allclose(graph.states_init[graph.node_for(key)], T_numeric)
    assert np.allclose(graph.metadata.priors[0].prior_value, T_numeric)


def test_fixed_variable_and_soft_prior_are_distinct_mechanisms():
    pose_timestamps = np.array([0.0, 0.5, 1.0])
    imu_t, gyro, _ = _imu_measurements()
    key = VariableKey("imu_0", VariableType.EXTRINSIC)
    config = VariableConfig(initial_source="constant", initial_value=_pose(0.0), fixed=True, prior_source="constant", prior_value=_pose(2.0), prior_information=1.0)
    graph = RollingGraph(
        streams=[GyroStream("imu_0", imu_t, gyro, samples_per_factor=None, time_offset_margin=0.2)],
        variable_configs={key: config},
        solver_config=SolverConfig(maxIters=0),
    )

    graph.build_problem(pose_timestamps=pose_timestamps, states=_poses(pose_timestamps))

    assert graph.metadata.calibration_variables[0].fixed is True
    assert graph.metadata.priors[0].fixed_node is True
    assert graph.metadata.priors[0].added is False
    assert "extrinsic_prior" not in graph.factor_counts


def test_rolling_state_carries_optimized_calibration_into_next_window():
    pose_timestamps = np.array([0.0, 0.5, 1.0])
    imu_t, gyro, _ = _imu_measurements()
    key = VariableKey("imu_0", VariableType.EXTRINSIC)
    T_first = _pose(1.5)
    T_fallback = _pose(9.0)
    graph = RollingGraph(
        streams=[GyroStream("imu_0", imu_t, gyro, samples_per_factor=None, time_offset_margin=0.2)],
        variable_configs={key: VariableConfig(initial_source="optimized", initial_value=T_first)},
        solver_config=SolverConfig(maxIters=0),
        trajectory_config=TrajectoryConfig(use_imu_gyr=True),
    )

    with patch.object(RollingGraph, "solve_problem", lambda self, **_: self.states):
        first = graph.generate_filter_window(window_index=0, window_start=0.0, window_end=0.5, pose_timestamps=pose_timestamps, states=_poses(pose_timestamps))
    assert np.allclose(graph.rolling_state.get_calibration(key), first.calibration_values[key])

    graph.variable_configs[key] = VariableConfig(initial_source="optimized", initial_value=T_fallback).normalized()
    with patch.object(RollingGraph, "solve_problem", lambda self, **_: self.states):
        graph.generate_filter_window(window_index=1, window_start=0.5, window_end=1.0, pose_timestamps=pose_timestamps, states=_poses(pose_timestamps))

    assert np.allclose(graph.states_init[graph.node_for(key)], first.calibration_values[key])
    assert not np.allclose(graph.states_init[graph.node_for(key)], T_fallback)


def test_graph_metadata_is_visualization_neutral():
    pose_timestamps = np.array([0.0, 0.5, 1.0])
    graph = RollingGraph(streams=_base_streams(), solver_config=SolverConfig(maxIters=0))

    graph.build_problem(pose_timestamps=pose_timestamps, states=_poses(pose_timestamps))

    assert len(graph.metadata.trajectory_nodes) == 3
    assert len(graph.metadata.calibration_variables) >= 5
    assert "networkx" not in sys.modules
    assert "graphviz" not in sys.modules

