from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

MROB_ROOT = Path("/home/camel/Skoltech/Mobile_Robotics_Lab/mrob")
if (MROB_ROOT / "mrobpy").exists():
    sys.path.insert(0, str(MROB_ROOT / "mrobpy"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calib_observability.lie_se3 import se3_exp
from calib_observability.sensor_emulation import SE3CalibrationLaw, ScalarCalibrationLaw
from kaist_dataset.lidar_map import LidarMapRegistrationResult, save_lidar_map_pose_csv
from multi_sensor_pipeline.kaist_perturbation import (
    LidarPoseObservations,
    PerturbedKaistStreams,
    build_kaist_perturbation_laws,
    build_kaist_variable_configs,
    build_perturbed_kaist_streams,
    load_kaist_lidar_map_pose_observations,
    perturb_lidar_sensor_poses,
    run_kaist_perturbed_calibration_graph,
)
from multi_sensor_pipeline.variables import VariableKey, VariableType


def _pose(x=0.0, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def _sensor_pose_observations(tmp_path: Path, *, sensor_id="lidar_0"):
    timestamps = np.linspace(1.0, 5.0, 5)
    poses = np.stack([_pose(t, 2.0 * t, 0.5) for t in timestamps])
    return LidarPoseObservations(sensor_id=sensor_id, path=tmp_path / f"{sensor_id}.csv", timestamps_s=timestamps, poses_T_W_L=poses, dataframe=pd.DataFrame({"timestamp": timestamps, "success": True}))


def test_lidar_pose_perturbation_preserves_T_A_B_convention():
    timestamps = np.array([10.0, 11.0])
    T_B_L_reference = _pose(1.0, 0.0, 0.0)
    T_B_L_truth = _pose(2.0, 0.5, -0.25)
    body_poses = np.stack([_pose(0.0, 0.0, 0.0), _pose(5.0, 0.0, 0.0)])
    poses_T_W_L_reference = body_poses @ T_B_L_reference

    result = perturb_lidar_sensor_poses(
        timestamps,
        poses_T_W_L_reference,
        T_B_L_reference=T_B_L_reference,
        T_B_L_law=SE3CalibrationLaw("constant", reference=T_B_L_truth),
        tau_L_reference=0.1,
        tau_L_law=ScalarCalibrationLaw("constant", value=0.25),
    )

    np.testing.assert_allclose(result.poses_T_W_L, body_poses @ T_B_L_truth)
    np.testing.assert_allclose(result.timestamps_s, timestamps + 0.15)
    np.testing.assert_allclose(result.tau_L_truth, np.full(2, 0.25))


def test_perturbed_stream_builder_uses_notebook15_imu_emulation_shapes(tmp_path: Path):
    imu_timestamps = np.linspace(0.0, 6.0, 61)
    gyro = np.tile([0.01, -0.02, 0.03], (imu_timestamps.size, 1))
    accel = np.tile([0.0, 0.0, 9.81], (imu_timestamps.size, 1))
    lidar_left = _sensor_pose_observations(tmp_path)
    laws = build_kaist_perturbation_laws(
        t_start_s=0.0,
        t_end_s=6.0,
        T_B_I_reference=np.eye(4),
        tau_I_reference=0.0,
        T_B_LL_reference=np.eye(4),
        tau_LL_reference=0.0,
        T_B_I_mode="constant",
        tau_I_mode="constant",
        T_B_L_mode="constant",
        tau_L_mode="constant",
    )

    streams = build_perturbed_kaist_streams(
        imu_timestamps_s=imu_timestamps,
        gyroscope_radps=gyro,
        accelerometer_mps2=accel,
        lidar_left=lidar_left,
        laws=laws,
        T_B_I_reference=np.eye(4),
        tau_I_reference=0.0,
        T_B_LL_reference=np.eye(4),
        imu_time_offset_margin=0.1,
        lidar_time_offset_margin=0.1,
        lidar_samples_per_factor=2,
    )

    assert len(streams.streams) == 3
    assert streams.emulated_imu.gyroscope.shape == gyro.shape
    assert streams.emulated_imu.accelerometer.shape == accel.shape
    assert np.all(np.diff(streams.emulated_imu.sensor_timestamps) > 0.0)
    assert streams.pose_timestamps_s.size == lidar_left.timestamps_s.size
    assert streams.initial_body_poses_T_W_B.shape == (lidar_left.timestamps_s.size, 4, 4)


def test_variable_config_builder_switches_fixed_and_free_modes():
    fixed = build_kaist_variable_configs(True, T_B_I_initial=np.eye(4), T_B_LL_initial=np.eye(4), right_lidar_sensor_id="lidar_1", T_B_RL_initial=np.eye(4))
    free = build_kaist_variable_configs(False, T_B_I_initial=np.eye(4), T_B_LL_initial=np.eye(4), right_lidar_sensor_id="lidar_1", T_B_RL_initial=np.eye(4), T_B_I_prior_information=1.0)

    imu_key = VariableKey("imu_0", VariableType.EXTRINSIC)
    left_tau_key = VariableKey("lidar_0", VariableType.TIME_OFFSET)
    right_ext_key = VariableKey("lidar_1", VariableType.EXTRINSIC)

    assert fixed[imu_key].fixed is True
    assert fixed[left_tau_key].initial_source.value == "constant"
    assert free[imu_key].fixed is False
    assert free[imu_key].initial_source.value == "optimized"
    assert free[imu_key].prior_source.value == "optimized"
    assert right_ext_key in fixed
    assert right_ext_key in free

    numerical = build_kaist_variable_configs(False, T_B_I_initial=np.eye(4), T_B_LL_initial=np.eye(4), right_lidar_sensor_id="lidar_1", T_B_RL_initial=np.eye(4), T_B_I_prior_information=1.0, T_B_L_prior_information=1.0, tau_L_prior_information=1.0, initial_source="numerical", prior_source="numerical")
    assert numerical[imu_key].initial_source.value == "numerical"
    assert numerical[imu_key].prior_source.value == "numerical"
    assert numerical[left_tau_key].initial_source.value == "optimized"
    assert numerical[left_tau_key].prior_source.value == "optimized"
    assert numerical[right_ext_key].initial_source.value == "optimized"
    assert numerical[right_ext_key].prior_source.value == "optimized"


def test_graph_runner_uses_notebook19_window_scaling():
    calls = {}

    class FakeRollingGraph:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
            self.rolling_state = SimpleNamespace(rolling_trajectory=(np.array([1.0]), np.eye(4)[None, :, :]))

        def generate_filter_iterative(self, **kwargs):
            calls["generate"] = kwargs
            return []

    perturbed = PerturbedKaistStreams(streams=[], sensors=[], emulated_imu=None, lidar_left=None, lidar_right=None, pose_timestamps_s=np.array([1.0, 2.0, 3.0]), initial_body_poses_T_W_B=np.stack([_pose(), _pose(1), _pose(2)]))

    with patch("multi_sensor_pipeline.kaist_perturbation.RollingGraph", FakeRollingGraph):
        run = run_kaist_perturbed_calibration_graph(label="fixed", fixed_calibration=True, perturbed_streams=perturbed, variable_configs={}, window_size=5.0)

    assert calls["generate"]["window_size"] == 500.0
    assert calls["generate"]["step_size"] == 250.0
    assert run.window_size_s == 500.0
    assert run.step_size_s == 250.0


def test_missing_right_lidar_csv_is_optional(tmp_path: Path):
    T_est = _pose(1.0, 2.0, 3.0)
    csv_path = tmp_path / "left.csv"
    save_lidar_map_pose_csv([
        LidarMapRegistrationResult(timestamp_s=1.0, scan_path="scan.bin", T_W_L_initial=T_est, T_W_L_estimated=T_est, success=True, fitness=1.0, inlier_rmse=0.1)
    ], csv_path)

    left, right = load_kaist_lidar_map_pose_observations(csv_path, tmp_path / "missing_right.csv", require_right=False)

    assert left is not None
    assert right is None
    np.testing.assert_allclose(left.poses_T_W_L[0], T_est)
