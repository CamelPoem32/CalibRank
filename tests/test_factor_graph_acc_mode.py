import sys
from pathlib import Path

import numpy as np
import pytest

MROB_ROOT = Path("/home/camel/Skoltech/Mobile_Robotics_Lab/mrob")
if (MROB_ROOT / "mrobpy").exists():
    sys.path.insert(0, str(MROB_ROOT / "mrobpy"))

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from factor_graph_calibration import FactorGraphCalibration


def _pose(x):
    pose = np.eye(4)
    pose[0, 3] = float(x)
    return pose


def _imu_stream():
    timestamps = np.linspace(-0.5, 1.5, 51)
    accelerometer = np.tile([0.0, 0.0, -9.81], (len(timestamps), 1))
    gyroscope = np.tile([0.1, -0.05, 0.02], (len(timestamps), 1))
    return timestamps, accelerometer, gyroscope


def test_acc_mode_defaults_to_simple_and_preserves_no_bias_node_without_gyro():
    timestamps, accelerometer, _ = _imu_stream()
    graph = FactorGraphCalibration(include_gyro_factors=False, include_accel_factors=True, include_lidar_factors=False)

    graph.build_problem(
        pose_timestamps=[0.0, 0.5, 1.0],
        states=[_pose(0.0), _pose(0.5), _pose(1.0)],
        imu_timestamps=timestamps,
        specific_force_imu=accelerometer,
        T_B_I_initial=np.eye(4),
    )

    assert graph.acc_mode == "simple"
    assert graph.node_bias_g is None
    assert graph.factor_counts["accel"] == 3
    assert all(item["mode"] == "simple" for item in graph.factor_metadata["accel"])


def test_complex_acc_mode_requires_angular_velocity():
    timestamps, accelerometer, _ = _imu_stream()
    graph = FactorGraphCalibration(
        include_gyro_factors=False,
        include_accel_factors=True,
        include_lidar_factors=False,
        acc_mode="complex",
    )

    with pytest.raises(ValueError, match="angular_velocity_imu"):
        graph.build_problem(
            pose_timestamps=[0.0, 0.5, 1.0],
            states=[_pose(0.0), _pose(0.5), _pose(1.0)],
            imu_timestamps=timestamps,
            specific_force_imu=accelerometer,
            T_B_I_initial=np.eye(4),
        )


def test_complex_acc_mode_builds_bias_node_without_gyro_factors():
    timestamps, accelerometer, gyroscope = _imu_stream()
    graph = FactorGraphCalibration(
        include_gyro_factors=False,
        include_accel_factors=True,
        include_lidar_factors=False,
        acc_mode="complex",
    )

    graph.build_problem(
        pose_timestamps=[0.0, 0.5, 1.0],
        states=[_pose(0.0), _pose(0.5), _pose(1.0)],
        imu_timestamps=timestamps,
        angular_velocity_imu=gyroscope,
        specific_force_imu=accelerometer,
        T_B_I_initial=np.eye(4),
        bias_initial=np.zeros(3),
    )

    assert graph.acc_mode == "complex"
    assert graph.node_bias_g is not None
    assert graph.factor_counts["accel"] == 1
    assert graph.factor_metadata["accel"][0]["mode"] == "complex"


def test_invalid_acc_mode_raises():
    with pytest.raises(ValueError, match="acc_mode"):
        FactorGraphCalibration(acc_mode="dynamic")
