import numpy as np

from src.new_college_dataset.data import IMUData, LidarData
from src.new_college_dataset.imu import resample_imu
from src.transform import se3_to_angvels, se3_to_velocities


def test_imu_data_validates_shapes_and_monotonic_timestamps():
    timestamps = np.array([0.0, 0.01, 0.02])
    samples = np.zeros((3, 3))
    imu = IMUData(timestamps, samples, samples)
    assert imu.timestamps_s.shape == (3,)
    assert imu.accel_mps2.shape == (3, 3)
    assert imu.gyro_radps.shape == (3, 3)


def test_resample_imu_outputs_requested_frequency():
    timestamps = np.array([0.0, 0.5, 1.0])
    accel = np.column_stack([timestamps, timestamps * 2.0, timestamps * 3.0])
    gyro = accel + 1.0
    imu = IMUData(timestamps, accel, gyro)

    resampled = resample_imu(imu, target_frequency_hz=10.0)

    assert np.allclose(np.diff(resampled.timestamps_s), 0.1)
    assert resampled.accel_mps2.shape == (11, 3)
    assert resampled.gyro_radps.shape == (11, 3)


def test_se3_to_angvels_known_rotation():
    angle = 0.2
    pose = np.eye(4)
    pose[:3, :3] = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    angular_velocity = se3_to_angvels(np.asarray([pose]), np.array([0.0, 2.0]))

    assert np.allclose(angular_velocity, [[0.0, 0.0, 0.1]])


def test_se3_to_velocities_known_translation():
    pose = np.eye(4)
    pose[:3, 3] = np.array([2.0, 4.0, 6.0])

    velocity = se3_to_velocities(np.asarray([pose]), np.array([0.0, 2.0]))

    assert np.allclose(velocity, [[1.0, 2.0, 3.0]])


def test_lidar_data_accepts_relative_pose_shape():
    poses = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    lidar = LidarData(
        timestamps_s=np.array([0.05, 0.15]),
        relative_poses_se3=poses,
        scan_timestamps_s=np.array([0.0, 0.1, 0.2]),
    )
    assert lidar.relative_poses_se3.shape == (2, 4, 4)
