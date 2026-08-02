from __future__ import annotations

import numpy as np

from src.calib_observability.simulation import PlanarRoverConfig
from src.calib_observability.types import AccelerometerOptions
from src.calib_observability.workflows import (
    build_dataset_from_imported_sensor_streams,
    build_planar_rover_dataset,
    plot_local_crlb_accuracy,
    plot_observability_over_time,
    plot_rover_dataset_overview,
    run_rolling_observability_analysis,
)


def test_wrapped_rover_workflow_runs_modular_pipeline(tmp_path) -> None:
    config = PlanarRoverConfig(
        rectangle_width=2.0,
        rectangle_height=1.0,
        straight_speed=1.0,
        turn_duration=0.4,
        total_laps=1,
        imu_rate_hz=20.0,
        lidar_rate_hz=4.0,
        gyro_noise_std=1e-5,
        accel_noise_std=1e-5,
        random_seed=3,
        mode="one_rectangle",
    )
    dataset, provider = build_planar_rover_dataset(config, trajectory_mode="one_rectangle")

    overview_paths = plot_rover_dataset_overview(dataset, tmp_path, trajectory_samples=80)
    series = run_rolling_observability_analysis(
        dataset,
        provider,
        window_duration=1.0,
        window_step=1.0,
        accelerometer_options=AccelerometerOptions(mode="simple", factor_rate_hz=4.0, measurement_std_m_s2=0.05),
        lidar_rate_hz=config.lidar_rate_hz,
        display_variables=("T_B_I", "b_g", "tau_I", "tau_L"),
    )
    observability_paths = plot_observability_over_time(series, tmp_path, display_variables=("T_B_I", "b_g", "tau_I", "tau_L"))
    accuracy_paths = plot_local_crlb_accuracy(series, tmp_path, display_variables=("T_B_I", "b_g", "tau_I", "tau_L"))

    assert series.snapshots
    assert all(path.exists() for path in overview_paths.values())
    assert {"ranks", "condition_numbers", "factor_counts"} <= set(observability_paths)
    assert "tau_bounds" in accuracy_paths
    assert all(path.exists() for path in observability_paths.values())
    assert all(path.exists() for path in accuracy_paths.values())


def test_imported_sensor_streams_build_dataset_interface() -> None:
    scan_times = np.array([10.0, 10.5, 11.0], dtype=float)
    relative_poses = np.repeat(np.eye(4)[None, :, :], 2, axis=0)
    relative_poses[:, 0, 3] = 0.5
    imu_times = np.linspace(10.0, 11.0, 21)
    gyroscope = np.zeros((imu_times.size, 3), dtype=float)
    accelerometer = np.tile(np.array([0.0, 0.0, 9.81], dtype=float), (imu_times.size, 1))

    dataset = build_dataset_from_imported_sensor_streams(
        imu_timestamps=imu_times,
        gyroscope=gyroscope,
        accelerometer=accelerometer,
        lidar_scan_timestamps=scan_times,
        lidar_relative_poses=relative_poses,
    )

    assert dataset.start_time == 0.0
    assert dataset.end_time == 1.0
    assert dataset.imu.gyroscope.shape == (21, 3)
    assert dataset.lidar.measurements.shape == (2, 4, 4)
    assert np.allclose(dataset.trajectory.position_at(0.75), [0.75, 0.0, 0.0])
