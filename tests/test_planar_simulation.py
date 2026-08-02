from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.calib_observability.backend import estimate_poses_dummy
from src.calib_observability.linalg import numerical_rank_dense
from src.calib_observability.observability import build_motion_only_matrix_dense, build_motion_only_matrix_dense_se2
from src.calib_observability.simulation import PlanarRoverConfig, simulate_planar_rover
from src.calib_observability.lie_se2 import se2_exp


def test_planar_simulation_shapes_and_dummy_backend() -> None:
    simulation_config = PlanarRoverConfig(imu_rate_hz=20.0, lidar_rate_hz=4.0, random_seed=10)
    dataset = simulate_planar_rover(simulation_config, mode="one_rectangle")
    assert dataset.imu.gyroscope.shape[1] == 3
    assert dataset.lidar.measurements.shape[1:] == (4, 4)
    provider = estimate_poses_dummy(dataset)
    poses = provider.poses_at(np.array([dataset.start_time, min(dataset.start_time + 1.0, dataset.end_time)]))
    assert poses.shape == (2, 4, 4)


def test_planar_embedded_se3_has_out_of_plane_degeneracy() -> None:
    simulation_config = PlanarRoverConfig(imu_rate_hz=20.0, lidar_rate_hz=4.0, random_seed=11)
    dataset = simulate_planar_rover(simulation_config, mode="one_rectangle")
    provider = estimate_poses_dummy(dataset)
    bundle, motions, counts = dataset.window_jacobians(
        dataset.start_time,
        dataset.end_time,
        provider,
        use_sparse=False,
    )
    C_X = build_motion_only_matrix_dense(motions)
    assert C_X.shape[1] == 6
    assert numerical_rank_dense(C_X, tolerance=1e-7) < 6
    assert bundle.J_C.shape[1] == 11
    assert "T_B_L" not in bundle.calibration_column_slices
    assert counts["imu"] > 0
    assert counts["lidar"] > 0


def test_reduced_se2_motion_can_have_full_rank() -> None:
    As = [
        se2_exp(np.array([0.0, 1.0, 0.0])),
        se2_exp(np.array([np.pi / 2, 1.0, 0.5])),
        se2_exp(np.array([-np.pi / 3, 0.5, 0.8])),
    ]
    C = build_motion_only_matrix_dense_se2(As)
    assert numerical_rank_dense(C, tolerance=1e-9) == 3
