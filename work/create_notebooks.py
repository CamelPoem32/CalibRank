from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": dedent(text).strip().splitlines(True),
    }


def notebook(cells: list[dict]) -> dict:
    for i, cell in enumerate(cells):
        cell.setdefault("id", f"cell-{i:03d}")
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


nb1_cells = [
    md(
        """
        # Math And Dense-Sparse Validation

        Executable checks for the Lie groups, left-perturbation Jacobians,
        whitening, scaling, projection, Schur reduction, and dense/sparse
        invariants used by the observability package.
        """
    ),
    code(
        """
        from pathlib import Path
        import sys, time
        ROOT = Path.cwd()
        for candidate in (ROOT, *ROOT.parents):
            if (candidate / "src" / "calib_observability").exists():
                ROOT = candidate
                break
        sys.path.insert(0, str(ROOT / "src"))
        OUT = ROOT / "outputs" / "math_validation"
        OUT.mkdir(parents=True, exist_ok=True)

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import sparse

        from calib_observability.lie_so3 import *
        from calib_observability.lie_se3 import *
        from calib_observability.lie_se2 import *
        from calib_observability.finite_difference import finite_difference_left_jacobian_se3, finite_difference_left_jacobian_se2
        from calib_observability.residuals import *
        from calib_observability.jacobians import *
        from calib_observability.whitening import whiten_residual_and_jacobian_dense
        from calib_observability.scaling import ParameterScales, build_parameter_scaling_dense, build_parameter_scaling_sparse, scale_jacobian_dense, scale_jacobian_sparse
        from calib_observability.assembly import VariableLayout, JacobianBlock, assemble_jacobian_dense, assemble_jacobian_sparse, make_residual_blocks
        from calib_observability.linalg import *
        from calib_observability.observability import *
        from calib_observability.plotting import save_singular_values_plot, save_sparsity_plot

        rng = np.random.default_rng(123)
        errors = []

        def record(name, A, B, atol=1e-6, rtol=1e-5):
            A = np.asarray(A, dtype=float)
            B = np.asarray(B, dtype=float)
            max_abs = float(np.max(np.abs(A - B))) if A.size else 0.0
            rel = float(np.linalg.norm(A - B) / max(np.linalg.norm(B), 1e-15))
            errors.append((name, max_abs, rel))
            assert max_abs <= atol or rel <= rtol, (name, max_abs, rel)
            return max_abs, rel
        """
    ),
    md("## 1. Imports And Reproducibility"),
    code("assert rng is not None\nprint('output directory:', OUT)"),
    md("## 2. SO(3) Exp/Log Round-Trip Tests"),
    code(
        """
        for _ in range(50):
            omega = rng.normal(0.0, 0.35, 3)
            R = so3_exp(omega)
            record("SO3 Log(Exp)", so3_log(R), omega, atol=1e-10)
            record("SO3 Exp(Log)", so3_exp(so3_log(R)), R, atol=1e-10)
        """
    ),
    md("## 3. SE(3) Exp/Log Round-Trip Tests"),
    code(
        """
        for _ in range(40):
            xi = rng.normal(0.0, [0.2, 0.2, 0.2, 0.5, 0.5, 0.5])
            T = se3_exp(xi)
            record("SE3 Log(Exp)", se3_log(T), xi, atol=1e-9)
            record("SE3 Exp(Log)", se3_exp(se3_log(T)), T, atol=1e-10)
        """
    ),
    md("## 4. SE(2) Exp/Log Round-Trip Tests"),
    code(
        """
        for _ in range(40):
            xi = rng.normal(0.0, [0.3, 0.4, 0.4])
            T = se2_exp(xi)
            record("SE2 Log(Exp)", se2_log(T), xi, atol=1e-10)
            record("SE2 Exp(Log)", se2_exp(se2_log(T)), T, atol=1e-10)
        """
    ),
    md("## 5. Adjoint Identity Checks"),
    code(
        """
        xi = np.array([0.1, -0.2, 0.05, 0.3, -0.1, 0.2])
        eta = np.array([-0.03, 0.04, 0.02, 0.1, 0.05, -0.02])
        T = se3_exp(xi)
        record("SE3 Adj identity", T @ se3_exp(eta) @ se3_inverse(T), se3_exp(se3_adjoint(T) @ eta), atol=1e-10)

        xi2 = np.array([0.3, 0.2, -0.1])
        eta2 = np.array([-0.05, 0.04, 0.02])
        T2 = se2_exp(xi2)
        record("SE2 Adj identity", T2 @ se2_exp(eta2) @ se2_inverse(T2), se2_exp(se2_adjoint(T2) @ eta2), atol=1e-10)
        """
    ),
    md("## 6. Left-Perturbation Checks"),
    code(
        """
        xi = np.array([0.08, -0.04, 0.03, 0.2, 0.1, -0.1])
        delta = np.array([1e-5, -2e-5, 1.5e-5, 2e-5, -1e-5, 1e-5])
        lhs = se3_exp(xi + delta)
        rhs = se3_exp(se3_left_jacobian(xi) @ delta) @ se3_exp(xi)
        record("SE3 left perturbation", lhs, rhs, atol=1e-9)

        xi2 = np.array([0.2, 0.3, -0.2])
        delta2 = np.array([1e-5, -2e-5, 1.5e-5])
        lhs2 = se2_exp(xi2 + delta2)
        rhs2 = se2_exp(se2_left_jacobian(xi2) @ delta2) @ se2_exp(xi2)
        record("SE2 left perturbation", lhs2, rhs2, atol=1e-9)
        """
    ),
    md("## 7. Gyroscope Temporal-Offset Analytic Versus Finite Difference"),
    code(
        """
        times = np.linspace(0.0, 2.0, 240)
        omega = np.c_[0.01 * times, 0.02 * np.sin(times), 0.1 + 0.03 * times**2]
        b_g = np.array([0.001, -0.002, 0.003])
        t0, t1, tau = 0.3, 1.1, 0.02
        R0 = np.eye(3)
        R1 = R0 @ so3_exp(gyro_increment_from_signal(times, omega, t0, t1, tau, b_g, interpolation="cubic"))
        H = gyro_temporal_offset_jacobian(R0, R1, times, omega, t0, t1, tau, b_g, interpolation="cubic")
        eps = 1e-7
        Hp = gyro_propagation_residual(R0, R1, times, omega, t0, t1, tau + eps, b_g, interpolation="cubic")
        Hm = gyro_propagation_residual(R0, R1, times, omega, t0, t1, tau - eps, b_g, interpolation="cubic")
        H_fd = (Hp - Hm) / (2 * eps)
        record("gyro dr/dtau", H, H_fd, atol=2e-6)
        """
    ),
    md("## 8. Spatial Smoothness Analytic Versus Finite Difference"),
    code(
        """
        X0 = se3_exp(np.array([0.05, -0.02, 0.03, 0.2, -0.1, 0.3]))
        X1 = se3_exp(np.array([0.06, -0.01, 0.035, 0.22, -0.09, 0.32]))
        sm = spatial_smoothness_jacobians_left(X0, X1)
        H0 = finite_difference_left_jacobian_se3(lambda X: spatial_smoothness_residual(X, X1), X0)
        H1 = finite_difference_left_jacobian_se3(lambda X: spatial_smoothness_residual(X0, X), X1)
        record("smooth H_X_m", sm.H_X_m, H0, atol=2e-6)
        record("smooth H_X_m1", sm.H_X_m1, H1, atol=2e-6)
        """
    ),
    md("## 9. Extrinsic-Prior Analytic Versus Finite Difference"),
    code(
        """
        pr = extrinsic_prior_jacobian_left(X0, X1)
        Hp = finite_difference_left_jacobian_se3(lambda X: extrinsic_prior_residual(X, X1), X0)
        record("prior H_X", pr.H_X, Hp, atol=2e-6)
        """
    ),
    md("## 10. Conjugation Pose-Residual Calibration Jacobian"),
    code(
        """
        T0 = se3_exp(np.array([0.02, -0.01, 0.03, 0.1, 0.0, 0.0]))
        T1 = se3_exp(np.array([0.04, -0.015, 0.08, 0.5, 0.2, 0.05]))
        X = se3_exp(np.array([0.01, 0.02, -0.02, 0.3, -0.1, 0.2]))
        Z_true = sensor_relative_prediction(T0, T1, X)
        Z = se3_exp(np.array([1e-4, -2e-4, 1.5e-4, 3e-4, -1e-4, 2e-4])) @ Z_true
        cal = pose_residual_calibration_jacobian_left(T0, T1, X, Z)
        Hfd = finite_difference_left_jacobian_se3(
            lambda Y: relative_pose_residual_prediction_first(sensor_relative_prediction(T0, T1, Y), Z), X
        )
        record("pose calibration H_X", cal.H_X, Hfd, atol=5e-6)
        """
    ),
    md("## 11. Whitening Validation"),
    code(
        """
        r = np.array([0.3, -0.2, 0.1])
        H = rng.normal(size=(3, 5))
        Sigma = np.array([[2.0, 0.2, 0.1], [0.2, 1.5, 0.0], [0.1, 0.0, 1.0]])
        rb, Hb = whiten_residual_and_jacobian_dense(r, H, Sigma)
        assert np.allclose(rb @ rb, r @ np.linalg.inv(Sigma) @ r, atol=1e-12)
        """
    ),
    md("## 12. Parameter-Scaling Validation"),
    code(
        """
        layout = VariableLayout.from_specs([("T_W_B_0", 6, "trajectory"), ("T_B_L", 6, "calibration"), ("b_g", 3, "calibration"), ("tau_L", 1, "calibration")])
        scales = ParameterScales(rotation_scale_rad=0.1, translation_scale_m=2.0, gyro_bias_scale_rad_s=0.01, time_offset_scale_s=0.05)
        Dd = build_parameter_scaling_dense(layout.blocks, scales)
        Ds = build_parameter_scaling_sparse(layout.blocks, scales)
        J = rng.normal(size=(12, layout.total_dim))
        record("dense sparse scaling", scale_jacobian_dense(J, Dd), scale_jacobian_sparse(sparse.csr_matrix(J), Ds).toarray(), atol=1e-12)
        assert np.linalg.matrix_rank(J) == np.linalg.matrix_rank(scale_jacobian_dense(J, Dd))
        """
    ),
    md("## 13. Gram-Matrix And Null-Space Examples"),
    code(
        """
        M = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        G = gram_matrix_dense(M)
        ns = null_space_dense(M, tolerance=1e-12)
        assert G.shape == (3, 3)
        assert ns.shape[1] == 1
        assert np.allclose(M @ ns, 0.0, atol=1e-12)
        """
    ),
    md("## 14. Tiny 2D Geometric Projector Example"),
    code(
        """
        J_T = np.array([[1.0], [0.0]])
        J_C = np.eye(2)
        O = effective_observability_dense(J_T, J_C)
        assert np.allclose(O, np.array([[0.0, 0.0], [0.0, 1.0]]))
        """
    ),
    md("## 15. Dense Explicit Projector Versus Dense QR Projection"),
    code(
        """
        J_T = rng.normal(size=(45, 8))
        J_C = rng.normal(size=(45, 5))
        O = effective_observability_dense(J_T, J_C)
        Oq = effective_observability_dense_qr(J_T, J_C)
        record("dense projector vs QR S", O.T @ O, Oq.T @ Oq, atol=1e-9)
        """
    ),
    md("## 16. Dense Projection Versus Sparse LSMR Projection"),
    code(
        """
        sp_result = effective_observability_sparse_lsmr(sparse.csr_matrix(J_T), sparse.csr_matrix(J_C))
        record("dense vs sparse LSMR S", O.T @ O, sp_result.S_C, atol=1e-8)
        """
    ),
    md("## 17. Dense Schur Complement Versus Projected Information"),
    code(
        """
        S_proj = O.T @ O
        S_schur = schur_complement_dense(J_T, J_C)
        record("Schur vs projection", S_schur, S_proj, atol=1e-9)
        """
    ),
    md("## 18. Verification: `S_C == O_C.T @ O_C`"),
    code(
        """
        S_C = reduced_information_dense(O)
        record("S_C identity", S_C, O.T @ O, atol=1e-12)
        """
    ),
    md("## 19. Random Sparse Block-Jacobian Assembly"),
    code(
        """
        layout = VariableLayout.from_specs([("T_W_B_0", 6, "trajectory"), ("T_W_B_1", 6, "trajectory"), ("T_B_L", 6, "calibration")])
        residual_blocks = make_residual_blocks([("r0", 6, np.eye(6), "measurement"), ("r1", 6, np.eye(6), "measurement")])
        blocks = [
            JacobianBlock("r0", "T_W_B_0", rng.normal(size=(6, 6))),
            JacobianBlock("r0", "T_B_L", rng.normal(size=(6, 6))),
            JacobianBlock("r1", "T_W_B_1", rng.normal(size=(6, 6))),
            JacobianBlock("r1", "T_B_L", rng.normal(size=(6, 6))),
        ]
        bd = assemble_jacobian_dense(layout, residual_blocks, blocks)
        bs = assemble_jacobian_sparse(layout, residual_blocks, blocks)
        record("dense sparse assembly J", bd.J, bs.J.toarray(), atol=1e-12)
        save_sparsity_plot(bs.J, OUT / "random_sparse_jacobian.png", "Random sparse block Jacobian")
        """
    ),
    md("## 20. Dense Versus Sparse Results"),
    code(
        """
        Od = effective_observability_dense(bd.J_T, bd.J_C)
        Os = effective_observability_sparse_lsmr(bs.J_T, bs.J_C)
        record("assembled dense sparse S", Od.T @ Od, Os.S_C, atol=1e-8)
        """
    ),
    md("## 21. Runtime And Memory Comparison"),
    code(
        """
        big_T = sparse.random(250, 70, density=0.035, random_state=1, format="csr")
        big_C = sparse.random(250, 8, density=0.2, random_state=2, format="csr")
        t0 = time.perf_counter()
        sparse_big = effective_observability_sparse_lsmr(big_T, big_C)
        sparse_runtime = time.perf_counter() - t0
        dense_nbytes = big_T.toarray().nbytes + big_C.toarray().nbytes
        sparse_nbytes = big_T.data.nbytes + big_T.indices.nbytes + big_T.indptr.nbytes + big_C.data.nbytes + big_C.indices.nbytes + big_C.indptr.nbytes
        assert sparse_big.S_C.shape == (8, 8)
        print({"sparse_runtime_s": sparse_runtime, "dense_nbytes": dense_nbytes, "sparse_nbytes": sparse_nbytes})
        """
    ),
    md("## 22. Motion-Only `C_X` Rank Test"),
    code(
        """
        A_list = [
            se3_exp(np.array([0.0, 0.0, 0.3, 1.0, 0.0, 0.0])),
            se3_exp(np.array([0.2, 0.1, 0.0, 0.0, 1.0, 0.2])),
            se3_exp(np.array([-0.1, 0.3, 0.2, 0.4, -0.2, 1.0])),
        ]
        C_X = build_motion_only_matrix_dense(A_list)
        assert numerical_rank_dense(C_X, tolerance=1e-9) == 6
        B = sparse.block_diag([rng.normal(size=(6, 6)) + 4.0 * np.eye(6) for _ in A_list]).toarray()
        BC = B @ C_X
        assert numerical_rank_dense(BC, tolerance=1e-9) == numerical_rank_dense(C_X, tolerance=1e-9)
        assert not np.allclose(np.linalg.svd(BC, compute_uv=False), np.linalg.svd(C_X, compute_uv=False))
        save_singular_values_plot(np.linalg.svd(C_X, compute_uv=False), OUT / "motion_only_singular_values.png", "C_X singular values")
        """
    ),
    md("## 23. Full-Rank `C_X` But Jointly Deficient Example"),
    code(
        """
        J_X = C_X.copy()
        J_N = J_X.copy()
        O_X = effective_observability_dense(J_N, J_X)
        assert numerical_rank_dense(J_X, tolerance=1e-9) == 6
        assert numerical_rank_dense(O_X, tolerance=1e-9) == 0
        """
    ),
    md("## 24. Summary Table Of Numerical Errors"),
    code(
        """
        for name, max_abs, rel in errors:
            print(f"{name:35s} max_abs={max_abs:.3e} rel={rel:.3e}")
        assert max(e[1] for e in errors) < 1e-5
        """
    ),
]


nb2_cells = [
    md(
        """
        # Planar Rover Observability

        This notebook simulates planar rover motion embedded in full SE(3),
        compares dense and sparse observability reductions, runs sliding-window
        analysis, and contrasts full SE(3) planar degeneracy with reduced SE(2)
        and synthetic 3D excitation.
        """
    ),
    code(
        """
        from pathlib import Path
        import sys, time
        ROOT = Path.cwd()
        for candidate in (ROOT, *ROOT.parents):
            if (candidate / "src" / "calib_observability").exists():
                ROOT = candidate
                break
        sys.path.insert(0, str(ROOT / "src"))
        OUT = ROOT / "outputs" / "planar_rover"
        OUT.mkdir(parents=True, exist_ok=True)

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import sparse

        from calib_observability.backend import estimate_poses_dummy
        from calib_observability.linalg import numerical_rank_dense
        from calib_observability.lie_se2 import se2_exp
        from calib_observability.lie_se3 import se3_log
        from calib_observability.observability import *
        from calib_observability.plotting import save_sparsity_plot, save_singular_values_plot
        from calib_observability.simulation import PlanarRoverConfig, simulate_planar_rover

        rng = np.random.default_rng(321)

        def rank_dense(M, tol=1e-7):
            return numerical_rank_dense(M.toarray() if sparse.issparse(M) else M, tolerance=tol)
        """
    ),
    md("## 1. Simulation Configuration"),
    code(
        """
        cfg = PlanarRoverConfig(imu_rate_hz=20.0, lidar_rate_hz=3.0, total_laps=1, random_seed=21)
        ds = simulate_planar_rover(cfg, mode="one_rectangle")
        print(cfg)
        assert ds.lidar.measurements.shape[0] > 10
        """
    ),
    md("## 2. Rectangular Rover Trajectory Plot"),
    code(
        """
        t, xyz, rpy = ds.trajectory.sample(500)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(xyz[:, 0], xyz[:, 1], lw=2)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Smooth rectangular rover trajectory")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "trajectory_xy.png", dpi=160)
        plt.close(fig)
        """
    ),
    md("## 3. Position, Yaw, Velocity, Yaw-Rate, And Acceleration"),
    code(
        """
        vel = np.vstack([ds.trajectory.velocity_at(float(ti)) for ti in t])
        acc = np.vstack([ds.trajectory.acceleration_at(float(ti)) for ti in t])
        yaw_rate = np.array([ds.trajectory.yaw_rate_at(float(ti)) for ti in t])
        fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        axs[0].plot(t, xyz[:, :2]); axs[0].set_ylabel("pos [m]")
        axs[1].plot(t, rpy[:, 2]); axs[1].set_ylabel("yaw [rad]")
        axs[2].plot(t, vel[:, :2]); axs[2].set_ylabel("vel [m/s]")
        axs[3].plot(t, yaw_rate, label="yaw rate"); axs[3].plot(t, np.linalg.norm(acc[:, :2], axis=1), label="|acc|"); axs[3].legend(); axs[3].set_ylabel("rate/acc")
        axs[3].set_xlabel("time [s]")
        for ax in axs:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "trajectory_timeseries.png", dpi=160)
        plt.close(fig)
        """
    ),
    md("## 4. IMU And LiDAR Timestamp Visualization"),
    code(
        """
        fig, ax = plt.subplots(figsize=(8, 2.4))
        ax.plot(ds.imu.sensor_timestamps, np.zeros_like(ds.imu.sensor_timestamps), "|", label="IMU")
        ax.plot(ds.lidar.sensor_timestamps, np.ones_like(ds.lidar.sensor_timestamps), "|", label="LiDAR")
        ax.set_yticks([0, 1], ["IMU", "LiDAR"])
        ax.set_xlabel("sensor timestamp [s]")
        ax.set_title("Sensor timestamps")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "timestamps.png", dpi=160)
        plt.close(fig)
        """
    ),
    md("## 5. Simulated Gyroscope And Accelerometer Streams"),
    code(
        """
        fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        axs[0].plot(ds.imu.sensor_timestamps, ds.imu.gyroscope); axs[0].set_ylabel("gyro [rad/s]")
        axs[1].plot(ds.imu.sensor_timestamps, ds.imu.accelerometer); axs[1].set_ylabel("accel [m/s^2]")
        axs[1].set_xlabel("sensor time [s]")
        for ax in axs:
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "imu_streams.png", dpi=160)
        plt.close(fig)
        """
    ),
    md("## 6. Simulated LiDAR Relative-Pose Measurements"),
    code(
        """
        z_xi = np.vstack([se3_log(Z) for Z in ds.lidar.measurements])
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(ds.lidar.relative_start_times, z_xi[:, 2], label="yaw")
        ax.plot(ds.lidar.relative_start_times, z_xi[:, 3], label="x")
        ax.plot(ds.lidar.relative_start_times, z_xi[:, 4], label="y")
        ax.set_xlabel("time [s]")
        ax.set_title("LiDAR relative measurement components")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "lidar_measurements.png", dpi=160)
        plt.close(fig)
        """
    ),
    md("## 7. True Extrinsic And Temporal-Offset Values"),
    code(
        """
        print("tau_I_true", ds.tau_I_true)
        print("tau_L_true", ds.tau_L_true)
        print("gyro_bias_true", ds.gyro_bias_true)
        print("T_B_L_true tangent", se3_log(ds.T_B_L_true))
        print("T_B_I_true tangent", se3_log(ds.T_B_I_true))
        """
    ),
    md("## 8. Dummy Pose-Provider Demonstration"),
    code(
        """
        provider = estimate_poses_dummy(ds)
        poses = provider.poses_at(np.array([ds.start_time, ds.start_time + 1.0]))
        assert poses.shape == (2, 4, 4)
        """
    ),
    md("## 9. One-Window Jacobian Layout And Sparsity Plot"),
    code(
        """
        w0, w1 = ds.start_time, min(ds.start_time + 6.0, ds.end_time)
        bd, motions, counts = ds.window_jacobians(w0, w1, provider, use_sparse=False)
        bs, _, _ = ds.window_jacobians(w0, w1, provider, use_sparse=True)
        assert bd.J.shape == bs.J.shape
        save_sparsity_plot(bs.J, OUT / "one_window_sparsity.png", "One-window sparse Jacobian")
        print({"shape": bd.J.shape, "J_T": bd.J_T.shape, "J_C": bd.J_C.shape, "counts": counts})
        """
    ),
    md("## 10. Dense Versus Sparse Observability Comparison"),
    code(
        """
        O_dense = effective_observability_dense(bd.J_T, bd.J_C)
        S_dense = O_dense.T @ O_dense
        sparse_result = effective_observability_sparse_lsmr(bs.J_T, bs.J_C)
        assert np.allclose(S_dense, sparse_result.S_C, atol=1e-7)
        print({"rank_O_dense": rank_dense(O_dense), "rank_O_sparse": rank_dense(sparse_result.O_C), "nnz": sparse_result.nnz})
        """
    ),
    md("## 11. Sliding-Window Observability Over The Whole Trajectory"),
    code(
        """
        results = analyze_observability_over_time(ds, provider, window_duration=6.0, window_step=3.0, include_priors=False, use_sparse=True, rank_tolerance=1e-7)
        assert len(results) >= 3
        centers = np.array([(r["window_start"] + r["window_end"]) / 2 for r in results])
        rank_JC = np.array([r["rank_J_C"] for r in results])
        rank_CX = np.array([r["rank_C_X"] for r in results])
        rank_OC = np.array([r["rank_O_C"] for r in results])
        smin = np.array([r["smallest_nonzero_singular_value"] for r in results])
        cond = np.array([r["condition_number"] for r in results])
        """
    ),
    md("## 12. Plot Ranks, Smallest Singular Value, Condition Number, And Selected Standard Deviations"),
    code(
        """
        fig, axs = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        axs[0].plot(centers, rank_JC, marker="o", label="rank(J_C)")
        axs[0].plot(centers, rank_CX, marker="o", label="rank(C_X)")
        axs[0].plot(centers, rank_OC, marker="o", label="rank(O_C)")
        axs[0].legend(); axs[0].set_ylabel("rank")
        axs[1].semilogy(centers, np.maximum(smin, 1e-16), marker="o"); axs[1].set_ylabel("smallest nonzero sigma")
        axs[2].semilogy(centers, np.maximum(cond, 1.0), marker="o"); axs[2].set_ylabel("condition"); axs[2].set_xlabel("time [s]")
        for ax in axs:
            for q in np.linspace(ds.start_time, ds.end_time, 5)[1:-1]:
                ax.axvline(q, color="0.6", lw=0.8, ls="--", alpha=0.5)
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / "sliding_window_observability.png", dpi=160)
        plt.close(fig)
        """
    ),
    md("## 13. Straight Segments And Turns Marked On Plots"),
    code(
        """
        # The dashed quartile markers in the previous figure approximate the four sides/turn regions
        # of the smooth rectangular trajectory. They are visual guides, not observability inputs.
        assert (OUT / "sliding_window_observability.png").exists()
        """
    ),
    md("## 14. Compare Motion Modes"),
    code(
        """
        modes = ["stationary", "straight_constant_velocity", "straight_accelerating", "single_turn", "one_rectangle", "multiple_rectangles"]
        mode_rows = []
        for mode in modes:
            dsm = simulate_planar_rover(cfg, mode=mode)
            ppm = estimate_poses_dummy(dsm)
            bm, motions_m, _ = dsm.window_jacobians(dsm.start_time, dsm.end_time, ppm, use_sparse=False)
            Cxm = build_motion_only_matrix_dense(motions_m)
            Om = effective_observability_dense(bm.J_T, bm.J_C) if bm.J.shape[0] else np.zeros((0, bm.J_C.shape[1]))
            mode_rows.append((mode, rank_dense(Cxm), rank_dense(bm.J_C), rank_dense(Om), bm.J.shape))
        for row in mode_rows:
            print(row)
        assert dict((m, c) for m, c, _, _, _ in mode_rows)["stationary"] == 0
        """
    ),
    md("## 15. Measurement-Only Observability Versus Regularized Uncertainty"),
    code(
        """
        b_meas, _, _ = ds.window_jacobians(w0, w1, provider, include_priors=False, use_sparse=False)
        b_reg, _, _ = ds.window_jacobians(w0, w1, provider, include_priors=True, include_smoothness=True, use_sparse=False)
        O_meas = effective_observability_dense(b_meas.J_T, b_meas.J_C)
        O_reg = effective_observability_dense(b_reg.J_T, b_reg.J_C)
        print({"measurement_only_rank": rank_dense(O_meas), "regularized_rank": rank_dense(O_reg)})
        assert rank_dense(O_reg) >= rank_dense(O_meas)
        print("Priors can make a regularized matrix full rank without proving measurement-based observability.")
        """
    ),
    md("## 16. SE(3)-Embedded Planar Versus Reduced SE(2) Versus Synthetic 3D"),
    code(
        """
        Cx_planar = build_motion_only_matrix_dense(motions)
        rank_planar = rank_dense(Cx_planar)
        As2 = [
            se2_exp(np.array([0.0, 1.0, 0.0])),
            se2_exp(np.array([np.pi / 2, 1.0, 0.5])),
            se2_exp(np.array([-np.pi / 3, 0.5, 0.8])),
        ]
        rank_se2 = rank_dense(build_motion_only_matrix_dense_se2(As2), tol=1e-9)
        ds3 = simulate_planar_rover(cfg, mode="multi-axis_3d_reference_motion")
        pp3 = estimate_poses_dummy(ds3)
        b3, motions3, _ = ds3.window_jacobians(ds3.start_time, ds3.end_time, pp3, use_sparse=False)
        rank_3d = rank_dense(build_motion_only_matrix_dense(motions3), tol=1e-7)
        print({"SE3_planar_CX_rank": rank_planar, "SE2_reference_CX_rank": rank_se2, "synthetic_3D_CX_rank": rank_3d})
        assert rank_planar < 6
        assert rank_se2 == 3
        assert rank_3d == 6
        """
    ),
    md("## 17. Weakest Calibration Right-Singular Vectors"),
    code(
        """
        labels = ["roll", "pitch", "yaw", "x", "y", "z", "b_gx", "b_gy", "b_gz", "tau_I", "tau_L"]
        U, s, Vt = np.linalg.svd(O_dense, full_matrices=False)
        weakest = Vt[-min(6, Vt.shape[0]):]
        fig, ax = plt.subplots(figsize=(8, 3.5))
        im = ax.imshow(weakest, aspect="auto", cmap="coolwarm", vmin=-np.max(np.abs(weakest)), vmax=np.max(np.abs(weakest)))
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
        ax.set_yticks(range(weakest.shape[0]), [f"v-{i}" for i in range(weakest.shape[0], 0, -1)])
        ax.set_title("Weak calibration directions")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(OUT / "weak_singular_vectors.png", dpi=160)
        plt.close(fig)
        save_singular_values_plot(s, OUT / "one_window_singular_values.png", "One-window O_C singular values")
        """
    ),
    md("## 18. Approximate Physical Interpretation"),
    code(
        """
        print("Weak directions in planar SE(3) commonly involve roll, pitch, z translation, and nuisance columns such as bias/time offsets.")
        print("Yaw, x translation, and y translation tend to improve during turns relative to stationary or straight-only windows.")
        """
    ),
    md("## 19. Matrix Sizes, Nonzero Counts, And Dense/Sparse Runtimes"),
    code(
        """
        t0 = time.perf_counter()
        _ = effective_observability_dense(bd.J_T, bd.J_C)
        dense_runtime = time.perf_counter() - t0
        t0 = time.perf_counter()
        sp = effective_observability_sparse_lsmr(bs.J_T, bs.J_C)
        sparse_runtime = time.perf_counter() - t0
        print({"J_shape": bd.J.shape, "J_sparse_nnz": bs.J.nnz, "dense_runtime_s": dense_runtime, "sparse_runtime_s": sparse_runtime, "sparse_density": bs.J.nnz / (bs.J.shape[0] * bs.J.shape[1])})
        """
    ),
    md("## 20. Figures Saved Under `outputs/planar_rover/`"),
    code(
        """
        saved = sorted(p.name for p in OUT.glob("*.png"))
        print(saved)
        assert {"trajectory_xy.png", "sliding_window_observability.png", "weak_singular_vectors.png"}.issubset(saved)
        """
    ),
]


def main() -> None:
    out = Path("notebooks")
    out.mkdir(exist_ok=True)
    (out / "01_math_and_dense_sparse_validation.ipynb").write_text(json.dumps(notebook(nb1_cells), indent=2), encoding="utf-8")
    (out / "02_planar_rover_observability.ipynb").write_text(json.dumps(notebook(nb2_cells), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
