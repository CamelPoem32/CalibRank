# Mathematical Mapping

| Mathematical object | Python function |
| --- | --- |
| `Adj(T)` | `calib_observability.lie_se3.se3_adjoint` |
| `r_smooth = Log(X_m^{-1} X_{m+1})` | `calib_observability.residuals.spatial_smoothness_residual` |
| `dr_smooth / d xi_m`, `dr_smooth / d xi_{m+1}` | `calib_observability.jacobians.spatial_smoothness_jacobians_left` |
| `r_prior = Log(X^{-1} X_0)` | `calib_observability.residuals.extrinsic_prior_residual` |
| `dr_prior / d xi` | `calib_observability.jacobians.extrinsic_prior_jacobian_left` |
| `Z_hat = X^{-1} A_m X` | `calib_observability.residuals.sensor_relative_prediction` |
| `r_cal = Log(Z_hat Z_m^{-1})` | `calib_observability.residuals.relative_pose_residual_prediction_first` |
| `dr_cal / d xi_X` | `calib_observability.jacobians.pose_residual_calibration_jacobian_left` |
| `P_T_perp = I - J_T J_T^\dagger` | `calib_observability.observability.trajectory_projector_dense` |
| `O_C = P_T_perp J_C` | `calib_observability.observability.effective_observability_dense` |
| `S_C = O_C.T O_C` | `calib_observability.observability.reduced_information_dense` |
| `S_C = C - B.T A^{-1} B` | `calib_observability.observability.schur_complement_dense` |
| `C_X = stack(Adj(A_m) - I_6)` | `calib_observability.observability.build_motion_only_matrix_dense` |

The smoothness and prior Jacobians use the later derivation in the source and
the implementation brief:

```text
H_m  = -J_l^{-1}(r_m) @ Adj(X_m^{-1})
H_m1 =  J_l^{-1}(r_m) @ Adj(X_m^{-1})
```

This is the convention tested by left-perturbation finite differences.
