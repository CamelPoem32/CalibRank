# Notation

This package follows `phd_proposal_draft.tex`.

| LaTeX notation | Python name | Meaning |
| --- | --- | --- |
| `${}^{W}T_B$` | `T_W_B` | body pose in the world frame |
| `${}^{B}T_I$` | `T_B_I` | IMU extrinsic in the body frame |
| `${}^{B}T_L$` | `T_B_L` | LiDAR extrinsic in the body frame |
| `\tau_I` | `tau_I` | IMU temporal offset |
| `\tau_L` | `tau_L` | LiDAR temporal offset |
| `b_g` | `b_g` | gyroscope bias |
| `J_T` | `J_T` | trajectory Jacobian block |
| `J_C` | `J_C` | calibration Jacobian block |
| `\mathcal O_C` | `O_C` | effective calibration observability matrix |
| `S_C` | `S_C` | reduced calibration information matrix |
| `\mathcal C_X` | `C_X` | motion-only extrinsic sensitivity matrix |

## Tangent Ordering

The central tangent-vector ordering is rotation-first:

```text
xi = [omega_x, omega_y, omega_z, v_x, v_y, v_z]
```

`omega` is the rotational perturbation in radians and `v` is the translational
perturbation in metres. The reduced SE(2) tangent is:

```text
xi = [omega, v_x, v_y]
```

Do not assume MROB uses this ordering. Use `tangent_from_mrob` and
`tangent_to_mrob` after the MROB ordering has been verified.

## Perturbation Convention

All perturbations are left perturbations:

```text
T_perturbed = Exp(delta_xi) @ T
```

Right perturbations are not used silently anywhere in the package.

## Residual Convention

The canonical relative-pose residual is prediction-first:

```text
r = Log(prediction @ inverse(measurement))
```

The measurement-first alternative is implemented only under an explicit name for
comparison and regression testing.
