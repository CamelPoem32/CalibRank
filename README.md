# Calibration Observability

Research-quality Python reference code for the mathematics needed to construct,
whiten, scale, partition, project, reduce, and analyze Jacobians for online
IMU-LiDAR spatial and temporal calibration.

This package follows the notation and derivations in
`phd_proposal_draft.tex`. It intentionally implements only the observability
mathematics and a simulation-backed Jacobian prototype.

## Scope

- Lie-group utilities for SO(3), SE(3), and SE(2).
- Prediction-first residuals, left-perturbation Jacobians, whitening, scaling,
  dense and sparse Jacobian assembly, projection, Schur reduction, rank, null
  space, and covariance diagnostics.
- A dummy pose backend that returns simulated true poses.
- Planar SE(3)-embedded rover and reduced SE(2) reference simulations.

## Non-goals

This repository does not implement factor-graph nodes, nonlinear optimization,
MROB wrappers, scan matching, mapping, or a production localization pipeline.
Future MROB integration is represented by explicit placeholder interfaces.

## Install

```bash
pip install -e .
pip install -e ".[dev]"
```

## Test

```bash
pytest -q
```

## Execute Notebooks

```bash
jupyter nbconvert --to notebook --execute notebooks/01_math_and_dense_sparse_validation.ipynb --output /tmp/notebook01.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_planar_rover_observability.ipynb --output /tmp/notebook02.ipynb
```

## Mathematical Conventions

- Homogeneous matrices are used for SO(3), SE(3), SO(2), and SE(2).
- Tangents are rotation-first: `xi = [omega_x, omega_y, omega_z, v_x, v_y, v_z]`.
- All state perturbations are left perturbations: `T_perturbed = Exp(delta_xi) @ T`.
- Documentation and code use `Adj`, for example `se3_adjoint(T)`.
- The canonical relative-pose residual is prediction-first:
  `r = Log(prediction @ inverse(measurement))`.

## Dense and Sparse Methods

Dense routines are transparent references and may form explicit projectors.
Sparse routines avoid the full residual-space projector and project each
calibration column by solving a sparse least-squares problem with LSMR.

## Observability Warnings

`rank(J_C)` tests fixed-trajectory calibration sensitivity. `rank(C_X)` tests
motion-only extrinsic sensitivity when body motions are known. Joint calibration
observability is tested with `rank(O_C)`, where `O_C = P_T_perp @ J_C`.
Priors can regularize a matrix and make it full rank without proving
measurement-based observability.

## Future MROB Integration Points

- `calib_observability.backend.MrobPoseProvider`
- `calib_observability.jacobians.get_pose_jacobians_from_mrob`
- `calib_observability.conventions.tangent_from_mrob`
- `calib_observability.conventions.tangent_to_mrob`

These placeholders deliberately raise `NotImplementedError` until MROB tangent
ordering and Jacobian block conventions have been verified.
# CalibRank
