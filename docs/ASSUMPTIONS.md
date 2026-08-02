# Assumptions

1. Tangent vectors are rotation-first: `[omega, v]` for SE(3) and
   `[omega, v_x, v_y]` for SE(2).
2. All perturbations are left perturbations:
   `T_perturbed = Exp(delta_xi) @ T`.
3. Relative pose residuals use the prediction-first convention
   `Log(prediction @ inverse(measurement))` unless an explicit
   measurement-first function is called.
4. Body-frame calibration variables use `${}^{B}T_I` and `${}^{B}T_L`.
5. Sensor clocks use `true_time = sensor_timestamp + tau_sensor`.
6. Gyroscope signals are interpolated with linear interpolation or SciPy cubic
   splines before integration.
7. Covariances are expected to be symmetric positive definite for whitening.
8. Gauge choices are explicit: one clock or one extrinsic may be fixed by
   omitting its variable block, and trajectory gauge should be fixed before
   interpreting full-system rank as full observability.
9. Planar rover motion is embedded in full SE(3), so out-of-plane calibration
   directions may remain unobservable even when a reduced SE(2) model is full
   rank.
10. The dummy backend returns true simulated poses. It validates observability
    mathematics but is not a substitute for future factor-graph estimates.
