# KAIST Observability MP4

From the repository root in the WSL Anaconda environment:

```bash
python src/kaist_dataset/run_kaist_observability.py \
  /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16
```

This uses the left LiDAR. Select `--lidar right` for the other sensor. The
selected `lidar_map_poses_vlp_<side>.csv` must already exist in the dataset root,
its `data` directory, or the repository's `data/KAISTDataset/<UrbanName>`.
Use `--processed-data-dir` to override discovery. No point clouds are copied or
registered by this command.

Results default to `<Urban directory>/outputs/calib_observability/<side>/`:

- `observability_dashboard.mp4`: subprocess-rendered dashboard, without HTML.
- `trajectory_comparison.png`: LiDAR-derived and KAIST reference trajectories.
- Overview, observability and local uncertainty PNGs from the notebook workflow.
- `observability_windows.csv`: per-window validity, ranks, conditions and bounds.
- `run_summary.json`: paths, configuration, sampling counts/rates and transforms.

Relative output and processed-data paths are resolved against the Urban directory.
Use separate output directories to retain multiple experiments.

## Bounded Runs

```bash
python src/kaist_dataset/run_kaist_observability.py \
  /mnt/d/Downloads/MobRobLab/KAISTDataset/Urban16 \
  --lidar right --end-time 20 \
  --max-analysis-windows 10 --max-rendered-frames 8 \
  --output-dir outputs/observability_preview/right
```

Start/end offsets are seconds from the common IMU/LiDAR/reference overlap.
Without those offsets, caps sample across the whole overlap rather than truncate
the sequence. IMU and reference streams keep bracketing interpolation support.

Defaults: IMU resampling at 100 Hz; at most 100000 IMU samples, 5000 LiDAR
poses, 5000 reference poses, 300 analysis windows, 300 video frames, and 700
path samples. Window length is 5 seconds and requested spacing is 1 second.
Video defaults to 5 fps at 100 DPI on a 17 x 10 inch figure.

Sample caps reduce effective measurement rates and therefore change information
and uncertainty values. The command reports these rates and cap activation.
Window caps select endpoints before expensive Jacobian assembly. Frame caps
apply before serialization. The final dataset timestamp is always included.
Raw CSV ingestion still reads the source tables before applying sample budgets.

## Frames And Assumptions

Absolute successful scan-to-map poses are sampled first, then converted to
`inverse(T_W_L[i]) @ T_W_L[i+1]`. The analysis uses LiDAR-derived body poses,
not a newly optimized trajectory. Ground truth is only a plotted comparison.
One common timestamp origin and world translation are applied to all streams.
The fixed-extrinsic body frame is the selected LiDAR; the reference is transformed
to that frame too. Calibration conventions match the existing KAIST runner.

Noise flags are whitening assumptions, not injected noise or measured truth.
The simple accelerometer uses notebook 11's low-dynamic gate and default
`--simple-accel-noise-std 1e-6`. Gravity defaults to `[0, 0, -9.81]` for the
observability specific-force convention. Use `--help` for all overrides.

FFmpeg must be available to Matplotlib. Both the CLI and child renderer use Agg.

## Console Output

Use `--verbose 0` (default) for a silent successful run, `--verbose 1` for
processing messages and live analysis-window/MP4-frame progress, or `--verbose 2` to also print
JSON metadata. `-v` and `--verbosity` are aliases. The tqdm frame counter shows
elapsed time, rate and estimated time remaining; it is hidden at level 0.
Saved result files are identical at every verbosity level. Failures still raise
an error with the subprocess diagnostics.

## Parallel Analysis

Gyro weighting for imported data uses one representative rotation standard
deviation per window: raw gyro sigma [rad/s] times
sqrt(median retained IMU dt * median usable gyro pose-interval dt).
Raw sigma comes from sqrt(trace(gyro_covariance) / 3). The resulting sigma [rad]
is squared to weight every gyro rotation residual in that window. This is a
simple independent white-noise approximation, not exact sample propagation.
The four quantities are recorded in each bundle's `gyro_noise_conversion`
metadata and `run_summary.json` under `gyro_noise_windows`; verbosity 2 prints
them too. Simulation weighting and accelerometer/LiDAR covariances are unchanged.

Pass `--n-processes 4 --verbose 1` to analyze windows in a four-process pool.
The default `--n-processes 1` keeps sequential execution. The analysis progress
bar counts completed windows out of the total retained window count, regardless
of how many workers are active. Results are restored to increasing timestamp
order before plots, CSV export and video generation.

For Python callers, set `KaistObservabilityConfig(n_processes=4, ...)`, pass
`n_processes=4` to `run_rolling_observability_analysis` or
`save_simple_accelerometer_dashboard`, or set
`QuasiRealtimeConfig(n_processes=4)` for `compute_quasi_realtime_snapshots`.
Pass `verbose=1` to show window progress in those shared functions.

Workers use the spawn start method and load a cloudpickle-serialized copy of the
dataset/provider once per process, including supported local trajectory closures.
Each worker needs memory for its dataset copy and active analysis window.
Standalone Python scripts must invoke parallel analysis under
`if __name__ == "__main__":`. The provided CLI already does this.

## Optional optimized analysis and notebook 23

The default remains optimize=False. Pass --optimize to the CLI, or set
KaistObservabilityConfig(dataset_root=..., optimize=True) in Python. Use
--no-optimize for an explicit legacy comparison. The flag also reaches rolling
analysis, its process workers, and dense/sparse per-target projection helpers.

The dense optimized branch applies the retained nuisance SVD basis directly
instead of allocating I - J_N @ pinv(J_N). Its cutoff remains 1e-15, matching the
legacy pseudoinverse. For retained nuisance condition numbers above 1e8, it
reconstructs the legacy projector from the same SVD to preserve floating-point
behavior; these exceptional cases still allocate the dense projector. Right SVDs are reused within each target calculation;
tall matrices use economy SVD, while wide matrices retain their full right null
space. Physical and column-filtered matrices keep separate decompositions when
filtering changes the matrix. No cache survives a target evaluation.
Sparse LSMR solves, Jacobian assembly, noise models and practical-rank/CRLB
definitions remain unchanged. The optimized rolling path omits redundant rank
recomputation; validate_stored_rank_against_matrix remains available for audits.
Roundoff-level machine ranks can differ between projection algorithms.

Notebook notebooks/23_kaist_observability_optimization_benchmark.ipynb loads
Urban16 through load_observability_inputs and calls analyze_observability_inputs,
the same numerical function used by run_observability. It needs the existing
KAIST map-pose CSV and the dev dependencies (including pandas and threadpoolctl).
Adjust its dataset, interval, window size, dense/sparse, process and BLAS controls
before running all cells. It warms both branches, checks practical ranks,
subspaces and uncertainty, then alternates timed runs and saves CSV comparisons,
a timing plot and environment metadata under outputs/kaist_observability_optimization.

Analysis timing excludes input loading and rendering. A second benchmark uses
a preassembled KAIST bundle to isolate projection and diagnostics. Equal BLAS
thread counts apply to both modes and spawned workers. Keep separate output
directories when comparing configurations. These optimizations run on CPU;
a GPU/autograd implementation is a separate step.
