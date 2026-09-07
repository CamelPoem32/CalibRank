'''Aggregate KAIST multi-sequence pipeline results and compare calibration against fixed calibration parameters.

The script inspects only the direct child directories of DATASET_ROOT. Each
child directory is treated as one dataset, for example:

    KAISTDataset/
        Urban13/
        Urban14/
        Urban15/
        Urban16/
        Urban17/

For a selected LiDAR mode, the expected input layout inside every dataset is:

    <dataset>/
        experiment_results/
            errors/
                poses/
                    with_calibration/
                        trajectory_errors.pkl
                    without_calibration/
                        trajectory_errors.pkl

or:

    <dataset>/
        experiment_results/
            errors/
                odometry/
                    with_calibration/
                        trajectory_errors.pkl
                    without_calibration/
                        trajectory_errors.pkl

Only direct children of DATASET_ROOT are inspected. Dataset discovery is not
recursive.

The raw trajectory errors stored in the pickle are reduced to one set of
statistics per dataset. This keeps each driving sequence as the primary
experimental unit when averaging improvements across datasets.

For an error metric e, calibration gain is defined as

    absolute_gain = e_without_calibration - e_with_calibration

and

    relative_gain_percent =
        100 * absolute_gain / e_without_calibration.

Therefore positive gain always means that calibration improved the result.

Default output:

    <dataset_root>/pipeline_results_summary/<lidar_mode>/
        with_calibration_metrics.csv
        without_calibration_metrics.csv
        calibration_gains.csv
        calibration_gains_long.csv
        calibration_gain_summary.csv
        plots/
            comparison_median.png
            comparison_rmse.png
            comparison_p90.png
            gain_median.png
            gain_rmse.png
            gain_p90.png
            average_gain_median.png
            average_gain_rmse.png
            average_gain_p90.png

Example:

    python process_kaist_pipeline_results.py /mnt/d/Downloads/MobRobLab/KAISTDataset

Example for LiDAR odometry:

    python process_kaist_pipeline_results.py /mnt/d/Downloads/MobRobLab/KAISTDataset --lidar-mode odometry

Example with a custom output directory:

    python process_kaist_pipeline_results.py /mnt/d/Downloads/MobRobLab/KAISTDataset --lidar-mode poses --output-dir /home/camel/Skoltech/phd_proposal/kaist_summary

Example selecting different statistics:

    python process_kaist_pipeline_results.py /mnt/d/Downloads/MobRobLab/KAISTDataset --statistics median mean rmse p90 p10
'''

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import re
import warnings
from typing import Any

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


##################################################
# Error definitions
##################################################


ERROR_TYPES = {
    'full_se3': {
        'title': 'Full SE(3) tangent error',
        'ylabel': r'$\|\mathrm{Log}(T_{ref}^{-1}T_{est})\|_2$',
    },
    'rotation_deg': {
        'title': 'Rotational error',
        'ylabel': 'error [deg]',
    },
    'translation_m': {
        'title': 'Translational error',
        'ylabel': 'error [m]',
    },
}

AVAILABLE_STATISTICS = ('median', 'mean', 'rmse', 'p90', 'p10')

RUN_CONFIG_COMPARE_KEYS = (
    'lidar_mode',
    'acc_mode',
    'window_size_s',
    'step_size_s',
    'imu_frequency_hz',
    'map_pose_factor_stride',
    'gyro_information',
    'accel_information',
    'lidar_rotation_information',
    'lidar_translation_information',
    'map_pose_rotation_information',
    'map_pose_translation_information',
    'gravity_z_mps2',
)


##################################################
# Generic helpers
##################################################


def natural_sort_key(value: str) -> list[Any]:
    '''Return a natural sorting key so Urban2 appears before Urban10.'''

    return [int(component) if component.isdigit() else component.lower() for component in re.split(r'(\d+)', value)]


def resolve_output_directory(dataset_root: Path, requested_path: Path | None, lidar_mode: str) -> Path:
    '''Resolve the summary output directory.'''

    if requested_path is None:
        return (dataset_root / 'pipeline_results_summary' / lidar_mode).resolve()

    output_dir = requested_path.expanduser()

    if not output_dir.is_absolute():
        output_dir = dataset_root / output_dir

    return output_dir.resolve()


def load_result_pickle(path: Path) -> dict[str, Any]:
    '''Load one trusted pipeline result pickle.'''

    with path.open('rb') as stream:
        payload = pickle.load(stream)

    if not isinstance(payload, dict):
        raise ValueError(f'Expected dictionary payload in {path}, got {type(payload).__name__}')

    return payload


def finite_error_array(payload: dict[str, Any], error_name: str, path: Path) -> np.ndarray:
    '''Extract one finite one-dimensional error array.'''

    if 'errors' not in payload or not isinstance(payload['errors'], dict):
        raise ValueError(f'Pickle does not contain an errors dictionary: {path}')

    if error_name not in payload['errors']:
        raise ValueError(f'Pickle does not contain errors["{error_name}"]: {path}')

    values = np.asarray(payload['errors'][error_name], dtype=float).reshape(-1)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        raise ValueError(f'Error array "{error_name}" contains no finite samples: {path}')

    return values


def compute_statistic(values: np.ndarray, statistic: str) -> float:
    '''Calculate one requested summary statistic.'''

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    if statistic == 'median':
        return float(np.median(values))

    if statistic == 'mean':
        return float(np.mean(values))

    if statistic == 'rmse':
        return float(np.sqrt(np.mean(values**2)))

    if statistic.startswith('p'):
        percentile = float(statistic[1:])
        return float(np.percentile(values, percentile))

    raise ValueError(f'Unknown statistic: {statistic}')


def values_match(value_a: Any, value_b: Any) -> bool:
    '''Compare simple run-configuration values robustly.'''

    if isinstance(value_a, (int, float, np.integer, np.floating)) and isinstance(value_b, (int, float, np.integer, np.floating)):
        return bool(np.isclose(float(value_a), float(value_b), rtol=1e-10, atol=1e-12, equal_nan=True))

    return value_a == value_b


def compare_run_configs(with_payload: dict[str, Any], without_payload: dict[str, Any]) -> list[str]:
    '''Return important configuration mismatches between paired runs.'''

    with_config = with_payload.get('run_config', {})
    without_config = without_payload.get('run_config', {})

    if not isinstance(with_config, dict) or not isinstance(without_config, dict):
        return []

    mismatches = []

    for key in RUN_CONFIG_COMPARE_KEYS:
        if key not in with_config or key not in without_config:
            continue

        if not values_match(with_config[key], without_config[key]):
            mismatches.append(f'{key}: with={with_config[key]!r}, without={without_config[key]!r}')

    return mismatches


def validate_result_payload(payload: dict[str, Any], *, path: Path, dataset_name: str, calibration_mode: str, lidar_mode: str) -> None:
    '''Validate one pipeline result pickle against its expected location.'''

    expected_payload_mode = calibration_mode.replace('_', '-')

    stored_dataset_name = payload.get('dataset_name')

    if stored_dataset_name is not None and str(stored_dataset_name) != dataset_name:
        warnings.warn(f'Dataset name mismatch in {path}: directory={dataset_name!r}, pickle={stored_dataset_name!r}')

    stored_mode = payload.get('mode')

    if stored_mode is not None and str(stored_mode) not in {calibration_mode, expected_payload_mode}:
        warnings.warn(f'Calibration mode mismatch in {path}: expected {calibration_mode!r}, pickle={stored_mode!r}')

    stored_lidar_mode = payload.get('lidar_mode')

    if stored_lidar_mode is not None and str(stored_lidar_mode) != lidar_mode:
        raise ValueError(f'LiDAR mode mismatch in {path}: expected {lidar_mode!r}, pickle={stored_lidar_mode!r}')

    for error_name in ERROR_TYPES:
        finite_error_array(payload, error_name, path)


##################################################
# Per-dataset metrics
##################################################


def result_to_metric_row(dataset_name: str, payload: dict[str, Any], path: Path, statistics: tuple[str, ...]) -> dict[str, Any]:
    '''Reduce one trajectory-error pickle to one paper-style dataset row.'''

    row: dict[str, Any] = {
        'dataset': dataset_name,
    }

    timestamps = np.asarray(payload.get('timestamps_s', []), dtype=float).reshape(-1)
    finite_timestamps = timestamps[np.isfinite(timestamps)]

    error_arrays = {error_name: finite_error_array(payload, error_name, path) for error_name in ERROR_TYPES}

    sample_counts = [len(values) for values in error_arrays.values()]
    row['samples'] = int(min(sample_counts))

    if len(finite_timestamps) >= 2:
        row['duration_s'] = float(finite_timestamps[-1] - finite_timestamps[0])
    else:
        row['duration_s'] = np.nan

    for error_name, values in error_arrays.items():
        for statistic in statistics:
            row[f'{error_name}_{statistic}'] = compute_statistic(values, statistic)

    return row


##################################################
# Dataset discovery
##################################################


def dataset_result_paths(dataset_dir: Path, experiment_dir_name: str, lidar_mode: str, pickle_name: str) -> tuple[Path, Path]:
    '''Construct the calibrated and uncalibrated pickle paths for one dataset.'''

    base_dir = dataset_dir / experiment_dir_name / 'errors' / lidar_mode

    return (
        base_dir / 'with_calibration' / pickle_name,
        base_dir / 'without_calibration' / pickle_name,
    )


def collect_results(dataset_root: Path, experiment_dir_name: str, lidar_mode: str, pickle_name: str, statistics: tuple[str, ...], strict: bool, strict_config: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Load paired calibration results from every direct dataset child directory.'''

    dataset_dirs = sorted((path for path in dataset_root.iterdir() if path.is_dir()), key=lambda path: natural_sort_key(path.name))

    with_rows = []
    without_rows = []

    for dataset_dir in dataset_dirs:
        with_path, without_path = dataset_result_paths(dataset_dir, experiment_dir_name, lidar_mode, pickle_name)

        if not with_path.is_file() and not without_path.is_file():
            continue

        if not with_path.is_file() or not without_path.is_file():
            missing_paths = [str(path) for path in (with_path, without_path) if not path.is_file()]
            message = f'Skipping dataset {dataset_dir.name}: incomplete calibration pair. Missing: {", ".join(missing_paths)}'

            if strict:
                raise FileNotFoundError(message)

            warnings.warn(message)
            continue

        with_payload = load_result_pickle(with_path)
        without_payload = load_result_pickle(without_path)

        validate_result_payload(with_payload, path=with_path, dataset_name=dataset_dir.name, calibration_mode='with_calibration', lidar_mode=lidar_mode)
        validate_result_payload(without_payload, path=without_path, dataset_name=dataset_dir.name, calibration_mode='without_calibration', lidar_mode=lidar_mode)

        config_mismatches = compare_run_configs(with_payload, without_payload)

        if config_mismatches:
            mismatch_text = '; '.join(config_mismatches)
            message = f'Run-configuration mismatch for {dataset_dir.name}: {mismatch_text}'

            if strict_config:
                raise ValueError(message)

            warnings.warn(message)

        with_row = result_to_metric_row(dataset_dir.name, with_payload, with_path, statistics)
        without_row = result_to_metric_row(dataset_dir.name, without_payload, without_path, statistics)

        with_rows.append(with_row)
        without_rows.append(without_row)

    if not with_rows:
        raise RuntimeError(f'No complete with/without-calibration result pairs were found under {dataset_root} for LiDAR mode "{lidar_mode}".')

    with_dataframe = pd.DataFrame(with_rows)
    without_dataframe = pd.DataFrame(without_rows)

    with_dataframe = with_dataframe.sort_values('dataset', key=lambda series: series.map(natural_sort_key)).reset_index(drop=True)
    without_dataframe = without_dataframe.sort_values('dataset', key=lambda series: series.map(natural_sort_key)).reset_index(drop=True)

    if with_dataframe['dataset'].tolist() != without_dataframe['dataset'].tolist():
        raise RuntimeError('Internal error: calibrated and uncalibrated dataset ordering differs.')

    return with_dataframe, without_dataframe


##################################################
# Calibration-gain tables
##################################################


def build_gain_tables(with_dataframe: pd.DataFrame, without_dataframe: pd.DataFrame, statistics: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''Build wide and long per-dataset calibration gain tables.'''

    datasets = with_dataframe['dataset'].tolist()

    wide_rows = []
    long_rows = []

    for row_index, dataset_name in enumerate(datasets):
        wide_row: dict[str, Any] = {
            'dataset': dataset_name,
        }

        for error_name in ERROR_TYPES:
            for statistic in statistics:
                metric_name = f'{error_name}_{statistic}'

                with_value = float(with_dataframe.loc[row_index, metric_name])
                without_value = float(without_dataframe.loc[row_index, metric_name])

                absolute_gain = without_value - with_value
                relative_gain_percent = 100.0 * absolute_gain / without_value if np.isfinite(without_value) and without_value != 0.0 else np.nan

                wide_row[f'{metric_name}_without'] = without_value
                wide_row[f'{metric_name}_with'] = with_value
                wide_row[f'{metric_name}_gain'] = absolute_gain
                wide_row[f'{metric_name}_gain_percent'] = relative_gain_percent

                long_rows.append({
                    'dataset': dataset_name,
                    'error_type': error_name,
                    'statistic': statistic,
                    'without_calibration': without_value,
                    'with_calibration': with_value,
                    'absolute_gain': absolute_gain,
                    'relative_gain_percent': relative_gain_percent,
                    'improved': bool(absolute_gain > 0.0),
                })

        wide_rows.append(wide_row)

    return pd.DataFrame(wide_rows), pd.DataFrame(long_rows)


def build_gain_summary(gain_long_dataframe: pd.DataFrame, statistics: tuple[str, ...]) -> pd.DataFrame:
    '''Average calibration gains across datasets while preserving sequence-level weighting.'''

    rows = []

    for error_name in ERROR_TYPES:
        for statistic in statistics:
            selected = gain_long_dataframe[(gain_long_dataframe['error_type'] == error_name) & (gain_long_dataframe['statistic'] == statistic)].copy()

            without_values = selected['without_calibration'].to_numpy(dtype=float)
            with_values = selected['with_calibration'].to_numpy(dtype=float)
            absolute_gains = selected['absolute_gain'].to_numpy(dtype=float)
            relative_gains = selected['relative_gain_percent'].to_numpy(dtype=float)

            finite_relative_gains = relative_gains[np.isfinite(relative_gains)]

            improved_count = int(np.sum(absolute_gains > 0.0))
            degraded_count = int(np.sum(absolute_gains < 0.0))
            unchanged_count = int(np.sum(np.isclose(absolute_gains, 0.0)))

            rows.append({
                'error_type': error_name,
                'statistic': statistic,
                'datasets': int(len(selected)),
                'without_calibration_mean': float(np.mean(without_values)),
                'without_calibration_std': float(np.std(without_values, ddof=1)) if len(without_values) > 1 else 0.0,
                'with_calibration_mean': float(np.mean(with_values)),
                'with_calibration_std': float(np.std(with_values, ddof=1)) if len(with_values) > 1 else 0.0,
                'absolute_gain_mean': float(np.mean(absolute_gains)),
                'absolute_gain_median': float(np.median(absolute_gains)),
                'relative_gain_percent_mean': float(np.mean(finite_relative_gains)) if len(finite_relative_gains) else np.nan,
                'relative_gain_percent_median': float(np.median(finite_relative_gains)) if len(finite_relative_gains) else np.nan,
                'relative_gain_percent_std': float(np.std(finite_relative_gains, ddof=1)) if len(finite_relative_gains) > 1 else 0.0,
                'improved_datasets': improved_count,
                'degraded_datasets': degraded_count,
                'unchanged_datasets': unchanged_count,
                'improved_fraction': improved_count / len(selected) if len(selected) else np.nan,
            })

    return pd.DataFrame(rows)


##################################################
# Plotting
##################################################


def plot_metric_comparison(with_dataframe: pd.DataFrame, without_dataframe: pd.DataFrame, statistic: str, output_path: Path, lidar_mode: str, dpi: int) -> None:
    '''Plot calibrated and uncalibrated values for every dataset and error type.'''

    datasets = with_dataframe['dataset'].tolist()
    x = np.arange(len(datasets), dtype=float)
    width = 0.38

    fig, axes = plt.subplots(len(ERROR_TYPES), 1, figsize=(max(10.0, 1.25 * len(datasets)), 11.0), sharex=True)

    if len(ERROR_TYPES) == 1:
        axes = np.asarray([axes])

    for axis, (error_name, error_info) in zip(axes, ERROR_TYPES.items()):
        metric_name = f'{error_name}_{statistic}'

        without_values = without_dataframe[metric_name].to_numpy(dtype=float)
        with_values = with_dataframe[metric_name].to_numpy(dtype=float)

        axis.bar(x - width / 2.0, without_values, width=width, label='Without calibration')
        axis.bar(x + width / 2.0, with_values, width=width, label='With calibration')

        axis.set_ylabel(error_info['ylabel'])
        axis.set_title(f"{error_info['title']}: {statistic.upper()}")
        axis.grid(True, axis='y', alpha=0.25)
        axis.legend()

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(datasets, rotation=45, ha='right')
    axes[-1].set_xlabel('KAIST sequence')

    fig.suptitle(f'Calibration comparison, LiDAR mode: {lidar_mode}')
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_metric_gain(gain_long_dataframe: pd.DataFrame, statistic: str, output_path: Path, lidar_mode: str, dpi: int) -> None:
    '''Plot per-dataset percentage gain produced by calibration.'''

    datasets = list(dict.fromkeys(gain_long_dataframe['dataset'].tolist()))
    x = np.arange(len(datasets), dtype=float)

    fig, axes = plt.subplots(len(ERROR_TYPES), 1, figsize=(max(10.0, 1.25 * len(datasets)), 11.0), sharex=True)

    if len(ERROR_TYPES) == 1:
        axes = np.asarray([axes])

    for axis, (error_name, error_info) in zip(axes, ERROR_TYPES.items()):
        selected = gain_long_dataframe[(gain_long_dataframe['error_type'] == error_name) & (gain_long_dataframe['statistic'] == statistic)].set_index('dataset')
        gain_values = np.asarray([selected.loc[dataset_name, 'relative_gain_percent'] for dataset_name in datasets], dtype=float)

        axis.bar(x, gain_values)
        axis.axhline(0.0, linewidth=1.0)

        axis.set_ylabel('gain [%]')
        axis.set_title(f"{error_info['title']}: {statistic.upper()}")
        axis.grid(True, axis='y', alpha=0.25)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(datasets, rotation=45, ha='right')
    axes[-1].set_xlabel('KAIST sequence')

    fig.suptitle(f'Calibration gain, LiDAR mode: {lidar_mode}\nPositive values mean lower error with calibration')
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def plot_average_gain(gain_summary_dataframe: pd.DataFrame, statistic: str, output_path: Path, lidar_mode: str, dpi: int) -> None:
    '''Plot sequence-balanced mean percentage calibration gain for all error types.'''

    selected = gain_summary_dataframe[gain_summary_dataframe['statistic'] == statistic].set_index('error_type')

    error_names = list(ERROR_TYPES.keys())
    labels = ['Full SE(3)', 'Rotation', 'Translation']
    mean_gains = np.asarray([selected.loc[error_name, 'relative_gain_percent_mean'] for error_name in error_names], dtype=float)
    std_gains = np.asarray([selected.loc[error_name, 'relative_gain_percent_std'] for error_name in error_names], dtype=float)

    x = np.arange(len(error_names), dtype=float)

    fig, axis = plt.subplots(figsize=(8.5, 5.5))

    axis.bar(x, mean_gains, yerr=std_gains, capsize=5)
    axis.axhline(0.0, linewidth=1.0)
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylabel('mean per-sequence gain [%]')
    axis.set_title(f'Average calibration gain: {statistic.upper()}, LiDAR mode: {lidar_mode}')
    axis.grid(True, axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


##################################################
# CSV formatting
##################################################


def ordered_metric_columns(statistics: tuple[str, ...]) -> list[str]:
    '''Return paper-friendly metric column ordering.'''

    columns = ['dataset', 'samples', 'duration_s']

    for error_name in ERROR_TYPES:
        for statistic in statistics:
            columns.append(f'{error_name}_{statistic}')

    return columns


def save_tables(with_dataframe: pd.DataFrame, without_dataframe: pd.DataFrame, gain_wide_dataframe: pd.DataFrame, gain_long_dataframe: pd.DataFrame, gain_summary_dataframe: pd.DataFrame, output_dir: Path, statistics: tuple[str, ...]) -> dict[str, Path]:
    '''Save all result tables.'''

    output_dir.mkdir(parents=True, exist_ok=True)

    metric_columns = ordered_metric_columns(statistics)

    with_path = output_dir / 'with_calibration_metrics.csv'
    without_path = output_dir / 'without_calibration_metrics.csv'
    gain_wide_path = output_dir / 'calibration_gains.csv'
    gain_long_path = output_dir / 'calibration_gains_long.csv'
    gain_summary_path = output_dir / 'calibration_gain_summary.csv'

    with_dataframe[metric_columns].to_csv(with_path, index=False, float_format='%.10g')
    without_dataframe[metric_columns].to_csv(without_path, index=False, float_format='%.10g')
    gain_wide_dataframe.to_csv(gain_wide_path, index=False, float_format='%.10g')
    gain_long_dataframe.to_csv(gain_long_path, index=False, float_format='%.10g')
    gain_summary_dataframe.to_csv(gain_summary_path, index=False, float_format='%.10g')

    return {
        'with_calibration': with_path,
        'without_calibration': without_path,
        'gain_wide': gain_wide_path,
        'gain_long': gain_long_path,
        'gain_summary': gain_summary_path,
    }


##################################################
# Console summary
##################################################


def print_compact_summary(gain_summary_dataframe: pd.DataFrame, statistic: str) -> None:
    '''Print the most useful aggregate comparison to the console.'''

    selected = gain_summary_dataframe[gain_summary_dataframe['statistic'] == statistic]

    print()
    print(f'Calibration comparison using {statistic.upper()}')
    print('========================================')

    for _, row in selected.iterrows():
        error_name = str(row['error_type'])
        title = ERROR_TYPES[error_name]['title']

        print()
        print(title)
        print(f"  datasets:                    {int(row['datasets'])}")
        print(f"  without calibration mean:    {row['without_calibration_mean']:.6g}")
        print(f"  with calibration mean:       {row['with_calibration_mean']:.6g}")
        print(f"  mean absolute gain:           {row['absolute_gain_mean']:.6g}")
        print(f"  mean relative gain:           {row['relative_gain_percent_mean']:.3f}%")
        print(f"  median relative gain:         {row['relative_gain_percent_median']:.3f}%")
        print(f"  improved datasets:            {int(row['improved_datasets'])}/{int(row['datasets'])}")


##################################################
# Argument parser
##################################################


def build_argument_parser() -> argparse.ArgumentParser:
    '''Construct the command-line interface.'''

    parser = argparse.ArgumentParser(description='Aggregate KAIST pipeline trajectory-error pickles across multiple dataset directories and compare calibration against fixed parameters.')

    parser.add_argument('dataset_root', type=Path, help='Directory whose direct child directories are datasets such as Urban13, Urban14, Urban15, Urban16, and Urban17.')
    parser.add_argument('--lidar-mode', choices=('poses', 'odometry'), default='poses', help='LiDAR result mode to process.')
    parser.add_argument('--experiment-dir-name', type=str, default='experiment_results', help='Experiment-results directory name inside every dataset.')
    parser.add_argument('--pickle-name', type=str, default='trajectory_errors.pkl', help='Result pickle filename.')
    parser.add_argument('--output-dir', type=Path, default=None, help='Summary output directory. Default: <dataset_root>/pipeline_results_summary/<lidar_mode>. Relative paths are interpreted relative to dataset_root.')
    parser.add_argument('--statistics', nargs='+', choices=AVAILABLE_STATISTICS, default=['median', 'rmse', 'p90'], help='Per-sequence statistics included in the tables and plots.')
    parser.add_argument('--strict', action=argparse.BooleanOptionalAction, default=False, help='Fail instead of skipping datasets with only one calibration mode available.')
    parser.add_argument('--strict-config', action=argparse.BooleanOptionalAction, default=False, help='Fail when important run parameters differ between calibrated and uncalibrated runs.')
    parser.add_argument('--plots', action=argparse.BooleanOptionalAction, default=True, help='Generate comparison and gain plots.')
    parser.add_argument('--dpi', type=int, default=180, help='Saved plot resolution.')
    parser.add_argument('--console-statistic', choices=AVAILABLE_STATISTICS, default='rmse', help='Statistic used for the compact final console summary.')

    return parser


##################################################
# Main processing
##################################################


def run(args: argparse.Namespace) -> None:
    '''Aggregate all direct dataset directories and save tables and plots.'''

    dataset_root = args.dataset_root.expanduser().resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f'Dataset root does not exist: {dataset_root}')

    statistics = tuple(dict.fromkeys(args.statistics))

    output_dir = resolve_output_directory(dataset_root, args.output_dir, args.lidar_mode)
    plots_dir = output_dir / 'plots'

    with_dataframe, without_dataframe = collect_results(
        dataset_root=dataset_root,
        experiment_dir_name=args.experiment_dir_name,
        lidar_mode=args.lidar_mode,
        pickle_name=args.pickle_name,
        statistics=statistics,
        strict=args.strict,
        strict_config=args.strict_config,
    )

    gain_wide_dataframe, gain_long_dataframe = build_gain_tables(with_dataframe, without_dataframe, statistics)
    gain_summary_dataframe = build_gain_summary(gain_long_dataframe, statistics)

    table_paths = save_tables(
        with_dataframe=with_dataframe,
        without_dataframe=without_dataframe,
        gain_wide_dataframe=gain_wide_dataframe,
        gain_long_dataframe=gain_long_dataframe,
        gain_summary_dataframe=gain_summary_dataframe,
        output_dir=output_dir,
        statistics=statistics,
    )

    if args.plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

        for statistic in statistics:
            plot_metric_comparison(with_dataframe, without_dataframe, statistic, plots_dir / f'comparison_{statistic}.png', args.lidar_mode, args.dpi)
            plot_metric_gain(gain_long_dataframe, statistic, plots_dir / f'gain_{statistic}.png', args.lidar_mode, args.dpi)
            plot_average_gain(gain_summary_dataframe, statistic, plots_dir / f'average_gain_{statistic}.png', args.lidar_mode, args.dpi)

    console_statistic = args.console_statistic

    if console_statistic not in statistics:
        console_statistic = 'rmse' if 'rmse' in statistics else statistics[0]

    print()
    print('Processed datasets')
    print('==================')
    print(', '.join(with_dataframe['dataset'].tolist()))

    print_compact_summary(gain_summary_dataframe, console_statistic)

    print()
    print('Saved tables')
    print('============')

    for path in table_paths.values():
        print(path)

    if args.plots:
        print()
        print('Saved plots')
        print('===========')
        print(plots_dir)


##################################################
# Entrypoint
##################################################


def main() -> None:
    '''Parse CLI arguments and process all datasets.'''

    parser = build_argument_parser()
    args = parser.parse_args()

    if args.dpi <= 0:
        parser.error('--dpi must be positive.')

    run(args)


if __name__ == '__main__':
    main()

# python process_kaist_pipeline_results.py /mnt/d/Downloads/MobRobLab/KAISTDataset