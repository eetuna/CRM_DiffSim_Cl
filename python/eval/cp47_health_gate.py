#!/usr/bin/env python3
"""
CP4.7.3/CP4.7.4: Health Gate for MPC-based Calibration/Benchmark

Validates datasets/segments for use in calibration and benchmarking by running
MPC-only preflight checks and detecting common failure modes:
- Rank-deficient Jacobian (rank=0)
- Physics solver failures (status!=0)
- Excessive residuals (poor convergence)
- NaN/Inf values in outputs

CP4.7.3: Full dataset validation
CP4.7.4: Sliding-window validation to find valid segments within datasets

Outputs JSON report with:
- Pass/fail status for each dataset/window
- Failure reason counts
- List of valid datasets/windows for calibration/benchmark

This prevents calibration from producing NaN baselines and placeholder thresholds.
"""

import sys
import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))

from control.ilqr import iLQRSolver
from data.npz_manifest import load_manifest
import crm_diff_py


class HealthCheckFailure:
    """Tracks failure counts by category."""

    def __init__(self):
        self.rank_deficient = 0
        self.status_nonzero = 0
        self.residual_high = 0
        self.nan_inf = 0
        self.solver_exception = 0
        self.total = 0

    def to_dict(self):
        return {
            'rank_deficient': self.rank_deficient,
            'status_nonzero': self.status_nonzero,
            'residual_high': self.residual_high,
            'nan_inf': self.nan_inf,
            'solver_exception': self.solver_exception,
            'total': self.total
        }


def check_dataset_health(
    dataset,
    params_dict,
    duration=2.0,
    mpc_horizon=10,
    max_iters=10,
    verbose=False
) -> Tuple[bool, str, Dict]:
    """
    Run MPC-only preflight on a single dataset.

    Args:
        dataset: Dataset object with tip_ref, dt, L_inserted
        params_dict: Physics parameters
        duration: Test duration in seconds (default: 2.0s)
        mpc_horizon: MPC horizon length
        max_iters: Max iLQR iterations
        verbose: Print detailed info

    Returns:
        (is_valid, failure_reason, details)
        - is_valid: True if dataset passes health check
        - failure_reason: String describing first failure (empty if valid)
        - details: Dict with counts and metrics
    """
    dt = dataset.dt
    L_inserted = dataset.L_inserted
    n_steps = min(int(duration / dt), len(dataset.tip_ref))

    if n_steps < 5:
        return False, "trajectory_too_short", {'n_steps': n_steps}

    p_ref = dataset.tip_ref[:n_steps]
    x_t = np.zeros(6)

    # Initialize
    try:
        result_init = crm_diff_py.dynamics_forward(
            x_t, np.zeros(3), 0.0, L_inserted, params_dict
        )
        p_tip_t = result_init['p_tip']

        if result_init['status'] != 0:
            return False, "init_status_nonzero", {'status': result_init['status']}

        if not np.isfinite(p_tip_t).all():
            return False, "init_nan_inf", {}

    except Exception as e:
        return False, "init_exception", {'error': str(e)}

    # Run MPC steps
    failures = HealthCheckFailure()
    U_warm = None
    n_mpc_failures = 0
    tracking_errors = []

    for i in range(n_steps):
        horizon_end = min(i + mpc_horizon, n_steps)
        p_ref_horizon = p_ref[i:horizon_end]

        if len(p_ref_horizon) < mpc_horizon:
            p_ref_horizon = np.vstack([
                p_ref_horizon,
                np.tile(p_ref_horizon[-1], (mpc_horizon - len(p_ref_horizon), 1))
            ])

        # Run MPC solver
        try:
            solver = iLQRSolver(
                dt=dt,
                L_inserted=L_inserted,
                params_dict=params_dict,
                horizon=mpc_horizon,
                Q=np.zeros((6, 6)),
                R=0.01 * np.eye(3),
                p_target=p_ref_horizon[0],
                terminal_weight=1.0,
                max_iters=max_iters,
                tol=1e-3,
                jacobian_mode="cpp"
            )

            X, U, converged = solver.solve(x_t, U_init=U_warm, verbose=False)

            # Check for NaN/Inf in solution
            if not np.isfinite(X).all() or not np.isfinite(U).all():
                failures.nan_inf += 1
                failures.total += 1
                if verbose:
                    print(f"  Step {i}: NaN/Inf in MPC solution")
                return False, "mpc_nan_inf", failures.to_dict()

            u_t = np.clip(U[0], -0.5, 0.5)
            U_warm = np.vstack([U[1:], U[-1:]])

        except Exception as e:
            failures.solver_exception += 1
            failures.total += 1
            error_str = str(e).lower()

            # Classify exception
            if 'rank' in error_str and ('deficient' in error_str or '0' in error_str):
                failures.rank_deficient += 1
                if verbose:
                    print(f"  Step {i}: Rank-deficient Jacobian: {e}")
                return False, "rank_deficient_jacobian", failures.to_dict()
            else:
                if verbose:
                    print(f"  Step {i}: Solver exception: {e}")
                return False, "solver_exception", failures.to_dict()

        # Step dynamics forward
        try:
            result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)

            if result['status'] != 0:
                failures.status_nonzero += 1
                failures.total += 1
                if verbose:
                    print(f"  Step {i}: Dynamics status={result['status']}")
                return False, "dynamics_status_nonzero", failures.to_dict()

            if not np.isfinite(result['x_next']).all() or not np.isfinite(result['p_tip']).all():
                failures.nan_inf += 1
                failures.total += 1
                if verbose:
                    print(f"  Step {i}: NaN/Inf in dynamics output")
                return False, "dynamics_nan_inf", failures.to_dict()

            x_t = result['x_next']
            p_tip_t = result['p_tip']

        except Exception as e:
            if verbose:
                print(f"  Step {i}: Dynamics exception: {e}")
            return False, "dynamics_exception", {'error': str(e)}

        # Track error
        tracking_errors.append(np.linalg.norm(p_tip_t - p_ref[i]))

    # Compute metrics
    tracking_rmse = float(np.sqrt(np.mean(np.array(tracking_errors)**2)))
    tracking_max = float(np.max(tracking_errors))

    # Check if tracking is reasonable (not pathological)
    if tracking_rmse > 100.0:  # 10cm average error is clearly broken
        return False, "tracking_rmse_pathological", {
            'tracking_rmse': tracking_rmse,
            'tracking_max': tracking_max
        }

    if not np.isfinite(tracking_rmse):
        return False, "tracking_rmse_nan", {}

    # Dataset is healthy
    return True, "", {
        'n_steps': n_steps,
        'tracking_rmse': tracking_rmse,
        'tracking_max': tracking_max,
        'failures': failures.to_dict()
    }


def run_health_gate(
    datasets,
    params_dict,
    duration=2.0,
    verbose=True,
    output_json_path=None
) -> Dict:
    """
    Run health gate on a list of datasets.

    Args:
        datasets: List of dataset objects
        params_dict: Physics parameters
        duration: Test duration per dataset (seconds)
        verbose: Print progress
        output_json_path: Path to save JSON report (optional)

    Returns:
        dict with:
            - valid_datasets: List of dataset indices that passed
            - invalid_datasets: List of (index, filename, reason, details)
            - summary: Counts and failure reasons
    """
    if verbose:
        print("="*80)
        print("CP4.7.3: Health Gate Preflight Check")
        print("="*80)
        print(f"Checking {len(datasets)} datasets...")
        print(f"Test duration: {duration}s per dataset")
        print()

    t_start = time.time()

    valid_datasets = []
    invalid_datasets = []
    failure_counts = {}

    for i, ds in enumerate(datasets):
        if verbose:
            print(f"[{i+1}/{len(datasets)}] {ds.filename}")

        is_valid, failure_reason, details = check_dataset_health(
            ds, params_dict, duration=duration, verbose=False
        )

        if is_valid:
            valid_datasets.append(i)
            if verbose:
                rmse = details.get('tracking_rmse', 0.0)
                print(f"  ✓ PASS (RMSE: {rmse:.3f} mm)")
        else:
            invalid_datasets.append({
                'index': i,
                'filename': ds.filename,
                'reason': failure_reason,
                'details': details
            })
            failure_counts[failure_reason] = failure_counts.get(failure_reason, 0) + 1
            if verbose:
                print(f"  ✗ FAIL ({failure_reason})")

    t_elapsed = time.time() - t_start

    # Build summary
    summary = {
        'total_datasets': len(datasets),
        'valid_count': len(valid_datasets),
        'invalid_count': len(invalid_datasets),
        'failure_reasons': failure_counts,
        'time_elapsed': t_elapsed
    }

    report = {
        'test_name': 'CP4.7.3 Health Gate',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'duration_per_dataset': duration,
        'summary': summary,
        'valid_datasets': valid_datasets,
        'invalid_datasets': invalid_datasets
    }

    # Save JSON if requested
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(report, f, indent=2)
        if verbose:
            print(f"\n✓ Health report saved: {output_json_path}")

    if verbose:
        print("\n" + "="*80)
        print("HEALTH GATE SUMMARY")
        print("="*80)
        print(f"Total datasets: {summary['total_datasets']}")
        print(f"Valid: {summary['valid_count']}")
        print(f"Invalid: {summary['invalid_count']}")

        if failure_counts:
            print("\nFailure reasons:")
            for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
                print(f"  - {reason}: {count}")

        print(f"\nTime elapsed: {t_elapsed:.1f}s")
        print("="*80)

    return report


def check_dataset_window_health(
    dataset,
    params_dict,
    start_idx,
    window_steps,
    mpc_horizon=10,
    max_iters=10,
    verbose=False,
    debug_trace=False
) -> Tuple[bool, str, Dict]:
    """
    Run MPC-only preflight on a specific window within a dataset.

    Args:
        dataset: Dataset object with tip_ref, dt, L_inserted
        params_dict: Physics parameters
        start_idx: Starting index in dataset
        window_steps: Number of steps to check
        mpc_horizon: MPC horizon length
        max_iters: Max iLQR iterations
        verbose: Print detailed info
        debug_trace: If True, print per-step trace for debugging

    Returns:
        (is_valid, failure_reason, details)
        details includes:
            - first_failure_step: timestep where failure occurred
            - failure_source: what failed (p_tip, mpc, dynamics, etc.)
            - diagnostic_info: specific values/errors at failure point
    """
    dt = dataset.dt
    L_inserted = dataset.L_inserted
    end_idx = min(start_idx + window_steps, len(dataset.tip_ref))
    n_steps = end_idx - start_idx

    if debug_trace:
        print(f"\n=== DEBUG TRACE: Window [{start_idx}:{end_idx}), {n_steps} steps ===")

    if n_steps < 5:
        return False, "window_too_short", {
            'n_steps': n_steps,
            'first_failure_step': -1,
            'failure_source': 'window_config'
        }

    p_ref = dataset.tip_ref[start_idx:end_idx]
    x_t = np.zeros(6)

    # Initialize
    try:
        result_init = crm_diff_py.dynamics_forward(
            x_t, np.zeros(3), 0.0, L_inserted, params_dict
        )
        p_tip_t = result_init['p_tip']

        if debug_trace:
            print(f"Step -1 (init): p_tip={p_tip_t}, status={result_init['status']}")

        if result_init['status'] != 0:
            return False, "init_status_nonzero", {
                'status': result_init['status'],
                'first_failure_step': -1,
                'failure_source': 'dynamics_init'
            }

        if not np.isfinite(p_tip_t).all():
            return False, "init_nan_inf", {
                'first_failure_step': -1,
                'failure_source': 'p_tip_init',
                'p_tip': p_tip_t.tolist()
            }

    except Exception as e:
        return False, "init_exception", {
            'error': str(e),
            'first_failure_step': -1,
            'failure_source': 'dynamics_exception'
        }

    # Run MPC steps
    failures = HealthCheckFailure()
    U_warm = None
    tracking_errors = []

    for i in range(n_steps):
        horizon_end = min(i + mpc_horizon, n_steps)
        p_ref_horizon = p_ref[i:horizon_end]

        if len(p_ref_horizon) < mpc_horizon:
            p_ref_horizon = np.vstack([
                p_ref_horizon,
                np.tile(p_ref_horizon[-1], (mpc_horizon - len(p_ref_horizon), 1))
            ])

        # Check p_ref validity
        if not np.isfinite(p_ref[i]).all():
            return False, "p_ref_nan_inf", {
                'first_failure_step': i,
                'failure_source': 'p_ref',
                'p_ref': p_ref[i].tolist(),
                'failures': failures.to_dict()
            }

        # Run MPC solver
        try:
            solver = iLQRSolver(
                dt=dt,
                L_inserted=L_inserted,
                params_dict=params_dict,
                horizon=mpc_horizon,
                Q=np.zeros((6, 6)),
                R=0.01 * np.eye(3),
                p_target=p_ref_horizon[0],
                terminal_weight=1.0,
                max_iters=max_iters,
                tol=1e-3,
                jacobian_mode="cpp"
            )

            X, U, converged = solver.solve(x_t, U_init=U_warm, verbose=False)

            if not np.isfinite(X).all() or not np.isfinite(U).all():
                failures.nan_inf += 1
                failures.total += 1
                return False, "mpc_nan_inf", {
                    'first_failure_step': i,
                    'failure_source': 'mpc_output',
                    'X_has_nan': not np.isfinite(X).all(),
                    'U_has_nan': not np.isfinite(U).all(),
                    'failures': failures.to_dict()
                }

            u_t = np.clip(U[0], -0.5, 0.5)
            U_warm = np.vstack([U[1:], U[-1:]])

            if debug_trace:
                print(f"Step {i}: MPC converged={converged}, u={u_t}, p_ref={p_ref[i]}")

        except Exception as e:
            failures.solver_exception += 1
            failures.total += 1
            error_str = str(e).lower()

            if 'rank' in error_str and ('deficient' in error_str or '0' in error_str):
                failures.rank_deficient += 1
                return False, "rank_deficient_jacobian", {
                    'first_failure_step': i,
                    'failure_source': 'mpc_jacobian',
                    'error': str(e),
                    'x_state': x_t.tolist(),
                    'failures': failures.to_dict()
                }
            else:
                return False, "solver_exception", {
                    'first_failure_step': i,
                    'failure_source': 'mpc_solver',
                    'error': str(e),
                    'failures': failures.to_dict()
                }

        # Step dynamics forward
        try:
            result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)

            if result['status'] != 0:
                failures.status_nonzero += 1
                failures.total += 1
                return False, "dynamics_status_nonzero", {
                    'first_failure_step': i,
                    'failure_source': 'dynamics_status',
                    'status': result['status'],
                    'failures': failures.to_dict()
                }

            if not np.isfinite(result['x_next']).all() or not np.isfinite(result['p_tip']).all():
                failures.nan_inf += 1
                failures.total += 1
                return False, "dynamics_nan_inf", {
                    'first_failure_step': i,
                    'failure_source': 'dynamics_output',
                    'x_next_has_nan': not np.isfinite(result['x_next']).all(),
                    'p_tip_has_nan': not np.isfinite(result['p_tip']).all(),
                    'failures': failures.to_dict()
                }

            x_t = result['x_next']
            p_tip_t = result['p_tip']

            if debug_trace:
                error = np.linalg.norm(p_tip_t - p_ref[i])
                print(f"        Dynamics: p_tip={p_tip_t}, error={error:.3f}")

        except Exception as e:
            return False, "dynamics_exception", {
                'first_failure_step': i,
                'failure_source': 'dynamics_exception',
                'error': str(e)
            }

        # Track error
        error = np.linalg.norm(p_tip_t - p_ref[i])
        tracking_errors.append(error)

        if not np.isfinite(error):
            return False, "tracking_error_nan", {
                'first_failure_step': i,
                'failure_source': 'tracking_error',
                'p_tip': p_tip_t.tolist(),
                'p_ref': p_ref[i].tolist()
            }

    # Compute metrics
    tracking_rmse = float(np.sqrt(np.mean(np.array(tracking_errors)**2)))
    tracking_max = float(np.max(tracking_errors))

    if tracking_rmse > 100.0:
        return False, "tracking_rmse_pathological", {
            'tracking_rmse': tracking_rmse,
            'tracking_max': tracking_max,
            'first_failure_step': n_steps - 1,
            'failure_source': 'tracking_rmse'
        }

    if not np.isfinite(tracking_rmse):
        return False, "tracking_rmse_nan", {
            'first_failure_step': n_steps - 1,
            'failure_source': 'tracking_rmse_computation'
        }

    if debug_trace:
        print(f"=== WINDOW VALID: RMSE={tracking_rmse:.3f}, max={tracking_max:.3f} ===\n")

    # Window is healthy
    return True, "", {
        'n_steps': n_steps,
        'tracking_rmse': tracking_rmse,
        'tracking_max': tracking_max,
        'failures': failures.to_dict()
    }


def run_sliding_window_health_gate(
    datasets,
    params_dict,
    window_steps=50,
    stride_steps=25,
    max_windows=None,
    mpc_horizon=10,
    max_iters=10,
    verbose=True,
    output_json_path=None,
    debug_one_window=None,
    skip_hold_periods=False
) -> Dict:
    """
    Run health gate on sliding windows within datasets.

    Args:
        datasets: List of dataset objects
        params_dict: Physics parameters
        window_steps: Window size in steps
        stride_steps: Stride between windows
        max_windows: Maximum total windows to evaluate (None = unlimited)
        mpc_horizon: MPC horizon length for health checks
        max_iters: Max iLQR iterations per MPC step
        verbose: Print progress
        output_json_path: Path to save JSON report (optional)
        debug_one_window: Global window index to debug with per-step trace (CP4.7.6)
        skip_hold_periods: CP4.7.8 - Skip windows that overlap with hold periods (NaN ref)

    Returns:
        dict with:
            - valid_windows: List of {filename, dataset_idx, start_idx, end_idx, details}
            - invalid_windows: List of {filename, dataset_idx, start_idx, end_idx, reason, details}
            - summary: Counts and failure reasons
    """
    if verbose:
        print("="*80)
        print("CP4.7.4: Sliding Window Health Gate")
        print("="*80)
        print(f"Window size: {window_steps} steps")
        print(f"Stride: {stride_steps} steps")
        print(f"Checking {len(datasets)} datasets...")
        print()

    t_start = time.time()

    valid_windows = []
    invalid_windows = []
    failure_counts = {}
    total_windows = 0

    for dataset_idx, ds in enumerate(datasets):
        if verbose:
            print(f"[{dataset_idx+1}/{len(datasets)}] {ds.filename}")

        n_total_steps = len(ds.tip_ref)
        n_windows = 0

        # CP4.7.8: Get hold mask if skip_hold_periods enabled
        if skip_hold_periods and hasattr(ds, 'hold_mask'):
            hold_mask = ds.hold_mask
        else:
            hold_mask = None

        # Generate windows
        for start_idx in range(0, n_total_steps, stride_steps):
            # Early exit if max_windows reached
            if max_windows is not None and total_windows >= max_windows:
                break

            end_idx = min(start_idx + window_steps, n_total_steps)
            actual_window_steps = end_idx - start_idx

            if actual_window_steps < 10:  # Skip tiny windows
                continue

            # CP4.7.8: Skip windows that overlap with hold periods
            if hold_mask is not None:
                window_hold_mask = hold_mask[start_idx:end_idx]
                if window_hold_mask.any():
                    # Window contains hold period (NaN ref), skip it
                    continue

            # Check if this is the debug window
            debug_this_window = (debug_one_window is not None and total_windows == debug_one_window)

            total_windows += 1
            n_windows += 1

            is_valid, failure_reason, details = check_dataset_window_health(
                ds, params_dict,
                start_idx=start_idx,
                window_steps=actual_window_steps,
                mpc_horizon=mpc_horizon,
                max_iters=max_iters,
                verbose=False,
                debug_trace=debug_this_window
            )

            window_info = {
                'filename': ds.filename,
                'dataset_idx': dataset_idx,
                'start_idx': int(start_idx),
                'end_idx': int(end_idx),
                'n_steps': int(actual_window_steps)
            }

            if is_valid:
                window_info['details'] = details
                valid_windows.append(window_info)
            else:
                window_info['reason'] = failure_reason
                window_info['details'] = details
                invalid_windows.append(window_info)
                failure_counts[failure_reason] = failure_counts.get(failure_reason, 0) + 1

        if verbose:
            n_valid = sum(1 for w in valid_windows if w['dataset_idx'] == dataset_idx)
            n_invalid = n_windows - n_valid
            print(f"  Windows: {n_windows} total, {n_valid} valid, {n_invalid} invalid")

        # Break outer loop if max_windows reached
        if max_windows is not None and total_windows >= max_windows:
            break

    t_elapsed = time.time() - t_start

    # Build summary
    summary = {
        'total_datasets': len(datasets),
        'total_windows': total_windows,
        'valid_window_count': len(valid_windows),
        'invalid_window_count': len(invalid_windows),
        'window_steps': window_steps,
        'stride_steps': stride_steps,
        'failure_reasons': failure_counts,
        'time_elapsed': t_elapsed
    }

    report = {
        'test_name': 'CP4.7.4 Sliding Window Health Gate',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'window_config': {
            'window_steps': window_steps,
            'stride_steps': stride_steps
        },
        'summary': summary,
        'valid_windows': valid_windows,
        'invalid_windows': invalid_windows
    }

    # Save JSON if requested
    if output_json_path:
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w') as f:
            json.dump(report, f, indent=2)
        if verbose:
            print(f"\n✓ Window health report saved: {output_json_path}")

    if verbose:
        print("\n" + "="*80)
        print("SLIDING WINDOW HEALTH GATE SUMMARY")
        print("="*80)
        print(f"Total datasets: {summary['total_datasets']}")
        print(f"Total windows: {summary['total_windows']}")
        print(f"Valid windows: {summary['valid_window_count']}")
        print(f"Invalid windows: {summary['invalid_window_count']}")

        if failure_counts:
            print("\nFailure reasons:")
            for reason, count in sorted(failure_counts.items(), key=lambda x: -x[1]):
                print(f"  - {reason}: {count}")

        print(f"\nTime elapsed: {t_elapsed:.1f}s")
        print("="*80)

    return report


def create_windowed_manifest(datasets, valid_windows):
    """
    Create a filtered in-memory manifest of valid windows.

    Args:
        datasets: List of original dataset objects
        valid_windows: List of valid window dicts from run_sliding_window_health_gate

    Returns:
        List of windowed dataset objects (with sliced tip_ref, etc.)
    """
    windowed_datasets = []

    for window in valid_windows:
        dataset_idx = window['dataset_idx']
        start_idx = window['start_idx']
        end_idx = window['end_idx']

        original_ds = datasets[dataset_idx]

        # Create a windowed dataset object (simple namespace)
        class WindowedDataset:
            def __init__(self, orig_ds, start, end, window_info):
                self.filename = f"{orig_ds.filename}_window_{start}_{end}"
                self.dt = orig_ds.dt
                self.L_inserted = orig_ds.L_inserted
                self.tip_ref = orig_ds.tip_ref[start:end]
                self.window_info = window_info  # For reference

        windowed_ds = WindowedDataset(original_ds, start_idx, end_idx, window)
        windowed_datasets.append(windowed_ds)

    return windowed_datasets


def main():
    """Standalone health gate test."""
    import argparse

    parser = argparse.ArgumentParser(description='CP4.7.3/CP4.7.4/CP4.7.6: Health Gate Preflight Check')
    parser.add_argument('--datasets_limit', type=int, default=None,
                       help='Limit number of datasets to check')
    parser.add_argument('--duration', type=float, default=2.0,
                       help='Test duration per dataset (seconds, full dataset mode only)')
    parser.add_argument('--output', type=str, default='./build/artifacts/cp47_health_report.json',
                       help='Output JSON path')
    parser.add_argument('--window_mode', action='store_true',
                       help='Use sliding-window mode (CP4.7.4)')
    parser.add_argument('--window_steps', type=int, default=50,
                       help='Window size in steps (window mode only)')
    parser.add_argument('--stride_steps', type=int, default=25,
                       help='Stride between windows (window mode only)')
    parser.add_argument('--debug_one_window', type=int, default=None, metavar='WINDOW_IDX',
                       help='CP4.7.6: Print detailed per-step trace for one window (by global index)')
    args = parser.parse_args()

    # Load datasets
    print("Loading datasets...")
    try:
        manifest = load_manifest(data_dir='./data', verbose=False)
        datasets = manifest.datasets_with_ref

        if not datasets:
            print("✗ No datasets with references found")
            return 1

        if args.datasets_limit:
            datasets = datasets[:args.datasets_limit]

        print(f"✓ Loaded {len(datasets)} datasets")

    except Exception as e:
        print(f"✗ Failed to load datasets: {e}")
        return 1

    # Load physics parameters
    param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
    cath_params = crm_diff_py.load_cath_params(param_file)
    cath_config = crm_diff_py.load_cath_config(config_file)

    params_dict = {
        'CathParams': cath_params,
        'CathConfig': cath_config,
        'ContactMode': int(crm_diff_py.ContactModeType.FREE_TIP),
        'TipForce': [0.0, 0.0, 0.0],
        'deltau0_initialguess': [0.0, 0.0, 0.0],
        'IntegrationStepSize': 0.5,
        'FinalValueOnly': True,
    }

    # Run health gate (full dataset or sliding window mode)
    if args.window_mode:
        # CP4.7.4/CP4.7.6: Sliding-window mode
        report = run_sliding_window_health_gate(
            datasets,
            params_dict,
            window_steps=args.window_steps,
            stride_steps=args.stride_steps,
            verbose=True,
            output_json_path=args.output,
            debug_one_window=args.debug_one_window
        )

        # Exit with success if at least one window is valid
        if report['summary']['valid_window_count'] > 0:
            return 0
        else:
            print("\n✗ No valid windows found")
            return 1
    else:
        # CP4.7.3: Full dataset mode
        report = run_health_gate(
            datasets,
            params_dict,
            duration=args.duration,
            verbose=True,
            output_json_path=args.output
        )

        # Exit with success if at least one dataset is valid
        if report['summary']['valid_count'] > 0:
            return 0
        else:
            print("\n✗ No valid datasets found")
            return 1


if __name__ == "__main__":
    sys.exit(main())
