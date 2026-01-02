#!/usr/bin/env python3
"""
CP4.7.1: Threshold Calibration for Hybrid Controller

Sweeps (tau_low, tau_high) grid to find optimal thresholds that achieve:
- MPC call rate ≤ 30%
- Tracking RMSE ≤ MPC-only + 0.5 mm
- Speedup ≥ 2× vs MPC-only
- Zero safety violations

Requires trained ensemble from CP4.5.
"""

import sys
import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from control.hybrid_controller import HybridController, HybridControllerMetrics
from control.ilqr import iLQRSolver
from models.ensemble_policy import EnsemblePolicy
from models.recurrent_policy import GRUPolicy
from data.npz_manifest import load_manifest
import crm_diff_py


def load_ensemble_from_cp45() -> Optional[EnsemblePolicy]:
    """
    Load trained ensemble from CP4.5 artifacts.

    Returns:
        EnsemblePolicy or None if not found
    """
    metadata_path = './build/artifacts/cp45_ensemble_dagger_ensemble_metadata.json'

    if not os.path.exists(metadata_path):
        print(f"✗ Ensemble metadata not found: {metadata_path}")
        print("  Run: python3 python/train_cp45_ensemble_dagger.py first")
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Check all checkpoints exist
    missing = [p for p in metadata['checkpoint_paths'] if not os.path.exists(p)]
    if missing:
        print(f"✗ Missing checkpoints: {missing}")
        return None

    ensemble = EnsemblePolicy(
        policy_paths=metadata['checkpoint_paths'],
        policy_class=GRUPolicy,
        policy_kwargs=metadata['policy_kwargs']
    )

    print(f"✓ Loaded ensemble: {metadata['n_members']} members")
    return ensemble


def run_mpc_baseline(dataset, params_dict, duration=5.0, mpc_horizon=10):
    """Run MPC-only baseline for comparison."""
    dt = dataset.dt
    L_inserted = dataset.L_inserted
    n_steps = min(int(duration / dt), len(dataset.tip_ref))

    p_ref = dataset.tip_ref[:n_steps]
    x_t = np.zeros(6)

    result_init = crm_diff_py.dynamics_forward(
        x_t, np.zeros(3), 0.0, L_inserted, params_dict
    )
    p_tip_t = result_init['p_tip']

    p_tips = []
    mpc_times = []
    n_failures = 0

    U_warm = None
    t_start_total = time.time()

    for i in range(n_steps):
        horizon_end = min(i + mpc_horizon, n_steps)
        p_ref_horizon = p_ref[i:horizon_end]

        if len(p_ref_horizon) < mpc_horizon:
            p_ref_horizon = np.vstack([
                p_ref_horizon,
                np.tile(p_ref_horizon[-1], (mpc_horizon - len(p_ref_horizon), 1))
            ])

        t_mpc_start = time.time()

        solver = iLQRSolver(
            dt=dt,
            L_inserted=L_inserted,
            params_dict=params_dict,
            horizon=mpc_horizon,
            Q=np.zeros((6, 6)),
            R=0.01 * np.eye(3),
            p_target=p_ref_horizon[0],
            terminal_weight=1.0,
            max_iters=10,
            tol=1e-3,
            jacobian_mode="cpp"
        )

        try:
            X, U, converged = solver.solve(x_t, U_init=U_warm, verbose=False)
            u_t = np.clip(U[0], -0.5, 0.5)
            U_warm = np.vstack([U[1:], U[-1:]])
        except:
            u_t = np.zeros(3)
            U_warm = None
            n_failures += 1

        mpc_times.append((time.time() - t_mpc_start) * 1000.0)

        result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)

        if result['status'] == 0:
            x_t = result['x_next']
            p_tip_t = result['p_tip']
        else:
            n_failures += 1

        p_tips.append(p_tip_t)

    total_time = time.time() - t_start_total
    p_tips = np.array(p_tips)
    tracking_errors = np.linalg.norm(p_tips - p_ref[:n_steps], axis=1)

    return {
        'tracking_rmse': float(np.sqrt(np.mean(tracking_errors**2))),
        'tracking_max': float(np.max(tracking_errors)),
        'total_time': total_time,
        'mean_mpc_time_ms': float(np.mean(mpc_times)),
        'n_failures': n_failures
    }


def run_hybrid_with_thresholds(
    dataset,
    ensemble,
    params_dict,
    tau_low,
    tau_high,
    duration=5.0,
    mpc_horizon=10
):
    """Run hybrid controller with given thresholds."""
    dt = dataset.dt
    L_inserted = dataset.L_inserted
    n_steps = min(int(duration / dt), len(dataset.tip_ref))

    controller = HybridController(
        ensemble=ensemble,
        dt=dt,
        L_inserted=L_inserted,
        params_dict=params_dict,
        tau_low=tau_low,
        tau_high=tau_high,
        tracking_safety_limit=5.0,
        mpc_horizon=mpc_horizon,
        mpc_max_iters=10,
        mpc_tol=1e-3
    )

    metrics = HybridControllerMetrics()
    p_ref = dataset.tip_ref[:n_steps]
    x_t = np.zeros(6)

    result_init = crm_diff_py.dynamics_forward(
        x_t, np.zeros(3), 0.0, L_inserted, params_dict
    )
    p_tip_t = result_init['p_tip']

    p_tips = []
    n_failures = 0
    t_start_total = time.time()

    for i in range(n_steps):
        horizon_end = min(i + controller.mpc_horizon, n_steps)
        p_ref_horizon = p_ref[i:horizon_end]

        if len(p_ref_horizon) < controller.mpc_horizon:
            p_ref_horizon = np.vstack([
                p_ref_horizon,
                np.tile(p_ref_horizon[-1], (controller.mpc_horizon - len(p_ref_horizon), 1))
            ])

        u, info = controller.step(x_t, p_tip_t, p_ref_horizon, hiddens=None)

        result = crm_diff_py.dynamics_forward(x_t, u, dt, L_inserted, params_dict)

        if result['status'] == 0:
            x_t = result['x_next']
            p_tip_t = result['p_tip']
        else:
            n_failures += 1

        p_tips.append(p_tip_t)
        metrics.add_step(info, p_tip_t, p_ref[i])

    total_time = time.time() - t_start_total
    summary = metrics.get_summary()
    summary['total_time'] = total_time
    summary['n_failures'] = n_failures

    return summary


def calibrate_thresholds(
    ensemble,
    datasets,
    params_dict,
    tau_low_range,
    tau_high_range,
    duration=5.0,
    max_configs=None,
    max_seconds=None
):
    """
    Sweep threshold grid and find best configuration.

    Args:
        max_configs: Maximum number of configs to test (None = unlimited)
        max_seconds: Maximum time budget in seconds (None = unlimited)

    Returns:
        dict with sweep results and best thresholds
    """
    print("\n" + "="*60)
    print("THRESHOLD CALIBRATION")
    print("="*60)

    if max_configs is not None:
        print(f"Budget: max {max_configs} configurations")
    if max_seconds is not None:
        print(f"Budget: max {max_seconds}s time limit")

    calibration_start_time = time.time()

    # First, establish MPC baseline
    print("\n[1/3] Computing MPC-only baseline...")
    mpc_baselines = []

    for ds in datasets:
        print(f"  Running MPC on {ds.filename}...")
        baseline = run_mpc_baseline(ds, params_dict, duration=duration)
        mpc_baselines.append(baseline)
        print(f"    RMSE: {baseline['tracking_rmse']:.3f} mm, Time: {baseline['total_time']:.2f}s")

    avg_mpc_rmse = np.mean([b['tracking_rmse'] for b in mpc_baselines])
    avg_mpc_time = np.mean([b['total_time'] for b in mpc_baselines])

    print(f"\n  Baseline avg: RMSE={avg_mpc_rmse:.3f} mm, Time={avg_mpc_time:.2f}s")

    # Sweep threshold grid
    print(f"\n[2/3] Sweeping threshold grid...")
    print(f"  tau_low range: {tau_low_range}")
    print(f"  tau_high range: {tau_high_range}")

    sweep_results = []
    configs_tested = 0
    budget_exceeded = False

    for tau_low in tau_low_range:
        for tau_high in tau_high_range:
            if tau_low >= tau_high:
                continue  # Invalid: tau_low must be < tau_high

            # Check budget constraints
            if max_configs is not None and configs_tested >= max_configs:
                print(f"\n  Budget limit reached: {max_configs} configs tested")
                budget_exceeded = True
                break

            if max_seconds is not None:
                elapsed = time.time() - calibration_start_time
                if elapsed >= max_seconds:
                    print(f"\n  Time budget exceeded: {elapsed:.1f}s >= {max_seconds}s")
                    budget_exceeded = True
                    break

            print(f"\n  Testing tau_low={tau_low:.6f}, tau_high={tau_high:.6f} ({configs_tested+1})")

            results_per_dataset = []

            for ds in datasets:
                result = run_hybrid_with_thresholds(
                    ds, ensemble, params_dict,
                    tau_low, tau_high,
                    duration=duration
                )
                results_per_dataset.append(result)

            # Aggregate metrics
            avg_mpc_call_rate = np.mean([r['mpc_call_rate'] for r in results_per_dataset])
            avg_rmse = np.mean([r['tracking_rmse'] for r in results_per_dataset])
            avg_time = np.mean([r['total_time'] for r in results_per_dataset])
            total_failures = sum(r['n_failures'] for r in results_per_dataset)

            speedup = avg_mpc_time / avg_time if avg_time > 0 else 0.0
            rmse_diff = avg_rmse - avg_mpc_rmse

            # Check acceptance criteria
            pass_mpc_rate = avg_mpc_call_rate <= 0.30
            pass_rmse = avg_rmse <= avg_mpc_rmse + 0.5
            pass_speedup = speedup >= 2.0
            pass_safety = total_failures == 0

            all_pass = pass_mpc_rate and pass_rmse and pass_speedup and pass_safety

            result_entry = {
                'tau_low': tau_low,
                'tau_high': tau_high,
                'mpc_call_rate': float(avg_mpc_call_rate),
                'tracking_rmse': float(avg_rmse),
                'total_time': float(avg_time),
                'speedup': float(speedup),
                'rmse_diff': float(rmse_diff),
                'n_failures': int(total_failures),
                'acceptance': {
                    'mpc_rate': pass_mpc_rate,
                    'rmse': pass_rmse,
                    'speedup': pass_speedup,
                    'safety': pass_safety,
                    'all_pass': all_pass
                }
            }

            sweep_results.append(result_entry)
            configs_tested += 1

            status = "✓ PASS" if all_pass else "✗ FAIL"
            print(f"    {status} MPC rate={avg_mpc_call_rate:.1%}, RMSE={avg_rmse:.3f}mm, speedup={speedup:.2f}×")

        if budget_exceeded:
            break

    # Find best configuration
    print(f"\n[3/3] Selecting best thresholds...")

    if not sweep_results:
        print("  ERROR: No configurations tested (budget too tight)")
        # Return minimal valid response
        return {
            'mpc_baseline': {
                'avg_rmse': float(avg_mpc_rmse),
                'avg_time': float(avg_mpc_time),
                'per_dataset': mpc_baselines
            },
            'sweep_results': [],
            'best_thresholds': None,
            'n_passing': 0,
            'n_total': 0,
            'budget_exceeded': True,
            'configs_tested': configs_tested
        }

    # Filter to only passing configurations
    passing = [r for r in sweep_results if r['acceptance']['all_pass']]

    if not passing:
        print("  WARNING: No configuration met all criteria")
        print("  Selecting best compromise (lowest MPC call rate with safety)")
        safe_configs = [r for r in sweep_results if r['n_failures'] == 0]
        if safe_configs:
            best = min(safe_configs, key=lambda r: r['mpc_call_rate'])
        else:
            best = min(sweep_results, key=lambda r: r['n_failures'])
    else:
        # Among passing, prefer lowest MPC call rate (most policy usage)
        best = min(passing, key=lambda r: r['mpc_call_rate'])

    print(f"\n  Best configuration (from {configs_tested} tested):")
    print(f"    tau_low:  {best['tau_low']:.6f}")
    print(f"    tau_high: {best['tau_high']:.6f}")
    print(f"    MPC call rate: {best['mpc_call_rate']:.1%}")
    print(f"    Tracking RMSE: {best['tracking_rmse']:.3f} mm")
    print(f"    Speedup: {best['speedup']:.2f}×")
    print(f"    Safety violations: {best['n_failures']}")

    if budget_exceeded:
        print(f"\n  ⚠ Budget limit reached - results may be incomplete")

    return {
        'mpc_baseline': {
            'avg_rmse': float(avg_mpc_rmse),
            'avg_time': float(avg_mpc_time),
            'per_dataset': mpc_baselines
        },
        'sweep_results': sweep_results,
        'best_thresholds': best,
        'n_passing': len(passing),
        'n_total': len(sweep_results),
        'budget_exceeded': budget_exceeded,
        'configs_tested': configs_tested
    }


def main():
    """Main calibration routine."""
    import argparse

    parser = argparse.ArgumentParser(description='CP4.7.1/CP4.7.2: Threshold Calibration')
    parser.add_argument('--max_configs', type=int, default=None,
                       help='Maximum number of configurations to test')
    parser.add_argument('--max_seconds', type=int, default=None,
                       help='Maximum time budget in seconds')
    parser.add_argument('--datasets_limit', type=int, default=2,
                       help='Limit number of datasets to use (default: 2)')
    args = parser.parse_args()

    print("="*60)
    print("CP4.7.1/CP4.7.2: Threshold Calibration")
    print("="*60)

    # Load ensemble
    print("\n[Loading ensemble...]")
    ensemble = load_ensemble_from_cp45()

    if ensemble is None:
        print("\n✗ Cannot proceed without trained ensemble")
        print("  Run: python3 python/train_cp45_ensemble_dagger.py")
        return 1

    # Load datasets
    print("\n[Loading datasets...]")
    try:
        manifest = load_manifest(data_dir='./data', verbose=False)
        datasets = manifest.datasets_with_ref

        if not datasets:
            print("✗ No datasets with references found")
            return 1

        # Apply dataset limit
        datasets = datasets[:args.datasets_limit]

        print(f"✓ Using {len(datasets)} datasets:")
        for ds in datasets:
            print(f"  - {ds.filename}")

    except Exception as e:
        print(f"✗ Failed to load datasets: {e}")
        return 1

    # Load physics parameters
    print("\n[Loading physics parameters...]")
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
        'IntegrationStepSize': 0.5,  # Will be overridden by dataset
        'FinalValueOnly': True,
    }

    # Define threshold ranges
    tau_low_range = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
    tau_high_range = [5e-3, 1e-2, 2e-2, 5e-2]

    # Run calibration
    t_start = time.time()

    results = calibrate_thresholds(
        ensemble,
        datasets,
        params_dict,
        tau_low_range,
        tau_high_range,
        duration=3.0,  # 3 seconds per dataset (fast calibration)
        max_configs=args.max_configs,
        max_seconds=args.max_seconds
    )

    t_elapsed = time.time() - t_start

    # Save results
    output_dir = './build/artifacts'
    os.makedirs(output_dir, exist_ok=True)

    # Full sweep results
    sweep_path = os.path.join(output_dir, 'cp47_threshold_sweep.json')
    with open(sweep_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved sweep results: {sweep_path}")

    # Best thresholds (short summary)
    best_path = os.path.join(output_dir, 'cp47_threshold_best.json')

    if results['best_thresholds'] is not None:
        best_summary = {
            'tau_low': results['best_thresholds']['tau_low'],
            'tau_high': results['best_thresholds']['tau_high'],
            'metrics': {
                'mpc_call_rate': results['best_thresholds']['mpc_call_rate'],
                'tracking_rmse': results['best_thresholds']['tracking_rmse'],
                'speedup': results['best_thresholds']['speedup'],
                'safety_violations': results['best_thresholds']['n_failures']
            },
            'acceptance': results['best_thresholds']['acceptance'],
            'calibration_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'budget_exceeded': results['budget_exceeded']
        }

        with open(best_path, 'w') as f:
            json.dump(best_summary, f, indent=2)
        print(f"✓ Saved best thresholds: {best_path}")
    else:
        print(f"⚠ No thresholds found (budget too tight)")

    # Summary
    print("\n" + "="*60)
    print("CALIBRATION COMPLETE")
    print("="*60)
    print(f"Total time: {t_elapsed:.1f}s")
    print(f"Configurations tested: {results.get('configs_tested', results['n_total'])}")
    print(f"Passing all criteria: {results['n_passing']}")

    if results['best_thresholds'] is not None:
        print(f"\nRecommended thresholds:")
        print(f"  tau_low:  {results['best_thresholds']['tau_low']:.6f}")
        print(f"  tau_high: {results['best_thresholds']['tau_high']:.6f}")

    if results.get('budget_exceeded', False):
        print("\n⚠ Budget limit reached - calibration incomplete")

    print("="*60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
