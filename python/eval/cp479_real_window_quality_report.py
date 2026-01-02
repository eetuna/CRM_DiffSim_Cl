#!/usr/bin/env python3
"""
CP4.7.9: Real NPZ Window Quality Report

Nightly report that validates tracking quality on real NPZ datasets using
windowed validation with hold-period mitigation.

Workflow:
1. Run windowed health gate with skip_hold_periods=True
2. For each valid window, run MPC baseline + hybrid (if ensemble available)
3. Generate comprehensive quality report with distributions

Outputs:
- Window pass/fail counts
- RMSE, MPC call rate, speedup distributions
- Top-K worst windows with diagnostics
- Artifact JSON: build/artifacts/cp479_real_window_quality_report.json

Labeled nightly. Budget-controlled. Never crashes.
"""

import sys
import os
import json
import time
import argparse
import numpy as np
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))

from eval.cp47_health_gate import run_sliding_window_health_gate, create_windowed_manifest
from control.hybrid_controller import HybridController
from control.ilqr import iLQRSolver
from models.ensemble_policy import EnsemblePolicy
from models.recurrent_policy import GRUPolicy
from data.npz_manifest import load_manifest
import crm_diff_py


def filter_real_datasets(manifest):
    """Filter out golden datasets, keeping only real NPZ datasets."""
    real_datasets = [
        ds for ds in manifest.datasets_with_ref
        if not ('cp476' in ds.filename.lower() and 'golden' in ds.filename.lower())
    ]
    return real_datasets


def check_ensemble_available():
    """Check if trained ensemble is available."""
    metadata_path = './build/artifacts/cp45_ensemble_dagger_ensemble_metadata.json'

    if not os.path.exists(metadata_path):
        return False, f"Ensemble metadata not found"

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    missing = [p for p in metadata['checkpoint_paths'] if not os.path.exists(p)]
    if missing:
        return False, f"Missing checkpoints"

    return True, "Ensemble available"


def load_ensemble():
    """Load trained ensemble from CP4.5 artifacts."""
    metadata_path = './build/artifacts/cp45_ensemble_dagger_ensemble_metadata.json'

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    ensemble = EnsemblePolicy(
        policy_paths=metadata['checkpoint_paths'],
        policy_class=GRUPolicy,
        policy_kwargs=metadata['policy_kwargs']
    )

    return ensemble


def load_calibrated_thresholds():
    """Load calibrated thresholds or use defaults."""
    best_path = './build/artifacts/cp47_threshold_best.json'

    if os.path.exists(best_path):
        with open(best_path, 'r') as f:
            data = json.load(f)
        return data['tau_low'], data['tau_high']
    else:
        return 0.001, 0.01


def run_mpc_baseline_on_window(windowed_ds, params_dict):
    """Run MPC baseline on a windowed dataset."""
    dt = windowed_ds.dt
    L_inserted = windowed_ds.L_inserted
    p_ref = windowed_ds.tip_ref
    n_steps = len(p_ref)

    x_t = np.zeros(6)

    result_init = crm_diff_py.dynamics_forward(
        x_t, np.zeros(3), 0.0, L_inserted, params_dict
    )
    p_tip_t = result_init['p_tip']

    p_tips = []
    n_failures = 0
    U_warm = None
    t_start = time.time()

    MPC_HORIZON = 10
    MPC_MAX_ITERS = 10

    for i in range(n_steps):
        horizon_end = min(i + MPC_HORIZON, n_steps)
        p_ref_horizon = p_ref[i:horizon_end]

        if len(p_ref_horizon) < MPC_HORIZON:
            p_ref_horizon = np.vstack([
                p_ref_horizon,
                np.tile(p_ref_horizon[-1], (MPC_HORIZON - len(p_ref_horizon), 1))
            ])

        solver = iLQRSolver(
            dt=dt,
            L_inserted=L_inserted,
            params_dict=params_dict,
            horizon=MPC_HORIZON,
            Q=np.zeros((6, 6)),
            R=0.01 * np.eye(3),
            p_target=p_ref_horizon[0],
            terminal_weight=1.0,
            max_iters=MPC_MAX_ITERS,
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

        result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)

        if result['status'] == 0:
            x_t = result['x_next']
            p_tip_t = result['p_tip']
        else:
            n_failures += 1

        p_tips.append(p_tip_t)

    total_time = time.time() - t_start
    p_tips = np.array(p_tips)
    errors = np.linalg.norm(p_tips - p_ref, axis=1)

    return {
        'tracking_rmse': float(np.sqrt(np.mean(errors**2))),
        'total_time': total_time,
        'n_failures': n_failures,
        'n_steps': n_steps
    }


def run_hybrid_on_window(windowed_ds, ensemble, params_dict, tau_low, tau_high):
    """Run hybrid controller on a windowed dataset."""
    from control.hybrid_controller import HybridControllerMetrics

    dt = windowed_ds.dt
    L_inserted = windowed_ds.L_inserted
    p_ref = windowed_ds.tip_ref
    n_steps = len(p_ref)

    controller = HybridController(
        ensemble=ensemble,
        dt=dt,
        L_inserted=L_inserted,
        params_dict=params_dict,
        tau_low=tau_low,
        tau_high=tau_high,
        tracking_safety_limit=5.0,
        mpc_horizon=10,
        mpc_max_iters=10,
        mpc_tol=1e-3
    )

    metrics = HybridControllerMetrics()
    x_t = np.zeros(6)

    result_init = crm_diff_py.dynamics_forward(
        x_t, np.zeros(3), 0.0, L_inserted, params_dict
    )
    p_tip_t = result_init['p_tip']

    n_failures = 0
    t_start = time.time()

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

        metrics.add_step(info, p_tip_t, p_ref[i])

    total_time = time.time() - t_start
    summary = metrics.get_summary()

    return {
        'tracking_rmse': summary['tracking_rmse'],
        'mpc_call_rate': summary['mpc_call_rate'],
        'safety_violations': n_failures,
        'total_time': total_time,
        'n_steps': n_steps
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='CP4.7.9: Real NPZ Window Quality Report'
    )
    parser.add_argument('--max_datasets', type=int, default=None,
                       help='Limit number of datasets to process')
    parser.add_argument('--max_windows', type=int, default=50,
                       help='Limit total windows to process (default: 50)')
    parser.add_argument('--window_steps', type=int, default=50,
                       help='Window size in steps (default: 50)')
    parser.add_argument('--stride_steps', type=int, default=25,
                       help='Stride between windows (default: 25)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of worst windows to report (default: 5)')
    parser.add_argument('--output', type=str,
                       default='./build/artifacts/cp479_real_window_quality_report.json',
                       help='Output JSON path')
    args = parser.parse_args()

    print("="*80)
    print("CP4.7.9: Real NPZ Window Quality Report")
    print("="*80)
    print(f"Budget: max {args.max_datasets or 'unlimited'} datasets, {args.max_windows} windows")
    print()

    t_start = time.time()

    try:
        # Load datasets
        print("[1/5] Loading real NPZ datasets...")
        manifest = load_manifest(data_dir='./data', verbose=False)
        real_datasets = filter_real_datasets(manifest)

        if not real_datasets:
            print("⏸️  SKIPPED: No real NPZ datasets found")

            artifact = {
                'test_name': 'CP4.7.9 Real NPZ Window Quality Report',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'skipped',
                'skipped_reason': 'no_real_datasets',
                'time_elapsed': time.time() - t_start
            }

            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(artifact, f, indent=2)

            print(f"\n✓ Report written: {args.output}")
            return 0

        if args.max_datasets:
            real_datasets = real_datasets[:args.max_datasets]

        print(f"✓ Loaded {len(real_datasets)} real dataset(s)")

        # Load physics parameters
        print("\n[2/5] Loading physics parameters...")
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
        print("✓ Loaded physics parameters")

        # Run windowed health gate WITH hold-period mitigation
        print("\n[3/5] Running windowed health gate (hold-aware)...")
        health_report = run_sliding_window_health_gate(
            real_datasets,
            params_dict,
            window_steps=args.window_steps,
            stride_steps=args.stride_steps,
            max_windows=args.max_windows,
            mpc_horizon=10,
            max_iters=10,
            verbose=True,
            skip_hold_periods=True  # CP4.7.8 mitigation
        )

        valid_window_count = health_report['summary']['valid_window_count']

        if valid_window_count == 0:
            print("\n⏸️  SKIPPED: No valid windows found after health gate + hold filtering")

            artifact = {
                'test_name': 'CP4.7.9 Real NPZ Window Quality Report',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'skipped',
                'skipped_reason': 'no_valid_windows',
                'health_summary': health_report['summary'],
                'time_elapsed': time.time() - t_start
            }

            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(artifact, f, indent=2)

            print(f"\n✓ Report written: {args.output}")
            return 0

        print(f"\n✓ Found {valid_window_count} valid window(s)")

        # Create windowed datasets
        windowed_datasets = create_windowed_manifest(real_datasets, health_report['valid_windows'])
        print(f"✓ Created windowed manifest with {len(windowed_datasets)} dataset(s)")

        # Run MPC baseline on all valid windows
        print("\n[4/5] Running MPC baseline on valid windows...")
        mpc_results = []
        for i, wds in enumerate(windowed_datasets):
            print(f"  [{i+1}/{len(windowed_datasets)}] {wds.filename}")
            res = run_mpc_baseline_on_window(wds, params_dict)
            mpc_results.append({
                'window_filename': wds.filename,
                'window_info': wds.window_info,
                'mpc': res
            })

        # Check ensemble availability
        print("\n[5/5] Checking ensemble availability...")
        ensemble_available, message = check_ensemble_available()

        if not ensemble_available:
            print(f"⏸️  {message}")
            print("  Skipping hybrid controller benchmarks")

            artifact = {
                'test_name': 'CP4.7.9 Real NPZ Window Quality Report',
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'status': 'partial',
                'note': 'MPC baseline only (ensemble not available)',
                'health_summary': health_report['summary'],
                'valid_window_count': valid_window_count,
                'mpc_results': mpc_results,
                'time_elapsed': time.time() - t_start
            }

            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(artifact, f, indent=2)

            print(f"\n✓ Report written: {args.output}")
            return 0

        # Run hybrid on all valid windows
        print(f"✓ {message}")
        ensemble = load_ensemble()
        tau_low, tau_high = load_calibrated_thresholds()
        print(f"  Thresholds: tau_low={tau_low:.6f}, tau_high={tau_high:.6f}")

        print("\nRunning hybrid controller on valid windows...")
        for i, (wds, mpc_res) in enumerate(zip(windowed_datasets, mpc_results)):
            print(f"  [{i+1}/{len(windowed_datasets)}] {wds.filename}")
            hybrid_res = run_hybrid_on_window(wds, ensemble, params_dict, tau_low, tau_high)
            mpc_res['hybrid'] = hybrid_res

        # Compute distributions and statistics
        mpc_rmses = [r['mpc']['tracking_rmse'] for r in mpc_results]
        hybrid_rmses = [r['hybrid']['tracking_rmse'] for r in mpc_results]
        mpc_call_rates = [r['hybrid']['mpc_call_rate'] for r in mpc_results]
        speedups = [r['mpc']['total_time'] / r['hybrid']['total_time'] if r['hybrid']['total_time'] > 0 else 0 for r in mpc_results]
        safety_violations = [r['hybrid']['safety_violations'] for r in mpc_results]

        quality_stats = {
            'mpc_rmse': {
                'mean': float(np.mean(mpc_rmses)),
                'median': float(np.median(mpc_rmses)),
                'min': float(np.min(mpc_rmses)),
                'max': float(np.max(mpc_rmses)),
                'p90': float(np.percentile(mpc_rmses, 90))
            },
            'hybrid_rmse': {
                'mean': float(np.mean(hybrid_rmses)),
                'median': float(np.median(hybrid_rmses)),
                'min': float(np.min(hybrid_rmses)),
                'max': float(np.max(hybrid_rmses)),
                'p90': float(np.percentile(hybrid_rmses, 90))
            },
            'mpc_call_rate': {
                'mean': float(np.mean(mpc_call_rates)),
                'median': float(np.median(mpc_call_rates)),
                'min': float(np.min(mpc_call_rates)),
                'max': float(np.max(mpc_call_rates)),
                'p90': float(np.percentile(mpc_call_rates, 90))
            },
            'speedup': {
                'mean': float(np.mean(speedups)),
                'median': float(np.median(speedups)),
                'min': float(np.min(speedups)),
                'max': float(np.max(speedups)),
                'p90': float(np.percentile(speedups, 90))
            },
            'total_safety_violations': int(sum(safety_violations))
        }

        # Find top-K worst windows (by hybrid RMSE)
        sorted_results = sorted(mpc_results, key=lambda r: r['hybrid']['tracking_rmse'], reverse=True)
        top_worst = sorted_results[:args.top_k]

        # Write artifact
        artifact = {
            'test_name': 'CP4.7.9 Real NPZ Window Quality Report',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'completed',
            'config': {
                'max_datasets': args.max_datasets,
                'max_windows': args.max_windows,
                'window_steps': args.window_steps,
                'stride_steps': args.stride_steps,
                'skip_hold_periods': True
            },
            'health_summary': health_report['summary'],
            'valid_window_count': valid_window_count,
            'quality_stats': quality_stats,
            'top_worst_windows': [
                {
                    'rank': i + 1,
                    'window_filename': r['window_filename'],
                    'window_info': r['window_info'],
                    'mpc_rmse': r['mpc']['tracking_rmse'],
                    'hybrid_rmse': r['hybrid']['tracking_rmse'],
                    'mpc_call_rate': r['hybrid']['mpc_call_rate'],
                    'safety_violations': r['hybrid']['safety_violations']
                }
                for i, r in enumerate(top_worst)
            ],
            'time_elapsed': time.time() - t_start
        }

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(artifact, f, indent=2)

        # Print summary
        print("\n" + "="*80)
        print("QUALITY REPORT SUMMARY")
        print("="*80)
        print(f"Valid windows tested: {valid_window_count}")
        print(f"\nMPC RMSE:         mean={quality_stats['mpc_rmse']['mean']:.3f}mm, p90={quality_stats['mpc_rmse']['p90']:.3f}mm")
        print(f"Hybrid RMSE:      mean={quality_stats['hybrid_rmse']['mean']:.3f}mm, p90={quality_stats['hybrid_rmse']['p90']:.3f}mm")
        print(f"MPC call rate:    mean={quality_stats['mpc_call_rate']['mean']*100:.1f}%, p90={quality_stats['mpc_call_rate']['p90']*100:.1f}%")
        print(f"Speedup:          mean={quality_stats['speedup']['mean']:.2f}×, p90={quality_stats['speedup']['p90']:.2f}×")
        print(f"Safety violations: {quality_stats['total_safety_violations']}")
        print("="*80)

        print(f"\n✓ Report written: {args.output}")
        print(f"✓ Time elapsed: {artifact['time_elapsed']:.1f}s")

        return 0

    except Exception as e:
        print(f"\n✗ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

        artifact = {
            'test_name': 'CP4.7.9 Real NPZ Window Quality Report',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'status': 'error',
            'error': str(e),
            'time_elapsed': time.time() - t_start
        }

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(artifact, f, indent=2)

        return 1


if __name__ == "__main__":
    sys.exit(main())
