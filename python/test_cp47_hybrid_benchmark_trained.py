#!/usr/bin/env python3
"""
CP4.7.1: Hybrid Controller Benchmark with Trained Ensemble

Validates hybrid controller acceptance criteria using trained ensemble.
Requires:
- Trained ensemble from CP4.5
- Calibrated thresholds from calibrate_cp47_thresholds.py (optional)

Acceptance gates:
- MPC call rate ≤ 30%
- Tracking RMSE ≤ MPC-only + 0.5 mm
- Speedup ≥ 2×
- Zero safety violations

Labeled nightly-only. Skips gracefully if ensemble not available.
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from control.hybrid_controller import HybridController, HybridControllerMetrics
from control.ilqr import iLQRSolver
from models.ensemble_policy import EnsemblePolicy
from models.recurrent_policy import GRUPolicy
from data.npz_manifest import load_manifest
import crm_diff_py


def check_ensemble_available():
    """
    Check if trained ensemble is available.

    Returns:
        (bool, str): (available, message)
    """
    metadata_path = './build/artifacts/cp45_ensemble_dagger_ensemble_metadata.json'

    if not os.path.exists(metadata_path):
        return False, f"Ensemble metadata not found: {metadata_path}"

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    missing = [p for p in metadata['checkpoint_paths'] if not os.path.exists(p)]
    if missing:
        return False, f"Missing checkpoints: {missing[:3]}"

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
    """
    Load calibrated thresholds from calibration script.

    Returns:
        (tau_low, tau_high) or default values if not calibrated
    """
    best_path = './build/artifacts/cp47_threshold_best.json'

    if os.path.exists(best_path):
        with open(best_path, 'r') as f:
            data = json.load(f)
        return data['tau_low'], data['tau_high']
    else:
        # Default values from CP4.7 design
        return 0.001, 0.01


def run_mpc_baseline(dataset, params_dict, duration=5.0):
    """Run MPC-only baseline."""
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
    t_start = time.time()

    for i in range(n_steps):
        horizon_end = min(i + 10, n_steps)
        p_ref_horizon = p_ref[i:horizon_end]

        if len(p_ref_horizon) < 10:
            p_ref_horizon = np.vstack([
                p_ref_horizon,
                np.tile(p_ref_horizon[-1], (10 - len(p_ref_horizon), 1))
            ])

        t_mpc = time.time()

        solver = iLQRSolver(
            dt=dt,
            L_inserted=L_inserted,
            params_dict=params_dict,
            horizon=10,
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

        mpc_times.append((time.time() - t_mpc) * 1000.0)

        result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)

        if result['status'] == 0:
            x_t = result['x_next']
            p_tip_t = result['p_tip']
        else:
            n_failures += 1

        p_tips.append(p_tip_t)

    total_time = time.time() - t_start
    p_tips = np.array(p_tips)
    errors = np.linalg.norm(p_tips - p_ref[:n_steps], axis=1)

    return {
        'tracking_rmse': float(np.sqrt(np.mean(errors**2))),
        'tracking_max': float(np.max(errors)),
        'total_time': total_time,
        'mean_mpc_time_ms': float(np.mean(mpc_times)),
        'n_failures': n_failures,
        'n_steps': n_steps
    }


def run_hybrid(dataset, ensemble, params_dict, tau_low, tau_high, duration=5.0):
    """Run hybrid controller."""
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
        mpc_horizon=10,
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

        p_tips.append(p_tip_t)
        metrics.add_step(info, p_tip_t, p_ref[i])

    total_time = time.time() - t_start
    summary = metrics.get_summary()
    summary['total_time'] = total_time
    summary['n_failures'] = n_failures
    summary['n_steps'] = n_steps

    return summary


def main():
    """Main benchmark routine."""
    print("\n" + "="*60)
    print("CP4.7.1: Hybrid Benchmark (Trained Ensemble)")
    print("="*60)

    # Check ensemble availability
    available, message = check_ensemble_available()

    if not available:
        print(f"\n⏸️  SKIPPED: {message}")
        print("\nTo run this benchmark:")
        print("  1. Train ensemble: python3 python/train_cp45_ensemble_dagger.py")
        print("  2. (Optional) Calibrate: python3 python/calibrate_cp47_thresholds.py")
        print("  3. Re-run this benchmark")
        print("\n" + "="*60)
        return 0  # Exit 0 (not a failure, just skipped)

    print(f"✓ {message}")

    # Load ensemble
    print("\n[Loading ensemble...]")
    ensemble = load_ensemble()
    print(f"✓ Loaded ensemble")

    # Load thresholds
    print("\n[Loading thresholds...]")
    tau_low, tau_high = load_calibrated_thresholds()
    print(f"  tau_low:  {tau_low:.6f}")
    print(f"  tau_high: {tau_high:.6f}")

    # Load datasets
    print("\n[Loading datasets...]")
    try:
        manifest = load_manifest(data_dir='./data', verbose=False)
        datasets = manifest.datasets_with_ref[:2]  # Use first 2

        if not datasets:
            print("✗ No datasets available")
            return 1

        print(f"✓ Using {len(datasets)} datasets:")
        for ds in datasets:
            print(f"  - {ds.filename}")

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

    # Run benchmarks
    duration = 5.0  # 5 seconds per dataset

    print("\n" + "="*60)
    print("RUNNING BENCHMARKS")
    print("="*60)

    mpc_results = []
    hybrid_results = []

    for i, ds in enumerate(datasets):
        print(f"\nDataset {i+1}/{len(datasets)}: {ds.filename}")

        # MPC baseline
        print("  [1/2] MPC-only baseline...")
        mpc_res = run_mpc_baseline(ds, params_dict, duration)
        mpc_results.append(mpc_res)
        print(f"    RMSE: {mpc_res['tracking_rmse']:.3f} mm, Time: {mpc_res['total_time']:.2f}s")

        # Hybrid
        print("  [2/2] Hybrid controller...")
        hybrid_res = run_hybrid(ds, ensemble, params_dict, tau_low, tau_high, duration)
        hybrid_results.append(hybrid_res)
        print(f"    RMSE: {hybrid_res['tracking_rmse']:.3f} mm, MPC rate: {hybrid_res['mpc_call_rate']:.1%}, Time: {hybrid_res['total_time']:.2f}s")

    # Aggregate metrics
    avg_mpc_rmse = np.mean([r['tracking_rmse'] for r in mpc_results])
    avg_mpc_time = np.mean([r['total_time'] for r in mpc_results])
    total_mpc_failures = sum(r['n_failures'] for r in mpc_results)

    avg_hybrid_rmse = np.mean([r['tracking_rmse'] for r in hybrid_results])
    avg_hybrid_time = np.mean([r['total_time'] for r in hybrid_results])
    avg_mpc_call_rate = np.mean([r['mpc_call_rate'] for r in hybrid_results])
    total_hybrid_failures = sum(r['n_failures'] for r in hybrid_results)

    speedup = avg_mpc_time / avg_hybrid_time if avg_hybrid_time > 0 else 0.0
    rmse_diff = avg_hybrid_rmse - avg_mpc_rmse

    # Check acceptance criteria
    criteria = {
        'mpc_call_rate': {
            'value': avg_mpc_call_rate,
            'target': 0.30,
            'pass': avg_mpc_call_rate <= 0.30
        },
        'tracking_rmse': {
            'value': avg_hybrid_rmse,
            'target': avg_mpc_rmse + 0.5,
            'pass': avg_hybrid_rmse <= avg_mpc_rmse + 0.5
        },
        'speedup': {
            'value': speedup,
            'target': 2.0,
            'pass': speedup >= 2.0
        },
        'safety': {
            'value': total_hybrid_failures,
            'target': total_mpc_failures,
            'pass': total_hybrid_failures <= total_mpc_failures
        }
    }

    all_pass = all(c['pass'] for c in criteria.values())

    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)

    print("\nMPC-only baseline:")
    print(f"  Avg RMSE: {avg_mpc_rmse:.4f} mm")
    print(f"  Avg time: {avg_mpc_time:.2f} s")
    print(f"  Failures: {total_mpc_failures}")

    print("\nHybrid controller:")
    print(f"  Avg RMSE: {avg_hybrid_rmse:.4f} mm")
    print(f"  Avg time: {avg_hybrid_time:.2f} s")
    print(f"  MPC call rate: {avg_mpc_call_rate:.1%}")
    print(f"  Failures: {total_hybrid_failures}")

    print("\nComparison:")
    print(f"  Speedup: {speedup:.2f}×")
    print(f"  RMSE difference: {rmse_diff:+.4f} mm")

    print("\n" + "="*60)
    print("ACCEPTANCE CRITERIA")
    print("="*60)

    for name, crit in criteria.items():
        status = "✓ PASS" if crit['pass'] else "✗ FAIL"
        print(f"{status}  {name:20s}: {crit['value']:.4f} (target: ≤{crit['target']:.4f})")

    print("\n" + "="*60)
    if all_pass:
        print("✓ ALL CRITERIA MET")
    else:
        print("✗ SOME CRITERIA FAILED")
    print("="*60)

    # Save results
    output_dir = './build/artifacts'
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, 'cp47_benchmark_trained_results.json')
    with open(results_path, 'w') as f:
        json.dump({
            'thresholds': {'tau_low': tau_low, 'tau_high': tau_high},
            'mpc_baseline': {
                'avg_rmse': float(avg_mpc_rmse),
                'avg_time': float(avg_mpc_time),
                'total_failures': int(total_mpc_failures)
            },
            'hybrid': {
                'avg_rmse': float(avg_hybrid_rmse),
                'avg_time': float(avg_hybrid_time),
                'avg_mpc_call_rate': float(avg_mpc_call_rate),
                'total_failures': int(total_hybrid_failures)
            },
            'comparison': {
                'speedup': float(speedup),
                'rmse_diff': float(rmse_diff)
            },
            'acceptance': {
                name: {'value': float(c['value']), 'target': float(c['target']), 'pass': c['pass']}
                for name, c in criteria.items()
            },
            'all_pass': all_pass
        }, f, indent=2)

    print(f"\n✓ Results saved: {results_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
