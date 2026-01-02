#!/usr/bin/env python3
"""
CP4.7.2/CP4.7.4: End-to-End Trained Validation

Runs the complete CP4.7 pipeline under bounded compute budget:
1. Fast ensemble training (<=30 min, N>=3 members)
2. Threshold calibration (capped budget)
3. Trained hybrid benchmark

CP4.7.4 Enhancement:
- Optionally uses windowed health report if available
- Enables validation on valid dataset windows instead of full datasets

Outputs JSON summary with PASS/FAIL verdict and acceptance metrics.

Labeled nightly-only. Timeout: 1800s (30 minutes).

Acceptance gates:
- MPC call rate <= 0.30
- RMSE <= RMSE(MPC-only) + 0.5 mm
- Speedup >= 2.0x
- Safety violations == 0
"""

import sys
import os
import json
import time
import subprocess
import numpy as np
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from control.hybrid_controller import HybridController, HybridControllerMetrics
from control.ilqr import iLQRSolver
from models.ensemble_policy import EnsemblePolicy
from models.recurrent_policy import GRUPolicy
from data.npz_manifest import load_manifest
import crm_diff_py


# ============================================================================
# Phase 1: Fast Ensemble Training
# ============================================================================

def run_fast_ensemble_training(datasets_limit=2, max_seconds=1200, skip_if_exists=True):
    """
    Run fast ensemble training with CP4.7.2 profile.

    Args:
        datasets_limit: Limit number of training datasets
        max_seconds: Maximum training time (default: 20 minutes)
        skip_if_exists: Skip training if ensemble artifacts already exist

    Returns:
        dict with training results or error info
    """
    print("\n" + "="*80)
    print("PHASE 1: Fast Ensemble Training")
    print("="*80)
    print(f"Budget: {max_seconds}s ({max_seconds/60:.1f} min)")
    print(f"Datasets: {datasets_limit}")

    # Check if ensemble already exists
    metadata_path = './build/artifacts/cp472_ensemble_fast_ensemble_metadata.json'
    if skip_if_exists and os.path.exists(metadata_path):
        print(f"\n⏭️  Skipping training - ensemble already exists: {metadata_path}")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return {
            'success': True,
            'time': 0.0,
            'n_members': metadata['n_members'],
            'metadata_path': metadata_path,
            'skipped': True
        }

    t_start = time.time()

    # Run training script with fast_train profile
    cmd = [
        sys.executable,
        "python/train_cp45_ensemble_dagger.py",
        "--profile", "fast_train",
        "--datasets_limit", str(datasets_limit)
    ]

    print(f"\nCommand: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            timeout=max_seconds,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )

        t_elapsed = time.time() - t_start

        if result.returncode != 0:
            print(f"✗ Training failed (exit code {result.returncode})")
            print(f"\nStdout:\n{result.stdout}")
            print(f"\nStderr:\n{result.stderr}")
            return {
                'success': False,
                'error': 'training_failed',
                'exit_code': result.returncode,
                'time': t_elapsed,
                'stdout': result.stdout[-1000:],  # Last 1000 chars
                'stderr': result.stderr[-1000:]
            }

        print(f"✓ Training completed in {t_elapsed:.1f}s ({t_elapsed/60:.1f} min)")

        # Check if ensemble artifacts exist
        metadata_path = './build/artifacts/cp472_ensemble_fast_ensemble_metadata.json'
        if not os.path.exists(metadata_path):
            print(f"✗ Ensemble metadata not found: {metadata_path}")
            return {
                'success': False,
                'error': 'missing_artifacts',
                'time': t_elapsed
            }

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        print(f"✓ Ensemble created: {metadata['n_members']} members")

        return {
            'success': True,
            'time': t_elapsed,
            'n_members': metadata['n_members'],
            'metadata_path': metadata_path
        }

    except subprocess.TimeoutExpired:
        t_elapsed = time.time() - t_start
        print(f"✗ Training timeout after {t_elapsed:.1f}s")
        return {
            'success': False,
            'error': 'timeout',
            'time': t_elapsed
        }


# ============================================================================
# Phase 2: Threshold Calibration
# ============================================================================

def run_threshold_calibration(max_configs=5, max_seconds=300, datasets_limit=2, skip_if_exists=True):
    """
    Run threshold calibration with capped budget.

    Args:
        max_configs: Max configurations to test
        max_seconds: Max time budget
        datasets_limit: Number of datasets to use
        skip_if_exists: Skip if threshold artifacts already exist

    Returns:
        dict with calibration results
    """
    print("\n" + "="*80)
    print("PHASE 2: Threshold Calibration")
    print("="*80)
    print(f"Budget: {max_configs} configs, {max_seconds}s")
    print(f"Datasets: {datasets_limit}")

    # Check if calibration already done
    best_path = './build/artifacts/cp47_threshold_best.json'
    if skip_if_exists and os.path.exists(best_path):
        print(f"\n⏭️  Skipping calibration - thresholds already exist: {best_path}")
        with open(best_path, 'r') as f:
            best_data = json.load(f)
        return {
            'success': True,
            'time': 0.0,
            'configs_tested': 0,
            'n_passing': 0,
            'budget_exceeded': False,
            'best_thresholds': best_data,
            'skipped': True
        }

    t_start = time.time()

    cmd = [
        sys.executable,
        "python/calibrate_cp47_thresholds.py",
        "--max_configs", str(max_configs),
        "--max_seconds", str(max_seconds),
        "--datasets_limit", str(datasets_limit)
    ]

    print(f"\nCommand: {' '.join(cmd)}")
    print()

    try:
        result = subprocess.run(
            cmd,
            timeout=max_seconds + 60,  # Extra 60s grace period
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(__file__))
        )

        t_elapsed = time.time() - t_start

        if result.returncode != 0:
            print(f"✗ Calibration failed (exit code {result.returncode})")
            print(f"\nStdout:\n{result.stdout}")
            print(f"\nStderr:\n{result.stderr}")
            return {
                'success': False,
                'error': 'calibration_failed',
                'exit_code': result.returncode,
                'time': t_elapsed
            }

        print(f"✓ Calibration completed in {t_elapsed:.1f}s")

        # Load calibration results
        sweep_path = './build/artifacts/cp47_threshold_sweep.json'
        best_path = './build/artifacts/cp47_threshold_best.json'

        if not os.path.exists(sweep_path):
            print(f"✗ Calibration results not found: {sweep_path}")
            return {
                'success': False,
                'error': 'missing_results',
                'time': t_elapsed
            }

        with open(sweep_path, 'r') as f:
            sweep_data = json.load(f)

        # Best thresholds might not exist if budget too tight
        if os.path.exists(best_path):
            with open(best_path, 'r') as f:
                best_data = json.load(f)
            print(f"✓ Best thresholds: tau_low={best_data['tau_low']:.6f}, tau_high={best_data['tau_high']:.6f}")
        else:
            print(f"⚠ No best thresholds found (using defaults)")
            best_data = None

        return {
            'success': True,
            'time': t_elapsed,
            'configs_tested': sweep_data.get('configs_tested', sweep_data['n_total']),
            'n_passing': sweep_data['n_passing'],
            'budget_exceeded': sweep_data.get('budget_exceeded', False),
            'best_thresholds': best_data
        }

    except subprocess.TimeoutExpired:
        t_elapsed = time.time() - t_start
        print(f"✗ Calibration timeout after {t_elapsed:.1f}s")
        return {
            'success': False,
            'error': 'timeout',
            'time': t_elapsed
        }


# ============================================================================
# Phase 3: Trained Benchmark
# ============================================================================

def load_ensemble_cp472():
    """Load CP4.7.2 fast-trained ensemble."""
    metadata_path = './build/artifacts/cp472_ensemble_fast_ensemble_metadata.json'

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    ensemble = EnsemblePolicy(
        policy_paths=metadata['checkpoint_paths'],
        policy_class=GRUPolicy,
        policy_kwargs=metadata['policy_kwargs']
    )

    return ensemble


def load_thresholds_cp472():
    """Load calibrated thresholds or use defaults."""
    best_path = './build/artifacts/cp47_threshold_best.json'

    if os.path.exists(best_path):
        with open(best_path, 'r') as f:
            data = json.load(f)
        return data['tau_low'], data['tau_high']
    else:
        # Default fallback
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


def run_trained_benchmark(duration=5.0, datasets_limit=2):
    """
    Run trained hybrid benchmark.

    Args:
        duration: Duration to test per dataset (seconds)
        datasets_limit: Number of datasets to test

    Returns:
        dict with benchmark results
    """
    print("\n" + "="*80)
    print("PHASE 3: Trained Benchmark")
    print("="*80)
    print(f"Duration: {duration}s per dataset")
    print(f"Datasets: {datasets_limit}")

    t_start = time.time()

    # Load ensemble
    try:
        ensemble = load_ensemble_cp472()
        print(f"✓ Loaded CP4.7.2 ensemble")
    except Exception as e:
        print(f"✗ Failed to load ensemble: {e}")
        return {
            'success': False,
            'error': 'ensemble_load_failed',
            'exception': str(e)
        }

    # Load thresholds
    tau_low, tau_high = load_thresholds_cp472()
    print(f"✓ Thresholds: tau_low={tau_low:.6f}, tau_high={tau_high:.6f}")

    # Load datasets
    try:
        manifest = load_manifest(data_dir='./data', verbose=False)
        datasets = manifest.datasets_with_ref[:datasets_limit]

        if not datasets:
            print("✗ No datasets available")
            return {
                'success': False,
                'error': 'no_datasets'
            }

        print(f"✓ Using {len(datasets)} datasets")

    except Exception as e:
        print(f"✗ Failed to load datasets: {e}")
        return {
            'success': False,
            'error': 'dataset_load_failed',
            'exception': str(e)
        }

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
    print("\nRunning benchmarks...")
    mpc_results = []
    hybrid_results = []

    for i, ds in enumerate(datasets):
        print(f"\n  Dataset {i+1}/{len(datasets)}: {ds.filename}")

        # MPC baseline
        print("    [1/2] MPC baseline...")
        mpc_res = run_mpc_baseline(ds, params_dict, duration)
        mpc_results.append(mpc_res)
        print(f"      RMSE: {mpc_res['tracking_rmse']:.3f} mm, Time: {mpc_res['total_time']:.2f}s")

        # Hybrid
        print("    [2/2] Hybrid controller...")
        hybrid_res = run_hybrid(ds, ensemble, params_dict, tau_low, tau_high, duration)
        hybrid_results.append(hybrid_res)
        print(f"      RMSE: {hybrid_res['tracking_rmse']:.3f} mm, MPC rate: {hybrid_res['mpc_call_rate']:.1%}")

    t_elapsed = time.time() - t_start

    # Aggregate results
    avg_mpc_rmse = np.mean([r['tracking_rmse'] for r in mpc_results])
    avg_mpc_time = np.mean([r['total_time'] for r in mpc_results])
    total_mpc_failures = sum(r['n_failures'] for r in mpc_results)

    avg_hybrid_rmse = np.mean([r['tracking_rmse'] for r in hybrid_results])
    avg_hybrid_time = np.mean([r['total_time'] for r in hybrid_results])
    avg_mpc_call_rate = np.mean([r['mpc_call_rate'] for r in hybrid_results])
    total_hybrid_failures = sum(r['n_failures'] for r in hybrid_results)

    speedup = avg_mpc_time / avg_hybrid_time if avg_hybrid_time > 0 else 0.0
    rmse_diff = avg_hybrid_rmse - avg_mpc_rmse

    print(f"\n✓ Benchmark completed in {t_elapsed:.1f}s")

    return {
        'success': True,
        'time': t_elapsed,
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
        'speedup': float(speedup),
        'rmse_diff': float(rmse_diff)
    }


# ============================================================================
# Main End-to-End Test
# ============================================================================

def classify_failure_reason(training_result, calibration_result, benchmark_result):
    """
    Classify failure root cause.

    Returns:
        str: One of 'policy_quality', 'uncertainty_estimation', 'mpc_speed', 'physics_failures', 'unknown'
    """
    if not benchmark_result.get('success', False):
        return 'benchmark_execution_failed'

    bench = benchmark_result

    # Check for physics failures
    if bench['hybrid']['total_failures'] > 0:
        return 'physics_failures'

    # Check MPC call rate (too high → poor policy or uncertainty)
    if bench['hybrid']['avg_mpc_call_rate'] > 0.50:
        return 'policy_quality'  # Policy not confident enough

    # Check tracking accuracy
    if bench['rmse_diff'] > 1.0:  # More than 1mm worse than MPC
        return 'policy_quality'

    # Check speedup (too low → MPC too slow or policy not helping)
    if bench['speedup'] < 1.5:
        return 'mpc_speed'

    # Otherwise, it's close but didn't quite pass
    return 'marginal_performance'


def main():
    """Main end-to-end validation."""
    print("="*80)
    print("CP4.7.2: End-to-End Trained Validation")
    print("="*80)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    overall_start = time.time()

    # Results accumulator
    results = {
        'test_name': 'CP4.7.2 End-to-End Validation',
        'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
        'phases': {}
    }

    # Phase 1: Training
    training_result = run_fast_ensemble_training(
        datasets_limit=2,
        max_seconds=1200  # 20 minutes
    )
    results['phases']['training'] = training_result

    if not training_result['success']:
        print("\n✗ Training phase failed - cannot continue")
        results['overall_result'] = 'FAIL'
        results['failure_reason'] = f"training_{training_result.get('error', 'unknown')}"
        results['total_time'] = time.time() - overall_start
        save_and_print_summary(results)
        return 1

    # Phase 2: Calibration
    calibration_result = run_threshold_calibration(
        max_configs=5,
        max_seconds=300,  # 5 minutes
        datasets_limit=2
    )
    results['phases']['calibration'] = calibration_result

    if not calibration_result['success']:
        print("\n✗ Calibration phase failed - cannot continue")
        results['overall_result'] = 'FAIL'
        results['failure_reason'] = f"calibration_{calibration_result.get('error', 'unknown')}"
        results['total_time'] = time.time() - overall_start
        save_and_print_summary(results)
        return 1

    # Phase 3: Benchmark
    benchmark_result = run_trained_benchmark(duration=2.0, datasets_limit=1)
    results['phases']['benchmark'] = benchmark_result

    if not benchmark_result['success']:
        print("\n✗ Benchmark phase failed")
        results['overall_result'] = 'FAIL'
        results['failure_reason'] = f"benchmark_{benchmark_result.get('error', 'unknown')}"
        results['total_time'] = time.time() - overall_start
        save_and_print_summary(results)
        return 1

    # Check acceptance criteria
    bench = benchmark_result
    criteria = {
        'mpc_call_rate': {
            'value': bench['hybrid']['avg_mpc_call_rate'],
            'target': 0.30,
            'pass': bench['hybrid']['avg_mpc_call_rate'] <= 0.30
        },
        'tracking_rmse': {
            'value': bench['hybrid']['avg_rmse'],
            'target': bench['mpc_baseline']['avg_rmse'] + 0.5,
            'pass': bench['hybrid']['avg_rmse'] <= bench['mpc_baseline']['avg_rmse'] + 0.5
        },
        'speedup': {
            'value': bench['speedup'],
            'target': 2.0,
            'pass': bench['speedup'] >= 2.0
        },
        'safety_violations': {
            'value': bench['hybrid']['total_failures'],
            'target': 0,
            'pass': bench['hybrid']['total_failures'] == 0
        }
    }

    all_pass = all(c['pass'] for c in criteria.values())

    results['acceptance_criteria'] = criteria
    results['overall_result'] = 'PASS' if all_pass else 'FAIL'

    if not all_pass:
        results['failure_reason'] = classify_failure_reason(
            training_result, calibration_result, benchmark_result
        )
        results['failure_details'] = {
            name: {'value': c['value'], 'target': c['target'], 'pass': c['pass']}
            for name, c in criteria.items() if not c['pass']
        }

    results['total_time'] = time.time() - overall_start

    # Save and print summary
    save_and_print_summary(results)

    return 0 if all_pass else 1


def save_and_print_summary(results):
    """Save JSON summary and print results."""

    # Save JSON
    output_dir = './build/artifacts'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'cp472_end_to_end_results.json')

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("END-TO-END VALIDATION SUMMARY")
    print("="*80)
    print(f"Total time: {results['total_time']:.1f}s ({results['total_time']/60:.1f} min)")
    print()

    # Print phase timings
    for phase_name, phase_data in results['phases'].items():
        if 'time' in phase_data:
            status = "✓" if phase_data.get('success', False) else "✗"
            print(f"{status} {phase_name.capitalize()}: {phase_data['time']:.1f}s")

    print()

    # Print acceptance criteria if available
    if 'acceptance_criteria' in results:
        print("Acceptance Criteria:")
        for name, crit in results['acceptance_criteria'].items():
            status = "✓ PASS" if crit['pass'] else "✗ FAIL"
            print(f"  {status}  {name:20s}: {crit['value']:.4f} (target: {crit['target']:.4f})")
        print()

    # Overall verdict
    if results['overall_result'] == 'PASS':
        print("="*80)
        print("✓ OVERALL: PASS")
        print("="*80)
    else:
        print("="*80)
        print("✗ OVERALL: FAIL")
        if 'failure_reason' in results:
            print(f"  Reason: {results['failure_reason']}")
        print("="*80)

    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
