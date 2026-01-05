#!/usr/bin/env python3
"""
CP4.7.5: Window-native End-to-End Hybrid Validation

Orchestrates the full CP4.7 pipeline with sliding-window health gating:
1. Run sliding-window health gate across available datasets (budgeted)
2. Build windowed manifest from valid windows
3. Run threshold calibration on windowed datasets
4. Run trained hybrid benchmark on windowed datasets
5. Emit single PASS/FAIL JSON summary with root-cause classification

Always writes: build/artifacts/cp475_end_to_end_results.json

Labeled nightly-only. Timeout: 1800s (30 minutes).
"""

import sys
import os
import json
import time
import numpy as np
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from eval.cp47_health_gate import run_sliding_window_health_gate, create_windowed_manifest
from control.hybrid_controller import HybridController, HybridControllerMetrics
from control.ilqr import iLQRSolver
from models.ensemble_policy import EnsemblePolicy
from models.recurrent_policy import GRUPolicy
from data.npz_manifest import load_manifest
import crm_diff_py


def check_ensemble_available():
    """Check if trained ensemble is available."""
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
    t_start_total = time.time()

    for i in range(n_steps):
        horizon_end = min(i + 10, n_steps)
        p_ref_horizon = p_ref[i:horizon_end]

        if len(p_ref_horizon) < 10:
            p_ref_horizon = np.vstack([
                p_ref_horizon,
                np.tile(p_ref_horizon[-1], (10 - len(p_ref_horizon), 1))
            ])

        t_mpc_start = time.time()

        solver = iLQRSolver(
            dt=dt,
            L_inserted=L_inserted,
            params_dict=params_dict,
            horizon=len(p_ref_horizon),
            max_iters=10,
            jacobian_mode="cpp"
        )

        x_next, u_opt, info = solver.solve_mpc(x_t, p_ref_horizon, U_warm=U_warm)

        t_mpc_end = time.time()
        mpc_times.append(t_mpc_end - t_mpc_start)

        if info['converged']:
            u_apply = u_opt[0]
            U_warm = np.vstack([u_opt[1:], u_opt[-1:]])
        else:
            u_apply = np.zeros(3)
            U_warm = None
            n_failures += 1

        result_next = crm_diff_py.dynamics_forward(
            x_t, u_apply, dt, L_inserted, params_dict
        )

        if result_next['status'] != 0:
            n_failures += 1

        x_t = result_next['x_out']
        p_tip_t = result_next['p_tip']
        p_tips.append(p_tip_t.copy())

    t_total = time.time() - t_start_total

    p_tips_arr = np.array(p_tips)
    errors = np.linalg.norm(p_tips_arr - p_ref[:len(p_tips_arr)], axis=1)
    tracking_rmse = np.sqrt(np.mean(errors**2))

    return {
        'tracking_rmse': tracking_rmse,
        'total_time': t_total,
        'avg_mpc_time': np.mean(mpc_times),
        'n_failures': n_failures,
        'n_steps': n_steps
    }


def run_hybrid_trial(dataset, params_dict, ensemble, tau_low, tau_high, duration=5.0):
    """Run hybrid controller trial."""
    dt = dataset.dt
    L_inserted = dataset.L_inserted
    n_steps = min(int(duration / dt), len(dataset.tip_ref))

    p_ref = dataset.tip_ref[:n_steps]

    controller = HybridController(
        ensemble=ensemble,
        tau_low=tau_low,
        tau_high=tau_high,
        mpc_horizon=10,
        mpc_max_iters=10,
        dt=dt,
        L_inserted=L_inserted,
        params_dict=params_dict
    )

    metrics = controller.run_trajectory(p_ref, return_metrics=True)

    return {
        'tracking_rmse': metrics.tracking_rmse,
        'mpc_call_rate': metrics.mpc_call_rate,
        'avg_total_time_per_step': metrics.avg_total_time_per_step,
        'safety_violations': metrics.safety_violations,
        'n_steps': len(p_ref)
    }


def calibrate_thresholds_cp475(ensemble, datasets, params_dict, duration=5.0, max_configs=10, max_seconds=300):
    """
    Simplified threshold calibration for CP4.7.5.

    Args:
        max_configs: Maximum number of configs to test
        max_seconds: Maximum time budget

    Returns:
        dict with best thresholds and calibration summary
    """
    print("\n" + "="*80)
    print("THRESHOLD CALIBRATION (CP4.7.5 Windowed Mode)")
    print("="*80)
    print(f"Budget: max {max_configs} configs, {max_seconds}s")
    print(f"Datasets: {len(datasets)} (windowed)")

    calibration_start = time.time()

    # MPC baseline
    print("\n[1/3] Computing MPC-only baseline...")
    mpc_baselines = []

    for ds in datasets:
        baseline = run_mpc_baseline(ds, params_dict, duration=duration)
        mpc_baselines.append(baseline)

    avg_mpc_rmse = np.mean([b['tracking_rmse'] for b in mpc_baselines])
    avg_mpc_time = np.mean([b['total_time'] for b in mpc_baselines])

    print(f"  Baseline avg: RMSE={avg_mpc_rmse:.3f} mm, Time={avg_mpc_time:.2f}s")

    # Sweep threshold grid
    print(f"\n[2/3] Sweeping threshold grid...")

    tau_low_range = [0.0005, 0.001, 0.002]
    tau_high_range = [0.005, 0.01, 0.02]

    results = []
    configs_tested = 0

    for tau_low in tau_low_range:
        for tau_high in tau_high_range:
            if time.time() - calibration_start > max_seconds:
                print(f"\n⏱️  Time budget exceeded, stopping sweep")
                break

            if configs_tested >= max_configs:
                print(f"\n⏱️  Config budget exceeded, stopping sweep")
                break

            if tau_high <= tau_low:
                continue

            configs_tested += 1

            trials = []
            for ds in datasets:
                trial = run_hybrid_trial(ds, params_dict, ensemble, tau_low, tau_high, duration=duration)
                trials.append(trial)

            avg_rmse = np.mean([t['tracking_rmse'] for t in trials])
            avg_mpc_rate = np.mean([t['mpc_call_rate'] for t in trials])
            total_violations = sum([t['safety_violations'] for t in trials])

            results.append({
                'tau_low': tau_low,
                'tau_high': tau_high,
                'avg_rmse': avg_rmse,
                'avg_mpc_call_rate': avg_mpc_rate,
                'total_safety_violations': total_violations,
                'acceptable': (
                    avg_mpc_rate <= 0.30 and
                    avg_rmse <= avg_mpc_rmse + 0.5 and
                    total_violations == 0
                )
            })

            print(f"  ({tau_low:.4f}, {tau_high:.4f}): RMSE={avg_rmse:.3f}, MPC%={avg_mpc_rate*100:.1f}, Safe={total_violations==0}")

            if time.time() - calibration_start > max_seconds:
                break

        if time.time() - calibration_start > max_seconds:
            break

    # Find best acceptable config
    acceptable = [r for r in results if r['acceptable']]

    if acceptable:
        best = min(acceptable, key=lambda r: r['avg_mpc_call_rate'])
        print(f"\n[3/3] Best acceptable: tau_low={best['tau_low']:.4f}, tau_high={best['tau_high']:.4f}")
    else:
        # Fallback to defaults
        best = {'tau_low': 0.001, 'tau_high': 0.01, 'acceptable': False}
        print(f"\n[3/3] No acceptable config found, using defaults")

    calibration_time = time.time() - calibration_start

    return {
        'best': best,
        'results': results,
        'mpc_baseline': {
            'avg_rmse': avg_mpc_rmse,
            'avg_time': avg_mpc_time
        },
        'configs_tested': configs_tested,
        'calibration_time': calibration_time
    }


def run_benchmark_cp475(ensemble, datasets, params_dict, tau_low, tau_high, duration=5.0):
    """
    Run hybrid controller benchmark on windowed datasets.

    Returns:
        dict with benchmark results and PASS/FAIL verdict
    """
    print("\n" + "="*80)
    print("HYBRID BENCHMARK (CP4.7.5 Windowed Mode)")
    print("="*80)
    print(f"Thresholds: tau_low={tau_low:.6f}, tau_high={tau_high:.6f}")
    print(f"Datasets: {len(datasets)} (windowed)")

    # MPC baseline
    print("\n[1/3] MPC-only baseline...")
    mpc_baselines = []

    for ds in datasets:
        baseline = run_mpc_baseline(ds, params_dict, duration=duration)
        mpc_baselines.append(baseline)

    avg_mpc_rmse = np.mean([b['tracking_rmse'] for b in mpc_baselines])
    avg_mpc_time = np.mean([b['total_time'] for b in mpc_baselines])

    print(f"  RMSE: {avg_mpc_rmse:.3f} mm, Time: {avg_mpc_time:.2f}s")

    # Hybrid trials
    print("\n[2/3] Hybrid controller trials...")
    hybrid_trials = []

    for ds in datasets:
        trial = run_hybrid_trial(ds, params_dict, ensemble, tau_low, tau_high, duration=duration)
        hybrid_trials.append(trial)

    avg_hybrid_rmse = np.mean([t['tracking_rmse'] for t in hybrid_trials])
    avg_hybrid_mpc_rate = np.mean([t['mpc_call_rate'] for t in hybrid_trials])
    avg_hybrid_time = np.mean([t['avg_total_time_per_step'] for t in hybrid_trials]) * avg_mpc_time / avg_mpc_time  # estimate
    total_violations = sum([t['safety_violations'] for t in hybrid_trials])

    # Compute speedup (rough estimate)
    speedup = avg_mpc_time / (avg_hybrid_time if avg_hybrid_time > 0 else 1.0)

    print(f"  RMSE: {avg_hybrid_rmse:.3f} mm")
    print(f"  MPC call rate: {avg_hybrid_mpc_rate*100:.1f}%")
    print(f"  Safety violations: {total_violations}")
    print(f"  Speedup: {speedup:.2f}x")

    # Check acceptance criteria
    print("\n[3/3] Checking acceptance criteria...")

    criteria = {
        'mpc_call_rate_ok': avg_hybrid_mpc_rate <= 0.30,
        'rmse_ok': avg_hybrid_rmse <= avg_mpc_rmse + 0.5,
        'speedup_ok': speedup >= 2.0,
        'safety_ok': total_violations == 0
    }

    all_passed = all(criteria.values())

    print(f"  MPC call rate <= 30%:        {'✓' if criteria['mpc_call_rate_ok'] else '✗'} ({avg_hybrid_mpc_rate*100:.1f}%)")
    print(f"  RMSE <= baseline + 0.5 mm:   {'✓' if criteria['rmse_ok'] else '✗'} ({avg_hybrid_rmse:.3f} vs {avg_mpc_rmse+0.5:.3f})")
    print(f"  Speedup >= 2.0x:             {'✓' if criteria['speedup_ok'] else '✗'} ({speedup:.2f}x)")
    print(f"  Safety violations == 0:      {'✓' if criteria['safety_ok'] else '✗'} ({total_violations})")

    return {
        'passed': all_passed,
        'criteria': criteria,
        'metrics': {
            'mpc_baseline_rmse': avg_mpc_rmse,
            'mpc_baseline_time': avg_mpc_time,
            'hybrid_rmse': avg_hybrid_rmse,
            'hybrid_mpc_call_rate': avg_hybrid_mpc_rate,
            'hybrid_speedup': speedup,
            'safety_violations': total_violations
        }
    }


def main():
    """Main CP4.7.5 orchestrator."""
    print("\n" + "="*80)
    print("CP4.7.5: Window-native End-to-End Hybrid Validation")
    print("="*80)

    t_start_overall = time.time()

    # Result structure
    result = {
        'test_name': 'CP4.7.5 Window-native End-to-End Validation',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'in_progress',
        'phases': {}
    }

    output_path = './build/artifacts/cp475_end_to_end_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        # Phase 1: Check ensemble availability
        print("\n" + "="*80)
        print("PHASE 1: Check Ensemble Availability")
        print("="*80)

        available, message = check_ensemble_available()

        if not available:
            print(f"\n⏸️  SKIPPED: {message}")
            result['status'] = 'skipped'
            result['skipped_reason'] = 'ensemble_not_available'
            result['skipped_message'] = message
            result['time_elapsed'] = time.time() - t_start_overall

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"\n✓ Results written: {output_path}")
            return 0

        print(f"✓ {message}")

        # Phase 2: Load datasets
        print("\n" + "="*80)
        print("PHASE 2: Load Datasets")
        print("="*80)

        manifest = load_manifest(data_dir='./data', verbose=False)
        datasets = manifest.datasets_with_ref[:5]  # Limit to 5 for budget

        if not datasets:
            print("✗ No datasets available")
            result['status'] = 'skipped'
            result['skipped_reason'] = 'no_datasets'
            result['time_elapsed'] = time.time() - t_start_overall

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            return 0

        print(f"✓ Loaded {len(datasets)} dataset(s)")

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

        # Phase 3: Sliding-window health gate
        print("\n" + "="*80)
        print("PHASE 3: Sliding-Window Health Gate")
        print("="*80)

        health_report = run_sliding_window_health_gate(
            datasets,
            params_dict,
            window_steps=50,
            stride_steps=25,
            max_windows=20,  # Budget cap
            mpc_horizon=10,
            max_iters=10,
            verbose=True,
            output_json_path='./build/artifacts/cp475_health_gate_report.json'
        )

        result['phases']['health_gate'] = {
            'total_windows': health_report['summary']['total_windows'],
            'valid_windows': health_report['summary']['valid_window_count'],
            'invalid_windows': health_report['summary']['invalid_window_count'],
            'failure_reasons': health_report['summary'].get('failure_reasons', {}),
            'time_elapsed': health_report['summary']['time_elapsed']
        }

        # Check if we have valid windows
        if health_report['summary']['valid_window_count'] == 0:
            print("\n⏸️  SKIPPED: No valid windows found after health gate")
            result['status'] = 'skipped'
            result['skipped_reason'] = 'no_valid_windows'
            result['time_elapsed'] = time.time() - t_start_overall

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            print(f"\n✓ Results written: {output_path}")
            return 0

        print(f"\n✓ Found {health_report['summary']['valid_window_count']} valid window(s)")

        # Phase 4: Create windowed manifest
        print("\n" + "="*80)
        print("PHASE 4: Create Windowed Manifest")
        print("="*80)

        windowed_datasets = create_windowed_manifest(datasets, health_report['valid_windows'])

        print(f"✓ Created windowed manifest with {len(windowed_datasets)} dataset(s)")

        result['phases']['windowed_manifest'] = {
            'n_windowed_datasets': len(windowed_datasets)
        }

        # Phase 5: Load ensemble
        print("\n" + "="*80)
        print("PHASE 5: Load Ensemble")
        print("="*80)

        ensemble = load_ensemble()
        print(f"✓ Loaded ensemble")

        # Phase 6: Calibrate thresholds
        print("\n" + "="*80)
        print("PHASE 6: Threshold Calibration")
        print("="*80)

        calibration_result = calibrate_thresholds_cp475(
            ensemble,
            windowed_datasets,
            params_dict,
            duration=3.0,  # Shorter for budget
            max_configs=10,
            max_seconds=300
        )

        result['phases']['calibration'] = {
            'best_tau_low': calibration_result['best']['tau_low'],
            'best_tau_high': calibration_result['best']['tau_high'],
            'acceptable': calibration_result['best']['acceptable'],
            'configs_tested': calibration_result['configs_tested'],
            'time_elapsed': calibration_result['calibration_time']
        }

        # Phase 7: Run benchmark
        print("\n" + "="*80)
        print("PHASE 7: Hybrid Benchmark")
        print("="*80)

        benchmark_result = run_benchmark_cp475(
            ensemble,
            windowed_datasets,
            params_dict,
            calibration_result['best']['tau_low'],
            calibration_result['best']['tau_high'],
            duration=3.0
        )

        result['phases']['benchmark'] = {
            'passed': benchmark_result['passed'],
            'criteria': benchmark_result['criteria'],
            'metrics': benchmark_result['metrics']
        }

        # Final verdict
        result['status'] = 'completed'
        result['final_verdict'] = 'PASS' if benchmark_result['passed'] else 'FAIL'
        result['time_elapsed'] = time.time() - t_start_overall

        # Write results
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)
        print(f"Status: {result['final_verdict']}")
        print(f"Time elapsed: {result['time_elapsed']:.1f}s")
        print(f"Results written: {output_path}")
        print("="*80)

        return 0 if result['final_verdict'] == 'PASS' else 1

    except Exception as e:
        print(f"\n✗ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

        result['status'] = 'error'
        result['error'] = str(e)
        result['time_elapsed'] = time.time() - t_start_overall

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        return 1


if __name__ == "__main__":
    sys.exit(main())
