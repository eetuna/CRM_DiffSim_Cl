#!/usr/bin/env python3
"""
CP5.1: Real-NPZ Windowed Ensemble - Full Closure Report

Nightly test that runs full CP5.1 pipeline:
1. Train ensemble on real NPZ windows (or load existing)
2. Run threshold calibration (budgeted)
3. Run quality benchmark on real windows
4. Generate closure report with quality gates

This validates whether training on real windows closes the CP5.0 domain shift gap.
"""

import sys
import os
import json
import time
import subprocess

sys.path.insert(0, os.path.dirname(__file__))


def check_ensemble_exists(prefix='cp51_real_ensemble'):
    """Check if CP5.1 ensemble already exists."""
    metadata_path = f'./build/artifacts/{prefix}_ensemble_metadata.json'
    return os.path.exists(metadata_path)


def train_ensemble(max_windows=20, window_steps=20):
    """Train CP5.1 ensemble on real windows."""
    print("="*80)
    print("[1/3] Training CP5.1 Ensemble on Real Windows")
    print("="*80)
    print()

    cmd = [
        'python3', 'python/train_cp51_real_window_ensemble.py',
        '--max_windows', str(max_windows),
        '--window_steps', str(window_steps),
        '--n_members', '3',
        '--epochs', '50'
    ]

    env = {**os.environ, 'PYTHONPATH': 'build:python'}
    result = subprocess.run(cmd, env=env)

    return result.returncode == 0


def run_quality_report(max_windows=20, window_steps=20):
    """Run quality benchmark with CP5.1 ensemble."""
    print("\n" + "="*80)
    print("[2/3] Running Quality Benchmark")
    print("="*80)
    print()

    # Temporarily override ensemble metadata to use CP5.1
    cp51_meta = './build/artifacts/cp51_real_ensemble_ensemble_metadata.json'
    cp45_meta = './build/artifacts/cp45_ensemble_dagger_ensemble_metadata.json'

    # Backup and override
    if os.path.exists(cp45_meta):
        backup_meta = './build/artifacts/cp45_ensemble_dagger_ensemble_metadata.json.cp51_backup'
        if not os.path.exists(backup_meta):
            os.system(f'cp {cp45_meta} {backup_meta}')

    os.system(f'cp {cp51_meta} {cp45_meta}')
    print("✓ Configured to use CP5.1 ensemble")

    # Run quality report
    cmd = [
        'python3', 'python/eval/cp479_real_window_quality_report.py',
        '--max_datasets', '2',
        '--max_windows', str(max_windows),
        '--window_steps', str(window_steps),
        '--stride_steps', str(window_steps),
        '--top_k', '3',
        '--output', './build/artifacts/cp51_quality_report.json'
    ]

    env = {**os.environ, 'PYTHONPATH': 'build:python'}
    result = subprocess.run(cmd, env=env)

    return result.returncode == 0


def evaluate_closure(quality_report_path='./build/artifacts/cp51_quality_report.json'):
    """Evaluate CP5.1 closure criteria."""
    print("\n" + "="*80)
    print("[3/3] Evaluating CP5.1 Closure Criteria")
    print("="*80)
    print()

    if not os.path.exists(quality_report_path):
        print("✗ Quality report not found")
        return None

    with open(quality_report_path, 'r') as f:
        quality_report = json.load(f)

    if quality_report.get('status') != 'completed':
        print("✗ Quality report incomplete")
        return None

    stats = quality_report.get('quality_stats', {})

    # CP5.1 closure gates (same as CP5.0, but expecting better performance)
    gates = {
        'mpc_rmse_ok': stats['mpc_rmse']['mean'] <= 50.0,  # Relaxed for real data
        'hybrid_rmse_ok': stats['hybrid_rmse']['mean'] <= stats['mpc_rmse']['mean'] + 0.5,
        'mpc_call_rate_ok': stats['mpc_call_rate']['mean'] <= 0.30,  # Target
        'speedup_ok': stats['speedup']['mean'] >= 2.0,
        'safety_ok': stats['total_safety_violations'] == 0
    }

    closure_report = {
        'test_name': 'CP5.1 Real-NPZ Windowed Ensemble Closure',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'quality_stats': stats,
        'quality_gates': gates,
        'closure_achieved': all(gates.values())
    }

    # Print evaluation
    print("Quality Metrics:")
    print(f"  MPC RMSE:         {stats['mpc_rmse']['mean']:.2f}mm (mean)")
    print(f"  Hybrid RMSE:      {stats['hybrid_rmse']['mean']:.2f}mm (mean)")
    print(f"  MPC call rate:    {stats['mpc_call_rate']['mean']*100:.1f}%")
    print(f"  Speedup:          {stats['speedup']['mean']:.2f}×")
    print(f"  Safety violations: {stats['total_safety_violations']}")
    print()

    print("Quality Gates:")
    for gate_name, passed in gates.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status:10s} {gate_name}")
    print()

    if closure_report['closure_achieved']:
        print("="*80)
        print("✓ CP5.1 CLOSURE ACHIEVED")
        print("  Real-NPZ windowed training resolved domain shift!")
        print("="*80)
    else:
        print("="*80)
        print("⏸ CP5.1 PARTIAL CLOSURE")
        failed_gates = [name for name, passed in gates.items() if not passed]
        print(f"  Failed gates: {', '.join(failed_gates)}")
        print("="*80)

    return closure_report


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CP5.1: Real-NPZ Closure Report')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training if ensemble exists')
    parser.add_argument('--max_windows', type=int, default=20,
                       help='Max windows for training and quality report')
    parser.add_argument('--window_steps', type=int, default=20,
                       help='Window size in steps')
    args = parser.parse_args()

    print("="*80)
    print("CP5.1: Real-NPZ Windowed Ensemble - Closure Report")
    print("="*80)
    print()

    t_start = time.time()

    # Step 1: Train ensemble (or skip if exists)
    if args.skip_training and check_ensemble_exists():
        print("[1/3] Using existing CP5.1 ensemble")
        print("  (--skip_training enabled)")
        training_success = True
    else:
        training_success = train_ensemble(
            max_windows=args.max_windows,
            window_steps=args.window_steps
        )

        if not training_success:
            print("\n✗ Training failed")
            return 1

    # Step 2: Run quality benchmark
    quality_success = run_quality_report(
        max_windows=args.max_windows,
        window_steps=args.window_steps
    )

    if not quality_success:
        print("\n✗ Quality benchmark failed")
        return 1

    # Step 3: Evaluate closure
    closure_report = evaluate_closure()

    if closure_report is None:
        print("\n✗ Closure evaluation failed")
        return 1

    # Save closure report
    output_path = './build/artifacts/cp51_closure_report.json'
    closure_report['total_time'] = time.time() - t_start

    with open(output_path, 'w') as f:
        json.dump(closure_report, f, indent=2)

    print(f"\n✓ Closure report saved: {output_path}")
    print(f"✓ Total time: {closure_report['total_time']/60:.1f} minutes")

    return 0 if closure_report['closure_achieved'] else 1


if __name__ == "__main__":
    sys.exit(main())
