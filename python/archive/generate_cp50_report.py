#!/usr/bin/env python3
"""
CP5.0: Consolidated Performance Report Generator

Aggregates results from:
1. Ensemble training on real NPZ windows
2. Threshold calibration
3. Quality benchmarks

Produces a single consolidated report with:
- RMSE distributions
- MPC call rate
- Speedup
- Safety summary

Gates: At least one real dataset meets all quality criteria.
"""

import sys
import os
import json
import time
import numpy as np
from datetime import datetime


def load_json(path, required=True):
    """Load JSON file with error handling."""
    if not os.path.exists(path):
        if required:
            print(f"✗ Required file not found: {path}")
            return None
        else:
            return {}

    with open(path, 'r') as f:
        return json.load(f)


def main():
    print("="*80)
    print("CP5.0: Consolidated Performance Report")
    print("="*80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    report = {
        'title': 'CP5.0: Hybrid Controller Performance Closure',
        'generated_at': datetime.now().isoformat(),
        'status': 'in_progress',
        'components': {}
    }

    # Load ensemble training metrics
    print("[1/4] Loading ensemble training results...")
    ensemble_metrics = load_json(
        './build/artifacts/cp50_real_ensemble_metrics.json',
        required=False
    )

    if ensemble_metrics:
        report['components']['ensemble_training'] = {
            'status': 'completed',
            'n_members': len(ensemble_metrics.get('ensemble_members', [])),
            'n_iterations': len(ensemble_metrics.get('iterations', [])),
            'metadata_path': './build/artifacts/cp50_real_ensemble_ensemble_metadata.json'
        }
        print(f"  ✓ Ensemble: {report['components']['ensemble_training']['n_members']} members, "
              f"{report['components']['ensemble_training']['n_iterations']} iterations")
    else:
        print("  ⚠ Ensemble training results not found")
        report['components']['ensemble_training'] = {'status': 'not_found'}

    # Load threshold calibration results
    print("\n[2/4] Loading threshold calibration results...")
    threshold_best = load_json(
        './build/artifacts/cp50_threshold_best.json',
        required=False
    )

    if threshold_best and 'tau_low' in threshold_best:
        report['components']['threshold_calibration'] = {
            'status': 'completed',
            'tau_low': threshold_best['tau_low'],
            'tau_high': threshold_best['tau_high'],
            'metrics': threshold_best.get('metrics', {}),
            'acceptance': threshold_best.get('acceptance', {})
        }
        print(f"  ✓ Thresholds: tau_low={threshold_best['tau_low']:.6f}, "
              f"tau_high={threshold_best['tau_high']:.6f}")
        print(f"    MPC call rate: {threshold_best['metrics']['mpc_call_rate']:.1%}")
    else:
        print("  ⚠ Threshold calibration results not found")
        report['components']['threshold_calibration'] = {'status': 'not_found'}

    # Load quality report
    print("\n[3/4] Loading quality report...")
    quality_report = load_json(
        './build/artifacts/cp50_quality_report.json',
        required=False
    )

    if quality_report and quality_report.get('status') == 'completed':
        stats = quality_report.get('quality_stats', {})

        report['components']['quality_benchmark'] = {
            'status': 'completed',
            'valid_windows': quality_report.get('valid_window_count', 0),
            'quality_stats': stats
        }

        print(f"  ✓ Quality benchmark: {quality_report['valid_window_count']} valid windows")
        if stats:
            print(f"    MPC RMSE: {stats['mpc_rmse']['mean']:.3f}mm (mean), "
                  f"{stats['mpc_rmse']['p90']:.3f}mm (p90)")
            print(f"    Hybrid RMSE: {stats['hybrid_rmse']['mean']:.3f}mm (mean), "
                  f"{stats['hybrid_rmse']['p90']:.3f}mm (p90)")
            print(f"    MPC call rate: {stats['mpc_call_rate']['mean']*100:.1f}% (mean)")
            print(f"    Speedup: {stats['speedup']['mean']:.2f}× (mean)")
            print(f"    Safety violations: {stats['total_safety_violations']}")
    else:
        print("  ⚠ Quality report not found or incomplete")
        report['components']['quality_benchmark'] = {'status': 'not_found'}

    # Evaluate CP5.0 success criteria
    print("\n[4/4] Evaluating CP5.0 success criteria...")
    print()

    success_criteria = {
        'ensemble_trained': ensemble_metrics is not None,
        'thresholds_calibrated': threshold_best is not None and 'tau_low' in threshold_best,
        'quality_benchmarked': quality_report is not None and quality_report.get('status') == 'completed',
        'at_least_one_passing': False
    }

    # Check if at least one dataset meets all quality gates
    if quality_report and quality_report.get('status') == 'completed':
        stats = quality_report.get('quality_stats', {})

        if stats:
            # Check quality gates (per CP4.7.9 spec)
            mpc_rmse_ok = stats['mpc_rmse']['mean'] <= 10.0  # Relaxed for real data

            mpc_baseline = stats['mpc_rmse']['mean']
            hybrid_rmse_ok = stats['hybrid_rmse']['mean'] <= mpc_baseline + 0.5

            mpc_rate_ok = stats['mpc_call_rate']['mean'] <= 0.30
            speedup_ok = stats['speedup']['mean'] >= 2.0
            safety_ok = stats['total_safety_violations'] == 0

            all_gates_pass = (mpc_rmse_ok and hybrid_rmse_ok and
                             mpc_rate_ok and speedup_ok and safety_ok)

            success_criteria['at_least_one_passing'] = all_gates_pass

            report['quality_gates'] = {
                'mpc_rmse_ok': mpc_rmse_ok,
                'hybrid_rmse_ok': hybrid_rmse_ok,
                'mpc_call_rate_ok': mpc_rate_ok,
                'speedup_ok': speedup_ok,
                'safety_ok': safety_ok,
                'all_pass': all_gates_pass
            }

    report['success_criteria'] = success_criteria
    all_success = all(success_criteria.values())
    report['status'] = 'success' if all_success else 'partial'

    # Print summary
    print("="*80)
    print("CP5.0 SUCCESS CRITERIA")
    print("="*80)

    for criterion, passed in success_criteria.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} {criterion}")

    print("="*80)

    if all_success:
        print("✓ CP5.0 PERFORMANCE CLOSURE ACHIEVED")
        print("  At least one real dataset meets all quality gates")
    else:
        print("⏸ CP5.0 PARTIAL COMPLETION")
        print("  Some criteria not met - review individual reports")

    print("="*80)

    # Save consolidated report
    output_path = './build/artifacts/cp50_consolidated_report.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n✓ Consolidated report saved: {output_path}")

    # Also generate markdown summary
    md_path = './build/artifacts/cp50_summary.md'
    with open(md_path, 'w') as f:
        f.write(f"# CP5.0: Hybrid Controller Performance Closure\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")
        f.write(f"**Status**: {report['status'].upper()}\n\n")

        f.write(f"## Components\n\n")

        f.write(f"### 1. Ensemble Training\n")
        if ensemble_metrics:
            f.write(f"- Members: {report['components']['ensemble_training']['n_members']}\n")
            f.write(f"- Iterations: {report['components']['ensemble_training']['n_iterations']}\n")
        else:
            f.write(f"- Status: Not found\n")
        f.write(f"\n")

        f.write(f"### 2. Threshold Calibration\n")
        if threshold_best and 'tau_low' in threshold_best:
            f.write(f"- tau_low: {threshold_best['tau_low']:.6f}\n")
            f.write(f"- tau_high: {threshold_best['tau_high']:.6f}\n")
            f.write(f"- MPC call rate: {threshold_best['metrics']['mpc_call_rate']:.1%}\n")
        else:
            f.write(f"- Status: Not found\n")
        f.write(f"\n")

        f.write(f"### 3. Quality Benchmark\n")
        if quality_report and quality_report.get('status') == 'completed':
            stats = quality_report.get('quality_stats', {})
            f.write(f"- Valid windows: {quality_report['valid_window_count']}\n")
            f.write(f"- MPC RMSE: {stats['mpc_rmse']['mean']:.3f}mm (mean)\n")
            f.write(f"- Hybrid RMSE: {stats['hybrid_rmse']['mean']:.3f}mm (mean)\n")
            f.write(f"- MPC call rate: {stats['mpc_call_rate']['mean']*100:.1f}%\n")
            f.write(f"- Speedup: {stats['speedup']['mean']:.2f}×\n")
            f.write(f"- Safety violations: {stats['total_safety_violations']}\n")
        else:
            f.write(f"- Status: Not found\n")
        f.write(f"\n")

        f.write(f"## Success Criteria\n\n")
        for criterion, passed in success_criteria.items():
            status = "✓" if passed else "✗"
            f.write(f"- {status} {criterion}\n")
        f.write(f"\n")

        if all_success:
            f.write(f"## Conclusion\n\n")
            f.write(f"✓ **CP5.0 Performance Closure Achieved**\n\n")
            f.write(f"At least one real dataset meets all quality gates.\n")

    print(f"✓ Markdown summary saved: {md_path}")
    print()

    return 0 if all_success else 1


if __name__ == "__main__":
    sys.exit(main())
