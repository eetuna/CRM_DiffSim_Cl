#!/usr/bin/env python3
"""
CP5.0: Quality Report on Real NPZ Windows

Runs CP4.7.9 quality report with CP5.0 ensemble and calibrated thresholds.
Validates performance on real NPZ windows with realistic budgets.

Outputs comprehensive quality metrics for CP5.0 performance closure.
"""

import sys
import os
import subprocess
import json

sys.path.insert(0, os.path.dirname(__file__))


def check_cp50_ensemble():
    """Check if CP5.0 ensemble is available."""
    metadata_path = './build/artifacts/cp50_real_ensemble_ensemble_metadata.json'
    return os.path.exists(metadata_path)


def check_cp50_thresholds():
    """Check if CP5.0 thresholds are calibrated."""
    best_path = './build/artifacts/cp50_threshold_best.json'
    return os.path.exists(best_path)


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CP5.0: Quality Report on Real NPZ')
    parser.add_argument('--max_windows', type=int, default=50,
                       help='Maximum windows to test (default: 50)')
    parser.add_argument('--window_steps', type=int, default=50,
                       help='Window size in steps (default: 50)')
    parser.add_argument('--stride_steps', type=int, default=25,
                       help='Stride between windows (default: 25)')
    args = parser.parse_args()

    print("="*80)
    print("CP5.0: Quality Report on Real NPZ Windows")
    print("="*80)
    print(f"Config: max_windows={args.max_windows}, window_steps={args.window_steps}, stride={args.stride_steps}")
    print()

    # Check prerequisites
    print("[Prerequisites]")

    if not check_cp50_ensemble():
        print("✗ CP5.0 ensemble not found")
        print("  Run: python3 python/train_cp50_real_ensemble.py first")
        return 1
    else:
        print("✓ CP5.0 ensemble found")

    if not check_cp50_thresholds():
        print("⚠ CP5.0 thresholds not calibrated (will use CP5.0 ensemble with default thresholds)")
    else:
        print("✓ CP5.0 thresholds calibrated")

    print()

    # Update ensemble symlink to use CP5.0 ensemble
    print("[Setting up CP5.0 ensemble for quality report...]")

    # Temporarily override ensemble metadata to use CP5.0
    cp50_meta = './build/artifacts/cp50_real_ensemble_ensemble_metadata.json'
    cp45_meta = './build/artifacts/cp45_ensemble_dagger_ensemble_metadata.json'

    # Backup original if it exists
    if os.path.exists(cp45_meta):
        backup_meta = './build/artifacts/cp45_ensemble_dagger_ensemble_metadata.json.cp50_backup'
        if not os.path.exists(backup_meta):
            os.system(f'cp {cp45_meta} {backup_meta}')

    # Copy CP5.0 metadata to CP4.5 location (scripts expect this path)
    os.system(f'cp {cp50_meta} {cp45_meta}')
    print("✓ Configured to use CP5.0 ensemble")

    # Also update threshold metadata if available
    if check_cp50_thresholds():
        cp50_thresh = './build/artifacts/cp50_threshold_best.json'
        cp47_thresh = './build/artifacts/cp47_threshold_best.json'

        if os.path.exists(cp47_thresh):
            backup_thresh = './build/artifacts/cp47_threshold_best.json.cp50_backup'
            if not os.path.exists(backup_thresh):
                os.system(f'cp {cp47_thresh} {backup_thresh}')

        os.system(f'cp {cp50_thresh} {cp47_thresh}')
        print("✓ Configured to use CP5.0 thresholds")

    print()

    # Run quality report
    print("[Running quality report...]")
    print()

    cmd = [
        'python3', 'python/eval/cp479_real_window_quality_report.py',
        '--max_windows', str(args.max_windows),
        '--window_steps', str(args.window_steps),
        '--stride_steps', str(args.stride_steps),
        '--output', './build/artifacts/cp50_quality_report.json'
    ]

    result = subprocess.run(cmd, env={**os.environ, 'PYTHONPATH': 'build:python'})

    print()

    if result.returncode == 0:
        print("="*80)
        print("✓ CP5.0 Quality Report Complete")
        print("="*80)
        print("Report: ./build/artifacts/cp50_quality_report.json")
        print("="*80)
    else:
        print("="*80)
        print("✗ Quality report failed")
        print("="*80)

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
