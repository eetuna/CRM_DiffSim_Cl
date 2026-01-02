#!/usr/bin/env python3
"""
CP4.7.7: Real NPZ Triage Report

Runs sliding-window health gate across ALL real NPZ datasets (excluding goldens)
and generates comprehensive triage report for debugging/analysis.

Outputs:
- Summary table: counts by failure_reason
- Top-K failing windows with first_failure_step + failure_source
- Artifact JSON: build/artifacts/cp477_real_npz_triage.json

Labeled nightly. Budget-conscious with configurable limits.
"""

import sys
import os
import json
import time
import argparse
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))

from eval.cp47_health_gate import run_sliding_window_health_gate
from data.npz_manifest import load_manifest
import crm_diff_py


def filter_real_datasets(manifest):
    """
    Filter out golden datasets, keeping only real NPZ datasets.

    Args:
        manifest: DatasetManifest object

    Returns:
        List of real (non-golden) datasets
    """
    real_datasets = [
        ds for ds in manifest.datasets_with_ref
        if not ('cp476' in ds.filename.lower() and 'golden' in ds.filename.lower())
    ]
    return real_datasets


def generate_triage_summary(health_report, top_k=10):
    """
    Generate triage summary from health report.

    Args:
        health_report: Health gate report dict
        top_k: Number of top failing windows to include

    Returns:
        dict with summary, top failures, and statistics
    """
    summary = health_report['summary']
    invalid_windows = health_report['invalid_windows']

    # Count failures by reason
    failure_counts = summary.get('failure_reasons', {})

    # Sort invalid windows by dataset/position for consistent reporting
    sorted_invalid = sorted(
        invalid_windows,
        key=lambda w: (w['dataset_idx'], w['start_idx'])
    )

    # Extract top-K failures with detailed diagnostics
    top_failures = []
    for i, window in enumerate(sorted_invalid[:top_k]):
        details = window.get('details', {})
        failure_entry = {
            'rank': i + 1,
            'filename': window['filename'],
            'dataset_idx': window['dataset_idx'],
            'window_range': [window['start_idx'], window['end_idx']],
            'n_steps': window['n_steps'],
            'failure_reason': window['reason'],
            'first_failure_step': details.get('first_failure_step', -1),
            'failure_source': details.get('failure_source', 'unknown'),
            'diagnostic_info': {
                k: v for k, v in details.items()
                if k not in ['first_failure_step', 'failure_source', 'failures']
            }
        }
        top_failures.append(failure_entry)

    # Compute statistics
    stats = {
        'total_datasets': summary['total_datasets'],
        'total_windows': summary['total_windows'],
        'valid_windows': summary['valid_window_count'],
        'invalid_windows': summary['invalid_window_count'],
        'pass_rate': (
            summary['valid_window_count'] / summary['total_windows']
            if summary['total_windows'] > 0 else 0.0
        )
    }

    # Analyze failure step ranges (CP4.7.8)
    failure_step_ranges = {
        'early': 0,      # first_failure_step < 5
        'mid': 0,        # 5 <= first_failure_step < 20
        'late': 0,       # first_failure_step >= 20
        'unknown': 0     # first_failure_step == -1
    }

    for window in invalid_windows:
        step = window.get('details', {}).get('first_failure_step', -1)
        if step == -1:
            failure_step_ranges['unknown'] += 1
        elif step < 5:
            failure_step_ranges['early'] += 1
        elif step < 20:
            failure_step_ranges['mid'] += 1
        else:
            failure_step_ranges['late'] += 1

    return {
        'summary': failure_counts,
        'top_failures': top_failures,
        'statistics': stats,
        'failure_step_ranges': failure_step_ranges
    }


def print_triage_report(triage_summary):
    """
    Print formatted triage report to console.

    Args:
        triage_summary: Triage summary dict
    """
    print("\n" + "="*80)
    print("TRIAGE SUMMARY")
    print("="*80)

    stats = triage_summary['statistics']
    print(f"Total datasets:  {stats['total_datasets']}")
    print(f"Total windows:   {stats['total_windows']}")
    print(f"Valid windows:   {stats['valid_windows']} ({stats['pass_rate']*100:.1f}%)")
    print(f"Invalid windows: {stats['invalid_windows']} ({(1-stats['pass_rate'])*100:.1f}%)")

    print("\n" + "="*80)
    print("FAILURE COUNTS BY REASON")
    print("="*80)

    failure_counts = triage_summary['summary']
    if failure_counts:
        # Sort by count (descending)
        sorted_failures = sorted(
            failure_counts.items(),
            key=lambda x: -x[1]
        )
        for reason, count in sorted_failures:
            pct = (count / stats['invalid_windows'] * 100) if stats['invalid_windows'] > 0 else 0
            print(f"  {reason:30s}: {count:4d} ({pct:5.1f}%)")
    else:
        print("  (No failures)")

    # Print failure step ranges (CP4.7.8)
    print("\n" + "="*80)
    print("FAILURE STEP RANGES (when failures occur)")
    print("="*80)

    step_ranges = triage_summary['failure_step_ranges']
    total_invalid = stats['invalid_windows']
    for range_name, count in step_ranges.items():
        pct = (count / total_invalid * 100) if total_invalid > 0 else 0
        print(f"  {range_name:10s}: {count:4d} ({pct:5.1f}%)")

    print("\n" + "="*80)
    print(f"TOP-{len(triage_summary['top_failures'])} FAILING WINDOWS")
    print("="*80)

    for failure in triage_summary['top_failures']:
        print(f"\n[{failure['rank']}] {failure['filename']}")
        print(f"    Window: [{failure['window_range'][0]}:{failure['window_range'][1]}), {failure['n_steps']} steps")
        print(f"    Reason: {failure['failure_reason']}")
        print(f"    First failure step: {failure['first_failure_step']}")
        print(f"    Failure source: {failure['failure_source']}")
        if failure['diagnostic_info']:
            print(f"    Diagnostics: {failure['diagnostic_info']}")

    print("\n" + "="*80)


def main():
    """Main triage entry point."""
    parser = argparse.ArgumentParser(
        description='CP4.7.7: Real NPZ Triage Report'
    )
    parser.add_argument('--max_datasets', type=int, default=None,
                       help='Limit number of datasets to process')
    parser.add_argument('--max_windows', type=int, default=100,
                       help='Limit total windows to process (default: 100)')
    parser.add_argument('--window_steps', type=int, default=50,
                       help='Window size in steps (default: 50)')
    parser.add_argument('--stride_steps', type=int, default=25,
                       help='Stride between windows (default: 25)')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top failures to report (default: 10)')
    parser.add_argument('--output', type=str,
                       default='./build/artifacts/cp477_real_npz_triage.json',
                       help='Output JSON path')
    args = parser.parse_args()

    print("="*80)
    print("CP4.7.7: Real NPZ Triage Report")
    print("="*80)
    print(f"Budget: max {args.max_datasets or 'unlimited'} datasets, {args.max_windows} windows")
    print(f"Window config: {args.window_steps} steps, stride {args.stride_steps}")
    print()

    t_start = time.time()

    # Load datasets
    print("[1/3] Loading real NPZ datasets...")
    try:
        manifest = load_manifest(data_dir='./data', verbose=False)
        real_datasets = filter_real_datasets(manifest)

        if not real_datasets:
            print("⏸️  SKIPPED: No real NPZ datasets found")
            print("  All available datasets are either golden or have no references.")
            return 0

        if args.max_datasets:
            real_datasets = real_datasets[:args.max_datasets]

        print(f"✓ Loaded {len(real_datasets)} real dataset(s):")
        for ds in real_datasets:
            print(f"  - {ds.filename}")

    except Exception as e:
        print(f"✗ Failed to load datasets: {e}")
        return 1

    # Load physics parameters
    print("\n[2/3] Loading physics parameters...")
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

    # Run sliding-window health gate
    print("\n[3/3] Running sliding-window health gate (budgeted)...")
    health_report = run_sliding_window_health_gate(
        real_datasets,
        params_dict,
        window_steps=args.window_steps,
        stride_steps=args.stride_steps,
        max_windows=args.max_windows,
        mpc_horizon=10,
        max_iters=10,
        verbose=True,
        output_json_path=None  # We'll write our own JSON below
    )

    # Generate triage summary
    triage_summary = generate_triage_summary(health_report, top_k=args.top_k)

    # Print to console
    print_triage_report(triage_summary)

    # Write artifact JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    artifact = {
        'test_name': 'CP4.7.7 Real NPZ Triage Report',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'max_datasets': args.max_datasets,
            'max_windows': args.max_windows,
            'window_steps': args.window_steps,
            'stride_steps': args.stride_steps,
            'top_k': args.top_k
        },
        'triage_summary': triage_summary,
        'time_elapsed': time.time() - t_start
    }

    with open(args.output, 'w') as f:
        json.dump(artifact, f, indent=2)

    print(f"\n✓ Triage report written: {args.output}")
    print(f"✓ Time elapsed: {artifact['time_elapsed']:.1f}s")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
