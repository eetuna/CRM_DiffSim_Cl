#!/usr/bin/env python3
"""
CP4.7.4: Health Gate Sliding-Window Smoke Test

Fast smoke test for sliding-window health gate preflight checks.
- Tests with 1 dataset only (fast)
- Uses small window size for speed
- Verifies JSON report is written
- Checks schema contains required fields
- Ensures at least one window is classified (valid or invalid)
- Ensures no crash/hang

Target runtime: <60s
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from eval.cp47_health_gate import run_sliding_window_health_gate, create_windowed_manifest
from data.npz_manifest import load_manifest
import crm_diff_py


def main():
    """Sliding-window health gate smoke test."""
    print("="*60)
    print("CP4.7.4: Health Gate Sliding-Window Smoke Test")
    print("="*60)

    t_start = time.time()

    # Load datasets (limit to 1 for speed)
    print("\n[1/3] Loading datasets...")
    try:
        manifest = load_manifest(data_dir='./data', verbose=False)
        datasets = manifest.datasets_with_ref

        if not datasets:
            print("⏸️  SKIPPED: No datasets with references found")
            return 0  # Skip, not fail

        datasets = datasets[:1]  # Only test first dataset
        print(f"✓ Loaded {len(datasets)} dataset(s)")

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
    print("\n[3/3] Running sliding-window health gate...")
    output_path = './build/artifacts/cp47_health_gate_window_smoke_report.json'

    # Use minimal params for smoke test speed (<60s requirement)
    report = run_sliding_window_health_gate(
        datasets,
        params_dict,
        window_steps=10,   # Minimal window for speed
        stride_steps=10,   # No overlap
        max_windows=3,     # Only test 3 windows
        mpc_horizon=5,     # Short horizon for speed
        max_iters=3,       # Minimal iterations for speed
        verbose=True,
        output_json_path=output_path
    )

    t_elapsed = time.time() - t_start

    # Verify JSON was written
    if not os.path.exists(output_path):
        print(f"\n✗ FAIL: JSON report not written to {output_path}")
        return 1

    # Load and verify JSON schema
    with open(output_path, 'r') as f:
        saved_report = json.load(f)

    # Check required fields
    required_fields = ['test_name', 'timestamp', 'window_config', 'summary', 'valid_windows', 'invalid_windows']
    missing = [field for field in required_fields if field not in saved_report]

    if missing:
        print(f"\n✗ FAIL: Missing required fields in JSON: {missing}")
        return 1

    # Verify window_config contains window_steps and stride_steps
    window_config = saved_report['window_config']
    if 'window_steps' not in window_config or 'stride_steps' not in window_config:
        print(f"\n✗ FAIL: Missing window config fields")
        return 1

    # Verify summary contains counts
    summary = saved_report['summary']
    required_summary_fields = ['total_datasets', 'total_windows', 'valid_window_count', 'invalid_window_count']
    missing_summary = [field for field in required_summary_fields if field not in summary]

    if missing_summary:
        print(f"\n✗ FAIL: Missing required summary fields: {missing_summary}")
        return 1

    # Check that at least one window was tested
    total_windows = summary['total_windows']
    if total_windows == 0:
        print(f"\n✗ FAIL: No windows were tested")
        return 1

    # Check that we have at least one outcome (valid or invalid)
    valid = summary['valid_window_count']
    invalid = summary['invalid_window_count']

    if total_windows != (valid + invalid):
        print(f"\n✗ FAIL: Count mismatch: {total_windows} != {valid} + {invalid}")
        return 1

    # Test windowed manifest creation
    if valid > 0:
        print("\n[Bonus] Testing windowed manifest creation...")
        try:
            windowed_datasets = create_windowed_manifest(datasets, saved_report['valid_windows'])
            if len(windowed_datasets) != valid:
                print(f"✗ FAIL: Windowed manifest count mismatch: {len(windowed_datasets)} != {valid}")
                return 1
            print(f"✓ Created windowed manifest with {len(windowed_datasets)} dataset(s)")
        except Exception as e:
            print(f"✗ FAIL: Windowed manifest creation failed: {e}")
            return 1

    # Print summary
    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)
    print(f"Time elapsed: {t_elapsed:.1f}s")
    print(f"Windows tested: {total_windows}")
    print(f"Valid: {valid}")
    print(f"Invalid: {invalid}")

    if invalid > 0:
        print("\nFailure reasons:")
        for reason, count in summary.get('failure_reasons', {}).items():
            print(f"  - {reason}: {count}")

    print("="*60)
    print("✓ SMOKE TEST PASSED")
    print("="*60)

    # Check timing (should be < 60s)
    if t_elapsed > 60.0:
        print(f"\n⚠ WARNING: Test took {t_elapsed:.1f}s (target: <60s)")
        print("  This may cause CI timeout issues")

    return 0


if __name__ == "__main__":
    sys.exit(main())
