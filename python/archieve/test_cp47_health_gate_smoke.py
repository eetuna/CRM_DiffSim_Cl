#!/usr/bin/env python3
"""
CP4.7.3: Health Gate Smoke Test

Fast smoke test for health gate preflight checks.
- Tests with 1 dataset only (fast)
- Verifies JSON report is written
- Checks schema contains required fields
- Ensures no crash/hang

Target runtime: <30s
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from eval.cp47_health_gate import run_health_gate
from data.npz_manifest import load_manifest
import crm_diff_py


def main():
    """Health gate smoke test."""
    print("="*60)
    print("CP4.7.3: Health Gate Smoke Test")
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

    # Run health gate
    print("\n[3/3] Running health gate...")
    output_path = './build/artifacts/cp47_health_gate_smoke_report.json'

    report = run_health_gate(
        datasets,
        params_dict,
        duration=1.0,  # Fast test: only 1 second per dataset
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
    required_fields = ['test_name', 'timestamp', 'summary', 'valid_datasets', 'invalid_datasets']
    missing = [field for field in required_fields if field not in saved_report]

    if missing:
        print(f"\n✗ FAIL: Missing required fields in JSON: {missing}")
        return 1

    # Verify summary contains counts
    summary = saved_report['summary']
    required_summary_fields = ['total_datasets', 'valid_count', 'invalid_count']
    missing_summary = [field for field in required_summary_fields if field not in summary]

    if missing_summary:
        print(f"\n✗ FAIL: Missing required summary fields: {missing_summary}")
        return 1

    # Check that we have at least one outcome (valid or invalid)
    total = summary['total_datasets']
    valid = summary['valid_count']
    invalid = summary['invalid_count']

    if total != (valid + invalid):
        print(f"\n✗ FAIL: Count mismatch: {total} != {valid} + {invalid}")
        return 1

    # Print summary
    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)
    print(f"Time elapsed: {t_elapsed:.1f}s")
    print(f"Datasets tested: {total}")
    print(f"Valid: {valid}")
    print(f"Invalid: {invalid}")

    if invalid > 0:
        print("\nFailure reasons:")
        for reason, count in summary.get('failure_reasons', {}).items():
            print(f"  - {reason}: {count}")

    print("="*60)
    print("✓ SMOKE TEST PASSED")
    print("="*60)

    # Check timing (should be < 30s)
    if t_elapsed > 30.0:
        print(f"\n⚠ WARNING: Test took {t_elapsed:.1f}s (target: <30s)")
        print("  This may cause CI timeout issues")

    return 0


if __name__ == "__main__":
    sys.exit(main())
