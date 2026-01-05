#!/usr/bin/env python3
"""
CP4.7.6: Golden Windows Health Gate Smoke Test

Validates that CP4.7.6 golden NPZ datasets pass health gate checks.
Acceptance: At least 1 valid window exists across all golden datasets.

Target runtime: <60s
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from eval.cp47_health_gate import run_sliding_window_health_gate
from data.npz_manifest import load_manifest
import crm_diff_py


def main():
    """Golden health gate smoke test."""
    print("="*60)
    print("CP4.7.6: Golden Health Gate Smoke Test")
    print("="*60)

    t_start = time.time()

    output_path = './build/artifacts/cp476_golden_health_smoke_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result = {
        'test_name': 'CP4.7.6 Golden Health Gate Smoke Test',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'in_progress'
    }

    try:
        # Load datasets
        print("\n[1/3] Loading golden datasets...")
        try:
            manifest = load_manifest(data_dir='./data', verbose=False)

            # Filter to only cp476 golden datasets
            golden_datasets = [
                ds for ds in manifest.datasets_with_ref
                if 'cp476' in ds.filename.lower() and 'golden' in ds.filename.lower()
            ]

            if not golden_datasets:
                print("⏸️  SKIPPED: No CP4.7.6 golden datasets found")
                print("  Expected datasets:")
                print("    - cp476_circle_golden.npz")
                print("    - cp476_lemniscate_golden.npz")
                print("    - cp476_line_golden.npz")
                print("\n  Run: python3 python/data/generate_cp476_npz_goldens.py")

                result['status'] = 'skipped'
                result['skipped_reason'] = 'no_golden_datasets'
                result['time_elapsed'] = time.time() - t_start

                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)

                return 0  # Skip, not fail

            # For smoke test speed, only test first golden dataset
            golden_datasets = golden_datasets[:1]

            print(f"✓ Loaded {len(golden_datasets)} golden dataset(s) (smoke test limit):")
            for ds in golden_datasets:
                print(f"  - {ds.filename}")

        except Exception as e:
            print(f"✗ Failed to load datasets: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
            result['time_elapsed'] = time.time() - t_start

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

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

        # Run sliding-window health gate with minimal budget for smoke test
        print("\n[3/3] Running health gate on golden datasets...")
        health_report = run_sliding_window_health_gate(
            golden_datasets,
            params_dict,
            window_steps=20,    # Smaller windows for speed
            stride_steps=20,    # No overlap
            max_windows=10,     # Cap total windows for smoke test
            mpc_horizon=5,      # Short horizon
            max_iters=3,        # Minimal iterations
            verbose=True,
            output_json_path='./build/artifacts/cp476_golden_health_report.json'
        )

        total_windows = health_report['summary']['total_windows']
        valid_windows = health_report['summary']['valid_window_count']
        invalid_windows = health_report['summary']['invalid_window_count']

        # Acceptance check: At least 1 valid window must exist
        if valid_windows < 1:
            print("\n" + "="*60)
            print("SMOKE TEST FAILED")
            print("="*60)
            print(f"❌ Acceptance: valid_windows >= 1")
            print(f"   Got: {valid_windows}")
            print("\nFailure reasons:")
            for reason, count in health_report['summary'].get('failure_reasons', {}).items():
                print(f"  - {reason}: {count}")
            print("="*60)

            result['status'] = 'failed'
            result['verdict'] = 'FAIL'
            result['failure_reason'] = 'no_valid_windows'
            result['health_summary'] = {
                'total_windows': total_windows,
                'valid_windows': valid_windows,
                'invalid_windows': invalid_windows,
                'failure_reasons': health_report['summary'].get('failure_reasons', {})
            }
            result['time_elapsed'] = time.time() - t_start

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            return 1

        # Success
        result['status'] = 'completed'
        result['verdict'] = 'PASS'
        result['health_summary'] = {
            'total_windows': total_windows,
            'valid_windows': valid_windows,
            'invalid_windows': invalid_windows,
            'failure_reasons': health_report['summary'].get('failure_reasons', {})
        }
        result['time_elapsed'] = time.time() - t_start

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print("\n" + "="*60)
        print("SMOKE TEST SUMMARY")
        print("="*60)
        print(f"Time elapsed: {result['time_elapsed']:.1f}s")
        print(f"Windows tested: {total_windows}")
        print(f"Valid: {valid_windows} ✓")
        print(f"Invalid: {invalid_windows}")
        if invalid_windows > 0:
            print("\nFailure reasons:")
            for reason, count in health_report['summary'].get('failure_reasons', {}).items():
                print(f"  - {reason}: {count}")
        print("="*60)
        print("✓ SMOKE TEST PASSED")
        print(f"  Acceptance: valid_windows >= 1 ({valid_windows} found)")
        print("="*60)

        # Check timing (should be < 60s)
        if result['time_elapsed'] > 60.0:
            print(f"\n⚠ WARNING: Test took {result['time_elapsed']:.1f}s (target: <60s)")
            print("  This may cause CI timeout issues")

        return 0

    except Exception as e:
        print(f"\n✗ EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

        result['status'] = 'error'
        result['error'] = str(e)
        result['time_elapsed'] = time.time() - t_start

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        return 1


if __name__ == "__main__":
    sys.exit(main())
