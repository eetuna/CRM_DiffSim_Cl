#!/usr/bin/env python3
"""
CP4.7.5: Windowed End-to-End Smoke Test

Fast smoke test for windowed end-to-end validation pipeline.
- Minimal budget (<60s)
- Proves pipeline produces well-formed results JSON
- Can PASS or SKIP with reason
- Never hangs

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
    """Windowed end-to-end smoke test."""
    print("="*60)
    print("CP4.7.5: Windowed End-to-End Smoke Test")
    print("="*60)

    t_start = time.time()

    output_path = './build/artifacts/cp475_smoke_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result = {
        'test_name': 'CP4.7.5 Windowed End-to-End Smoke Test',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'in_progress'
    }

    try:
        # Load datasets (limit to 1 for speed)
        print("\n[1/4] Loading datasets...")
        try:
            manifest = load_manifest(data_dir='./data', verbose=False)
            datasets = manifest.datasets_with_ref[:1]

            if not datasets:
                print("⏸️  SKIPPED: No datasets with references found")
                result['status'] = 'skipped'
                result['skipped_reason'] = 'no_datasets'
                result['time_elapsed'] = time.time() - t_start

                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)

                return 0  # Skip, not fail

            print(f"✓ Loaded {len(datasets)} dataset(s)")

        except Exception as e:
            print(f"✗ Failed to load datasets: {e}")
            result['status'] = 'error'
            result['error'] = str(e)
            result['time_elapsed'] = time.time() - t_start

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            return 1

        # Load physics parameters
        print("\n[2/4] Loading physics parameters...")
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

        # Run sliding-window health gate (minimal budget)
        print("\n[3/4] Running sliding-window health gate...")
        health_report = run_sliding_window_health_gate(
            datasets,
            params_dict,
            window_steps=10,   # Minimal window
            stride_steps=10,   # No overlap
            max_windows=3,     # Hard cap for smoke test
            mpc_horizon=5,     # Short horizon
            max_iters=3,       # Minimal iterations
            verbose=True,
            output_json_path='./build/artifacts/cp475_smoke_health_report.json'
        )

        # Verify health report structure
        print("\n[4/4] Verifying pipeline output...")

        required_fields = ['test_name', 'timestamp', 'window_config', 'summary', 'valid_windows', 'invalid_windows']
        missing = [f for f in required_fields if f not in health_report]

        if missing:
            print(f"✗ Health report missing fields: {missing}")
            result['status'] = 'error'
            result['error'] = f"Missing fields: {missing}"
            result['time_elapsed'] = time.time() - t_start

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            return 1

        total_windows = health_report['summary']['total_windows']
        valid_windows = health_report['summary']['valid_window_count']
        invalid_windows = health_report['summary']['invalid_window_count']

        if total_windows != (valid_windows + invalid_windows):
            print(f"✗ Count mismatch: {total_windows} != {valid_windows} + {invalid_windows}")
            result['status'] = 'error'
            result['error'] = 'Window count mismatch'
            result['time_elapsed'] = time.time() - t_start

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            return 1

        # Test windowed manifest creation if valid windows exist
        if valid_windows > 0:
            print(f"\n[Bonus] Testing windowed manifest creation...")
            windowed_datasets = create_windowed_manifest(datasets, health_report['valid_windows'])

            if len(windowed_datasets) != valid_windows:
                print(f"✗ Windowed manifest count mismatch: {len(windowed_datasets)} != {valid_windows}")
                result['status'] = 'error'
                result['error'] = 'Windowed manifest count mismatch'
                result['time_elapsed'] = time.time() - t_start

                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)

                return 1

            print(f"✓ Created windowed manifest with {len(windowed_datasets)} dataset(s)")

        # Success
        result['status'] = 'completed'
        result['verdict'] = 'PASS'
        result['health_summary'] = {
            'total_windows': total_windows,
            'valid_windows': valid_windows,
            'invalid_windows': invalid_windows
        }
        result['time_elapsed'] = time.time() - t_start

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print("\n" + "="*60)
        print("SMOKE TEST SUMMARY")
        print("="*60)
        print(f"Time elapsed: {result['time_elapsed']:.1f}s")
        print(f"Windows tested: {total_windows}")
        print(f"Valid: {valid_windows}")
        print(f"Invalid: {invalid_windows}")
        print("="*60)
        print("✓ SMOKE TEST PASSED")
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
