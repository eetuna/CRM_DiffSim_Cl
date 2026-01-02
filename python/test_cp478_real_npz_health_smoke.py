#!/usr/bin/env python3
"""
CP4.7.8: Real NPZ Health Gate Smoke Test (with Hold-Period Mitigation)

Fast CI test that validates:
1. Real NPZ datasets can be processed with hold-aware windowing
2. At least one real dataset yields >=1 valid window after mitigation
3. Triage produces well-formed JSON artifact

Target runtime: <30s (CI gate)
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
    """Real NPZ health gate smoke test with hold-period mitigation."""
    print("="*80)
    print("CP4.7.8: Real NPZ Health Gate Smoke Test (Hold-Aware)")
    print("="*80)

    t_start = time.time()

    output_path = './build/artifacts/cp478_real_health_smoke_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result = {
        'test_name': 'CP4.7.8 Real NPZ Health Gate Smoke Test',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'in_progress'
    }

    try:
        # Load real datasets (exclude goldens)
        print("\n[1/3] Loading real NPZ datasets...")
        try:
            manifest = load_manifest(data_dir='./data', verbose=False)

            # Filter to only real (non-golden) datasets with references
            real_datasets = [
                ds for ds in manifest.datasets_with_ref
                if not ('cp476' in ds.filename.lower() and 'golden' in ds.filename.lower())
            ]

            if not real_datasets:
                print("⏸️  SKIPPED: No real NPZ datasets found")
                print("  All available datasets are either golden or have no references.")

                result['status'] = 'skipped'
                result['skipped_reason'] = 'no_real_datasets'
                result['time_elapsed'] = time.time() - t_start

                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)

                return 0  # Skip, not fail

            # Limit to first dataset for speed
            real_datasets = real_datasets[:1]

            print(f"✓ Loaded {len(real_datasets)} real dataset(s) for smoke test:")
            for ds in real_datasets:
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

        # Run health gate WITH hold-period mitigation (tight budget for <30s)
        print("\n[3/3] Running health gate with hold-aware windowing...")
        health_report = run_sliding_window_health_gate(
            real_datasets,
            params_dict,
            window_steps=10,       # Very small for speed
            stride_steps=10,       # No overlap
            max_windows=3,         # Very tight cap for <30s
            mpc_horizon=3,         # Minimal horizon
            max_iters=2,           # Very minimal iterations
            verbose=True,
            skip_hold_periods=True,  # CP4.7.8 mitigation
            output_json_path='./build/artifacts/cp478_real_health_report.json'
        )

        # Verify acceptance criteria
        valid_windows = health_report['summary']['valid_window_count']
        invalid_windows = health_report['summary']['invalid_window_count']
        total_windows = health_report['summary']['total_windows']

        # Acceptance: At least 1 valid window after mitigation
        if valid_windows >= 1:
            result['status'] = 'completed'
            result['verdict'] = 'PASS'
        else:
            result['status'] = 'completed'
            result['verdict'] = 'FAIL'
            result['failure_reason'] = f'No valid windows after mitigation (tested {total_windows} windows)'

        result['health_summary'] = {
            'total_windows': total_windows,
            'valid_windows': valid_windows,
            'invalid_windows': invalid_windows,
            'failure_reasons': health_report['summary'].get('failure_reasons', {})
        }
        result['mitigation_enabled'] = True
        result['time_elapsed'] = time.time() - t_start

        # Write results
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        # Print summary
        print("\n" + "="*80)
        print("SMOKE TEST SUMMARY")
        print("="*80)
        print(f"Time elapsed: {result['time_elapsed']:.1f}s")
        print(f"Mitigation: hold-aware windowing ENABLED")
        print(f"Windows tested: {total_windows}")
        print(f"Valid: {valid_windows}")
        print(f"Invalid: {invalid_windows}")
        print()
        print(f"Acceptance: valid_windows >= 1: {'✓ PASS' if valid_windows >= 1 else '✗ FAIL'}")
        print("="*80)

        if result['verdict'] == 'PASS':
            print("✓ SMOKE TEST PASSED")
            print(f"  Mitigation successfully enabled {valid_windows} valid window(s) on real datasets")
            print("="*80)
            return 0
        else:
            print("✗ SMOKE TEST FAILED")
            print(f"  {result['failure_reason']}")
            print("="*80)
            return 1

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
