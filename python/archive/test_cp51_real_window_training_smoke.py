#!/usr/bin/env python3
"""
CP5.1: Real-NPZ Windowed Ensemble Training - Smoke Test

Fast CI test (<60s) that verifies windowed ensemble training works.

Acceptance:
- Collects at least 1 valid window from real NPZ
- Trains tiny ensemble (1 member, 10 epochs) without errors
- Artifacts exist and metrics are finite
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from train_cp51_real_window_ensemble import (
    filter_real_datasets,
    train_ensemble_on_windows
)
from eval.cp47_health_gate import run_sliding_window_health_gate, create_windowed_manifest
from data.npz_manifest import load_manifest
import crm_diff_py


def main():
    print("="*80)
    print("CP5.1: Real-NPZ Windowed Training - Smoke Test")
    print("="*80)
    print("Target: <60s, 1 member, 1 dataset, 2 windows, 10 epochs")
    print()

    t_start = time.time()

    output_path = './build/artifacts/cp51_smoke_results.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    result = {
        'test_name': 'CP5.1 Windowed Training Smoke Test',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'status': 'in_progress'
    }

    try:
        # Load 1 real dataset
        print("[1/4] Loading real NPZ datasets...")
        manifest = load_manifest(data_dir='./data', verbose=False)
        real_datasets = filter_real_datasets(manifest.datasets_with_ref)

        if not real_datasets:
            print("⏸️  SKIPPED: No real datasets found")
            result['status'] = 'skipped'
            result['skipped_reason'] = 'no_real_datasets'
            result['time_elapsed'] = time.time() - t_start

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            return 0

        # Use only first dataset
        real_datasets = real_datasets[:1]
        print(f"✓ Using: {real_datasets[0].filename}")

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

        # Collect 2 windows (minimal)
        print("\n[3/4] Collecting valid windows...")
        health_report = run_sliding_window_health_gate(
            real_datasets,
            params_dict,
            window_steps=10,  # Small windows
            stride_steps=10,
            max_windows=2,    # Minimal
            mpc_horizon=3,    # Fast
            max_iters=3,      # Fast
            verbose=False,
            skip_hold_periods=True
        )

        valid_window_count = health_report['summary']['valid_window_count']

        if valid_window_count == 0:
            print("✗ No valid windows found")
            result['status'] = 'failed'
            result['failure_reason'] = 'no_valid_windows'
            result['time_elapsed'] = time.time() - t_start

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            return 1

        print(f"✓ Found {valid_window_count} valid window(s)")

        # Create windowed manifest
        windowed_datasets = create_windowed_manifest(real_datasets, health_report['valid_windows'])

        # Train tiny ensemble (1 member, 10 epochs)
        print("\n[4/4] Training tiny ensemble...")
        checkpoint_paths, metrics = train_ensemble_on_windows(
            windowed_datasets,
            params_dict,
            n_members=1,           # Minimal
            seeds=[42],
            epochs=10,             # Fast
            batch_size=32,
            mpc_horizon=3,         # Fast
            max_iters=3,
            output_prefix='cp51_smoke_ensemble',
            verbose=False
        )

        if checkpoint_paths is None:
            print("✗ Training failed")
            result['status'] = 'failed'
            result['failure_reason'] = 'training_failed'
            result['time_elapsed'] = time.time() - t_start

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            return 1

        # Verify artifacts
        metadata_path = './build/artifacts/cp51_smoke_ensemble_ensemble_metadata.json'

        if not os.path.exists(metadata_path):
            print("✗ Ensemble metadata not created")
            result['status'] = 'failed'
            result['failure_reason'] = 'metadata_missing'
            result['time_elapsed'] = time.time() - t_start

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            return 1

        # Verify metrics
        if not np.isfinite(metrics['member_metrics'][0]['best_val_loss']):
            print("✗ Non-finite validation loss")
            result['status'] = 'failed'
            result['failure_reason'] = 'non_finite_loss'
            result['time_elapsed'] = time.time() - t_start

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            return 1

        # Success
        t_elapsed = time.time() - t_start

        result['status'] = 'passed'
        result['verdict'] = 'PASS'
        result['time_elapsed'] = t_elapsed
        result['windows_collected'] = metrics['n_windows_collected']
        result['samples_collected'] = metrics['n_samples']
        result['best_val_loss'] = float(metrics['member_metrics'][0]['best_val_loss'])

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        # Print summary
        print("\n" + "="*80)
        print("SMOKE TEST SUMMARY")
        print("="*80)
        print(f"Time elapsed: {t_elapsed:.1f}s")
        print(f"Windows collected: {metrics['n_windows_collected']}")
        print(f"Training samples: {metrics['n_samples']}")
        print(f"Best val loss: {metrics['member_metrics'][0]['best_val_loss']:.6f}")
        print()
        print(f"Acceptance: time < 60s: {'✓ PASS' if t_elapsed < 60 else '✗ FAIL'}")
        print(f"Acceptance: windows >= 1: ✓ PASS")
        print(f"Acceptance: finite loss: ✓ PASS")
        print("="*80)
        print("✓ SMOKE TEST PASSED")
        print("="*80)

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
