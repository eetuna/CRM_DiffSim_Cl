#!/usr/bin/env python3
"""
CP5.0: Threshold Calibration for Real NPZ Performance

Calibrates (tau_low, tau_high) thresholds using CP5.0 ensemble trained on real data.
Optimizes for minimal MPC call rate while preserving RMSE ≤ MPC + 0.5 mm.

Uses real NPZ datasets with hold-aware windowing (CP4.7.8).
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from calibrate_cp47_thresholds import (
    calibrate_thresholds,
    load_ensemble_from_cp45,
    MPCExpert
)
from models.ensemble_policy import EnsemblePolicy
from models.recurrent_policy import GRUPolicy
from data.npz_manifest import load_manifest
from eval.cp47_health_gate import run_health_gate
import crm_diff_py


def filter_real_datasets(datasets):
    """Filter to real NPZ datasets only."""
    return [
        ds for ds in datasets
        if not ('cp476' in ds.filename.lower() and 'golden' in ds.filename.lower())
    ]


def load_cp50_ensemble():
    """Load CP5.0 ensemble trained on real NPZ data."""
    metadata_path = './build/artifacts/cp50_real_ensemble_ensemble_metadata.json'

    if not os.path.exists(metadata_path):
        print(f"✗ CP5.0 ensemble not found: {metadata_path}")
        print("  Run: python3 python/train_cp50_real_ensemble.py first")
        return None

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    missing = [p for p in metadata['checkpoint_paths'] if not os.path.exists(p)]
    if missing:
        print(f"✗ Missing checkpoints: {missing}")
        return None

    ensemble = EnsemblePolicy(
        policy_paths=metadata['checkpoint_paths'],
        policy_class=GRUPolicy,
        policy_kwargs=metadata['policy_kwargs']
    )

    print(f"✓ Loaded CP5.0 ensemble: {metadata['n_members']} members")
    return ensemble


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CP5.0: Threshold Calibration on Real NPZ')
    parser.add_argument('--max_configs', type=int, default=10,
                       help='Maximum number of configurations to test (default: 10)')
    parser.add_argument('--max_seconds', type=int, default=None,
                       help='Maximum time budget in seconds')
    args = parser.parse_args()

    print("="*80)
    print("CP5.0: Threshold Calibration for Real NPZ Performance")
    print("="*80)
    print(f"Budget: max {args.max_configs} configurations")
    print()

    # Load CP5.0 ensemble
    print("[Loading CP5.0 ensemble...]")
    ensemble = load_cp50_ensemble()

    if ensemble is None:
        print("\n✗ Cannot proceed without CP5.0 ensemble")
        return 1

    # Load real datasets only
    print("\n[Loading real NPZ datasets...]")
    manifest = load_manifest(data_dir='./data', verbose=False)
    all_datasets = filter_real_datasets(manifest.datasets_with_ref)

    if not all_datasets:
        print("✗ No real datasets found")
        return 1

    # Limit to 2 datasets for faster calibration
    datasets = all_datasets[:2]
    print(f"✓ Using {len(datasets)} real dataset(s):")
    for ds in datasets:
        print(f"  - {ds.filename}")

    # Load physics parameters
    print("\n[Loading physics parameters...]")
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

    # Run health gate first (CP4.7.3 requirement)
    print("\n[Running health gate...]")
    health_report = run_health_gate(
        datasets,
        params_dict,
        duration=2.0,
        verbose=True,
        output_json_path='./build/artifacts/cp50_health_report.json'
    )

    valid_indices = health_report['valid_datasets']
    if not valid_indices:
        print("\n✗ No valid datasets after health gate")
        return 1

    datasets = [datasets[i] for i in valid_indices]
    print(f"\n✓ Using {len(datasets)} valid dataset(s)")

    # Define threshold ranges (focused search)
    tau_low_range = [1e-4, 5e-4, 1e-3, 2e-3]
    tau_high_range = [5e-3, 1e-2, 2e-2]

    # Run calibration
    t_start = time.time()

    results = calibrate_thresholds(
        ensemble,
        datasets,
        params_dict,
        tau_low_range,
        tau_high_range,
        duration=3.0,
        max_configs=args.max_configs,
        max_seconds=args.max_seconds
    )

    t_elapsed = time.time() - t_start

    # Save results
    output_dir = './build/artifacts'
    os.makedirs(output_dir, exist_ok=True)

    # Full sweep results
    sweep_path = os.path.join(output_dir, 'cp50_threshold_sweep.json')
    with open(sweep_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved sweep results: {sweep_path}")

    # Best thresholds
    best_path = os.path.join(output_dir, 'cp50_threshold_best.json')

    if results['best_thresholds'] is not None:
        best_summary = {
            'tau_low': results['best_thresholds']['tau_low'],
            'tau_high': results['best_thresholds']['tau_high'],
            'metrics': {
                'mpc_call_rate': results['best_thresholds']['mpc_call_rate'],
                'tracking_rmse': results['best_thresholds']['tracking_rmse'],
                'speedup': results['best_thresholds']['speedup'],
                'safety_violations': results['best_thresholds']['n_failures']
            },
            'acceptance': results['best_thresholds']['acceptance'],
            'calibration_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'ensemble': 'cp50_real_ensemble'
        }

        with open(best_path, 'w') as f:
            json.dump(best_summary, f, indent=2)
        print(f"✓ Saved best thresholds: {best_path}")
    else:
        print(f"⚠ No thresholds found (budget too tight)")

    # Summary
    print("\n" + "="*80)
    print("CALIBRATION COMPLETE")
    print("="*80)
    print(f"Total time: {t_elapsed:.1f}s")
    print(f"Configurations tested: {results.get('configs_tested', results['n_total'])}")
    print(f"Passing all criteria: {results['n_passing']}")

    if results['best_thresholds'] is not None:
        print(f"\nRecommended thresholds:")
        print(f"  tau_low:  {results['best_thresholds']['tau_low']:.6f}")
        print(f"  tau_high: {results['best_thresholds']['tau_high']:.6f}")
        print(f"  MPC call rate: {results['best_thresholds']['mpc_call_rate']:.1%}")

    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
