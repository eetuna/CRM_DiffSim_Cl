#!/usr/bin/env python3
"""
CP5.0: Train Ensemble on Real NPZ Windows

Trains ensemble using ONLY real NPZ datasets (excludes golden synthetic data).
Leverages CP4.7.8 hold-aware processing and CP4.7.4 windowing infrastructure.

Key differences from CP4.5:
- Filters to real-only datasets (excludes cp476_*_golden.npz)
- Uses fast_train profile for reasonable runtime
- Optimized for real-world performance validation
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from train_cp45_ensemble_dagger import run_ensemble_dagger, EnsembleDAggerConfig
from data.npz_manifest import load_manifest, create_splits
import crm_diff_py
import time


def filter_real_datasets(datasets):
    """Filter to real NPZ datasets only (exclude golden synthetic data)."""
    real_datasets = [
        ds for ds in datasets
        if not ('cp476' in ds.filename.lower() and 'golden' in ds.filename.lower())
    ]
    return real_datasets


def main():
    print("="*80)
    print("CP5.0: Ensemble Training on Real NPZ Windows")
    print("="*80)
    print("Strategy: Train on real-only datasets with hold-aware processing")
    print()

    # Load manifest and filter to real datasets
    print("[1/3] Loading real NPZ datasets...")
    manifest = load_manifest(data_dir='./data', verbose=False)

    # Create splits from full manifest first
    datasets_train_all, datasets_val_all, datasets_test_all = create_splits(
        manifest,
        val_ratio=0.2,
        test_ratio=0.2
    )

    # Filter each split to real datasets only
    datasets_train = filter_real_datasets(datasets_train_all)
    datasets_val = filter_real_datasets(datasets_val_all)
    datasets_test = filter_real_datasets(datasets_test_all)

    if not datasets_train:
        print("✗ No real NPZ datasets found in training split")
        return 1

    print(f"✓ Found {len(datasets_train) + len(datasets_val) + len(datasets_test)} real dataset(s) total:")
    print(f"  Train: {len(datasets_train)}")
    for ds in datasets_train:
        print(f"    - {ds.filename}")
    print(f"  Val: {len(datasets_val)}")
    for ds in datasets_val:
        print(f"    - {ds.filename}")
    print(f"  Test: {len(datasets_test)}")
    for ds in datasets_test:
        print(f"    - {ds.filename}")

    # Load catheter parameters
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

    # Use fast training profile for CP5.0
    print("\n[3/3] Starting ensemble training (fast profile)...")
    print("  Profile: fast_train (3 members, 2 iterations)")
    print("  Note: Hold periods automatically skipped via CP4.7.8 infrastructure")

    config = EnsembleDAggerConfig.create_fast_train()
    config.checkpoint_prefix = 'cp50_real_ensemble'  # CP5.0 prefix

    # Run ensemble DAgger
    t_start = time.time()
    ensemble_paths, metrics = run_ensemble_dagger(
        config,
        datasets_train,
        datasets_val,
        params_dict,
        verbose=True
    )
    t_elapsed = time.time() - t_start

    print("\n" + "="*80)
    print("✓ CP5.0 Ensemble Training Complete")
    print("="*80)
    print(f"Training time: {t_elapsed/60:.1f} minutes")
    print(f"Ensemble members: {len(ensemble_paths)}")
    print(f"Checkpoint prefix: {config.checkpoint_prefix}")
    print(f"Metadata: ./build/artifacts/{config.checkpoint_prefix}_ensemble_metadata.json")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
