#!/usr/bin/env python3
"""
CP4.6: Multi-task DAgger Smoke Test

Quick validation that:
1. Manifest loads both circle and lemniscate datasets
2. Both trajectory types have references
3. Training data can be mixed
4. Single policy can handle both types

Runtime target: <60 seconds
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from data.npz_manifest import load_manifest
from models.recurrent_policy import GRUPolicy


def test_manifest_has_both_types():
    """Test 1: Manifest has circle and lemniscate datasets with references."""
    print("Test 1: Manifest Loading")
    print("-"*60)

    manifest = load_manifest(data_dir='./data', verbose=False)

    # Check trajectory types
    types_found = list(manifest.by_trajectory_type.keys())
    print(f"  Trajectory types: {types_found}")

    has_circle = 'circle' in types_found
    has_lemniscate = 'lemniscate' in types_found

    assert has_circle, "No circle datasets found"
    assert has_lemniscate, "No lemniscate datasets found"

    # Check all have references
    n_with_ref = len(manifest.datasets_with_ref)
    n_no_ref = len(manifest.datasets_no_ref)

    print(f"  Datasets with reference: {n_with_ref}")
    print(f"  Datasets without reference: {n_no_ref}")

    # Check circle datasets specifically
    circle_datasets = manifest.by_trajectory_type.get('circle', [])
    circle_with_ref = [ds for ds in circle_datasets if ds.has_reference]

    print(f"  Circle datasets: {len(circle_datasets)}")
    print(f"  Circle with reference: {len(circle_with_ref)}")

    assert len(circle_with_ref) > 0, "No circle datasets with reference"

    # Check lemniscate datasets
    lem_datasets = manifest.by_trajectory_type.get('lemniscate', [])
    lem_with_ref = [ds for ds in lem_datasets if ds.has_reference]

    print(f"  Lemniscate datasets: {len(lem_datasets)}")
    print(f"  Lemniscate with reference: {len(lem_with_ref)}")

    assert len(lem_with_ref) > 0, "No lemniscate datasets with reference"

    print("  ✓ Both trajectory types present with references")
    print()

    return manifest


def test_mixed_training_data(manifest):
    """Test 2: Can create mixed training dataset."""
    print("Test 2: Mixed Training Data")
    print("-"*60)

    # Sample from both types
    circle_datasets = [ds for ds in manifest.datasets_with_ref
                       if 'circle' in ds.filename.lower()]
    lem_datasets = [ds for ds in manifest.datasets_with_ref
                    if 'lem' in ds.filename.lower()]

    if not circle_datasets or not lem_datasets:
        print("  Warning: Missing one type, using available datasets")
        all_datasets = manifest.datasets_with_ref
    else:
        # Mix: take 1 circle + 1 lemniscate
        all_datasets = [circle_datasets[0], lem_datasets[0]]

    print(f"  Selected {len(all_datasets)} datasets:")
    for ds in all_datasets:
        print(f"    - {ds.filename}")

    # Collect samples
    p_tips_all = []
    p_refs_all = []

    for ds in all_datasets:
        valid_mask = ~ds.hold_mask
        p_tip = ds.tip_dyn[valid_mask]
        p_ref = ds.tip_ref[valid_mask]

        p_tips_all.append(p_tip)
        p_refs_all.append(p_ref)

    p_tips = np.vstack(p_tips_all)
    p_refs = np.vstack(p_refs_all)

    print(f"  Total samples: {len(p_tips)}")
    print(f"  p_tip shape: {p_tips.shape}")
    print(f"  p_ref shape: {p_refs.shape}")

    assert p_tips.shape == p_refs.shape
    assert p_tips.shape[1] == 3

    print("  ✓ Mixed dataset created successfully")
    print()

    return p_tips, p_refs


def test_single_policy_multi_task():
    """Test 3: Single policy handles both trajectory types."""
    print("Test 3: Single Policy Multi-task")
    print("-"*60)

    # Create policy
    policy = GRUPolicy(input_dim=6, hidden_dim=32, output_dim=3, num_layers=1)

    print(f"  Policy: {type(policy).__name__}")
    print(f"  Parameters: {sum(p.numel() for p in policy.parameters())}")

    # Test with circle-like input
    p_tip_circle = np.array([10.0, 20.0, 85.0])  # Typical circle coords
    p_ref_circle = np.array([12.0, 22.0, 85.0])

    hidden = policy.init_hidden(batch_size=1)
    u_circle, hidden = policy.predict(p_tip_circle, p_ref_circle, hidden)

    print(f"  Circle input → u={u_circle}")

    # Test with lemniscate-like input
    p_tip_lem = np.array([5.0, 40.0, 50.0])  # Typical lemniscate coords
    p_ref_lem = np.array([7.0, 42.0, 50.0])

    u_lem, hidden = policy.predict(p_tip_lem, p_ref_lem, hidden)

    print(f"  Lemniscate input → u={u_lem}")

    # Check outputs are valid
    assert u_circle.shape == (3,)
    assert u_lem.shape == (3,)
    assert np.all(np.isfinite(u_circle))
    assert np.all(np.isfinite(u_lem))

    print("  ✓ Single policy handles both trajectory types")
    print()


def main():
    print("="*80)
    print("CP4.6: Multi-task DAgger Smoke Test")
    print("="*80)
    print()

    try:
        # Test 1: Manifest
        manifest = test_manifest_has_both_types()

        # Test 2: Mixed data
        p_tips, p_refs = test_mixed_training_data(manifest)

        # Test 3: Single policy
        test_single_policy_multi_task()

        print("="*80)
        print("✓ All Tests PASSED")
        print("="*80)
        return 0

    except Exception as e:
        print("="*80)
        print(f"✗ Test FAILED: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
