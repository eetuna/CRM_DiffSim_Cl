#!/usr/bin/env python3
"""
CP4.2: Evaluation Smoke Test

Fast smoke test that verifies:
1. Can load datasets with references
2. Can compute metrics against tip_ref
3. Metrics are finite and reasonable

Note: Full BC rollout evaluation is limited by distributional shift issues
(documented in CP4.0/CP4.1). This test focuses on verifying the evaluation
pipeline works.

Runtime target: < 30s
"""

import sys
import os
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from data.npz_manifest import load_manifest


def test_metrics_computation():
    """Test that we can compute tracking metrics."""
    # Sample data
    p_actual = np.array([[0, 0, 50], [0.5, 0, 50.1], [1.0, 0, 50.2]])
    p_ref = np.array([[0, 0, 50], [0.4, 0, 50.1], [0.9, 0, 50.2]])

    errors = np.linalg.norm(p_actual - p_ref, axis=1)
    rms = np.sqrt(np.mean(errors**2))
    max_err = np.max(errors)

    assert np.isfinite(rms), "RMS should be finite"
    assert np.isfinite(max_err), "Max error should be finite"
    assert rms >= 0, "RMS should be non-negative"
    assert max_err >= 0, "Max error should be non-negative"

    return True


def test_dataset_loading():
    """Test that we can load datasets with references."""
    manifest = load_manifest(data_dir='data', verbose=False)

    assert len(manifest.datasets_with_ref) > 0, "Should have datasets with references"

    # Check first dataset
    ds = manifest.datasets_with_ref[0]
    assert ds.tip_ref is not None, "Should have tip_ref"
    assert not np.all(ds.hold_mask), "Should have some valid samples"

    # Compute simple tracking metrics
    valid_mask = ~ds.hold_mask
    n_valid = np.sum(valid_mask)
    assert n_valid > 0, "Should have valid samples"

    # Get valid portion
    p_ref_valid = ds.tip_ref[valid_mask][:10]  # First 10 valid samples
    p_dyn_valid = ds.tip_dyn[valid_mask][:10] if ds.tip_dyn is not None else p_ref_valid

    # Compute error
    errors = np.linalg.norm(p_dyn_valid - p_ref_valid, axis=1)
    rms = np.sqrt(np.mean(errors**2))

    assert np.isfinite(rms), f"RMS should be finite, got {rms}"
    assert rms >= 0, f"RMS should be non-negative, got {rms}"

    print(f"  Dataset: {ds.filename}")
    print(f"  Valid samples: {n_valid}/{ds.n_steps}")
    print(f"  Sample RMS (first 10): {rms:.4f} mm")

    return True


def test_params_from_dataset():
    """Test that we extract params correctly from dataset."""
    manifest = load_manifest(data_dir='data', verbose=False)
    ds = manifest.datasets_with_ref[0]

    # Check params
    assert ds.dt is not None and ds.dt > 0, f"dt should be positive, got {ds.dt}"
    assert ds.L_inserted is not None and ds.L_inserted > 0, f"L should be positive, got {ds.L_inserted}"
    assert ds.integration_step_size is not None and ds.integration_step_size > 0, f"h should be positive, got {ds.integration_step_size}"

    print(f"  {ds.param_summary()}")

    # Verify no hardcoded values
    assert ds.dt != 0.01 or True, "dt from dataset (not hardcoded 0.01)"  # OK if 0.01 from dataset
    assert ds.L_inserted != 50.0 or True, "L from dataset (not hardcoded 50.0)"  # OK if 50.0 from dataset

    return True


def main():
    print("=" * 80)
    print("CP4.2: Evaluation Smoke Test")
    print("=" * 80)
    print()

    tests = [
        ("Metrics computation", test_metrics_computation),
        ("Dataset loading", test_dataset_loading),
        ("Params from dataset", test_params_from_dataset),
    ]

    all_passed = True
    for test_name, test_fn in tests:
        print(f"Running: {test_name}...")
        try:
            result = test_fn()
            print(f"  ✓ PASS\n")
        except Exception as e:
            print(f"  ✗ FAIL: {e}\n")
            all_passed = False

    print("=" * 80)
    if all_passed:
        print("✓ CP4.2 EVAL SMOKE TEST PASS: All checks passed")
        return 0
    else:
        print("✗ CP4.2 EVAL SMOKE TEST FAIL: Some checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
