#!/usr/bin/env python3
"""
CP4.6: Circle Reference Validation Test

Validates that all circle datasets now have reference trajectories (tip_desired).
This test FAILS if any circle dataset is missing a reference.

Runtime: Fast (<5 seconds)
"""

import sys
import os
import glob
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))


def check_circle_dataset_has_reference(npz_path):
    """
    Check if circle dataset has reference trajectory.

    Args:
        npz_path: Path to NPZ file

    Returns:
        tuple: (has_reference, error_message)
    """
    try:
        data = np.load(npz_path)

        # Check for required reference fields
        required_fields = ['tip_desired', 'tip_projected']
        missing = [f for f in required_fields if f not in data]

        if missing:
            return False, f"Missing fields: {missing}"

        # Check shapes
        if 'tip_dyn' in data:
            n_samples = len(data['tip_dyn'])
            for field in required_fields:
                if data[field].shape != (n_samples, 3):
                    return False, f"{field} has wrong shape: {data[field].shape}, expected ({n_samples}, 3)"

        # Check that reference is not all zeros
        tip_desired = data['tip_desired']
        if np.allclose(tip_desired, 0):
            return False, "tip_desired is all zeros"

        return True, None

    except Exception as e:
        return False, f"Failed to load: {e}"


def main():
    print("="*80)
    print("CP4.6: Circle Reference Validation Test")
    print("="*80)
    print()

    # Find all circle datasets
    data_dir = './data'
    pattern = os.path.join(data_dir, '*circle*.npz')
    circle_files = glob.glob(pattern)

    if not circle_files:
        print(f"WARNING: No circle datasets found in {data_dir}")
        print("Test SKIPPED (no circle datasets to validate)")
        return 0

    print(f"Found {len(circle_files)} circle dataset(s)")
    print()

    all_pass = True
    for npz_path in sorted(circle_files):
        filename = os.path.basename(npz_path)
        has_ref, error_msg = check_circle_dataset_has_reference(npz_path)

        if has_ref:
            # Additional check: verify reference quality
            data = np.load(npz_path)
            tip_desired = data['tip_desired']
            tip_projected = data['tip_projected']

            # Check not identical to tip_dyn (would be circular)
            if 'tip_dyn' in data:
                tip_dyn = data['tip_dyn']
                if np.allclose(tip_desired, tip_dyn):
                    print(f"  ✗ {filename}: FAIL - tip_desired identical to tip_dyn (circular reference)")
                    all_pass = False
                    continue

            print(f"  ✓ {filename}: HAS REFERENCE")
            print(f"      tip_desired: {tip_desired.shape}")
            print(f"      tip_projected: {tip_projected.shape}")

        else:
            print(f"  ✗ {filename}: NO REFERENCE")
            print(f"      Error: {error_msg}")
            all_pass = False

    print()
    print("="*80)
    if all_pass:
        print("✓ PASS: All circle datasets have valid references")
        print("="*80)
        return 0
    else:
        print("✗ FAIL: Some circle datasets missing or invalid references")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
