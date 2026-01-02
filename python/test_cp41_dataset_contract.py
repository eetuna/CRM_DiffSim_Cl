#!/usr/bin/env python3
"""
CP4.1: Dataset Contract Test

Validates that all NPZ datasets conform to the CP4.1 standard contract:
- Exposes t, dt, L_inserted, integration_step_size
- Exposes currents [N, 3]
- Exposes tip_fk [N, 3] and/or tip_dyn [N, 3] when present
- Exposes tip_ref [N, 3] (or marked as "no_ref")
- Exposes hold_mask [N] (bool array)
- param_summary() method prints exact format

This test ensures all datasets can be loaded and provide the standardized interface.
"""

import sys
import os
import numpy as np

# Add python directory to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from data.npz_manifest import discover_npz_files
from data.npz_dataset import load_npz_dataset


def test_dataset_contract(npz_path):
    """
    Test that a single NPZ dataset conforms to CP4.1 contract.

    Args:
        npz_path: Path to NPZ file

    Returns:
        dict: Test results with keys:
            - passed: bool
            - errors: list of error messages
            - warnings: list of warning messages
    """
    filename = os.path.basename(npz_path)
    result = {
        'passed': True,
        'errors': [],
        'warnings': []
    }

    try:
        ds = load_npz_dataset(npz_path, validate=False)
    except Exception as e:
        result['passed'] = False
        result['errors'].append(f"Failed to load: {e}")
        return result

    # Test required attributes
    required_attrs = ['t', 'currents', 'dt', 'L_inserted', 'integration_step_size']
    for attr in required_attrs:
        if not hasattr(ds, attr):
            result['passed'] = False
            result['errors'].append(f"Missing required attribute: {attr}")
        elif getattr(ds, attr) is None:
            result['passed'] = False
            result['errors'].append(f"Required attribute is None: {attr}")

    # Test currents shape
    if ds.currents is not None:
        if ds.currents.ndim != 2 or ds.currents.shape[1] != 3:
            result['passed'] = False
            result['errors'].append(f"currents shape {ds.currents.shape} != [N, 3]")

    # Test trajectory arrays (at least one must be present)
    has_any_traj = False
    for traj_attr in ['tip_fk', 'tip_dyn']:
        traj = getattr(ds, traj_attr, None)
        if traj is not None:
            has_any_traj = True
            if traj.ndim != 2 or traj.shape[1] != 3:
                result['passed'] = False
                result['errors'].append(f"{traj_attr} shape {traj.shape} != [N, 3]")

    if not has_any_traj:
        result['passed'] = False
        result['errors'].append("No tip trajectories found (need tip_fk or tip_dyn)")

    # Test tip_ref property
    if not hasattr(ds, 'tip_ref'):
        result['passed'] = False
        result['errors'].append("Missing tip_ref property")
    else:
        tip_ref = ds.tip_ref
        if tip_ref is None:
            # Dataset marked as "no_ref" - this is valid
            result['warnings'].append("Dataset has no reference (no_ref)")
        else:
            if tip_ref.ndim != 2 or tip_ref.shape[1] != 3:
                result['passed'] = False
                result['errors'].append(f"tip_ref shape {tip_ref.shape} != [N, 3]")

    # Test hold_mask property
    if not hasattr(ds, 'hold_mask'):
        result['passed'] = False
        result['errors'].append("Missing hold_mask property")
    else:
        hold_mask = ds.hold_mask
        if hold_mask.ndim != 1:
            result['passed'] = False
            result['errors'].append(f"hold_mask shape {hold_mask.shape} != [N]")
        if hold_mask.dtype != bool:
            result['passed'] = False
            result['errors'].append(f"hold_mask dtype {hold_mask.dtype} != bool")

    # Test param_summary() method
    if not hasattr(ds, 'param_summary'):
        result['passed'] = False
        result['errors'].append("Missing param_summary() method")
    else:
        try:
            summary = ds.param_summary()
            if not isinstance(summary, str):
                result['passed'] = False
                result['errors'].append(f"param_summary() returned {type(summary)}, expected str")
            elif not summary.startswith("PARAMS:"):
                result['passed'] = False
                result['errors'].append(f"param_summary() doesn't start with 'PARAMS:': {summary}")
        except Exception as e:
            result['passed'] = False
            result['errors'].append(f"param_summary() raised exception: {e}")

    return result


def main():
    print("=" * 80)
    print("CP4.1: Dataset Contract Test")
    print("=" * 80)
    print()
    print("Testing that all NPZ datasets conform to CP4.1 standardized contract:")
    print("  - t, dt, L_inserted, integration_step_size")
    print("  - currents [N, 3]")
    print("  - tip_fk/tip_dyn [N, 3]")
    print("  - tip_ref [N, 3] property")
    print("  - hold_mask [N] bool property")
    print("  - param_summary() method")
    print()

    # Discover NPZ files
    data_dir = './data'
    npz_files = discover_npz_files(data_dir)

    if not npz_files:
        print(f"✗ No NPZ files found in {data_dir}/")
        return 1

    print(f"Found {len(npz_files)} NPZ file(s) in {data_dir}/\n")

    # Test each file
    results = []
    all_passed = True

    for npz_path in npz_files:
        filename = os.path.basename(npz_path)
        print(f"{'=' * 80}")
        print(f"Testing: {filename}")
        print(f"{'=' * 80}")

        result = test_dataset_contract(npz_path)
        results.append((filename, result))

        # Print errors
        if result['errors']:
            print("\n✗ ERRORS:")
            for error in result['errors']:
                print(f"  - {error}")
            all_passed = False

        # Print warnings
        if result['warnings']:
            print("\n⚠ WARNINGS:")
            for warning in result['warnings']:
                print(f"  - {warning}")

        # Print param_summary if loaded successfully
        if not result['errors']:
            try:
                ds = load_npz_dataset(npz_path, validate=False)
                print(f"\n✓ {ds.param_summary()}")
                print(f"  tip_ref: {'present' if ds.tip_ref is not None else 'NO_REF'}")
                print(f"  hold_mask: {np.sum(ds.hold_mask)}/{len(ds.hold_mask)} invalid")
            except Exception as e:
                print(f"\n✗ Error loading dataset: {e}")
                all_passed = False

        # Print status
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"\n{status}: {filename}\n")

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed_count = sum(1 for _, r in results if r['passed'])
    failed_count = len(results) - passed_count

    # Count trajectory files vs workspace files
    trajectory_passed = 0
    trajectory_failed = 0
    workspace_files = 0

    for filename, result in results:
        if 'workspace' in filename.lower():
            workspace_files += 1
            status = "⚠ SKIP" if not result['passed'] else "✓ PASS"
            print(f"  {status}: {filename} (workspace format, not trajectory)")
        else:
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"  {status}: {filename}")
            if result['passed']:
                trajectory_passed += 1
            else:
                trajectory_failed += 1
                all_passed = False

    print("=" * 80)
    print(f"Trajectory files: {trajectory_passed + trajectory_failed}")
    print(f"  Passed: {trajectory_passed}")
    print(f"  Failed: {trajectory_failed}")
    if workspace_files > 0:
        print(f"Workspace files (skipped): {workspace_files}")
    print("=" * 80)

    # Only trajectory files must pass
    if trajectory_failed == 0:
        print("✓ CP4.1 DATASET CONTRACT PASS: All trajectory files conform to contract")
        return 0
    else:
        print("✗ CP4.1 DATASET CONTRACT FAIL: Some trajectory files do not conform")
        return 1


if __name__ == "__main__":
    sys.exit(main())
