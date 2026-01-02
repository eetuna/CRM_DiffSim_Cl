#!/usr/bin/env python3
"""
CP4.0: NPZ Dataset Metadata Audit

This test verifies that all NPZ files in data/ contain the required metadata
for learning and replay:
- currents (control inputs)
- tip trajectories (tip_fk and/or tip_dyn)
- dt (timestep)
- L_inserted (insertion length)
- integration_step_size (optional for old files)

Prints a one-line PARAMS summary for each file.
"""

import sys
import os
import glob
import numpy as np

# Add python directory to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from crm_config import print_param_summary


def audit_npz_file(npz_path):
    """
    Audit a single NPZ file for required metadata.

    Returns:
        dict: Audit results with keys:
            - passed: bool
            - errors: list of error strings
            - warnings: list of warning strings
            - metadata: dict of extracted metadata
    """
    result = {
        'passed': True,
        'errors': [],
        'warnings': [],
        'metadata': {}
    }

    try:
        data = np.load(npz_path)
    except Exception as e:
        result['passed'] = False
        result['errors'].append(f"Failed to load NPZ: {e}")
        return result

    # Check for required fields
    required_fields = ['currents']
    for field in required_fields:
        if field not in data:
            result['passed'] = False
            result['errors'].append(f"Missing required field: '{field}'")

    # Check for at least one tip trajectory
    tip_fields = ['tip_fk', 'tip_dyn', 'tip_desired', 'tip_projected']
    has_tip = any(field in data for field in tip_fields)
    if not has_tip:
        result['passed'] = False
        result['errors'].append("No tip trajectory found (need tip_fk, tip_dyn, tip_desired, or tip_projected)")

    # Check for required parameters
    if 'dt' not in data:
        result['passed'] = False
        result['errors'].append("Missing required parameter: 'dt'")
    else:
        result['metadata']['dt'] = float(data['dt'])

    # Check for L_inserted (may be stored as 'insertion_length')
    if 'insertion_length' in data:
        result['metadata']['L_inserted'] = float(data['insertion_length'])
    elif 'L_inserted' in data:
        result['metadata']['L_inserted'] = float(data['L_inserted'])
    else:
        result['passed'] = False
        result['errors'].append("Missing required parameter: 'insertion_length' or 'L_inserted'")

    # Check for integration_step_size (optional for old files)
    if 'integration_step_size' in data:
        result['metadata']['integration_step_size'] = float(data['integration_step_size'])
    else:
        result['warnings'].append("Missing 'integration_step_size' (assumed 0.5 for old files)")
        result['metadata']['integration_step_size'] = 0.5  # Default

    # Extract array metadata
    if 'currents' in data:
        currents = data['currents']
        result['metadata']['n_steps'] = len(currents)
        result['metadata']['n_controls'] = currents.shape[1] if currents.ndim > 1 else 1

        # Compute duration if dt is available
        if 'dt' in result['metadata']:
            duration = (len(currents) - 1) * result['metadata']['dt']
            result['metadata']['duration'] = duration

    # List all available fields
    result['metadata']['available_fields'] = list(data.keys())

    return result


def main():
    print("=" * 80)
    print("CP4.0: NPZ Dataset Metadata Audit")
    print("=" * 80)
    print()
    print("This test verifies that all NPZ files contain required metadata:")
    print("  - currents (control inputs)")
    print("  - tip trajectories (tip_fk, tip_dyn, tip_desired, or tip_projected)")
    print("  - dt (timestep)")
    print("  - L_inserted (insertion length)")
    print("  - integration_step_size (optional)")
    print()

    # Find all NPZ files in data/
    data_dir = './data'
    npz_files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))

    if not npz_files:
        print(f"✗ No NPZ files found in {data_dir}/")
        return 1

    print(f"Found {len(npz_files)} NPZ file(s) in {data_dir}/\n")

    # Audit each file
    results = []
    all_passed = True

    for npz_path in npz_files:
        filename = os.path.basename(npz_path)
        print(f"{'=' * 80}")
        print(f"Auditing: {filename}")
        print(f"{'=' * 80}")

        result = audit_npz_file(npz_path)
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

        # Print metadata
        if result['metadata']:
            print("\nMetadata:")
            meta = result['metadata']

            # Print parameter summary
            if 'dt' in meta and 'L_inserted' in meta and 'integration_step_size' in meta:
                summary = print_param_summary(
                    meta['dt'],
                    meta['L_inserted'],
                    meta['integration_step_size']
                )
                print(f"  {summary}")

            # Print array info
            if 'n_steps' in meta:
                print(f"  N_steps: {meta['n_steps']}")
            if 'duration' in meta:
                print(f"  Duration: {meta['duration']:.3f} s")

            # Print available fields
            if 'available_fields' in meta:
                fields_str = ', '.join(meta['available_fields'])
                print(f"  Fields: {fields_str}")

        # Print status
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"\n{status}: {filename}\n")

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed_count = sum(1 for _, r in results if r['passed'])
    failed_count = len(results) - passed_count

    for filename, result in results:
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"  {status}: {filename}")

    print("=" * 80)
    print(f"Total: {len(results)} files")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print("=" * 80)

    if all_passed:
        print("✓ CP4.0 NPZ METADATA AUDIT PASS: All files contain required metadata")
        return 0
    else:
        print("✗ CP4.0 NPZ METADATA AUDIT FAIL: Some files missing required metadata")
        return 1


if __name__ == "__main__":
    sys.exit(main())
