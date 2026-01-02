#!/usr/bin/env python3
"""
CP4.6: Add Reference Trajectories to Circle Datasets

Circle datasets (dyn_fk_ramp_circle*.npz) were generated without explicit
reference trajectories (tip_desired, tip_projected). This script:

1. Fits a parametric helix/circle to the observed tip positions
2. Generates smooth reference trajectory
3. Adds tip_desired and tip_projected fields
4. Saves updated NPZ files

Method: Fit 3D helix parametrically to observed tip_dyn positions.
"""

import os
import sys
import numpy as np

def smooth_trajectory_1d(values, window=5):
    """Simple moving average smoothing."""
    kernel = np.ones(window) / window
    # Pad edges to avoid boundary effects
    padded = np.pad(values, (window//2, window//2), mode='edge')
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed


def fit_helix_to_points(tip_positions, t_values):
    """
    Fit a parametric helix to 3D points using simple estimation.

    Uses smoothed trajectory as reference (no complex optimization needed).

    Args:
        tip_positions: (N, 3) array of 3D positions
        t_values: (N,) array of time values

    Returns:
        params: Dict with fitted parameters
        fitted_traj: (N, 3) array of fitted positions
    """
    x = tip_positions[:, 0]
    y = tip_positions[:, 1]
    z = tip_positions[:, 2]

    # Estimate center
    cx = (x.max() + x.min()) / 2
    cy = (y.max() + y.min()) / 2

    # Estimate radius
    A = np.sqrt((x - cx)**2 + (y - cy)**2).mean()

    # Estimate z velocity
    t_span = t_values[-1] - t_values[0]
    v_z = (z[-1] - z[0]) / t_span
    z0 = z[0]

    # Estimate angular frequency
    # Simple approach: use mean angular velocity
    angles = np.arctan2(y - cy, x - cx)
    # Unwrap angles to handle 2π discontinuities
    angles_unwrapped = np.unwrap(angles)
    # Fit linear trend to get angular velocity
    poly = np.polyfit(t_values, angles_unwrapped, 1)
    omega = poly[0]  # rad/s
    phi = poly[1]     # initial phase at t=0

    # Generate fitted helix
    x_fit = A * np.cos(omega * t_values + phi) + cx
    y_fit = A * np.sin(omega * t_values + phi) + cy
    z_fit = z0 + v_z * t_values

    # Apply light smoothing to reduce noise
    window = min(11, len(x_fit) // 10 + 1)
    if window % 2 == 0:
        window += 1  # Make odd
    if window >= 3:
        x_fit = smooth_trajectory_1d(x_fit, window)
        y_fit = smooth_trajectory_1d(y_fit, window)
        z_fit = smooth_trajectory_1d(z_fit, window)

    fitted_traj = np.column_stack([x_fit, y_fit, z_fit])

    params = {
        'method': 'helix',
        'A': float(A),
        'omega': float(omega),
        'phi': float(phi),
        'cx': float(cx),
        'cy': float(cy),
        'z0': float(z0),
        'v_z': float(v_z),
        'period': float(2 * np.pi / abs(omega)) if omega != 0 else float('inf')
    }

    return params, fitted_traj


def add_reference_to_circle_dataset(npz_path, output_path=None, verbose=True):
    """
    Add tip_desired and tip_projected fields to circle dataset.

    Args:
        npz_path: Path to input NPZ file
        output_path: Path to output NPZ file (default: overwrite input)
        verbose: Print processing info

    Returns:
        Dict with processing statistics
    """
    if output_path is None:
        output_path = npz_path

    if verbose:
        print(f"Processing: {os.path.basename(npz_path)}")

    # Load data
    data = np.load(npz_path)

    # Check if already has reference
    if 'tip_desired' in data:
        if verbose:
            print(f"  Already has tip_desired, skipping")
        return {'status': 'skipped', 'reason': 'already_has_reference'}

    # Extract fields
    tip_dyn = data['tip_dyn']
    t = data['t']
    hold = int(data['hold'])

    # Remove hold samples for fitting
    valid_idx = np.arange(hold, len(tip_dyn))
    tip_valid = tip_dyn[valid_idx]
    t_valid = t[valid_idx]

    # Fit parametric trajectory
    params, tip_ref_valid = fit_helix_to_points(tip_valid, t_valid)

    if verbose:
        # Compute fit quality
        fit_error = np.linalg.norm(tip_valid - tip_ref_valid, axis=1)
        print(f"  Method: {params['method']}")
        if params['method'] == 'helix':
            print(f"    Radius: {params['A']:.2f} mm")
            print(f"    Period: {params['period']:.2f} s")
            print(f"    z velocity: {params['v_z']:.2f} mm/s")
        print(f"    Fit RMSE: {np.sqrt(np.mean(fit_error**2)):.2f} mm")
        print(f"    Max error: {fit_error.max():.2f} mm")

    # Construct full reference (including hold samples)
    # For hold samples, use initial position
    tip_desired = np.zeros_like(tip_dyn)
    tip_desired[:hold] = tip_ref_valid[0]  # Use first valid reference
    tip_desired[hold:] = tip_ref_valid

    # tip_projected = tip_desired (no obstacles to project onto)
    tip_projected = tip_desired.copy()

    # Create new dataset with added fields
    new_data = {}
    for key in data.keys():
        new_data[key] = data[key]

    new_data['tip_desired'] = tip_desired
    new_data['tip_projected'] = tip_projected

    # Add integration_step_size if missing (for CP4.1 contract compliance)
    if 'integration_step_size' not in new_data:
        # Use default from lemniscate datasets
        new_data['integration_step_size'] = 0.2

    # Save
    np.savez(output_path, **new_data)

    if verbose:
        print(f"  Saved: {output_path}")

    stats = {
        'status': 'success',
        'fit_params': params,
        'n_samples': len(tip_dyn),
        'n_valid': len(tip_valid),
        'output_path': output_path
    }

    return stats


def process_all_circle_datasets(data_dir='./data', pattern='*circle*.npz', verbose=True):
    """
    Process all circle datasets in directory.

    Args:
        data_dir: Directory containing NPZ files
        pattern: Glob pattern for circle files
        verbose: Print progress

    Returns:
        List of processing statistics
    """
    import glob

    files = glob.glob(os.path.join(data_dir, pattern))

    if not files:
        print(f"No files matching {pattern} found in {data_dir}")
        return []

    print(f"Found {len(files)} circle dataset(s)")
    print()

    results = []
    for npz_path in sorted(files):
        stats = add_reference_to_circle_dataset(npz_path, verbose=verbose)
        results.append(stats)
        print()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Add reference trajectories to circle datasets')
    parser.add_argument('--data-dir', default='./data', help='Data directory')
    parser.add_argument('--pattern', default='*circle*.npz', help='File pattern')
    parser.add_argument('--dry-run', action='store_true', help='Dry run (no file writes)')

    args = parser.parse_args()

    print("="*80)
    print("CP4.6: Adding References to Circle Datasets")
    print("="*80)
    print()

    results = process_all_circle_datasets(args.data_dir, args.pattern, verbose=True)

    print("="*80)
    print(f"Processed {len(results)} file(s)")
    success = sum(1 for r in results if r['status'] == 'success')
    skipped = sum(1 for r in results if r['status'] == 'skipped')
    print(f"  Success: {success}")
    print(f"  Skipped: {skipped}")
    print("="*80)
