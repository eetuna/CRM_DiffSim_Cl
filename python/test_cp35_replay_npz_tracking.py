"""
CP3.5: Dataset-Driven NPZ Trajectory Visualization (FIXED)

This test loads NPZ trajectory data and visualizes the recorded trajectories.
It shows the intrinsic FK vs Dyn mismatch present in the dataset.

Dataset regime: L_inserted = 94.3 mm

Note: This test does NOT recompute rollouts. It visualizes the NPZ data as-is.
For replay verification, see test_cp35_npz_replay_contract.py
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def load_npz_data(npz_path):
    """Load all relevant data from NPZ."""
    data = np.load(npz_path)

    result = {
        't': data['t'],
        'currents': data['currents'],
        'tip_fk': data['tip_fk'],
        'tip_dyn': data['tip_dyn'],
        'dt': float(data['dt']),
        'L_inserted': float(data['insertion_length']),
    }

    # Optional fields
    if 'tip_desired' in data:
        result['tip_desired'] = data['tip_desired']
    if 'tip_projected' in data:
        result['tip_projected'] = data['tip_projected']
    if 'integration_step_size' in data:
        result['integration_step_size'] = float(data['integration_step_size'])
    else:
        result['integration_step_size'] = 0.5  # Default

    return result


def plot_npz_visualization(data, traj_name, output_path):
    """
    Generate comprehensive visualization of NPZ trajectory data.

    Shows:
    - 3D + XY/XZ/YZ projections
    - FK vs Dyn comparison from NPZ
    - Control inputs
    - FK-Dyn error over time
    """
    t = data['t']
    currents = data['currents']
    tip_fk = data['tip_fk']
    tip_dyn = data['tip_dyn']

    # Check for desired/projected
    has_desired = 'tip_desired' in data
    has_projected = 'tip_projected' in data

    # Compute FK-Dyn error
    fk_dyn_err = np.linalg.norm(tip_fk - tip_dyn, axis=1)

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Plot 1: 3D trajectory
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')

    if has_projected:
        p_ref = data['tip_projected']
        ax1.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2],
                 'g--', linewidth=2, label='Projected', alpha=0.7)
    elif has_desired:
        p_ref = data['tip_desired']
        ax1.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2],
                 'g--', linewidth=2, label='Desired', alpha=0.7)

    ax1.plot(tip_fk[:, 0], tip_fk[:, 1], tip_fk[:, 2],
             'b-', linewidth=1.5, label='FK (NPZ)', alpha=0.8)
    ax1.plot(tip_dyn[:, 0], tip_dyn[:, 1], tip_dyn[:, 2],
             'r-', linewidth=1.5, label='Dyn (NPZ)', alpha=0.8)
    ax1.scatter(tip_fk[0, 0], tip_fk[0, 1], tip_fk[0, 2],
                c='k', s=100, marker='o', label='Start')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: XY projection
    ax2 = fig.add_subplot(gs[0, 1])
    if has_projected or has_desired:
        ax2.plot(p_ref[:, 0], p_ref[:, 1], 'g--', linewidth=2,
                 label='Projected' if has_projected else 'Desired', alpha=0.7)
    ax2.plot(tip_fk[:, 0], tip_fk[:, 1], 'b-', linewidth=1.5, label='FK', alpha=0.8)
    ax2.plot(tip_dyn[:, 0], tip_dyn[:, 1], 'r-', linewidth=1.5, label='Dyn', alpha=0.8)
    ax2.scatter(tip_fk[0, 0], tip_fk[0, 1], c='k', s=100, marker='o')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('XY Projection')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    # Plot 3: XZ projection
    ax3 = fig.add_subplot(gs[0, 2])
    if has_projected or has_desired:
        ax3.plot(p_ref[:, 0], p_ref[:, 2], 'g--', linewidth=2, alpha=0.7)
    ax3.plot(tip_fk[:, 0], tip_fk[:, 2], 'b-', linewidth=1.5, label='FK', alpha=0.8)
    ax3.plot(tip_dyn[:, 0], tip_dyn[:, 2], 'r-', linewidth=1.5, label='Dyn', alpha=0.8)
    ax3.scatter(tip_fk[0, 0], tip_fk[0, 2], c='k', s=100, marker='o')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('XZ Projection')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')

    # Plot 4: YZ projection
    ax4 = fig.add_subplot(gs[0, 3])
    if has_projected or has_desired:
        ax4.plot(p_ref[:, 1], p_ref[:, 2], 'g--', linewidth=2, alpha=0.7)
    ax4.plot(tip_fk[:, 1], tip_fk[:, 2], 'b-', linewidth=1.5, label='FK', alpha=0.8)
    ax4.plot(tip_dyn[:, 1], tip_dyn[:, 2], 'r-', linewidth=1.5, label='Dyn', alpha=0.8)
    ax4.scatter(tip_fk[0, 1], tip_fk[0, 2], c='k', s=100, marker='o')
    ax4.set_xlabel('Y (mm)')
    ax4.set_ylabel('Z (mm)')
    ax4.set_title('YZ Projection')
    ax4.legend()
    ax4.grid(True)
    ax4.axis('equal')

    # Plot 5: Control inputs
    ax5 = fig.add_subplot(gs[1, 0:2])
    for i in range(3):
        ax5.plot(t, currents[:, i], linewidth=2, label=f'i_{i}')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Current (A)')
    ax5.set_title('Control Inputs (Currents)')
    ax5.legend()
    ax5.grid(True)

    # Plot 6: FK-Dyn error over time
    ax6 = fig.add_subplot(gs[1, 2:4])
    ax6.plot(t, fk_dyn_err, 'b-', linewidth=2, label='||FK - Dyn||')
    rms_err = np.sqrt(np.mean(fk_dyn_err**2))
    ax6.axhline(y=rms_err, color='b', linestyle=':', linewidth=2,
                label=f'RMS = {rms_err:.3f} mm')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Error (mm)')
    ax6.set_title('FK vs Dyn Mismatch (Intrinsic to NPZ)')
    ax6.legend()
    ax6.grid(True)

    # Plot 7: Dyn vs Projected/Desired error (if available)
    ax7 = fig.add_subplot(gs[2, 0:2])
    if has_projected:
        err = np.linalg.norm(tip_dyn - data['tip_projected'], axis=1)
        ax7.plot(t, err, 'r-', linewidth=2, label='||Dyn - Projected||')
        rms = np.sqrt(np.mean(err**2))
        ax7.axhline(y=rms, color='r', linestyle=':', linewidth=2,
                    label=f'RMS = {rms:.3f} mm')
        ax7.set_title('Dyn vs Projected Tracking Error')
    elif has_desired:
        err = np.linalg.norm(tip_dyn - data['tip_desired'], axis=1)
        ax7.plot(t, err, 'r-', linewidth=2, label='||Dyn - Desired||')
        rms = np.sqrt(np.mean(err**2))
        ax7.axhline(y=rms, color='r', linestyle=':', linewidth=2,
                    label=f'RMS = {rms:.3f} mm')
        ax7.set_title('Dyn vs Desired Tracking Error')
    else:
        ax7.text(0.5, 0.5, 'No reference trajectory available',
                 ha='center', va='center', transform=ax7.transAxes)
        ax7.set_title('Tracking Error (N/A)')

    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Error (mm)')
    ax7.legend()
    ax7.grid(True)

    # Plot 8: Summary metrics
    ax8 = fig.add_subplot(gs[2, 2:4])
    ax8.axis('off')

    # Compute metrics
    fk_dyn_rms = np.sqrt(np.mean(fk_dyn_err**2))
    fk_dyn_max = np.max(fk_dyn_err)
    fk_dyn_mean = np.mean(fk_dyn_err)

    summary_text = "NPZ Trajectory Metrics\n" + "="*40 + "\n\n"
    summary_text += f"FK vs Dyn (intrinsic to NPZ):\n"
    summary_text += f"  RMS:  {fk_dyn_rms:.4f} mm\n"
    summary_text += f"  Max:  {fk_dyn_max:.4f} mm\n"
    summary_text += f"  Mean: {fk_dyn_mean:.4f} mm\n\n"

    if has_projected:
        p_err = np.linalg.norm(tip_dyn - data['tip_projected'], axis=1)
        summary_text += f"Dyn vs Projected:\n"
        summary_text += f"  RMS:  {np.sqrt(np.mean(p_err**2)):.4f} mm\n"
        summary_text += f"  Max:  {np.max(p_err):.4f} mm\n"
        summary_text += f"  Mean: {np.mean(p_err):.4f} mm\n\n"
    elif has_desired:
        d_err = np.linalg.norm(tip_dyn - data['tip_desired'], axis=1)
        summary_text += f"Dyn vs Desired:\n"
        summary_text += f"  RMS:  {np.sqrt(np.mean(d_err**2)):.4f} mm\n"
        summary_text += f"  Max:  {np.max(d_err):.4f} mm\n"
        summary_text += f"  Mean: {np.mean(d_err):.4f} mm\n\n"

    summary_text += f"Dataset Parameters:\n"
    summary_text += f"  L_inserted = {data['L_inserted']:.1f} mm\n"
    summary_text += f"  dt = {data['dt']:.4f} s\n"
    summary_text += f"  IntegrationStepSize = {data['integration_step_size']:.1f}\n"
    summary_text += f"  N_steps = {len(t)}\n"
    summary_text += f"  Duration = {t[-1]:.2f} s\n"

    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'CP3.5: NPZ Trajectory Visualization - {traj_name}',
                 fontsize=14, fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to {output_path}")


def test_npz_visualization(npz_path, traj_name, output_dir):
    """Load NPZ and generate visualization."""
    print(f"\n{'='*70}")
    print(f"NPZ Visualization: {traj_name}")
    print(f"NPZ file: {npz_path}")
    print(f"{'='*70}")

    # Load NPZ data
    data = load_npz_data(npz_path)

    t = data['t']
    tip_fk = data['tip_fk']
    tip_dyn = data['tip_dyn']

    print(f"\nNPZ Info:")
    print(f"  Duration: {t[-1]:.3f} s")
    print(f"  Steps: {len(t)}")
    print(f"  dt: {data['dt']:.4f} s")
    print(f"  L_inserted: {data['L_inserted']:.1f} mm")
    print(f"  IntegrationStepSize: {data['integration_step_size']:.1f}")

    # Compute intrinsic FK-Dyn error
    fk_dyn_err = np.linalg.norm(tip_fk - tip_dyn, axis=1)
    print(f"\nIntrinsic FK-Dyn Error (from NPZ):")
    print(f"  RMS:  {np.sqrt(np.mean(fk_dyn_err**2)):.6f} mm")
    print(f"  Max:  {np.max(fk_dyn_err):.6f} mm")
    print(f"  Mean: {np.mean(fk_dyn_err):.6f} mm")

    # If projected/desired available, compute tracking error
    if 'tip_projected' in data:
        track_err = np.linalg.norm(tip_dyn - data['tip_projected'], axis=1)
        print(f"\nDyn vs Projected Error (from NPZ):")
        print(f"  RMS:  {np.sqrt(np.mean(track_err**2)):.6f} mm")
        print(f"  Max:  {np.max(track_err):.6f} mm")
        print(f"  Mean: {np.mean(track_err):.6f} mm")
    elif 'tip_desired' in data:
        track_err = np.linalg.norm(tip_dyn - data['tip_desired'], axis=1)
        print(f"\nDyn vs Desired Error (from NPZ):")
        print(f"  RMS:  {np.sqrt(np.mean(track_err**2)):.6f} mm")
        print(f"  Max:  {np.max(track_err):.6f} mm")
        print(f"  Mean: {np.mean(track_err):.6f} mm")

    # Generate plot
    plot_filename = os.path.join(output_dir, f'cp35_{traj_name}.png')
    plot_npz_visualization(data, traj_name, plot_filename)

    print(f"\n✓ Visualization complete: {traj_name}")
    return True


def main():
    print("="*70)
    print("CP3.5: NPZ Trajectory Visualization")
    print("="*70)
    print()
    print("This test visualizes NPZ-recorded trajectories and shows")
    print("the intrinsic FK vs Dyn mismatch present in the dataset.")
    print()

    # Create output directory
    output_dir = './build/artifacts/cp35'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Test trajectories
    test_cases = [
        ('data/dyn_fk_ramp_circle1_hold1.npz', 'circle'),
        ('data/dyn_fk_lem1_y40_a10_hold1.npz', 'lemniscate'),
    ]

    results = []
    for npz_path, traj_name in test_cases:
        if not os.path.exists(npz_path):
            print(f"\n✗ SKIP: {traj_name} - File not found: {npz_path}")
            results.append((traj_name, False))
            continue

        passed = test_npz_visualization(npz_path, traj_name, output_dir)
        results.append((traj_name, passed))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    all_passed = True
    for traj_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {traj_name}")
        if not passed:
            all_passed = False

    print(f"{'='*70}")
    if all_passed:
        print("✓ CP3.5 PASS: All NPZ visualizations generated")
        print(f"  Plots saved to {output_dir}/")
        print(f"{'='*70}")
        return 0
    else:
        print("✗ CP3.5 FAIL: Some visualizations failed")
        print(f"{'='*70}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
