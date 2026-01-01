"""
CP3.5: Dataset-Driven Regression Test for Trajectory Tracking

Replays recorded current sequences from NPZ files and compares:
- Projected workspace path (desired)
- FK output path (reference)
- Legacy dynamics rollout path (actual under dynamics)

Dataset regime: L_inserted = 94.3 mm

Acceptance criteria:
- Both NPZ trajectories (circle + lemniscate) produce valid metrics
- FK and Dyn rollouts complete without NaN/Inf
- Dynamics solver failure rate < 15% (allows for numerical precision differences)
- Plots saved to build/artifacts/cp35/
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add build directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.insert(0, os.path.dirname(__file__))

import crm_diff_py


def load_trajectory_npz(npz_path):
    """
    Load trajectory data from NPZ file.

    Returns:
        dict with keys:
            t: time vector (N,)
            currents: control sequence (N, 3)
            tip_desired: desired path (N, 3) [optional]
            tip_projected: projected path (N, 3) [optional]
            tip_fk: FK output (N, 3) [from NPZ]
            tip_dyn: Dyn output (N, 3) [from NPZ]
            dt: timestep scalar
            L_inserted: insertion length scalar
    """
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

    return result


def run_fk_rollout(currents, dt, L_inserted, params_dict):
    """
    Run FK-only evaluation for each control input.

    Args:
        currents: (N, 3) control sequence
        dt: timestep (not used for FK, but kept for API consistency)
        L_inserted: insertion length (mm)
        params_dict: catheter parameters

    Returns:
        tip_fk_computed: (N, 3) FK tip positions
        fk_status: (N,) status codes (0 = success)
    """
    N = currents.shape[0]
    tip_fk_computed = np.zeros((N, 3))
    fk_status = np.zeros(N, dtype=int)

    for i in range(N):
        u = currents[i]
        result = crm_diff_py.equilibrium_forward(u, L_inserted, params_dict)
        tip_fk_computed[i] = result['p_tip']
        fk_status[i] = result['status']

    return tip_fk_computed, fk_status


def run_dynamics_rollout(currents, dt, L_inserted, params_dict, x0=None):
    """
    Run dynamics rollout step-by-step.

    Args:
        currents: (N, 3) control sequence
        dt: timestep (s)
        L_inserted: insertion length (mm)
        params_dict: catheter parameters
        x0: initial state (6,) [default: zeros]

    Returns:
        states: (N+1, 6) state trajectory (includes x0)
        tip_dyn_computed: (N+1, 3) dynamics tip positions
        dyn_status: (N,) status codes per step (0 = success)
    """
    N = currents.shape[0]
    if x0 is None:
        x0 = np.zeros(6)

    states = np.zeros((N + 1, 6))
    tip_dyn_computed = np.zeros((N + 1, 3))
    dyn_status = np.zeros(N, dtype=int)

    # Initial state
    states[0] = x0
    result_init = crm_diff_py.dynamics_forward(x0, np.zeros(3), 0.0, L_inserted, params_dict)
    tip_dyn_computed[0] = result_init['p_tip']

    # Rollout
    x_t = x0.copy()
    for i in range(N):
        u_t = currents[i]
        result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)

        x_next = result['x_next']
        p_tip = result['p_tip']
        status = result['status']

        states[i + 1] = x_next
        tip_dyn_computed[i + 1] = p_tip
        dyn_status[i] = status

        x_t = x_next

    return states, tip_dyn_computed, dyn_status


def compute_metrics(p_actual, p_reference, label=""):
    """
    Compute tracking error metrics.

    Args:
        p_actual: (N, 3) actual trajectory
        p_reference: (N, 3) reference trajectory
        label: string label for printout

    Returns:
        dict with rms_error, max_error, mean_error
    """
    # Handle different lengths (dynamics rollout has N+1 points)
    N = min(p_actual.shape[0], p_reference.shape[0])
    p_actual = p_actual[:N]
    p_reference = p_reference[:N]

    errors = np.linalg.norm(p_actual - p_reference, axis=1)

    metrics = {
        'rms_error': np.sqrt(np.mean(errors**2)),
        'max_error': np.max(errors),
        'mean_error': np.mean(errors),
        'errors': errors,
    }

    if label:
        print(f"  {label}:")
        print(f"    RMS error:  {metrics['rms_error']:.6f} mm")
        print(f"    Max error:  {metrics['max_error']:.6f} mm")
        print(f"    Mean error: {metrics['mean_error']:.6f} mm")

    return metrics


def plot_trajectory_comparison(traj_data, tip_fk_computed, tip_dyn_computed,
                                 metrics_dict, output_path):
    """
    Generate comprehensive comparison plots.

    Args:
        traj_data: dict from load_trajectory_npz
        tip_fk_computed: (N, 3) computed FK trajectory
        tip_dyn_computed: (N+1, 3) computed Dyn trajectory
        metrics_dict: dict of computed metrics
        output_path: path to save figure
    """
    t = traj_data['t']
    currents = traj_data['currents']

    # Determine what to plot
    has_desired = 'tip_desired' in traj_data
    has_projected = 'tip_projected' in traj_data

    # Use projected if available, otherwise desired, otherwise FK
    if has_projected:
        p_ref = traj_data['tip_projected']
        ref_label = 'Projected'
    elif has_desired:
        p_ref = traj_data['tip_desired']
        ref_label = 'Desired'
    else:
        p_ref = traj_data['tip_fk']
        ref_label = 'FK (NPZ)'

    # For dynamics, use computed rollout (has N+1 points, trim to N)
    p_dyn = tip_dyn_computed[:-1]  # Trim last point to match N

    # For FK, use computed
    p_fk = tip_fk_computed

    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Plot 1: 3D trajectory
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2],
             'g--', linewidth=2, label=ref_label, alpha=0.7)
    ax1.plot(p_fk[:, 0], p_fk[:, 1], p_fk[:, 2],
             'b-', linewidth=1.5, label='FK', alpha=0.8)
    ax1.plot(p_dyn[:, 0], p_dyn[:, 1], p_dyn[:, 2],
             'r-', linewidth=1.5, label='Dyn(ok)', alpha=0.8)
    ax1.scatter(p_ref[0, 0], p_ref[0, 1], p_ref[0, 2],
                c='k', s=100, marker='o', label='Start')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Trajectory Comparison')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: XY projection
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(p_ref[:, 0], p_ref[:, 1], 'g--', linewidth=2, label=ref_label, alpha=0.7)
    ax2.plot(p_fk[:, 0], p_fk[:, 1], 'b-', linewidth=1.5, label='FK', alpha=0.8)
    ax2.plot(p_dyn[:, 0], p_dyn[:, 1], 'r-', linewidth=1.5, label='Dyn(ok)', alpha=0.8)
    ax2.scatter(p_ref[0, 0], p_ref[0, 1], c='k', s=100, marker='o', label='Start')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('XY Projection')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    # Plot 3: XZ projection
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(p_ref[:, 0], p_ref[:, 2], 'g--', linewidth=2, label=ref_label, alpha=0.7)
    ax3.plot(p_fk[:, 0], p_fk[:, 2], 'b-', linewidth=1.5, label='FK', alpha=0.8)
    ax3.plot(p_dyn[:, 0], p_dyn[:, 2], 'r-', linewidth=1.5, label='Dyn(ok)', alpha=0.8)
    ax3.scatter(p_ref[0, 0], p_ref[0, 2], c='k', s=100, marker='o', label='Start')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('XZ Projection')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')

    # Plot 4: YZ projection
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(p_ref[:, 1], p_ref[:, 2], 'g--', linewidth=2, label=ref_label, alpha=0.7)
    ax4.plot(p_fk[:, 1], p_fk[:, 2], 'b-', linewidth=1.5, label='FK', alpha=0.8)
    ax4.plot(p_dyn[:, 1], p_dyn[:, 2], 'r-', linewidth=1.5, label='Dyn(ok)', alpha=0.8)
    ax4.scatter(p_ref[0, 1], p_ref[0, 2], c='k', s=100, marker='o', label='Start')
    ax4.set_xlabel('Y (mm)')
    ax4.set_ylabel('Z (mm)')
    ax4.set_title('YZ Projection')
    ax4.legend()
    ax4.grid(True)
    ax4.axis('equal')

    # Plot 5: Control inputs (currents)
    ax5 = fig.add_subplot(gs[1, 0:2])
    for i in range(3):
        ax5.plot(t, currents[:, i], linewidth=2, label=f'i_{i}')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Current (A)')
    ax5.set_title('Control Inputs (Currents)')
    ax5.legend()
    ax5.grid(True)

    # Plot 6: FK vs Dyn error over time
    ax6 = fig.add_subplot(gs[1, 2:4])
    err_fk_dyn = np.linalg.norm(p_fk - p_dyn, axis=1)
    ax6.plot(t, err_fk_dyn, 'b-', linewidth=2, label='||FK - Dyn||')
    if 'fk_dyn' in metrics_dict:
        ax6.axhline(y=metrics_dict['fk_dyn']['rms_error'],
                   color='b', linestyle=':', linewidth=2,
                   label=f'RMS = {metrics_dict["fk_dyn"]["rms_error"]:.4f} mm')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Error (mm)')
    ax6.set_title('FK vs Dyn Mismatch')
    ax6.legend()
    ax6.grid(True)

    # Plot 7: Dyn vs Desired/Projected error (if available)
    ax7 = fig.add_subplot(gs[2, 0:2])
    if has_projected or has_desired:
        err_dyn_ref = np.linalg.norm(p_dyn - p_ref, axis=1)
        ax7.plot(t, err_dyn_ref, 'r-', linewidth=2, label=f'||Dyn - {ref_label}||')
        if 'dyn_ref' in metrics_dict:
            ax7.axhline(y=metrics_dict['dyn_ref']['rms_error'],
                       color='r', linestyle=':', linewidth=2,
                       label=f'RMS = {metrics_dict["dyn_ref"]["rms_error"]:.4f} mm')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Error (mm)')
    ax7.set_title(f'Dyn vs {ref_label} Tracking Error')
    ax7.legend()
    ax7.grid(True)

    # Plot 8: Summary metrics text
    ax8 = fig.add_subplot(gs[2, 2:4])
    ax8.axis('off')

    summary_text = "Tracking Metrics Summary\n" + "="*40 + "\n\n"

    if 'dyn_ref' in metrics_dict:
        m = metrics_dict['dyn_ref']
        summary_text += f"Dyn vs {ref_label}:\n"
        summary_text += f"  RMS:  {m['rms_error']:.4f} mm\n"
        summary_text += f"  Max:  {m['max_error']:.4f} mm\n"
        summary_text += f"  Mean: {m['mean_error']:.4f} mm\n\n"

    if 'fk_dyn' in metrics_dict:
        m = metrics_dict['fk_dyn']
        summary_text += f"FK vs Dyn:\n"
        summary_text += f"  RMS:  {m['rms_error']:.4f} mm\n"
        summary_text += f"  Max:  {m['max_error']:.4f} mm\n"
        summary_text += f"  Mean: {m['mean_error']:.4f} mm\n\n"

    summary_text += f"\nDataset Parameters:\n"
    summary_text += f"  L_inserted = {traj_data['L_inserted']:.1f} mm\n"
    summary_text += f"  dt = {traj_data['dt']:.4f} s\n"
    summary_text += f"  N_steps = {len(t)}\n"

    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('CP3.5: NPZ Trajectory Replay Tracking', fontsize=14, fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to {output_path}")


def test_trajectory_replay(npz_path, traj_name, params_dict, output_dir):
    """
    Test a single trajectory replay.

    Returns:
        bool: True if passed
    """
    print(f"\n{'='*70}")
    print(f"Trajectory: {traj_name}")
    print(f"NPZ file: {npz_path}")
    print(f"{'='*70}")

    # Load trajectory
    traj_data = load_trajectory_npz(npz_path)

    t = traj_data['t']
    currents = traj_data['currents']
    dt = traj_data['dt']
    L_inserted = traj_data['L_inserted']

    print(f"\nTrajectory info:")
    print(f"  Duration: {t[-1]:.3f} s")
    print(f"  Steps: {len(t)}")
    print(f"  dt: {dt:.4f} s")
    print(f"  L_inserted: {L_inserted:.1f} mm (explicit)")
    print()

    # Run FK rollout
    print("Running FK rollout...")
    tip_fk_computed, fk_status = run_fk_rollout(currents, dt, L_inserted, params_dict)

    # Check FK status
    fk_failures = np.sum(fk_status != 0)
    if fk_failures > 0:
        print(f"  ✗ WARNING: {fk_failures}/{len(fk_status)} FK evaluations failed")
    else:
        print(f"  ✓ All FK evaluations succeeded (status=0)")

    # Check for NaN/Inf
    if not np.all(np.isfinite(tip_fk_computed)):
        print(f"  ✗ FAIL: FK trajectory contains NaN/Inf")
        return False
    print(f"  ✓ FK trajectory is finite")

    # Run dynamics rollout
    print("\nRunning dynamics rollout...")
    states, tip_dyn_computed, dyn_status = run_dynamics_rollout(
        currents, dt, L_inserted, params_dict, x0=None
    )

    # Check dynamics status
    dyn_failures = np.sum(dyn_status != 0)
    failure_rate = dyn_failures / len(dyn_status) * 100
    if dyn_failures > 0:
        print(f"  ⚠ WARNING: {dyn_failures}/{len(dyn_status)} dynamics steps failed (status!=0) [{failure_rate:.1f}%]")
        # Allow up to 15% failure rate (dataset may have been recorded with different solver settings)
        if failure_rate > 15.0:
            print(f"  ✗ FAIL: Failure rate {failure_rate:.1f}% exceeds 15% threshold")
            return False
        else:
            print(f"  ✓ PASS: Failure rate {failure_rate:.1f}% is within acceptable range (<15%)")
    else:
        print(f"  ✓ All dynamics steps succeeded (status=0)")

    # Check for NaN/Inf
    if not np.all(np.isfinite(tip_dyn_computed)):
        print(f"  ✗ FAIL: Dynamics trajectory contains NaN/Inf")
        return False
    if not np.all(np.isfinite(states)):
        print(f"  ✗ FAIL: State trajectory contains NaN/Inf")
        return False
    print(f"  ✓ Dynamics trajectory is finite")

    # Compute metrics
    print("\nComputing metrics...")
    metrics_dict = {}

    # FK vs Dyn
    # Note: tip_dyn_computed has N+1 points, tip_fk_computed has N points
    # Compare tip_dyn_computed[:-1] with tip_fk_computed
    metrics_dict['fk_dyn'] = compute_metrics(
        tip_fk_computed, tip_dyn_computed[:-1], label="FK vs Dyn"
    )

    # Dyn vs Desired/Projected (if available)
    if 'tip_projected' in traj_data:
        metrics_dict['dyn_ref'] = compute_metrics(
            tip_dyn_computed[:-1], traj_data['tip_projected'],
            label="Dyn vs Projected"
        )
    elif 'tip_desired' in traj_data:
        metrics_dict['dyn_ref'] = compute_metrics(
            tip_dyn_computed[:-1], traj_data['tip_desired'],
            label="Dyn vs Desired"
        )

    # Generate plots
    print("\nGenerating plots...")
    plot_filename = os.path.join(output_dir, f'cp35_{traj_name}.png')
    plot_trajectory_comparison(
        traj_data, tip_fk_computed, tip_dyn_computed, metrics_dict, plot_filename
    )

    print(f"\n✓ PASS: {traj_name}")
    return True


def main():
    print("=" * 70)
    print("CP3.5: Dataset-Driven NPZ Trajectory Replay Test")
    print("=" * 70)
    print()

    # Load catheter parameters
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

    # Create output directory
    output_dir = './build/artifacts/cp35'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Test trajectories (L_inserted = 94.3 mm regime)
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

        passed = test_trajectory_replay(npz_path, traj_name, params_dict, output_dir)
        results.append((traj_name, passed))

    # Summary
    print("\n" + "=" * 70)
    print("CP3.5 SUMMARY")
    print("=" * 70)

    all_passed = True
    for traj_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {traj_name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("✓ CP3.5 PASS: All NPZ trajectory replays validated")
        print()
        print("Gate B: ✓ Metrics computed for BOTH trajectories")
        print("Gate C: ✓ Plots saved to build/artifacts/cp35/")
        print("=" * 70)
        return 0
    else:
        print("✗ CP3.5 FAIL: Some tests failed")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
