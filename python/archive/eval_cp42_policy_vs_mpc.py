#!/usr/bin/env python3
"""
CP4.2: Policy vs MPC Benchmark Evaluation

Evaluates and compares:
1. Behavior Cloning (BC) policy rollout through dynamics
2. MPC tracking rollout through dynamics

Against NPZ reference trajectories (tip_ref).

Key features:
- Uses CP4.1 manifest + dataset contract
- Only evaluates datasets with ds.has_reference == True
- Uses dt/L_inserted/integration_step_size from dataset metadata
- Generates comprehensive plots and metrics JSON
"""

import sys
import os
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from data.npz_manifest import load_manifest
from data.npz_dataset import NPZDataset
import crm_diff_py

# Import BC policy
import torch.nn as nn

class BehaviorCloningPolicy(nn.Module):
    """BC policy (same architecture as training)."""
    def __init__(self, input_dim=6, hidden_dim=64, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, p_tip, p_ref):
        x = torch.cat([p_tip, p_ref], dim=-1)
        return self.net(x)

    def predict(self, p_tip, p_ref):
        self.eval()
        with torch.no_grad():
            p_tip_t = torch.from_numpy(p_tip).float().unsqueeze(0)
            p_ref_t = torch.from_numpy(p_ref).float().unsqueeze(0)
            u_t = self.forward(p_tip_t, p_ref_t)
            return u_t.squeeze(0).numpy()


# Import MPC (simplified version for evaluation)
from control.ilqr import iLQRSolver

def run_mpc_tracking(dataset, duration, params_dict, horizon=10, verbose=False):
    """
    Run MPC tracking on dataset reference trajectory.

    Args:
        dataset: NPZDataset with reference
        duration: Rollout duration (seconds)
        params_dict: Physics parameters
        horizon: MPC planning horizon
        verbose: Print progress

    Returns:
        dict: {p_tips, us, statuses, n_steps}
    """
    dt = dataset.dt
    L_inserted = dataset.L_inserted
    n_steps = min(int(duration / dt), len(dataset.tip_ref))

    # Get reference
    p_ref = dataset.tip_ref[:n_steps]

    # Initial state
    x_t = np.zeros(6)

    # MPC setup
    Q_tip = 1.0
    R = 0.01 * np.eye(3)

    # Storage
    p_tips = []
    us = []
    statuses = []

    # Warm-start for iLQR
    U_init = None

    for i in range(n_steps):
        if verbose and i % 50 == 0:
            print(f"  MPC step {i}/{n_steps}")

        # Reference for this horizon
        horizon_end = min(i + horizon, n_steps)
        p_ref_horizon = p_ref[i:horizon_end]

        # Pad if needed
        if len(p_ref_horizon) < horizon:
            p_ref_horizon = np.vstack([
                p_ref_horizon,
                np.tile(p_ref_horizon[-1], (horizon - len(p_ref_horizon), 1))
            ])

        # Solve iLQR
        solver = iLQRSolver(
            dt=dt,
            L_inserted=L_inserted,
            params_dict=params_dict,
            horizon=horizon,
            Q_tip=Q_tip,
            R=R,
            max_iters=5,  # Fast for evaluation
            cost_decrease_tol=1e-2
        )

        try:
            result = solver.solve(x_t, p_ref_horizon, U_init=U_init)
            u_t = result['U'][0]  # First control
            U_init = np.vstack([result['U'][1:], result['U'][-1:]])  # Warm-start
        except:
            # Fallback: zero control
            u_t = np.zeros(3)
            U_init = None

        # Clip controls
        u_t = np.clip(u_t, -0.5, 0.5)

        # Step dynamics
        dyn_result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)
        x_t = dyn_result['x_next']
        p_tip = dyn_result['p_tip']
        status = dyn_result['status']

        p_tips.append(p_tip)
        us.append(u_t)
        statuses.append(status)

    return {
        'p_tips': np.array(p_tips),
        'us': np.array(us),
        'statuses': np.array(statuses),
        'n_steps': n_steps
    }


def run_bc_rollout(policy, dataset, duration, params_dict, verbose=False):
    """
    Run BC policy rollout through dynamics.

    Args:
        policy: Trained BC policy
        dataset: NPZDataset with reference
        duration: Rollout duration (seconds)
        params_dict: Physics parameters
        verbose: Print progress

    Returns:
        dict: {p_tips, us, statuses, n_steps}
    """
    dt = dataset.dt
    L_inserted = dataset.L_inserted
    n_steps = min(int(duration / dt), len(dataset.tip_ref))

    # Get reference
    p_ref = dataset.tip_ref[:n_steps]

    # Initial state
    x_t = np.zeros(6)

    # Storage
    p_tips = []
    us = []
    statuses = []

    # Get initial tip position
    result_init = crm_diff_py.dynamics_forward(x_t, np.zeros(3), 0.0, L_inserted, params_dict)
    p_tip_t = result_init['p_tip']

    for i in range(n_steps):
        if verbose and i % 50 == 0:
            print(f"  BC step {i}/{n_steps}")

        # Predict control
        u_t = policy.predict(p_tip_t.astype(np.float32), p_ref[i].astype(np.float32))
        u_t = u_t.astype(np.float64)

        # Clip controls
        u_t = np.clip(u_t, -0.5, 0.5)

        # Step dynamics
        result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)
        x_t = result['x_next']
        p_tip_t = result['p_tip']
        status = result['status']

        p_tips.append(p_tip_t)
        us.append(u_t)
        statuses.append(status)

    return {
        'p_tips': np.array(p_tips),
        'us': np.array(us),
        'statuses': np.array(statuses),
        'n_steps': n_steps
    }


def compute_tracking_metrics(p_actual, p_ref):
    """Compute tracking error metrics."""
    errors = np.linalg.norm(p_actual - p_ref, axis=1)
    return {
        'rms': float(np.sqrt(np.mean(errors**2))),
        'max': float(np.max(errors)),
        'mean': float(np.mean(errors)),
        'std': float(np.std(errors)),
        'errors': errors
    }


def plot_evaluation(dataset, bc_result, mpc_result, output_path):
    """
    Generate comprehensive evaluation plot.

    Shows:
    - 3D trajectory + XY/XZ/YZ projections
    - Control inputs over time
    - Tracking errors over time
    """
    n_steps = bc_result['n_steps']
    t = np.arange(n_steps) * dataset.dt
    p_ref = dataset.tip_ref[:n_steps]

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Plot 1: 3D trajectory
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.plot(p_ref[:, 0], p_ref[:, 1], p_ref[:, 2], 'g--', linewidth=2, label='Reference', alpha=0.7)
    ax1.plot(bc_result['p_tips'][:, 0], bc_result['p_tips'][:, 1], bc_result['p_tips'][:, 2],
             'b-', linewidth=1.5, label='BC', alpha=0.8)
    ax1.plot(mpc_result['p_tips'][:, 0], mpc_result['p_tips'][:, 1], mpc_result['p_tips'][:, 2],
             'r-', linewidth=1.5, label='MPC', alpha=0.8)
    ax1.scatter(p_ref[0, 0], p_ref[0, 1], p_ref[0, 2], c='k', s=100, marker='o', label='Start')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_zlabel('Z (mm)')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: XY projection
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(p_ref[:, 0], p_ref[:, 1], 'g--', linewidth=2, label='Ref', alpha=0.7)
    ax2.plot(bc_result['p_tips'][:, 0], bc_result['p_tips'][:, 1], 'b-', linewidth=1.5, label='BC', alpha=0.8)
    ax2.plot(mpc_result['p_tips'][:, 0], mpc_result['p_tips'][:, 1], 'r-', linewidth=1.5, label='MPC', alpha=0.8)
    ax2.scatter(p_ref[0, 0], p_ref[0, 1], c='k', s=100, marker='o')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_title('XY Projection')
    ax2.legend()
    ax2.grid(True)
    ax2.axis('equal')

    # Plot 3: XZ projection
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(p_ref[:, 0], p_ref[:, 2], 'g--', linewidth=2, alpha=0.7)
    ax3.plot(bc_result['p_tips'][:, 0], bc_result['p_tips'][:, 2], 'b-', linewidth=1.5, label='BC', alpha=0.8)
    ax3.plot(mpc_result['p_tips'][:, 0], mpc_result['p_tips'][:, 2], 'r-', linewidth=1.5, label='MPC', alpha=0.8)
    ax3.scatter(p_ref[0, 0], p_ref[0, 2], c='k', s=100, marker='o')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Z (mm)')
    ax3.set_title('XZ Projection')
    ax3.legend()
    ax3.grid(True)
    ax3.axis('equal')

    # Plot 4: YZ projection
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.plot(p_ref[:, 1], p_ref[:, 2], 'g--', linewidth=2, alpha=0.7)
    ax4.plot(bc_result['p_tips'][:, 1], bc_result['p_tips'][:, 2], 'b-', linewidth=1.5, label='BC', alpha=0.8)
    ax4.plot(mpc_result['p_tips'][:, 1], mpc_result['p_tips'][:, 2], 'r-', linewidth=1.5, label='MPC', alpha=0.8)
    ax4.scatter(p_ref[0, 1], p_ref[0, 2], c='k', s=100, marker='o')
    ax4.set_xlabel('Y (mm)')
    ax4.set_ylabel('Z (mm)')
    ax4.set_title('YZ Projection')
    ax4.legend()
    ax4.grid(True)
    ax4.axis('equal')

    # Plot 5: BC controls
    ax5 = fig.add_subplot(gs[1, 0:2])
    for i in range(3):
        ax5.plot(t, bc_result['us'][:, i], linewidth=2, label=f'BC i_{i}')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Current (A)')
    ax5.set_title('BC Control Inputs')
    ax5.legend()
    ax5.grid(True)

    # Plot 6: MPC controls
    ax6 = fig.add_subplot(gs[1, 2:4])
    for i in range(3):
        ax6.plot(t, mpc_result['us'][:, i], linewidth=2, label=f'MPC i_{i}')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Current (A)')
    ax6.set_title('MPC Control Inputs')
    ax6.legend()
    ax6.grid(True)

    # Plot 7: Tracking errors
    bc_metrics = compute_tracking_metrics(bc_result['p_tips'], p_ref)
    mpc_metrics = compute_tracking_metrics(mpc_result['p_tips'], p_ref)

    ax7 = fig.add_subplot(gs[2, 0:2])
    ax7.plot(t, bc_metrics['errors'], 'b-', linewidth=2, label=f'BC (RMS={bc_metrics["rms"]:.3f}mm)')
    ax7.plot(t, mpc_metrics['errors'], 'r-', linewidth=2, label=f'MPC (RMS={mpc_metrics["rms"]:.3f}mm)')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Tracking Error (mm)')
    ax7.set_title('Tracking Error vs Time')
    ax7.legend()
    ax7.grid(True)

    # Plot 8: Summary metrics
    ax8 = fig.add_subplot(gs[2, 2:4])
    ax8.axis('off')

    summary_text = "Evaluation Metrics\n" + "="*40 + "\n\n"
    summary_text += f"BC Policy:\n"
    summary_text += f"  RMS:  {bc_metrics['rms']:.4f} mm\n"
    summary_text += f"  Max:  {bc_metrics['max']:.4f} mm\n"
    summary_text += f"  Mean: {bc_metrics['mean']:.4f} mm\n"
    summary_text += f"  Solver failures: {np.sum(bc_result['statuses'] != 0)}/{n_steps}\n\n"

    summary_text += f"MPC Tracking:\n"
    summary_text += f"  RMS:  {mpc_metrics['rms']:.4f} mm\n"
    summary_text += f"  Max:  {mpc_metrics['max']:.4f} mm\n"
    summary_text += f"  Mean: {mpc_metrics['mean']:.4f} mm\n"
    summary_text += f"  Solver failures: {np.sum(mpc_result['statuses'] != 0)}/{n_steps}\n\n"

    summary_text += f"Dataset:\n"
    summary_text += f"  {dataset.param_summary()}\n"
    summary_text += f"  Duration: {t[-1]:.2f} s\n"
    summary_text += f"  Steps: {n_steps}\n"

    ax8.text(0.1, 0.9, summary_text, transform=ax8.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'CP4.2: BC vs MPC Evaluation - {dataset.filename}',
                 fontsize=14, fontweight='bold')

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot: {output_path}")


def evaluate_dataset(dataset, policy, params_dict, duration, output_dir):
    """Evaluate BC and MPC on a single dataset."""
    print(f"\n{'='*80}")
    print(f"Evaluating: {dataset.filename}")
    print(f"{'='*80}")
    print(f"  {dataset.param_summary()}")
    print(f"  Duration: {duration:.1f}s")

    # Run BC rollout
    print("\n  Running BC rollout...")
    bc_result = run_bc_rollout(policy, dataset, duration, params_dict, verbose=True)
    bc_metrics = compute_tracking_metrics(bc_result['p_tips'], dataset.tip_ref[:bc_result['n_steps']])
    print(f"    BC RMS error: {bc_metrics['rms']:.4f} mm")

    # Run MPC rollout
    print("\n  Running MPC rollout...")
    mpc_result = run_mpc_tracking(dataset, duration, params_dict, verbose=True)
    mpc_metrics = compute_tracking_metrics(mpc_result['p_tips'], dataset.tip_ref[:mpc_result['n_steps']])
    print(f"    MPC RMS error: {mpc_metrics['rms']:.4f} mm")

    # Generate plot
    plot_path = os.path.join(output_dir, f'cp42_{dataset.filename.replace(".npz", ".png")}')
    plot_evaluation(dataset, bc_result, mpc_result, plot_path)

    return {
        'filename': dataset.filename,
        'params': {
            'dt': dataset.dt,
            'L_inserted': dataset.L_inserted,
            'integration_step_size': dataset.integration_step_size
        },
        'bc': bc_metrics,
        'mpc': mpc_metrics,
        'n_steps': bc_result['n_steps'],
        'duration': bc_result['n_steps'] * dataset.dt
    }


def main():
    print("=" * 80)
    print("CP4.2: Policy vs MPC Benchmark Evaluation")
    print("=" * 80)
    print()

    # Load manifest
    manifest = load_manifest(data_dir='data', verbose=False)
    datasets_with_ref = manifest.datasets_with_ref

    if not datasets_with_ref:
        print("✗ No datasets with references available")
        return 1

    print(f"Found {len(datasets_with_ref)} datasets with references")

    # Load BC policy
    policy_path = './build/artifacts/cp41_bc_multitraj_policy.pth'
    if not os.path.exists(policy_path):
        print(f"✗ BC policy not found: {policy_path}")
        print("  Run: python3 python/train_cp41_bc_multitraj.py first")
        return 1

    policy = BehaviorCloningPolicy()
    policy.load_state_dict(torch.load(policy_path))
    policy.eval()
    print(f"✓ Loaded BC policy from {policy_path}")

    # Load physics parameters
    param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
    cath_params = crm_diff_py.load_cath_params(param_file)
    cath_config = crm_diff_py.load_cath_config(config_file)

    # Create output directory
    output_dir = './build/artifacts/cp42_eval'
    os.makedirs(output_dir, exist_ok=True)

    # Evaluate each dataset
    all_metrics = []
    eval_duration = 5.0  # 5 seconds per dataset

    for dataset in datasets_with_ref[:2]:  # Limit to 2 for speed
        # Create params_dict with dataset-specific parameters
        params_dict = {
            'CathParams': cath_params,
            'CathConfig': cath_config,
            'ContactMode': int(crm_diff_py.ContactModeType.FREE_TIP),
            'TipForce': [0.0, 0.0, 0.0],
            'deltau0_initialguess': [0.0, 0.0, 0.0],
            'IntegrationStepSize': dataset.integration_step_size,
            'FinalValueOnly': True,
        }

        metrics = evaluate_dataset(dataset, policy, params_dict, eval_duration, output_dir)
        all_metrics.append(metrics)

    # Compute aggregate metrics
    bc_rms_all = [m['bc']['rms'] for m in all_metrics]
    mpc_rms_all = [m['mpc']['rms'] for m in all_metrics]

    aggregate = {
        'bc': {
            'mean_rms': float(np.mean(bc_rms_all)),
            'std_rms': float(np.std(bc_rms_all)),
            'min_rms': float(np.min(bc_rms_all)),
            'max_rms': float(np.max(bc_rms_all))
        },
        'mpc': {
            'mean_rms': float(np.mean(mpc_rms_all)),
            'std_rms': float(np.std(mpc_rms_all)),
            'min_rms': float(np.min(mpc_rms_all)),
            'max_rms': float(np.max(mpc_rms_all))
        }
    }

    # Save metrics
    metrics_output = {
        'per_trajectory': all_metrics,
        'aggregate': aggregate,
        'n_datasets': len(all_metrics)
    }

    metrics_path = './build/artifacts/cp42_eval_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print(f"\n✓ Metrics saved to {metrics_path}")

    # Summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Evaluated: {len(all_metrics)} datasets")
    print(f"\nBC Policy:")
    print(f"  Mean RMS: {aggregate['bc']['mean_rms']:.4f} ± {aggregate['bc']['std_rms']:.4f} mm")
    print(f"  Range: [{aggregate['bc']['min_rms']:.4f}, {aggregate['bc']['max_rms']:.4f}] mm")
    print(f"\nMPC Tracking:")
    print(f"  Mean RMS: {aggregate['mpc']['mean_rms']:.4f} ± {aggregate['mpc']['std_rms']:.4f} mm")
    print(f"  Range: [{aggregate['mpc']['min_rms']:.4f}, {aggregate['mpc']['max_rms']:.4f}] mm")
    print("=" * 80)
    print(f"\n✓ Plots saved to {output_dir}/")

    return 0


if __name__ == "__main__":
    sys.exit(main())
