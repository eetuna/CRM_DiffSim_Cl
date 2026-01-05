#!/usr/bin/env python3
"""
CP5.1: Real-NPZ Windowed Ensemble Training

Trains ensemble on WINDOWS extracted from real NPZ data (not full trajectories).
Uses hold-aware filtering + health gate to collect valid training windows.

Strategy:
1. Extract valid windows from real NPZ using CP4.7.8 health gate
2. For each valid window, run MPC expert to get reference actions
3. Train ensemble members via supervised learning (behavioral cloning)

This addresses CP5.0's domain shift problem by training directly on real data.
"""

import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from eval.cp47_health_gate import run_sliding_window_health_gate, create_windowed_manifest
from data.npz_manifest import load_manifest
from control.ilqr import iLQRSolver
from models.recurrent_policy import GRUPolicy
from models.ensemble_policy import EnsemblePolicy
import crm_diff_py


class WindowedBCDataset(Dataset):
    """Dataset for behavioral cloning on windows."""

    def __init__(self, p_tips, p_refs, u_experts, verbose=True):
        self.p_tips = p_tips.astype(np.float32)
        self.p_refs = p_refs.astype(np.float32)
        self.us = u_experts.astype(np.float32)

        if verbose:
            print(f"  Dataset: {len(self)} samples")

    def __len__(self):
        return len(self.p_tips)

    def __getitem__(self, idx):
        return self.p_tips[idx], self.p_refs[idx], self.us[idx]


def filter_real_datasets(datasets):
    """Filter to real NPZ datasets only."""
    return [
        ds for ds in datasets
        if not ('cp476' in ds.filename.lower() and 'golden' in ds.filename.lower())
    ]


def collect_window_expert_data(windowed_datasets, params_dict, mpc_horizon=10, max_iters=10, verbose=True):
    """
    Collect expert MPC actions for each windowed dataset.

    Returns:
        dict with p_tips, p_refs, u_experts arrays
    """
    all_p_tips = []
    all_p_refs = []
    all_u_experts = []

    n_windows = len(windowed_datasets)
    n_collected = 0

    for i, wds in enumerate(windowed_datasets):
        if verbose:
            print(f"  [{i+1}/{n_windows}] {wds.filename}")

        dt = wds.dt
        L_inserted = wds.L_inserted
        p_ref = wds.tip_ref
        n_steps = len(p_ref)

        # Initialize
        x_t = np.zeros(6)
        result_init = crm_diff_py.dynamics_forward(
            x_t, np.zeros(3), 0.0, L_inserted, params_dict
        )
        p_tip_t = result_init['p_tip']

        p_tips_window = []
        p_refs_window = []
        u_experts_window = []
        U_warm = None

        # Run MPC expert on this window
        for step in range(n_steps):
            # Get MPC horizon
            horizon_end = min(step + mpc_horizon, n_steps)
            p_ref_horizon = p_ref[step:horizon_end]

            if len(p_ref_horizon) < mpc_horizon:
                p_ref_horizon = np.vstack([
                    p_ref_horizon,
                    np.tile(p_ref_horizon[-1], (mpc_horizon - len(p_ref_horizon), 1))
                ])

            # Query MPC expert
            solver = iLQRSolver(
                dt=dt,
                L_inserted=L_inserted,
                params_dict=params_dict,
                horizon=mpc_horizon,
                Q=np.zeros((6, 6)),
                R=0.01 * np.eye(3),
                p_target=p_ref_horizon[0],
                terminal_weight=1.0,
                max_iters=max_iters,
                tol=1e-3,
                jacobian_mode="cpp"
            )

            try:
                X, U, converged = solver.solve(x_t, U_init=U_warm, verbose=False)
                u_expert = np.clip(U[0], -0.5, 0.5)
                U_warm = np.vstack([U[1:], U[-1:]])
            except:
                # Skip this window if MPC fails
                if verbose:
                    print(f"    MPC failure at step {step}/{n_steps}, skipping window")
                break

            # Execute action
            result = crm_diff_py.dynamics_forward(x_t, u_expert, dt, L_inserted, params_dict)

            if result['status'] != 0:
                # Skip this window if dynamics fails
                if verbose:
                    print(f"    Dynamics failure at step {step}/{n_steps}, skipping window")
                break

            # Record sample
            p_tips_window.append(p_tip_t)
            p_refs_window.append(p_ref[step])
            u_experts_window.append(u_expert)

            # Update state
            x_t = result['x_next']
            p_tip_t = result['p_tip']

        # Add window if complete
        if len(p_tips_window) == n_steps:
            all_p_tips.extend(p_tips_window)
            all_p_refs.extend(p_refs_window)
            all_u_experts.extend(u_experts_window)
            n_collected += 1

    if verbose:
        print(f"\n  Collected {n_collected}/{n_windows} complete windows")
        print(f"  Total samples: {len(all_p_tips)}")

    return {
        'p_tips': np.array(all_p_tips),
        'p_refs': np.array(all_p_refs),
        'u_experts': np.array(all_u_experts),
        'n_windows': n_collected
    }


def train_policy_bc(policy, train_loader, val_loader, epochs=50, lr=1e-3, patience=10, verbose=True):
    """Train a single policy via behavioral cloning."""
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        policy.train()
        train_loss = 0.0
        for p_tips, p_refs, u_experts in train_loader:
            # GRUPolicy handles 2D inputs [batch, 3] by adding sequence dim internally
            hidden = policy.init_hidden(batch_size=p_tips.size(0), device=p_tips.device)

            # Forward (policy adds/removes sequence dimension internally)
            u_pred, _ = policy(p_tips, p_refs, hidden)

            loss = criterion(u_pred, u_experts)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for p_tips, p_refs, u_experts in val_loader:
                hidden = policy.init_hidden(batch_size=p_tips.size(0), device=p_tips.device)
                u_pred, _ = policy(p_tips, p_refs, hidden)
                loss = criterion(u_pred, u_experts)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
            print(f"    Epoch {epoch:3d}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch}")
                break

    return {
        'final_train_loss': float(train_losses[-1]),
        'final_val_loss': float(val_losses[-1]),
        'best_val_loss': float(best_val_loss),
        'epochs_trained': len(train_losses)
    }


def train_ensemble_on_windows(
    windowed_datasets,
    params_dict,
    n_members=3,
    seeds=[42, 123, 456],
    epochs=50,
    batch_size=64,
    mpc_horizon=10,
    max_iters=10,
    output_prefix='cp51_real_ensemble',
    verbose=True
):
    """
    Train ensemble on windowed real NPZ data.

    Args:
        windowed_datasets: List of windowed datasets from health gate
        params_dict: Physics parameters
        n_members: Number of ensemble members
        seeds: Random seeds for each member
        epochs: Training epochs per member
        batch_size: Batch size
        mpc_horizon: MPC horizon for expert
        max_iters: MPC max iterations
        output_prefix: Output file prefix
        verbose: Verbosity

    Returns:
        tuple of (checkpoint_paths, metrics)
    """
    print("="*80)
    print("CP5.1: Windowed Ensemble Training")
    print("="*80)
    print(f"Windows: {len(windowed_datasets)}")
    print(f"Ensemble members: {n_members}")
    print(f"Epochs: {epochs}")
    print()

    # Collect expert data on windows
    print("[1/2] Collecting MPC expert data on windows...")
    t_collect_start = time.time()

    expert_data = collect_window_expert_data(
        windowed_datasets,
        params_dict,
        mpc_horizon=mpc_horizon,
        max_iters=max_iters,
        verbose=verbose
    )

    t_collect = time.time() - t_collect_start
    print(f"  Collection time: {t_collect:.1f}s")

    if expert_data['n_windows'] == 0:
        print("\n✗ No complete windows collected - cannot train")
        return None, None

    # Create train/val split
    n_samples = len(expert_data['p_tips'])
    indices = np.random.permutation(n_samples)
    n_val = max(1, int(0.2 * n_samples))
    n_train = n_samples - n_val

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = WindowedBCDataset(
        expert_data['p_tips'][train_indices],
        expert_data['p_refs'][train_indices],
        expert_data['u_experts'][train_indices],
        verbose=verbose
    )

    val_dataset = WindowedBCDataset(
        expert_data['p_tips'][val_indices],
        expert_data['p_refs'][val_indices],
        expert_data['u_experts'][val_indices],
        verbose=False
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"\n  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")

    # Train ensemble members
    print(f"\n[2/2] Training {n_members} ensemble members...")

    checkpoint_paths = []
    member_metrics = []

    output_dir = './build/artifacts'
    os.makedirs(output_dir, exist_ok=True)

    for member_id in range(n_members):
        seed = seeds[member_id]
        print(f"\n  Member {member_id} (seed={seed})")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create policy
        policy = GRUPolicy(input_dim=6, hidden_dim=64, output_dim=3, num_layers=1)

        # Train
        t_train_start = time.time()
        metrics = train_policy_bc(
            policy, train_loader, val_loader,
            epochs=epochs,
            lr=1e-3,
            patience=10,
            verbose=verbose
        )
        t_train = time.time() - t_train_start

        # Save checkpoint
        checkpoint_path = os.path.join(
            output_dir,
            f"{output_prefix}_member{member_id}_seed{seed}_policy.pth"
        )
        torch.save(policy.state_dict(), checkpoint_path)
        checkpoint_paths.append(checkpoint_path)

        metrics['member_id'] = member_id
        metrics['seed'] = seed
        metrics['train_time'] = t_train
        metrics['checkpoint_path'] = checkpoint_path
        member_metrics.append(metrics)

        print(f"    ✓ Trained in {t_train:.1f}s, val_loss={metrics['best_val_loss']:.6f}")

    # Save ensemble metadata
    metadata_path = os.path.join(output_dir, f"{output_prefix}_ensemble_metadata.json")
    metadata = {
        'n_members': n_members,
        'checkpoint_paths': checkpoint_paths,
        'policy_kwargs': {
            'input_dim': 6,
            'hidden_dim': 64,
            'output_dim': 3,
            'num_layers': 1
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✓ Ensemble metadata saved: {metadata_path}")

    # Save training metrics
    metrics_path = os.path.join(output_dir, f"{output_prefix}_metrics.json")
    full_metrics = {
        'n_members': n_members,
        'n_windows_collected': expert_data['n_windows'],
        'n_samples': n_samples,
        'collection_time': t_collect,
        'member_metrics': member_metrics
    }

    with open(metrics_path, 'w') as f:
        json.dump(full_metrics, f, indent=2)

    print(f"✓ Training metrics saved: {metrics_path}")

    return checkpoint_paths, full_metrics


def main():
    import argparse

    parser = argparse.ArgumentParser(description='CP5.1: Windowed Ensemble Training on Real NPZ')
    parser.add_argument('--max_datasets', type=int, default=None,
                       help='Limit number of datasets to use')
    parser.add_argument('--max_windows', type=int, default=20,
                       help='Max windows to collect (default: 20)')
    parser.add_argument('--window_steps', type=int, default=20,
                       help='Window size in steps (default: 20)')
    parser.add_argument('--stride_steps', type=int, default=20,
                       help='Stride between windows (default: 20)')
    parser.add_argument('--n_members', type=int, default=3,
                       help='Number of ensemble members (default: 3)')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Training epochs per member (default: 50)')
    parser.add_argument('--fast', action='store_true',
                       help='Fast mode: fewer windows, fewer epochs')
    args = parser.parse_args()

    if args.fast:
        args.max_windows = 5
        args.epochs = 20
        print("[FAST MODE: 5 windows, 20 epochs]")

    print("="*80)
    print("CP5.1: Real-NPZ Windowed Ensemble Training")
    print("="*80)
    print()

    # Load real datasets
    print("[Loading real NPZ datasets...]")
    manifest = load_manifest(data_dir='./data', verbose=False)
    real_datasets = filter_real_datasets(manifest.datasets_with_ref)

    if not real_datasets:
        print("✗ No real datasets found")
        return 1

    if args.max_datasets:
        real_datasets = real_datasets[:args.max_datasets]

    print(f"✓ Using {len(real_datasets)} real dataset(s)")
    for ds in real_datasets:
        print(f"  - {ds.filename}")

    # Load physics parameters
    print("\n[Loading physics parameters...]")
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

    # Run health gate to get valid windows
    print(f"\n[Running health gate to collect valid windows...]")
    print(f"  Window size: {args.window_steps} steps")
    print(f"  Stride: {args.stride_steps} steps")
    print(f"  Max windows: {args.max_windows}")

    t_health_start = time.time()

    health_report = run_sliding_window_health_gate(
        real_datasets,
        params_dict,
        window_steps=args.window_steps,
        stride_steps=args.stride_steps,
        max_windows=args.max_windows,
        mpc_horizon=10,
        max_iters=10,
        verbose=True,
        skip_hold_periods=True  # CP4.7.8 mitigation
    )

    t_health = time.time() - t_health_start

    valid_window_count = health_report['summary']['valid_window_count']

    if valid_window_count == 0:
        print("\n✗ No valid windows found")
        return 1

    print(f"\n✓ Found {valid_window_count} valid window(s) in {t_health:.1f}s")

    # Create windowed manifest
    windowed_datasets = create_windowed_manifest(real_datasets, health_report['valid_windows'])

    # Train ensemble
    print("\n[Training ensemble on windows...]")

    t_train_start = time.time()

    checkpoint_paths, metrics = train_ensemble_on_windows(
        windowed_datasets,
        params_dict,
        n_members=args.n_members,
        seeds=[42, 123, 456, 789, 1024][:args.n_members],
        epochs=args.epochs,
        batch_size=64,
        mpc_horizon=10,
        max_iters=10,
        output_prefix='cp51_real_ensemble',
        verbose=True
    )

    t_train_total = time.time() - t_train_start

    if checkpoint_paths is None:
        print("\n✗ Training failed")
        return 1

    print("\n" + "="*80)
    print("✓ CP5.1 Windowed Ensemble Training Complete")
    print("="*80)
    print(f"Total time: {(t_health + t_train_total)/60:.1f} minutes")
    print(f"Ensemble members: {len(checkpoint_paths)}")
    print(f"Windows used: {metrics['n_windows_collected']}")
    print(f"Training samples: {metrics['n_samples']}")
    print("="*80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
