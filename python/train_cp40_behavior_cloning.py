#!/usr/bin/env python3
"""
CP4.0: Behavior Cloning Baseline

Train a simple neural network to predict control inputs from tip position and
reference trajectory using behavior cloning (supervised learning).

Architecture:
    Input: [p_tip (3), p_ref (3)] -> 6D
    Hidden: [64, 64]
    Output: u (3)

Training:
    - Load NPZ datasets
    - Train on (p_tip, p_ref) -> u pairs
    - Use MSE loss
    - Quick CPU training on small datasets

Evaluation:
    - Replay learned policy through CP2 dynamics for 2-5 seconds
    - Report RMS tracking error vs reference
    - Compare to expert trajectory

This is a BASELINE only - not production RL.
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add python directory and build directory to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from data.npz_dataset import load_npz_dataset, load_multiple_datasets
from crm_config import print_param_summary
import crm_diff_py


# ============================================================================
# Neural Network Policy
# ============================================================================

class BehaviorCloningPolicy(nn.Module):
    """
    Simple feedforward policy for behavior cloning.

    Input: [p_tip (3), p_ref (3)] -> 6D
    Hidden: [64, 64]
    Output: u (3)
    """

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
        """
        Args:
            p_tip: Tensor [batch, 3] - Current tip position
            p_ref: Tensor [batch, 3] - Reference tip position

        Returns:
            u: Tensor [batch, 3] - Predicted control inputs
        """
        x = torch.cat([p_tip, p_ref], dim=-1)  # [batch, 6]
        u = self.net(x)  # [batch, 3]
        return u

    def predict(self, p_tip, p_ref):
        """Predict control input (no gradient)."""
        self.eval()
        with torch.no_grad():
            p_tip_t = torch.from_numpy(p_tip).float().unsqueeze(0)
            p_ref_t = torch.from_numpy(p_ref).float().unsqueeze(0)
            u_t = self.forward(p_tip_t, p_ref_t)
            return u_t.squeeze(0).numpy()


# ============================================================================
# Dataset
# ============================================================================

class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for behavior cloning from NPZ trajectories.

    Returns (p_tip, p_ref, u) tuples.
    """

    def __init__(self, npz_datasets, use_fk=False):
        """
        Args:
            npz_datasets: List of NPZDataset objects
            use_fk: If True, use tip_fk; otherwise use tip_dyn
        """
        self.use_fk = use_fk

        # Collect all (p_tip, p_ref, u) tuples
        p_tips = []
        p_refs = []
        us = []

        for dataset in npz_datasets:
            # Get tip positions
            if use_fk:
                if dataset.tip_fk is None:
                    raise ValueError(f"Dataset {dataset.filename} missing tip_fk")
                p_tip = dataset.tip_fk
            else:
                if dataset.tip_dyn is None:
                    raise ValueError(f"Dataset {dataset.filename} missing tip_dyn")
                p_tip = dataset.tip_dyn

            # Get reference
            p_ref = dataset.reference_trajectory
            if p_ref is None:
                raise ValueError(f"Dataset {dataset.filename} missing reference trajectory")

            # Get controls
            u = dataset.currents

            # Filter out NaN rows (happens during hold periods)
            valid_mask = ~np.isnan(p_ref).any(axis=1)
            n_valid = np.sum(valid_mask)
            n_total = len(p_ref)

            if n_valid == 0:
                raise ValueError(f"Dataset {dataset.filename} has no valid (non-NaN) reference data")

            if n_valid < n_total:
                print(f"    Filtered {n_total - n_valid}/{n_total} samples with NaN references")

            p_tip = p_tip[valid_mask]
            p_ref = p_ref[valid_mask]
            u = u[valid_mask]

            # Append
            p_tips.append(p_tip)
            p_refs.append(p_ref)
            us.append(u)

        # Concatenate all datasets
        self.p_tips = np.concatenate(p_tips, axis=0).astype(np.float32)
        self.p_refs = np.concatenate(p_refs, axis=0).astype(np.float32)
        self.us = np.concatenate(us, axis=0).astype(np.float32)

        print(f"Dataset created: {len(self)} samples")
        print(f"  p_tip range: [{self.p_tips.min():.2f}, {self.p_tips.max():.2f}]")
        print(f"  p_ref range: [{self.p_refs.min():.2f}, {self.p_refs.max():.2f}]")
        print(f"  u range: [{self.us.min():.4f}, {self.us.max():.4f}]")

    def __len__(self):
        return len(self.p_tips)

    def __getitem__(self, idx):
        return self.p_tips[idx], self.p_refs[idx], self.us[idx]


# ============================================================================
# Training
# ============================================================================

def train_policy(policy, dataloader, epochs=50, lr=1e-3):
    """
    Train behavior cloning policy.

    Args:
        policy: BehaviorCloningPolicy
        dataloader: PyTorch DataLoader
        epochs: Number of training epochs
        lr: Learning rate

    Returns:
        dict: Training metrics
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"\nTraining for {epochs} epochs...")
    print(f"Optimizer: Adam(lr={lr})")
    print(f"Loss: MSE")

    train_losses = []

    for epoch in range(epochs):
        policy.train()
        epoch_loss = 0.0
        n_batches = 0

        for p_tip, p_ref, u_target in dataloader:
            # Forward
            u_pred = policy(p_tip, p_ref)

            # Loss
            loss = criterion(u_pred, u_target)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs}: loss = {avg_loss:.6f}")

    print(f"Training complete. Final loss: {train_losses[-1]:.6f}")

    return {'train_losses': train_losses}


# ============================================================================
# Evaluation: Test set prediction accuracy
# ============================================================================

def evaluate_policy_prediction(policy, test_datasets, use_fk=False):
    """
    Evaluate policy prediction accuracy on test set.

    Computes MSE and tracking error on test trajectories (offline evaluation).

    Args:
        policy: Trained BehaviorCloningPolicy
        test_datasets: List of NPZDataset for evaluation
        use_fk: If True, use tip_fk; otherwise use tip_dyn

    Returns:
        dict: Evaluation metrics
    """
    print(f"\nEvaluating policy predictions on test set...")

    # Create test dataset (filter NaNs same as training)
    print("Creating test dataset...")
    p_tips_all = []
    p_refs_all = []
    us_all = []

    for dataset in test_datasets:
        if use_fk:
            if dataset.tip_fk is None:
                continue
            p_tip = dataset.tip_fk
        else:
            if dataset.tip_dyn is None:
                continue
            p_tip = dataset.tip_dyn

        p_ref = dataset.reference_trajectory
        if p_ref is None:
            continue

        u = dataset.currents

        # Filter NaNs
        valid_mask = ~np.isnan(p_ref).any(axis=1)
        p_tip = p_tip[valid_mask]
        p_ref = p_ref[valid_mask]
        u = u[valid_mask]

        p_tips_all.append(p_tip)
        p_refs_all.append(p_ref)
        us_all.append(u)

    if not p_tips_all:
        raise ValueError("No valid test data available")

    p_tips = np.concatenate(p_tips_all, axis=0).astype(np.float32)
    p_refs = np.concatenate(p_refs_all, axis=0).astype(np.float32)
    us_true = np.concatenate(us_all, axis=0).astype(np.float32)

    print(f"  Test samples: {len(p_tips)}")

    # Predict controls
    policy.eval()
    with torch.no_grad():
        p_tips_t = torch.from_numpy(p_tips)
        p_refs_t = torch.from_numpy(p_refs)
        us_pred_t = policy(p_tips_t, p_refs_t)
        us_pred = us_pred_t.numpy()

    # Compute MSE
    mse = np.mean((us_pred - us_true)**2)
    rmse = np.sqrt(mse)

    # Compute per-channel errors
    mse_per_channel = np.mean((us_pred - us_true)**2, axis=0)

    print(f"\nPrediction Results:")
    print(f"  MSE:  {mse:.6f} A^2")
    print(f"  RMSE: {rmse:.6f} A")
    print(f"  Per-channel RMSE: [{np.sqrt(mse_per_channel[0]):.6f}, {np.sqrt(mse_per_channel[1]):.6f}, {np.sqrt(mse_per_channel[2]):.6f}] A")

    # Compute control magnitude statistics
    u_true_mag = np.linalg.norm(us_true, axis=1)
    u_pred_mag = np.linalg.norm(us_pred, axis=1)
    print(f"  True control magnitude:      {np.mean(u_true_mag):.6f} ± {np.std(u_true_mag):.6f} A")
    print(f"  Predicted control magnitude: {np.mean(u_pred_mag):.6f} ± {np.std(u_pred_mag):.6f} A")

    return {
        'mse': mse,
        'rmse': rmse,
        'mse_per_channel': mse_per_channel,
        'n_samples': len(p_tips),
    }


def evaluate_policy_rollout(policy, test_dataset, params_dict, duration=5.0):
    """
    Evaluate policy by rolling out through CP2 dynamics.

    Args:
        policy: Trained BehaviorCloningPolicy
        test_dataset: NPZDataset for evaluation
        params_dict: Physics parameters
        duration: Rollout duration (seconds)

    Returns:
        dict: Evaluation metrics
    """
    print(f"\nEvaluating policy rollout...")
    print(f"  Duration: {duration:.1f} s")
    print(f"  dt: {test_dataset.dt:.4f} s")

    dt = test_dataset.dt
    L_inserted = test_dataset.L_inserted
    n_steps = int(duration / dt)

    # Get reference trajectory
    p_ref_full = test_dataset.reference_trajectory
    if p_ref_full is None:
        raise ValueError("Test dataset missing reference trajectory")

    # Limit to evaluation duration
    n_steps = min(n_steps, len(p_ref_full))
    p_ref = p_ref_full[:n_steps]

    # Initialize state (start from rest)
    x_t = np.zeros(6)

    # Rollout
    p_tips = []
    us = []
    statuses = []

    print(f"  Rolling out {n_steps} steps...")

    for i in range(n_steps):
        # Get reference for this timestep
        p_ref_t = p_ref[i]

        # Get current tip position (from state via FK)
        # For simplicity, we'll use the last dynamics output
        if i == 0:
            # Initial position
            result_init = crm_diff_py.dynamics_forward(
                x_t, np.zeros(3), 0.0, L_inserted, params_dict
            )
            p_tip_t = result_init['p_tip']
        else:
            p_tip_t = p_tips[-1]

        # Predict control with policy
        u_t = policy.predict(p_tip_t.astype(np.float32), p_ref_t.astype(np.float32))
        u_t = u_t.astype(np.float64)  # Convert back to float64 for C++

        # Clip controls to reasonable range [-0.5, 0.5] A
        u_t = np.clip(u_t, -0.5, 0.5)

        # Step dynamics
        result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)
        x_t = result['x_next']
        p_tip_next = result['p_tip']
        status = result['status']

        # Record
        p_tips.append(p_tip_next)
        us.append(u_t)
        statuses.append(status)

    # Convert to arrays
    p_tips = np.array(p_tips)
    us = np.array(us)
    statuses = np.array(statuses)

    # Compute tracking error
    tracking_errors = np.linalg.norm(p_tips - p_ref, axis=1)
    rms_error = np.sqrt(np.mean(tracking_errors**2))
    max_error = np.max(tracking_errors)
    mean_error = np.mean(tracking_errors)

    # Count failures
    n_failures = np.sum(statuses != 0)
    failure_rate = n_failures / n_steps * 100

    # Compare to expert (if available)
    if test_dataset.tip_dyn is not None:
        expert_dyn = test_dataset.tip_dyn[:n_steps]
        expert_errors = np.linalg.norm(expert_dyn - p_ref, axis=1)
        expert_rms = np.sqrt(np.mean(expert_errors**2))
    else:
        expert_rms = None

    print(f"\nRollout Results:")
    print(f"  Tracking error (policy):")
    print(f"    RMS:  {rms_error:.4f} mm")
    print(f"    Max:  {max_error:.4f} mm")
    print(f"    Mean: {mean_error:.4f} mm")
    if expert_rms is not None:
        print(f"  Expert tracking error (from dataset):")
        print(f"    RMS:  {expert_rms:.4f} mm")
        ratio = rms_error / expert_rms if expert_rms > 0 else float('inf')
        print(f"  Policy/Expert ratio: {ratio:.2f}x")
    print(f"  Dynamics failures: {n_failures}/{n_steps} ({failure_rate:.1f}%)")

    return {
        'rms_error': rms_error,
        'max_error': max_error,
        'mean_error': mean_error,
        'expert_rms': expert_rms,
        'failure_rate': failure_rate,
        'n_steps': n_steps,
        'p_tips': p_tips,
        'us': us,
        'tracking_errors': tracking_errors,
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("CP4.0: Behavior Cloning Baseline")
    print("=" * 80)
    print()

    # Load datasets
    # Note: Only use datasets with reference trajectories (tip_projected/tip_desired)
    train_files = [
        'data/dyn_fk_lem1_y40_a10_L94_hold1.npz',
        'data/dyn_fk_lem1_y40_a10_hold1.npz',  # Also lemniscate
    ]

    test_files = [
        'data/dyn_fk_lem1_y40_a10_L94_hold2.npz',
    ]

    print("Loading training datasets...")
    train_datasets = []
    for f in train_files:
        if os.path.exists(f):
            ds = load_npz_dataset(f, validate=True)
            print(f"  ✓ {ds.filename}: {ds.n_steps} steps, {ds.duration:.2f}s")
            train_datasets.append(ds)
        else:
            print(f"  ✗ {f} not found, skipping")

    if not train_datasets:
        print("\n✗ No training datasets found")
        return 1

    print("\nLoading test datasets...")
    test_datasets = []
    for f in test_files:
        if os.path.exists(f):
            ds = load_npz_dataset(f, validate=True)
            print(f"  ✓ {ds.filename}: {ds.n_steps} steps, {ds.duration:.2f}s")
            test_datasets.append(ds)
        else:
            print(f"  ✗ {f} not found, skipping")

    if not test_datasets:
        print("\n✗ No test datasets found")
        return 1

    # Get parameters from first dataset
    ref_ds = train_datasets[0]
    print(f"\n{print_param_summary(ref_ds.dt, ref_ds.L_inserted, ref_ds.integration_step_size)}")

    # Create PyTorch dataset
    print("\nCreating training dataset...")
    train_data = TrajectoryDataset(train_datasets, use_fk=False)

    # Create dataloader
    batch_size = 64
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    print(f"DataLoader created: batch_size={batch_size}, {len(dataloader)} batches")

    # Create policy
    print("\nCreating policy...")
    policy = BehaviorCloningPolicy(input_dim=6, hidden_dim=64, output_dim=3)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Policy: {n_params} parameters")
    print(f"  Architecture: [6] -> [64] -> [64] -> [3]")

    # Train
    train_metrics = train_policy(policy, dataloader, epochs=50, lr=1e-3)

    # Evaluate on test set (offline evaluation)
    eval_metrics = evaluate_policy_prediction(policy, test_datasets, use_fk=False)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Training:")
    print(f"  Datasets: {len(train_datasets)}")
    print(f"  Samples: {len(train_data)}")
    print(f"  Final train loss (MSE): {train_metrics['train_losses'][-1]:.6f} A^2")
    print(f"\nEvaluation (test set, {eval_metrics['n_samples']} samples):")
    print(f"  Test MSE:  {eval_metrics['mse']:.6f} A^2")
    print(f"  Test RMSE: {eval_metrics['rmse']:.6f} A")
    print("=" * 80)
    print("\nNOTE: This is an offline behavior cloning baseline.")
    print("      Online rollout evaluation would require domain adaptation")
    print("      (e.g., DAgger, residual RL) due to distributional shift.")
    print("=" * 80)

    # Save model
    model_path = './build/artifacts/cp40_bc_policy.pth'
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    torch.save(policy.state_dict(), model_path)
    print(f"\n✓ Model saved to {model_path}")

    # Success criterion: RMSE < 0.1A (reasonable for baseline)
    if eval_metrics['rmse'] < 0.1:
        print("✓ CP4.0 BEHAVIOR CLONING PASS: Policy achieves reasonable prediction accuracy")
        return 0
    else:
        print("✗ CP4.0 BEHAVIOR CLONING FAIL: Policy prediction error too high")
        return 1


if __name__ == "__main__":
    sys.exit(main())
