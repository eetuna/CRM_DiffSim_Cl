#!/usr/bin/env python3
"""
CP4.1: Multi-Trajectory Behavior Cloning

Enhanced behavior cloning that trains on ALL available datasets with references,
stratified by trajectory type (circle + lemniscate).

Key improvements over CP4.0:
- Uses manifest system to discover all datasets
- Creates proper train/val/test splits
- Reports per-trajectory RMSE
- Saves metrics to JSON file
- Handles datasets without references explicitly

Architecture: Same as CP4.0
    Input: [p_tip (3), p_ref (3)] → 6D
    Hidden: [64, 64]
    Output: u (3)
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add python directory and build directory to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from data.npz_manifest import load_manifest, create_splits, print_manifest_summary
from data.npz_dataset import NPZDataset


# ============================================================================
# Neural Network Policy (same as CP4.0)
# ============================================================================

class BehaviorCloningPolicy(nn.Module):
    """
    Simple feedforward policy for behavior cloning.

    Input: [p_tip (3), p_ref (3)] → 6D
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

class MultiTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for behavior cloning from multiple NPZ trajectories.

    Handles hold masks and filters invalid data.
    Returns (p_tip, p_ref, u) tuples.
    """

    def __init__(self, npz_datasets, use_fk=False, verbose=True):
        """
        Args:
            npz_datasets: List of NPZDataset objects
            use_fk: If True, use tip_fk; otherwise use tip_dyn
            verbose: If True, print loading progress
        """
        self.use_fk = use_fk

        # Collect all (p_tip, p_ref, u) tuples
        p_tips = []
        p_refs = []
        us = []
        dataset_labels = []  # Track which dataset each sample came from

        for ds_idx, dataset in enumerate(npz_datasets):
            # Check if dataset has reference
            if not dataset.has_reference:
                if verbose:
                    print(f"    Skipping {dataset.filename}: no reference (no_ref)")
                continue

            # Get tip positions
            if use_fk:
                if dataset.tip_fk is None:
                    if verbose:
                        print(f"    Skipping {dataset.filename}: missing tip_fk")
                    continue
                p_tip = dataset.tip_fk
            else:
                if dataset.tip_dyn is None:
                    if verbose:
                        print(f"    Skipping {dataset.filename}: missing tip_dyn")
                    continue
                p_tip = dataset.tip_dyn

            # Get reference
            p_ref = dataset.tip_ref

            # Get controls
            u = dataset.currents

            # Filter using hold_mask (exclude invalid/NaN samples)
            valid_mask = ~dataset.hold_mask
            n_valid = np.sum(valid_mask)
            n_total = len(p_ref)

            if n_valid == 0:
                if verbose:
                    print(f"    Skipping {dataset.filename}: no valid samples")
                continue

            if verbose:
                print(f"    ✓ {dataset.filename}: {n_valid}/{n_total} valid samples")

            p_tip = p_tip[valid_mask]
            p_ref = p_ref[valid_mask]
            u = u[valid_mask]

            # Append
            p_tips.append(p_tip)
            p_refs.append(p_ref)
            us.append(u)
            dataset_labels.extend([ds_idx] * len(p_tip))

        if not p_tips:
            raise ValueError("No valid data available from any dataset")

        # Concatenate all datasets
        self.p_tips = np.concatenate(p_tips, axis=0).astype(np.float32)
        self.p_refs = np.concatenate(p_refs, axis=0).astype(np.float32)
        self.us = np.concatenate(us, axis=0).astype(np.float32)
        self.labels = np.array(dataset_labels)

        if verbose:
            print(f"\n  Total samples: {len(self)}")
            print(f"    p_tip range: [{self.p_tips.min():.2f}, {self.p_tips.max():.2f}] mm")
            print(f"    p_ref range: [{self.p_refs.min():.2f}, {self.p_refs.max():.2f}] mm")
            print(f"    u range: [{self.us.min():.4f}, {self.us.max():.4f}] A")

    def __len__(self):
        return len(self.p_tips)

    def __getitem__(self, idx):
        return self.p_tips[idx], self.p_refs[idx], self.us[idx]


# ============================================================================
# Training
# ============================================================================

def train_policy(policy, train_loader, val_loader, epochs=50, lr=1e-3, verbose=True):
    """
    Train behavior cloning policy.

    Args:
        policy: BehaviorCloningPolicy
        train_loader: PyTorch DataLoader for training
        val_loader: PyTorch DataLoader for validation
        epochs: Number of training epochs
        lr: Learning rate
        verbose: If True, print progress

    Returns:
        dict: Training metrics
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if verbose:
        print(f"\nTraining for {epochs} epochs...")
        print(f"  Optimizer: Adam(lr={lr})")
        print(f"  Loss: MSE")

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Train
        policy.train()
        epoch_train_loss = 0.0
        n_train_batches = 0

        for p_tip, p_ref, u_target in train_loader:
            u_pred = policy(p_tip, p_ref)
            loss = criterion(u_pred, u_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            n_train_batches += 1

        avg_train_loss = epoch_train_loss / n_train_batches
        train_losses.append(avg_train_loss)

        # Validation
        policy.eval()
        epoch_val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for p_tip, p_ref, u_target in val_loader:
                u_pred = policy(p_tip, p_ref)
                loss = criterion(u_pred, u_target)
                epoch_val_loss += loss.item()
                n_val_batches += 1

        avg_val_loss = epoch_val_loss / n_val_batches if n_val_batches > 0 else 0.0
        val_losses.append(avg_val_loss)

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"  Epoch {epoch+1:3d}/{epochs}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")

    if verbose:
        print(f"Training complete. Final: train={train_losses[-1]:.6f}, val={val_losses[-1]:.6f}")

    return {
        'train_losses': train_losses,
        'val_losses': val_losses
    }


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_per_trajectory(policy, test_datasets, use_fk=False, verbose=True):
    """
    Evaluate policy on test set with per-trajectory breakdown.

    Args:
        policy: Trained BehaviorCloningPolicy
        test_datasets: List of NPZDataset for evaluation
        use_fk: If True, use tip_fk; otherwise use tip_dyn
        verbose: If True, print results

    Returns:
        dict: Evaluation metrics (overall + per-trajectory)
    """
    if verbose:
        print(f"\nEvaluating policy on test set...")

    all_errors_sq = []
    per_trajectory_metrics = []

    for dataset in test_datasets:
        if not dataset.has_reference:
            continue

        # Get data
        if use_fk:
            if dataset.tip_fk is None:
                continue
            p_tip = dataset.tip_fk
        else:
            if dataset.tip_dyn is None:
                continue
            p_tip = dataset.tip_dyn

        p_ref = dataset.tip_ref
        u_true = dataset.currents

        # Filter with hold_mask
        valid_mask = ~dataset.hold_mask
        p_tip = p_tip[valid_mask]
        p_ref = p_ref[valid_mask]
        u_true = u_true[valid_mask]

        if len(p_tip) == 0:
            continue

        # Predict
        policy.eval()
        with torch.no_grad():
            p_tips_t = torch.from_numpy(p_tip.astype(np.float32))
            p_refs_t = torch.from_numpy(p_ref.astype(np.float32))
            us_pred_t = policy(p_tips_t, p_refs_t)
            us_pred = us_pred_t.numpy()

        # Compute errors
        errors_sq = (us_pred - u_true) ** 2
        mse = np.mean(errors_sq)
        rmse = np.sqrt(mse)

        all_errors_sq.append(errors_sq)

        traj_metrics = {
            'filename': dataset.filename,
            'n_samples': len(p_tip),
            'mse': float(mse),
            'rmse': float(rmse)
        }
        per_trajectory_metrics.append(traj_metrics)

        if verbose:
            print(f"  • {dataset.filename}: RMSE={rmse:.6f} A ({len(p_tip)} samples)")

    # Overall metrics
    all_errors_sq_concat = np.concatenate(all_errors_sq, axis=0)
    overall_mse = np.mean(all_errors_sq_concat)
    overall_rmse = np.sqrt(overall_mse)

    if verbose:
        print(f"\nOverall Test Performance:")
        print(f"  RMSE: {overall_rmse:.6f} A")
        print(f"  MSE:  {overall_mse:.6f} A²")

    return {
        'overall_mse': float(overall_mse),
        'overall_rmse': float(overall_rmse),
        'per_trajectory': per_trajectory_metrics,
        'n_trajectories': len(per_trajectory_metrics),
        'total_samples': sum(m['n_samples'] for m in per_trajectory_metrics)
    }


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("CP4.1: Multi-Trajectory Behavior Cloning")
    print("=" * 80)
    print()

    # Load manifest
    print("Loading dataset manifest...")
    manifest = load_manifest(data_dir='data', verbose=False)
    print_manifest_summary(manifest)

    # Check if we have datasets with references
    if not manifest.datasets_with_ref:
        print("\n✗ ERROR: No datasets with references available for training")
        print("  Circle datasets lack tip_projected/tip_desired (marked as 'no_ref')")
        print("  Only lemniscate datasets have references")
        return 1

    # Create splits
    print(f"\nCreating train/val/test splits...")
    train_ds, val_ds, test_ds = create_splits(
        manifest,
        val_ratio=0.15,
        test_ratio=0.25,  # Use 1 dataset for test
        stratify_by_type=True,
        verbose=True
    )

    # Create PyTorch datasets
    print(f"\nCreating PyTorch datasets...")
    print(f"  Train:")
    train_data = MultiTrajectoryDataset(train_ds, use_fk=False)

    print(f"\n  Validation:")
    val_data = MultiTrajectoryDataset(val_ds, use_fk=False)

    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_loader)} batches")
    print(f"  Val:   {len(val_loader)} batches")

    # Create policy
    print(f"\nCreating policy...")
    policy = BehaviorCloningPolicy(input_dim=6, hidden_dim=64, output_dim=3)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"  Architecture: [6] -> [64] -> [64] -> [3]")
    print(f"  Parameters: {n_params}")

    # Train
    train_metrics = train_policy(
        policy, train_loader, val_loader,
        epochs=50, lr=1e-3, verbose=True
    )

    # Evaluate on test set
    test_metrics = evaluate_per_trajectory(policy, test_ds, use_fk=False, verbose=True)

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING SUMMARY")
    print("=" * 80)
    print(f"Datasets:")
    print(f"  Train: {len(train_ds)} datasets, {len(train_data)} samples")
    print(f"  Val:   {len(val_ds)} datasets, {len(val_data)} samples")
    print(f"  Test:  {len(test_ds)} datasets, {test_metrics['total_samples']} samples")
    print(f"\nTraining:")
    print(f"  Final train loss: {train_metrics['train_losses'][-1]:.6f} A²")
    print(f"  Final val loss:   {train_metrics['val_losses'][-1]:.6f} A²")
    print(f"\nTest Performance:")
    print(f"  Overall RMSE: {test_metrics['overall_rmse']:.6f} A")
    print(f"  Per-trajectory:")
    for traj_metric in test_metrics['per_trajectory']:
        print(f"    • {traj_metric['filename']}: RMSE={traj_metric['rmse']:.6f} A")
    print("=" * 80)

    # Save model and metrics
    output_dir = './build/artifacts'
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, 'cp41_bc_multitraj_policy.pth')
    torch.save(policy.state_dict(), model_path)
    print(f"\n✓ Model saved to {model_path}")

    metrics_path = os.path.join(output_dir, 'cp41_bc_multitraj_metrics.json')
    metrics = {
        'train': {
            'n_datasets': len(train_ds),
            'n_samples': len(train_data),
            'final_loss': train_metrics['train_losses'][-1],
            'losses': train_metrics['train_losses']
        },
        'val': {
            'n_datasets': len(val_ds),
            'n_samples': len(val_data),
            'final_loss': train_metrics['val_losses'][-1],
            'losses': train_metrics['val_losses']
        },
        'test': test_metrics,
        'config': {
            'epochs': 50,
            'lr': 1e-3,
            'batch_size': batch_size,
            'architecture': '[6] -> [64] -> [64] -> [3]'
        }
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")

    # Success criterion: RMSE < 0.1A
    if test_metrics['overall_rmse'] < 0.1:
        print("\n✓ CP4.1 MULTI-TRAJ BC PASS: RMSE < 0.1A")
        return 0
    else:
        print(f"\n✗ CP4.1 MULTI-TRAJ BC FAIL: RMSE {test_metrics['overall_rmse']:.6f} >= 0.1A")
        return 1


if __name__ == "__main__":
    sys.exit(main())
