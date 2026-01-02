#!/usr/bin/env python3
"""
CP4.3: DAgger (Dataset Aggregation) Training

Addresses distributional shift by:
1. Rolling out current policy on training trajectories
2. Querying MPC expert when policy unsafe or tracking error large
3. Aggregating expert data from on-policy states
4. Retraining policy on growing dataset

Key Features:
- Uses CP4.1 BC architecture and dataset system
- MPC expert with iLQR (horizon=10, warm-start)
- Safety checks: status==0, rank==6, residual<1e-10, error<5mm
- Adaptive epochs based on dataset size
- Comprehensive metrics logging
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

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from data.npz_manifest import load_manifest, create_splits
from data.npz_dataset import NPZDataset
from control.ilqr import iLQRSolver
import crm_diff_py

# Import BC policy from CP4.1
from train_cp41_bc_multitraj import BehaviorCloningPolicy


# ============================================================================
# Configuration
# ============================================================================

class DAggerConfig:
    """DAgger training configuration."""

    # DAgger iterations
    n_iterations = 5

    # Safety thresholds
    max_tracking_error = 5.0  # mm - trigger expert if exceeded
    residual_threshold = 1e-10

    # MPC expert
    mpc_horizon = 10
    mpc_max_iters = 10  # Fast expert queries
    mpc_cost_tol = 1e-2

    # Training
    batch_size = 64
    base_epochs = 50
    lr = 1e-3
    val_patience = 10  # Early stopping

    # Rollout
    rollout_timeout_per_traj = 300  # seconds (5 min)

    # Artifacts
    output_dir = './build/artifacts'
    checkpoint_prefix = 'cp43_dagger'


# ============================================================================
# DAgger Dataset (PyTorch wrapper)
# ============================================================================

class DAggerDataset(Dataset):
    """
    PyTorch Dataset for DAgger aggregated data.

    Concatenates data from all iterations.
    """

    def __init__(self, aggregated_data, verbose=True):
        """
        Args:
            aggregated_data: Dict mapping iteration -> {p_tips, p_refs, u_experts}
        """
        p_tips_list = []
        p_refs_list = []
        us_list = []

        for iter_key in sorted(aggregated_data.keys()):
            iter_data = aggregated_data[iter_key]
            p_tips_list.append(iter_data['p_tips'])
            p_refs_list.append(iter_data['p_refs'])
            us_list.append(iter_data['u_experts'])

        self.p_tips = np.concatenate(p_tips_list, axis=0).astype(np.float32)
        self.p_refs = np.concatenate(p_refs_list, axis=0).astype(np.float32)
        self.us = np.concatenate(us_list, axis=0).astype(np.float32)

        if verbose:
            print(f"  DAggerDataset: {len(self)} samples from {len(aggregated_data)} iterations")

    def __len__(self):
        return len(self.p_tips)

    def __getitem__(self, idx):
        return self.p_tips[idx], self.p_refs[idx], self.us[idx]


# ============================================================================
# MPC Expert
# ============================================================================

class MPCExpert:
    """
    MPC expert for DAgger expert queries.

    Uses iLQR with warm-starting for fast queries.
    """

    def __init__(self, config, params_dict):
        self.config = config
        self.params_dict = params_dict
        self.U_warm = None  # Warm-start buffer

        self.n_queries = 0
        self.n_failures = 0
        self.query_times = []

    def query(self, x_t, p_ref_horizon, dt, L_inserted, verbose=False):
        """
        Query MPC expert for control action.

        Args:
            x_t: Current state [6]
            p_ref_horizon: Reference trajectory [horizon, 3]
            dt: Timestep
            L_inserted: Insertion length
            verbose: Print debug info

        Returns:
            u_expert: Expert control [3] or None if failed
            info: Dict with convergence info
        """
        t_start = time.time()
        self.n_queries += 1

        # Create iLQR solver
        solver = iLQRSolver(
            dt=dt,
            L_inserted=L_inserted,
            params_dict=self.params_dict,
            horizon=self.config.mpc_horizon,
            p_target=p_ref_horizon[0],  # Target first reference point
            Q=np.zeros((6, 6)),
            R=0.01 * np.eye(3),
            terminal_weight=1.0,
            max_iters=self.config.mpc_max_iters,
            tol=self.config.mpc_cost_tol
        )

        try:
            # Solve iLQR
            X, U, converged = solver.solve(
                x_t,
                U_init=self.U_warm,
                verbose=False
            )

            if not converged:
                if verbose:
                    print(f"    [MPC Expert] Warning: iLQR did not converge")

            # Extract first control
            u_expert = U[0]

            # Update warm-start buffer (shift solution)
            self.U_warm = np.vstack([U[1:], U[-1:]])

            query_time = time.time() - t_start
            self.query_times.append(query_time)

            return u_expert, {
                'converged': converged,
                'cost': solver.cost_history[-1] if solver.cost_history else None,
                'query_time': query_time
            }

        except Exception as e:
            if verbose:
                print(f"    [MPC Expert] FAILED: {e}")

            self.n_failures += 1
            self.U_warm = None  # Reset warm-start on failure

            return None, {
                'converged': False,
                'error': str(e)
            }

    def reset(self):
        """Reset warm-start buffer (call between trajectories)."""
        self.U_warm = None

    def get_stats(self):
        """Get expert query statistics."""
        return {
            'n_queries': self.n_queries,
            'n_failures': self.n_failures,
            'failure_rate': self.n_failures / max(self.n_queries, 1),
            'mean_query_time': float(np.mean(self.query_times)) if self.query_times else 0.0,
            'total_query_time': float(np.sum(self.query_times)) if self.query_times else 0.0
        }


# ============================================================================
# DAgger Rollout
# ============================================================================

def run_dagger_rollout(policy, dataset, expert, params_dict, config, verbose=True):
    """
    Run DAgger rollout on a single trajectory.

    Policy executes, expert intervenes when:
    - Dynamics solver fails (status != 0, rank < 6, residual > threshold)
    - Tracking error exceeds threshold

    Args:
        policy: Current BC policy
        dataset: NPZDataset with reference trajectory
        expert: MPCExpert instance
        params_dict: Physics parameters
        config: DAggerConfig
        verbose: Print progress

    Returns:
        dict: {
            'p_tips': Actual tip positions [N, 3],
            'p_refs': Reference positions [N, 3],
            'u_experts': Expert controls [N, 3],
            'statuses': Solver statuses [N],
            'expert_queries': Number of expert interventions,
            'n_steps': Number of steps completed,
            'failed': Bool, whether rollout failed catastrophically
        }
    """
    dt = dataset.dt
    L_inserted = dataset.L_inserted
    p_ref_full = dataset.tip_ref

    # Filter to valid reference samples
    valid_mask = ~dataset.hold_mask
    p_ref = p_ref_full[valid_mask]
    n_steps = len(p_ref)

    if verbose:
        print(f"    Rollout: {dataset.filename}")
        print(f"      Steps: {n_steps}, dt={dt:.4f}s, L={L_inserted:.1f}mm")

    # Initialize state
    x_t = np.zeros(6)

    # Get initial tip position
    result_init = crm_diff_py.dynamics_forward(
        x_t, np.zeros(3), 0.0, L_inserted, params_dict
    )
    p_tip_t = result_init['p_tip']

    # Storage
    p_tips = []
    p_refs_collected = []
    u_experts = []
    statuses = []
    expert_queries = 0

    expert.reset()  # Reset warm-start buffer

    t_start = time.time()

    for i in range(n_steps):
        if verbose and i % 50 == 0:
            print(f"      Step {i}/{n_steps}")

        # Check timeout
        if time.time() - t_start > config.rollout_timeout_per_traj:
            print(f"      WARNING: Rollout timeout at step {i}/{n_steps}")
            break

        # Predict control with policy
        u_policy = policy.predict(
            p_tip_t.astype(np.float32),
            p_ref[i].astype(np.float32)
        ).astype(np.float64)

        # Clip control
        u_policy = np.clip(u_policy, -0.5, 0.5)

        # Try policy action
        result = crm_diff_py.dynamics_forward(
            x_t, u_policy, dt, L_inserted, params_dict
        )

        # Safety checks
        status = result['status']
        lu_rank = result['lu_rank']
        rel_residual = result.get('rel_solve_residual', 0.0)

        dynamics_safe = (status == 0 and
                        lu_rank == 6 and
                        rel_residual < config.residual_threshold)

        if dynamics_safe:
            # Policy succeeded dynamically, check tracking error
            x_next_policy = result['x_next']
            p_tip_next_policy = result['p_tip']
            tracking_error = np.linalg.norm(p_tip_next_policy - p_ref[i])

            if tracking_error < config.max_tracking_error:
                # Accept policy action
                x_t = x_next_policy
                p_tip_t = p_tip_next_policy
                u_expert = u_policy

            else:
                # Tracking error too large → query expert
                expert_queries += 1

                # Get reference horizon
                horizon_end = min(i + config.mpc_horizon, n_steps)
                p_ref_horizon = p_ref[i:horizon_end]

                # Pad if needed
                if len(p_ref_horizon) < config.mpc_horizon:
                    p_ref_horizon = np.vstack([
                        p_ref_horizon,
                        np.tile(p_ref_horizon[-1],
                               (config.mpc_horizon - len(p_ref_horizon), 1))
                    ])

                # Query expert
                u_expert_candidate, expert_info = expert.query(
                    x_t, p_ref_horizon, dt, L_inserted, verbose=False
                )

                if u_expert_candidate is not None:
                    # Expert succeeded
                    u_expert = u_expert_candidate
                    u_expert = np.clip(u_expert, -0.5, 0.5)

                    # Execute expert action
                    result_expert = crm_diff_py.dynamics_forward(
                        x_t, u_expert, dt, L_inserted, params_dict
                    )

                    if result_expert['status'] == 0:
                        x_t = result_expert['x_next']
                        p_tip_t = result_expert['p_tip']
                    else:
                        # Expert action also failed → abort trajectory
                        print(f"      FAILED: Expert action failed at step {i}")
                        return {
                            'failed': True,
                            'n_steps': i,
                            'expert_queries': expert_queries
                        }
                else:
                    # Expert query failed → abort trajectory
                    print(f"      FAILED: Expert query failed at step {i}")
                    return {
                        'failed': True,
                        'n_steps': i,
                        'expert_queries': expert_queries
                    }

        else:
            # Dynamics unsafe → query expert immediately
            expert_queries += 1

            # Get reference horizon
            horizon_end = min(i + config.mpc_horizon, n_steps)
            p_ref_horizon = p_ref[i:horizon_end]

            # Pad if needed
            if len(p_ref_horizon) < config.mpc_horizon:
                p_ref_horizon = np.vstack([
                    p_ref_horizon,
                    np.tile(p_ref_horizon[-1],
                           (config.mpc_horizon - len(p_ref_horizon), 1))
                ])

            # Query expert
            u_expert_candidate, expert_info = expert.query(
                x_t, p_ref_horizon, dt, L_inserted, verbose=False
            )

            if u_expert_candidate is not None:
                u_expert = u_expert_candidate
                u_expert = np.clip(u_expert, -0.5, 0.5)

                # Execute expert action
                result_expert = crm_diff_py.dynamics_forward(
                    x_t, u_expert, dt, L_inserted, params_dict
                )

                if result_expert['status'] == 0:
                    x_t = result_expert['x_next']
                    p_tip_t = result_expert['p_tip']
                else:
                    # Expert action failed → abort
                    print(f"      FAILED: Expert action failed at step {i}")
                    return {
                        'failed': True,
                        'n_steps': i,
                        'expert_queries': expert_queries
                    }
            else:
                # Expert query failed → abort
                print(f"      FAILED: Expert query failed at step {i}")
                return {
                    'failed': True,
                    'n_steps': i,
                    'expert_queries': expert_queries
                }

        # Store data
        p_tips.append(p_tip_t)
        p_refs_collected.append(p_ref[i])
        u_experts.append(u_expert)
        statuses.append(status)

    # Success
    return {
        'p_tips': np.array(p_tips),
        'p_refs': np.array(p_refs_collected),
        'u_experts': np.array(u_experts),
        'statuses': np.array(statuses),
        'expert_queries': expert_queries,
        'n_steps': len(p_tips),
        'failed': False
    }


# ============================================================================
# Training
# ============================================================================

def compute_adaptive_epochs(n_samples, config):
    """Compute epochs based on dataset size."""
    if n_samples < 1000:
        return config.base_epochs
    elif n_samples < 5000:
        return 30
    else:
        return 20


def train_policy_with_early_stopping(policy, train_loader, val_loader,
                                     epochs, lr, patience, verbose=True):
    """
    Train policy with early stopping.

    Returns:
        dict: Training history
    """
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

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

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
        else:
            patience_counter += 1

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"    Epoch {epoch+1:3d}/{epochs}: "
                  f"train={avg_train_loss:.6f}, val={avg_val_loss:.6f}, "
                  f"patience={patience_counter}/{patience}")

        # Early stop
        if patience_counter >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch+1}")
            break

    # Restore best weights
    if best_state is not None:
        policy.load_state_dict(best_state)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'stopped_early': patience_counter >= patience
    }


# ============================================================================
# Main DAgger Loop
# ============================================================================

def main():
    print("=" * 80)
    print("CP4.3: DAgger (Dataset Aggregation) Training")
    print("=" * 80)
    print()

    config = DAggerConfig()

    # Load manifest
    print("Loading dataset manifest...")
    manifest = load_manifest(data_dir='data', verbose=False)

    if not manifest.datasets_with_ref:
        print("ERROR: No datasets with references available")
        return 1

    # Create splits
    print("Creating train/val/test splits...")
    train_ds, val_ds, test_ds = create_splits(
        manifest,
        val_ratio=0.15,
        test_ratio=0.25,
        stratify_by_type=True,
        verbose=True
    )

    # Load physics parameters
    param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
    cath_params = crm_diff_py.load_cath_params(param_file)
    cath_config = crm_diff_py.load_cath_config(config_file)

    # Create params_dict (will be updated per-dataset for integration_step_size)
    base_params_dict = {
        'CathParams': cath_params,
        'CathConfig': cath_config,
        'ContactMode': int(crm_diff_py.ContactModeType.FREE_TIP),
        'TipForce': [0.0, 0.0, 0.0],
        'deltau0_initialguess': [0.0, 0.0, 0.0],
        'FinalValueOnly': True,
    }

    # Initialize policy (try loading CP4.1 checkpoint)
    policy_path_cp41 = os.path.join(config.output_dir, 'cp41_bc_multitraj_policy.pth')
    policy = BehaviorCloningPolicy()

    if os.path.exists(policy_path_cp41):
        print(f"\nLoading CP4.1 policy from {policy_path_cp41}")
        policy.load_state_dict(torch.load(policy_path_cp41))
        print("  Loaded successfully")
    else:
        print(f"\nNo CP4.1 policy found, starting with random initialization")

    # Initialize aggregated data storage
    aggregated_data = {}

    # Initialize expert
    expert = MPCExpert(config, base_params_dict)

    # DAgger iterations
    all_iteration_metrics = []

    for dagger_iter in range(config.n_iterations):
        print(f"\n{'='*80}")
        print(f"DAgger Iteration {dagger_iter}/{config.n_iterations}")
        print(f"{'='*80}")

        iter_start_time = time.time()

        # Rollout current policy on training trajectories
        print(f"\n[Iteration {dagger_iter}] Rolling out policy...")

        iter_p_tips = []
        iter_p_refs = []
        iter_u_experts = []
        iter_sources = []

        total_expert_queries = 0
        total_steps = 0
        n_failed_trajectories = 0

        for ds_idx, dataset in enumerate(train_ds):
            # Create dataset-specific params_dict
            params_dict = base_params_dict.copy()
            params_dict['IntegrationStepSize'] = dataset.integration_step_size

            # Run rollout
            rollout_result = run_dagger_rollout(
                policy, dataset, expert, params_dict, config, verbose=True
            )

            if rollout_result['failed']:
                n_failed_trajectories += 1
                print(f"    Trajectory {dataset.filename} FAILED, skipping")
                continue

            # Collect data
            iter_p_tips.append(rollout_result['p_tips'])
            iter_p_refs.append(rollout_result['p_refs'])
            iter_u_experts.append(rollout_result['u_experts'])
            iter_sources.append(dataset.filename)

            total_expert_queries += rollout_result['expert_queries']
            total_steps += rollout_result['n_steps']

            print(f"    ✓ {dataset.filename}: "
                  f"{rollout_result['n_steps']} steps, "
                  f"{rollout_result['expert_queries']} expert queries")

        # Aggregate iteration data
        if not iter_p_tips:
            print(f"\nERROR: All trajectories failed in iteration {dagger_iter}")
            break

        aggregated_data[f'iteration_{dagger_iter}'] = {
            'p_tips': np.concatenate(iter_p_tips, axis=0),
            'p_refs': np.concatenate(iter_p_refs, axis=0),
            'u_experts': np.concatenate(iter_u_experts, axis=0),
            'sources': iter_sources
        }

        expert_query_rate = total_expert_queries / max(total_steps, 1)

        print(f"\n[Iteration {dagger_iter}] Rollout Summary:")
        print(f"  Total steps: {total_steps}")
        print(f"  Expert queries: {total_expert_queries} ({expert_query_rate:.1%})")
        print(f"  Failed trajectories: {n_failed_trajectories}/{len(train_ds)}")
        print(f"  New samples collected: {len(aggregated_data[f'iteration_{dagger_iter}']['p_tips'])}")

        # Create aggregated dataset
        print(f"\n[Iteration {dagger_iter}] Creating aggregated dataset...")
        full_train_data = DAggerDataset(aggregated_data, verbose=True)

        # Split aggregated data 80/20 for train/val
        n_total = len(full_train_data)
        n_train_agg = int(0.8 * n_total)

        indices = np.random.permutation(n_total)
        train_indices = indices[:n_train_agg]
        val_indices = indices[n_train_agg:]

        train_subset = torch.utils.data.Subset(full_train_data, train_indices)
        val_subset = torch.utils.data.Subset(full_train_data, val_indices)

        train_loader = DataLoader(train_subset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=config.batch_size, shuffle=False)

        print(f"  Train: {len(train_subset)} samples")
        print(f"  Val:   {len(val_subset)} samples")

        # Determine epochs
        epochs = compute_adaptive_epochs(len(train_subset), config)
        print(f"\n[Iteration {dagger_iter}] Training policy for {epochs} epochs...")

        # Train policy
        train_metrics = train_policy_with_early_stopping(
            policy, train_loader, val_loader,
            epochs=epochs, lr=config.lr, patience=config.val_patience,
            verbose=True
        )

        print(f"\n[Iteration {dagger_iter}] Training complete:")
        print(f"  Best val loss: {train_metrics['best_val_loss']:.6f}")
        print(f"  Early stopped: {train_metrics['stopped_early']}")

        # Save checkpoint
        os.makedirs(config.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            config.output_dir,
            f'{config.checkpoint_prefix}_iter{dagger_iter}_policy.pth'
        )
        torch.save(policy.state_dict(), checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")

        # Iteration metrics
        iter_elapsed = time.time() - iter_start_time
        expert_stats = expert.get_stats()

        iter_metrics = {
            'iteration': dagger_iter,
            'rollout': {
                'total_steps': total_steps,
                'expert_queries': total_expert_queries,
                'expert_query_rate': expert_query_rate,
                'failed_trajectories': n_failed_trajectories,
                'successful_trajectories': len(iter_sources)
            },
            'expert_stats': expert_stats,
            'training': {
                'n_samples': len(train_subset),
                'epochs_run': len(train_metrics['train_losses']),
                'best_val_loss': float(train_metrics['best_val_loss']),
                'early_stopped': train_metrics['stopped_early']
            },
            'timing': {
                'iteration_time': iter_elapsed
            }
        }

        all_iteration_metrics.append(iter_metrics)

        print(f"\n[Iteration {dagger_iter}] Iteration time: {iter_elapsed:.1f}s")

    # Final evaluation and save
    print(f"\n{'='*80}")
    print("DAgger Training Complete")
    print(f"{'='*80}")

    # Save final model
    final_model_path = os.path.join(config.output_dir, f'{config.checkpoint_prefix}_final_policy.pth')
    torch.save(policy.state_dict(), final_model_path)
    print(f"\nFinal model saved: {final_model_path}")

    # Save metrics
    metrics_path = os.path.join(config.output_dir, f'{config.checkpoint_prefix}_metrics.json')
    metrics_output = {
        'config': {
            'n_iterations': config.n_iterations,
            'max_tracking_error': config.max_tracking_error,
            'mpc_horizon': config.mpc_horizon,
            'base_epochs': config.base_epochs,
            'lr': config.lr
        },
        'iterations': all_iteration_metrics
    }

    with open(metrics_path, 'w') as f:
        json.dump(metrics_output, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    # Summary
    print(f"\nSummary:")
    print(f"  Iterations completed: {len(all_iteration_metrics)}")
    print(f"  Final dataset size: {len(full_train_data)} samples")
    print(f"  Total expert queries: {sum(m['rollout']['expert_queries'] for m in all_iteration_metrics)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
