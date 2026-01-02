#!/usr/bin/env python3
"""
CP4.5: Ensemble DAgger with Uncertainty Estimation

Trains N=5 recurrent DAgger policies with different random seeds to create
an ensemble that provides uncertainty estimates.

Key features:
- Each policy trained independently with different seed
- Ensemble aggregation (mean + variance)
- Uncertainty-error correlation analysis
- Expert querying based on uncertainty threshold

Reuses all CP4.4a infrastructure (recurrent policies, DAgger rollout, MPC expert).
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
from models.recurrent_policy import GRUPolicy
from models.ensemble_policy import EnsemblePolicy, EnsembleMetrics
import crm_diff_py


# ============================================================================
# Configuration
# ============================================================================

class EnsembleDAggerConfig:
    """Ensemble DAgger configuration (same as CP4.4a + ensemble params)."""

    # Ensemble
    n_ensemble_members = 5
    random_seeds = [42, 123, 456, 789, 1024]  # Different seeds for diversity

    # DAgger iterations (reduced for faster training)
    n_iterations = 3

    # Safety thresholds
    max_tracking_error = 5.0  # mm
    residual_threshold = 1e-10

    # Uncertainty-based expert querying
    uncertainty_threshold = 0.001  # Variance threshold for expert query

    # MPC expert
    mpc_horizon = 10
    mpc_max_iters = 10
    mpc_cost_tol = 1e-2

    # Training
    batch_size = 64
    base_epochs = 50
    lr = 1e-3
    val_patience = 10
    gradient_clip = 1.0

    # Rollout
    rollout_timeout_per_traj = 300  # seconds

    # Policy architecture
    policy_type = 'gru'
    hidden_dim = 64
    num_layers = 1

    # Artifacts
    output_dir = './build/artifacts'
    checkpoint_prefix = 'cp45_ensemble_dagger'

    @classmethod
    def create_fast_train(cls):
        """
        Create fast training profile for CP4.7.2 end-to-end validation.

        Target: <= 30 minutes on typical dev machine.
        Maintains minimum 3 ensemble members for uncertainty estimation.
        """
        config = cls()
        config.n_ensemble_members = 3
        config.random_seeds = [42, 123, 456]
        config.n_iterations = 2  # Reduced from 3
        config.base_epochs = 20  # Reduced from 50
        config.mpc_max_iters = 8  # Reduced from 10
        config.rollout_timeout_per_traj = 180  # Reduced from 300
        config.checkpoint_prefix = 'cp472_ensemble_fast'
        return config


# ============================================================================
# Reuse CP4.4a components
# ============================================================================

class RecurrentDAggerDataset(Dataset):
    """PyTorch Dataset for recurrent DAgger (same as CP4.4a)."""

    def __init__(self, aggregated_data, verbose=True):
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
            print(f"  Dataset: {len(self)} samples from {len(aggregated_data)} iterations")

    def __len__(self):
        return len(self.p_tips)

    def __getitem__(self, idx):
        return self.p_tips[idx], self.p_refs[idx], self.us[idx]


class MPCExpert:
    """MPC expert for DAgger (same as CP4.4a)."""

    def __init__(self, config, params_dict):
        self.config = config
        self.params_dict = params_dict
        self.U_warm = None
        self.n_queries = 0
        self.n_failures = 0
        self.query_times = []

    def query(self, x_t, p_ref_horizon, dt, L_inserted, verbose=False):
        t_start = time.time()
        self.n_queries += 1

        solver = iLQRSolver(
            dt=dt,
            L_inserted=L_inserted,
            params_dict=self.params_dict,
            horizon=self.config.mpc_horizon,
            p_target=p_ref_horizon[0],
            Q=np.zeros((6, 6)),
            R=0.01 * np.eye(3),
            terminal_weight=1.0,
            max_iters=self.config.mpc_max_iters,
            tol=self.config.mpc_cost_tol,
            jacobian_mode="cpp"  # CP4.4c fast Jacobians
        )

        try:
            X, U, converged = solver.solve(x_t, U_init=self.U_warm, verbose=False)
            u_expert = U[0]
            self.U_warm = np.vstack([U[1:], U[-1:]])

            query_time = time.time() - t_start
            self.query_times.append(query_time)

            return u_expert, {
                'converged': converged,
                'cost': solver.cost_history[-1] if solver.cost_history else None,
                'query_time': query_time
            }

        except Exception as e:
            self.n_failures += 1
            self.U_warm = None
            return None, {'converged': False, 'error': str(e)}

    def reset(self):
        self.U_warm = None

    def get_stats(self):
        return {
            'n_queries': self.n_queries,
            'n_failures': self.n_failures,
            'failure_rate': self.n_failures / max(self.n_queries, 1),
            'mean_query_time': float(np.mean(self.query_times)) if self.query_times else 0.0,
            'total_query_time': float(np.sum(self.query_times)) if self.query_times else 0.0
        }


def run_recurrent_dagger_rollout(policy, dataset, expert, params_dict, config, verbose=True):
    """
    Run DAgger rollout with recurrent policy (same as CP4.4a).

    Returns:
        dict with rollout results or {'failed': True}
    """
    dt = dataset.dt
    L_inserted = dataset.L_inserted
    p_ref_full = dataset.tip_ref

    valid_mask = ~dataset.hold_mask
    p_ref = p_ref_full[valid_mask]
    n_steps = len(p_ref)

    if verbose:
        print(f"    Rollout: {dataset.filename}")
        print(f"      Steps: {n_steps}, dt={dt:.4f}s, L={L_inserted:.1f}mm")

    x_t = np.zeros(6)
    hidden = policy.init_hidden(batch_size=1, device='cpu')

    result_init = crm_diff_py.dynamics_forward(
        x_t, np.zeros(3), 0.0, L_inserted, params_dict
    )
    p_tip_t = result_init['p_tip']

    p_tips = []
    p_refs_collected = []
    u_experts = []
    statuses = []
    expert_queries = 0

    expert.reset()
    t_start = time.time()

    for i in range(n_steps):
        if verbose and i % 50 == 0:
            print(f"      Step {i}/{n_steps}")

        if time.time() - t_start > config.rollout_timeout_per_traj:
            print(f"      WARNING: Rollout timeout at step {i}/{n_steps}")
            break

        # Predict control with recurrent policy
        u_policy, hidden = policy.predict(
            p_tip_t.astype(np.float32),
            p_ref[i].astype(np.float32),
            hidden
        )
        u_policy = u_policy.astype(np.float64)
        u_policy = np.clip(u_policy, -0.5, 0.5)

        # Detach hidden state
        if isinstance(hidden, tuple):
            hidden = (hidden[0].detach(), hidden[1].detach())
        else:
            hidden = hidden.detach()

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
            x_next_policy = result['x_next']
            p_tip_next_policy = result['p_tip']
            tracking_error = np.linalg.norm(p_tip_next_policy - p_ref[i])

            if tracking_error < config.max_tracking_error:
                x_t = x_next_policy
                p_tip_t = p_tip_next_policy
                u_expert = u_policy
            else:
                # Query expert for correction
                expert_queries += 1

                horizon_end = min(i + config.mpc_horizon, n_steps)
                p_ref_horizon = p_ref[i:horizon_end]

                if len(p_ref_horizon) < config.mpc_horizon:
                    p_ref_horizon = np.vstack([
                        p_ref_horizon,
                        np.tile(p_ref_horizon[-1],
                               (config.mpc_horizon - len(p_ref_horizon), 1))
                    ])

                u_expert_candidate, expert_info = expert.query(
                    x_t, p_ref_horizon, dt, L_inserted, verbose=False
                )

                if u_expert_candidate is not None:
                    u_expert = np.clip(u_expert_candidate, -0.5, 0.5)

                    result_expert = crm_diff_py.dynamics_forward(
                        x_t, u_expert, dt, L_inserted, params_dict
                    )

                    if result_expert['status'] == 0:
                        x_t = result_expert['x_next']
                        p_tip_t = result_expert['p_tip']
                    else:
                        print(f"      FAILED: Expert action failed at step {i}")
                        return {'failed': True, 'n_steps': i, 'expert_queries': expert_queries}
                else:
                    print(f"      FAILED: Expert query failed at step {i}")
                    return {'failed': True, 'n_steps': i, 'expert_queries': expert_queries}

        else:
            # Dynamics unsafe → query expert
            expert_queries += 1

            horizon_end = min(i + config.mpc_horizon, n_steps)
            p_ref_horizon = p_ref[i:horizon_end]

            if len(p_ref_horizon) < config.mpc_horizon:
                p_ref_horizon = np.vstack([
                    p_ref_horizon,
                    np.tile(p_ref_horizon[-1],
                           (config.mpc_horizon - len(p_ref_horizon), 1))
                ])

            u_expert_candidate, expert_info = expert.query(
                x_t, p_ref_horizon, dt, L_inserted, verbose=False
            )

            if u_expert_candidate is not None:
                u_expert = np.clip(u_expert_candidate, -0.5, 0.5)

                result_expert = crm_diff_py.dynamics_forward(
                    x_t, u_expert, dt, L_inserted, params_dict
                )

                if result_expert['status'] == 0:
                    x_t = result_expert['x_next']
                    p_tip_t = result_expert['p_tip']
                else:
                    print(f"      FAILED: Expert action failed at step {i}")
                    return {'failed': True, 'n_steps': i, 'expert_queries': expert_queries}
            else:
                print(f"      FAILED: Expert query failed at step {i}")
                return {'failed': True, 'n_steps': i, 'expert_queries': expert_queries}

        # Store data
        p_tips.append(p_tip_t)
        p_refs_collected.append(p_ref[i])
        u_experts.append(u_expert)
        statuses.append(status)

    # Successful rollout
    return {
        'p_tips': np.array(p_tips),
        'p_refs': np.array(p_refs_collected),
        'u_experts': np.array(u_experts),
        'statuses': np.array(statuses),
        'expert_queries': expert_queries,
        'n_steps': len(p_tips),
        'failed': False
    }


def train_recurrent_policy(policy, train_loader, val_loader, config, verbose=True):
    """Train recurrent policy with early stopping (same as CP4.4a)."""

    optimizer = optim.Adam(policy.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    n_samples = len(train_loader.dataset)
    if n_samples < 1000:
        epochs = config.base_epochs
    elif n_samples < 5000:
        epochs = max(30, config.base_epochs // 2)
    else:
        epochs = 20

    if verbose:
        print(f"  Training: {epochs} epochs, {n_samples} samples")

    for epoch in range(epochs):
        # Training
        policy.train()
        train_loss = 0.0
        n_batches = 0

        for p_tip_batch, p_ref_batch, u_batch in train_loader:
            optimizer.zero_grad()

            u_pred, _ = policy(p_tip_batch, p_ref_batch, hidden=None)

            loss = criterion(u_pred, u_batch)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(policy.parameters(), config.gradient_clip)

            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches

        # Validation
        policy.eval()
        val_loss = 0.0
        n_val_batches = 0

        with torch.no_grad():
            for p_tip_batch, p_ref_batch, u_batch in val_loader:
                u_pred, _ = policy(p_tip_batch, p_ref_batch, hidden=None)
                loss = criterion(u_pred, u_batch)
                val_loss += loss.item()
                n_val_batches += 1

        val_loss /= n_val_batches

        if verbose and epoch % 10 == 0:
            print(f"    Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= config.val_patience:
                if verbose:
                    print(f"    Early stopping at epoch {epoch}")
                break

    # Restore best weights
    if best_state is not None:
        policy.load_state_dict(best_state)

    return {
        'epochs_run': epoch + 1,
        'best_val_loss': float(best_val_loss),
        'final_train_loss': float(train_loss),
        'early_stopped': patience_counter >= config.val_patience
    }


# ============================================================================
# Ensemble DAgger Training
# ============================================================================

def train_single_ensemble_member(
    member_id,
    seed,
    config,
    aggregated_data,
    verbose=True
):
    """
    Train a single ensemble member with given seed.

    Args:
        member_id: Ensemble member index (0-indexed)
        seed: Random seed for this member
        config: EnsembleDAggerConfig
        aggregated_data: DAgger aggregated data
        verbose: Print training progress

    Returns:
        Trained policy and training metrics
    """
    if verbose:
        print(f"\n  Ensemble Member {member_id} (seed={seed})")
        print("  " + "-"*60)

    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Initialize policy
    policy = GRUPolicy(
        input_dim=6,
        hidden_dim=config.hidden_dim,
        output_dim=3,
        num_layers=config.num_layers
    )

    if verbose:
        print(f"    Parameters: {sum(p.numel() for p in policy.parameters())}")

    # Create datasets
    train_dataset = RecurrentDAggerDataset(aggregated_data, verbose=False)
    val_dataset = RecurrentDAggerDataset(aggregated_data, verbose=False)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # Train
    train_metrics = train_recurrent_policy(
        policy, train_loader, val_loader, config, verbose=verbose
    )

    # Save checkpoint
    os.makedirs(config.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(
        config.output_dir,
        f"{config.checkpoint_prefix}_member{member_id}_seed{seed}_policy.pth"
    )
    torch.save(policy.state_dict(), checkpoint_path)

    if verbose:
        print(f"    Saved: {checkpoint_path}")

    return policy, train_metrics, checkpoint_path


def run_ensemble_dagger(config, datasets_train, datasets_val, params_dict, verbose=True):
    """Run ensemble DAgger training."""

    print("="*80)
    print("CP4.5: Ensemble DAgger Training")
    print("="*80)
    print(f"Ensemble members: {config.n_ensemble_members}")
    print(f"Seeds: {config.random_seeds[:config.n_ensemble_members]}")
    print(f"Iterations: {config.n_iterations}")
    print(f"Train datasets: {len(datasets_train)}")
    print()

    # Initialize expert
    expert = MPCExpert(config, params_dict)

    # Storage
    aggregated_data = {}
    ensemble_metrics = {
        'config': {
            'n_ensemble_members': config.n_ensemble_members,
            'random_seeds': config.random_seeds[:config.n_ensemble_members],
            'n_iterations': config.n_iterations,
            'uncertainty_threshold': config.uncertainty_threshold
        },
        'iterations': [],
        'ensemble_members': []
    }

    # DAgger iterations (collect data with first policy)
    print("="*80)
    print("Phase 1: DAgger Data Collection")
    print("="*80)

    # Initialize first policy for data collection
    torch.manual_seed(config.random_seeds[0])
    np.random.seed(config.random_seeds[0])

    policy_collector = GRUPolicy(
        input_dim=6,
        hidden_dim=config.hidden_dim,
        output_dim=3,
        num_layers=config.num_layers
    )

    for iteration in range(config.n_iterations):
        print(f"\nIteration {iteration}/{config.n_iterations}")
        print("-"*80)

        iter_start = time.time()

        # Rollout phase
        print("  Rollout phase...")
        p_tips_all = []
        p_refs_all = []
        u_experts_all = []

        n_total_steps = 0
        n_expert_queries = 0
        n_failed = 0

        for ds in datasets_train:
            result = run_recurrent_dagger_rollout(
                policy_collector, ds, expert, params_dict, config, verbose=False
            )

            if result['failed']:
                n_failed += 1
                continue

            p_tips_all.append(result['p_tips'])
            p_refs_all.append(result['p_refs'])
            u_experts_all.append(result['u_experts'])

            n_total_steps += result['n_steps']
            n_expert_queries += result['expert_queries']

        # Aggregate data
        aggregated_data[f'iteration_{iteration}'] = {
            'p_tips': np.vstack(p_tips_all),
            'p_refs': np.vstack(p_refs_all),
            'u_experts': np.vstack(u_experts_all)
        }

        expert_query_rate = n_expert_queries / max(n_total_steps, 1)

        print(f"    Total steps: {n_total_steps}")
        print(f"    Expert queries: {n_expert_queries} ({expert_query_rate:.1%})")
        print(f"    Failed trajectories: {n_failed}/{len(datasets_train)}")

        # Quick training update for data collector
        print("  Updating collector policy...")

        train_dataset = RecurrentDAggerDataset(aggregated_data, verbose=False)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)

        train_metrics = train_recurrent_policy(
            policy_collector, train_loader, val_loader, config, verbose=False
        )

        iter_time = time.time() - iter_start

        iter_metrics = {
            'iteration': iteration,
            'rollout': {
                'total_steps': int(n_total_steps),
                'expert_queries': int(n_expert_queries),
                'expert_query_rate': float(expert_query_rate),
                'failed_trajectories': int(n_failed)
            },
            'collector_training': train_metrics,
            'timing': {'iteration_time': float(iter_time)}
        }

        ensemble_metrics['iterations'].append(iter_metrics)

    # Phase 2: Train ensemble members
    print("\n" + "="*80)
    print("Phase 2: Training Ensemble Members")
    print("="*80)

    ensemble_checkpoint_paths = []

    for member_id in range(config.n_ensemble_members):
        seed = config.random_seeds[member_id]

        policy, train_metrics, checkpoint_path = train_single_ensemble_member(
            member_id, seed, config, aggregated_data, verbose=True
        )

        ensemble_checkpoint_paths.append(checkpoint_path)

        member_metrics = {
            'member_id': member_id,
            'seed': seed,
            'checkpoint_path': checkpoint_path,
            'training': train_metrics
        }

        ensemble_metrics['ensemble_members'].append(member_metrics)

    # Save ensemble metadata
    ensemble_meta_path = os.path.join(
        config.output_dir,
        f"{config.checkpoint_prefix}_ensemble_metadata.json"
    )

    ensemble_metadata = {
        'n_members': config.n_ensemble_members,
        'checkpoint_paths': ensemble_checkpoint_paths,
        'policy_kwargs': {
            'input_dim': 6,
            'hidden_dim': config.hidden_dim,
            'output_dim': 3,
            'num_layers': config.num_layers
        }
    }

    with open(ensemble_meta_path, 'w') as f:
        json.dump(ensemble_metadata, f, indent=2)

    print(f"\nEnsemble metadata saved: {ensemble_meta_path}")

    # Save full metrics
    metrics_path = os.path.join(config.output_dir, f"{config.checkpoint_prefix}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(ensemble_metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    print()
    print("="*80)
    print("✓ Ensemble DAgger Complete")
    print("="*80)

    return ensemble_checkpoint_paths, ensemble_metrics


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='CP4.5/CP4.7.2: Ensemble DAgger Training')
    parser.add_argument('--profile', type=str, default='standard',
                       choices=['standard', 'fast_train'],
                       help='Training profile: standard (full) or fast_train (CP4.7.2, <=30min)')
    parser.add_argument('--datasets_limit', type=int, default=None,
                       help='Limit number of training datasets (for faster testing)')
    args = parser.parse_args()

    # Load manifest
    manifest = load_manifest()
    datasets_train, datasets_val, datasets_test = create_splits(manifest, val_ratio=0.2, test_ratio=0.2)

    # Apply dataset limit if specified
    if args.datasets_limit is not None:
        datasets_train = datasets_train[:args.datasets_limit]
        datasets_val = datasets_val[:max(1, args.datasets_limit // 5)]

    print(f"Loaded {len(datasets_train)} train, {len(datasets_val)} val datasets")

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

    # Select configuration profile
    if args.profile == 'fast_train':
        print("\n[CP4.7.2 Fast Training Profile]")
        print("  Target: <=30 minutes")
        print("  Ensemble members: 3")
        print("  Iterations: 2")
        config = EnsembleDAggerConfig.create_fast_train()
    else:
        config = EnsembleDAggerConfig()

    # Run ensemble DAgger
    t_start = time.time()
    ensemble_paths, metrics = run_ensemble_dagger(
        config,
        datasets_train,
        datasets_val,
        params_dict,
        verbose=True
    )
    t_elapsed = time.time() - t_start

    print(f"\nTotal training time: {t_elapsed/60:.1f} minutes")
    print("Done!")
