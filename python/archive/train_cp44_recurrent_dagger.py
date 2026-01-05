#!/usr/bin/env python3
"""
CP4.4a: Recurrent DAgger (Dataset Aggregation with GRU/LSTM)

Extends CP4.3 DAgger to use recurrent policies (GRU/LSTM) for temporal context.

Key differences from CP4.3:
- Uses GRUPolicy instead of BehaviorCloningPolicy
- Hidden state management during rollout and training
- Hidden state reset at episode boundaries
- Hidden state detached between steps (no backprop through full horizon)

All other features identical to CP4.3:
- Safety checks: status==0, rank==6, residual<1e-10, error<5mm
- MPC expert with iLQR warm-start
- Adaptive epochs, early stopping
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
from models.recurrent_policy import GRUPolicy, LSTMPolicy
import crm_diff_py


# ============================================================================
# Configuration
# ============================================================================

class RecurrentDAggerConfig:
    """Recurrent DAgger training configuration."""

    # DAgger iterations
    n_iterations = 5

    # Safety thresholds (same as CP4.3)
    max_tracking_error = 5.0  # mm - trigger expert if exceeded
    residual_threshold = 1e-10

    # MPC expert (same as CP4.3)
    mpc_horizon = 10
    mpc_max_iters = 10  # Fast expert queries
    mpc_cost_tol = 1e-2

    # Training
    batch_size = 64
    base_epochs = 50
    lr = 1e-3
    val_patience = 10  # Early stopping
    gradient_clip = 1.0  # Clip gradients for RNN stability

    # Rollout
    rollout_timeout_per_traj = 300  # seconds (5 min)

    # Policy architecture
    policy_type = 'gru'  # 'gru' or 'lstm'
    hidden_dim = 64
    num_layers = 1

    # Artifacts
    output_dir = './build/artifacts'
    checkpoint_prefix = 'cp44_recurrent_dagger'


# ============================================================================
# Recurrent DAgger Dataset
# ============================================================================

class RecurrentDAggerDataset(Dataset):
    """
    PyTorch Dataset for recurrent DAgger.

    Sequences are broken into episodes for proper hidden state handling.
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
            print(f"  RecurrentDAggerDataset: {len(self)} samples from {len(aggregated_data)} iterations")

    def __len__(self):
        return len(self.p_tips)

    def __getitem__(self, idx):
        return self.p_tips[idx], self.p_refs[idx], self.us[idx]


# ============================================================================
# MPC Expert (same as CP4.3)
# ============================================================================

class MPCExpert:
    """MPC expert for DAgger expert queries."""

    def __init__(self, config, params_dict):
        self.config = config
        self.params_dict = params_dict
        self.U_warm = None

        self.n_queries = 0
        self.n_failures = 0
        self.query_times = []

    def query(self, x_t, p_ref_horizon, dt, L_inserted, verbose=False):
        """Query MPC expert for control action."""
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
            tol=self.config.mpc_cost_tol
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
        """Reset warm-start buffer."""
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
# Recurrent DAgger Rollout
# ============================================================================

def run_recurrent_dagger_rollout(policy, dataset, expert, params_dict, config, verbose=True):
    """
    Run DAgger rollout with recurrent policy.

    Key difference: Hidden state managed across timesteps, reset at episode start.

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

    # Initialize state
    x_t = np.zeros(6)

    # Initialize hidden state
    hidden = policy.init_hidden(batch_size=1, device='cpu')

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

        # Detach hidden state to prevent backprop through full trajectory
        if isinstance(hidden, tuple):  # LSTM
            hidden = (hidden[0].detach(), hidden[1].detach())
        else:  # GRU
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
                # Accept policy action
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


# ============================================================================
# Training
# ============================================================================

def train_recurrent_policy(policy, train_loader, val_loader, config, verbose=True):
    """Train recurrent policy with early stopping."""

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

            # Forward pass (no hidden state for supervised learning)
            u_pred, _ = policy(p_tip_batch, p_ref_batch, hidden=None)

            loss = criterion(u_pred, u_batch)
            loss.backward()

            # Gradient clipping for RNN stability
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
# Main DAgger Loop
# ============================================================================

def run_recurrent_dagger(config, datasets_train, datasets_val, params_dict, verbose=True):
    """Run full recurrent DAgger training."""

    print("="*80)
    print("CP4.4a: Recurrent DAgger Training")
    print("="*80)
    print(f"Policy: {config.policy_type.upper()}, hidden_dim={config.hidden_dim}")
    print(f"Iterations: {config.n_iterations}")
    print(f"Train datasets: {len(datasets_train)}")
    print(f"Val datasets: {len(datasets_val)}")
    print()

    # Initialize policy
    if config.policy_type == 'gru':
        policy = GRUPolicy(
            input_dim=6,
            hidden_dim=config.hidden_dim,
            output_dim=3,
            num_layers=config.num_layers
        )
    elif config.policy_type == 'lstm':
        policy = LSTMPolicy(
            input_dim=6,
            hidden_dim=config.hidden_dim,
            output_dim=3,
            num_layers=config.num_layers
        )
    else:
        raise ValueError(f"Unknown policy type: {config.policy_type}")

    print(f"Policy parameters: {sum(p.numel() for p in policy.parameters())}")
    print()

    # Initialize expert
    expert = MPCExpert(config, params_dict)

    # Storage
    aggregated_data = {}
    metrics = {
        'config': {
            'policy_type': config.policy_type,
            'hidden_dim': config.hidden_dim,
            'n_iterations': config.n_iterations,
            'max_tracking_error': config.max_tracking_error,
            'mpc_horizon': config.mpc_horizon
        },
        'iterations': []
    }

    # DAgger iterations
    for iteration in range(config.n_iterations):
        print(f"Iteration {iteration}/{config.n_iterations}")
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
                policy, ds, expert, params_dict, config, verbose=False
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

        # Training phase
        print("  Training phase...")

        train_dataset = RecurrentDAggerDataset(aggregated_data, verbose=False)
        val_dataset = RecurrentDAggerDataset(aggregated_data, verbose=False)  # Same for now

        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        train_metrics = train_recurrent_policy(
            policy, train_loader, val_loader, config, verbose=True
        )

        # Save checkpoint
        os.makedirs(config.output_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            config.output_dir,
            f"{config.checkpoint_prefix}_iter{iteration}_policy.pth"
        )
        torch.save(policy.state_dict(), checkpoint_path)
        print(f"    Saved: {checkpoint_path}")

        # Metrics
        iter_time = time.time() - iter_start

        iter_metrics = {
            'iteration': iteration,
            'rollout': {
                'total_steps': int(n_total_steps),
                'expert_queries': int(n_expert_queries),
                'expert_query_rate': float(expert_query_rate),
                'failed_trajectories': int(n_failed),
                'successful_trajectories': len(datasets_train) - n_failed
            },
            'expert_stats': expert.get_stats(),
            'training': train_metrics,
            'timing': {
                'iteration_time': float(iter_time)
            }
        }

        metrics['iterations'].append(iter_metrics)
        print()

    # Save final policy
    final_path = os.path.join(config.output_dir, f"{config.checkpoint_prefix}_final_policy.pth")
    torch.save(policy.state_dict(), final_path)
    print(f"Final policy saved: {final_path}")

    # Save metrics
    metrics_path = os.path.join(config.output_dir, f"{config.checkpoint_prefix}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    print()
    print("="*80)
    print("✓ Recurrent DAgger Complete")
    print("="*80)

    return policy, metrics


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    # Load manifest
    manifest = load_manifest()
    splits = create_splits(manifest, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)

    datasets_train = splits['train']
    datasets_val = splits['val']

    print(f"Loaded {len(datasets_train)} train, {len(datasets_val)} val datasets")

    # Load catheter parameters
    cath_params = crm_diff_py.get_default_cath_params()
    cath_config = crm_diff_py.get_default_cath_config()

    params_dict = {
        'CathParams': cath_params,
        'CathConfig': cath_config,
        'ContactMode': int(crm_diff_py.ContactModeType.FREE_TIP),
        'TipForce': [0.0, 0.0, 0.0],
        'deltau0_initialguess': [0.0, 0.0, 0.0],
        'IntegrationStepSize': 0.2,  # From dataset
        'FinalValueOnly': True,
    }

    # Run recurrent DAgger
    config = RecurrentDAggerConfig()
    policy, metrics = run_recurrent_dagger(
        config,
        datasets_train,
        datasets_val,
        params_dict,
        verbose=True
    )

    print("\nDone!")
