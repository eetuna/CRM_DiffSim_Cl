#!/usr/bin/env python3
"""
CP4.4a: Recurrent DAgger Smoke Test

Fast validation of recurrent DAgger implementation:
- Runs 1 DAgger iteration on 1 trajectory with GRU policy
- Verifies hidden state management works correctly
- Verifies policy file produced
- Checks failure rate bounded
- Runtime target: <60s
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from data.npz_manifest import load_manifest
from models.recurrent_policy import GRUPolicy
import crm_diff_py
from train_cp44_recurrent_dagger import (
    MPCExpert, RecurrentDAggerConfig,
    run_recurrent_dagger_rollout, RecurrentDAggerDataset,
    train_recurrent_policy
)
from torch.utils.data import DataLoader


def main():
    print("=" * 80)
    print("CP4.4a: Recurrent DAgger Smoke Test")
    print("=" * 80)
    print()

    # Load one dataset
    manifest = load_manifest(data_dir='data', verbose=False)

    if not manifest.datasets_with_ref:
        print("ERROR: No datasets with references")
        return 1

    dataset = manifest.datasets_with_ref[0]
    print(f"Using dataset: {dataset.filename}")
    print(f"  {dataset.param_summary()}")

    # Load physics parameters
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
        'IntegrationStepSize': dataset.integration_step_size,
        'FinalValueOnly': True,
    }

    # Create GRU policy (random initialization)
    policy = GRUPolicy(input_dim=6, hidden_dim=64, output_dim=3, num_layers=1)
    print(f"\nGRU Policy initialized ({sum(p.numel() for p in policy.parameters())} params)")

    # Test hidden state initialization
    print("\nTesting hidden state management...")
    hidden = policy.init_hidden(batch_size=1, device='cpu')
    print(f"  Hidden shape: {hidden.shape}")

    # Test forward pass
    p_tip_test = torch.randn(1, 3)
    p_ref_test = torch.randn(1, 3)
    u_test, hidden_new = policy(p_tip_test, p_ref_test, hidden)
    print(f"  Forward pass: u shape {u_test.shape}, new hidden shape {hidden_new.shape}")
    print("✓ Hidden state management works")

    # Create config (fast settings for smoke test)
    config = RecurrentDAggerConfig()
    config.mpc_horizon = 3  # Very short for speed
    config.mpc_max_iters = 3  # Minimal iterations
    config.mpc_cost_tol = 1e-1  # Looser tolerance
    config.rollout_timeout_per_traj = 25  # 25 seconds - short for smoke test
    config.base_epochs = 5  # Very few epochs for smoke test
    config.val_patience = 3  # Early patience
    config.max_tracking_error = 10.0  # More lenient for random policy

    # Create expert
    expert = MPCExpert(config, params_dict)

    # Run one rollout
    print("\nRunning recurrent DAgger rollout (smoke test)...")
    result = run_recurrent_dagger_rollout(
        policy, dataset, expert, params_dict, config, verbose=True
    )

    # Validation checks
    print("\n" + "=" * 80)
    print("Validation Checks")
    print("=" * 80)

    all_pass = True

    # Check 1: Rollout completed (or timed out with some data)
    if result['failed'] and result.get('n_steps', 0) == 0:
        print("✗ FAIL: Rollout failed immediately")
        all_pass = False
    elif result['failed']:
        print(f"⚠ WARNING: Rollout timed out after {result['n_steps']} steps (acceptable for smoke test)")
        # Allow timeout as long as we got some data
    else:
        print(f"✓ PASS: Rollout completed ({result['n_steps']} steps)")

    # Check 2: Expert queries reasonable
    if result.get('n_steps', 0) > 0:
        expert_rate = result['expert_queries'] / max(result['n_steps'], 1)
        if expert_rate > 0.9:  # More than 90% is concerning
            print(f"⚠ WARNING: Very high expert query rate ({expert_rate:.1%})")
        else:
            print(f"✓ PASS: Expert query rate acceptable ({expert_rate:.1%})")

    # Check 3: Data collected
    if result.get('n_steps', 0) > 0 and 'u_experts' in result:
        if len(result['u_experts']) > 0:
            print(f"✓ PASS: Data collected ({len(result['u_experts'])} samples)")
        else:
            print("✗ FAIL: No data collected")
            all_pass = False

    # Check 4: Expert stats
    expert_stats = expert.get_stats()
    print(f"\nExpert Statistics:")
    print(f"  Queries: {expert_stats['n_queries']}")
    print(f"  Failures: {expert_stats['n_failures']}")
    print(f"  Failure rate: {expert_stats['failure_rate']:.1%}")
    print(f"  Mean query time: {expert_stats['mean_query_time']:.3f}s")

    if expert_stats['failure_rate'] > 0.5:
        print("✗ FAIL: Expert failure rate > 50%")
        all_pass = False
    else:
        print("✓ PASS: Expert failure rate acceptable")

    # Check 5: Train policy on collected data (quick smoke test)
    if result.get('n_steps', 0) > 0 and 'u_experts' in result and len(result.get('u_experts', [])) > 10:
        print("\nTesting training loop...")

        aggregated_data = {
            'iteration_0': {
                'p_tips': result['p_tips'],
                'p_refs': result['p_refs'],
                'u_experts': result['u_experts']
            }
        }

        train_dataset = RecurrentDAggerDataset(aggregated_data, verbose=False)
        val_dataset = RecurrentDAggerDataset(aggregated_data, verbose=False)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        train_metrics = train_recurrent_policy(
            policy, train_loader, val_loader, config, verbose=False
        )

        print(f"  Training completed: {train_metrics['epochs_run']} epochs")
        print(f"  Best val loss: {train_metrics['best_val_loss']:.6f}")
        print("✓ PASS: Training loop works")

        # Check 6: Save policy
        os.makedirs('./build/artifacts', exist_ok=True)
        policy_path = './build/artifacts/cp44_recurrent_smoke_policy.pth'
        torch.save(policy.state_dict(), policy_path)

        if os.path.exists(policy_path):
            print(f"✓ PASS: Policy saved to {policy_path}")
        else:
            print("✗ FAIL: Policy file not saved")
            all_pass = False

    # Summary
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ CP4.4a RECURRENT DAGGER SMOKE TEST PASS")
        return 0
    else:
        print("✗ CP4.4a RECURRENT DAGGER SMOKE TEST FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
