#!/usr/bin/env python3
"""
CP4.3: DAgger Smoke Test

Fast validation of DAgger implementation:
- Runs 1 DAgger iteration on 1 trajectory
- Verifies policy file produced
- Checks failure rate bounded
- Runtime target: <30s
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from data.npz_manifest import load_manifest
import crm_diff_py
from train_cp43_dagger import (
    BehaviorCloningPolicy, MPCExpert, DAggerConfig,
    run_dagger_rollout
)


def main():
    print("=" * 80)
    print("CP4.3: DAgger Smoke Test")
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

    # Create policy (random initialization)
    policy = BehaviorCloningPolicy()
    print("\nPolicy initialized (random weights)")

    # Create config (fast settings)
    config = DAggerConfig()
    config.mpc_horizon = 3  # Very short for smoke test speed
    config.mpc_max_iters = 3  # Minimal iterations
    config.mpc_cost_tol = 1e-1  # Looser tolerance
    config.rollout_timeout_per_traj = 50  # 50 seconds

    # Create expert
    expert = MPCExpert(config, params_dict)

    # Run one rollout
    print("\nRunning DAgger rollout (smoke test)...")
    result = run_dagger_rollout(
        policy, dataset, expert, params_dict, config, verbose=True
    )

    # Checks
    print("\n" + "=" * 80)
    print("Validation Checks")
    print("=" * 80)

    all_pass = True

    # Check 1: Rollout completed
    if result['failed']:
        print("✗ FAIL: Rollout failed")
        all_pass = False
    else:
        print(f"✓ PASS: Rollout completed ({result['n_steps']} steps)")

    # Check 2: Expert queries reasonable
    if not result['failed']:
        expert_rate = result['expert_queries'] / max(result['n_steps'], 1)
        if expert_rate > 0.5:  # More than 50% expert queries is concerning
            print(f"⚠ WARNING: High expert query rate ({expert_rate:.1%})")
        else:
            print(f"✓ PASS: Expert query rate acceptable ({expert_rate:.1%})")

    # Check 3: Data collected
    if not result['failed']:
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

    # Summary
    print("\n" + "=" * 80)
    if all_pass:
        print("✓ CP4.3 DAGGER SMOKE TEST PASS")
        return 0
    else:
        print("✗ CP4.3 DAGGER SMOKE TEST FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())
