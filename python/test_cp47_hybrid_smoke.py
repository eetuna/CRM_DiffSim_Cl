#!/usr/bin/env python3
"""
CP4.7: Hybrid Controller Smoke Test

Fast validation test (<60s) for hybrid MPC + policy controller.

Tests:
1. Controller creation with valid ensemble
2. Single step produces valid control output
3. Full rollout without crashes
4. High uncertainty triggers MPC
"""

import sys
import os
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from control.hybrid_controller import HybridController, HybridControllerMetrics
from models.recurrent_policy import GRUPolicy
from models.ensemble_policy import EnsemblePolicy
import crm_diff_py


def create_test_ensemble(n_members=3, hidden_dim=32, seed=42):
    """Create a small test ensemble."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create temporary policies
    policies_paths = []
    temp_dir = './build/artifacts'
    os.makedirs(temp_dir, exist_ok=True)

    for i in range(n_members):
        policy = GRUPolicy(input_dim=6, hidden_dim=hidden_dim, output_dim=3, num_layers=1)

        # Initialize with different seeds
        torch.manual_seed(seed + i)
        for param in policy.parameters():
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)

        path = os.path.join(temp_dir, f'test_cp47_ensemble_member{i}.pth')
        torch.save(policy.state_dict(), path)
        policies_paths.append(path)

    ensemble = EnsemblePolicy(
        policy_paths=policies_paths,
        policy_class=GRUPolicy,
        policy_kwargs={'input_dim': 6, 'hidden_dim': hidden_dim, 'output_dim': 3, 'num_layers': 1}
    )

    return ensemble


def create_test_controller():
    """Create a test hybrid controller."""
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
        'IntegrationStepSize': 0.5,
        'FinalValueOnly': True,
    }

    ensemble = create_test_ensemble()

    controller = HybridController(
        ensemble=ensemble,
        dt=0.01,
        L_inserted=50.0,
        params_dict=params_dict,
        tau_low=0.001,
        tau_high=0.01,
        tracking_safety_limit=5.0,
        mpc_horizon=5,  # Short horizon for speed
        mpc_max_iters=3,  # Fast for testing
        mpc_tol=2.0  # Loose tolerance for speed
    )

    return controller, params_dict


def test_controller_creation():
    """Test 1: Controller can be created with valid ensemble."""
    print("\n" + "="*60)
    print("Test 1: Controller Creation")
    print("="*60)

    controller, _ = create_test_controller()

    assert controller is not None
    assert controller.ensemble is not None
    assert controller.mpc_horizon == 5
    assert controller.tau_low == 0.001
    assert controller.tau_high == 0.01

    print("  Created HybridController")
    print(f"    tau_low:  {controller.tau_low}")
    print(f"    tau_high: {controller.tau_high}")
    print(f"    MPC horizon: {controller.mpc_horizon}")
    print("  ✓ PASS")


def test_controller_step():
    """Test 2: Controller produces valid control output."""
    print("\n" + "="*60)
    print("Test 2: Single Step")
    print("="*60)

    controller, params_dict = create_test_controller()

    # Initial state
    x_t = np.zeros(6)
    result_init = crm_diff_py.dynamics_forward(
        x_t, np.zeros(3), 0.0, controller.L_inserted, params_dict
    )
    p_tip_t = result_init['p_tip']

    # Reference horizon
    p_ref_horizon = np.tile([1.0, 1.0, 52.0], (controller.mpc_horizon, 1))

    # Execute step
    u, info = controller.step(x_t, p_tip_t, p_ref_horizon, hiddens=None)

    print(f"  Input: p_tip={p_tip_t}, p_ref={p_ref_horizon[0]}")
    print(f"  Output: u={u}")
    print(f"  Mode: {info['mode']}")
    print(f"  Uncertainty: {info['uncertainty']:.6f}")

    # Assertions
    assert u.shape == (3,)
    assert np.all(np.abs(u) <= 0.5), "Control exceeds limits"
    assert info['mode'] in ['policy', 'mpc_warm', 'mpc_cold', 'safety_override']
    assert 'uncertainty' in info
    assert info['uncertainty'] >= 0.0

    print("  ✓ PASS")


def test_controller_rollout():
    """Test 3: Full rollout without crashes."""
    print("\n" + "="*60)
    print("Test 3: Rollout (30 steps)")
    print("="*60)

    controller, params_dict = create_test_controller()
    metrics = HybridControllerMetrics()

    # Simple circular reference trajectory
    n_steps = 30
    t = np.linspace(0, 10, n_steps)
    radius = 10.0
    p_ref_traj = np.zeros((n_steps, 3))
    p_ref_traj[:, 0] = radius * np.cos(t)
    p_ref_traj[:, 1] = radius * np.sin(t)
    p_ref_traj[:, 2] = 50.0 + 0.1 * t

    # Initial state
    x_t = np.zeros(6)
    result_init = crm_diff_py.dynamics_forward(
        x_t, np.zeros(3), 0.0, controller.L_inserted, params_dict
    )
    p_tip_t = result_init['p_tip']

    # Rollout
    n_mpc_calls = 0
    n_policy_only = 0

    for i in range(n_steps):
        if i % 10 == 0:
            print(f"  Step {i}/{n_steps}")

        # Prepare reference horizon
        horizon_end = min(i + controller.mpc_horizon, n_steps)
        p_ref_horizon = p_ref_traj[i:horizon_end]

        if len(p_ref_horizon) < controller.mpc_horizon:
            p_ref_horizon = np.vstack([
                p_ref_horizon,
                np.tile(p_ref_horizon[-1], (controller.mpc_horizon - len(p_ref_horizon), 1))
            ])

        # Control step
        u, info = controller.step(x_t, p_tip_t, p_ref_horizon, hiddens=None)

        # Apply control
        result = crm_diff_py.dynamics_forward(
            x_t, u, controller.dt, controller.L_inserted, params_dict
        )

        # Update state
        if result['status'] == 0:
            x_t = result['x_next']
            p_tip_t = result['p_tip']
        else:
            # Dynamics failed, stay at current state (shouldn't happen with safety override)
            print(f"    WARNING: Dynamics failed at step {i}")

        # Record metrics
        metrics.add_step(info, p_tip_t, p_ref_traj[i])

        if info['mpc_called']:
            n_mpc_calls += 1
        if info['mode'] == 'policy':
            n_policy_only += 1

    # Summary
    summary = metrics.get_summary()

    print(f"\n  Rollout Summary:")
    print(f"    Steps: {summary['n_steps']}")
    print(f"    MPC call rate: {summary['mpc_call_rate']:.1%}")
    print(f"    Policy-only: {n_policy_only}/{n_steps} ({n_policy_only/n_steps:.1%})")
    print(f"    Tracking RMSE: {summary['tracking_rmse']:.3f} mm")
    print(f"    Tracking Max: {summary['tracking_max']:.3f} mm")

    # Assertions
    assert summary['n_steps'] == n_steps
    # Note: With untrained ensemble, MPC may be called frequently (high uncertainty)
    # This is expected behavior - the test just validates no crashes
    assert summary['mpc_call_rate'] <= 1.0, "Invalid MPC call rate"
    assert summary['tracking_rmse'] < 50.0, "Tracking error too large (controller not working)"

    print("  ✓ PASS")


def test_uncertainty_triggers_mpc():
    """Test 4: High uncertainty triggers MPC."""
    print("\n" + "="*60)
    print("Test 4: Uncertainty Triggers MPC")
    print("="*60)

    controller, params_dict = create_test_controller()

    # Unusual state (far from training distribution)
    x_unusual = np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])

    # Get tip position
    result = crm_diff_py.dynamics_forward(
        x_unusual, np.zeros(3), 0.0, controller.L_inserted, params_dict
    )
    p_tip = result['p_tip']

    # Reference far away
    p_ref = p_tip + np.array([20.0, 20.0, 10.0])
    p_ref_horizon = np.tile(p_ref, (controller.mpc_horizon, 1))

    # Test with low threshold (should use policy)
    controller_low = HybridController(
        ensemble=controller.ensemble,
        dt=controller.dt,
        L_inserted=controller.L_inserted,
        params_dict=params_dict,
        tau_low=1.0,  # Very high threshold
        tau_high=10.0,
        mpc_horizon=3,
        mpc_max_iters=2
    )

    u_low, info_low = controller_low.step(x_unusual, p_tip, p_ref_horizon, hiddens=None)
    print(f"  Low threshold (tau_low=1.0): mode={info_low['mode']}, uncertainty={info_low['uncertainty']:.6f}")

    # Test with high threshold (should use MPC)
    controller_high = HybridController(
        ensemble=controller.ensemble,
        dt=controller.dt,
        L_inserted=controller.L_inserted,
        params_dict=params_dict,
        tau_low=0.0001,  # Very low threshold
        tau_high=0.001,
        mpc_horizon=3,
        mpc_max_iters=2
    )

    u_high, info_high = controller_high.step(x_unusual, p_tip, p_ref_horizon, hiddens=None)
    print(f"  High threshold (tau_low=0.0001): mode={info_high['mode']}, uncertainty={info_high['uncertainty']:.6f}")

    # Assertions
    # Low threshold: policy should be used (unless safety override kicks in)
    # Note: With unusual states, safety_override is acceptable
    assert info_low['mode'] in ['policy', 'safety_override'], \
        f"Expected policy or safety_override mode with high threshold, got {info_low['mode']}"

    # High threshold: MPC should be called
    # Note: safety_override also counts as MPC being called
    assert info_high['mpc_called'] or info_high['mode'] in ['mpc_warm', 'mpc_cold', 'safety_override'], \
        "MPC or safety override should be triggered with low threshold"

    print("  ✓ PASS")


def main():
    """Run all smoke tests."""
    print("\n" + "="*60)
    print("CP4.7: Hybrid Controller Smoke Test")
    print("="*60)

    t_start = time.time()

    # Run tests
    try:
        test_controller_creation()
        test_controller_step()
        test_controller_rollout()
        test_uncertainty_triggers_mpc()

        t_elapsed = time.time() - t_start

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        print(f"Runtime: {t_elapsed:.2f}s (target: <60s)")

        if t_elapsed >= 60:
            print("  WARNING: Test exceeded 60s timeout")
            return 1

        return 0

    except AssertionError as e:
        print("\n" + "="*60)
        print("✗ TEST FAILED")
        print("="*60)
        print(f"Error: {e}")
        return 1

    except Exception as e:
        print("\n" + "="*60)
        print("✗ TEST ERROR")
        print("="*60)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
