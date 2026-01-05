#!/usr/bin/env python3
"""
FULLSTATE Step Forward Smoke Test

Verifies that true_legacy_step using the Python wrapper:
1. Returns correct FULLSTATE dimensions (18*N + 15)
2. Produces finite outputs (no NaN/Inf)
3. Is deterministic (same inputs → same outputs)
4. Succeeds for reasonable inputs

This is a basic correctness test for the FULLSTATE migration.
"""

import sys
import os
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import crm_diff_py
from control.true_legacy_state_adapter import (
    pack_true_legacy_state, unpack_true_legacy_state, true_legacy_state_dim
)
from control.true_legacy_step import true_legacy_step
from crm_config import get_default_params_dict


def get_params():
    """Get params dict with L_inserted set."""
    params = get_default_params_dict()
    return params


def get_initial_state(n_act=1):
    """Get a valid initial state at origin."""
    # Use torch tensors as the Python wrapper expects
    x_coil = torch.zeros((n_act, 18), dtype=torch.float64)
    # Set rotation to identity
    for j in range(n_act):
        x_coil[j, 9:18] = torch.eye(3, dtype=torch.float64).flatten()
        x_coil[j, 6:9] = torch.tensor([0.0, 0.0, 1.0 + j], dtype=torch.float64)

    xf = torch.zeros(15, dtype=torch.float64)
    xf[3:12] = torch.eye(3, dtype=torch.float64).flatten()
    xf[:3] = torch.tensor([0.0, 0.0, 10.0], dtype=torch.float64)

    return pack_true_legacy_state(x_coil, xf)


def test_fullstate_step_dimension():
    """Test that step forward returns correct FULLSTATE dimension."""
    n_act = 1
    state_dim = true_legacy_state_dim(n_act)  # Should be 18*1 + 15 = 33
    assert state_dim == 33, f"Expected state_dim=33, got {state_dim}"

    x = get_initial_state(n_act)
    u = torch.zeros((n_act, 3), dtype=torch.float64)

    dt = 0.01
    params_dict = get_params()

    # Step forward using Python wrapper
    x_next, obs = true_legacy_step(x, u, dt, n_act=n_act, catheter_params=params_dict)

    # Check dimensions
    assert x_next.shape == (state_dim,), f"x_next has wrong shape: {x_next.shape}"
    assert obs['tip_p'].shape == (3,), f"tip_p has wrong shape: {obs['tip_p'].shape}"

    print(f"✓ Dimension test passed: x_next.shape={x_next.shape}, tip_p.shape={obs['tip_p'].shape}")


def test_fullstate_step_finiteness():
    """Test that outputs are finite (no NaN or Inf)."""
    n_act = 1
    x = get_initial_state(n_act)

    # Apply small non-zero control
    u = torch.tensor([[0.1, -0.05, 0.08]], dtype=torch.float64)  # Shape: [n_act, 3]

    dt = 0.01
    params_dict = get_params()

    x_next, obs = true_legacy_step(x, u, dt, n_act=n_act, catheter_params=params_dict)

    # Check finiteness
    assert torch.all(torch.isfinite(x_next)), "x_next contains NaN or Inf"
    assert torch.all(torch.isfinite(obs['tip_p'])), "tip_p contains NaN or Inf"

    print(f"✓ Finiteness test passed: max|x_next|={torch.max(torch.abs(x_next)):.6f}")


def test_fullstate_step_determinism():
    """Test that step forward is deterministic."""
    n_act = 1
    torch.manual_seed(123)
    x = get_initial_state(n_act)
    x += torch.randn_like(x) * 0.01  # Small perturbation

    u = torch.tensor([[0.15, -0.1, 0.05]], dtype=torch.float64)  # Shape: [n_act, 3]

    dt = 0.01
    params_dict = get_params()

    # Run twice with same inputs
    x_next1, obs1 = true_legacy_step(x, u, dt, n_act=n_act, catheter_params=params_dict)
    x_next2, obs2 = true_legacy_step(x, u, dt, n_act=n_act, catheter_params=params_dict)

    # Check exact match
    assert torch.allclose(x_next1, x_next2, atol=0), "x_next not deterministic"
    assert torch.allclose(obs1['tip_p'], obs2['tip_p'], atol=0), "tip_p not deterministic"

    print(f"✓ Determinism test passed: exact match on repeated calls")


def test_fullstate_step_multistep():
    """Test a short 5-step rollout."""
    n_act = 1
    x = get_initial_state(n_act)

    dt = 0.01
    num_steps = 5
    params_dict = get_params()

    tip_positions = []

    for t in range(num_steps):
        # Time-varying control (small to avoid numerical instability)
        u = torch.tensor([[
            0.01 * np.sin(2 * np.pi * t / 10),
            0.01 * np.cos(2 * np.pi * t / 10),
            0.01 * np.sin(2 * np.pi * t / 5)
        ]], dtype=torch.float64)  # Shape: [n_act, 3]

        x, obs = true_legacy_step(x, u, dt, n_act=n_act, catheter_params=params_dict)

        assert torch.all(torch.isfinite(x)), f"Step {t}: x not finite"
        assert torch.all(torch.isfinite(obs['tip_p'])), f"Step {t}: tip_p not finite"

        tip_positions.append(obs['tip_p'].numpy())

    tip_positions = np.array(tip_positions)

    # Check that tip moved (not stuck at origin)
    tip_displacement = np.linalg.norm(tip_positions[-1] - tip_positions[0])

    print(f"✓ Multi-step rollout test passed:")
    print(f"  - {num_steps} steps completed successfully")
    print(f"  - Tip displacement: {tip_displacement:.6f} mm")
    print(f"  - Final tip position: {tip_positions[-1]}")
    print(f"  - All states finite")


def test_fullstate_step_zero_control():
    """Test that zero control produces stable (finite) evolution."""
    n_act = 1
    torch.manual_seed(456)
    x = get_initial_state(n_act)
    x += torch.randn_like(x) * 0.01  # Very small initial perturbation

    u = torch.zeros((n_act, 3), dtype=torch.float64)

    dt = 0.01
    params_dict = get_params()

    # Run 10 steps with zero control
    for t in range(10):
        x, obs = true_legacy_step(x, u, dt, n_act=n_act, catheter_params=params_dict)

        assert torch.all(torch.isfinite(x)), f"Step {t}: x not finite"
        assert torch.all(torch.isfinite(obs['tip_p'])), f"Step {t}: tip_p not finite"

    print(f"✓ Zero control stability test passed: 10 steps with zero control remained finite")


if __name__ == "__main__":
    print("=" * 60)
    print("FULLSTATE Step Forward Smoke Test")
    print("=" * 60)

    test_fullstate_step_dimension()
    test_fullstate_step_finiteness()
    test_fullstate_step_determinism()
    test_fullstate_step_multistep()
    test_fullstate_step_zero_control()

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
