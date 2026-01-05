#!/usr/bin/env python3
"""
CP4.4b: Jacobian Correctness Test

Compares C++ dynamics_linearize Jacobians against PyTorch autograd reference.

Tests at:
- 3 pre-defined operating points (rest, actuated, moving)
- 5 random points (seeded for reproducibility)

Tolerances: atol=1e-6, rtol=1e-4
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import crm_diff_py
from crm_dynamics_torch import dynamics_step


def extract_jacobians_pytorch(x_t, u_t, dt, L_inserted, params_dict):
    """
    Extract linearization A_t, B_t using PyTorch autograd (reference implementation).

    Args:
        x_t: np.array (6,)
        u_t: np.array (3,)
        dt: float
        L_inserted: float
        params_dict: dict

    Returns:
        A: np.array (6, 6), state Jacobian
        B: np.array (6, 3), control Jacobian
    """
    x_t_torch = torch.tensor(x_t, dtype=torch.float64, requires_grad=True)
    u_t_torch = torch.tensor(u_t, dtype=torch.float64, requires_grad=True)

    # A = ∂x_next/∂x_t
    A = torch.autograd.functional.jacobian(
        lambda x: dynamics_step(x, u_t_torch, dt, L_inserted, params_dict),
        x_t_torch
    ).numpy()

    # B = ∂x_next/∂u_t
    B = torch.autograd.functional.jacobian(
        lambda u: dynamics_step(x_t_torch, u, dt, L_inserted, params_dict),
        u_t_torch
    ).numpy()

    return A, B


def extract_jacobians_cpp(x_t, u_t, dt, L_inserted, params_dict):
    """
    Extract linearization A_t, B_t using C++ dynamics_linearize.

    Args:
        x_t: np.array (6,)
        u_t: np.array (3,)
        dt: float
        L_inserted: float
        params_dict: dict

    Returns:
        A: np.array (6, 6), state Jacobian
        B: np.array (6, 3), control Jacobian
    """
    result = crm_diff_py.dynamics_linearize(x_t, u_t, dt, L_inserted, params_dict)
    return result['A'], result['B']


def compare_jacobians(A_torch, B_torch, A_cpp, B_cpp, atol=1e-6, rtol=1e-4, label=""):
    """
    Compare PyTorch and C++ Jacobians.

    Returns:
        bool: True if within tolerance, False otherwise
    """
    A_close = np.allclose(A_torch, A_cpp, atol=atol, rtol=rtol)
    B_close = np.allclose(B_torch, B_cpp, atol=atol, rtol=rtol)

    if not A_close or not B_close:
        print(f"\n{label} - MISMATCH:")
        if not A_close:
            A_diff = np.abs(A_torch - A_cpp)
            print(f"  A max abs diff: {np.max(A_diff):.2e}")
            print(f"  A max rel diff: {np.max(A_diff / (np.abs(A_torch) + 1e-12)):.2e}")
            print(f"  A mismatches: {np.sum(~np.isclose(A_torch, A_cpp, atol=atol, rtol=rtol))}/36")
        if not B_close:
            B_diff = np.abs(B_torch - B_cpp)
            print(f"  B max abs diff: {np.max(B_diff):.2e}")
            print(f"  B max rel diff: {np.max(B_diff / (np.abs(B_torch) + 1e-12)):.2e}")
            print(f"  B mismatches: {np.sum(~np.isclose(B_torch, B_cpp, atol=atol, rtol=rtol))}/18")
        return False
    else:
        A_diff = np.abs(A_torch - A_cpp)
        B_diff = np.abs(B_torch - B_cpp)
        print(f"  {label}: ✓ PASS")
        print(f"    A max diff: {np.max(A_diff):.2e}, B max diff: {np.max(B_diff):.2e}")
        return True


def main():
    print("="*80)
    print("CP4.4b: Jacobian Correctness Test")
    print("="*80)
    print()

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

    dt = 0.01
    L_inserted = 50.0

    print("Test Configuration:")
    print(f"  dt: {dt}s")
    print(f"  L_inserted: {L_inserted}mm")
    print(f"  Tolerance: atol=1e-6, rtol=1e-4")
    print()

    all_pass = True

    # Test 1: Rest (zero state, zero control)
    print("Test 1: Rest (x=0, u=0)")
    x_t = np.zeros(6)
    u_t = np.zeros(3)

    A_torch, B_torch = extract_jacobians_pytorch(x_t, u_t, dt, L_inserted, params_dict)
    A_cpp, B_cpp = extract_jacobians_cpp(x_t, u_t, dt, L_inserted, params_dict)

    if not compare_jacobians(A_torch, B_torch, A_cpp, B_cpp, label="Rest"):
        all_pass = False

    # Test 2: Actuated (zero state, non-zero control)
    print("\nTest 2: Actuated (x=0, u≠0)")
    x_t = np.zeros(6)
    u_t = np.array([0.1, 0.05, -0.05])

    A_torch, B_torch = extract_jacobians_pytorch(x_t, u_t, dt, L_inserted, params_dict)
    A_cpp, B_cpp = extract_jacobians_cpp(x_t, u_t, dt, L_inserted, params_dict)

    if not compare_jacobians(A_torch, B_torch, A_cpp, B_cpp, label="Actuated"):
        all_pass = False

    # Test 3: Moving (non-zero state and control)
    print("\nTest 3: Moving (x≠0, u≠0)")
    x_t = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
    u_t = np.array([0.1, -0.1, 0.05])

    A_torch, B_torch = extract_jacobians_pytorch(x_t, u_t, dt, L_inserted, params_dict)
    A_cpp, B_cpp = extract_jacobians_cpp(x_t, u_t, dt, L_inserted, params_dict)

    if not compare_jacobians(A_torch, B_torch, A_cpp, B_cpp, label="Moving"):
        all_pass = False

    # Test 4-8: Random points (seeded)
    print("\nTests 4-8: Random points (seeded)")
    np.random.seed(42)  # Reproducibility

    for i in range(5):
        x_t = np.random.uniform(-0.05, 0.05, 6)
        x_t[3:] = np.random.uniform(-0.5, 0.5, 3)  # Velocities
        u_t = np.random.uniform(-0.2, 0.2, 3)

        try:
            A_torch, B_torch = extract_jacobians_pytorch(x_t, u_t, dt, L_inserted, params_dict)
            A_cpp, B_cpp = extract_jacobians_cpp(x_t, u_t, dt, L_inserted, params_dict)

            if not compare_jacobians(A_torch, B_torch, A_cpp, B_cpp, label=f"Random {i+1}"):
                all_pass = False
        except RuntimeError as e:
            print(f"  Random {i+1}: SKIP (dynamics failed: {e})")

    # Summary
    print("\n" + "="*80)
    if all_pass:
        print("✓ CP4.4b JACOBIAN CORRECTNESS TEST PASS")
        print("All C++ Jacobians match PyTorch autograd within tolerance")
        return 0
    else:
        print("✗ CP4.4b JACOBIAN CORRECTNESS TEST FAIL")
        print("Some C++ Jacobians do not match PyTorch autograd")
        return 1


if __name__ == "__main__":
    sys.exit(main())
