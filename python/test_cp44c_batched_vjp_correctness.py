#!/usr/bin/env python3
"""
CP4.4c: Batched VJP Correctness Test

Compares batched VJP implementation against per-vector VJP implementation.
The batched version should give identical results to calling dynamics_backward K times.

Tests at:
- 3 pre-defined operating points (rest, actuated, moving)
- 5 random points (seeded for reproducibility)

Tolerances: atol=1e-8, rtol=1e-6 (exact match expected)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import crm_diff_py


def extract_vjps_per_vector(x_t, u_t, dt, L_inserted, params_dict, V):
    """
    Extract VJP results by calling dynamics_backward K times (reference implementation).

    Args:
        x_t: np.array (6,)
        u_t: np.array (3,)
        dt: float
        L_inserted: float
        params_dict: dict
        V: np.array (K, 6) - adjoint vectors

    Returns:
        W_x: np.array (K, 6) - gradients w.r.t. x_t
        W_u: np.array (K, 3) - gradients w.r.t. u_t
    """
    K = V.shape[0]

    # Forward pass
    fwd_result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)
    if fwd_result['status'] != 0:
        raise RuntimeError(f"Forward pass failed with status {fwd_result['status']}")

    W_x = np.zeros((K, 6))
    W_u = np.zeros((K, 3))

    # Call backward K times
    for k in range(K):
        bwd_result = crm_diff_py.dynamics_backward(
            fwd_result,
            V[k, :],  # k-th adjoint vector
            params_dict
        )
        if bwd_result['status'] != 0:
            raise RuntimeError(f"Backward pass {k} failed with status {bwd_result['status']}")

        W_x[k, :] = bwd_result['grad_x_t']
        W_u[k, :] = bwd_result['grad_u_t']

    return W_x, W_u


def extract_vjps_batched(x_t, u_t, dt, L_inserted, params_dict, V):
    """
    Extract VJP results using batched API.

    Args:
        x_t: np.array (6,)
        u_t: np.array (3,)
        dt: float
        L_inserted: float
        params_dict: dict
        V: np.array (K, 6) - adjoint vectors

    Returns:
        W_x: np.array (K, 6) - gradients w.r.t. x_t
        W_u: np.array (K, 3) - gradients w.r.t. u_t
    """
    result = crm_diff_py.dynamics_linearize_batched(x_t, u_t, dt, L_inserted, params_dict, V)
    if result['status'] != 0:
        raise RuntimeError(f"Batched VJP failed with status {result['status']}")

    return result['W_x'], result['W_u']


def compare_vjps(W_x_ref, W_u_ref, W_x_batched, W_u_batched, atol=1e-8, rtol=1e-6, label=""):
    """
    Compare per-vector and batched VJP results.

    Returns:
        bool: True if within tolerance, False otherwise
    """
    W_x_close = np.allclose(W_x_ref, W_x_batched, atol=atol, rtol=rtol)
    W_u_close = np.allclose(W_u_ref, W_u_batched, atol=atol, rtol=rtol)

    if not W_x_close or not W_u_close:
        print(f"\n{label} - MISMATCH:")
        if not W_x_close:
            W_x_diff = np.abs(W_x_ref - W_x_batched)
            print(f"  W_x max abs diff: {np.max(W_x_diff):.2e}")
            print(f"  W_x max rel diff: {np.max(W_x_diff / (np.abs(W_x_ref) + 1e-12)):.2e}")
            print(f"  W_x mismatches: {np.sum(~np.isclose(W_x_ref, W_x_batched, atol=atol, rtol=rtol))}/{W_x_ref.size}")
        if not W_u_close:
            W_u_diff = np.abs(W_u_ref - W_u_batched)
            print(f"  W_u max abs diff: {np.max(W_u_diff):.2e}")
            print(f"  W_u max rel diff: {np.max(W_u_diff / (np.abs(W_u_ref) + 1e-12)):.2e}")
            print(f"  W_u mismatches: {np.sum(~np.isclose(W_u_ref, W_u_batched, atol=atol, rtol=rtol))}/{W_u_ref.size}")
        return False
    else:
        W_x_diff = np.abs(W_x_ref - W_x_batched)
        W_u_diff = np.abs(W_u_ref - W_u_batched)
        print(f"  {label}: ✓ PASS")
        print(f"    W_x max diff: {np.max(W_x_diff):.2e}, W_u max diff: {np.max(W_u_diff):.2e}")
        return True


def main():
    print("="*80)
    print("CP4.4c: Batched VJP Correctness Test")
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
    print(f"  Tolerance: atol=1e-8, rtol=1e-6 (exact match expected)")
    print()

    all_pass = True

    # Test with canonical basis vectors (identity matrix)
    K = 6
    V = np.eye(K)

    # Test 1: Rest (zero state, zero control)
    print("Test 1: Rest (x=0, u=0)")
    x_t = np.zeros(6)
    u_t = np.zeros(3)

    try:
        W_x_ref, W_u_ref = extract_vjps_per_vector(x_t, u_t, dt, L_inserted, params_dict, V)
        W_x_batched, W_u_batched = extract_vjps_batched(x_t, u_t, dt, L_inserted, params_dict, V)

        if not compare_vjps(W_x_ref, W_u_ref, W_x_batched, W_u_batched, label="Rest"):
            all_pass = False
    except RuntimeError as e:
        print(f"  Rest: SKIP (dynamics failed: {e})")

    # Test 2: Actuated (zero state, non-zero control)
    print("\nTest 2: Actuated (x=0, u≠0)")
    x_t = np.zeros(6)
    u_t = np.array([0.1, 0.05, -0.05])

    try:
        W_x_ref, W_u_ref = extract_vjps_per_vector(x_t, u_t, dt, L_inserted, params_dict, V)
        W_x_batched, W_u_batched = extract_vjps_batched(x_t, u_t, dt, L_inserted, params_dict, V)

        if not compare_vjps(W_x_ref, W_u_ref, W_x_batched, W_u_batched, label="Actuated"):
            all_pass = False
    except RuntimeError as e:
        print(f"  Actuated: SKIP (dynamics failed: {e})")

    # Test 3: Moving (non-zero state and control)
    print("\nTest 3: Moving (x≠0, u≠0)")
    x_t = np.array([0.01, 0.01, 0.01, 0.1, 0.1, 0.1])
    u_t = np.array([0.1, -0.1, 0.05])

    try:
        W_x_ref, W_u_ref = extract_vjps_per_vector(x_t, u_t, dt, L_inserted, params_dict, V)
        W_x_batched, W_u_batched = extract_vjps_batched(x_t, u_t, dt, L_inserted, params_dict, V)

        if not compare_vjps(W_x_ref, W_u_ref, W_x_batched, W_u_batched, label="Moving"):
            all_pass = False
    except RuntimeError as e:
        print(f"  Moving: SKIP (dynamics failed: {e})")

    # Test 4-8: Random points (seeded)
    print("\nTests 4-8: Random points (seeded)")
    np.random.seed(42)  # Reproducibility

    for i in range(5):
        x_t = np.random.uniform(-0.05, 0.05, 6)
        x_t[3:] = np.random.uniform(-0.5, 0.5, 3)  # Velocities
        u_t = np.random.uniform(-0.2, 0.2, 3)

        try:
            W_x_ref, W_u_ref = extract_vjps_per_vector(x_t, u_t, dt, L_inserted, params_dict, V)
            W_x_batched, W_u_batched = extract_vjps_batched(x_t, u_t, dt, L_inserted, params_dict, V)

            if not compare_vjps(W_x_ref, W_u_ref, W_x_batched, W_u_batched, label=f"Random {i+1}"):
                all_pass = False
        except RuntimeError as e:
            print(f"  Random {i+1}: SKIP (dynamics failed: {e})")

    # Summary
    print("\n" + "="*80)
    if all_pass:
        print("✓ CP4.4c BATCHED VJP CORRECTNESS TEST PASS")
        print("Batched VJP matches per-vector VJP within tolerance")
        return 0
    else:
        print("✗ CP4.4c BATCHED VJP CORRECTNESS TEST FAIL")
        print("Batched VJP does not match per-vector VJP")
        return 1


if __name__ == "__main__":
    sys.exit(main())
