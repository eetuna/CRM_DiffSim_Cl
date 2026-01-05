"""
CP2.3: Python Bindings Smoke Test for Dynamics Primitives

Tests dynamics_forward and dynamics_backward bindings at three operating points:
- OP1: Rest (zero state, zero control)
- OP2: Actuated (zero state, non-zero control)
- OP3: Moving (non-zero state, non-zero control)

Acceptance criteria:
- Forward pass returns status==0, rank==6, finite outputs
- Backward pass returns status==0, rank==6, finite gradients
- Actuated cases produce non-zero control gradients
"""
import sys
import os
import numpy as np

# Add python directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

import crm_diff_py


def test_case(case_name, x_t, u_t, dt, L_inserted, params_dict):
    """
    Test a single operating point.

    Returns: bool (True if passed)
    """
    print(f"\n{'='*60}")
    print(f"Test Case: {case_name}")
    print(f"{'='*60}")
    print(f"  x_t = {x_t}")
    print(f"  u_t = {u_t}")
    print(f"  dt = {dt} s")
    print(f"  L_inserted = {L_inserted} mm")
    print()

    # Forward pass
    try:
        result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)
    except Exception as e:
        print(f"✗ FAIL: Forward pass raised exception: {e}")
        return False

    status = result['status']
    lu_rank = result['lu_rank']
    rel_residual = result['rel_solve_residual']
    x_next = result['x_next']
    p_tip = result['p_tip']

    print(f"Forward pass:")
    print(f"  status = {status}")
    print(f"  lu_rank = {lu_rank}")
    print(f"  rel_solve_residual = {rel_residual:.6e}")
    print(f"  x_next = {x_next}")
    print(f"  p_tip = {p_tip}")

    # Checks for forward pass
    passed = True

    if status != 0:
        print(f"✗ FAIL: Forward status = {status} (expected 0)")
        passed = False

    if lu_rank < 6:
        print(f"✗ FAIL: Rank-deficient matrix (rank = {lu_rank})")
        passed = False

    if not np.all(np.isfinite(x_next)):
        print(f"✗ FAIL: x_next contains non-finite values")
        passed = False

    if not np.all(np.isfinite(p_tip)):
        print(f"✗ FAIL: p_tip contains non-finite values")
        passed = False

    if rel_residual >= 1e-10:
        print(f"✗ FAIL: Forward solve residual too large ({rel_residual:.6e})")
        passed = False

    # Check shapes
    if x_next.shape != (6,):
        print(f"✗ FAIL: x_next shape = {x_next.shape} (expected (6,))")
        passed = False

    if p_tip.shape != (3,):
        print(f"✗ FAIL: p_tip shape = {p_tip.shape} (expected (3,))")
        passed = False

    # Backward pass test
    print()
    print(f"Backward pass:")

    # Use canonical direction e_0 = [1,0,0,0,0,0]
    grad_x_next = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    try:
        bwd_result = crm_diff_py.dynamics_backward(result, grad_x_next, params_dict)
    except Exception as e:
        print(f"✗ FAIL: Backward pass raised exception: {e}")
        return False

    bwd_status = bwd_result['status']
    bwd_rank = bwd_result['lu_rank']
    bwd_residual = bwd_result['rel_residual']
    grad_x_t = bwd_result['grad_x_t']
    grad_u_t = bwd_result['grad_u_t']

    print(f"  status = {bwd_status}")
    print(f"  lu_rank = {bwd_rank}")
    print(f"  rel_residual = {bwd_residual:.6e}")
    print(f"  grad_x_t = {grad_x_t}")
    print(f"  grad_u_t = {grad_u_t}")

    # Checks for backward pass
    if bwd_status != 0:
        print(f"✗ FAIL: Backward status = {bwd_status} (expected 0)")
        passed = False

    if bwd_rank < 6:
        print(f"✗ FAIL: Backward rank-deficient (rank = {bwd_rank})")
        passed = False

    if bwd_residual >= 1e-10:
        print(f"✗ FAIL: Backward solve residual too large ({bwd_residual:.6e})")
        passed = False

    if not np.all(np.isfinite(grad_x_t)):
        print(f"✗ FAIL: grad_x_t contains non-finite values")
        passed = False

    if not np.all(np.isfinite(grad_u_t)):
        print(f"✗ FAIL: grad_u_t contains non-finite values")
        passed = False

    # Check shapes
    if grad_x_t.shape != (6,):
        print(f"✗ FAIL: grad_x_t shape = {grad_x_t.shape} (expected (6,))")
        passed = False

    if grad_u_t.shape != (3,):
        print(f"✗ FAIL: grad_u_t shape = {grad_u_t.shape} (expected (3,))")
        passed = False

    # For actuated cases, expect non-zero control gradient
    u_norm = np.linalg.norm(u_t)
    grad_u_norm = np.linalg.norm(grad_u_t)

    if u_norm > 0.01:  # Actuated case
        if grad_u_norm < 1e-12:
            print(f"✗ FAIL: Actuated case but grad_u_t is essentially zero (||grad_u_t|| = {grad_u_norm:.6e})")
            passed = False
        else:
            print(f"  ||grad_u_t|| = {grad_u_norm:.6e} (non-zero, as expected for actuated case)")

    print()
    if passed:
        print(f"✓ PASS: {case_name}")
    else:
        print(f"✗ FAIL: {case_name}")

    return passed


def main():
    print("CP2.3 Dynamics Python Bindings Smoke Test")
    print("=" * 60)

    # Load parameters (same as C++ test)
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

    # Test configuration
    dt = 0.01  # 10 ms timestep
    L_inserted = 50.0  # mm

    # Test cases (matching C++ test_cp21_dynamics_smoke.cpp)
    test_cases = [
        ("OP1: Rest (zero state, zero control)",
         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
         np.array([0.0, 0.0, 0.0], dtype=np.float64)),

        ("OP2: Actuated (zero state, non-zero control)",
         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
         np.array([0.1, 0.0, 0.0], dtype=np.float64)),

        ("OP3: Moving (non-zero state, non-zero control)",
         np.array([0.01, 0.0, 0.0, 0.1, 0.0, 0.0], dtype=np.float64),
         np.array([0.1, 0.0, 0.0], dtype=np.float64)),
    ]

    # Run tests
    results = []
    for case_name, x_t, u_t in test_cases:
        passed = test_case(case_name, x_t, u_t, dt, L_inserted, params_dict)
        results.append((case_name, passed))

    # Summary
    print("\n" + "=" * 60)
    print("CP2.3 SUMMARY")
    print("=" * 60)

    all_passed = True
    for case_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {case_name}")
        if not passed:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("PASS: All test cases succeeded")
        print("=" * 60)
        return 0
    else:
        print("FAIL: One or more test cases failed")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
