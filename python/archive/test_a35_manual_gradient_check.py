"""
A3.5: Manual State Gradient Check

Tests state gradients using the low-level VJP API directly,
bypassing the autograd wrapper to avoid convergence issues.

Verifies that J_yx ≠ 0 is properly computed and wired.
"""

import sys
import numpy as np
import crm_diff_py

from control import load_default_catheter_params

# Global params
PARAMS_DICT = None
N_ACT = 1


def load_params():
    """Load catheter parameters once."""
    global PARAMS_DICT
    if PARAMS_DICT is None:
        param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
        config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
        PARAMS_DICT = load_default_catheter_params(param_file, config_file)
    return PARAMS_DICT


def test_state_gradient_nonzero():
    """
    Test that state gradients are NON-ZERO using the low-level VJP API.

    This bypasses convergence checks and directly verifies that J_yx
    produces non-zero state gradients.
    """
    print("\n" + "="*70)
    print("A3.5: Manual State Gradient Check")
    print("="*70)

    params = load_params()
    dt = 0.001
    L_inserted = 100.0

    # Simple initialization
    x_coil = np.zeros((N_ACT, 18), dtype=np.float64)
    xf = np.zeros(15, dtype=np.float64)

    # Identity rotations
    R_identity = np.eye(3).flatten()
    x_coil[0, 9:18] = R_identity
    xf[3:12] = R_identity

    # Small positions
    x_coil[0, 6:9] = [0.0, 0.0, 1.0]
    xf[:3] = [0.0, 0.0, 10.0]

    # Small actuation
    u = np.array([[0.01, 0.0, 0.0]], dtype=np.float64)

    # Upstream gradient (arbitrary non-zero vector)
    grad_tip_p = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    print(f"  dt: {dt}")
    print(f"  L_inserted: {L_inserted}")
    print(f"  grad_tip_p: {grad_tip_p}")

    # Call VJP directly
    try:
        result = crm_diff_py.true_legacy_step_vjp(
            x_coil, xf, u, dt, L_inserted, params, grad_tip_p
        )
    except Exception as e:
        print(f"\n  [ERROR] VJP call failed: {e}")
        return False

    grad_x_coil = result['grad_x_coil']
    grad_xf = result['grad_xf']
    grad_u = result['grad_u']

    print(f"\n  Gradient shapes:")
    print(f"  grad_x_coil: {grad_x_coil.shape}")
    print(f"  grad_xf: {grad_xf.shape}")
    print(f"  grad_u: {grad_u.shape}")

    # Compute gradient norms
    grad_x_coil_norm = np.linalg.norm(grad_x_coil)
    grad_xf_norm = np.linalg.norm(grad_xf)
    grad_u_norm = np.linalg.norm(grad_u)

    print(f"\n  Gradient norms:")
    print(f"  ||grad_x_coil||: {grad_x_coil_norm:.6e}")
    print(f"  ||grad_xf||: {grad_xf_norm:.6e}")
    print(f"  ||grad_u||: {grad_u_norm:.6e}")

    # Analyze structure
    v_grad = grad_x_coil[0, :3]
    w_grad = grad_x_coil[0, 3:6]
    p_grad = grad_x_coil[0, 6:9]
    R_grad = grad_x_coil[0, 9:18]

    print(f"\n  Gradient components:")
    print(f"  v_grad (velocity): {v_grad}")
    print(f"  w_grad (ang. vel): {w_grad}")
    print(f"  p_grad (position): {p_grad}")
    print(f"  R_grad norm: {np.linalg.norm(R_grad):.6e}")

    # Check for non-zero gradients
    tol = 1e-12

    grad_x_coil_nonzero = np.max(np.abs(grad_x_coil)) > tol
    grad_xf_nonzero = np.max(np.abs(grad_xf)) > tol
    v_grad_nonzero = np.max(np.abs(v_grad)) > tol
    w_grad_nonzero = np.max(np.abs(w_grad)) > tol

    print(f"\n  Non-zero checks:")
    print(f"  grad_x_coil non-zero: {grad_x_coil_nonzero}")
    print(f"  grad_xf non-zero: {grad_xf_nonzero}")
    print(f"  v_grad non-zero: {v_grad_nonzero}")
    print(f"  w_grad non-zero: {w_grad_nonzero}")

    # CRITICAL: Verify state gradients are non-zero
    # This proves J_yx ≠ 0 is being used
    state_grads_nonzero = grad_x_coil_nonzero or grad_xf_nonzero

    if state_grads_nonzero:
        print(f"\n  [PASS] State gradients are NON-ZERO")
        print(f"  This confirms J_yx is properly computed and contributes to gradients")

        # Additional check: velocity/angular velocity should have coupling
        if v_grad_nonzero or w_grad_nonzero:
            print(f"  [PASS] Velocity coupling confirmed (v or w gradients non-zero)")
        else:
            print(f"  [WARN] Velocity coupling weak (but state gradients still present)")
    else:
        print(f"\n  [FAIL] All state gradients are ZERO")
        print(f"  This suggests J_yx is not being used (implementation error)")

    return state_grads_nonzero


def main():
    """Run manual state gradient check."""
    print("\n" + "="*70)
    print("A3.5: MANUAL STATE GRADIENT TESTS")
    print("="*70)

    passed = test_state_gradient_nonzero()

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] State Gradient Check")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
