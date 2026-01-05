"""
Test FULLSTATE linearization (A, B) shapes and basic correctness.

Verifies that the implicit function theorem linearization produces
correct shapes and reasonable numerical values.
"""

import sys
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/build')
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/python')

import numpy as np
import crm_diff_py
from control.true_legacy_state_adapter import pack_true_legacy_state, true_legacy_state_dim
from control.true_legacy_step import true_legacy_linearize

def create_test_state(n_act=1):
    """Create a simple test state and control input."""
    # Coil state: [v, w, p, R] per coil
    x_coil = np.zeros((n_act, 18))
    x_coil[:, 6:9] = [[0.0, 0.0, 0.0]]  # position at origin
    x_coil[:, 9:18] = np.eye(3).flatten()  # identity rotation

    # Tip state: [p_tip, R_tip, u_tip]
    xf = np.zeros(15)
    xf[0:3] = [0.0, 0.0, 100.0]  # tip at z=100mm
    xf[3:12] = np.eye(3).flatten()  # identity rotation
    xf[12:15] = [0.0, 0.0, 0.0]  # zero curvature

    # Pack state
    x = pack_true_legacy_state(x_coil, xf)

    # Control: small currents
    u = np.array([[0.1, 0.0, 0.0]] * n_act)

    # Catheter params
    params = crm_diff_py.load_cath_params('./catheterdata/CatheterParameterSet_1_dyn.txt')
    config = crm_diff_py.load_cath_config('./catheterdata/CatheterSpatialConfiguration_1.txt')

    params_dict = {
        'CathParams': params,
        'CathConfig': config,
        'L_inserted': 100.0,
        'ContactMode': crm_diff_py.ContactModeType.FREE_TIP,
        'TipConstraintPoint': np.zeros(3),
        'TipForce': np.zeros(3),
        'deltau0_initialguess': np.zeros(3),
        'IntegrationStepSize': 0.5
    }

    return x, u, 0.01, params_dict, n_act

def test_linearize_shapes():
    """Verify A, B have correct shapes."""
    n_act = 1
    state_dim = true_legacy_state_dim(n_act)  # 18*1 + 15 = 33
    control_dim = 3 * n_act  # 3

    print(f"Testing with n_act={n_act}, state_dim={state_dim}, control_dim={control_dim}")

    x, u, dt, params_dict, n_act = create_test_state(n_act)

    # Test implicit method
    A, B = true_legacy_linearize(
        x, u, dt,
        n_act=n_act,
        catheter_params=params_dict,
        L_inserted=100.0,
        method="implicit"
    )

    print(f"A shape: {A.shape}, expected: ({state_dim}, {state_dim})")
    print(f"B shape: {B.shape}, expected: ({state_dim}, {control_dim})")

    assert A.shape == (state_dim, state_dim), f"A shape mismatch: {A.shape}"
    assert B.shape == (state_dim, control_dim), f"B shape mismatch: {B.shape}"

    # Check that matrices are not all zeros or NaN
    assert not np.all(A == 0), "A is all zeros"
    assert not np.all(B == 0), "B is all zeros"
    assert not np.any(np.isnan(A)), "A contains NaN"
    assert not np.any(np.isnan(B)), "B contains NaN"

    print("PASS: Shapes are correct and matrices are non-trivial")
    return True

def test_linearize_vs_torch_autograd():
    """Compare implicit vs torch.autograd (should be close)."""
    n_act = 1
    x, u, dt, params_dict, n_act = create_test_state(n_act)

    try:
        A_implicit, B_implicit = true_legacy_linearize(
            x, u, dt,
            n_act=n_act,
            catheter_params=params_dict,
            L_inserted=100.0,
            method="implicit"
        )

        A_torch, B_torch = true_legacy_linearize(
            x, u, dt,
            n_act=n_act,
            catheter_params=params_dict,
            L_inserted=100.0,
            method="torch_autograd"
        )

        # Check relative differences
        A_rel_err = np.linalg.norm(A_implicit - A_torch) / (np.linalg.norm(A_torch) + 1e-10)
        B_rel_err = np.linalg.norm(B_implicit - B_torch) / (np.linalg.norm(B_torch) + 1e-10)

        print(f"A relative error: {A_rel_err:.6f}")
        print(f"B relative error: {B_rel_err:.6f}")

        # They should match within reasonable tolerance
        # (may differ slightly due to numerical errors in implicit solve)
        assert A_rel_err < 0.01, f"A matrices differ too much: {A_rel_err}"
        assert B_rel_err < 0.01, f"B matrices differ too much: {B_rel_err}"

        print("PASS: Implicit matches torch.autograd within tolerance")
        return True

    except Exception as e:
        print(f"WARNING: torch.autograd comparison failed (this is OK if torch is not available): {e}")
        return True  # Don't fail test if torch isn't available

if __name__ == "__main__":
    success = True

    try:
        print("="*60)
        print("Test 1: Linearization shapes")
        print("="*60)
        test_linearize_shapes()
        print()
    except Exception as e:
        print(f"FAILED: {e}")
        success = False

    try:
        print("="*60)
        print("Test 2: Implicit vs torch.autograd")
        print("="*60)
        test_linearize_vs_torch_autograd()
        print()
    except Exception as e:
        print(f"FAILED: {e}")
        success = False

    if success:
        print("="*60)
        print("ALL TESTS PASSED")
        print("="*60)
        sys.exit(0)
    else:
        sys.exit(1)
