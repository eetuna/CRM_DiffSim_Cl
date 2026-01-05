"""
Test FULLSTATE VJP gradient correctness for control inputs without finite differences.
"""

import sys
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/build')
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/python')

import numpy as np
import crm_diff_py
from control.true_legacy_state_adapter import pack_true_legacy_state, true_legacy_state_dim
from control.true_legacy_step import true_legacy_linearize

def create_test_state(n_act=1):
    """Create a simple test state."""
    x_coil = np.zeros((n_act, 18))
    x_coil[:, 6:9] = [[0.0, 0.0, 0.0]]
    x_coil[:, 9:18] = np.eye(3).flatten()

    xf = np.zeros(15)
    xf[0:3] = [0.0, 0.0, 100.0]
    xf[3:12] = np.eye(3).flatten()

    x = pack_true_legacy_state(x_coil, xf)

    u = np.array([[0.1, 0.05, -0.05]] * n_act)

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

def test_vjp_u_gradients():
    """Verify ∂L/∂u using implicit linearization consistency."""
    n_act = 1
    x, u, dt, params_dict, n_act = create_test_state(n_act)

    # Compute gradient via VJP (using batched)
    x_coil = x[:n_act*18].reshape(n_act, 18)
    xf = x[n_act*18:]

    # Forward pass
    fwd_result = crm_diff_py.true_legacy_step_forward(
        x_coil, xf, u, dt, params_dict
    )

    tip_p = fwd_result['tip_p']

    # Upstream gradient: nonzero tip position cotangent
    grad_tip_p = np.array([1.0, -0.5, 0.25], dtype=np.float64)

    # VJP (single)
    vjp_result = crm_diff_py.true_legacy_step_vjp(
        x_coil, xf, u, dt, params_dict['L_inserted'], params_dict, grad_tip_p
    )

    grad_u_vjp = vjp_result['grad_u']

    # Gradient via implicit linearization: grad_u = B^T * v_xf_next
    A, B = true_legacy_linearize(
        x, u, dt,
        n_act=n_act,
        catheter_params=params_dict,
        L_inserted=params_dict['L_inserted'],
        method="implicit"
    )

    state_dim = true_legacy_state_dim(n_act)
    v_xf_next = np.zeros(state_dim, dtype=np.float64)
    v_xf_next[n_act * 18:n_act * 18 + 3] = grad_tip_p
    grad_u_linearize = (B.T @ v_xf_next).reshape(n_act, 3)

    # Compare
    rel_err = np.linalg.norm(grad_u_vjp - grad_u_linearize) / (np.linalg.norm(grad_u_linearize) + 1e-10)

    print(f"VJP gradient:        {grad_u_vjp.flatten()}")
    print(f"Linearize gradient:  {grad_u_linearize.flatten()}")
    print(f"Relative error: {rel_err:.6e}")

    # Should match within reasonable tolerance
    assert rel_err < 1e-10, f"Gradient mismatch: {rel_err}"

    print("PASS: VJP gradients match implicit linearization")
    return True

if __name__ == "__main__":
    try:
        print("="*60)
        print("Test: VJP gradient check for control inputs")
        print("="*60)
        test_vjp_u_gradients()
        print()
        print("="*60)
        print("ALL TESTS PASSED")
        print("="*60)
        sys.exit(0)
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
