"""
Test FULLSTATE VJP gradient correctness for control inputs using finite differences.

Verifies that ∂L/∂u computed via VJP matches finite difference approximation.
"""

import sys
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/build')
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/python')

import numpy as np
import crm_diff_py
from control.true_legacy_state_adapter import pack_true_legacy_state
from control.true_legacy_step import _true_legacy_step_single

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
    """Verify ∂L/∂u using finite differences."""
    n_act = 1
    x, u, dt, params_dict, n_act = create_test_state(n_act)

    # Define simple scalar loss: L = ||tip_p||^2
    def loss_fn(u_test):
        x_next, obs = _true_legacy_step_single(
            x, u_test, dt, n_act, params_dict, warmstart=None, return_orientation=False
        )
        tip_p = obs['tip_p']
        return np.sum(tip_p ** 2)

    # Compute loss at u
    L0 = loss_fn(u)

    # Compute gradient via VJP (using batched)
    x_coil = x[:n_act*18].reshape(n_act, 18)
    xf = x[n_act*18:]

    # Forward pass
    fwd_result = crm_diff_py.true_legacy_step_forward(
        x_coil, xf, u, dt, params_dict
    )

    tip_p = fwd_result['tip_p']

    # Upstream gradient: dL/d(tip_p) = 2 * tip_p
    grad_tip_p = 2.0 * tip_p

    # VJP
    vjp_result = crm_diff_py.true_legacy_step_vjp(
        x_coil, xf, u, dt, params_dict['L_inserted'], params_dict, grad_tip_p
    )

    grad_u_vjp = vjp_result['grad_u']

    # Compute gradient via finite differences
    eps = 1e-5
    grad_u_fd = np.zeros_like(u)

    for i in range(n_act):
        for j in range(3):
            u_pert = u.copy()
            u_pert[i, j] += eps
            L_pert = loss_fn(u_pert)
            grad_u_fd[i, j] = (L_pert - L0) / eps

    # Compare
    rel_err = np.linalg.norm(grad_u_vjp - grad_u_fd) / (np.linalg.norm(grad_u_fd) + 1e-10)

    print(f"VJP gradient:   {grad_u_vjp.flatten()}")
    print(f"FD gradient:    {grad_u_fd.flatten()}")
    print(f"Relative error: {rel_err:.6e}")

    # Should match within reasonable tolerance
    assert rel_err < 1e-3, f"Gradient mismatch: {rel_err}"

    print("PASS: VJP gradients match finite differences")
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
