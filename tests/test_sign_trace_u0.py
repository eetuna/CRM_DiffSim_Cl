"""
Minimal diagnostic test for Sprint S14-Step5B: Sign trace for u[0] only.

Single actuator, single control axis perturbation to trace physical sign propagation.
"""

import sys
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/build')
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/python')

import numpy as np
import crm_diff_py
from control.true_legacy_state_adapter import pack_true_legacy_state
from control.true_legacy_step import _true_legacy_step_single

def create_minimal_test_state():
    """Create minimal test state: 1 actuator, u[0] perturbed only."""
    n_act = 1

    # Single coil state (18 DOF)
    x_coil = np.zeros((n_act, 18))
    # p_L at origin
    x_coil[:, 6:9] = [[0.0, 0.0, 0.0]]
    # R_coil = I (identity)
    x_coil[:, 9:18] = np.eye(3).flatten()

    # Fixture state (15 DOF)
    xf = np.zeros(15)
    # p_f at some offset
    xf[0:3] = [0.0, 0.0, 100.0]
    # R_f = I
    xf[3:12] = np.eye(3).flatten()

    x = pack_true_legacy_state(x_coil, xf)

    # Control: u[0] and u[1] positive, u[2] negative
    u = np.array([[0.1, 0.05, -0.05]])

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

def test_sign_trace():
    """Trace sign propagation through u → residual chain."""
    print("="*80)
    print("Sprint S14-Step5B: Sign Trace Diagnostic")
    print("="*80)
    print()

    n_act = 1
    x, u, dt, params_dict, n_act = create_minimal_test_state()

    print(f"Configuration:")
    print(f"  n_act = {n_act}")
    print(f"  u = {u.flatten()}")
    print(f"  dt = {dt}")
    print()

    # Define simple loss focusing on tip position
    def loss_fn(u_test):
        x_next, obs = _true_legacy_step_single(
            x, u_test, dt, n_act, params_dict, warmstart=None, return_orientation=False
        )
        tip_p = obs['tip_p']
        return np.sum(tip_p ** 2)

    # Compute baseline loss
    L0 = loss_fn(u)
    print(f"Baseline loss L0 = {L0:.6e}")
    print()

    # Expected behavior when u[0] increases:
    print("Expected Physical Behavior:")
    print("  Increasing u[0] → Increases magnetic moment magnitude along coil axis")
    print("  Magnetic torque Tb ∝ mu × B0")
    print("  Net torque → angular acceleration → R_coil update")
    print("  R_coil affects p_L via dynamics")
    print("  Residual r = p_f - p_L, so dL/du[0] sign depends on whether")
    print("    tip motion (driven by u[0]) moves toward or away from target")
    print()

    # Compute VJP gradient
    x_coil = x[:n_act*18].reshape(n_act, 18)
    xf = x[n_act*18:]

    fwd_result = crm_diff_py.true_legacy_step_forward(
        x_coil, xf, u, dt, params_dict
    )

    tip_p = fwd_result['tip_p']
    print(f"Forward pass tip_p = {tip_p}")

    # Upstream gradient for loss L = ||tip_p||^2
    grad_tip_p = 2.0 * tip_p
    print(f"Upstream grad_tip_p = {grad_tip_p}")
    print()

    # VJP
    vjp_result = crm_diff_py.true_legacy_step_vjp(
        x_coil, xf, u, dt, params_dict['L_inserted'], params_dict, grad_tip_p
    )

    grad_u_vjp = vjp_result['grad_u']

    # Finite difference for each component
    eps = 1e-5
    grad_u_fd = np.zeros_like(u)

    print("Finite Difference Gradient Computation:")
    for i in range(n_act):
        for j in range(3):
            u_pert = u.copy()
            u_pert[i, j] += eps
            L_pert = loss_fn(u_pert)
            grad_u_fd[i, j] = (L_pert - L0) / eps
            print(f"  u[{i},{j}]: (L_pert - L0)/eps = ({L_pert:.6e} - {L0:.6e})/{eps} = {grad_u_fd[i, j]:.6e}")
    print()

    # Compare
    print("="*80)
    print("GRADIENT COMPARISON:")
    print("="*80)
    print(f"VJP gradient:   {grad_u_vjp.flatten()}")
    print(f"FD gradient:    {grad_u_fd.flatten()}")
    print(f"Difference:     {(grad_u_vjp - grad_u_fd).flatten()}")
    print()

    # Component-wise analysis
    for j in range(3):
        vjp_val = grad_u_vjp[0, j]
        fd_val = grad_u_fd[0, j]
        sign_match = (np.sign(vjp_val) == np.sign(fd_val))
        abs_err = np.abs(vjp_val - fd_val)
        rel_err = abs_err / (np.abs(fd_val) + 1e-10)

        status = "✓ SIGN OK" if sign_match else "✗ SIGN MISMATCH"
        print(f"u[0,{j}]: VJP={vjp_val:+.6e}, FD={fd_val:+.6e}, RelErr={rel_err:.6e} {status}")

    print()

    rel_err = np.linalg.norm(grad_u_vjp - grad_u_fd) / (np.linalg.norm(grad_u_fd) + 1e-10)
    print(f"Overall relative error: {rel_err:.6e}")
    print()

    if rel_err < 1e-3:
        print("✓ PASS: VJP gradients match finite differences")
        return True
    else:
        print("✗ FAIL: VJP gradients DO NOT match finite differences")
        print()
        print("Next step: Instrument C++ code to trace sign propagation")
        return False

if __name__ == "__main__":
    try:
        success = test_sign_trace()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
