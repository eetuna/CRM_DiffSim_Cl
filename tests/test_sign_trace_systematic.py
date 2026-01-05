"""
Systematic sign trace test: Test each control component individually with both signs.

Tests all combinations:
- u[0] only: positive and negative
- u[1] only: positive and negative
- u[2] only: positive and negative
"""

import sys
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/build')
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/python')

import numpy as np
import crm_diff_py
from control.true_legacy_state_adapter import pack_true_legacy_state
from control.true_legacy_step import _true_legacy_step_single

def create_test_state(u_vals):
    """Create test state with specified control inputs."""
    n_act = 1

    # Single coil state (18 DOF)
    x_coil = np.zeros((n_act, 18))
    x_coil[:, 6:9] = [[0.0, 0.0, 0.0]]  # p_L at origin
    x_coil[:, 9:18] = np.eye(3).flatten()  # R_coil = I

    # Fixture state (15 DOF)
    xf = np.zeros(15)
    xf[0:3] = [0.0, 0.0, 100.0]  # p_f at offset
    xf[3:12] = np.eye(3).flatten()  # R_f = I

    x = pack_true_legacy_state(x_coil, xf)

    u = np.array([u_vals])

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

def test_single_component(component_idx, sign_str, u_vals):
    """Test a single control component."""
    print(f"\n{'='*80}")
    print(f"Test: u[{component_idx}] = {sign_str}, others = 0")
    print(f"Control: u = {u_vals}")
    print('='*80)

    n_act = 1
    x, u, dt, params_dict, n_act = create_test_state(u_vals)

    # Define loss function
    def loss_fn(u_test):
        x_next, obs = _true_legacy_step_single(
            x, u_test, dt, n_act, params_dict, warmstart=None, return_orientation=False
        )
        tip_p = obs['tip_p']
        return np.sum(tip_p ** 2)

    # Baseline loss
    L0 = loss_fn(u)

    # VJP gradient
    x_coil = x[:n_act*18].reshape(n_act, 18)
    xf = x[n_act*18:]

    fwd_result = crm_diff_py.true_legacy_step_forward(
        x_coil, xf, u, dt, params_dict
    )

    tip_p = fwd_result['tip_p']
    grad_tip_p = 2.0 * tip_p

    vjp_result = crm_diff_py.true_legacy_step_vjp(
        x_coil, xf, u, dt, params_dict['L_inserted'], params_dict, grad_tip_p
    )

    grad_u_vjp = vjp_result['grad_u']

    # Finite difference gradient
    eps = 1e-5
    grad_u_fd = np.zeros_like(u)

    for i in range(n_act):
        for j in range(3):
            u_pert = u.copy()
            u_pert[i, j] += eps
            L_pert = loss_fn(u_pert)
            grad_u_fd[i, j] = (L_pert - L0) / eps

    # Compare
    print(f"\nVJP gradient:   {grad_u_vjp.flatten()}")
    print(f"FD gradient:    {grad_u_fd.flatten()}")
    print(f"\nComponent-wise analysis:")

    results = {}
    for j in range(3):
        vjp_val = grad_u_vjp[0, j]
        fd_val = grad_u_fd[0, j]
        sign_match = (np.sign(vjp_val) == np.sign(fd_val)) or (abs(fd_val) < 1e-10 and abs(vjp_val) < 1e-10)
        abs_err = np.abs(vjp_val - fd_val)
        rel_err = abs_err / (np.abs(fd_val) + 1e-10)

        status = "✓ SIGN OK" if sign_match else "✗ SIGN MISMATCH"
        print(f"  u[0,{j}]: VJP={vjp_val:+.6e}, FD={fd_val:+.6e}, RelErr={rel_err:.6e} {status}")
        results[j] = sign_match

    rel_err_overall = np.linalg.norm(grad_u_vjp - grad_u_fd) / (np.linalg.norm(grad_u_fd) + 1e-10)
    print(f"\nOverall relative error: {rel_err_overall:.6e}")

    return results

def main():
    print("="*80)
    print("SYSTEMATIC SIGN TRACE TEST")
    print("Testing each control component individually with both signs")
    print("="*80)

    # Test configurations: (component_idx, sign_str, u_values)
    test_configs = [
        (0, "+0.1", [0.1, 0.0, 0.0]),
        (0, "-0.1", [-0.1, 0.0, 0.0]),
        (1, "+0.05", [0.0, 0.05, 0.0]),
        (1, "-0.05", [0.0, -0.05, 0.0]),
        (2, "+0.05", [0.0, 0.0, 0.05]),
        (2, "-0.05", [0.0, 0.0, -0.05]),
    ]

    all_results = {}
    for comp_idx, sign_str, u_vals in test_configs:
        results = test_single_component(comp_idx, sign_str, u_vals)
        all_results[(comp_idx, sign_str)] = results

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Test':<20} {'∂L/∂u[0]':<15} {'∂L/∂u[1]':<15} {'∂L/∂u[2]':<15}")
    print("-"*80)

    for comp_idx, sign_str, u_vals in test_configs:
        test_name = f"u[{comp_idx}]={sign_str}"
        results = all_results[(comp_idx, sign_str)]
        r0 = "✓ OK" if results[0] else "✗ MISMATCH"
        r1 = "✓ OK" if results[1] else "✗ MISMATCH"
        r2 = "✓ OK" if results[2] else "✗ MISMATCH"
        print(f"{test_name:<20} {r0:<15} {r1:<15} {r2:<15}")

    print("="*80)

if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
