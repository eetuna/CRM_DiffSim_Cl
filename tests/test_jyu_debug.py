"""
Debug test to check J_yu values directly.
"""

import sys
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/build')
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/python')

import numpy as np
import crm_diff_py

def test_jyu_direct():
    """Test J_yu computation directly."""
    n_act = 1

    # Create simple test state
    x_coil = np.zeros((n_act, 18))
    x_coil[:, 6:9] = [[0.0, 0.0, 0.0]]
    x_coil[:, 9:18] = np.eye(3).flatten()

    xf = np.zeros(15)
    xf[0:3] = [0.0, 0.0, 100.0]
    xf[3:12] = np.eye(3).flatten()

    u = np.array([[0.1, 0.05, -0.05]] * n_act)
    dt = 0.01

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

    # Forward pass to get BVP solution
    fwd_result = crm_diff_py.true_legacy_step_forward(
        x_coil, xf, u, dt, params_dict
    )

    print("Forward pass completed")
    print(f"tip_p: {fwd_result['tip_p']}")

    # Check if J_yu is available
    if 'J_yu' in fwd_result:
        J_yu = fwd_result['J_yu']
        print(f"\nJ_yu shape: {J_yu.shape}")
        print(f"J_yu norm: {np.linalg.norm(J_yu)}")
        print(f"J_yu nonzero entries: {np.count_nonzero(J_yu)}")
        print(f"J_yu max abs value: {np.max(np.abs(J_yu))}")
        print(f"\nJ_yu sample values (first 3x3):\n{J_yu[:3, :3]}")
    else:
        print("J_yu not in forward result")

    # Now check VJP
    grad_tip_p = np.ones(3)
    vjp_result = crm_diff_py.true_legacy_step_vjp(
        x_coil, xf, u, dt, params_dict['L_inserted'], params_dict, grad_tip_p
    )

    grad_u = vjp_result['grad_u']
    print(f"\nVJP grad_u: {grad_u}")
    print(f"VJP grad_u norm: {np.linalg.norm(grad_u)}")

    return True

if __name__ == "__main__":
    test_jyu_direct()
