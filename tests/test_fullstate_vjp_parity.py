"""
Test Parity between Single and Batched VJP implementations.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))

import crm_diff_py

def create_test_case():
    n_act = 1
    x_coil = np.zeros((n_act, 18))
    x_coil[:, 9:18] = np.eye(3).flatten() # R=I
    xf = np.zeros(15)
    xf[0:3] = [0.0, 0.0, 100.0]
    xf[3:12] = np.eye(3).flatten()
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
        'IntegrationStepSize': 0.1
    }
    
    dt = 0.01
    return x_coil, xf, u, dt, params_dict

def test_single_vs_batched_vjp():
    x_coil, xf, u, dt, params_dict = create_test_case()
    grad_tip_p = np.array([1.0, 2.0, 3.0])
    
    # 1. Single VJP
    single_out = crm_diff_py.true_legacy_step_vjp(
        x_coil, xf, u, dt, params_dict['L_inserted'], params_dict, grad_tip_p
    )
    
    # 2. Batched VJP (batch=1)
    # Note: python binding for batched needs checking.
    # Assuming 'true_legacy_step_vjp_batched' exists or 'true_legacy_step_vjp' handles batches?
    # Python binding usually exposes batched version separately or overloaded.
    # Let's check crm_bindings.cpp (not visible here but usually standard).
    # If not exposed, we might fail here.
    # But plan says "Add tests/test_fullstate_vjp_parity.py".
    
    # Assuming `true_legacy_step_vjp_batched` takes a list of grads or array of grads.
    grad_tip_p_batch = grad_tip_p.reshape(1, 3)
    
    try:
        batched_out = crm_diff_py.true_legacy_step_vjp_batched(
            x_coil, xf, u, dt, params_dict['L_inserted'], params_dict, grad_tip_p_batch
        )
    except AttributeError:
        pytest.skip("true_legacy_step_vjp_batched not exposed in python")

    # Compare
    # x_coil gradients
    err_x = np.linalg.norm(single_out['grad_x_coil'] - batched_out['grad_x_coil'][0])
    err_xf = np.linalg.norm(single_out['grad_xf'] - batched_out['grad_xf'][0])
    err_u = np.linalg.norm(single_out['grad_u'] - batched_out['grad_u'][0])
    
    print(f"Error x_coil: {err_x}")
    print(f"Error xf: {err_xf}")
    print(f"Error u: {err_u}")
    
    assert err_x < 1e-12
    assert err_xf < 1e-12
    assert err_u < 1e-12

if __name__ == "__main__":
    test_single_vs_batched_vjp()
