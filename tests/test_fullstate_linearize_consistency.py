"""
Test Linearization Consistency with VJP using random cotangents.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))

import crm_diff_py

def test_linearize_consistency_random():
    n_act = 1
    x_coil = np.zeros((n_act, 18))
    x_coil[:, 9:18] = np.eye(3).flatten()
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

    # 1. Linearize
    A, B = crm_diff_py.true_legacy_linearize(x_coil, xf, u, dt, params_dict)
    
    # 2. VJP with random vectors
    np.random.seed(42)
    
    for _ in range(5):
        grad_tip_p = np.random.randn(3)
        
        # Cotangent vector on full state x_next
        v_full = np.zeros(n_act * 18 + 15)
        # Map tip_p to xf positions (first 3 of xf part)
        v_full[n_act*18 + 0:n_act*18 + 3] = grad_tip_p
        
        # Expected
        grad_x_exp = A.T @ v_full
        grad_u_exp = B.T @ v_full
        
        # VJP
        vjp_out = crm_diff_py.true_legacy_step_vjp(
            x_coil, xf, u, dt, params_dict['L_inserted'], params_dict, grad_tip_p
        )
        
        grad_x_vjp = np.concatenate([vjp_out['grad_x_coil'].flatten(), vjp_out['grad_xf']])
        grad_u_vjp = vjp_out['grad_u'].flatten()
        
        # Check
        err_x = np.linalg.norm(grad_x_vjp - grad_x_exp)
        err_u = np.linalg.norm(grad_u_vjp - grad_u_exp)
        
        assert err_x < 1e-10
        assert err_u < 1e-10

if __name__ == "__main__":
    test_linearize_consistency_random()
