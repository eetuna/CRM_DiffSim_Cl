"""
Test BVP Jacobian Sanity via Linearization properties.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))

import crm_diff_py

def test_jacobian_sanity():
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

    A, B = crm_diff_py.true_legacy_linearize(x_coil, xf, u, dt, params_dict)
    
    # A should capture dynamics: x_{t+1} approx x_t + dt * f(x_t)
    # So A approx I + dt * J_f
    # It should not be exactly Identity (implies no dynamics)
    diff_I = np.linalg.norm(A - np.eye(A.shape[0]))
    print(f"Norm(A - I): {diff_I}")
    assert diff_I > 1e-5, "A matrix implies trivial dynamics (A approx I)"
    
    # B should be non-zero (control affects state)
    norm_B = np.linalg.norm(B)
    print(f"Norm(B): {norm_B}")
    assert norm_B > 1e-5, "B matrix is zero (no control authority)"

if __name__ == "__main__":
    test_jacobian_sanity()
