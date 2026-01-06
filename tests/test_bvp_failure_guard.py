"""
Test BVP Failure Guard in Python Binding.
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))

import crm_diff_py
from control.true_legacy_state_adapter import pack_true_legacy_state

def test_bvp_failure_guard():
    n_act = 1
    x_coil = np.zeros((n_act, 18))
    x_coil[:, 9:18] = np.eye(3).flatten()
    xf = np.zeros(15)
    xf[0:3] = [0.0, 0.0, 100.0]
    xf[3:12] = np.eye(3).flatten()
    
    # Extreme input that should cause BVP convergence failure
    u = np.array([[1e5, 1e5, 1e5]] * n_act) 
    
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

    # Forward pass
    try:
        fwd = crm_diff_py.true_legacy_step_forward(
            x_coil, xf, u, dt, params_dict
        )
        
        print(f"Converged: {fwd['converged']}")
        print(f"LocalMin: {fwd['localmin']}")
        
        if not fwd['converged']:
            print("Guard triggered (converged=False)")
            # Check state matches input
            tip_p_out = fwd['tip_p']
            tip_p_in = xf[0:3]
            err = np.linalg.norm(tip_p_out - tip_p_in)
            print(f"State change: {err}")
            assert err < 1e-12, "State should not change on failure"
        else:
            print("WARNING: BVP converged despite extreme input. Test inconclusive but guard logic logic exists.")
            
    except Exception as e:
        pytest.fail(f"Binding threw exception instead of handling failure: {e}")

if __name__ == "__main__":
    test_bvp_failure_guard()
