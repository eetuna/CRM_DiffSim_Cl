"""
Test FULLSTATE VJP correctness by comparing against Linearization (A, B).

Consistency Check:
VJP(v_xf_next) should equal [A^T * v; B^T * v] roughly, 
where v is mapped to the full state space.

Wait, VJP input is grad_tip_p (3 dim).
Linearization output is full state (18N + 15).
We need to map grad_tip_p to v_xf_next (zeros everywhere except tip p).
Then VJP output grad_x_coil, grad_xf, grad_u should match:
[grad_x; grad_u] = [A^T; B^T] * v_full_state
where v_full_state corresponds to grad_tip_p placed in xf_next positions.
"""

import sys
import os
import numpy as np
import pytest

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../build'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../python'))

import crm_diff_py

def create_test_case():
    n_act = 1
    x_coil = np.zeros((n_act, 18))
    # Identity orientation, zero position/velocity
    x_coil[:, 9:18] = np.eye(3).flatten()
    
    xf = np.zeros(15)
    xf[0:3] = [0.0, 0.0, 100.0]
    xf[3:12] = np.eye(3).flatten()
    
    u = np.array([[0.1, 0.05, -0.05]] * n_act)
    
    # Load params
    params = crm_diff_py.load_cath_params('./catheterdata/CatheterParameterSet_1_dyn.txt')
    config = crm_diff_py.load_cath_config('./catheterdata/CatheterSpatialConfiguration_1.txt')
    
    params_dict = {
        'CathParams': params,
        'CathConfig': config,
        'L_inserted': 100.0,
        'ContactMode': crm_diff_py.ContactModeType.FREE_TIP,
        'TipConstraintPoint': np.zeros(3),
        'TipForce': np.zeros(3),
        'IntegrationStepSize': 0.1 # Coarse for speed
    }
    
    dt = 0.01
    return x_coil, xf, u, dt, params_dict

def test_vjp_consistency_with_linearization():
    x_coil, xf, u, dt, params_dict = create_test_case()
    n_act = x_coil.shape[0]
    
    # 1. Linearize
    A, B = crm_diff_py.true_legacy_linearize(
        x_coil, xf, u, dt, params_dict
    )
    
    # A: (18N+15) x (18N+15)
    # B: (18N+15) x (3N)
    
    # 2. VJP
    # VJP takes grad_tip_p (size 3).
    # This corresponds to a cotangent vector v on x_next where only the first 3 components 
    # of the TIP PART (xf_next) are non-zero.
    # The state vector structure in Linearization is [x_coil (18N); xf (15)].
    # xf structure is [p (3), R (9), u (3)].
    # So tip_p corresponds to indices 18N + 0, 18N + 1, 18N + 2.
    
    grad_tip_p = np.array([1.0, -0.5, 0.2])
    
    v_full = np.zeros(n_act * 18 + 15)
    v_full[n_act*18 + 0] = grad_tip_p[0]
    v_full[n_act*18 + 1] = grad_tip_p[1]
    v_full[n_act*18 + 2] = grad_tip_p[2]
    
    # Expected gradients from Linearization
    # grad_x = A^T * v
    # grad_u = B^T * v
    
    grad_x_expected = A.T @ v_full
    grad_u_expected = B.T @ v_full
    
    # Compute via VJP
    vjp_out = crm_diff_py.true_legacy_step_vjp(
        x_coil, xf, u, dt, params_dict['L_inserted'], params_dict, grad_tip_p
    )
    
    grad_x_coil_vjp = vjp_out['grad_x_coil']
    grad_xf_vjp = vjp_out['grad_xf']
    grad_u_vjp = vjp_out['grad_u']
    
    # Pack VJP outputs into full vector
    grad_x_vjp_packed = np.concatenate([grad_x_coil_vjp.flatten(), grad_xf_vjp])
    grad_u_vjp_packed = grad_u_vjp.flatten()
    
    # Compare
    # Tolerance: Since both use exact analytic methods, they should match very closely (machine precision)
    # unless there are implementation divergences.
    
    err_x = np.linalg.norm(grad_x_vjp_packed - grad_x_expected)
    err_u = np.linalg.norm(grad_u_vjp_packed - grad_u_expected)
    
    print(f"Error grad_x: {err_x}")
    print(f"Error grad_u: {err_u}")
    
    assert err_x < 1e-10
    assert err_u < 1e-10

if __name__ == "__main__":
    test_vjp_consistency_with_linearization()