"""
Test FULLSTATE linearization consistency with VJP.
"""

import sys
sys.path.insert(0, "/workspaces/CRM_DiffSim_Cl/build")
sys.path.insert(0, "/workspaces/CRM_DiffSim_Cl/python")

import numpy as np
import crm_diff_py
from control.true_legacy_state_adapter import pack_true_legacy_state, true_legacy_state_dim
from control.true_legacy_step import true_legacy_linearize


def create_test_state(n_act=1):
    x_coil = np.zeros((n_act, 18), dtype=np.float64)
    x_coil[:, 6:9] = [[0.0, 0.0, 0.0]]
    x_coil[:, 9:18] = np.eye(3).flatten()

    xf = np.zeros(15, dtype=np.float64)
    xf[0:3] = [0.0, 0.0, 100.0]
    xf[3:12] = np.eye(3).flatten()

    x = pack_true_legacy_state(x_coil, xf)
    u = np.array([[0.1, 0.0, 0.0]] * n_act, dtype=np.float64)

    params = crm_diff_py.load_cath_params("./catheterdata/CatheterParameterSet_1_dyn.txt")
    config = crm_diff_py.load_cath_config("./catheterdata/CatheterSpatialConfiguration_1.txt")

    params_dict = {
        "CathParams": params,
        "CathConfig": config,
        "L_inserted": 100.0,
        "ContactMode": crm_diff_py.ContactModeType.FREE_TIP,
        "TipConstraintPoint": np.zeros(3),
        "TipForce": np.zeros(3),
        "deltau0_initialguess": np.zeros(3),
        "IntegrationStepSize": 0.5,
    }

    return x, x_coil, xf, u, 0.01, params_dict


def test_linearize_matches_vjp_tip_p():
    n_act = 1
    x, x_coil, xf, u, dt, params_dict = create_test_state(n_act)

    grad_tip_p = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    vjp_result = crm_diff_py.true_legacy_step_vjp(
        x_coil, xf, u, dt, params_dict["L_inserted"], params_dict, grad_tip_p
    )

    grad_x_coil_vjp = vjp_result["grad_x_coil"].reshape(n_act * 18)
    grad_xf_vjp = vjp_result["grad_xf"]
    grad_u_vjp = vjp_result["grad_u"].reshape(n_act * 3)

    A, B = true_legacy_linearize(
        x, u, dt,
        n_act=n_act,
        catheter_params=params_dict,
        L_inserted=params_dict["L_inserted"],
        method="implicit"
    )

    state_dim = true_legacy_state_dim(n_act)
    v_xf_next = np.zeros(state_dim, dtype=np.float64)
    v_xf_next[n_act * 18:n_act * 18 + 3] = grad_tip_p

    grad_x_linear = A.T @ v_xf_next
    grad_u_linear = B.T @ v_xf_next

    grad_x_vjp = np.concatenate([grad_x_coil_vjp, grad_xf_vjp])

    atol = 1e-10
    rtol = 1e-10

    assert np.allclose(grad_x_linear, grad_x_vjp, rtol=rtol, atol=atol)
    assert np.allclose(grad_u_linear, grad_u_vjp, rtol=rtol, atol=atol)
