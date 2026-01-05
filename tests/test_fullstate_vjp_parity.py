"""
Test FULLSTATE VJP parity between single and batched paths.
"""

import sys
sys.path.insert(0, "/workspaces/CRM_DiffSim_Cl/build")
sys.path.insert(0, "/workspaces/CRM_DiffSim_Cl/python")

import numpy as np
import crm_diff_py


def test_vjp_single_vs_batched_parity():
    n_act = 1

    x_coil = np.zeros((n_act, 18), dtype=np.float64)
    xf = np.zeros(15, dtype=np.float64)
    x_coil[:, 9:18] = np.eye(3).flatten()
    xf[3:12] = np.eye(3).flatten()
    x_coil[:, 6:9] = [0.0, 0.0, 1.0]
    xf[:3] = [0.0, 0.0, 10.0]
    u = np.array([[0.01, 0.0, 0.0]], dtype=np.float64)

    params = crm_diff_py.load_cath_params("./catheterdata/CatheterParameterSet_1_dyn.txt")
    config = crm_diff_py.load_cath_config("./catheterdata/CatheterSpatialConfiguration_1.txt")

    params_dict = {
        "CathParams": params,
        "CathConfig": config,
        "L_inserted": 100.0,
        "ContactMode": int(crm_diff_py.ContactModeType.FREE_TIP),
        "TipForce": [0.0, 0.0, 0.0],
        "deltau0_initialguess": [0.0, 0.0, 0.0],
        "IntegrationStepSize": 0.5,
        "FinalValueOnly": True,
    }

    grad_tip_p = np.array([1.0, -0.5, 0.25], dtype=np.float64)

    single = crm_diff_py.true_legacy_step_vjp(
        x_coil, xf, u, 0.001, params_dict["L_inserted"], params_dict, grad_tip_p
    )
    batched = crm_diff_py.true_legacy_step_vjp_batched(
        x_coil, xf, u, 0.001, params_dict["L_inserted"], params_dict, grad_tip_p.reshape(1, 3)
    )

    grad_xf_single = single["grad_xf"]
    grad_x_coil_single = single["grad_x_coil"]
    grad_u_single = single["grad_u"]

    grad_xf_batch = batched["grad_xf"][0]
    grad_x_coil_batch = batched["grad_x_coil"][0]
    grad_u_batch = batched["grad_u"][0]

    atol = 1e-12
    rtol = 1e-12

    assert np.allclose(grad_xf_single, grad_xf_batch, rtol=rtol, atol=atol)
    assert np.allclose(grad_x_coil_single, grad_x_coil_batch, rtol=rtol, atol=atol)
    assert np.allclose(grad_u_single, grad_u_batch, rtol=rtol, atol=atol)
