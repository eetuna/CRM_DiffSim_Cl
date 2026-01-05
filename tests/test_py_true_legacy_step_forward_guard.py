"""
Test python binding guard for non-converged BVP.
"""

import sys
sys.path.insert(0, "/workspaces/CRM_DiffSim_Cl/build")
sys.path.insert(0, "/workspaces/CRM_DiffSim_Cl/python")

import numpy as np
import crm_diff_py


def test_py_true_legacy_step_forward_guard():
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

    mL_guess = np.full((n_act, 3), np.nan, dtype=np.float64)
    nL_guess = np.full((n_act, 3), np.nan, dtype=np.float64)

    result = crm_diff_py.true_legacy_step_forward(
        x_coil, xf, u, 0.001, params_dict, mL_guess, nL_guess
    )

    assert result["converged"] is False
    assert np.allclose(result["x_coil_next"], x_coil, atol=0.0, rtol=0.0, equal_nan=True)
    assert np.allclose(result["xf_next"], xf, atol=0.0, rtol=0.0, equal_nan=True)
