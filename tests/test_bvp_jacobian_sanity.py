"""
Sanity checks for FULLSTATE BVP Jacobians.
"""

import sys
sys.path.insert(0, "/workspaces/CRM_DiffSim_Cl/build")
sys.path.insert(0, "/workspaces/CRM_DiffSim_Cl/python")

import numpy as np
import crm_diff_py


def test_bvp_jacobians_sanity():
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

    fwd = crm_diff_py.true_legacy_step_forward(
        x_coil, xf, u, 0.001, params_dict
    )

    mL = fwd["mL_next"]
    nL = fwd["nL_next"]

    jac = crm_diff_py.bvp_jacobians_fullstate(
        mL, nL, x_coil, xf, u, 0.001, params_dict["L_inserted"], params_dict
    )

    J_yy = jac["J_yy"]
    J_yx = jac["J_yx"]

    assert np.linalg.norm(J_yx) > 0.0

    rhs = np.arange(J_yy.shape[0], dtype=np.float64)
    sol = np.linalg.lstsq(J_yy, rhs, rcond=None)[0]
    residual = J_yy @ sol - rhs

    assert np.all(np.isfinite(residual))
