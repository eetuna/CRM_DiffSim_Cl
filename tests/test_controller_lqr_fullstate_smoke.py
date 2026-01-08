"""
FULLSTATE LQR controller smoke test (end-to-end).
"""

import os
import sys
import numpy as np

repo_root = os.path.join(os.path.dirname(__file__), '..')
build_dir = os.path.join(repo_root, 'build_s15')
if not os.path.isdir(build_dir):
    build_dir = os.path.join(repo_root, 'build')
sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(repo_root, 'python'))

import crm_diff_py
from control.lqr import finite_horizon_lqr
from control.true_legacy_state_adapter import pack_true_legacy_state


def make_params_dict():
    params = crm_diff_py.load_cath_params('./catheterdata/CatheterParameterSet_1_dyn.txt')
    config = crm_diff_py.load_cath_config('./catheterdata/CatheterSpatialConfiguration_1.txt')
    return {
        'CathParams': params,
        'CathConfig': config,
        'L_inserted': 100.0,
        'ContactMode': crm_diff_py.ContactModeType.FREE_TIP,
        'TipConstraintPoint': np.zeros(3),
        'TipForce': np.zeros(3),
        'deltau0_initialguess': np.zeros(3),
        'IntegrationStepSize': 0.5,
    }


def make_initial_state(params_dict, n_act=1, L_inserted=100.0):
    u_eq = np.zeros(3 * n_act, dtype=np.float64)
    eq = crm_diff_py.equilibrium_forward(u_eq, L_inserted, params_dict)
    assert eq['converged'] == 0, "equilibrium_forward did not converge"

    x_coil = np.zeros((n_act, 18), dtype=np.float64)
    x_coil[:, 6:9] = eq['coil_p']
    x_coil[:, 9:18] = eq['coil_R']

    xf = np.zeros(15, dtype=np.float64)
    xf[0:3] = eq['p_tip']
    xf[3:12] = np.eye(3).flatten()

    x0 = pack_true_legacy_state(x_coil, xf)
    return x0, xf[:3].copy()


def test_controller_lqr_fullstate_smoke():
    np.random.seed(0)
    n_act = 1
    horizon = 1
    dt = 0.01

    params_dict = make_params_dict()
    x0, p_tip_0 = make_initial_state(params_dict, n_act)
    p_target = p_tip_0.copy()

    u_nominal = np.zeros((horizon, 3 * n_act), dtype=np.float64)
    U_lqr, X_lqr, P_tip_lqr = finite_horizon_lqr(
        x0,
        p_target,
        dt,
        100.0,
        params_dict,
        horizon,
        n_act=n_act,
        Q=None,
        R=0.01 * np.eye(3 * n_act),
        terminal_weight=10.0,
        u_nominal=u_nominal,
        verbose=False,
    )

    assert U_lqr.shape == (horizon, 3 * n_act)
    assert X_lqr.shape == (horizon + 1, 18 * n_act + 15)
    assert P_tip_lqr.shape == (horizon + 1, 3)
    assert np.all(np.isfinite(U_lqr)), "U_lqr contains NaN/Inf"
    assert np.all(np.isfinite(X_lqr)), "X_lqr contains NaN/Inf"
    assert np.all(np.isfinite(P_tip_lqr)), "P_tip_lqr contains NaN/Inf"

    cost = float(np.sum(U_lqr * U_lqr) + np.sum(P_tip_lqr[-1] ** 2))
    assert np.isfinite(cost), "Cost is not finite"

    assert np.linalg.norm(X_lqr[1] - X_lqr[0]) > 0.0, "State did not advance"
