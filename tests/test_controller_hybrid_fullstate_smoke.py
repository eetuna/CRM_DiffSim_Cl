"""
FULLSTATE Hybrid controller smoke test (end-to-end).
"""

import os
import sys
import numpy as np
import torch

repo_root = os.path.join(os.path.dirname(__file__), '..')
build_dir = os.path.join(repo_root, 'build_s15')
if not os.path.isdir(build_dir):
    build_dir = os.path.join(repo_root, 'build')
sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(repo_root, 'python'))

import crm_diff_py
from control.hybrid_controller import HybridController
from control.true_legacy_state_adapter import pack_true_legacy_state, unpack_true_legacy_state
from models.ensemble_policy import EnsemblePolicy
from models.recurrent_policy import GRUPolicy


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


def make_ensemble(tmp_path):
    torch.manual_seed(0)
    policy_paths = []
    for idx in range(2):
        torch.manual_seed(100 + idx)
        policy = GRUPolicy()
        policy_path = tmp_path / f"policy_{idx}.pth"
        torch.save(policy.state_dict(), policy_path)
        policy_paths.append(str(policy_path))

    return EnsemblePolicy(policy_paths, GRUPolicy, policy_kwargs={}, device='cpu')


def test_controller_hybrid_fullstate_smoke(tmp_path):
    np.random.seed(0)
    n_act = 1
    dt = 0.01

    params_dict = make_params_dict()
    x0, p_tip_0 = make_initial_state(params_dict, n_act)

    ensemble = make_ensemble(tmp_path)
    controller = HybridController(
        ensemble=ensemble,
        dt=dt,
        L_inserted=100.0,
        params_dict=params_dict,
        n_act=n_act,
        tau_low=1e6,
        tau_high=1e6,
        tracking_safety_limit=1e6,
        mpc_horizon=6,
        mpc_max_iters=1,
        mpc_tol=1e-2,
        dynamics_safety_checks=True,
        control_limits=(-0.05, 0.05),
    )

    p_ref_horizon = np.tile(p_tip_0 + np.array([1.0, 0.0, 0.0]), (6, 1))

    u_t, info = controller.step(x0, p_tip_0, p_ref_horizon)
    assert u_t.shape == (3 * n_act,)
    assert np.all(np.isfinite(u_t)), "u_t contains NaN/Inf"
    assert 'mode' in info

    x_coil_0, xf_0 = unpack_true_legacy_state(x0, n_act)
    result = crm_diff_py.true_legacy_step_forward(
        x_coil_0, xf_0, u_t.reshape(n_act, 3), dt, params_dict
    )
    assert result['converged'], "Dynamics step did not converge"

    x_next = pack_true_legacy_state(result['x_coil_next'], result['xf_next'])
    assert np.linalg.norm(x_next - x0) > 0.0, "State did not advance"
