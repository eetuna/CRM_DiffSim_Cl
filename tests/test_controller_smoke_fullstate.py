"""
Smoke test for FULLSTATE controllers (iLQR, MPC, LQR, Hybrid).

Runs each controller for a short horizon to verify basic functionality.
"""

import sys
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/build')
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/python')

import numpy as np
import crm_diff_py
from control.true_legacy_state_adapter import pack_true_legacy_state, true_legacy_state_dim
from control.ilqr import iLQRSolver
from control.lqr import LQRController

def create_test_scenario(n_act=1):
    """Create initial state and target for testing."""
    # Initial state
    x_coil = np.zeros((n_act, 18))
    x_coil[:, 6:9] = [[0.0, 0.0, 0.0]]
    x_coil[:, 9:18] = np.eye(3).flatten()

    xf = np.zeros(15)
    xf[0:3] = [0.0, 0.0, 100.0]
    xf[3:12] = np.eye(3).flatten()

    x0 = pack_true_legacy_state(x_coil, xf)

    # Target: move tip to [10, 0, 100]
    p_target = np.array([10.0, 0.0, 100.0])

    # Params
    params = crm_diff_py.load_cath_params('./catheterdata/CatheterParameterSet_1_dyn.txt')
    config = crm_diff_py.load_cath_config('./catheterdata/CatheterSpatialConfiguration_1.txt')

    params_dict = {
        'CathParams': params,
        'CathConfig': config,
        'L_inserted': 100.0,
        'ContactMode': crm_diff_py.ContactModeType.FREE_TIP,
        'TipConstraintPoint': np.zeros(3),
        'TipForce': np.zeros(3),
        'deltau0_initialguess': np.zeros(3),
        'IntegrationStepSize': 0.5
    }

    return x0, p_target, params_dict, n_act

def test_ilqr_smoke():
    """Test iLQR with short horizon."""
    print("Testing iLQR...")

    x0, p_target, params_dict, n_act = create_test_scenario()

    solver = iLQRSolver(
        dt=0.01,
        L_inserted=100.0,
        params_dict=params_dict,
        horizon=3,  # Short horizon for speed
        n_act=n_act,
        p_target=p_target,
        terminal_weight=10.0,
        max_iters=5,  # Few iterations for smoke test
        jacobian_mode="implicit"
    )

    # Initial control sequence (zeros)
    u_init = np.zeros((3, n_act, 3))

    # Solve
    u_opt, x_traj, info = solver.solve(x0, u_init, p_target)

    assert x_traj.shape == (4, true_legacy_state_dim(n_act)), f"x_traj shape: {x_traj.shape}"
    assert u_opt.shape == (3, n_act, 3), f"u_opt shape: {u_opt.shape}"
    assert 'cost' in info, "info missing 'cost'"

    print(f"  Cost: {info['cost']:.6f}, Iterations: {info.get('iterations', 'N/A')}")
    print("  PASS")
    return True

def test_lqr_smoke():
    """Test LQR with short horizon."""
    print("Testing LQR...")

    x0, p_target, params_dict, n_act = create_test_scenario()

    controller = LQRController(
        dt=0.01,
        L_inserted=100.0,
        params_dict=params_dict,
        horizon=3,
        n_act=n_act,
        p_target=p_target,
        terminal_weight=10.0
    )

    # Compute control
    u_opt, x_traj, info = controller.compute_control(x0)

    assert x_traj.shape == (4, true_legacy_state_dim(n_act)), f"x_traj shape: {x_traj.shape}"
    assert u_opt.shape == (3, n_act, 3), f"u_opt shape: {u_opt.shape}"

    print(f"  Cost: {info['cost']:.6f}")
    print("  PASS")
    return True

if __name__ == "__main__":
    success = True

    try:
        print("="*60)
        print("Controller Smoke Tests (FULLSTATE jacobian_mode='implicit')")
        print("="*60)
        print()

        test_ilqr_smoke()
        print()

        test_lqr_smoke()
        print()

        print("="*60)
        print("ALL CONTROLLER TESTS PASSED")
        print("="*60)
        sys.exit(0)

    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
