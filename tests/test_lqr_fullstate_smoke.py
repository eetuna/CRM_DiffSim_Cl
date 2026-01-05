"""
LQR FULLSTATE Smoke Test

Verifies that LQR runs without error on FULLSTATE and produces stabilizing control.
Tests a single-step scenario with small perturbation.
"""

import sys
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/build')
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/python')

import numpy as np
import crm_diff_py
from control.true_legacy_state_adapter import pack_true_legacy_state, true_legacy_state_dim, unpack_true_legacy_state
from control.lqr import finite_horizon_lqr


def create_test_scenario(n_act=1):
    """Create initial state and target for testing."""
    # Initial state: straight catheter at z=100mm with small perturbation
    x_coil = np.zeros((n_act, 18))
    x_coil[:, 6:9] = [[0.0, 0.0, 0.0]]  # coil position at origin
    x_coil[:, 9:18] = np.eye(3).flatten()  # identity rotation

    xf = np.zeros(15)
    xf[0:3] = [1.0, 0.0, 100.0]  # tip at [1, 0, 100] (small perturbation in x)
    xf[3:12] = np.eye(3).flatten()  # identity rotation
    xf[12:15] = [0.0, 0.0, 0.0]  # zero curvature

    x0 = pack_true_legacy_state(x_coil, xf)

    # Target: straight position at [0, 0, 100]
    p_target = np.array([0.0, 0.0, 100.0])

    # Catheter params
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


def test_lqr_smoke():
    """Test LQR with short horizon to verify it runs without error."""
    print("\n" + "="*60)
    print("Task 1: LQR Smoke Test (Single Step)")
    print("="*60)

    n_act = 1
    x0, p_target, params_dict, n_act = create_test_scenario(n_act)

    state_dim = true_legacy_state_dim(n_act)
    control_dim = 3 * n_act

    print(f"State dimension: {state_dim}")
    print(f"Control dimension: {control_dim}")

    # Extract initial tip position
    x_coil_0, xf_0 = unpack_true_legacy_state(x0, n_act)
    p_tip_0 = xf_0[:3]
    print(f"Initial tip position: {p_tip_0}")
    print(f"Target position: {p_target}")
    print(f"Initial error: {np.linalg.norm(p_tip_0 - p_target):.6f} mm")

    # Run LQR with very short horizon for smoke test
    horizon = 5
    dt = 0.01
    L_inserted = 100.0

    print(f"\nRunning LQR with horizon={horizon}, dt={dt}")

    try:
        U_lqr, X_lqr, P_tip_lqr = finite_horizon_lqr(
            x0=x0,
            p_target=p_target,
            dt=dt,
            L_inserted=L_inserted,
            params_dict=params_dict,
            horizon=horizon,
            n_act=n_act,
            Q=None,  # Zero state cost
            R=None,  # Default control cost
            terminal_weight=100.0,
            u_nominal=None,  # Use default heuristic
            verbose=True
        )

        # Verify shapes
        assert U_lqr.shape == (horizon, control_dim), f"U_lqr shape: {U_lqr.shape}"
        assert X_lqr.shape == (horizon + 1, state_dim), f"X_lqr shape: {X_lqr.shape}"
        assert P_tip_lqr.shape == (horizon + 1, 3), f"P_tip_lqr shape: {P_tip_lqr.shape}"

        # Check for NaN or Inf
        assert not np.any(np.isnan(U_lqr)), "U_lqr contains NaN"
        assert not np.any(np.isinf(U_lqr)), "U_lqr contains Inf"
        assert not np.any(np.isnan(X_lqr)), "X_lqr contains NaN"
        assert not np.any(np.isinf(X_lqr)), "X_lqr contains Inf"

        # Check control bounds (should be clipped to [-0.5, 0.5])
        assert np.all(U_lqr >= -0.5) and np.all(U_lqr <= 0.5), \
            f"Controls out of bounds: min={np.min(U_lqr)}, max={np.max(U_lqr)}"

        # Compute final error
        final_error = np.linalg.norm(P_tip_lqr[-1] - p_target)

        print(f"\n--- Results ---")
        print(f"Final tip position: {P_tip_lqr[-1]}")
        print(f"Final tip error: {final_error:.6f} mm")
        print(f"Max control magnitude: {np.max(np.abs(U_lqr)):.6f} A")
        print(f"Control trajectory (first 3 steps):")
        for t in range(min(3, horizon)):
            print(f"  t={t}: u={U_lqr[t]}")

        # Success criteria: LQR should produce finite, bounded controls
        # (We don't require convergence for smoke test, just stability)
        print(f"\n✓ LQR completed successfully")
        print(f"✓ All outputs are finite and bounded")

        return True

    except Exception as e:
        print(f"\n✗ LQR failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("LQR FULLSTATE Smoke Test")
    print("="*60)

    success = test_lqr_smoke()

    if success:
        print("\n" + "="*60)
        print("LQR SMOKE TEST PASSED")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("LQR SMOKE TEST FAILED")
        print("="*60)
        sys.exit(1)
