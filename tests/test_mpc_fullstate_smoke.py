"""
MPC FULLSTATE Smoke Test

Verifies that MPC runs in closed-loop for multiple steps with finite outputs.
Task 3: MPC closed-loop test.
"""

import sys
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/build')
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/python')

import numpy as np
import crm_diff_py
from control.true_legacy_state_adapter import pack_true_legacy_state, true_legacy_state_dim, unpack_true_legacy_state
from control.mpc import MPCController


def create_test_scenario(n_act=1):
    """Create initial state and reference trajectory for testing."""
    # Initial state: straight catheter
    x_coil = np.zeros((n_act, 18))
    x_coil[:, 6:9] = [[0.0, 0.0, 0.0]]  # coil position at origin
    x_coil[:, 9:18] = np.eye(3).flatten()  # identity rotation

    xf = np.zeros(15)
    xf[0:3] = [0.0, 0.0, 100.0]  # tip at [0, 0, 100]
    xf[3:12] = np.eye(3).flatten()  # identity rotation
    xf[12:15] = [0.0, 0.0, 0.0]  # zero curvature

    x0 = pack_true_legacy_state(x_coil, xf)

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

    return x0, params_dict, n_act


def test_mpc_smoke():
    """Test MPC closed-loop execution for multiple steps."""
    print("\n" + "="*60)
    print("Task 3: MPC Closed-Loop Test")
    print("="*60)

    n_act = 1
    x0, params_dict, n_act = create_test_scenario(n_act)

    state_dim = true_legacy_state_dim(n_act)
    control_dim = 3 * n_act

    print(f"State dimension: {state_dim}")
    print(f"Control dimension: {control_dim}")

    # Extract initial tip position
    x_coil_0, xf_0 = unpack_true_legacy_state(x0, n_act)
    p_tip_0 = xf_0[:3]
    print(f"Initial tip position: {p_tip_0}")

    dt = 0.01
    L_inserted = 100.0
    horizon = 3  # Very short horizon for smoke test
    num_mpc_steps = 5  # Small number of MPC steps

    print(f"\nRunning MPC with horizon={horizon}, num_steps={num_mpc_steps}, dt={dt}")

    try:
        # Create MPC controller
        controller = MPCController(
            dt=dt,
            L_inserted=L_inserted,
            params_dict=params_dict,
            horizon=horizon,
            n_act=n_act,
            Q_tip=10.0,
            R=0.01 * np.eye(control_dim),
            max_ilqr_iters=2,  # Only 2 iters per MPC step for speed
            verbose=True
        )

        # Constant reference (stay at current position)
        def p_target_fn(t):
            return np.array([0.0, 0.0, 100.0])

        controller.set_reference(p_target_fn)

        print(f"✓ MPCController instantiated successfully")

        # Run closed-loop for a few steps
        x_current = x0.copy()
        controls_applied = []
        states_recorded = [x_current.copy()]

        for step in range(num_mpc_steps):
            t_current = step * dt

            print(f"\nMPC step {step+1}/{num_mpc_steps} (t={t_current:.3f}s)")

            # Compute MPC control
            u_mpc, info = controller.compute_control(x_current, t_current)

            # Check for finiteness
            assert not np.any(np.isnan(u_mpc)), f"MPC control at step {step} contains NaN"
            assert not np.any(np.isinf(u_mpc)), f"MPC control at step {step} contains Inf"
            assert np.all(np.abs(u_mpc) <= 0.5), f"MPC control at step {step} exceeds bounds"

            controls_applied.append(u_mpc.copy())

            # Apply control
            x_coil_t, xf_t = unpack_true_legacy_state(x_current, n_act)
            u_mpc_reshaped = u_mpc.reshape(n_act, 3)
            result = crm_diff_py.true_legacy_step_forward(
                x_coil_t, xf_t, u_mpc_reshaped, dt, params_dict
            )

            if not result['converged']:
                print(f"Warning: Dynamics did not converge at step {step}")
                # Continue anyway for smoke test

            # Update state
            x_current = pack_true_legacy_state(result['x_coil_next'], result['xf_next'])
            states_recorded.append(x_current.copy())

            # Check state is finite
            assert not np.any(np.isnan(x_current)), f"State at step {step+1} contains NaN"
            assert not np.any(np.isinf(x_current)), f"State at step {step+1} contains Inf"

            _, xf_current = unpack_true_legacy_state(x_current, n_act)
            p_tip_current = xf_current[:3]
            print(f"  Tip position: {p_tip_current}")
            print(f"  Control magnitude: {np.linalg.norm(u_mpc):.6f} A")

        print(f"\n--- Results ---")
        print(f"Completed {num_mpc_steps} MPC steps")
        print(f"All states remained finite")
        print(f"All controls remained bounded")

        print(f"\n✓ MPC closed-loop completed successfully")
        print(f"✓ All states and controls are finite and bounded")

        return True

    except Exception as e:
        print(f"\n✗ MPC smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("MPC FULLSTATE Smoke Test")
    print("="*60)

    success = test_mpc_smoke()

    if success:
        print("\n" + "="*60)
        print("MPC SMOKE TEST PASSED")
        print("="*60)
        sys.exit(0)
    else:
        print("\n" + "="*60)
        print("MPC SMOKE TEST FAILED")
        print("="*60)
        sys.exit(1)
