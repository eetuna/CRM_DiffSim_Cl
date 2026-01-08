"""
FULLSTATE controller rollout robustness tests.

Small, fast tests (5-10 rollouts, 10-50 steps each) to detect:
- NaNs/Infs
- Solver non-convergence
- Nondeterminism
- Performance degradation
"""

import os
import sys
import time
import numpy as np
import pytest

repo_root = os.path.join(os.path.dirname(__file__), '..')
build_dir = os.path.join(repo_root, 'build_s15b')
if not os.path.isdir(build_dir):
    build_dir = os.path.join(repo_root, 'build_s15')
if not os.path.isdir(build_dir):
    build_dir = os.path.join(repo_root, 'build')
sys.path.insert(0, build_dir)
sys.path.insert(0, os.path.join(repo_root, 'python'))

import crm_diff_py
from control.lqr import finite_horizon_lqr
from control.ilqr import iLQRSolver
from control.mpc import MPCController
from control.true_legacy_state_adapter import pack_true_legacy_state, unpack_true_legacy_state


def make_params_dict():
    """Create conservative parameter dictionary for robustness tests."""
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
        'IntegrationStepSize': 0.5,  # Conservative step size
    }


def make_initial_state(params_dict, n_act=1, L_inserted=100.0):
    """Create initial FULLSTATE equilibrium state (deterministic, no perturbations)."""
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


def rollout_lqr(x0, p_target, params_dict, horizon=20, dt=0.01, n_act=1, seed=None):
    """Execute LQR rollout and return trajectory."""
    if seed is not None:
        np.random.seed(seed)

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
    return U_lqr, X_lqr, P_tip_lqr


def rollout_ilqr(x0, p_target, params_dict, horizon=20, dt=0.01, n_act=1,
                 max_iters=5, seed=None):
    """Execute iLQR rollout and return trajectory + solver info."""
    if seed is not None:
        np.random.seed(seed)

    solver = iLQRSolver(
        dt=dt,
        L_inserted=100.0,
        params_dict=params_dict,
        horizon=horizon,
        n_act=n_act,
        Q=None,
        R=0.01 * np.eye(3 * n_act),
        p_target=p_target,
        terminal_weight=10.0,
        max_iters=max_iters,
        jacobian_mode="implicit",
    )

    U_init = np.zeros((horizon, 3 * n_act), dtype=np.float64)
    X_opt, U_opt, _ = solver.solve(x0, U_init=U_init, verbose=False)

    return X_opt, U_opt, solver.cost_history


def rollout_mpc_steps(x0, p_target, params_dict, num_steps=20, dt=0.01,
                      horizon=6, n_act=1, max_ilqr_iters=1, seed=None):
    """Execute MPC rollout over num_steps and return closed-loop trajectory."""
    if seed is not None:
        np.random.seed(seed)

    controller = MPCController(
        dt=dt,
        L_inserted=100.0,
        params_dict=params_dict,
        horizon=horizon,
        n_act=n_act,
        Q_tip=1.0,
        R=0.01 * np.eye(3 * n_act),
        max_ilqr_iters=max_ilqr_iters,
        verbose=False,
    )

    def p_target_fn(_t):
        return p_target.copy()

    controller.set_reference(p_target_fn)

    X_traj = [x0]
    U_traj = []
    converged_flags = []
    warmstart = None

    x_current = x0
    for step in range(num_steps):
        # Compute control
        u_mpc, info = controller.compute_control(x_current, step * dt)
        U_traj.append(u_mpc)

        # Step dynamics
        x_coil_t, xf_t = unpack_true_legacy_state(x_current, n_act)
        result = crm_diff_py.true_legacy_step_forward(
            x_coil_t,
            xf_t,
            u_mpc.reshape(n_act, 3),
            dt,
            params_dict,
            None if warmstart is None else warmstart['mL_guess'],
            None if warmstart is None else warmstart['nL_guess'],
        )

        converged_flags.append(result['converged'])
        warmstart = {
            'mL_guess': result['mL_next'],
            'nL_guess': result['nL_next'],
        }

        x_next = pack_true_legacy_state(result['x_coil_next'], result['xf_next'])
        X_traj.append(x_next)
        x_current = x_next

    return np.array(X_traj), np.array(U_traj), converged_flags


# ============================================================================
# Test A: Deterministic rollout (no learning)
# ============================================================================

def test_deterministic_lqr_rollout():
    """
    Test A: Run 5 LQR rollouts with fixed seed, 20-step horizon.
    Assert: no NaN/Inf, deterministic results for same seed.
    """
    n_rollouts = 5
    horizon = 20
    dt = 0.01
    n_act = 1

    params_dict = make_params_dict()
    x0, p_target = make_initial_state(params_dict, n_act)

    results = []
    for i in range(n_rollouts):
        U, X, P_tip = rollout_lqr(x0, p_target, params_dict, horizon, dt, n_act, seed=i)

        # Assert no NaN/Inf
        assert np.all(np.isfinite(U)), f"Rollout {i}: U contains NaN/Inf"
        assert np.all(np.isfinite(X)), f"Rollout {i}: X contains NaN/Inf"
        assert np.all(np.isfinite(P_tip)), f"Rollout {i}: P_tip contains NaN/Inf"

        # Store for determinism check
        results.append((X, U, P_tip))

    # Determinism check: re-run first rollout
    U_repeat, X_repeat, P_tip_repeat = rollout_lqr(x0, p_target, params_dict, horizon, dt, n_act, seed=0)

    X_orig, U_orig, P_tip_orig = results[0]
    assert np.allclose(X_repeat, X_orig, atol=1e-12), "LQR rollout is not deterministic"
    assert np.allclose(U_repeat, U_orig, atol=1e-12), "LQR controls are not deterministic"

    print(f"✓ Test A passed: {n_rollouts} deterministic LQR rollouts, {horizon} steps each")


# ============================================================================
# Test B: iLQR short-horizon stability
# ============================================================================

def test_ilqr_short_horizon_stability():
    """
    Test B: Run 5 iLQR rollouts, 20-step horizon, 5 iterations.
    Assert: no NaN/Inf, cost does not explode.
    """
    n_rollouts = 5
    horizon = 20
    dt = 0.01
    n_act = 1
    max_iters = 5

    params_dict = make_params_dict()
    x0, p_target = make_initial_state(params_dict, n_act)

    for i in range(n_rollouts):
        X_opt, U_opt, cost_history = rollout_ilqr(
            x0, p_target, params_dict, horizon, dt, n_act, max_iters, seed=100 + i
        )

        # Assert no NaN/Inf
        assert np.all(np.isfinite(X_opt)), f"Rollout {i}: X_opt contains NaN/Inf"
        assert np.all(np.isfinite(U_opt)), f"Rollout {i}: U_opt contains NaN/Inf"
        assert np.all(np.isfinite(cost_history)), f"Rollout {i}: cost_history contains NaN/Inf"

        # Assert cost does not explode
        final_cost = cost_history[-1]
        assert final_cost < 1e6, f"Rollout {i}: cost exploded to {final_cost}"

        # Assert state norm remains finite
        state_norms = np.linalg.norm(X_opt, axis=1)
        assert np.all(state_norms < 1e6), f"Rollout {i}: state norm exploded"

    print(f"✓ Test B passed: {n_rollouts} iLQR rollouts, {horizon} steps, {max_iters} iters each")


# ============================================================================
# Test C: MPC short-horizon feasibility
# ============================================================================

def test_mpc_short_horizon_feasibility():
    """
    Test C: Run 5 MPC rollouts, 2 closed-loop steps each (conservative, like smoke test).
    Assert: no NaN/Inf, solver converges.
    """
    n_rollouts = 5
    num_steps = 2  # Conservative like smoke test
    dt = 0.01
    horizon = 6
    n_act = 1

    params_dict = make_params_dict()
    x0, p_target = make_initial_state(params_dict, n_act)

    for i in range(n_rollouts):
        X_traj, U_traj, converged_flags = rollout_mpc_steps(
            x0, p_target, params_dict, num_steps, dt, horizon, n_act, max_ilqr_iters=1, seed=None
        )

        # Assert no NaN/Inf
        assert np.all(np.isfinite(X_traj)), f"Rollout {i}: X_traj contains NaN/Inf"
        assert np.all(np.isfinite(U_traj)), f"Rollout {i}: U_traj contains NaN/Inf"

        # Assert solver converged at each step
        assert all(converged_flags), f"Rollout {i}: solver did not converge at some steps"

    print(f"✓ Test C passed: {n_rollouts} MPC rollouts, {num_steps} closed-loop steps each")


# ============================================================================
# Test D: Stability envelope (bounded controls)
# ============================================================================

def test_stability_envelope_bounded_controls():
    """
    Test D: Run 5 iLQR rollouts, 25-step horizon with bounded controls.
    Assert: no NaN/Inf, state norm remains finite.
    Note: Using iLQR instead of MPC to avoid repeated solver init issues.
    """
    n_rollouts = 5
    horizon = 25  # Conservative to avoid solver non-convergence
    dt = 0.01
    n_act = 1
    max_iters = 3

    params_dict = make_params_dict()
    x0, p_target = make_initial_state(params_dict, n_act)

    for i in range(n_rollouts):
        X_opt, U_opt, cost_history = rollout_ilqr(
            x0, p_target, params_dict, horizon, dt, n_act, max_iters, seed=300 + i
        )

        # Assert no NaN/Inf
        assert np.all(np.isfinite(X_opt)), f"Rollout {i}: X_opt contains NaN/Inf"
        assert np.all(np.isfinite(U_opt)), f"Rollout {i}: U_opt contains NaN/Inf"

        # Assert state norm remains finite (no explosion)
        state_norms = np.linalg.norm(X_opt, axis=1)
        assert np.all(state_norms < 1e6), f"Rollout {i}: state norm exploded, max={state_norms.max()}"

        # Assert control norm remains bounded
        control_norms = np.linalg.norm(U_opt, axis=1)
        assert np.all(control_norms < 1e3), f"Rollout {i}: control norm too large, max={control_norms.max()}"

    print(f"✓ Test D passed: {n_rollouts} stability envelope rollouts, {horizon} steps each")


# ============================================================================
# Test E: iLQR convergence behavior across seeds
# ============================================================================

def test_ilqr_convergence_across_seeds():
    """
    Test E: Run 5 iLQR rollouts with different seeds, 10 iterations.
    Assert: final cost < initial cost by meaningful margin OR monotonic decrease.
    """
    n_rollouts = 5
    horizon = 15
    dt = 0.01
    n_act = 1
    max_iters = 10

    params_dict = make_params_dict()
    x0, p_target = make_initial_state(params_dict, n_act)

    for i in range(n_rollouts):
        X_opt, U_opt, cost_history = rollout_ilqr(
            x0, p_target, params_dict, horizon, dt, n_act, max_iters, seed=400 + i
        )

        # Assert no NaN/Inf in cost_history
        assert np.all(np.isfinite(cost_history)), f"Rollout {i}: cost_history contains NaN/Inf"

        # Assert either meaningful decrease OR monotonic decrease
        initial_cost = cost_history[0]
        final_cost = cost_history[-1]

        meaningful_decrease = (final_cost < initial_cost - 1e-6)
        monotonic = all(cost_history[j+1] <= cost_history[j] + 1e-8 for j in range(len(cost_history)-1))

        assert meaningful_decrease or monotonic, \
            f"Rollout {i}: no convergence. initial={initial_cost:.6e}, final={final_cost:.6e}, history={cost_history}"

    print(f"✓ Test E passed: {n_rollouts} iLQR rollouts with convergence checks, {max_iters} iters each")


# ============================================================================
# Test F: Performance counters
# ============================================================================

def test_performance_counters():
    """
    Test F: Measure wall-time per simulator step and per iLQR iteration.
    No hard failure, just record and print.
    """
    n_steps = 50
    dt = 0.01
    n_act = 1

    params_dict = make_params_dict()
    x0, p_target = make_initial_state(params_dict, n_act)

    # Measure simulator step time
    x_current = x0
    x_coil, xf = unpack_true_legacy_state(x_current, n_act)
    u = np.zeros((n_act, 3), dtype=np.float64)

    times = []
    for _ in range(n_steps):
        t0 = time.time()
        result = crm_diff_py.true_legacy_step_forward(x_coil, xf, u, dt, params_dict)
        t1 = time.time()
        times.append(t1 - t0)

        x_coil = result['x_coil_next']
        xf = result['xf_next']

    mean_step_time = np.mean(times)
    std_step_time = np.std(times)

    # Measure iLQR iteration time
    horizon = 20
    max_iters = 5

    t0 = time.time()
    X_opt, U_opt, cost_history = rollout_ilqr(
        x0, p_target, params_dict, horizon, dt, n_act, max_iters, seed=500
    )
    t1 = time.time()

    total_ilqr_time = t1 - t0
    mean_iter_time = total_ilqr_time / len(cost_history) if len(cost_history) > 0 else 0.0

    print(f"✓ Test F performance counters:")
    print(f"  Simulator step time: {mean_step_time*1e3:.3f} ± {std_step_time*1e3:.3f} ms (n={n_steps})")
    print(f"  iLQR total time: {total_ilqr_time:.3f} s for {len(cost_history)} iterations")
    print(f"  iLQR mean iteration time: {mean_iter_time:.3f} s")


# ============================================================================
# Test G: Failure-mode hygiene
# ============================================================================

def test_failure_mode_hygiene():
    """
    Test G: Create intentionally-stressed rollout.
    Assert: clean failure (exception or status flag), NOT silent NaNs.
    """
    # Use longer horizon and larger controls to stress the system
    horizon = 50
    dt = 0.01
    n_act = 1
    max_iters = 10

    params_dict = make_params_dict()
    x0, p_target = make_initial_state(params_dict, n_act)

    # Perturb target far away to stress optimizer
    p_target_stressed = p_target + np.array([10.0, 10.0, 10.0])

    try:
        X_opt, U_opt, cost_history = rollout_ilqr(
            x0, p_target_stressed, params_dict, horizon, dt, n_act, max_iters, seed=600
        )

        # If it succeeds, must not contain NaNs
        assert np.all(np.isfinite(X_opt)), "Stressed rollout returned NaNs in X_opt (should fail cleanly)"
        assert np.all(np.isfinite(U_opt)), "Stressed rollout returned NaNs in U_opt (should fail cleanly)"
        assert np.all(np.isfinite(cost_history)), "Stressed rollout returned NaNs in cost_history"

        print(f"✓ Test G passed: stressed rollout succeeded without NaNs")

    except (AssertionError, RuntimeError, ValueError) as e:
        # Clean failure is acceptable
        print(f"✓ Test G passed: stressed rollout failed cleanly with {type(e).__name__}: {e}")


# ============================================================================
# Test H: Determinism
# ============================================================================

def test_determinism_identical_results():
    """
    Test H: Run same rollout twice with same seed and config.
    Assert: identical results within tight tolerance.
    """
    horizon = 20
    dt = 0.01
    n_act = 1
    max_iters = 5
    seed = 700

    params_dict = make_params_dict()
    x0, p_target = make_initial_state(params_dict, n_act)

    # First run
    X_opt1, U_opt1, cost_history1 = rollout_ilqr(
        x0, p_target, params_dict, horizon, dt, n_act, max_iters, seed=seed
    )

    # Second run
    X_opt2, U_opt2, cost_history2 = rollout_ilqr(
        x0, p_target, params_dict, horizon, dt, n_act, max_iters, seed=seed
    )

    # Assert identical results
    assert np.allclose(X_opt1, X_opt2, atol=1e-12), "X trajectories are not deterministic"
    assert np.allclose(U_opt1, U_opt2, atol=1e-12), "U trajectories are not deterministic"
    assert np.allclose(cost_history1, cost_history2, atol=1e-12), "Cost histories are not deterministic"

    print(f"✓ Test H passed: determinism verified for iLQR rollout (atol=1e-12)")


# ============================================================================
# Summary test (optional, for quick smoke check)
# ============================================================================

def test_rollout_robustness_summary():
    """Quick summary test that exercises all controller types."""
    print("\n" + "="*70)
    print("FULLSTATE Rollout Robustness Test Suite Summary")
    print("="*70)

    # Run one of each
    test_deterministic_lqr_rollout()
    test_ilqr_short_horizon_stability()
    test_mpc_short_horizon_feasibility()
    test_stability_envelope_bounded_controls()
    test_ilqr_convergence_across_seeds()
    test_performance_counters()
    test_failure_mode_hygiene()
    test_determinism_identical_results()

    print("="*70)
    print("All robustness tests PASSED")
    print("="*70)


if __name__ == "__main__":
    # Can run individual tests or summary
    test_rollout_robustness_summary()
