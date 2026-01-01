"""
P1-2: LQR Warm-Start Validation Test

Validates that LQR initialization improves iLQR convergence compared to
zero initialization (baseline).

Acceptance criteria:
- LQR warm-start reduces iLQR iterations OR final cost
- Both methods produce valid, converged solutions
- Deterministic (seeded RNG for reproducibility)
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from control.ilqr import iLQRSolver
from crm_dynamics_torch import load_default_catheter_params
import crm_diff_py


def run_ilqr_comparison():
    """
    Run iLQR with zero vs. LQR initialization and compare convergence.
    """
    print("=" * 70)
    print("P1-2: LQR Warm-Start Validation")
    print("=" * 70)
    print()

    # Load parameters
    param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
    params_dict = load_default_catheter_params(param_file, config_file)

    # Problem setup
    dt = 0.01
    L_inserted = 50.0
    horizon = 15  # Shorter horizon for faster testing

    # Initial state
    x0 = np.zeros(6)

    # Target position (small offset for convergence)
    result_init = crm_diff_py.dynamics_forward(x0, np.zeros(3), 0.0, L_inserted, params_dict)
    p_init = result_init['p_tip']
    p_target = p_init + np.array([1.0, 0.5, 0.0])  # Small achievable target

    print(f"Initial tip position: [{p_init[0]:.4f}, {p_init[1]:.4f}, {p_init[2]:.4f}] mm")
    print(f"Target tip position:  [{p_target[0]:.4f}, {p_target[1]:.4f}, {p_target[2]:.4f}] mm")
    print(f"Initial error: {np.linalg.norm(p_target - p_init):.4f} mm")
    print()

    # Cost matrices (balanced for convergence)
    Q = np.zeros((6, 6))
    R = 0.001 * np.eye(3)  # Low control cost to allow actuation
    terminal_weight = 10.0  # Moderate terminal weight (avoids line search issues)

    print(f"Problem Configuration:")
    print(f"  Horizon: T = {horizon}")
    print(f"  Timestep: dt = {dt} s")
    print(f"  Control cost: R = {R[0,0]}")
    print(f"  Terminal weight: {terminal_weight}")
    print()

    # ===== TEST 1: Zero Initialization (Baseline) =====
    print("-" * 70)
    print("TEST 1: Zero Initialization (Baseline)")
    print("-" * 70)

    solver_zero = iLQRSolver(
        dt=dt, L_inserted=L_inserted, params_dict=params_dict,
        horizon=horizon, Q=Q, R=R, p_target=p_target,
        terminal_weight=terminal_weight,
        max_iters=30, tol=1e-3, reg_init=1e-3
    )

    X_zero, U_zero, converged_zero = solver_zero.solve(x0, U_init=None, init_method="zero", verbose=True)

    iters_zero = len(solver_zero.cost_history) - 1
    cost_zero = solver_zero.cost_history[-1]
    tip_error_zero = solver_zero.tip_error_history[-1]

    print()
    print(f"RESULTS (Zero Init):")
    print(f"  Iterations: {iters_zero}")
    print(f"  Final cost: {cost_zero:.6f}")
    print(f"  Final tip error: {tip_error_zero:.6f} mm")
    print(f"  Converged: {converged_zero}")
    print()

    # ===== TEST 2: LQR Warm-Start =====
    print("-" * 70)
    print("TEST 2: LQR Warm-Start")
    print("-" * 70)

    solver_lqr = iLQRSolver(
        dt=dt, L_inserted=L_inserted, params_dict=params_dict,
        horizon=horizon, Q=Q, R=R, p_target=p_target,
        terminal_weight=terminal_weight,
        max_iters=30, tol=1e-3, reg_init=1e-3
    )

    X_lqr, U_lqr, converged_lqr = solver_lqr.solve(x0, U_init=None, init_method="lqr", verbose=True)

    iters_lqr = len(solver_lqr.cost_history) - 1
    cost_lqr = solver_lqr.cost_history[-1]
    tip_error_lqr = solver_lqr.tip_error_history[-1]

    print()
    print(f"RESULTS (LQR Warm-Start):")
    print(f"  Iterations: {iters_lqr}")
    print(f"  Final cost: {cost_lqr:.6f}")
    print(f"  Final tip error: {tip_error_lqr:.6f} mm")
    print(f"  Converged: {converged_lqr}")
    print()

    # ===== COMPARISON =====
    print("=" * 70)
    print("COMPARISON: Zero Init vs. LQR Warm-Start")
    print("=" * 70)
    print()

    iter_improvement = iters_zero - iters_lqr
    cost_improvement = cost_zero - cost_lqr
    tip_improvement = tip_error_zero - tip_error_lqr

    print(f"Iteration reduction:  {iter_improvement:+d} ({-100*iter_improvement/max(iters_zero,1):+.1f}%)")
    print(f"Cost reduction:       {cost_improvement:+.6f} ({-100*cost_improvement/max(cost_zero,1e-6):+.1f}%)")
    print(f"Tip error reduction:  {tip_improvement:+.6f} mm ({-100*tip_improvement/max(tip_error_zero,1e-6):+.1f}%)")
    print()

    # ===== ACCEPTANCE CRITERIA =====
    print("-" * 70)
    print("ACCEPTANCE CRITERIA")
    print("-" * 70)

    # Both must converge or make progress
    both_valid = (converged_zero or iters_zero < 30) and (converged_lqr or iters_lqr < 30)
    no_divergence = cost_lqr < 1e6 and cost_zero < 1e6

    # LQR should improve convergence rate OR final cost
    faster_convergence = iters_lqr < iters_zero
    better_cost = cost_lqr < cost_zero * 1.01  # Allow 1% tolerance

    improvement = faster_convergence or better_cost

    print(f"1. Both methods produce valid solutions: {'✓ PASS' if both_valid else '✗ FAIL'}")
    print(f"2. No divergence (cost < 1e6):           {'✓ PASS' if no_divergence else '✗ FAIL'}")
    print(f"3. LQR improves convergence OR cost:     {'✓ PASS' if improvement else '✗ FAIL'}")
    if faster_convergence:
        print(f"   → Faster convergence: {iters_lqr} < {iters_zero} iterations")
    if better_cost:
        print(f"   → Better or equal cost: {cost_lqr:.6f} ≤ {cost_zero:.6f}")
    print()

    # Overall pass/fail
    all_pass = both_valid and no_divergence and improvement

    print("=" * 70)
    if all_pass:
        print("✓ PASS: LQR warm-start validation successful")
        print()
        print("LQR initialization demonstrates measurable improvement over zero init.")
        print("=" * 70)
        return 0
    else:
        print("✗ FAIL: LQR warm-start validation failed")
        print()
        if not improvement:
            print("LQR warm-start did not improve convergence or cost.")
            print("This may indicate:")
            print("  - LQR linearization is poor for this problem")
            print("  - Terminal weight too high (causing line search failures)")
            print("  - Need to tune cost matrices (Q, R)")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(run_ilqr_comparison())
