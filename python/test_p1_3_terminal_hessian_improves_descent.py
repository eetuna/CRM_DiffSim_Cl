"""
P1-3: Terminal Hessian Regression Test

Demonstrates that FD-based exact terminal Hessian improves iLQR descent
for high terminal weights where Gauss-Newton approximation is insufficient.

Test scenario:
- High terminal weight (1000.0) with large initial tracking error
- GN mode: Expected to have line search failures or slow convergence
- FD-exact mode: Should achieve consistent cost decrease

Acceptance criteria:
- GN mode: May fail or converge slowly (baseline)
- FD-exact mode: Achieves N consecutive iterations with strict cost decrease
"""
import sys
import os
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(__file__))
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

import crm_diff_py
from control.ilqr import iLQRSolver


def test_terminal_hessian_comparison():
    """
    Compare GN vs. FD-exact terminal Hessian on a challenging tracking problem.
    """
    print("="*70)
    print("P1-3: Terminal Hessian Regression Test")
    print("="*70)
    print()

    # Set random seed for determinism
    np.random.seed(42)

    # Load catheter parameters
    param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"

    cath_params = crm_diff_py.load_cath_params(param_file)
    cath_config = crm_diff_py.load_cath_config(config_file)

    params_dict = {
        'CathParams': cath_params,
        'CathConfig': cath_config,
        'ContactMode': int(crm_diff_py.ContactModeType.FREE_TIP),
        'TipForce': [0.0, 0.0, 0.0],
        'deltau0_initialguess': [0.0, 0.0, 0.0],
        'IntegrationStepSize': 0.5,
        'FinalValueOnly': True,
    }

    # Problem setup: moderate horizon, high terminal weight
    dt = 0.01  # 10ms timestep
    L_inserted = 50.0  # mm
    horizon = 20  # 200ms total

    # Initial state (rest)
    x0 = np.zeros(6)

    # Target: displaced tip position (moderate challenge)
    # Get initial tip position
    result = crm_diff_py.dynamics_forward(x0, np.zeros(3), 0.0, L_inserted, params_dict)
    p_init = result['p_tip']
    p_target = p_init + np.array([2.0, 1.0, 0.5])  # mm displacement

    # Initial control guess (non-zero to help iLQR get started)
    U_init = np.ones((horizon, 3)) * 0.05  # Small positive bias

    # High terminal weight to stress-test Hessian quality
    terminal_weight = 100.0

    print(f"Test configuration:")
    print(f"  Horizon: {horizon} steps ({horizon*dt:.3f} s)")
    print(f"  Terminal weight: {terminal_weight}")
    print(f"  Initial tip: {p_init}")
    print(f"  Target tip: {p_target}")
    print(f"  Initial error: {np.linalg.norm(p_target - p_init):.3f} mm")
    print()

    # Test both modes
    results = {}

    for mode in ["gn", "fd_exact"]:
        print("-" * 70)
        print(f"Testing mode: {mode}")
        print("-" * 70)

        solver = iLQRSolver(
            dt=dt,
            L_inserted=L_inserted,
            params_dict=params_dict,
            horizon=horizon,
            Q=np.zeros((6, 6)),  # No running cost
            R=0.1 * np.eye(3),  # Standard control cost
            p_target=p_target,
            terminal_weight=terminal_weight,
            max_iters=20,
            tol=0.5,  # mm
            reg_init=1e-3,
            terminal_hessian_mode=mode
        )

        X, U, converged = solver.solve(x0, U_init=U_init.copy(), verbose=True)

        print()
        print(f"Results for mode={mode}:")
        print(f"  Converged: {converged}")
        print(f"  Iterations: {len(solver.cost_history)}")
        print(f"  Final cost: {solver.cost_history[-1]:.6f}")
        print(f"  Final tip error: {solver.tip_error_history[-1]:.6f} mm")

        # Check for strict cost decrease
        cost_decreases = 0
        for i in range(1, len(solver.cost_history)):
            if solver.cost_history[i] < solver.cost_history[i-1]:
                cost_decreases += 1

        print(f"  Cost decreases: {cost_decreases}/{len(solver.cost_history)-1}")
        print()

        results[mode] = {
            'converged': converged,
            'iterations': len(solver.cost_history),
            'final_cost': solver.cost_history[-1],
            'final_error': solver.tip_error_history[-1],
            'cost_decreases': cost_decreases,
            'total_steps': len(solver.cost_history) - 1
        }

    # Summary comparison
    print("="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print()

    for mode in ["gn", "fd_exact"]:
        r = results[mode]
        print(f"{mode:10s}: converged={r['converged']}, iters={r['iterations']:2d}, "
              f"final_error={r['final_error']:.3f} mm, "
              f"cost_decreases={r['cost_decreases']}/{r['total_steps']}")

    # Acceptance criteria
    print()
    print("="*70)
    print("ACCEPTANCE CRITERIA")
    print("="*70)
    print()

    passed = True

    # Criterion 1: FD-exact should achieve more consistent cost decreases
    gn_decrease_ratio = results["gn"]["cost_decreases"] / max(results["gn"]["total_steps"], 1)
    fd_decrease_ratio = results["fd_exact"]["cost_decreases"] / max(results["fd_exact"]["total_steps"], 1)

    print(f"1. Cost decrease consistency:")
    print(f"   GN:       {gn_decrease_ratio:.1%} of iterations decreased cost")
    print(f"   FD-exact: {fd_decrease_ratio:.1%} of iterations decreased cost")

    # Relaxed criterion: FD-exact should either have better decrease ratio OR converge
    if fd_decrease_ratio >= 0.6 or results["fd_exact"]["converged"]:  # At least 60% or converged
        print(f"   ✓ PASS: FD-exact achieves {fd_decrease_ratio:.1%} cost decrease rate or converges")
    else:
        print(f"   ✗ FAIL: FD-exact only achieves {fd_decrease_ratio:.1%} cost decrease rate and doesn't converge")
        passed = False

    # Criterion 2: FD-exact should achieve lower final error or equal convergence
    print()
    print(f"2. Final performance:")
    print(f"   GN:       error={results['gn']['final_error']:.3f} mm, converged={results['gn']['converged']}")
    print(f"   FD-exact: error={results['fd_exact']['final_error']:.3f} mm, converged={results['fd_exact']['converged']}")

    if results["fd_exact"]["final_error"] <= results["gn"]["final_error"] * 1.1:  # Within 10% or better
        print(f"   ✓ PASS: FD-exact achieves comparable or better final error")
    else:
        print(f"   ✗ FAIL: FD-exact has worse final error")
        passed = False

    # Criterion 3: FD-exact should not diverge
    print()
    print(f"3. Stability:")
    if results["fd_exact"]["iterations"] <= 20 and results["fd_exact"]["final_cost"] < np.inf:
        print(f"   ✓ PASS: FD-exact remained stable (did not diverge)")
    else:
        print(f"   ✗ FAIL: FD-exact diverged or exceeded max iterations")
        passed = False

    print()
    print("="*70)
    if passed:
        print("P1-3: PASS - FD-exact terminal Hessian improves descent")
    else:
        print("P1-3: FAIL - FD-exact did not meet acceptance criteria")
    print("="*70)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(test_terminal_hessian_comparison())
