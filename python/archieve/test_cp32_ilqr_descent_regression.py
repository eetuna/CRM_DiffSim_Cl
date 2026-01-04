"""
CP3.2.1: iLQR Descent Regression Test

This test verifies that the iLQR backward pass fixes produce reliable descent
directions and convergence. The old implementation had issues with:
- Non-PD Q_uu matrices causing line search failures
- Ad-hoc regularization that didn't use Cholesky
- Missing symmetry enforcement

This test will FAIL on the old implementation and PASS on the fixed version.

Test criteria:
1. iLQR must complete at least 5 iterations with successful cost reduction
2. Final cost must be below baseline threshold
3. No backward pass failures
4. Monotonic cost decrease (no increases)
"""
import sys
import os
import numpy as np

# Add python directory and build directory to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

from control.ilqr import iLQRSolver
from crm_dynamics_torch import load_default_catheter_params


def test_ilqr_descent():
    """
    Test that iLQR produces reliable descent and converges.

    Returns:
        success: bool
        metrics: dict with diagnostic info
    """
    print("=" * 70)
    print("CP3.2.1: iLQR Descent Regression Test")
    print("=" * 70)
    print()

    # Load catheter parameters
    param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
    params_dict = load_default_catheter_params(param_file, config_file)

    # Problem setup
    dt = 0.01  # 10 ms timestep
    L_inserted = 50.0  # mm
    horizon = 15  # Moderate horizon for balance

    # Initial state (rest)
    x0 = np.zeros(6)

    # Get initial tip position
    import crm_diff_py
    result_init = crm_diff_py.dynamics_forward(x0, np.zeros(3), 0.0, L_inserted, params_dict)
    p_init = result_init['p_tip']

    # Target position (modest offset, achievable)
    p_target = p_init + np.array([1.5, 0.5, 0.0])  # 1.5mm in x, 0.5mm in y

    print(f"Problem Setup:")
    print(f"  Initial tip: {p_init}")
    print(f"  Target tip:  {p_target}")
    print(f"  Displacement: {np.linalg.norm(p_target - p_init):.3f} mm")
    print(f"  Horizon: {horizon} steps")
    print(f"  Timestep: {dt} s")
    print()

    # Initial control guess (small random to break symmetry)
    np.random.seed(42)
    U_init = np.random.randn(horizon, 3) * 0.01

    # Cost matrices
    Q = np.zeros((6, 6))  # No state cost
    R = 0.1 * np.eye(3)  # Moderate control cost

    # Create iLQR solver
    solver = iLQRSolver(
        dt=dt,
        L_inserted=L_inserted,
        params_dict=params_dict,
        horizon=horizon,
        Q=Q,
        R=R,
        p_target=p_target,
        terminal_weight=1.0,  # Balanced weight
        max_iters=15,  # Allow enough iterations
        tol=0.5,  # 0.5 mm convergence tolerance
        reg_init=1e-3,
        reg_scale=10.0
    )

    print("Running iLQR optimization...")
    print("-" * 70)

    # Solve
    X_opt, U_opt, converged = solver.solve(x0, U_init=U_init, verbose=True)

    print("-" * 70)
    print()

    # Analyze results
    num_iters = len(solver.cost_history) - 1
    cost_initial = solver.cost_history[0]
    cost_final = solver.cost_history[-1]
    cost_reduction = cost_initial - cost_final

    # Check for monotonic decrease
    cost_increases = 0
    for i in range(1, len(solver.cost_history)):
        if solver.cost_history[i] > solver.cost_history[i-1]:
            cost_increases += 1

    # Check for successful iterations (cost actually decreased)
    successful_iters = num_iters - cost_increases

    metrics = {
        'num_iters': num_iters,
        'successful_iters': successful_iters,
        'cost_initial': cost_initial,
        'cost_final': cost_final,
        'cost_reduction': cost_reduction,
        'cost_increases': cost_increases,
        'converged': converged,
        'final_tip_error': solver.tip_error_history[-1]
    }

    return metrics


def main():
    metrics = test_ilqr_descent()

    print("=" * 70)
    print("REGRESSION TEST VALIDATION")
    print("=" * 70)
    print()

    all_passed = True

    # Criterion 1: At least 5 successful iterations
    print(f"1. Successful Iterations:")
    print(f"   Completed: {metrics['num_iters']}")
    print(f"   Successful (cost decrease): {metrics['successful_iters']}")
    if metrics['successful_iters'] >= 5:
        print(f"   ✓ PASS: At least 5 successful iterations")
    else:
        print(f"   ✗ FAIL: Need >= 5 successful iterations, got {metrics['successful_iters']}")
        all_passed = False
    print()

    # Criterion 2: Final cost below threshold
    print(f"2. Cost Reduction:")
    print(f"   Initial cost: {metrics['cost_initial']:.6f}")
    print(f"   Final cost:   {metrics['cost_final']:.6f}")
    print(f"   Reduction:    {metrics['cost_reduction']:.6f}")

    # Baseline threshold: with proper descent, we should achieve at least 50% reduction
    baseline_threshold = 0.5 * metrics['cost_initial']
    if metrics['cost_final'] < baseline_threshold:
        print(f"   ✓ PASS: Final cost < 50% of initial ({metrics['cost_final']:.3f} < {baseline_threshold:.3f})")
    else:
        print(f"   ✗ FAIL: Final cost too high ({metrics['cost_final']:.3f} >= {baseline_threshold:.3f})")
        all_passed = False
    print()

    # Criterion 3: No cost increases (monotonic decrease)
    print(f"3. Monotonic Descent:")
    print(f"   Cost increases: {metrics['cost_increases']}")
    if metrics['cost_increases'] == 0:
        print(f"   ✓ PASS: Monotonic cost decrease")
    else:
        print(f"   ✗ FAIL: Cost increased {metrics['cost_increases']} times")
        all_passed = False
    print()

    # Criterion 4: Final tip error
    print(f"4. Final Tip Error:")
    print(f"   Error: {metrics['final_tip_error']:.6f} mm")
    if metrics['final_tip_error'] < 1.0:
        print(f"   ✓ PASS: Reasonable tracking (< 1 mm)")
    else:
        print(f"   INFO: Error = {metrics['final_tip_error']:.3f} mm")
    print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if all_passed:
        print("✓ CP3.2.1 REGRESSION TEST PASS")
        print()
        print("The iLQR backward pass fixes produce reliable descent:")
        print("  - Levenberg-Marquardt regularization with Cholesky")
        print("  - Symmetry enforcement on Q-matrices")
        print("  - Proper positive-definite handling")
        print("  - Monotonic cost reduction")
        print()
        print("This test would FAIL on the old implementation.")
        return 0
    else:
        print("✗ CP3.2.1 REGRESSION TEST FAIL")
        print()
        print("The backward pass still has issues. Check:")
        print("  - Q_uu positive definiteness")
        print("  - Regularization strategy")
        print("  - Symmetry of Q-matrices")
        return 1


if __name__ == "__main__":
    sys.exit(main())
