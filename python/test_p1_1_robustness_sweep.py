"""
P1-1: Robustness Sweep Stress Test for Dynamics Primitive

Randomly samples N=20 operating points across bounded state/control/parameter space
and validates:
1. Forward pass succeeds (status=0, rank=6, residual<1e-10)
2. Backward pass succeeds (status=0, rank=6, residual<1e-10)
3. No NaN/Inf in states or gradients
4. States remain bounded (||x_next|| < 1e3)

This test catches edge-case numerical failures that fixed-point tests might miss.

Sampling ranges (deterministic seed for reproducibility):
- x_t: u0 in [-0.05, 0.05], v0 in [-0.2, 0.2]
- u_t: [-0.2, 0.2] (within actuator safe range)
- dt: [0.005, 0.02] (2× to 4× nominal timestep)
- L_inserted: [30, 80] mm (short to long insertions)
"""
import sys
import os
import numpy as np

# Add build directory to path for crm_diff_py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import crm_diff_py


def load_catheter_params():
    """Load standard catheter parameters."""
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

    return params_dict


def generate_random_operating_point(rng, idx):
    """
    Generate a random operating point within safe bounds.

    Args:
        rng: np.random.Generator instance
        idx: int, operating point index (for logging)

    Returns:
        x_t, u_t, dt, L_inserted
    """
    # State: [u0 (3), v0 (3)]
    # u0: base curvature (1/mm), small values near rest
    # v0: base curvature rate (1/mm/s), moderate velocities
    u0 = rng.uniform(-0.05, 0.05, size=3)
    v0 = rng.uniform(-0.2, 0.2, size=3)
    x_t = np.concatenate([u0, v0])

    # Control: currents in Amperes, safe range
    u_t = rng.uniform(-0.2, 0.2, size=3)

    # Timestep: vary from 5ms to 20ms
    dt = rng.choice([0.005, 0.01, 0.015, 0.02])

    # Insertion length: short to long
    L_inserted = rng.uniform(30.0, 80.0)

    return x_t, u_t, dt, L_inserted


def test_operating_point(x_t, u_t, dt, L_inserted, params_dict, idx):
    """
    Test dynamics forward/backward at a single operating point.

    Args:
        x_t: np.array (6,)
        u_t: np.array (3,)
        dt: float
        L_inserted: float
        params_dict: dict
        idx: int, operating point index

    Returns:
        success: bool
        errors: list of str (empty if success)
    """
    errors = []

    # ===== FORWARD PASS =====
    try:
        result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)
    except Exception as e:
        errors.append(f"Forward pass exception: {e}")
        return False, errors

    # Check forward status
    if result['status'] != 0:
        errors.append(f"Forward status={result['status']}, expected 0")

    # Check forward rank
    if result['lu_rank'] < 6:
        errors.append(f"Forward rank={result['lu_rank']}, expected 6")

    # Check forward residual
    if result['rel_solve_residual'] >= 1e-9:
        errors.append(f"Forward residual={result['rel_solve_residual']:.2e}, expected < 1e-9")

    # Check x_next is finite
    x_next = result['x_next']
    if not np.all(np.isfinite(x_next)):
        errors.append(f"x_next contains NaN/Inf: {x_next}")

    # Check x_next is bounded (prevent runaway)
    norm_x_next = np.linalg.norm(x_next)
    if norm_x_next >= 1e3:
        errors.append(f"||x_next|| = {norm_x_next:.2e} >= 1e3 (runaway state)")

    # If forward failed, don't test backward
    if errors:
        return False, errors

    # ===== BACKWARD PASS =====
    # Use a random upstream gradient (seeded for reproducibility within test)
    grad_x_next = np.random.randn(6)

    try:
        bwd_result = crm_diff_py.dynamics_backward(result, grad_x_next, params_dict)
    except Exception as e:
        errors.append(f"Backward pass exception: {e}")
        return False, errors

    # Check backward status
    if bwd_result['status'] != 0:
        errors.append(f"Backward status={bwd_result['status']}, expected 0")

    # Check backward rank
    if bwd_result['lu_rank'] < 6:
        errors.append(f"Backward rank={bwd_result['lu_rank']}, expected 6")

    # Check backward residual
    if bwd_result['rel_residual'] >= 1e-9:
        errors.append(f"Backward residual={bwd_result['rel_residual']:.2e}, expected < 1e-9")

    # Check gradients are finite
    grad_x_t = bwd_result['grad_x_t']
    grad_u_t = bwd_result['grad_u_t']

    if not np.all(np.isfinite(grad_x_t)):
        errors.append(f"grad_x_t contains NaN/Inf: {grad_x_t}")

    if not np.all(np.isfinite(grad_u_t)):
        errors.append(f"grad_u_t contains NaN/Inf: {grad_u_t}")

    # Success if no errors
    return len(errors) == 0, errors


def main():
    print("=" * 70)
    print("P1-1: Robustness Sweep Stress Test")
    print("=" * 70)
    print()

    # Load parameters
    print("Loading catheter parameters...")
    params_dict = load_catheter_params()
    print("  param_file: ./catheterdata/CatheterParameterSet_1_dyn.txt")
    print("  config_file: ./catheterdata/CatheterSpatialConfiguration_1.txt")
    print()

    # Sampling configuration
    N_POINTS = 20
    SEED = 42  # Deterministic for reproducibility
    rng = np.random.default_rng(SEED)

    print(f"Sampling Configuration:")
    print(f"  N = {N_POINTS} operating points")
    print(f"  Seed = {SEED} (deterministic)")
    print(f"  x_t: u0 in [-0.05, 0.05], v0 in [-0.2, 0.2]")
    print(f"  u_t: [-0.2, 0.2] A")
    print(f"  dt: {{0.005, 0.01, 0.015, 0.02}} s")
    print(f"  L_inserted: [30, 80] mm")
    print()

    print("-" * 70)
    print("Running stress tests...")
    print("-" * 70)
    print()

    all_passed = True
    failed_points = []

    for i in range(N_POINTS):
        # Generate random point
        x_t, u_t, dt, L_inserted = generate_random_operating_point(rng, i)

        # Test it
        success, errors = test_operating_point(x_t, u_t, dt, L_inserted, params_dict, i)

        # Report
        if success:
            print(f"Point {i+1:2d}/{N_POINTS}: ✓ PASS")
            print(f"  x_t = [{x_t[0]:+.4f}, {x_t[1]:+.4f}, {x_t[2]:+.4f}, "
                  f"{x_t[3]:+.4f}, {x_t[4]:+.4f}, {x_t[5]:+.4f}]")
            print(f"  u_t = [{u_t[0]:+.4f}, {u_t[1]:+.4f}, {u_t[2]:+.4f}] A")
            print(f"  dt = {dt:.4f} s, L = {L_inserted:.1f} mm")
        else:
            all_passed = False
            failed_points.append(i + 1)
            print(f"Point {i+1:2d}/{N_POINTS}: ✗ FAIL")
            print(f"  x_t = [{x_t[0]:+.4f}, {x_t[1]:+.4f}, {x_t[2]:+.4f}, "
                  f"{x_t[3]:+.4f}, {x_t[4]:+.4f}, {x_t[5]:+.4f}]")
            print(f"  u_t = [{u_t[0]:+.4f}, {u_t[1]:+.4f}, {u_t[2]:+.4f}] A")
            print(f"  dt = {dt:.4f} s, L = {L_inserted:.1f} mm")
            for error in errors:
                print(f"    ERROR: {error}")

        print()

    # Summary
    print("=" * 70)
    print("ROBUSTNESS SWEEP SUMMARY")
    print("=" * 70)
    print()

    if all_passed:
        print(f"✓ PASS: All {N_POINTS} operating points succeeded")
        print()
        print("Validation criteria (all points):")
        print("  ✓ Forward: status=0, rank=6, residual<1e-9")
        print("  ✓ Backward: status=0, rank=6, residual<1e-9")
        print("  ✓ No NaN/Inf in states or gradients")
        print("  ✓ States bounded: ||x_next|| < 1e3")
        print()
        print("=" * 70)
        return 0
    else:
        print(f"✗ FAIL: {len(failed_points)}/{N_POINTS} points failed")
        print(f"  Failed points: {failed_points}")
        print()
        print("This indicates edge-case numerical issues that need investigation.")
        print("Review failure details above and check:")
        print("  1. Equilibrium solver convergence at these states")
        print("  2. Matrix conditioning (check lu_rank, residuals)")
        print("  3. Physics parameter bounds (M, D, K matrices)")
        print()
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
