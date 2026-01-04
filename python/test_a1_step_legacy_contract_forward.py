"""
A1 Tests: Step Legacy Contract (Forward Only)

Tests for the forward dynamics wrapper.
Validates contract-exact I/O with the C++ binding.

Ground truth: docs/contracts/LEGACY_HYBRID_STATE_CONTRACT.md
"""

import sys
import numpy as np

from control.legacy_state import LegacyState
from control.step_legacy_contract import (
    step_legacy_contract,
    load_default_catheter_params,
)


# Global params (loaded once)
PARAMS_DICT = None


def load_params():
    """Load catheter parameters once."""
    global PARAMS_DICT
    if PARAMS_DICT is None:
        param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
        config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
        PARAMS_DICT = load_default_catheter_params(param_file, config_file)
    return PARAMS_DICT


def test_forward_rest_state():
    """Test forward at rest state (OP1: zero state, zero control)."""
    print("\n" + "="*70)
    print("Test 1: Forward at Rest State (OP1)")
    print("="*70)

    params = load_params()

    # Operating point 1: rest
    x_t = LegacyState.zeros()
    u_t = np.zeros(3, dtype=np.float64)
    dt = 0.01
    L_inserted = 100.0

    # Forward step
    result = step_legacy_contract(x_t, u_t, dt, L_inserted, params)

    # Check success
    if not result.success:
        print(f"  [FAIL] Forward failed: status={result.diagnostics['status']}")
        print(f"         exit_code={result.diagnostics['exit_code']}")
        return False

    # Check observables exist
    if 'p_tip' not in result.observables:
        print("  [FAIL] Missing p_tip in observables")
        return False
    if 'u_tip' not in result.observables:
        print("  [FAIL] Missing u_tip in observables")
        return False

    print(f"  Status: {result.diagnostics['status']}")
    print(f"  p_tip: {result.observables['p_tip']}")
    print(f"  [PASS] Forward at rest state")
    return True


def test_forward_actuated():
    """Test forward with actuation (OP2: zero state, non-zero control)."""
    print("\n" + "="*70)
    print("Test 2: Forward with Actuation (OP2)")
    print("="*70)

    params = load_params()

    # Operating point 2: actuated
    x_t = LegacyState.zeros()
    u_t = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    dt = 0.01
    L_inserted = 100.0

    # Forward step
    result = step_legacy_contract(x_t, u_t, dt, L_inserted, params)

    # Check success
    if not result.success:
        print(f"  [FAIL] Forward failed: status={result.diagnostics['status']}")
        return False

    print(f"  Status: {result.diagnostics['status']}")
    print(f"  p_tip: {result.observables['p_tip']}")
    print(f"  [PASS] Forward with actuation")
    return True


def test_forward_moving():
    """Test forward with moving state (OP3: non-zero state, non-zero control)."""
    print("\n" + "="*70)
    print("Test 3: Forward with Moving State (OP3)")
    print("="*70)

    params = load_params()

    # Operating point 3: moving
    x_t = LegacyState(
        u_0=np.array([0.01, 0.0, 0.0], dtype=np.float64),
        v_0=np.array([0.1, 0.0, 0.0], dtype=np.float64),
    )
    u_t = np.array([0.1, 0.05, 0.0], dtype=np.float64)
    dt = 0.01
    L_inserted = 100.0

    # Forward step
    result = step_legacy_contract(x_t, u_t, dt, L_inserted, params)

    # Check success
    if not result.success:
        print(f"  [FAIL] Forward failed: status={result.diagnostics['status']}")
        return False

    print(f"  Status: {result.diagnostics['status']}")
    print(f"  p_tip: {result.observables['p_tip']}")
    print(f"  [PASS] Forward with moving state")
    return True


def test_output_shapes():
    """Test that output shapes are correct."""
    print("\n" + "="*70)
    print("Test 4: Output Shapes")
    print("="*70)

    params = load_params()

    x_t = LegacyState.zeros()
    u_t = np.zeros(3, dtype=np.float64)
    dt = 0.01
    L_inserted = 100.0

    result = step_legacy_contract(x_t, u_t, dt, L_inserted, params)

    # Check x_next is LegacyState
    if not isinstance(result.x_next, LegacyState):
        print(f"  [FAIL] x_next is {type(result.x_next)}, expected LegacyState")
        return False

    # Check x_next has correct numpy shape
    x_next_np = result.x_next.to_numpy()
    if x_next_np.shape != (6,):
        print(f"  [FAIL] x_next shape: {x_next_np.shape}, expected (6,)")
        return False

    # Check p_tip shape
    if result.observables['p_tip'].shape != (3,):
        print(f"  [FAIL] p_tip shape: {result.observables['p_tip'].shape}")
        return False

    # Check u_tip shape
    if result.observables['u_tip'].shape != (3,):
        print(f"  [FAIL] u_tip shape: {result.observables['u_tip'].shape}")
        return False

    print(f"  x_next: LegacyState → (6,) numpy")
    print(f"  p_tip: (3,)")
    print(f"  u_tip: (3,)")
    print("  [PASS] Output shapes")
    return True


def test_finiteness():
    """Test that all outputs are finite (no NaN/Inf)."""
    print("\n" + "="*70)
    print("Test 5: Finiteness")
    print("="*70)

    params = load_params()

    x_t = LegacyState.zeros()
    u_t = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    dt = 0.01
    L_inserted = 100.0

    result = step_legacy_contract(x_t, u_t, dt, L_inserted, params)

    # Check x_next
    x_next_np = result.x_next.to_numpy()
    if not np.all(np.isfinite(x_next_np)):
        print(f"  [FAIL] x_next contains non-finite values")
        return False

    # Check p_tip
    if not np.all(np.isfinite(result.observables['p_tip'])):
        print(f"  [FAIL] p_tip contains non-finite values")
        return False

    # Check u_tip
    if not np.all(np.isfinite(result.observables['u_tip'])):
        print(f"  [FAIL] u_tip contains non-finite values")
        return False

    print("  All outputs are finite")
    print("  [PASS] Finiteness")
    return True


def test_determinism():
    """Test that same inputs produce same outputs."""
    print("\n" + "="*70)
    print("Test 6: Determinism")
    print("="*70)

    params = load_params()

    x_t = LegacyState.zeros()
    u_t = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    dt = 0.01
    L_inserted = 100.0

    # Run twice
    result1 = step_legacy_contract(x_t, u_t, dt, L_inserted, params)
    result2 = step_legacy_contract(x_t, u_t, dt, L_inserted, params)

    # Compare x_next
    x_next_1 = result1.x_next.to_numpy()
    x_next_2 = result2.x_next.to_numpy()
    err_x = np.linalg.norm(x_next_1 - x_next_2)

    # Compare p_tip
    err_p = np.linalg.norm(
        result1.observables['p_tip'] - result2.observables['p_tip']
    )

    print(f"  x_next error: {err_x:.3e}")
    print(f"  p_tip error: {err_p:.3e}")

    passed = err_x < 1e-14 and err_p < 1e-14

    if passed:
        print("  [PASS] Determinism")
    else:
        print("  [FAIL] Determinism (same inputs → different outputs)")

    return passed


def test_multi_step_rollout():
    """Test 10-step trajectory doesn't diverge."""
    print("\n" + "="*70)
    print("Test 7: Multi-Step Rollout")
    print("="*70)

    params = load_params()

    x_t = LegacyState.zeros()
    u_t = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    dt = 0.01
    L_inserted = 100.0
    n_steps = 10

    trajectory_x = [x_t.to_numpy().copy()]

    for step in range(n_steps):
        result = step_legacy_contract(x_t, u_t, dt, L_inserted, params)

        if not result.success:
            print(f"  [FAIL] Step {step} failed: status={result.diagnostics['status']}")
            return False

        x_t = result.x_next
        trajectory_x.append(x_t.to_numpy().copy())

    trajectory_x = np.array(trajectory_x)

    # Check trajectory changed (not degenerate)
    total_change = np.linalg.norm(trajectory_x[-1] - trajectory_x[0])
    print(f"  Total state change over {n_steps} steps: {total_change:.3e}")

    if total_change < 1e-6:
        print("  [FAIL] Trajectory is degenerate (no movement)")
        return False

    # Check trajectory didn't explode
    max_norm = np.max(np.linalg.norm(trajectory_x, axis=1))
    if max_norm > 1e3:
        print(f"  [FAIL] Trajectory diverged (max norm: {max_norm:.3e})")
        return False

    print(f"  Completed {n_steps}-step rollout successfully")
    print("  [PASS] Multi-step rollout")
    return True


def main():
    """Run all A1 forward tests."""
    print("\n" + "="*70)
    print("A1 STEP LEGACY CONTRACT (FORWARD ONLY) TESTS")
    print("="*70)

    tests = [
        ("Forward at Rest State (OP1)", test_forward_rest_state),
        ("Forward with Actuation (OP2)", test_forward_actuated),
        ("Forward with Moving State (OP3)", test_forward_moving),
        ("Output Shapes", test_output_shapes),
        ("Finiteness", test_finiteness),
        ("Determinism", test_determinism),
        ("Multi-Step Rollout", test_multi_step_rollout),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
        except Exception as e:
            print(f"\n  [EXCEPTION] {name}: {e}")
            import traceback
            traceback.print_exc()
            passed = False

        results.append((name, passed))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    total = len(results)
    passed_count = sum(1 for _, p in results if p)
    print(f"\n  Total: {passed_count}/{total} passed")

    all_passed = all(p for _, p in results if p)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
