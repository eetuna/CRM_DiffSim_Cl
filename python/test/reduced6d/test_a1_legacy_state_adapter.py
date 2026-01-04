"""
A1 Tests: Legacy State Adapter

Tests for the 6D legacy state representation and conversion functions.
Validates that the adapter layer enforces the contract exactly.

Ground truth: docs/contracts/LEGACY_HYBRID_STATE_CONTRACT.md
"""

import sys
import numpy as np

from control.legacy_state import LegacyState, STATE_DIM_LEGACY, U0_DIM, V0_DIM
from control.legacy_state_adapter import (
    pack_legacy_state,
    unpack_legacy_state,
    legacy_state_to_numpy,
    numpy_to_legacy_state,
    batch_to_legacy_states,
)


def test_pack_unpack_roundtrip():
    """Test that pack/unpack are inverses."""
    print("\n" + "="*70)
    print("Test 1: Pack/Unpack Roundtrip")
    print("="*70)

    np.random.seed(42)
    u_0 = np.random.randn(3) * 0.01
    v_0 = np.random.randn(3) * 0.1

    # Pack
    state = pack_legacy_state(u_0, v_0)

    # Unpack
    u_0_r, v_0_r = unpack_legacy_state(state)

    # Validate roundtrip
    err_u0 = np.linalg.norm(u_0 - u_0_r)
    err_v0 = np.linalg.norm(v_0 - v_0_r)

    print(f"  u_0 error: {err_u0:.3e}")
    print(f"  v_0 error: {err_v0:.3e}")

    passed = err_u0 < 1e-14 and err_v0 < 1e-14

    if passed:
        print("  [PASS] Pack/unpack roundtrip")
    else:
        print(f"  [FAIL] Pack/unpack roundtrip (tolerance 1e-14)")

    return passed


def test_numpy_conversion_roundtrip():
    """Test numpy → state → numpy preserves values."""
    print("\n" + "="*70)
    print("Test 2: Numpy Conversion Roundtrip")
    print("="*70)

    np.random.seed(43)
    x_original = np.random.randn(6) * 0.01

    # numpy → state → numpy
    state = numpy_to_legacy_state(x_original)
    x_recovered = legacy_state_to_numpy(state)

    err = np.linalg.norm(x_original - x_recovered)
    print(f"  Roundtrip error: {err:.3e}")

    passed = err < 1e-14

    if passed:
        print("  [PASS] Numpy conversion roundtrip")
    else:
        print(f"  [FAIL] Numpy conversion roundtrip (tolerance 1e-14)")

    return passed


def test_batch_conversion():
    """Test (B,6) batch array handling."""
    print("\n" + "="*70)
    print("Test 3: Batch Conversion")
    print("="*70)

    B = 5
    np.random.seed(44)
    x_batch = np.random.randn(B, 6) * 0.01

    # Convert batch to list of states
    states = batch_to_legacy_states(x_batch)

    # Validate
    if len(states) != B:
        print(f"  [FAIL] Expected {B} states, got {len(states)}")
        return False

    # Check each state matches
    for i in range(B):
        x_i = legacy_state_to_numpy(states[i])
        err = np.linalg.norm(x_batch[i] - x_i)
        if err > 1e-14:
            print(f"  [FAIL] State {i} error: {err:.3e}")
            return False

    print(f"  Converted batch ({B}, 6) → {B} LegacyState objects")
    print("  [PASS] Batch conversion")
    return True


def test_zero_state():
    """Test LegacyState.zeros() creates valid zero state."""
    print("\n" + "="*70)
    print("Test 4: Zero State")
    print("="*70)

    state = LegacyState.zeros()

    # Check fields
    if state.u_0.shape != (3,):
        print(f"  [FAIL] u_0 shape: {state.u_0.shape}")
        return False
    if state.v_0.shape != (3,):
        print(f"  [FAIL] v_0 shape: {state.v_0.shape}")
        return False

    # Check all zeros
    if not np.allclose(state.u_0, 0.0):
        print(f"  [FAIL] u_0 not zero: {state.u_0}")
        return False
    if not np.allclose(state.v_0, 0.0):
        print(f"  [FAIL] v_0 not zero: {state.v_0}")
        return False

    # Check dtype
    if state.u_0.dtype != np.float64:
        print(f"  [FAIL] u_0 dtype: {state.u_0.dtype}")
        return False

    print("  [PASS] Zero state")
    return True


def test_shape_validation():
    """Test that wrong shapes are rejected with clear errors."""
    print("\n" + "="*70)
    print("Test 5: Shape Validation")
    print("="*70)

    # Test wrong u_0 shape
    try:
        pack_legacy_state(np.zeros(2), np.zeros(3))
        print("  [FAIL] Did not reject u_0 shape (2,)")
        return False
    except ValueError as e:
        if "u_0 must be (3,)" in str(e):
            print(f"  [OK] Rejected u_0 shape (2,): {e}")
        else:
            print(f"  [FAIL] Wrong error message: {e}")
            return False

    # Test wrong v_0 shape
    try:
        pack_legacy_state(np.zeros(3), np.zeros(4))
        print("  [FAIL] Did not reject v_0 shape (4,)")
        return False
    except ValueError as e:
        if "v_0 must be (3,)" in str(e):
            print(f"  [OK] Rejected v_0 shape (4,): {e}")
        else:
            print(f"  [FAIL] Wrong error message: {e}")
            return False

    # Test wrong state array shape
    try:
        numpy_to_legacy_state(np.zeros(5))
        print("  [FAIL] Did not reject state shape (5,)")
        return False
    except ValueError as e:
        if "must be (6,)" in str(e):
            print(f"  [OK] Rejected state shape (5,): {e}")
        else:
            print(f"  [FAIL] Wrong error message: {e}")
            return False

    print("  [PASS] Shape validation")
    return True


def test_dtype_conversion():
    """Test that float32 is accepted and converted to float64."""
    print("\n" + "="*70)
    print("Test 6: Dtype Conversion")
    print("="*70)

    # Create float32 arrays
    x_float32 = np.zeros(6, dtype=np.float32)

    # Convert to state (should convert to float64)
    state = numpy_to_legacy_state(x_float32)

    # Check dtype
    if state.u_0.dtype != np.float64:
        print(f"  [FAIL] u_0 dtype: {state.u_0.dtype} (expected float64)")
        return False
    if state.v_0.dtype != np.float64:
        print(f"  [FAIL] v_0 dtype: {state.v_0.dtype} (expected float64)")
        return False

    print("  Converted float32 → float64")
    print("  [PASS] Dtype conversion")
    return True


def test_no_observable_leakage():
    """Test that LegacyState has no p_tip/u_tip fields."""
    print("\n" + "="*70)
    print("Test 7: No Observable Leakage")
    print("="*70)

    state = LegacyState.zeros()

    # Check that p_tip and u_tip are NOT attributes
    if hasattr(state, 'p_tip'):
        print("  [FAIL] LegacyState has 'p_tip' attribute (should not)")
        return False
    if hasattr(state, 'u_tip'):
        print("  [FAIL] LegacyState has 'u_tip' attribute (should not)")
        return False

    # Check only expected fields exist
    expected_fields = {'u_0', 'v_0'}
    actual_fields = {f.name for f in state.__dataclass_fields__.values()}

    if actual_fields != expected_fields:
        print(f"  [FAIL] Unexpected fields: {actual_fields - expected_fields}")
        return False

    print("  LegacyState has ONLY u_0, v_0 (no observables)")
    print("  [PASS] No observable leakage")
    return True


def main():
    """Run all A1 adapter tests."""
    print("\n" + "="*70)
    print("A1 LEGACY STATE ADAPTER TESTS")
    print("="*70)

    tests = [
        ("Pack/Unpack Roundtrip", test_pack_unpack_roundtrip),
        ("Numpy Conversion Roundtrip", test_numpy_conversion_roundtrip),
        ("Batch Conversion", test_batch_conversion),
        ("Zero State", test_zero_state),
        ("Shape Validation", test_shape_validation),
        ("Dtype Conversion", test_dtype_conversion),
        ("No Observable Leakage", test_no_observable_leakage),
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

    all_passed = all(p for _, p in results)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
