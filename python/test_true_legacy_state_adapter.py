"""
A0 Tests: TRUE Legacy State Adapter

Tests for the TRUE legacy state representation (18*N+15, no reduction).
Validates pack/unpack for core state and optional warm-start guesses.

Ground truth: A0 frozen contract specification
"""

import sys
import torch

from control.true_legacy_state_adapter import (
    COIL_STATE_DIM, TIP_STATE_DIM,
    true_legacy_state_dim, true_legacy_warmstart_dim,
    pack_true_legacy_state, unpack_true_legacy_state,
    pack_true_legacy_warmstart, unpack_true_legacy_warmstart,
)


def test_constants():
    """Test that constants are correctly defined."""
    print("\n" + "="*70)
    print("Test 1: Constants")
    print("="*70)

    # v[3] + w[3] + p[3] + R[9] = 18
    if COIL_STATE_DIM != 18:
        print(f"  [FAIL] COIL_STATE_DIM = {COIL_STATE_DIM}, expected 18")
        return False

    # p_tip[3] + R_tip[9] + u_tip[3] = 15
    if TIP_STATE_DIM != 15:
        print(f"  [FAIL] TIP_STATE_DIM = {TIP_STATE_DIM}, expected 15")
        return False

    print(f"  COIL_STATE_DIM = {COIL_STATE_DIM}")
    print(f"  TIP_STATE_DIM = {TIP_STATE_DIM}")
    print("  [PASS] Constants")
    return True


def test_dimension_helpers():
    """Test dimension calculation helpers."""
    print("\n" + "="*70)
    print("Test 2: Dimension Helpers")
    print("="*70)

    # Test for N=1
    dim_n1 = true_legacy_state_dim(1)
    expected_n1 = 18 * 1 + 15
    if dim_n1 != expected_n1:
        print(f"  [FAIL] true_legacy_state_dim(1) = {dim_n1}, expected {expected_n1}")
        return False

    # Test for N=2
    dim_n2 = true_legacy_state_dim(2)
    expected_n2 = 18 * 2 + 15
    if dim_n2 != expected_n2:
        print(f"  [FAIL] true_legacy_state_dim(2) = {dim_n2}, expected {expected_n2}")
        return False

    # Test warm-start dimensions
    ws_n1 = true_legacy_warmstart_dim(1)
    if ws_n1 != 6:
        print(f"  [FAIL] true_legacy_warmstart_dim(1) = {ws_n1}, expected 6")
        return False

    ws_n2 = true_legacy_warmstart_dim(2)
    if ws_n2 != 12:
        print(f"  [FAIL] true_legacy_warmstart_dim(2) = {ws_n2}, expected 12")
        return False

    print(f"  true_legacy_state_dim(1) = {dim_n1}")
    print(f"  true_legacy_state_dim(2) = {dim_n2}")
    print(f"  true_legacy_warmstart_dim(1) = {ws_n1}")
    print(f"  true_legacy_warmstart_dim(2) = {ws_n2}")
    print("  [PASS] Dimension helpers")
    return True


def test_unbatched_state_roundtrip_n1():
    """Test unbatched pack/unpack roundtrip for N=1."""
    print("\n" + "="*70)
    print("Test 3: Unbatched State Roundtrip (N=1)")
    print("="*70)

    torch.manual_seed(42)
    n_act = 1

    # Create random state
    x_coil = torch.randn(n_act, 18, dtype=torch.float64)
    xf = torch.randn(15, dtype=torch.float64)

    # Pack
    packed = pack_true_legacy_state(x_coil, xf)

    # Verify shape
    expected_dim = true_legacy_state_dim(n_act)
    if packed.shape != (expected_dim,):
        print(f"  [FAIL] Packed shape {packed.shape}, expected ({expected_dim},)")
        return False

    # Unpack
    x_coil_recovered, xf_recovered = unpack_true_legacy_state(packed, n_act)

    # Verify shapes
    if x_coil_recovered.shape != (n_act, 18):
        print(f"  [FAIL] x_coil_recovered shape {x_coil_recovered.shape}")
        return False
    if xf_recovered.shape != (15,):
        print(f"  [FAIL] xf_recovered shape {xf_recovered.shape}")
        return False

    # Check roundtrip accuracy
    err_coil = (x_coil - x_coil_recovered).abs().max().item()
    err_xf = (xf - xf_recovered).abs().max().item()

    print(f"  x_coil error: {err_coil:.3e}")
    print(f"  xf error: {err_xf:.3e}")

    if err_coil > 1e-14 or err_xf > 1e-14:
        print(f"  [FAIL] Roundtrip error too large")
        return False

    print("  [PASS] Unbatched roundtrip (N=1)")
    return True


def test_unbatched_state_roundtrip_n2():
    """Test unbatched pack/unpack roundtrip for N=2."""
    print("\n" + "="*70)
    print("Test 4: Unbatched State Roundtrip (N=2)")
    print("="*70)

    torch.manual_seed(43)
    n_act = 2

    # Create random state
    x_coil = torch.randn(n_act, 18, dtype=torch.float32)
    xf = torch.randn(15, dtype=torch.float32)

    # Pack
    packed = pack_true_legacy_state(x_coil, xf)

    # Verify shape
    expected_dim = true_legacy_state_dim(n_act)
    if packed.shape != (expected_dim,):
        print(f"  [FAIL] Packed shape {packed.shape}, expected ({expected_dim},)")
        return False

    # Unpack
    x_coil_recovered, xf_recovered = unpack_true_legacy_state(packed, n_act)

    # Check roundtrip accuracy
    err_coil = (x_coil - x_coil_recovered).abs().max().item()
    err_xf = (xf - xf_recovered).abs().max().item()

    print(f"  x_coil error: {err_coil:.3e}")
    print(f"  xf error: {err_xf:.3e}")

    if err_coil > 1e-6 or err_xf > 1e-6:  # float32 tolerance
        print(f"  [FAIL] Roundtrip error too large")
        return False

    print("  [PASS] Unbatched roundtrip (N=2)")
    return True


def test_batched_state_roundtrip():
    """Test batched pack/unpack roundtrip."""
    print("\n" + "="*70)
    print("Test 5: Batched State Roundtrip")
    print("="*70)

    torch.manual_seed(44)
    n_act = 2
    batch_size = 3

    # Create random batched state
    x_coil = torch.randn(batch_size, n_act, 18, dtype=torch.float64)
    xf = torch.randn(batch_size, 15, dtype=torch.float64)

    # Pack
    packed = pack_true_legacy_state(x_coil, xf)

    # Verify shape
    expected_dim = true_legacy_state_dim(n_act)
    if packed.shape != (batch_size, expected_dim):
        print(f"  [FAIL] Packed shape {packed.shape}, expected ({batch_size}, {expected_dim})")
        return False

    # Unpack
    x_coil_recovered, xf_recovered = unpack_true_legacy_state(packed, n_act)

    # Verify shapes
    if x_coil_recovered.shape != (batch_size, n_act, 18):
        print(f"  [FAIL] x_coil_recovered shape {x_coil_recovered.shape}")
        return False
    if xf_recovered.shape != (batch_size, 15):
        print(f"  [FAIL] xf_recovered shape {xf_recovered.shape}")
        return False

    # Check roundtrip accuracy
    err_coil = (x_coil - x_coil_recovered).abs().max().item()
    err_xf = (xf - xf_recovered).abs().max().item()

    print(f"  Batch size: {batch_size}, N_act: {n_act}")
    print(f"  x_coil error: {err_coil:.3e}")
    print(f"  xf error: {err_xf:.3e}")

    if err_coil > 1e-14 or err_xf > 1e-14:
        print(f"  [FAIL] Roundtrip error too large")
        return False

    print("  [PASS] Batched roundtrip")
    return True


def test_state_packing_order():
    """Test that packing order is correct: coil states then tip state."""
    print("\n" + "="*70)
    print("Test 6: State Packing Order")
    print("="*70)

    torch.manual_seed(45)
    n_act = 2

    # Create easily identifiable state
    x_coil = torch.arange(n_act * 18, dtype=torch.float32).reshape(n_act, 18)
    xf = torch.arange(15, dtype=torch.float32) + 1000

    # Pack
    packed = pack_true_legacy_state(x_coil, xf)

    # Verify packing order: coil states first
    coil_flat = x_coil.reshape(-1)
    err_coil = (packed[:n_act * 18] - coil_flat).abs().max().item()

    # Then tip state
    err_xf = (packed[n_act * 18:] - xf).abs().max().item()

    print(f"  Coil order error: {err_coil:.3e}")
    print(f"  Tip order error: {err_xf:.3e}")

    if err_coil > 1e-14 or err_xf > 1e-14:
        print(f"  [FAIL] Packing order incorrect")
        return False

    print("  [PASS] Packing order")
    return True


def test_unbatched_warmstart_roundtrip():
    """Test unbatched warm-start pack/unpack roundtrip."""
    print("\n" + "="*70)
    print("Test 7: Unbatched Warm-start Roundtrip")
    print("="*70)

    torch.manual_seed(46)
    n_act = 2

    # Create random warm-start guesses
    mL_guess = torch.randn(n_act, 3, dtype=torch.float64)
    nL_guess = torch.randn(n_act, 3, dtype=torch.float64)

    # Pack
    packed = pack_true_legacy_warmstart(mL_guess, nL_guess)

    # Verify shape
    expected_dim = true_legacy_warmstart_dim(n_act)
    if packed.shape != (expected_dim,):
        print(f"  [FAIL] Packed shape {packed.shape}, expected ({expected_dim},)")
        return False

    # Unpack
    mL_recovered, nL_recovered = unpack_true_legacy_warmstart(packed, n_act)

    # Check roundtrip accuracy
    err_mL = (mL_guess - mL_recovered).abs().max().item()
    err_nL = (nL_guess - nL_recovered).abs().max().item()

    print(f"  mL error: {err_mL:.3e}")
    print(f"  nL error: {err_nL:.3e}")

    if err_mL > 1e-14 or err_nL > 1e-14:
        print(f"  [FAIL] Roundtrip error too large")
        return False

    print("  [PASS] Unbatched warm-start roundtrip")
    return True


def test_batched_warmstart_roundtrip():
    """Test batched warm-start pack/unpack roundtrip."""
    print("\n" + "="*70)
    print("Test 8: Batched Warm-start Roundtrip")
    print("="*70)

    torch.manual_seed(47)
    n_act = 2
    batch_size = 5

    # Create random batched warm-start guesses
    mL_guess = torch.randn(batch_size, n_act, 3, dtype=torch.float64)
    nL_guess = torch.randn(batch_size, n_act, 3, dtype=torch.float64)

    # Pack
    packed = pack_true_legacy_warmstart(mL_guess, nL_guess)

    # Verify shape
    expected_dim = true_legacy_warmstart_dim(n_act)
    if packed.shape != (batch_size, expected_dim):
        print(f"  [FAIL] Packed shape {packed.shape}, expected ({batch_size}, {expected_dim})")
        return False

    # Unpack
    mL_recovered, nL_recovered = unpack_true_legacy_warmstart(packed, n_act)

    # Check roundtrip accuracy
    err_mL = (mL_guess - mL_recovered).abs().max().item()
    err_nL = (nL_guess - nL_recovered).abs().max().item()

    print(f"  Batch size: {batch_size}")
    print(f"  mL error: {err_mL:.3e}")
    print(f"  nL error: {err_nL:.3e}")

    if err_mL > 1e-14 or err_nL > 1e-14:
        print(f"  [FAIL] Roundtrip error too large")
        return False

    print("  [PASS] Batched warm-start roundtrip")
    return True


def test_warmstart_packing_order():
    """Test that warm-start packing order is: mL[3], nL[3] per coil."""
    print("\n" + "="*70)
    print("Test 9: Warm-start Packing Order")
    print("="*70)

    torch.manual_seed(48)
    n_act = 2

    # Create easily identifiable warm-start
    mL_guess = torch.arange(n_act * 3, dtype=torch.float32).reshape(n_act, 3)
    nL_guess = torch.arange(n_act * 3, dtype=torch.float32).reshape(n_act, 3) + 100

    # Pack
    packed = pack_true_legacy_warmstart(mL_guess, nL_guess)

    # Verify packing order: for each coil, mL[3] then nL[3]
    all_correct = True
    for j in range(n_act):
        offset = j * 6
        err_mL = (packed[offset:offset + 3] - mL_guess[j]).abs().max().item()
        err_nL = (packed[offset + 3:offset + 6] - nL_guess[j]).abs().max().item()

        if err_mL > 1e-14 or err_nL > 1e-14:
            print(f"  [FAIL] Coil {j}: mL error {err_mL:.3e}, nL error {err_nL:.3e}")
            all_correct = False

    if not all_correct:
        return False

    print("  Verified packing order: [mL[3], nL[3]] per coil")
    print("  [PASS] Warm-start packing order")
    return True


def test_shape_validation():
    """Test that invalid input shapes raise clear errors."""
    print("\n" + "="*70)
    print("Test 10: Shape Validation")
    print("="*70)

    # Test wrong dimension for x_coil
    try:
        pack_true_legacy_state(torch.randn(18), torch.randn(15))
        print("  [FAIL] Did not reject wrong x_coil dimension")
        return False
    except ValueError as e:
        if "x_coil must be 2D" in str(e):
            print(f"  [OK] Rejected wrong x_coil dimension")
        else:
            print(f"  [FAIL] Wrong error message: {e}")
            return False

    # Test wrong last dimension for x_coil
    try:
        pack_true_legacy_state(torch.randn(2, 10), torch.randn(15))
        print("  [FAIL] Did not reject x_coil last dimension")
        return False
    except ValueError as e:
        if "x_coil last dimension must be 18" in str(e):
            print(f"  [OK] Rejected x_coil last dimension")
        else:
            print(f"  [FAIL] Wrong error message: {e}")
            return False

    # Test batch size mismatch
    try:
        pack_true_legacy_state(torch.randn(3, 2, 18), torch.randn(2, 15))
        print("  [FAIL] Did not reject batch size mismatch")
        return False
    except ValueError as e:
        if "Batch size mismatch" in str(e):
            print(f"  [OK] Rejected batch size mismatch")
        else:
            print(f"  [FAIL] Wrong error message: {e}")
            return False

    # Test wrong packed dimension for unpack
    try:
        unpack_true_legacy_state(torch.randn(30), n_act=1)
        print("  [FAIL] Did not reject wrong packed dimension")
        return False
    except ValueError as e:
        if "x dimension must be 33" in str(e):
            print(f"  [OK] Rejected wrong packed dimension")
        else:
            print(f"  [FAIL] Wrong error message: {e}")
            return False

    print("  [PASS] Shape validation")
    return True


def main():
    """Run all TRUE legacy state adapter tests."""
    print("\n" + "="*70)
    print("A0 TRUE LEGACY STATE ADAPTER TESTS")
    print("="*70)

    tests = [
        ("Constants", test_constants),
        ("Dimension Helpers", test_dimension_helpers),
        ("Unbatched State Roundtrip (N=1)", test_unbatched_state_roundtrip_n1),
        ("Unbatched State Roundtrip (N=2)", test_unbatched_state_roundtrip_n2),
        ("Batched State Roundtrip", test_batched_state_roundtrip),
        ("State Packing Order", test_state_packing_order),
        ("Unbatched Warm-start Roundtrip", test_unbatched_warmstart_roundtrip),
        ("Batched Warm-start Roundtrip", test_batched_warmstart_roundtrip),
        ("Warm-start Packing Order", test_warmstart_packing_order),
        ("Shape Validation", test_shape_validation),
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
