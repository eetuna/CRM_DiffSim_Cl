"""
A3 Tests: Batched Implicit VJP

Tests for the batched backward pass (VJP) using dynamics_backward_batched.
Validates that batched VJP matches looped single-vector VJP (exact equivalence).

Ground truth: docs/contracts/LEGACY_HYBRID_STATE_CONTRACT.md
Evidence: src/CRM_DiffDynamics.hpp:75-84 (dynamics_backward_batched)
"""

import sys
import time
import numpy as np

from control.legacy_state import LegacyState
from control.step_legacy_contract import (
    step_legacy_contract,
    vjp_legacy_contract,
    vjp_legacy_contract_batched,
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


def test_batched_vs_looped_equivalence():
    """Test that batched VJP equals looped single-vector VJP (exact match)."""
    print("\n" + "="*70)
    print("Test 1: Batched vs Looped Equivalence")
    print("="*70)

    params = load_params()

    # Operating point
    x_t = LegacyState.zeros()
    u_t = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    dt = 0.01
    L_inserted = 100.0

    # Forward pass (once)
    fwd_result = step_legacy_contract(x_t, u_t, dt, L_inserted, params)
    if not fwd_result.success:
        print(f"  [FAIL] Forward failed: {fwd_result.diagnostics['status']}")
        return False

    all_passed = True
    for K in [4, 6, 8]:
        print(f"\n  Testing K={K} adjoint vectors...")

        # Generate random adjoint vectors
        np.random.seed(42 + K)
        V = np.random.randn(K, 6).astype(np.float64)

        # Batched VJP
        batched_result = vjp_legacy_contract_batched(fwd_result, V, params)
        if not batched_result.success:
            print(f"    [FAIL] Batched VJP failed: {batched_result.diagnostics}")
            all_passed = False
            continue

        # Looped single-vector VJP
        looped_grad_x_t = np.zeros((K, 6), dtype=np.float64)
        looped_grad_u_t = np.zeros((K, 3), dtype=np.float64)

        for k in range(K):
            vjp_result = vjp_legacy_contract(fwd_result, V[k], params)
            if not vjp_result.success:
                print(f"    [FAIL] Single VJP failed at k={k}")
                all_passed = False
                break
            looped_grad_x_t[k] = vjp_result.grad_x_t
            looped_grad_u_t[k] = vjp_result.grad_u_t

        # Compare (exact match expected)
        err_x = np.max(np.abs(batched_result.grad_x_t - looped_grad_x_t))
        err_u = np.max(np.abs(batched_result.grad_u_t - looped_grad_u_t))

        if err_x > 1e-12:
            print(f"    [FAIL] grad_x_t mismatch: max_err={err_x:.3e}")
            all_passed = False
        else:
            print(f"    [OK] grad_x_t: max_err={err_x:.3e}")

        if err_u > 1e-12:
            print(f"    [FAIL] grad_u_t mismatch: max_err={err_u:.3e}")
            all_passed = False
        else:
            print(f"    [OK] grad_u_t: max_err={err_u:.3e}")

    if all_passed:
        print("\n  [PASS] Batched vs Looped equivalence")
    else:
        print("\n  [FAIL] Batched vs Looped mismatch")

    return all_passed


def test_shape_coverage():
    """Test shapes for various K values."""
    print("\n" + "="*70)
    print("Test 2: Shape Coverage")
    print("="*70)

    params = load_params()

    x_t = LegacyState.zeros()
    u_t = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    dt = 0.01
    L_inserted = 100.0

    fwd_result = step_legacy_contract(x_t, u_t, dt, L_inserted, params)
    if not fwd_result.success:
        print(f"  [FAIL] Forward failed")
        return False

    all_passed = True
    for K in [1, 4, 8]:
        print(f"\n  Testing K={K}...")

        V = np.random.randn(K, 6).astype(np.float64)
        result = vjp_legacy_contract_batched(fwd_result, V, params)

        # Check shapes
        expected_grad_x_shape = (K, 6)
        expected_grad_u_shape = (K, 3)

        if result.grad_x_t.shape != expected_grad_x_shape:
            print(f"    [FAIL] grad_x_t shape: expected {expected_grad_x_shape}, got {result.grad_x_t.shape}")
            all_passed = False
        else:
            print(f"    [OK] grad_x_t shape: {result.grad_x_t.shape}")

        if result.grad_u_t.shape != expected_grad_u_shape:
            print(f"    [FAIL] grad_u_t shape: expected {expected_grad_u_shape}, got {result.grad_u_t.shape}")
            all_passed = False
        else:
            print(f"    [OK] grad_u_t shape: {result.grad_u_t.shape}")

    if all_passed:
        print("\n  [PASS] Shape coverage")
    else:
        print("\n  [FAIL] Shape mismatch")

    return all_passed


def test_diagnostics_propagation():
    """Test that diagnostics are properly propagated."""
    print("\n" + "="*70)
    print("Test 3: Diagnostics Propagation")
    print("="*70)

    params = load_params()

    x_t = LegacyState.zeros()
    u_t = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    dt = 0.01
    L_inserted = 100.0

    fwd_result = step_legacy_contract(x_t, u_t, dt, L_inserted, params)
    if not fwd_result.success:
        print(f"  [FAIL] Forward failed")
        return False

    K = 4
    V = np.random.randn(K, 6).astype(np.float64)
    result = vjp_legacy_contract_batched(fwd_result, V, params)

    # Check diagnostics
    checks = []

    if 'status' not in result.diagnostics:
        print("  [FAIL] Missing 'status'")
        checks.append(False)
    else:
        print(f"  status: {result.diagnostics['status']}")
        checks.append(True)

    if 'lu_rank' not in result.diagnostics:
        print("  [FAIL] Missing 'lu_rank'")
        checks.append(False)
    else:
        print(f"  lu_rank: {result.diagnostics['lu_rank']} (expected 6)")
        checks.append(True)

    if 'rel_residual' not in result.diagnostics:
        print("  [FAIL] Missing 'rel_residual'")
        checks.append(False)
    else:
        print(f"  rel_residual: {result.diagnostics['rel_residual']:.2e} (should be < 1e-10)")
        checks.append(True)

    if 'K' not in result.diagnostics:
        print("  [FAIL] Missing 'K'")
        checks.append(False)
    else:
        print(f"  K: {result.diagnostics['K']} (expected {K})")
        if result.diagnostics['K'] != K:
            print(f"    [WARN] K mismatch: expected {K}, got {result.diagnostics['K']}")
        checks.append(True)

    # Validate expected values
    if result.diagnostics.get('lu_rank', -1) != 6:
        print(f"  [WARN] lu_rank is {result.diagnostics['lu_rank']}, expected 6")

    if result.diagnostics.get('rel_residual', 1.0) > 1e-10:
        print(f"  [WARN] rel_residual > 1e-10")

    passed = all(checks)
    if passed:
        print("  [PASS] Diagnostics propagation")
    else:
        print("  [FAIL] Diagnostics propagation")

    return passed


def test_three_operating_points():
    """Test batched VJP at 3 operating points."""
    print("\n" + "="*70)
    print("Test 4: Three Operating Points")
    print("="*70)

    params = load_params()
    K = 6
    dt = 0.01
    L_inserted = 100.0

    operating_points = [
        ("OP1: Rest", np.zeros(6), np.zeros(3)),
        ("OP2: Actuated", np.zeros(6), np.array([0.1, 0., 0.])),
        ("OP3: Moving", np.array([0.01, 0., 0., 0.1, 0., 0.]), np.array([0.1, 0.05, 0.])),
    ]

    all_passed = True

    for name, x_t_np, u_t_np in operating_points:
        print(f"\n  Testing {name}...")

        x_t = LegacyState.from_numpy(x_t_np.astype(np.float64))
        u_t = u_t_np.astype(np.float64)

        # Forward pass
        fwd_result = step_legacy_contract(x_t, u_t, dt, L_inserted, params)
        if not fwd_result.success:
            print(f"    [FAIL] Forward failed")
            all_passed = False
            continue

        # Generate adjoint vectors
        np.random.seed(123)
        V = np.random.randn(K, 6).astype(np.float64)

        # Batched VJP
        result = vjp_legacy_contract_batched(fwd_result, V, params)

        if not result.success:
            print(f"    [FAIL] Batched VJP failed: {result.diagnostics}")
            all_passed = False
            continue

        # Check finite values
        if not np.all(np.isfinite(result.grad_x_t)):
            print(f"    [FAIL] grad_x_t has NaN/Inf")
            all_passed = False
        elif not np.all(np.isfinite(result.grad_u_t)):
            print(f"    [FAIL] grad_u_t has NaN/Inf")
            all_passed = False
        else:
            print(f"    [OK] {name}: grad_x_t shape={result.grad_x_t.shape}, grad_u_t shape={result.grad_u_t.shape}")

    if all_passed:
        print("\n  [PASS] Three operating points")
    else:
        print("\n  [FAIL] Some operating points failed")

    return all_passed


def test_performance_sanity():
    """Performance sanity check (non-failing, just logs timing)."""
    print("\n" + "="*70)
    print("Test 5: Performance Sanity (Informational)")
    print("="*70)

    params = load_params()

    x_t = LegacyState.zeros()
    u_t = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    dt = 0.01
    L_inserted = 100.0
    K = 6

    fwd_result = step_legacy_contract(x_t, u_t, dt, L_inserted, params)
    if not fwd_result.success:
        print(f"  [SKIP] Forward failed")
        return True  # Non-failing test

    V = np.eye(K, 6, dtype=np.float64)

    # Warmup
    for _ in range(3):
        vjp_legacy_contract_batched(fwd_result, V, params)
        for k in range(K):
            vjp_legacy_contract(fwd_result, V[k], params)

    # Time batched
    n_iters = 10
    t_start = time.perf_counter()
    for _ in range(n_iters):
        vjp_legacy_contract_batched(fwd_result, V, params)
    t_batched = (time.perf_counter() - t_start) / n_iters

    # Time looped
    t_start = time.perf_counter()
    for _ in range(n_iters):
        for k in range(K):
            vjp_legacy_contract(fwd_result, V[k], params)
    t_looped = (time.perf_counter() - t_start) / n_iters

    speedup = t_looped / t_batched if t_batched > 0 else float('inf')

    print(f"  Batched (K={K}): {t_batched*1000:.3f} ms")
    print(f"  Looped (K={K}):  {t_looped*1000:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")

    # Always pass (informational only)
    print("  [INFO] Performance numbers logged (no strict threshold)")
    return True


def test_pytorch_batched_vjp():
    """Test PyTorch batched VJP helper."""
    print("\n" + "="*70)
    print("Test 6: PyTorch Batched VJP Helper")
    print("="*70)

    try:
        import torch
        from control.step_legacy_contract import legacy_dynamics_batched_vjp
    except ImportError:
        print("  [SKIP] PyTorch not available")
        return True

    if legacy_dynamics_batched_vjp is None:
        print("  [SKIP] legacy_dynamics_batched_vjp not available")
        return True

    params = load_params()

    x_t = LegacyState.zeros()
    u_t = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    dt = 0.01
    L_inserted = 100.0
    K = 4

    # Forward pass
    fwd_result = step_legacy_contract(x_t, u_t, dt, L_inserted, params)
    if not fwd_result.success:
        print(f"  [FAIL] Forward failed")
        return False

    # PyTorch batched VJP
    V_tensor = torch.randn(K, 6, dtype=torch.float64)

    try:
        grad_x_t, grad_u_t = legacy_dynamics_batched_vjp(fwd_result, V_tensor, params)
    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        return False

    # Check shapes
    if grad_x_t.shape != (K, 6):
        print(f"  [FAIL] grad_x_t shape: expected ({K}, 6), got {grad_x_t.shape}")
        return False

    if grad_u_t.shape != (K, 3):
        print(f"  [FAIL] grad_u_t shape: expected ({K}, 3), got {grad_u_t.shape}")
        return False

    # Check vs numpy version
    V_np = V_tensor.numpy()
    numpy_result = vjp_legacy_contract_batched(fwd_result, V_np, params)

    err_x = np.max(np.abs(grad_x_t.numpy() - numpy_result.grad_x_t))
    err_u = np.max(np.abs(grad_u_t.numpy() - numpy_result.grad_u_t))

    if err_x > 1e-12 or err_u > 1e-12:
        print(f"  [FAIL] PyTorch vs numpy mismatch: err_x={err_x:.3e}, err_u={err_u:.3e}")
        return False

    print(f"  grad_x_t shape: {grad_x_t.shape}")
    print(f"  grad_u_t shape: {grad_u_t.shape}")
    print(f"  PyTorch vs numpy: max_err_x={err_x:.3e}, max_err_u={err_u:.3e}")
    print("  [PASS] PyTorch batched VJP helper")
    return True


def main():
    """Run all A3 batched VJP tests."""
    print("\n" + "="*70)
    print("A3 BATCHED IMPLICIT VJP TESTS")
    print("="*70)

    tests = [
        ("Batched vs Looped Equivalence", test_batched_vs_looped_equivalence),
        ("Shape Coverage", test_shape_coverage),
        ("Diagnostics Propagation", test_diagnostics_propagation),
        ("Three Operating Points", test_three_operating_points),
        ("Performance Sanity", test_performance_sanity),
        ("PyTorch Batched VJP Helper", test_pytorch_batched_vjp),
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
