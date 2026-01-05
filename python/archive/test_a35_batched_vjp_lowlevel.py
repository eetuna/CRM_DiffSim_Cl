"""
A3.5: Batched VJP Test (Low-Level API)

Tests the batched VJP implementation using low-level bindings.
Verifies that batched computation matches looped baseline.
"""

import sys
import numpy as np
import crm_diff_py

from control import load_default_catheter_params

# Global params
PARAMS_DICT = None
N_ACT = 1


def load_params():
    """Load catheter parameters once."""
    global PARAMS_DICT
    if PARAMS_DICT is None:
        param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
        config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
        PARAMS_DICT = load_default_catheter_params(param_file, config_file)
    return PARAMS_DICT


def test_batched_vs_looped_equivalence():
    """
    Test that batched VJP matches looped per-RHS VJP.

    Uses the batched binding which does multi-RHS solve with one factorization.
    """
    print("\n" + "="*70)
    print("A3.5: Batched vs Looped VJP Equivalence")
    print("="*70)

    params = load_params()
    dt = 0.001
    L_inserted = 100.0

    # Simple state
    x_coil = np.zeros((N_ACT, 18), dtype=np.float64)
    xf = np.zeros(15, dtype=np.float64)

    R_identity = np.eye(3).flatten()
    x_coil[0, 9:18] = R_identity
    xf[3:12] = R_identity

    x_coil[0, 6:9] = [0.0, 0.0, 1.0]
    xf[:3] = [0.0, 0.0, 10.0]

    u = np.array([[0.01, 0.0, 0.0]], dtype=np.float64)

    # Multiple upstream gradients (num_rhs=4)
    num_rhs = 4
    np.random.seed(42)
    grad_tip_p_batch = np.random.randn(num_rhs, 3)

    print(f"  num_rhs: {num_rhs}")
    print(f"  grad_tip_p_batch shape: {grad_tip_p_batch.shape}")

    # Looped VJP (baseline)
    print("\n  Running looped VJP...")
    grad_x_coil_looped = []
    grad_xf_looped = []
    grad_u_looped = []

    for i in range(num_rhs):
        result = crm_diff_py.true_legacy_step_vjp(
            x_coil, xf, u, dt, L_inserted, params, grad_tip_p_batch[i]
        )
        grad_x_coil_looped.append(result['grad_x_coil'])
        grad_xf_looped.append(result['grad_xf'])
        grad_u_looped.append(result['grad_u'])

    grad_x_coil_looped = np.array(grad_x_coil_looped)
    grad_xf_looped = np.array(grad_xf_looped)
    grad_u_looped = np.array(grad_u_looped)

    print(f"  Looped shapes: grad_x_coil {grad_x_coil_looped.shape}, grad_xf {grad_xf_looped.shape}, grad_u {grad_u_looped.shape}")

    # Batched VJP
    print("\n  Running batched VJP...")
    result_batched = crm_diff_py.true_legacy_step_vjp_batched(
        x_coil, xf, u, dt, L_inserted, params, grad_tip_p_batch
    )

    grad_x_coil_batched = result_batched['grad_x_coil']
    grad_xf_batched = result_batched['grad_xf']
    grad_u_batched = result_batched['grad_u']

    print(f"  Batched shapes: grad_x_coil {grad_x_coil_batched.shape}, grad_xf {grad_xf_batched.shape}, grad_u {grad_u_batched.shape}")

    # Compare
    print("\n  Comparing gradients...")
    diff_x_coil = np.abs(grad_x_coil_batched - grad_x_coil_looped)
    diff_xf = np.abs(grad_xf_batched - grad_xf_looped)
    diff_u = np.abs(grad_u_batched - grad_u_looped)

    max_diff_x_coil = np.max(diff_x_coil)
    max_diff_xf = np.max(diff_xf)
    max_diff_u = np.max(diff_u)

    print(f"  Max abs diff grad_x_coil: {max_diff_x_coil:.6e}")
    print(f"  Max abs diff grad_xf: {max_diff_xf:.6e}")
    print(f"  Max abs diff grad_u: {max_diff_u:.6e}")

    # Relative error
    rel_err_x_coil = max_diff_x_coil / (np.max(np.abs(grad_x_coil_looped)) + 1e-12)
    rel_err_xf = max_diff_xf / (np.max(np.abs(grad_xf_looped)) + 1e-12)
    rel_err_u = max_diff_u / (np.max(np.abs(grad_u_looped)) + 1e-12)

    print(f"  Rel error grad_x_coil: {rel_err_x_coil:.6e}")
    print(f"  Rel error grad_xf: {rel_err_xf:.6e}")
    print(f"  Rel error grad_u: {rel_err_u:.6e}")

    # Pass criteria: exact match (should be bit-exact since same computation)
    tol = 1e-10

    passed = (max_diff_x_coil < tol and max_diff_xf < tol and max_diff_u < tol)

    if passed:
        print(f"\n  [PASS] Batched VJP matches looped VJP")
    else:
        print(f"\n  [FAIL] Batched VJP differs from looped VJP")

    return passed


def test_single_rhs_equivalence():
    """Test that batched with num_rhs=1 matches non-batched."""
    print("\n" + "="*70)
    print("A3.5: Single RHS Batched vs Non-Batched")
    print("="*70)

    params = load_params()
    dt = 0.001
    L_inserted = 100.0

    x_coil = np.zeros((N_ACT, 18), dtype=np.float64)
    xf = np.zeros(15, dtype=np.float64)

    R_identity = np.eye(3).flatten()
    x_coil[0, 9:18] = R_identity
    xf[3:12] = R_identity

    x_coil[0, 6:9] = [0.0, 0.0, 1.0]
    xf[:3] = [0.0, 0.0, 10.0]

    u = np.array([[0.01, 0.0, 0.0]], dtype=np.float64)

    grad_tip_p = np.array([1.0, 0.5, 0.2], dtype=np.float64)

    # Non-batched VJP
    print("  Running non-batched VJP...")
    result_single = crm_diff_py.true_legacy_step_vjp(
        x_coil, xf, u, dt, L_inserted, params, grad_tip_p
    )

    # Batched VJP with num_rhs=1
    print("  Running batched VJP (num_rhs=1)...")
    grad_tip_p_batch = grad_tip_p.reshape(1, 3)
    result_batched = crm_diff_py.true_legacy_step_vjp_batched(
        x_coil, xf, u, dt, L_inserted, params, grad_tip_p_batch
    )

    # Compare (extract first index from batched)
    diff_x_coil = np.max(np.abs(result_batched['grad_x_coil'][0] - result_single['grad_x_coil']))
    diff_xf = np.max(np.abs(result_batched['grad_xf'][0] - result_single['grad_xf']))
    diff_u = np.max(np.abs(result_batched['grad_u'][0] - result_single['grad_u']))

    print(f"  Max diff grad_x_coil: {diff_x_coil:.6e}")
    print(f"  Max diff grad_xf: {diff_xf:.6e}")
    print(f"  Max diff grad_u: {diff_u:.6e}")

    tol = 1e-10
    passed = (diff_x_coil < tol and diff_xf < tol and diff_u < tol)

    if passed:
        print(f"  [PASS] Single RHS batched matches non-batched")
    else:
        print(f"  [FAIL] Single RHS batched differs from non-batched")

    return passed


def main():
    """Run batched VJP tests."""
    print("\n" + "="*70)
    print("A3.5: BATCHED VJP TESTS (LOW-LEVEL)")
    print("="*70)

    tests = [
        ("Batched vs Looped Equivalence", test_batched_vs_looped_equivalence),
        ("Single RHS Equivalence", test_single_rhs_equivalence),
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
