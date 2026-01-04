"""
CP1.5 Test: PyTorch gradcheck and finite difference validation
"""
import torch
import numpy as np
import sys
import os

# Add python directory to path
sys.path.insert(0, os.path.dirname(__file__))

import crm_equilibrium


def finite_difference_check(u_test, L_inserted, params_dict, eps=1e-5):
    """
    Central finite difference check for ∂p_tip/∂u.

    Args:
        u_test: Tensor [3], actuation currents
        L_inserted: float, insertion length
        params_dict: catheter parameters
        eps: finite difference step size

    Returns:
        J_fd: ndarray [3, 3], finite difference Jacobian
        J_auto: ndarray [3, 3], autograd Jacobian
        max_abs_err: float
        max_rel_err: float
    """
    u_test = u_test.clone().detach().requires_grad_(True)
    num_u = u_test.shape[0]

    # Compute autograd Jacobian
    p_tip = crm_equilibrium.equilibrium_tip_position(u_test, L_inserted, params_dict)
    J_auto = []
    for i in range(3):
        grad_out = torch.zeros(3, dtype=torch.float64)
        grad_out[i] = 1.0
        grad_u = torch.autograd.grad(p_tip, u_test, grad_out, retain_graph=True)[0]
        J_auto.append(grad_u.numpy())
    J_auto = np.array(J_auto)  # [3, 3]

    # Compute finite difference Jacobian (central difference)
    J_fd = np.zeros((3, num_u))
    u_np = u_test.detach().numpy().copy()

    for j in range(num_u):
        # Forward perturbation
        u_plus = u_np.copy()
        u_plus[j] += eps
        p_plus = crm_equilibrium.equilibrium_tip_position(
            torch.from_numpy(u_plus), L_inserted, params_dict
        ).detach().numpy()

        # Backward perturbation
        u_minus = u_np.copy()
        u_minus[j] -= eps
        p_minus = crm_equilibrium.equilibrium_tip_position(
            torch.from_numpy(u_minus), L_inserted, params_dict
        ).detach().numpy()

        # Central difference
        J_fd[:, j] = (p_plus - p_minus) / (2 * eps)

    # Compute errors
    abs_err = np.abs(J_auto - J_fd)
    max_abs_err = np.max(abs_err)

    # Relative error (avoid division by zero)
    denom = np.maximum(np.abs(J_fd), 1e-10)
    rel_err = abs_err / denom
    max_rel_err = np.max(rel_err)

    return J_fd, J_auto, max_abs_err, max_rel_err


def test_operating_point(u_test, L_inserted, params_dict, point_name):
    """
    Test gradcheck and FD at a single operating point.

    Returns:
        bool: True if PASS, False if FAIL
    """
    print(f"\n{'='*60}")
    print(f"Testing: {point_name}")
    print(f"{'='*60}")
    print(f"u = {u_test.numpy()}")
    print(f"L_inserted = {L_inserted}")

    # Compute forward pass to verify convergence
    p_tip = crm_equilibrium.equilibrium_tip_position(u_test.clone(), L_inserted, params_dict)
    print(f"p_tip = {p_tip.numpy()}")

    # Test 1: torch.autograd.gradcheck
    print(f"\n[1/2] Running torch.autograd.gradcheck...")
    u_gradcheck = u_test.clone().detach().requires_grad_(True)

    def func(u_in):
        return crm_equilibrium.equilibrium_tip_position(u_in, L_inserted, params_dict)

    try:
        gradcheck_pass = torch.autograd.gradcheck(
            func,
            u_gradcheck,
            eps=1e-5,
            atol=1e-4,
            rtol=1e-3,
            raise_exception=False
        )
        if gradcheck_pass:
            print("  ✓ gradcheck PASS")
        else:
            print("  ✗ gradcheck FAIL")
    except Exception as e:
        print(f"  ✗ gradcheck FAIL (exception: {e})")
        gradcheck_pass = False

    # Test 2: Manual finite difference check
    print(f"\n[2/2] Running finite difference check...")
    J_fd, J_auto, max_abs_err, max_rel_err = finite_difference_check(
        u_test, L_inserted, params_dict, eps=1e-5
    )

    print(f"\nFinite Difference Jacobian (∂p_tip/∂u):")
    print(J_fd)
    print(f"\nAutograd Jacobian (∂p_tip/∂u):")
    print(J_auto)
    print(f"\nMax absolute error: {max_abs_err:.6e}")
    print(f"Max relative error: {max_rel_err:.6e}")

    # Check against thresholds
    fd_pass = (max_abs_err < 1e-4) and (max_rel_err < 1e-3)
    if fd_pass:
        print("  ✓ Finite difference check PASS")
    else:
        print("  ✗ Finite difference check FAIL")

    overall_pass = gradcheck_pass and fd_pass
    print(f"\n{'─'*60}")
    if overall_pass:
        print(f"  {point_name}: PASS")
    else:
        print(f"  {point_name}: FAIL")
    print(f"{'─'*60}")

    return overall_pass


def main():
    print("CP1.5 Test: PyTorch gradcheck + finite difference validation")
    print("="*60)

    # Load catheter parameters
    cath_params_file = "../catheterdata/CatheterParameterSet_1_new.txt"
    cath_config_file = "../catheterdata/CatheterSpatialConfiguration_1.txt"

    if not os.path.exists(cath_params_file):
        print(f"ERROR: Catheter parameter file not found: {cath_params_file}")
        return 1

    if not os.path.exists(cath_config_file):
        print(f"ERROR: Catheter configuration file not found: {cath_config_file}")
        return 1

    params_dict = crm_equilibrium.load_default_catheter_params(
        cath_params_file, cath_config_file
    )

    # Operating points (3 currents each, NUM_ACT_SET=1)
    L_inserted = 50.0

    test_cases = [
        ("Zero currents", torch.tensor([0.0, 0.0, 0.0], dtype=torch.float64)),
        ("Small currents", torch.tensor([0.1, -0.05, 0.08], dtype=torch.float64)),
        ("Moderate currents", torch.tensor([0.3, 0.2, -0.15], dtype=torch.float64)),
        ("Large currents", torch.tensor([0.5, -0.4, 0.3], dtype=torch.float64)),
    ]

    results = []
    for point_name, u_test in test_cases:
        try:
            passed = test_operating_point(u_test, L_inserted, params_dict, point_name)
            results.append((point_name, passed))
        except Exception as e:
            print(f"\n✗ {point_name}: EXCEPTION - {e}")
            import traceback
            traceback.print_exc()
            results.append((point_name, False))

    # Summary
    print(f"\n{'='*60}")
    print("CP1.5 TEST SUMMARY")
    print(f"{'='*60}")
    for point_name, passed in results:
        status = "PASS" if passed else "FAIL"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {point_name}: {status}")

    all_pass = all(passed for _, passed in results)
    print(f"{'='*60}")
    if all_pass:
        print("CP1.5: PASS - All gradcheck and FD tests passed")
        print(f"{'='*60}")
        return 0
    else:
        print("CP1.5: FAIL - Some tests failed")
        print(f"{'='*60}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
