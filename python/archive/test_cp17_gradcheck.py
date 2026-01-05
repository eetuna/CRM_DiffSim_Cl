"""
CP1.7: PyTorch gradcheck Validation for Equilibrium Primitive

Uses torch.autograd.gradcheck to validate implicit-VJP implementation.
Tests at 3 operating points with deterministic behavior.
"""
import sys
import os
import torch
import numpy as np
import json
from datetime import datetime

# Add python directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

import crm_equilibrium
import crm_diff_py


def dump_failure_case(test_name, u, L_inserted, p_tip, gradcheck_result):
    """Dump failing case for replay."""
    failure_data = {
        'timestamp': datetime.now().isoformat(),
        'test_name': test_name,
        'u': u.detach().cpu().numpy().tolist(),
        'L_inserted': L_inserted,
        'p_tip': p_tip.detach().cpu().numpy().tolist(),
        'gradcheck_passed': gradcheck_result,
        'eps': 1e-5,
        'atol': 1e-4,
        'rtol': 1e-3
    }

    filename = f"failure_cp17_{test_name}.json"
    with open(filename, 'w') as f:
        json.dump(failure_data, f, indent=2)

    print(f"  Dumped failure case to: {filename}")


def test_operating_point(test_name, u_np, L_inserted, params_dict,
                         eps=1e-5, atol=1e-4, rtol=1e-3):
    """
    Test a single operating point with torch.autograd.gradcheck.

    Args:
        test_name: str
        u_np: ndarray [3]
        L_inserted: float
        params_dict: catheter parameters
        eps: FD step size for gradcheck
        atol: absolute tolerance
        rtol: relative tolerance

    Returns:
        passed: bool
    """
    print(f"\n{'='*70}")
    print(f"Test: {test_name}")
    print(f"{'='*70}")
    print(f"u = {u_np}")
    print(f"L_inserted = {L_inserted}")

    # Convert to torch tensor (float64, requires_grad)
    u = torch.from_numpy(u_np).clone().requires_grad_(True)

    # Compute forward pass to check convergence
    try:
        p_tip = crm_equilibrium.equilibrium_tip_position(u.clone(), L_inserted, params_dict)
        print(f"p_tip = {p_tip.detach().numpy()}")
    except RuntimeError as e:
        print(f"✗ FAIL: Forward pass failed: {e}")
        return False

    # Define function for gradcheck
    def func(u_in):
        return crm_equilibrium.equilibrium_tip_position(u_in, L_inserted, params_dict)

    # Run gradcheck
    print(f"\nRunning torch.autograd.gradcheck...")
    print(f"  eps={eps}, atol={atol}, rtol={rtol}")

    try:
        # Force float64 for numerical stability
        u_test = u.clone().detach().requires_grad_(True).to(dtype=torch.float64)

        gradcheck_result = torch.autograd.gradcheck(
            func,
            u_test,
            eps=eps,
            atol=atol,
            rtol=rtol,
            raise_exception=False
        )

        if gradcheck_result:
            print(f"\n✓ PASS: gradcheck succeeded")
        else:
            print(f"\n✗ FAIL: gradcheck failed")
            dump_failure_case(test_name, u_test, L_inserted, p_tip, gradcheck_result)

        return gradcheck_result

    except Exception as e:
        print(f"\n✗ FAIL: gradcheck exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("="*70)
    print("CP1.7: PyTorch gradcheck Validation")
    print("="*70)
    print()

    # Load parameters
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

    # Test parameters
    eps = 1e-5
    atol = 1e-4
    rtol = 1e-3

    print(f"Parameters:")
    print(f"  eps = {eps}")
    print(f"  atol = {atol}")
    print(f"  rtol = {rtol}")
    print(f"  dtype = torch.float64")

    # Operating points
    test_cases = [
        ("OP1_zero_current", np.array([0.0, 0.0, 0.0], dtype=np.float64), 50.0),
        ("OP2_single_axis", np.array([0.3, 0.0, 0.0], dtype=np.float64), 50.0),
        ("OP3_multi_axis", np.array([0.2, 0.1, 0.05], dtype=np.float64), 100.0),
    ]

    results = []
    for test_name, u, Li in test_cases:
        passed = test_operating_point(test_name, u, Li, params_dict, eps, atol, rtol)
        results.append((test_name, passed))

    # Summary
    print(f"\n{'='*70}")
    print("CP1.7 SUMMARY")
    print(f"{'='*70}")

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    overall_pass = all(passed for _, passed in results)

    print(f"\n{'='*70}")
    if overall_pass:
        print("CP1.7: PASS - All gradcheck tests passed")
    else:
        print("CP1.7: FAIL - Some tests failed")
    print(f"{'='*70}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
