"""
CP2.4: PyTorch gradcheck Validation for Dynamics Primitive

Uses torch.autograd.gradcheck to validate VJP implementation for one-step dynamics.
Tests at 3 operating points with deterministic behavior.
"""
import sys
import os
import torch
import numpy as np

# Add python directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

import crm_dynamics_torch
import crm_diff_py


def test_operating_point(test_name, x_t_np, u_t_np, dt, L_inserted, params_dict,
                         eps=1e-6, atol=1e-6, rtol=1e-4):
    """
    Test a single operating point with torch.autograd.gradcheck.

    Args:
        test_name: str
        x_t_np: ndarray [6] - initial state
        u_t_np: ndarray [3] - control inputs
        dt: float - time step
        L_inserted: float - insertion length
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
    print(f"x_t = {x_t_np}")
    print(f"u_t = {u_t_np}")
    print(f"dt = {dt}")
    print(f"L_inserted = {L_inserted}")

    # Convert to torch tensors (float64, requires_grad)
    x_t = torch.from_numpy(x_t_np).clone().requires_grad_(True)
    u_t = torch.from_numpy(u_t_np).clone().requires_grad_(True)

    # Compute forward pass to check convergence
    try:
        x_next = crm_dynamics_torch.dynamics_step(
            x_t.clone(), u_t.clone(), dt, L_inserted, params_dict
        )
        print(f"x_next = {x_next.detach().numpy()}")
    except RuntimeError as e:
        print(f"✗ FAIL: Forward pass failed: {e}")
        return False

    # Define function for gradcheck (needs to accept tuple of inputs)
    def func_x_t(x):
        """Test gradients w.r.t. x_t with u_t fixed."""
        return crm_dynamics_torch.dynamics_step(x, u_t.detach(), dt, L_inserted, params_dict)

    def func_u_t(u):
        """Test gradients w.r.t. u_t with x_t fixed."""
        return crm_dynamics_torch.dynamics_step(x_t.detach(), u, dt, L_inserted, params_dict)

    print(f"\nRunning torch.autograd.gradcheck...")
    print(f"  eps={eps}, atol={atol}, rtol={rtol}")

    # Test gradients w.r.t. x_t
    print(f"\nTesting ∂x_next/∂x_t...")
    try:
        x_t_test = x_t.clone().detach().requires_grad_(True).to(dtype=torch.float64)

        gradcheck_x_t = torch.autograd.gradcheck(
            func_x_t,
            x_t_test,
            eps=eps,
            atol=atol,
            rtol=rtol,
            raise_exception=False
        )

        if gradcheck_x_t:
            print(f"  ✓ PASS: gradcheck for ∂x_next/∂x_t succeeded")
        else:
            print(f"  ✗ FAIL: gradcheck for ∂x_next/∂x_t failed")

    except Exception as e:
        print(f"  ✗ FAIL: gradcheck exception for ∂x_next/∂x_t: {e}")
        import traceback
        traceback.print_exc()
        gradcheck_x_t = False

    # Test gradients w.r.t. u_t
    print(f"\nTesting ∂x_next/∂u_t...")
    try:
        u_t_test = u_t.clone().detach().requires_grad_(True).to(dtype=torch.float64)

        gradcheck_u_t = torch.autograd.gradcheck(
            func_u_t,
            u_t_test,
            eps=eps,
            atol=atol,
            rtol=rtol,
            raise_exception=False
        )

        if gradcheck_u_t:
            print(f"  ✓ PASS: gradcheck for ∂x_next/∂u_t succeeded")
        else:
            print(f"  ✗ FAIL: gradcheck for ∂x_next/∂u_t failed")

    except Exception as e:
        print(f"  ✗ FAIL: gradcheck exception for ∂x_next/∂u_t: {e}")
        import traceback
        traceback.print_exc()
        gradcheck_u_t = False

    # Overall pass requires both to pass
    overall_pass = gradcheck_x_t and gradcheck_u_t

    print()
    if overall_pass:
        print(f"✓ PASS: {test_name} (both ∂x_next/∂x_t and ∂x_next/∂u_t)")
    else:
        print(f"✗ FAIL: {test_name}")

    return overall_pass


def main():
    print("="*70)
    print("CP2.4: PyTorch gradcheck Validation for Dynamics")
    print("="*70)
    print()

    # Load parameters (same as CP2.3)
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
    dt = 0.01  # 10 ms timestep
    L_inserted = 50.0  # mm
    eps = 1e-6
    atol = 1e-5  # Relaxed for nested FD (our backward uses FD internally)
    rtol = 1e-3  # Relaxed for nested FD

    print(f"Parameters:")
    print(f"  dt = {dt} s")
    print(f"  L_inserted = {L_inserted} mm")
    print(f"  eps = {eps}")
    print(f"  atol = {atol}")
    print(f"  rtol = {rtol}")
    print(f"  dtype = torch.float64")

    # Operating points (matching CP2.2 specification)
    test_cases = [
        ("OP1: Rest",
         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
         np.array([0.0, 0.0, 0.0], dtype=np.float64)),

        ("OP2: Actuated",
         np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
         np.array([0.1, 0.0, 0.0], dtype=np.float64)),

        ("OP3: Moving",
         np.array([0.01, 0.0, 0.0, 0.1, 0.0, 0.0], dtype=np.float64),
         np.array([0.1, 0.0, 0.0], dtype=np.float64)),
    ]

    results = []
    for test_name, x_t, u_t in test_cases:
        passed = test_operating_point(
            test_name, x_t, u_t, dt, L_inserted, params_dict, eps, atol, rtol
        )
        results.append((test_name, passed))

    # Summary
    print(f"\n{'='*70}")
    print("CP2.4 SUMMARY")
    print(f"{'='*70}")

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    overall_pass = all(passed for _, passed in results)

    print(f"\n{'='*70}")
    if overall_pass:
        print("CP2.4: PASS - All gradcheck tests passed")
    else:
        print("CP2.4: FAIL - Some tests failed")
    print(f"{'='*70}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
