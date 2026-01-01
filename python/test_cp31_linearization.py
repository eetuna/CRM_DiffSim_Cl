"""
CP3.1: Linearization Validation (Python + PyTorch)

Validates that:
1. PyTorch jacobian() extracts correct Jacobians A, B
2. PyTorch Jacobians match C++ Jacobians within numerical precision
3. Linearization error is acceptable for iLQR use
"""

import sys
import numpy as np
import torch
import crm_diff_py
from crm_dynamics_torch import dynamics_step, load_default_catheter_params


def extract_jacobians_pytorch(x_t, u_t, dt, L_inserted, params_dict):
    """
    Extract Jacobians A = ∂x_next/∂x_t and B = ∂x_next/∂u_t using PyTorch.

    Returns:
        A: (6, 6) numpy array
        B: (6, 3) numpy array
    """
    x_t_torch = torch.tensor(x_t, dtype=torch.float64, requires_grad=True)
    u_t_torch = torch.tensor(u_t, dtype=torch.float64, requires_grad=True)

    # Extract A = ∂x_next/∂x_t
    A = torch.autograd.functional.jacobian(
        lambda x: dynamics_step(x, u_t_torch, dt, L_inserted, params_dict),
        x_t_torch
    ).numpy()

    # Extract B = ∂x_next/∂u_t
    B = torch.autograd.functional.jacobian(
        lambda u: dynamics_step(x_t_torch, u, dt, L_inserted, params_dict),
        u_t_torch
    ).numpy()

    return A, B


def extract_jacobians_cpp(x_t, u_t, dt, L_inserted, params_dict):
    """
    Extract Jacobians using C++ dynamics_backward with VJP.

    Returns:
        A: (6, 6) numpy array
        B: (6, 3) numpy array
    """
    # Forward pass
    result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)

    if result['status'] != 0 or result['lu_rank'] < 6:
        raise RuntimeError(f"Forward pass failed: status={result['status']}, rank={result['lu_rank']}")

    A = np.zeros((6, 6))
    B = np.zeros((6, 3))

    # Extract via VJP with canonical basis vectors
    for k in range(6):
        grad_x_next = np.zeros(6)
        grad_x_next[k] = 1.0

        bwd_result = crm_diff_py.dynamics_backward(result, grad_x_next, params_dict)

        if bwd_result['status'] != 0 or bwd_result['lu_rank'] < 6:
            raise RuntimeError(f"Backward pass failed: status={bwd_result['status']}, rank={bwd_result['lu_rank']}")

        # VJP gives row k of Jacobian
        A[k, :] = bwd_result['grad_x_t']
        B[k, :] = bwd_result['grad_u_t']

    return A, B


def test_operating_point(name, x_t, u_t, dt, L_inserted, params_dict):
    """
    Test linearization accuracy at a single operating point.

    Returns:
        True if all tests pass, False otherwise
    """
    print(f"Operating Point: {name}")
    print(f"  x_t = {x_t}")
    print(f"  u_t = {u_t}")
    print()

    # Extract Jacobians via PyTorch
    A_py, B_py = extract_jacobians_pytorch(x_t, u_t, dt, L_inserted, params_dict)

    # Extract Jacobians via C++
    A_cpp, B_cpp = extract_jacobians_cpp(x_t, u_t, dt, L_inserted, params_dict)

    # Compare PyTorch vs C++
    A_diff = np.linalg.norm(A_py - A_cpp, 'fro')
    B_diff = np.linalg.norm(B_py - B_cpp, 'fro')
    A_norm = np.linalg.norm(A_cpp, 'fro')
    B_norm = np.linalg.norm(B_cpp, 'fro')

    A_rel_err = A_diff / max(A_norm, 1e-12)
    B_rel_err = B_diff / max(B_norm, 1e-12)

    print("  Jacobian Comparison (PyTorch vs C++):")
    print(f"    ||A_py - A_cpp||_F = {A_diff:.6e}")
    print(f"    ||A_cpp||_F        = {A_norm:.6e}")
    print(f"    Relative error A   = {A_rel_err:.6e}")
    print()
    print(f"    ||B_py - B_cpp||_F = {B_diff:.6e}")
    print(f"    ||B_cpp||_F        = {B_norm:.6e}")
    print(f"    Relative error B   = {B_rel_err:.6e}")
    print()

    # Acceptance: relative error < 1e-6
    threshold = 1e-6
    pass_A = A_rel_err < threshold
    pass_B = B_rel_err < threshold

    if pass_A and pass_B:
        print(f"  PASS: PyTorch Jacobians match C++ (rel_err < {threshold})")
    else:
        print(f"  FAIL: PyTorch Jacobians do NOT match C++ (rel_err >= {threshold})")
    print()

    # Linearization accuracy test (same as C++)
    dx = np.array([5e-7, 5e-7, 5e-7, 5e-6, 5e-6, 5e-6])
    du = np.array([5e-5, 5e-5, 5e-5])

    norm_dx = np.linalg.norm(dx)
    norm_du = np.linalg.norm(du)

    # Compute f(x_t, u_t)
    result_base = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)
    if result_base['status'] != 0:
        print("  FAIL: Base forward pass failed")
        return False
    x_base = result_base['x_next']

    # Compute f(x_t + dx, u_t + du)
    x_pert = x_t + dx
    u_pert = u_t + du
    result_pert = crm_diff_py.dynamics_forward(x_pert, u_pert, dt, L_inserted, params_dict)
    if result_pert['status'] != 0:
        print("  FAIL: Perturbed forward pass failed")
        return False
    x_actual = result_pert['x_next']

    # Linearized prediction
    x_linear = x_base + A_cpp @ dx + B_cpp @ du

    # Error
    error = x_actual - x_linear
    error_norm = np.linalg.norm(error)
    rel_error = error_norm / norm_dx

    print("  Linearization Test:")
    print(f"    ||dx|| = {norm_dx:.6e}")
    print(f"    ||du|| = {norm_du:.6e}")
    print(f"    ||x_actual - x_linear|| = {error_norm:.6e}")
    print(f"    Relative error (||error|| / ||dx||) = {rel_error:.6e}")
    print()

    # Acceptance: relative error < 0.1
    lin_threshold = 0.1
    pass_lin = rel_error < lin_threshold

    if pass_lin:
        print(f"  PASS: Linearization accurate (rel_error < {lin_threshold})")
    else:
        print(f"  FAIL: Linearization inaccurate (rel_error >= {lin_threshold})")
    print()

    return pass_A and pass_B and pass_lin


def main():
    print("CP3.1 Linearization Validation Test (Python + PyTorch)")
    print("=======================================================")
    print()

    # Load parameters
    param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
    params_dict = load_default_catheter_params(param_file, config_file)

    dt = 0.01  # s
    L_inserted = 50.0  # mm (explicit, not from dataset)

    # Operating points
    x_op1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u_op1 = np.array([0.0, 0.0, 0.0])

    x_op2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u_op2 = np.array([0.1, 0.0, 0.0])

    x_op3 = np.array([0.01, 0.0, 0.0, 0.1, 0.0, 0.0])
    u_op3 = np.array([0.1, 0.0, 0.0])

    # Test all operating points
    pass1 = test_operating_point("OP1 - Rest", x_op1, u_op1, dt, L_inserted, params_dict)
    pass2 = test_operating_point("OP2 - Actuated", x_op2, u_op2, dt, L_inserted, params_dict)
    pass3 = test_operating_point("OP3 - Moving", x_op3, u_op3, dt, L_inserted, params_dict)

    # Overall verdict
    print("=" * 55)
    print("Overall Results:")
    print(f"  OP1 (Rest):     {'PASS' if pass1 else 'FAIL'}")
    print(f"  OP2 (Actuated): {'PASS' if pass2 else 'FAIL'}")
    print(f"  OP3 (Moving):   {'PASS' if pass3 else 'FAIL'}")
    print()

    all_pass = pass1 and pass2 and pass3
    if all_pass:
        print("CP3.1 PASS: All tests validated")
        return 0
    else:
        print("CP3.1 FAIL: Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
