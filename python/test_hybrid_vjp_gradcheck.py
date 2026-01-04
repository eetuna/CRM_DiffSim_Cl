"""
Milestone A: Hybrid VJP Gradcheck Tests

Tests the Vector-Jacobian Product implementation using torch.autograd.gradcheck.
Validates gradients w.r.t. x_t_hybrid, u_t, and theta at multiple operating points.

Constraints:
- Fast: Must complete in <60s for CI smoke tests
- Accurate: Gradcheck tolerance ~1e-4 for FD-based VJP
"""
import sys
import os
import time
import torch
import numpy as np

# Add python directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

import crm_diff_py
from control.hybrid_state_contract import (
    STATE_DIM_HYBRID, CONTROL_DIM, THETA_DIM,
    pack_hybrid_state, make_default_theta,
)
from control.step_hybrid_legacy_contract import (
    hybrid_dynamics_step, load_default_catheter_params,
)


def test_gradcheck_operating_point(
    test_name: str,
    x_t_hybrid_np: np.ndarray,
    u_t_np: np.ndarray,
    theta_np: np.ndarray,
    dt: float,
    L_inserted: float,
    params_dict: dict,
    eps: float = 1e-6,
    atol: float = 1e-4,
    rtol: float = 1e-3,
) -> bool:
    """
    Test VJP at a single operating point using torch.autograd.gradcheck.

    Args:
        test_name: Name of this test case
        x_t_hybrid_np: Initial hybrid state (9,)
        u_t_np: Control input (3,)
        theta_np: Parameter vector (3,)
        dt: Time step
        L_inserted: Insertion length
        params_dict: Catheter parameters
        eps: Finite difference step for gradcheck
        atol: Absolute tolerance
        rtol: Relative tolerance

    Returns:
        passed: True if all gradchecks pass
    """
    print(f"\n{'='*70}")
    print(f"Test: {test_name}")
    print(f"{'='*70}")
    print(f"x_t_hybrid = {x_t_hybrid_np}")
    print(f"u_t = {u_t_np}")
    print(f"theta = {theta_np}")
    print(f"dt = {dt}, L_inserted = {L_inserted}")

    # Convert to torch tensors
    x_t_hybrid = torch.from_numpy(x_t_hybrid_np.copy()).requires_grad_(True)
    u_t = torch.from_numpy(u_t_np.copy()).requires_grad_(True)
    theta = torch.from_numpy(theta_np.copy()).requires_grad_(True)

    # Test forward pass first
    try:
        x_next = hybrid_dynamics_step(
            x_t_hybrid.clone(), u_t.clone(), theta.clone(),
            dt, L_inserted, params_dict
        )
        print(f"x_next = {x_next.detach().numpy()}")
    except RuntimeError as e:
        print(f"FAIL: Forward pass failed: {e}")
        return False

    # Define functions for gradcheck
    def func_x_t(x):
        """Test gradients w.r.t. x_t_hybrid."""
        return hybrid_dynamics_step(x, u_t.detach(), theta.detach(),
                                     dt, L_inserted, params_dict)

    def func_u_t(u):
        """Test gradients w.r.t. u_t."""
        return hybrid_dynamics_step(x_t_hybrid.detach(), u, theta.detach(),
                                     dt, L_inserted, params_dict)

    def func_theta(t):
        """Test gradients w.r.t. theta."""
        return hybrid_dynamics_step(x_t_hybrid.detach(), u_t.detach(), t,
                                     dt, L_inserted, params_dict)

    results = {}

    # Test gradients w.r.t. x_t_hybrid (9D)
    print(f"\nTesting d(x_next)/d(x_t_hybrid)...")
    try:
        x_test = x_t_hybrid.clone().detach().requires_grad_(True)
        gradcheck_x = torch.autograd.gradcheck(
            func_x_t, x_test,
            eps=eps, atol=atol, rtol=rtol,
            raise_exception=False
        )
        results['x_t_hybrid'] = gradcheck_x
        status = "PASS" if gradcheck_x else "FAIL"
        print(f"  [{status}] d(x_next)/d(x_t_hybrid)")
    except Exception as e:
        print(f"  [FAIL] d(x_next)/d(x_t_hybrid): {e}")
        results['x_t_hybrid'] = False

    # Test gradients w.r.t. u_t (3D)
    print(f"\nTesting d(x_next)/d(u_t)...")
    try:
        u_test = u_t.clone().detach().requires_grad_(True)
        gradcheck_u = torch.autograd.gradcheck(
            func_u_t, u_test,
            eps=eps, atol=atol, rtol=rtol,
            raise_exception=False
        )
        results['u_t'] = gradcheck_u
        status = "PASS" if gradcheck_u else "FAIL"
        print(f"  [{status}] d(x_next)/d(u_t)")
    except Exception as e:
        print(f"  [FAIL] d(x_next)/d(u_t): {e}")
        results['u_t'] = False

    # Test gradients w.r.t. theta (3D)
    print(f"\nTesting d(x_next)/d(theta)...")
    try:
        theta_test = theta.clone().detach().requires_grad_(True)
        gradcheck_theta = torch.autograd.gradcheck(
            func_theta, theta_test,
            eps=eps, atol=atol, rtol=rtol,
            raise_exception=False
        )
        results['theta'] = gradcheck_theta
        status = "PASS" if gradcheck_theta else "FAIL"
        print(f"  [{status}] d(x_next)/d(theta)")
    except Exception as e:
        print(f"  [FAIL] d(x_next)/d(theta): {e}")
        results['theta'] = False

    # Summary for this operating point
    all_passed = all(results.values())

    print(f"\n{'-'*40}")
    if all_passed:
        print(f"PASS: {test_name}")
    else:
        failed = [k for k, v in results.items() if not v]
        print(f"FAIL: {test_name} (failed: {failed})")

    return all_passed


def test_manual_gradient_check(
    test_name: str,
    x_t_hybrid_np: np.ndarray,
    u_t_np: np.ndarray,
    theta_np: np.ndarray,
    dt: float,
    L_inserted: float,
    params_dict: dict,
    eps: float = 1e-6,
    tol: float = 1e-3,
) -> bool:
    """
    Manual finite-difference gradient check (more detailed output).

    This provides detailed error analysis when torch.autograd.gradcheck fails.
    """
    print(f"\n{'='*70}")
    print(f"Manual FD Check: {test_name}")
    print(f"{'='*70}")

    # Convert to torch tensors
    x_t_hybrid = torch.from_numpy(x_t_hybrid_np.copy()).requires_grad_(True)
    u_t = torch.from_numpy(u_t_np.copy()).requires_grad_(True)
    theta = torch.from_numpy(theta_np.copy()).requires_grad_(True)

    # Forward pass
    x_next = hybrid_dynamics_step(x_t_hybrid, u_t, theta, dt, L_inserted, params_dict)

    # Compute analytical gradient via backward pass
    grad_output = torch.ones_like(x_next)
    x_next.backward(grad_output)

    grad_x_analytical = x_t_hybrid.grad.numpy().copy()
    grad_u_analytical = u_t.grad.numpy().copy()
    grad_theta_analytical = theta.grad.numpy().copy()

    print(f"\nAnalytical gradients:")
    print(f"  d(sum(x_next))/d(x_t_hybrid) = {grad_x_analytical}")
    print(f"  d(sum(x_next))/d(u_t) = {grad_u_analytical}")
    print(f"  d(sum(x_next))/d(theta) = {grad_theta_analytical}")

    # Compute finite-difference gradients
    print(f"\nComputing FD gradients (eps={eps})...")

    def compute_fd_gradient(param_np, idx, eps, param_name):
        """Compute FD gradient for one parameter index."""
        param_plus = param_np.copy()
        param_plus[idx] += eps

        param_minus = param_np.copy()
        param_minus[idx] -= eps

        if param_name == 'x':
            x_plus = hybrid_dynamics_step(
                torch.from_numpy(param_plus),
                torch.from_numpy(u_t_np),
                torch.from_numpy(theta_np),
                dt, L_inserted, params_dict
            )
            x_minus = hybrid_dynamics_step(
                torch.from_numpy(param_minus),
                torch.from_numpy(u_t_np),
                torch.from_numpy(theta_np),
                dt, L_inserted, params_dict
            )
        elif param_name == 'u':
            x_plus = hybrid_dynamics_step(
                torch.from_numpy(x_t_hybrid_np),
                torch.from_numpy(param_plus),
                torch.from_numpy(theta_np),
                dt, L_inserted, params_dict
            )
            x_minus = hybrid_dynamics_step(
                torch.from_numpy(x_t_hybrid_np),
                torch.from_numpy(param_minus),
                torch.from_numpy(theta_np),
                dt, L_inserted, params_dict
            )
        else:  # theta
            x_plus = hybrid_dynamics_step(
                torch.from_numpy(x_t_hybrid_np),
                torch.from_numpy(u_t_np),
                torch.from_numpy(param_plus),
                dt, L_inserted, params_dict
            )
            x_minus = hybrid_dynamics_step(
                torch.from_numpy(x_t_hybrid_np),
                torch.from_numpy(u_t_np),
                torch.from_numpy(param_minus),
                dt, L_inserted, params_dict
            )

        # Central difference: d(sum(x_next))/d(param[idx])
        fd_grad = (x_plus.sum() - x_minus.sum()).item() / (2 * eps)
        return fd_grad

    # Compute FD for x_t_hybrid
    grad_x_fd = np.zeros(STATE_DIM_HYBRID)
    for i in range(STATE_DIM_HYBRID):
        grad_x_fd[i] = compute_fd_gradient(x_t_hybrid_np, i, eps, 'x')

    # Compute FD for u_t
    grad_u_fd = np.zeros(CONTROL_DIM)
    for i in range(CONTROL_DIM):
        grad_u_fd[i] = compute_fd_gradient(u_t_np, i, eps, 'u')

    # Compute FD for theta
    grad_theta_fd = np.zeros(THETA_DIM)
    for i in range(THETA_DIM):
        grad_theta_fd[i] = compute_fd_gradient(theta_np, i, eps, 'theta')

    print(f"\nFD gradients:")
    print(f"  d(sum(x_next))/d(x_t_hybrid) = {grad_x_fd}")
    print(f"  d(sum(x_next))/d(u_t) = {grad_u_fd}")
    print(f"  d(sum(x_next))/d(theta) = {grad_theta_fd}")

    # Compare
    err_x = np.abs(grad_x_analytical - grad_x_fd)
    err_u = np.abs(grad_u_analytical - grad_u_fd)
    err_theta = np.abs(grad_theta_analytical - grad_theta_fd)

    print(f"\nAbsolute errors:")
    print(f"  x_t_hybrid: max={err_x.max():.2e}, mean={err_x.mean():.2e}")
    print(f"  u_t: max={err_u.max():.2e}, mean={err_u.mean():.2e}")
    print(f"  theta: max={err_theta.max():.2e}, mean={err_theta.mean():.2e}")

    # Relative errors (avoid division by zero)
    def safe_rel_err(analytical, fd):
        denom = np.maximum(np.abs(analytical), np.abs(fd))
        denom = np.maximum(denom, 1e-10)
        return np.abs(analytical - fd) / denom

    rel_err_x = safe_rel_err(grad_x_analytical, grad_x_fd)
    rel_err_u = safe_rel_err(grad_u_analytical, grad_u_fd)
    rel_err_theta = safe_rel_err(grad_theta_analytical, grad_theta_fd)

    print(f"\nRelative errors:")
    print(f"  x_t_hybrid: max={rel_err_x.max():.2e}, mean={rel_err_x.mean():.2e}")
    print(f"  u_t: max={rel_err_u.max():.2e}, mean={rel_err_u.mean():.2e}")
    print(f"  theta: max={rel_err_theta.max():.2e}, mean={rel_err_theta.mean():.2e}")

    # Pass criteria: max relative error < tol
    passed_x = rel_err_x.max() < tol or err_x.max() < 1e-8
    passed_u = rel_err_u.max() < tol or err_u.max() < 1e-8
    passed_theta = rel_err_theta.max() < tol or err_theta.max() < 1e-8

    all_passed = passed_x and passed_u and passed_theta

    print(f"\nResults (tol={tol}):")
    print(f"  x_t_hybrid: {'PASS' if passed_x else 'FAIL'}")
    print(f"  u_t: {'PASS' if passed_u else 'FAIL'}")
    print(f"  theta: {'PASS' if passed_theta else 'FAIL'}")

    return all_passed


def main():
    print("="*70)
    print("Milestone A: Hybrid VJP Gradcheck Tests")
    print("="*70)

    start_time = time.time()

    # Load catheter parameters
    param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
    params_dict = load_default_catheter_params(param_file, config_file)

    # Test parameters
    dt = 0.01
    L_inserted = 50.0
    eps = 1e-6
    atol = 1e-4  # Relaxed for FD-based VJP
    rtol = 1e-3  # Relaxed for FD-based VJP

    print(f"\nParameters:")
    print(f"  dt = {dt}")
    print(f"  L_inserted = {L_inserted}")
    print(f"  eps = {eps}")
    print(f"  atol = {atol}")
    print(f"  rtol = {rtol}")

    # Operating points (2-3 as specified)
    operating_points = [
        (
            "OP1: Rest state",
            np.zeros(STATE_DIM_HYBRID, dtype=np.float64),
            np.zeros(CONTROL_DIM, dtype=np.float64),
            make_default_theta(),
        ),
        (
            "OP2: Actuated",
            np.zeros(STATE_DIM_HYBRID, dtype=np.float64),
            np.array([0.1, 0.0, 0.0], dtype=np.float64),
            make_default_theta(),
        ),
        (
            "OP3: Moving with scaled theta",
            np.array([0.01, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
            np.array([0.1, 0.05, 0.0], dtype=np.float64),
            np.array([1.1, 0.9, 1.05], dtype=np.float64),  # Slightly perturbed theta
        ),
    ]

    results = []

    # Run gradcheck tests
    for name, x_t, u_t, theta in operating_points:
        passed = test_gradcheck_operating_point(
            name, x_t, u_t, theta, dt, L_inserted, params_dict,
            eps=eps, atol=atol, rtol=rtol
        )
        results.append((name, passed))

    # Run manual FD check for OP2 (more detailed output)
    print("\n" + "="*70)
    print("Additional: Manual FD Validation")
    print("="*70)

    manual_passed = test_manual_gradient_check(
        "OP2 Manual FD",
        operating_points[1][1],  # x_t
        operating_points[1][2],  # u_t
        operating_points[1][3],  # theta
        dt, L_inserted, params_dict,
        eps=eps, tol=rtol
    )
    results.append(("Manual FD Validation", manual_passed))

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Hybrid VJP Gradcheck Tests")
    print("="*70)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(p for _, p in results)
    n_passed = sum(1 for _, p in results if p)
    n_total = len(results)

    print(f"\nTotal: {n_passed}/{n_total} tests passed")
    print(f"Elapsed time: {elapsed:.2f}s")
    print("="*70)

    if elapsed > 60:
        print(f"WARNING: Tests took {elapsed:.2f}s, exceeds 60s CI limit")

    if all_passed:
        print("MILESTONE A (VJP Gradcheck): PASS")
    else:
        print("MILESTONE A (VJP Gradcheck): FAIL")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
