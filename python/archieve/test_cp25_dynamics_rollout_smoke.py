"""
CP2.5: Multi-step Rollout + Backprop Smoke Test

Tests differentiable multi-step rollouts through PyTorch autograd:
- OP-A: Zero control (u=0) for 20 steps
- OP-B: Constant control (u=[0.1,0,0]) for 20 steps
- OP-C: Sinusoidal control (u=[0.1*sin(2πt/T),0,0]) for 20 steps

Acceptance criteria:
- All states remain finite (no NaN/Inf)
- States remain bounded (||x|| < 1e3)
- Forward passes succeed (status==0)
- Backward pass produces finite gradients
- Non-zero controls produce non-zero gradients
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


def test_rollout(rollout_name, x0_np, control_fn, T, dt, L_inserted, params_dict,
                 state_bound=1e3, check_nonzero_grad=True):
    """
    Test a single rollout with differentiable backprop.

    Args:
        rollout_name: str - name of this rollout
        x0_np: ndarray [6] - initial state
        control_fn: callable(t, T) -> ndarray [3] - control function
        T: int - number of steps
        dt: float - time step
        L_inserted: float - insertion length
        params_dict: catheter parameters
        state_bound: float - maximum allowed state norm
        check_nonzero_grad: bool - whether to expect non-zero gradients

    Returns:
        passed: bool
    """
    print(f"\n{'='*70}")
    print(f"Rollout: {rollout_name}")
    print(f"{'='*70}")
    print(f"x0 = {x0_np}")
    print(f"T = {T} steps")
    print(f"dt = {dt} s")
    print(f"L_inserted = {L_inserted} mm")
    print()

    # Initialize control sequence as a tensor with requires_grad
    # Shape: [T, 3]
    u_seq_np = np.array([control_fn(t, T) for t in range(T)], dtype=np.float64)
    u_seq = torch.from_numpy(u_seq_np).requires_grad_(True)

    # Initial state
    x_t = torch.from_numpy(x0_np).clone()

    # Rollout forward
    trajectory = [x_t.clone()]
    tip_positions = []

    passed = True

    print(f"Rolling forward {T} steps...")
    for t in range(T):
        u_t = u_seq[t]

        # Take one step
        try:
            x_next = crm_dynamics_torch.dynamics_step(
                x_t, u_t, dt, L_inserted, params_dict
            )
        except RuntimeError as e:
            print(f"✗ FAIL: Forward step {t} failed: {e}")
            return False

        # Check finiteness
        if not torch.all(torch.isfinite(x_next)):
            print(f"✗ FAIL: Step {t}: x_next contains non-finite values: {x_next}")
            passed = False
            break

        # Check boundedness
        x_norm = torch.linalg.norm(x_next).item()
        if x_norm >= state_bound:
            print(f"✗ FAIL: Step {t}: ||x|| = {x_norm:.6e} exceeds bound {state_bound}")
            passed = False
            break

        # Optionally check status via a direct C++ call (lightweight check every 5 steps)
        if t % 5 == 0:
            x_np = x_t.detach().numpy()
            u_np = u_t.detach().numpy()
            try:
                result = crm_diff_py.dynamics_forward(x_np, u_np, dt, L_inserted, params_dict)
                if result['status'] != 0:
                    print(f"✗ FAIL: Step {t}: C++ forward status = {result['status']}")
                    passed = False
                    break
                # Store tip position for loss computation
                tip_positions.append(torch.from_numpy(result['p_tip'].copy()))
            except Exception as e:
                print(f"✗ FAIL: Step {t}: C++ forward exception: {e}")
                passed = False
                break

        trajectory.append(x_next.clone())
        x_t = x_next

    if not passed:
        print(f"✗ FAIL: {rollout_name} - Forward rollout failed")
        return False

    print(f"  ✓ Forward rollout succeeded")
    print(f"  Final state x_T = {x_t.detach().numpy()}")

    # Compute loss: sum of squared state norms
    # L = sum_t ||x_t||^2
    loss = sum(torch.sum(x**2) for x in trajectory)
    loss_value = loss.item()

    print(f"\nLoss computation:")
    print(f"  L = sum_t ||x_t||^2 = {loss_value:.6e}")

    # Backward pass
    print(f"\nBackward pass...")
    try:
        loss.backward()
    except Exception as e:
        print(f"✗ FAIL: Backward pass raised exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check gradients
    if u_seq.grad is None:
        print(f"✗ FAIL: u_seq.grad is None after backward()")
        return False

    grad_u = u_seq.grad.detach().numpy()

    # Check finiteness
    if not np.all(np.isfinite(grad_u)):
        print(f"✗ FAIL: grad_u contains non-finite values")
        return False

    grad_norm = np.linalg.norm(grad_u)
    print(f"  ||grad_u|| = {grad_norm:.6e}")

    # Check non-zero gradient for actuated cases
    if check_nonzero_grad:
        if grad_norm < 1e-12:
            print(f"✗ FAIL: Expected non-zero gradient but ||grad_u|| = {grad_norm:.6e}")
            passed = False
        else:
            print(f"  ✓ Gradient is non-zero (as expected)")
    else:
        # For zero control, gradient may be tiny or zero
        if grad_norm < 1e-12:
            print(f"  Note: ||grad_u|| ≈ 0 (expected for zero control)")
        else:
            print(f"  Note: ||grad_u|| = {grad_norm:.6e}")

    print()
    if passed:
        print(f"✓ PASS: {rollout_name}")
        print(f"  Summary: T={T}, loss={loss_value:.6e}, ||grad_u||={grad_norm:.6e}")
    else:
        print(f"✗ FAIL: {rollout_name}")

    return passed


def main():
    print("="*70)
    print("CP2.5: Multi-step Rollout + Backprop Smoke Test")
    print("="*70)
    print()

    # Load parameters (same as CP2.3/CP2.4)
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

    # Rollout parameters
    T = 20  # number of steps
    dt = 0.01  # 10 ms timestep
    L_inserted = 50.0  # mm

    print(f"Rollout configuration:")
    print(f"  T = {T} steps")
    print(f"  dt = {dt} s")
    print(f"  L_inserted = {L_inserted} mm")
    print(f"  Total time = {T * dt} s")

    # Define control functions
    def zero_control(t, T):
        """OP-A: Zero control"""
        return np.array([0.0, 0.0, 0.0], dtype=np.float64)

    def constant_control(t, T):
        """OP-B: Constant control"""
        return np.array([0.1, 0.0, 0.0], dtype=np.float64)

    def sinusoidal_control(t, T):
        """OP-C: Sinusoidal control"""
        amplitude = 0.1
        freq = 2.0 * np.pi / T
        return np.array([amplitude * np.sin(freq * t), 0.0, 0.0], dtype=np.float64)

    # Initial states
    # OP-A and OP-B: start from rest (zeros)
    # OP-C: start with small perturbation to avoid trivial trajectory
    x0_rest = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    x0_perturbed = np.array([0.01, 0.0, 0.0, 0.1, 0.0, 0.0], dtype=np.float64)

    # Test cases
    test_cases = [
        ("OP-A: Zero control (u=0)",
         x0_rest, zero_control, False),  # expect zero/tiny gradient

        ("OP-B: Constant control (u=[0.1,0,0])",
         x0_rest, constant_control, True),  # expect non-zero gradient

        ("OP-C: Sinusoidal control (u=[0.1*sin(2πt/T),0,0])",
         x0_perturbed, sinusoidal_control, True),  # expect non-zero gradient
    ]

    results = []
    for rollout_name, x0, control_fn, check_nonzero_grad in test_cases:
        passed = test_rollout(
            rollout_name, x0, control_fn, T, dt, L_inserted, params_dict,
            state_bound=1e3, check_nonzero_grad=check_nonzero_grad
        )
        results.append((rollout_name, passed))

    # Summary
    print(f"\n{'='*70}")
    print("CP2.5 SUMMARY")
    print(f"{'='*70}")

    for rollout_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {rollout_name}")

    overall_pass = all(passed for _, passed in results)

    print(f"\n{'='*70}")
    if overall_pass:
        print("CP2.5: PASS - All rollout tests passed")
        print("  - All states remained finite and bounded")
        print("  - Forward passes succeeded")
        print("  - Backward passes produced valid gradients")
    else:
        print("CP2.5: FAIL - Some tests failed")
    print(f"{'='*70}")

    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
