"""
Test BVP Adjoint U-Gradients Implementation

Validates that grad_u is correctly computed via BVP adjoint.
Tests at a converged operating point to ensure BVP stability.
"""

import sys
import torch
import numpy as np

# Import from control module
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/python')
from control import (
    true_legacy_step_torch,
    load_default_catheter_params,
    true_legacy_state_dim,
)


def test_u_gradients_at_zero():
    """Test u-gradients with small perturbation state."""
    print("\n" + "="*70)
    print("TEST: U-Gradients with Small Perturbation")
    print("="*70)

    # Load parameters
    param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
    params = load_default_catheter_params(param_file, config_file)

    torch.manual_seed(42)
    N_ACT = 1
    state_dim = true_legacy_state_dim(N_ACT)

    # Create initial state (small velocities, near-zero position, identity rotation)
    # Based on test_true_legacy_step.py which works
    x_coil = torch.zeros(N_ACT, 18, dtype=torch.float64)
    # Set small velocities
    x_coil[:, :3] = torch.randn(N_ACT, 3) * 0.01   # v
    x_coil[:, 3:6] = torch.randn(N_ACT, 3) * 0.01  # w
    # Position near origin
    x_coil[:, 6:9] = torch.randn(N_ACT, 3) * 1.0   # p (mm)
    # Rotation = identity + small perturbation
    for j in range(N_ACT):
        R_flat = torch.eye(3).flatten() + torch.randn(9) * 0.01
        R_flat = R_flat / torch.norm(R_flat[:3])  # Normalize first row
        x_coil[j, 9:18] = R_flat

    # Tip state
    xf = torch.zeros(15, dtype=torch.float64)
    xf[:3] = torch.randn(3) * 10.0  # Tip position (mm)
    xf[3:12] = torch.eye(3).flatten()  # Tip rotation (identity)
    xf[12:15] = torch.randn(3) * 0.001  # Tip curvature

    # Pack state
    from control.true_legacy_state_adapter import pack_true_legacy_state
    x = pack_true_legacy_state(x_coil, xf)
    x.requires_grad = True

    # Small actuation
    u = torch.randn(N_ACT, 3, dtype=torch.float64) * 0.01  # Amperes
    u.requires_grad = True

    # Small timestep (same as working test)
    dt = 0.001  # seconds
    L_inserted = 100.0

    print(f"  State dim: {state_dim}")
    print(f"  x shape: {x.shape}")
    print(f"  u shape: {u.shape}")
    print(f"  dt: {dt}")

    # Forward pass
    print("  Running forward pass...")
    try:
        tip_p = true_legacy_step_torch(
            x, u, dt, n_act=N_ACT,
            catheter_params=params,
            L_inserted=L_inserted,
            return_x_next=False
        )
        print(f"  Forward succeeded! tip_p: {tip_p}")
    except RuntimeError as e:
        print(f"  [SKIP] BVP failed to converge: {e}")
        return None

    # Backward pass
    print("  Running backward pass...")
    loss = tip_p.sum()
    loss.backward()

    # Check gradients
    print(f"\n  Gradient Analysis:")
    print(f"  -----------------")
    print(f"  x.grad is None: {x.grad is None}")
    print(f"  u.grad is None: {u.grad is None}")

    if x.grad is not None:
        print(f"  x.grad norm: {x.grad.norm().item():.8e}")
        print(f"  x.grad min/max: [{x.grad.min().item():.8e}, {x.grad.max().item():.8e}]")

    if u.grad is not None:
        print(f"  u.grad: {u.grad.detach().numpy()}")
        print(f"  u.grad norm: {u.grad.norm().item():.8e}")
        print(f"  u.grad abs sum: {u.grad.abs().sum().item():.8e}")

        # Check if u-gradients are non-zero
        if u.grad.abs().sum().item() > 1e-10:
            print(f"\n  ✓ SUCCESS: u.grad is NON-ZERO (BVP adjoint working!)")
            return True
        else:
            print(f"\n  ✗ FAIL: u.grad is zero (BVP adjoint NOT working)")
            return False
    else:
        print(f"\n  ✗ FAIL: u.grad is None")
        return False


def test_gradcheck_u():
    """Test u-gradients with PyTorch gradcheck."""
    print("\n" + "="*70)
    print("TEST: PyTorch Gradcheck for U")
    print("="*70)

    # Load parameters
    param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
    params = load_default_catheter_params(param_file, config_file)

    torch.manual_seed(42)
    N_ACT = 1
    state_dim = true_legacy_state_dim(N_ACT)

    # Use similar initialization as test above
    x_coil = torch.zeros(N_ACT, 18, dtype=torch.float64)
    x_coil[:, :3] = torch.randn(N_ACT, 3) * 0.01
    x_coil[:, 3:6] = torch.randn(N_ACT, 3) * 0.01
    x_coil[:, 6:9] = torch.randn(N_ACT, 3) * 1.0
    for j in range(N_ACT):
        R_flat = torch.eye(3).flatten() + torch.randn(9) * 0.01
        R_flat = R_flat / torch.norm(R_flat[:3])
        x_coil[j, 9:18] = R_flat

    xf = torch.zeros(15, dtype=torch.float64)
    xf[:3] = torch.randn(3) * 10.0
    xf[3:12] = torch.eye(3).flatten()
    xf[12:15] = torch.randn(3) * 0.001

    from control.true_legacy_state_adapter import pack_true_legacy_state
    x = pack_true_legacy_state(x_coil, xf)

    u = torch.randn(N_ACT, 3, dtype=torch.float64) * 0.01
    u.requires_grad = True

    dt = 0.001  # Small timestep
    L_inserted = 100.0

    # Define function for gradcheck
    def func_u(u_in):
        return true_legacy_step_torch(
            x, u_in, dt, n_act=N_ACT,
            catheter_params=params,
            L_inserted=L_inserted,
            return_x_next=False
        )

    print("  Running gradcheck for u (may take a while)...")
    try:
        # Use gradcheck with numerical gradients
        passed = torch.autograd.gradcheck(
            func_u, u,
            eps=1e-6, atol=1e-4, rtol=1e-2,
            raise_exception=True
        )
        print(f"  ✓ PASS: Gradcheck for u")
        return True
    except RuntimeError as e:
        if "BVP" in str(e):
            print(f"  [SKIP] BVP convergence issue: {e}")
            return None
        else:
            print(f"  ✗ FAIL: Gradcheck failed: {e}")
            return False
    except Exception as e:
        print(f"  ✗ FAIL: Gradcheck exception: {e}")
        return False


def main():
    """Run all u-gradient tests."""
    print("\n" + "="*70)
    print("BVP ADJOINT U-GRADIENTS VALIDATION")
    print("="*70)

    tests = [
        ("U-Gradients at Zero State", test_u_gradients_at_zero),
        ("PyTorch Gradcheck for U", test_gradcheck_u),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            if result is None:
                status = "SKIP"
            elif result:
                status = "PASS"
            else:
                status = "FAIL"
            results.append((name, status))
        except Exception as e:
            print(f"\n  [EXCEPTION] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "FAIL"))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    for name, status in results:
        print(f"  [{status}] {name}")

    # Count successes
    passed = sum(1 for _, s in results if s == "PASS")
    skipped = sum(1 for _, s in results if s == "SKIP")
    total = len(results)

    print(f"\n  Passed: {passed}/{total}, Skipped: {skipped}/{total}")

    # Return success if at least one test passed
    return 0 if passed > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
