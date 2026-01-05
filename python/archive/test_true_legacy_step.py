"""
A0 Tests: TRUE Legacy Step Wrapper

End-to-end tests for the TRUE legacy stepping function (DynamicsBVP → DYNSolverIVP).
Tests execute the real C++ stepping path.

Ground truth: docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md
"""

import sys
import torch
import numpy as np

from control.true_legacy_step import true_legacy_step
from control.true_legacy_state_adapter import (
    true_legacy_state_dim,
    pack_true_legacy_state,
)
from control.step_legacy_contract import load_default_catheter_params


# Default catheter parameter files
CATH_PARAMS_FILE = "./catheterdata/CatheterParameterSet_1_dyn.txt"
CATH_CONFIG_FILE = "./catheterdata/CatheterSpatialConfiguration_1.txt"


def test_unbatched_n1_basic():
    """Test unbatched execution with N=1 (single coil)."""
    print("\n" + "="*70)
    print("Test 1: Unbatched N=1 Basic Execution")
    print("="*70)

    torch.manual_seed(42)
    n_act = 1

    # Create initial state (small velocities, near-zero position, identity rotation)
    state_dim = true_legacy_state_dim(n_act)
    x_coil = torch.zeros(n_act, 18, dtype=torch.float64)
    # Set small velocities
    x_coil[:, :3] = torch.randn(n_act, 3) * 0.01   # v
    x_coil[:, 3:6] = torch.randn(n_act, 3) * 0.01  # w
    # Position near origin
    x_coil[:, 6:9] = torch.randn(n_act, 3) * 1.0   # p (mm)
    # Rotation = identity + small perturbation
    for j in range(n_act):
        R_flat = torch.eye(3).flatten() + torch.randn(9) * 0.01
        R_flat = R_flat / torch.norm(R_flat[:3])  # Normalize first row
        x_coil[j, 9:18] = R_flat

    # Tip state
    xf = torch.zeros(15, dtype=torch.float64)
    xf[:3] = torch.randn(3) * 10.0  # Tip position (mm)
    xf[3:12] = torch.eye(3).flatten()  # Tip rotation (identity)
    xf[12:15] = torch.randn(3) * 0.001  # Tip curvature

    x = pack_true_legacy_state(x_coil, xf)

    # Small actuation
    u = torch.randn(n_act, 3, dtype=torch.float64) * 0.01  # Amperes

    # Small timestep
    dt = 0.001  # seconds

    # Load params
    params = load_default_catheter_params(CATH_PARAMS_FILE, CATH_CONFIG_FILE)

    try:
        # Call step
        x_next, obs = true_legacy_step(x, u, dt, n_act=n_act, catheter_params=params)

        # Validate output shapes
        if x_next.shape != x.shape:
            print(f"  [FAIL] x_next shape {x_next.shape}, expected {x.shape}")
            return False

        if obs['tip_p'].shape != (3,):
            print(f"  [FAIL] tip_p shape {obs['tip_p'].shape}, expected (3,)")
            return False

        if obs['tip_R'].shape != (9,):
            print(f"  [FAIL] tip_R shape {obs['tip_R'].shape}, expected (9,)")
            return False

        if obs['tip_u'].shape != (3,):
            print(f"  [FAIL] tip_u shape {obs['tip_u'].shape}, expected (3,)")
            return False

        # Check for finite values
        if not torch.all(torch.isfinite(x_next)):
            print(f"  [FAIL] x_next contains non-finite values")
            return False

        if not torch.all(torch.isfinite(obs['tip_p'])):
            print(f"  [FAIL] tip_p contains non-finite values")
            return False

        print(f"  Converged: {obs['converged']}")
        print(f"  Tip position: {obs['tip_p'].numpy()}")
        print(f"  Tip position norm: {torch.norm(obs['tip_p']).item():.2f} mm")

        print("  [PASS] Unbatched N=1 basic execution")
        return True

    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batched_n1():
    """Test batched execution with N=1."""
    print("\n" + "="*70)
    print("Test 2: Batched N=1 Execution")
    print("="*70)

    torch.manual_seed(43)
    n_act = 1
    batch_size = 2

    # Create batched state
    state_dim = true_legacy_state_dim(n_act)
    x_coil = torch.zeros(batch_size, n_act, 18, dtype=torch.float64)
    for b in range(batch_size):
        x_coil[b, :, :3] = torch.randn(n_act, 3) * 0.01   # v
        x_coil[b, :, 3:6] = torch.randn(n_act, 3) * 0.01  # w
        x_coil[b, :, 6:9] = torch.randn(n_act, 3) * 1.0   # p
        for j in range(n_act):
            R_flat = torch.eye(3).flatten() + torch.randn(9) * 0.01
            x_coil[b, j, 9:18] = R_flat

    xf = torch.zeros(batch_size, 15, dtype=torch.float64)
    for b in range(batch_size):
        xf[b, :3] = torch.randn(3) * 10.0
        xf[b, 3:12] = torch.eye(3).flatten()
        xf[b, 12:15] = torch.randn(3) * 0.001

    x_list = [pack_true_legacy_state(x_coil[b], xf[b]) for b in range(batch_size)]
    x = torch.stack(x_list)

    u = torch.randn(batch_size, n_act, 3, dtype=torch.float64) * 0.01
    dt = 0.001

    params = load_default_catheter_params(CATH_PARAMS_FILE, CATH_CONFIG_FILE)

    try:
        x_next, obs = true_legacy_step(x, u, dt, n_act=n_act, catheter_params=params)

        # Validate shapes
        if x_next.shape != (batch_size, state_dim):
            print(f"  [FAIL] x_next shape {x_next.shape}, expected ({batch_size}, {state_dim})")
            return False

        if obs['tip_p'].shape != (batch_size, 3):
            print(f"  [FAIL] tip_p shape {obs['tip_p'].shape}, expected ({batch_size}, 3)")
            return False

        if obs['tip_R'].shape != (batch_size, 9):
            print(f"  [FAIL] tip_R shape {obs['tip_R'].shape}, expected ({batch_size}, 9)")
            return False

        # Check finite
        if not torch.all(torch.isfinite(x_next)):
            print(f"  [FAIL] x_next contains non-finite values")
            return False

        print(f"  Batch size: {batch_size}")
        print(f"  Converged: {obs['converged'].numpy()}")
        print(f"  Tip positions shape: {obs['tip_p'].shape}")

        print("  [PASS] Batched N=1 execution")
        return True

    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_warmstart_passthrough():
    """Test that warm-start guesses are passed through and updated."""
    print("\n" + "="*70)
    print("Test 3: Warm-start Pass-through")
    print("="*70)

    torch.manual_seed(44)
    n_act = 1

    # Create state
    state_dim = true_legacy_state_dim(n_act)
    x_coil = torch.zeros(n_act, 18, dtype=torch.float64)
    x_coil[:, :3] = torch.randn(n_act, 3) * 0.01
    x_coil[:, 3:6] = torch.randn(n_act, 3) * 0.01
    x_coil[:, 6:9] = torch.randn(n_act, 3) * 1.0
    for j in range(n_act):
        x_coil[j, 9:18] = torch.eye(3).flatten()

    xf = torch.zeros(15, dtype=torch.float64)
    xf[:3] = torch.randn(3) * 10.0
    xf[3:12] = torch.eye(3).flatten()

    x = pack_true_legacy_state(x_coil, xf)
    u = torch.randn(n_act, 3, dtype=torch.float64) * 0.01
    dt = 0.001

    # Provide initial warm-start guess
    warmstart = {
        'mL_guess': torch.randn(n_act, 3, dtype=torch.float64) * 0.001,
        'nL_guess': torch.randn(n_act, 3, dtype=torch.float64) * 0.001,
    }

    params = load_default_catheter_params(CATH_PARAMS_FILE, CATH_CONFIG_FILE)

    try:
        x_next, obs = true_legacy_step(
            x, u, dt, n_act=n_act, catheter_params=params, warmstart=warmstart
        )

        # Check that warmstart_next is in obs
        if 'warmstart_next' not in obs:
            print(f"  [FAIL] warmstart_next not in obs")
            return False

        if 'mL_guess' not in obs['warmstart_next']:
            print(f"  [FAIL] mL_guess not in warmstart_next")
            return False

        if 'nL_guess' not in obs['warmstart_next']:
            print(f"  [FAIL] nL_guess not in warmstart_next")
            return False

        # Check shapes
        if obs['warmstart_next']['mL_guess'].shape != (n_act, 3):
            print(f"  [FAIL] mL_guess shape {obs['warmstart_next']['mL_guess'].shape}")
            return False

        if obs['warmstart_next']['nL_guess'].shape != (n_act, 3):
            print(f"  [FAIL] nL_guess shape {obs['warmstart_next']['nL_guess'].shape}")
            return False

        # Check finite
        if not torch.all(torch.isfinite(obs['warmstart_next']['mL_guess'])):
            print(f"  [FAIL] mL_guess contains non-finite values")
            return False

        print(f"  Warm-start mL magnitude: {torch.norm(obs['warmstart_next']['mL_guess']).item():.3e}")
        print(f"  Warm-start nL magnitude: {torch.norm(obs['warmstart_next']['nL_guess']).item():.3e}")

        print("  [PASS] Warm-start pass-through")
        return True

    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_output_shapes_all_observables():
    """Test that all observables have correct shapes."""
    print("\n" + "="*70)
    print("Test 4: Output Shapes - All Observables")
    print("="*70)

    torch.manual_seed(45)
    n_act = 1

    state_dim = true_legacy_state_dim(n_act)
    x_coil = torch.zeros(n_act, 18, dtype=torch.float64)
    xf = torch.zeros(15, dtype=torch.float64)
    xf[3:12] = torch.eye(3).flatten()

    x = pack_true_legacy_state(x_coil, xf)
    u = torch.zeros(n_act, 3, dtype=torch.float64)
    dt = 0.001

    params = load_default_catheter_params(CATH_PARAMS_FILE, CATH_CONFIG_FILE)

    try:
        x_next, obs = true_legacy_step(
            x, u, dt, n_act=n_act, catheter_params=params, return_orientation=True
        )

        # Check all observables
        required_keys = ['tip_p', 'tip_R', 'tip_u', 'converged', 'warmstart_next']
        for key in required_keys:
            if key not in obs:
                print(f"  [FAIL] Missing observable: {key}")
                return False

        # Check types and shapes
        checks = [
            ('tip_p', torch.Tensor, (3,)),
            ('tip_R', torch.Tensor, (9,)),
            ('tip_u', torch.Tensor, (3,)),
            ('converged', bool, None),
        ]

        for key, expected_type, expected_shape in checks:
            if expected_type == torch.Tensor:
                if not isinstance(obs[key], torch.Tensor):
                    print(f"  [FAIL] {key} is not a Tensor")
                    return False
                if obs[key].shape != expected_shape:
                    print(f"  [FAIL] {key} shape {obs[key].shape}, expected {expected_shape}")
                    return False
            else:
                if not isinstance(obs[key], expected_type):
                    print(f"  [FAIL] {key} type {type(obs[key])}, expected {expected_type}")
                    return False

        print("  All observables present with correct shapes")
        print("  [PASS] Output shapes - all observables")
        return True

    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_finite_outputs_small_dt():
    """Test that outputs are finite for small dt."""
    print("\n" + "="*70)
    print("Test 5: Finite Outputs (Small dt)")
    print("="*70)

    torch.manual_seed(46)
    n_act = 1

    # Create reasonable initial state
    x_coil = torch.zeros(n_act, 18, dtype=torch.float64)
    x_coil[:, 9:18] = torch.eye(3).flatten()  # Identity rotation

    xf = torch.zeros(15, dtype=torch.float64)
    xf[:3] = torch.tensor([0.0, 0.0, 50.0])  # Tip at 50mm
    xf[3:12] = torch.eye(3).flatten()

    x = pack_true_legacy_state(x_coil, xf)
    u = torch.zeros(n_act, 3, dtype=torch.float64)
    dt = 0.0001  # Very small dt

    params = load_default_catheter_params(CATH_PARAMS_FILE, CATH_CONFIG_FILE)

    try:
        x_next, obs = true_legacy_step(x, u, dt, n_act=n_act, catheter_params=params)

        # Check all outputs for finiteness
        if not torch.all(torch.isfinite(x_next)):
            nan_count = torch.sum(~torch.isfinite(x_next)).item()
            print(f"  [FAIL] x_next has {nan_count} non-finite values")
            return False

        for key in ['tip_p', 'tip_R', 'tip_u']:
            if not torch.all(torch.isfinite(obs[key])):
                print(f"  [FAIL] {key} contains non-finite values")
                return False

        for key in ['mL_guess', 'nL_guess']:
            if not torch.all(torch.isfinite(obs['warmstart_next'][key])):
                print(f"  [FAIL] warmstart_next['{key}'] contains non-finite values")
                return False

        print(f"  All outputs are finite")
        print(f"  State norm: {torch.norm(x_next).item():.2f}")
        print("  [PASS] Finite outputs (small dt)")
        return True

    except Exception as e:
        print(f"  [FAIL] Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all TRUE legacy step tests."""
    print("\n" + "="*70)
    print("A0 TRUE LEGACY STEP END-TO-END TESTS")
    print("="*70)
    print("\nThese tests execute the real DynamicsBVP → DYNSolverIVP stepping path.")

    tests = [
        ("Unbatched N=1 Basic Execution", test_unbatched_n1_basic),
        ("Batched N=1 Execution", test_batched_n1),
        ("Warm-start Pass-through", test_warmstart_passthrough),
        ("Output Shapes - All Observables", test_output_shapes_all_observables),
        ("Finite Outputs (Small dt)", test_finite_outputs_small_dt),
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
