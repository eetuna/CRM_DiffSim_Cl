"""
Milestone A: Hybrid State Contract Roundtrip Tests

Tests:
1. Pack/unpack roundtrip for hybrid state
2. Forward pass at multiple operating points
3. Legacy compatibility (hybrid state agrees with legacy state dynamics)
"""
import sys
import os
import numpy as np

# Add python directory to path
sys.path.insert(0, os.path.dirname(__file__))

# Add build directory to path
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

import crm_diff_py
from control.hybrid_state_contract import (
    HybridState, pack_hybrid_state, unpack_hybrid_state,
    hybrid_to_legacy, legacy_to_hybrid,
    STATE_DIM_HYBRID, STATE_DIM_DYNAMICS, STATE_DIM_OBSERVABLE,
    CONTROL_DIM, THETA_DIM,
    validate_hybrid_state, validate_control, validate_theta,
    make_default_theta,
)
from control.step_hybrid_legacy_contract import (
    step_hybrid_legacy_contract, load_default_catheter_params,
)


def test_pack_unpack_roundtrip():
    """Test that pack/unpack are inverses."""
    print("\n" + "="*70)
    print("Test: Pack/Unpack Roundtrip")
    print("="*70)

    # Create random hybrid state
    np.random.seed(42)
    u_0 = np.random.randn(3) * 0.01
    v_0 = np.random.randn(3) * 0.1
    p_tip = np.random.randn(3) * 10

    # Pack
    x_hybrid = pack_hybrid_state(u_0, v_0, p_tip)
    print(f"Original: u_0={u_0}, v_0={v_0}, p_tip={p_tip}")
    print(f"Packed: x_hybrid={x_hybrid}")

    # Unpack
    u_0_r, v_0_r, p_tip_r = unpack_hybrid_state(x_hybrid)
    print(f"Unpacked: u_0={u_0_r}, v_0={v_0_r}, p_tip={p_tip_r}")

    # Check
    err_u0 = np.linalg.norm(u_0 - u_0_r)
    err_v0 = np.linalg.norm(v_0 - v_0_r)
    err_ptip = np.linalg.norm(p_tip - p_tip_r)

    print(f"Errors: u_0={err_u0:.2e}, v_0={err_v0:.2e}, p_tip={err_ptip:.2e}")

    passed = err_u0 < 1e-14 and err_v0 < 1e-14 and err_ptip < 1e-14

    if passed:
        print("PASS: Pack/Unpack roundtrip")
    else:
        print("FAIL: Pack/Unpack roundtrip")

    return passed


def test_hybrid_state_class():
    """Test HybridState dataclass methods."""
    print("\n" + "="*70)
    print("Test: HybridState Class Methods")
    print("="*70)

    np.random.seed(43)
    u_0 = np.random.randn(3).astype(np.float64) * 0.01
    v_0 = np.random.randn(3).astype(np.float64) * 0.1
    p_tip = np.random.randn(3).astype(np.float64) * 10

    # Create via constructor
    hs = HybridState(u_0=u_0, v_0=v_0, p_tip=p_tip)

    # Pack to vector
    x_packed = hs.to_packed()
    print(f"Packed vector: {x_packed}")

    # Create from packed
    hs2 = HybridState.from_packed(x_packed)
    print(f"Reconstructed: u_0={hs2.u_0}, v_0={hs2.v_0}, p_tip={hs2.p_tip}")

    # Check agreement
    err = np.linalg.norm(hs.to_packed() - hs2.to_packed())
    print(f"Roundtrip error: {err:.2e}")

    # Test to_legacy_state
    x_legacy = hs.to_legacy_state()
    print(f"Legacy state: {x_legacy}")
    assert x_legacy.shape == (STATE_DIM_DYNAMICS,), f"Bad legacy shape: {x_legacy.shape}"

    # Test from_legacy_and_observable
    hs3 = HybridState.from_legacy_and_observable(x_legacy, p_tip)
    err2 = np.linalg.norm(hs.to_packed() - hs3.to_packed())
    print(f"from_legacy_and_observable error: {err2:.2e}")

    passed = err < 1e-14 and err2 < 1e-14

    if passed:
        print("PASS: HybridState class methods")
    else:
        print("FAIL: HybridState class methods")

    return passed


def test_validation_utilities():
    """Test input validation."""
    print("\n" + "="*70)
    print("Test: Validation Utilities")
    print("="*70)

    passed = True

    # Good inputs should not raise
    try:
        x_hybrid = np.zeros(9, dtype=np.float64)
        validate_hybrid_state(x_hybrid)
        print("  validate_hybrid_state(good): OK")
    except Exception as e:
        print(f"  validate_hybrid_state(good): FAIL - {e}")
        passed = False

    # Bad shape should raise
    try:
        x_bad = np.zeros(8, dtype=np.float64)
        validate_hybrid_state(x_bad)
        print("  validate_hybrid_state(bad shape): FAIL - should have raised")
        passed = False
    except ValueError:
        print("  validate_hybrid_state(bad shape): OK - raised ValueError")

    # Bad dtype should raise
    try:
        x_bad = np.zeros(9, dtype=np.float32)
        validate_hybrid_state(x_bad)
        print("  validate_hybrid_state(bad dtype): FAIL - should have raised")
        passed = False
    except ValueError:
        print("  validate_hybrid_state(bad dtype): OK - raised ValueError")

    # Good control
    try:
        u_t = np.zeros(CONTROL_DIM, dtype=np.float64)
        validate_control(u_t)
        print("  validate_control(good): OK")
    except Exception as e:
        print(f"  validate_control(good): FAIL - {e}")
        passed = False

    # Good theta
    try:
        theta = make_default_theta()
        validate_theta(theta)
        print("  validate_theta(good): OK")
    except Exception as e:
        print(f"  validate_theta(good): FAIL - {e}")
        passed = False

    if passed:
        print("PASS: Validation utilities")
    else:
        print("FAIL: Validation utilities")

    return passed


def test_forward_pass_at_rest():
    """Test forward pass at rest state."""
    print("\n" + "="*70)
    print("Test: Forward Pass at Rest")
    print("="*70)

    # Load catheter parameters
    param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
    params_dict = load_default_catheter_params(param_file, config_file)

    # Rest state
    u_0 = np.zeros(3, dtype=np.float64)
    v_0 = np.zeros(3, dtype=np.float64)
    p_tip = np.zeros(3, dtype=np.float64)  # Will be overwritten by equilibrium

    x_t_hybrid = pack_hybrid_state(u_0, v_0, p_tip)
    u_t = np.zeros(CONTROL_DIM, dtype=np.float64)
    theta = make_default_theta()

    dt = 0.01
    L_inserted = 50.0

    print(f"x_t_hybrid = {x_t_hybrid}")
    print(f"u_t = {u_t}")
    print(f"theta = {theta}")
    print(f"dt = {dt}, L_inserted = {L_inserted}")

    # Call forward
    result = step_hybrid_legacy_contract(x_t_hybrid, u_t, dt, L_inserted, theta, params_dict)

    print(f"\nResult:")
    print(f"  status = {result.status}")
    print(f"  x_next_hybrid = {result.x_next_hybrid}")
    print(f"  x_next_legacy = {result.x_next_legacy}")
    print(f"  p_tip_next = {result.p_tip_next}")
    print(f"  success = {result.success}")

    passed = result.success

    if passed:
        print("PASS: Forward pass at rest")
    else:
        print(f"FAIL: Forward pass at rest (status={result.status})")

    return passed


def test_forward_pass_actuated():
    """Test forward pass with actuation."""
    print("\n" + "="*70)
    print("Test: Forward Pass with Actuation")
    print("="*70)

    # Load catheter parameters
    param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
    params_dict = load_default_catheter_params(param_file, config_file)

    # Rest state with non-zero actuation
    x_t_hybrid = np.zeros(STATE_DIM_HYBRID, dtype=np.float64)
    u_t = np.array([0.1, 0.0, 0.0], dtype=np.float64)  # 0.1 A on first actuator
    theta = make_default_theta()

    dt = 0.01
    L_inserted = 50.0

    print(f"x_t_hybrid = {x_t_hybrid}")
    print(f"u_t = {u_t}")
    print(f"theta = {theta}")
    print(f"dt = {dt}, L_inserted = {L_inserted}")

    # Call forward
    result = step_hybrid_legacy_contract(x_t_hybrid, u_t, dt, L_inserted, theta, params_dict)

    print(f"\nResult:")
    print(f"  status = {result.status}")
    print(f"  x_next_hybrid = {result.x_next_hybrid}")
    print(f"  x_next_legacy = {result.x_next_legacy}")
    print(f"  p_tip_next = {result.p_tip_next}")
    print(f"  success = {result.success}")

    # Check that state changed (actuation should cause motion)
    state_change = np.linalg.norm(result.x_next_legacy)
    print(f"  state change norm = {state_change:.4e}")

    passed = result.success and state_change > 1e-10

    if passed:
        print("PASS: Forward pass with actuation")
    else:
        print(f"FAIL: Forward pass with actuation")

    return passed


def test_legacy_compatibility():
    """Test that hybrid dynamics agrees with legacy dynamics."""
    print("\n" + "="*70)
    print("Test: Legacy Compatibility")
    print("="*70)

    # Load catheter parameters
    param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
    params_dict = load_default_catheter_params(param_file, config_file)

    # Test state
    np.random.seed(44)
    x_t_legacy = np.array([0.01, 0.0, 0.0, 0.1, 0.0, 0.0], dtype=np.float64)
    u_t = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    theta = make_default_theta()

    dt = 0.01
    L_inserted = 50.0

    # Run legacy dynamics directly
    print("Running legacy dynamics...")
    legacy_result = crm_diff_py.dynamics_forward(x_t_legacy, u_t, dt, L_inserted, params_dict)
    print(f"  status = {legacy_result['status']}")
    print(f"  x_next = {legacy_result['x_next']}")
    print(f"  p_tip = {legacy_result['p_tip']}")

    if legacy_result['status'] != 0:
        print("FAIL: Legacy dynamics failed")
        return False

    # Create hybrid state with arbitrary p_tip (will be ignored)
    p_tip_dummy = np.zeros(3, dtype=np.float64)
    x_t_hybrid = legacy_to_hybrid(x_t_legacy, p_tip_dummy)

    # Run hybrid dynamics
    print("\nRunning hybrid dynamics...")
    hybrid_result = step_hybrid_legacy_contract(
        x_t_hybrid, u_t, dt, L_inserted, theta, params_dict
    )
    print(f"  status = {hybrid_result.status}")
    print(f"  x_next_legacy = {hybrid_result.x_next_legacy}")
    print(f"  p_tip_next = {hybrid_result.p_tip_next}")

    if not hybrid_result.success:
        print("FAIL: Hybrid dynamics failed")
        return False

    # Compare x_next (legacy vs hybrid)
    err_x_next = np.linalg.norm(legacy_result['x_next'] - hybrid_result.x_next_legacy)
    err_p_tip = np.linalg.norm(legacy_result['p_tip'] - hybrid_result.p_tip_next)

    print(f"\nComparison:")
    print(f"  x_next error = {err_x_next:.2e}")
    print(f"  p_tip error = {err_p_tip:.2e}")

    passed = err_x_next < 1e-12 and err_p_tip < 1e-12

    if passed:
        print("PASS: Legacy compatibility")
    else:
        print("FAIL: Legacy compatibility")

    return passed


def test_multi_step_rollout():
    """Test multi-step trajectory rollout."""
    print("\n" + "="*70)
    print("Test: Multi-Step Rollout (10 steps)")
    print("="*70)

    # Load catheter parameters
    param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"
    params_dict = load_default_catheter_params(param_file, config_file)

    # Initial state
    x_t = np.zeros(STATE_DIM_HYBRID, dtype=np.float64)
    u_t = np.array([0.1, 0.0, 0.0], dtype=np.float64)
    theta = make_default_theta()

    dt = 0.01
    L_inserted = 50.0
    n_steps = 10

    print(f"Initial state: {x_t}")
    print(f"Control: {u_t}")
    print(f"Steps: {n_steps}")

    trajectory = [x_t.copy()]

    for step in range(n_steps):
        result = step_hybrid_legacy_contract(x_t, u_t, dt, L_inserted, theta, params_dict)

        if not result.success:
            print(f"FAIL: Rollout failed at step {step} (status={result.status})")
            return False

        x_t = result.x_next_hybrid
        trajectory.append(x_t.copy())

    trajectory = np.array(trajectory)
    print(f"\nTrajectory shape: {trajectory.shape}")
    print(f"Final state: {trajectory[-1]}")

    # Check trajectory is non-degenerate
    total_change = np.linalg.norm(trajectory[-1] - trajectory[0])
    print(f"Total state change: {total_change:.4e}")

    passed = total_change > 1e-6

    if passed:
        print("PASS: Multi-step rollout")
    else:
        print("FAIL: Multi-step rollout (no state change)")

    return passed


def main():
    print("="*70)
    print("Milestone A: Hybrid State Contract Roundtrip Tests")
    print("="*70)

    tests = [
        ("Pack/Unpack Roundtrip", test_pack_unpack_roundtrip),
        ("HybridState Class", test_hybrid_state_class),
        ("Validation Utilities", test_validation_utilities),
        ("Forward Pass at Rest", test_forward_pass_at_rest),
        ("Forward Pass Actuated", test_forward_pass_actuated),
        ("Legacy Compatibility", test_legacy_compatibility),
        ("Multi-Step Rollout", test_multi_step_rollout),
    ]

    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
        except Exception as e:
            print(f"\nEXCEPTION in {name}: {e}")
            import traceback
            traceback.print_exc()
            passed = False
        results.append((name, passed))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Hybrid State Contract Roundtrip Tests")
    print("="*70)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(p for _, p in results)
    n_passed = sum(1 for _, p in results if p)
    n_total = len(results)

    print(f"\nTotal: {n_passed}/{n_total} tests passed")
    print("="*70)

    if all_passed:
        print("MILESTONE A (Contract Roundtrip): PASS")
    else:
        print("MILESTONE A (Contract Roundtrip): FAIL")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
