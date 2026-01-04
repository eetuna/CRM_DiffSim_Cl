#!/usr/bin/env python3
"""
P1-5: API Versioning Test

Tests that the crm_diff_py module exposes version information and
that all forward/backward entrypoints return dicts with api_version
and api_contract keys, ensuring no regressions in existing keys.

Runtime target: < 2s
"""

import sys
import os
import re
import numpy as np

# Add build directory to path for crm_diff_py import
build_dir = os.path.join(os.path.dirname(__file__), '..', 'build')
sys.path.insert(0, build_dir)

import crm_diff_py


def is_semver(version_string):
    """Check if a string matches semantic versioning (major.minor.patch)."""
    pattern = r'^\d+\.\d+\.\d+$'
    return bool(re.match(pattern, version_string))


def test_module_version_attributes():
    """Test that __version__ and __api_version__ exist and are valid semver."""
    print("TEST: Module version attributes")

    assert hasattr(crm_diff_py, '__version__'), "Module missing __version__"
    assert hasattr(crm_diff_py, '__api_version__'), "Module missing __api_version__"

    version = crm_diff_py.__version__
    api_version = crm_diff_py.__api_version__

    print(f"  __version__ = {version}")
    print(f"  __api_version__ = {api_version}")

    assert is_semver(version), f"__version__ '{version}' is not valid semver"
    assert is_semver(api_version), f"__api_version__ '{api_version}' is not valid semver"

    print("  PASSED: Version attributes are valid semver\n")


def test_check_api_compat():
    """Test the check_api_compat helper function."""
    print("TEST: check_api_compat function")

    current_version = crm_diff_py.__api_version__
    print(f"  Current API version: {current_version}")

    # Should succeed: same version
    try:
        result = crm_diff_py.check_api_compat(current_version)
        assert result is True, "check_api_compat should return True for same version"
        print(f"  PASSED: check_api_compat('{current_version}') = True")
    except RuntimeError as e:
        raise AssertionError(f"check_api_compat failed for same version: {e}")

    # Should succeed: older minor version (backward compatible)
    try:
        result = crm_diff_py.check_api_compat("1.0.0")
        assert result is True, "check_api_compat should return True for older version"
        print(f"  PASSED: check_api_compat('1.0.0') = True (backward compatible)")
    except RuntimeError as e:
        raise AssertionError(f"check_api_compat failed for older version: {e}")

    # Should fail: different major version
    try:
        crm_diff_py.check_api_compat("2.0.0")
        raise AssertionError("check_api_compat should raise RuntimeError for major version mismatch")
    except RuntimeError as e:
        assert "major version mismatch" in str(e).lower(), f"Wrong error message: {e}"
        print(f"  PASSED: check_api_compat('2.0.0') raises RuntimeError (major mismatch)")

    # Should fail: newer minor version
    try:
        crm_diff_py.check_api_compat("1.99.0")
        raise AssertionError("check_api_compat should raise RuntimeError for newer version")
    except RuntimeError as e:
        assert "too old" in str(e).lower(), f"Wrong error message: {e}"
        print(f"  PASSED: check_api_compat('1.99.0') raises RuntimeError (version too old)")

    print("  PASSED: check_api_compat validation works correctly\n")


def load_test_params():
    """Load catheter parameters and configuration for testing."""
    params_file = "./catheterdata/CatheterParameterSet_1_dyn.txt"
    config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt"

    cath_params = crm_diff_py.load_cath_params(params_file)
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

    return params_dict


def test_equilibrium_forward_versioning():
    """Test that equilibrium_forward returns dict with version info and no regressions."""
    print("TEST: equilibrium_forward versioning")

    params_dict = load_test_params()

    # Test inputs
    u = np.array([0.1, -0.05, 0.0], dtype=np.float64)
    L_inserted = 100.0

    result = crm_diff_py.equilibrium_forward(u, L_inserted, params_dict)

    # Check versioning keys are present
    assert 'api_version' in result, "equilibrium_forward missing 'api_version' key"
    assert 'api_contract' in result, "equilibrium_forward missing 'api_contract' key"

    api_version = result['api_version']
    api_contract = result['api_contract']

    print(f"  api_version = {api_version}")
    print(f"  api_contract = {api_contract}")

    assert is_semver(api_version), f"api_version '{api_version}' is not valid semver"
    assert isinstance(api_contract, str), "api_contract should be a string"
    assert len(api_contract) > 0, "api_contract should not be empty"
    assert "equilibrium" in api_contract.lower(), "api_contract should mention 'equilibrium'"

    # Check that existing keys are still present (no regressions)
    expected_keys = [
        'status', 'p_tip', 'deltau0', 'converged',
        'J_p_u0', 'J_u_u0', 'J_p_zc', 'J_u_zc', 'K_tip',
        'nl_iterations', 'final_residual', 'lu_rank', 'rel_solve_residual', 'exit_code'
    ]

    for key in expected_keys:
        assert key in result, f"equilibrium_forward missing expected key '{key}' (regression)"

    print(f"  PASSED: All expected keys present (no regressions)")
    print(f"  PASSED: equilibrium_forward returns version info\n")


def test_equilibrium_backward_versioning():
    """Test that equilibrium_backward returns dict with version info and no regressions."""
    print("TEST: equilibrium_backward versioning")

    params_dict = load_test_params()

    # Run forward pass to get cached data
    u = np.array([0.1, -0.05, 0.0], dtype=np.float64)
    L_inserted = 100.0
    fwd_result = crm_diff_py.equilibrium_forward(u, L_inserted, params_dict)

    # Test backward pass
    grad_p_tip = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    result = crm_diff_py.equilibrium_backward(fwd_result, grad_p_tip)

    # Check versioning keys are present
    assert 'api_version' in result, "equilibrium_backward missing 'api_version' key"
    assert 'api_contract' in result, "equilibrium_backward missing 'api_contract' key"

    api_version = result['api_version']
    api_contract = result['api_contract']

    print(f"  api_version = {api_version}")
    print(f"  api_contract = {api_contract}")

    assert is_semver(api_version), f"api_version '{api_version}' is not valid semver"
    assert isinstance(api_contract, str), "api_contract should be a string"
    assert "equilibrium" in api_contract.lower(), "api_contract should mention 'equilibrium'"

    # Check that existing keys are still present (no regressions)
    expected_keys = ['status', 'grad_u', 'lu_rank', 'rel_residual']

    for key in expected_keys:
        assert key in result, f"equilibrium_backward missing expected key '{key}' (regression)"

    print(f"  PASSED: All expected keys present (no regressions)")
    print(f"  PASSED: equilibrium_backward returns version info\n")


def test_dynamics_forward_versioning():
    """Test that dynamics_forward returns dict with version info and no regressions."""
    print("TEST: dynamics_forward versioning")

    params_dict = load_test_params()

    # Test inputs
    x_t = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    u_t = np.array([0.1, -0.05, 0.0], dtype=np.float64)
    dt = 0.01
    L_inserted = 100.0

    result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)

    # Check versioning keys are present
    assert 'api_version' in result, "dynamics_forward missing 'api_version' key"
    assert 'api_contract' in result, "dynamics_forward missing 'api_contract' key"

    api_version = result['api_version']
    api_contract = result['api_contract']

    print(f"  api_version = {api_version}")
    print(f"  api_contract = {api_contract}")

    assert is_semver(api_version), f"api_version '{api_version}' is not valid semver"
    assert isinstance(api_contract, str), "api_contract should be a string"
    assert "dynamics" in api_contract.lower(), "api_contract should mention 'dynamics'"

    # Check that existing keys are still present (no regressions)
    expected_keys = [
        'status', 'x_next', 'p_tip', 'u_tip',
        'J_G_xnext', 'J_G_xt', 'J_G_ut',
        'J_p_u0', 'J_p_ut',
        'M', 'D', 'K',
        'u_t_cached', 'dt_cached', 'L_inserted_cached',
        'K_tip_cached', 'J_u_zc_cached',
        'lu_rank', 'rel_solve_residual', 'solve_residual', 'converged', 'exit_code'
    ]

    for key in expected_keys:
        assert key in result, f"dynamics_forward missing expected key '{key}' (regression)"

    print(f"  PASSED: All expected keys present (no regressions)")
    print(f"  PASSED: dynamics_forward returns version info\n")


def test_dynamics_backward_versioning():
    """Test that dynamics_backward returns dict with version info and no regressions."""
    print("TEST: dynamics_backward versioning")

    params_dict = load_test_params()

    # Run forward pass to get cached data
    x_t = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    u_t = np.array([0.1, -0.05, 0.0], dtype=np.float64)
    dt = 0.01
    L_inserted = 100.0
    fwd_result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)

    # Test backward pass
    grad_x_next = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    result = crm_diff_py.dynamics_backward(fwd_result, grad_x_next, params_dict)

    # Check versioning keys are present
    assert 'api_version' in result, "dynamics_backward missing 'api_version' key"
    assert 'api_contract' in result, "dynamics_backward missing 'api_contract' key"

    api_version = result['api_version']
    api_contract = result['api_contract']

    print(f"  api_version = {api_version}")
    print(f"  api_contract = {api_contract}")

    assert is_semver(api_version), f"api_version '{api_version}' is not valid semver"
    assert isinstance(api_contract, str), "api_contract should be a string"
    assert "dynamics" in api_contract.lower(), "api_contract should mention 'dynamics'"

    # Check that existing keys are still present (no regressions)
    expected_keys = ['status', 'grad_x_t', 'grad_u_t', 'lu_rank', 'rel_residual']

    for key in expected_keys:
        assert key in result, f"dynamics_backward missing expected key '{key}' (regression)"

    print(f"  PASSED: All expected keys present (no regressions)")
    print(f"  PASSED: dynamics_backward returns version info\n")


if __name__ == '__main__':
    print("=" * 70)
    print("P1-5: API Versioning Test")
    print("=" * 70)
    print()

    try:
        test_module_version_attributes()
        test_check_api_compat()
        test_equilibrium_forward_versioning()
        test_equilibrium_backward_versioning()
        test_dynamics_forward_versioning()
        test_dynamics_backward_versioning()

        print("=" * 70)
        print("ALL TESTS PASSED")
        print("=" * 70)
        sys.exit(0)

    except AssertionError as e:
        print(f"\nTEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
