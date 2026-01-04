#!/usr/bin/env python3
"""
A3.5: Verify grad_xf Fix

Tests that the pybind11 array contiguity fix resolves the grad_xf corruption bug.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import crm_diff_py

def main():
    print("="*80)
    print("A3.5: Verify grad_xf Pybind11 Fix")
    print("="*80)

    # Load params
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

    # Test state
    x_coil = np.zeros((1, 18), dtype=np.float64)
    xf = np.zeros(15, dtype=np.float64)
    R_identity = np.eye(3).flatten()
    x_coil[0, 9:18] = R_identity
    xf[3:12] = R_identity
    x_coil[0, 6:9] = [0.0, 0.0, 1.0]
    xf[:3] = [0.0, 0.0, 10.0]
    u = np.array([[0.01, 0.0, 0.0]], dtype=np.float64)

    grad_tip_p = np.array([1.0, 0.5, 0.2], dtype=np.float64)

    # Single-sample VJP
    result = crm_diff_py.true_legacy_step_vjp(
        x_coil, xf, u, 0.001, 100.0, params_dict, grad_tip_p
    )

    grad_xf = result['grad_xf']

    print(f"\nInput grad_tip_p: {grad_tip_p}")
    print(f"\nResult grad_xf:")
    print(f"  Shape: {grad_xf.shape}")
    print(f"  C_CONTIGUOUS: {grad_xf.flags['C_CONTIGUOUS']}")
    print(f"  Values: {grad_xf}")

    # Expected: first 3 elements should match grad_tip_p
    expected = np.array([1.0, 0.5, 0.2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    if np.allclose(grad_xf, expected):
        print(f"\n✓ PASS: grad_xf is CORRECT (pybind11 fix works!)")
        print(f"  First 3 elements match grad_tip_p: {grad_xf[:3]}")
        return 0
    else:
        print(f"\n✗ FAIL: grad_xf is corrupted")
        print(f"  Expected: {expected[:5]}...")
        print(f"  Got:      {grad_xf[:5]}...")
        return 1


if __name__ == "__main__":
    sys.exit(main())
