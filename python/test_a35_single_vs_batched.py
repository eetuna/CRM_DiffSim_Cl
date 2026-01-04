#!/usr/bin/env python3
"""
A3.5 Regression Test: Single-sample VJP == Batched[0] VJP

Verifies that single-sample and batched VJP produce identical results.
This test ensures the pybind11 binding bug (non-contiguous array) is fixed.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))
import crm_diff_py

def main():
    print("="*80)
    print("A3.5 Regression Test: Single vs Batched VJP")
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
    N_ACT = 1
    x_coil = np.zeros((N_ACT, 18), dtype=np.float64)
    xf = np.zeros(15, dtype=np.float64)
    R_identity = np.eye(3).flatten()
    x_coil[0, 9:18] = R_identity
    xf[3:12] = R_identity
    x_coil[0, 6:9] = [0.0, 0.0, 1.0]
    xf[:3] = [0.0, 0.0, 10.0]
    u = np.array([[0.01, 0.0, 0.0]], dtype=np.float64)

    # Test with 3 different upstream gradients
    test_cases = [
        ("Standard", np.array([1.0, 0.5, 0.2])),
        ("Unit X", np.array([1.0, 0.0, 0.0])),
        ("Random", np.array([0.7, -0.3, 0.9])),
    ]

    all_passed = True

    for name, grad_tip_p in test_cases:
        print(f"\nTest Case: {name}")
        print(f"  grad_tip_p: {grad_tip_p}")

        # Single-sample VJP
        result_single = crm_diff_py.true_legacy_step_vjp(
            x_coil, xf, u, 0.001, 100.0, params_dict, grad_tip_p
        )

        # Batched VJP with num_rhs=1
        result_batch = crm_diff_py.true_legacy_step_vjp_batched(
            x_coil, xf, u, 0.001, 100.0, params_dict, grad_tip_p.reshape(1, 3)
        )

        # Extract results (copy to avoid any view/memory aliasing issues)
        grad_xf_single = result_single['grad_xf'].copy()
        grad_x_coil_single = result_single['grad_x_coil'].copy()
        grad_u_single = result_single['grad_u'].copy()

        grad_xf_batch = result_batch['grad_xf'][0].copy()
        grad_x_coil_batch = result_batch['grad_x_coil'][0].copy()
        grad_u_batch = result_batch['grad_u'][0].copy()

        # Compare with tight tolerances (same algorithm, minor floating-point differences)
        atol = 1e-10  # Tight but allows for minor numerical differences
        rtol = 1e-10

        grad_xf_match = np.allclose(grad_xf_single, grad_xf_batch, atol=atol, rtol=rtol)
        grad_x_coil_match = np.allclose(grad_x_coil_single, grad_x_coil_batch, atol=atol, rtol=rtol)
        grad_u_match = np.allclose(grad_u_single, grad_u_batch, atol=atol, rtol=rtol)

        # Compute max differences
        diff_xf = np.max(np.abs(grad_xf_single - grad_xf_batch))
        diff_x_coil = np.max(np.abs(grad_x_coil_single - grad_x_coil_batch))
        diff_u = np.max(np.abs(grad_u_single - grad_u_batch))

        print(f"  grad_xf match: {grad_xf_match} (max diff: {diff_xf:.2e})")
        print(f"  grad_x_coil match: {grad_x_coil_match} (max diff: {diff_x_coil:.2e})")
        print(f"  grad_u match: {grad_u_match} (max diff: {diff_u:.2e})")

        case_passed = grad_xf_match and grad_x_coil_match and grad_u_match
        if case_passed:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL")
            if not grad_xf_match:
                print(f"    Single grad_xf: {grad_xf_single[:5]}...")
                print(f"    Batch grad_xf:  {grad_xf_batch[:5]}...")
            if not grad_x_coil_match:
                print(f"    Single grad_x_coil shape: {grad_x_coil_single.shape}")
                print(f"    Batch grad_x_coil shape: {grad_x_coil_batch.shape}")
                print(f"    Single has NaN: {np.any(np.isnan(grad_x_coil_single))}")
                print(f"    Batch has NaN: {np.any(np.isnan(grad_x_coil_batch))}")
                diff_per_elem = np.abs(grad_x_coil_single - grad_x_coil_batch)
                print(f"    Max diff per element: {diff_per_elem.flatten()}")
                max_idx = np.unravel_index(np.argmax(diff_per_elem), diff_per_elem.shape)
                print(f"    Max diff at index {max_idx}: single={grad_x_coil_single[max_idx]:.6e}, batch={grad_x_coil_batch[max_idx]:.6e}")
            all_passed = False

    print("\n" + "="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("Single-sample VJP == Batched[0] VJP (to floating-point precision)")
        return 0
    else:
        print("✗ TESTS FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
