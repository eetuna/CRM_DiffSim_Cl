"""
A3.5 Debug: Pinpoint mismatch between batched and looped VJP

This test compares intermediate values to identify where batched != looped.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import crm_diff_py
from control import load_default_catheter_params


def main():
    print("="*80)
    print("A3.5 DEBUG: Batched vs Looped Intermediate Comparison")
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

    dt = 0.001
    L_inserted = 100.0
    N_ACT = 1

    # Simple state
    x_coil = np.zeros((N_ACT, 18), dtype=np.float64)
    xf = np.zeros(15, dtype=np.float64)

    R_identity = np.eye(3).flatten()
    x_coil[0, 9:18] = R_identity
    xf[3:12] = R_identity

    x_coil[0, 6:9] = [0.0, 0.0, 1.0]
    xf[:3] = [0.0, 0.0, 10.0]

    u = np.array([[0.01, 0.0, 0.0]], dtype=np.float64)

    # Single upstream gradient for comparison
    grad_tip_p = np.array([1.0, 0.5, 0.2], dtype=np.float64)

    print("\nTest setup:")
    print(f"  x_coil shape: {x_coil.shape}")
    print(f"  xf shape: {xf.shape}")
    print(f"  u shape: {u.shape}")
    print(f"  grad_tip_p: {grad_tip_p}")
    print(f"  dt: {dt}s")
    print(f"  L_inserted: {L_inserted}mm")

    # Run single-sample VJP
    print("\n" + "-"*80)
    print("SINGLE-SAMPLE VJP (looped baseline)")
    print("-"*80)
    result_single = crm_diff_py.true_legacy_step_vjp(
        x_coil, xf, u, dt, L_inserted, params_dict, grad_tip_p
    )
    print(f"[PYTHON] result_single keys: {list(result_single.keys())}")
    print(f"[PYTHON] result_single['grad_xf'] id: {id(result_single['grad_xf'])}")
    print(f"[PYTHON] result_single['grad_xf'] dtype: {result_single['grad_xf'].dtype}")
    print(f"[PYTHON] result_single['grad_xf'] shape: {result_single['grad_xf'].shape}")
    print(f"[PYTHON] result_single['grad_xf'] immediately after call: {result_single['grad_xf']}")
    print(f"[PYTHON] input xf (for comparison): {xf}")
    print(f"[PYTHON] input grad_tip_p (for comparison): {grad_tip_p}")

    grad_x_coil_single = result_single['grad_x_coil']
    grad_xf_single = result_single['grad_xf']
    grad_u_single = result_single['grad_u']
    print(f"[PYTHON] grad_xf_single after extraction: {grad_xf_single}")

    print(f"grad_x_coil shape: {grad_x_coil_single.shape}")
    print(f"grad_xf shape: {grad_xf_single.shape}")
    print(f"grad_u shape: {grad_u_single.shape}")
    print(f"\ngrad_x_coil[0, :6] (velocities): {grad_x_coil_single[0, :6]}")
    print(f"grad_x_coil[0, 6:9] (position): {grad_x_coil_single[0, 6:9]}")
    print(f"grad_x_coil[0, 9:18] (rotation): {grad_x_coil_single[0, 9:18]}")
    print(f"\ngrad_xf[:3] (tip_p): {grad_xf_single[:3]}")
    print(f"grad_xf[3:12] (tip_R): {grad_xf_single[3:12]}")
    print(f"grad_xf[12:15] (tip_u): {grad_xf_single[12:15]}")
    print(f"\ngrad_u: {grad_u_single}")

    # Run batched VJP with num_rhs=1
    print("\n" + "-"*80)
    print("BATCHED VJP (num_rhs=1)")
    print("-"*80)
    grad_tip_p_batch = grad_tip_p.reshape(1, 3)
    result_batched = crm_diff_py.true_legacy_step_vjp_batched(
        x_coil, xf, u, dt, L_inserted, params_dict, grad_tip_p_batch
    )

    grad_x_coil_batched = result_batched['grad_x_coil'][0]  # Extract first RHS
    grad_xf_batched = result_batched['grad_xf'][0]
    grad_u_batched = result_batched['grad_u'][0]

    print(f"grad_x_coil shape: {grad_x_coil_batched.shape}")
    print(f"grad_xf shape: {grad_xf_batched.shape}")
    print(f"grad_u shape: {grad_u_batched.shape}")
    print(f"\ngrad_x_coil[0, :6] (velocities): {grad_x_coil_batched[0, :6]}")
    print(f"grad_x_coil[0, 6:9] (position): {grad_x_coil_batched[0, 6:9]}")
    print(f"grad_x_coil[0, 9:18] (rotation): {grad_x_coil_batched[0, 9:18]}")
    print(f"\ngrad_xf[:3] (tip_p): {grad_xf_batched[:3]}")
    print(f"grad_xf[3:12] (tip_R): {grad_xf_batched[3:12]}")
    print(f"grad_xf[12:15] (tip_u): {grad_xf_batched[12:15]}")
    print(f"\ngrad_u: {grad_u_batched}")

    # Compare
    print("\n" + "="*80)
    print("COMPARISON: Batched vs Single")
    print("="*80)

    diff_x_coil = grad_x_coil_batched - grad_x_coil_single
    diff_xf = grad_xf_batched - grad_xf_single
    diff_u = grad_u_batched - grad_u_single

    print(f"\ngrad_u difference:")
    print(f"  Max abs: {np.max(np.abs(diff_u)):.6e}")
    print(f"  Matches: {np.allclose(grad_u_batched, grad_u_single, atol=1e-10)}")

    print(f"\ngrad_xf difference:")
    print(f"  Max abs: {np.max(np.abs(diff_xf)):.6e}")
    print(f"  diff[:3] (tip_p): {diff_xf[:3]}")
    print(f"  diff[3:12] (tip_R): {diff_xf[3:12]}")
    print(f"  diff[12:15] (tip_u): {diff_xf[12:15]}")
    print(f"  Matches: {np.allclose(grad_xf_batched, grad_xf_single, atol=1e-10)}")

    print(f"\ngrad_x_coil difference:")
    print(f"  Max abs: {np.max(np.abs(diff_x_coil)):.6e}")
    print(f"  diff[0, :6] (velocities): {diff_x_coil[0, :6]}")
    print(f"  diff[0, 6:9] (position): {diff_x_coil[0, 6:9]}")
    print(f"  diff[0, 9:18] (rotation): {diff_x_coil[0, 9:18]}")
    print(f"  Matches: {np.allclose(grad_x_coil_batched, grad_x_coil_single, atol=1e-10)}")

    # Check for NaN/Inf
    print(f"\nNumerical health:")
    print(f"  Single grad_x_coil has NaN: {np.any(np.isnan(grad_x_coil_single))}")
    print(f"  Single grad_x_coil has Inf: {np.any(np.isinf(grad_x_coil_single))}")
    print(f"  Batched grad_x_coil has NaN: {np.any(np.isnan(grad_x_coil_batched))}")
    print(f"  Batched grad_x_coil has Inf: {np.any(np.isinf(grad_x_coil_batched))}")

    # Determine if test passes
    all_match = (
        np.allclose(grad_u_batched, grad_u_single, atol=1e-10) and
        np.allclose(grad_xf_batched, grad_xf_single, atol=1e-10) and
        np.allclose(grad_x_coil_batched, grad_x_coil_single, atol=1e-10)
    )

    print("\n" + "="*80)
    if all_match:
        print("✓ PASS: Batched matches single-sample VJP")
        return 0
    else:
        print("✗ FAIL: Batched differs from single-sample VJP")
        return 1


if __name__ == "__main__":
    sys.exit(main())
