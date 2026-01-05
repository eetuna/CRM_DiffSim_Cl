"""
Diagnostic script to understand gradient pathways and identify double-counting.

Traces all contributions to grad_u:
1. Direct pathway: u → τ_mag → w_next
2. Implicit BVP pathway: u → (BVP residual) → λ → grad_u

Goal: Verify these don't double-count and identify correct J_yu structure.
"""

import sys
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/build')
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/python')

import numpy as np
import crm_diff_py
from control.true_legacy_state_adapter import pack_true_legacy_state

def create_test_state(n_act=1):
    """Create test state matching the failing test."""
    x_coil = np.zeros((n_act, 18))
    x_coil[:, 6:9] = [[0.0, 0.0, 0.0]]
    x_coil[:, 9:18] = np.eye(3).flatten()

    xf = np.zeros(15)
    xf[0:3] = [0.0, 0.0, 100.0]
    xf[3:12] = np.eye(3).flatten()

    x = pack_true_legacy_state(x_coil, xf)
    u = np.array([[0.1, 0.05, -0.05]] * n_act)

    params = crm_diff_py.load_cath_params('./catheterdata/CatheterParameterSet_1_dyn.txt')
    config = crm_diff_py.load_cath_config('./catheterdata/CatheterSpatialConfiguration_1.txt')

    params_dict = {
        'CathParams': params,
        'CathConfig': config,
        'L_inserted': 100.0,
        'ContactMode': crm_diff_py.ContactModeType.FREE_TIP,
        'TipConstraintPoint': np.zeros(3),
        'TipForce': np.zeros(3),
        'deltau0_initialguess': np.zeros(3),
        'IntegrationStepSize': 0.5
    }

    return x, u, 0.01, params_dict, n_act

def test_magnetic_torque_pathway():
    """Test: Does u→τ→w pathway exist? If so, where does it contribute to grad_u?"""
    print("="*80)
    print("TEST 1: Trace magnetic torque pathway u → τ → w → ...")
    print("="*80)

    n_act = 1
    x, u, dt, params_dict, n_act = create_test_state(n_act)

    # Forward pass
    x_coil = x[:n_act*18].reshape(n_act, 18)
    xf = x[n_act*18:]

    fwd_result = crm_diff_py.true_legacy_step_forward(
        x_coil, xf, u, dt, params_dict
    )

    print(f"\nForward pass completed:")
    print(f"  u (input currents): {u.flatten()}")
    print(f"  mL_next (interface moments): {fwd_result['mL_next'].flatten()}")
    print(f"  nL_next (interface forces): {fwd_result['nL_next'].flatten()}")
    print(f"  x_coil_next[3:6] (w_next, angular velocity): {fwd_result['x_coil_next'][0, 3:6]}")

    print(f"\nChecking if w_next depends on u by finite difference:")
    print(f"  (This is the DIRECT pathway: u → τ_mag → w_next)")
    eps = 1e-6
    for i in range(3):
        u_pert = u.copy()
        u_pert[0, i] += eps
        fwd_pert = crm_diff_py.true_legacy_step_forward(
            x_coil, xf, u_pert, dt, params_dict
        )
        dw_du_fd = (fwd_pert['x_coil_next'][0, 3:6] - fwd_result['x_coil_next'][0, 3:6]) / eps
        print(f"  ∂w_next/∂u[{i}] (finite diff): {dw_du_fd}")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("  If ∂τ/∂u is non-zero, then u directly affects τ (magnetic torque).")
    print("  This τ affects w_next via Euler integration: w_next = w + dt * (τ/I + ...)")
    print("  The question: Does this pathway contribute via:")
    print("    (A) grad_u_direct only (line 419-442 in CRM_TrueLegacyDynamics.cpp:419-442)?")
    print("    (B) J_yu implicit term only (line 445)?")
    print("    (C) BOTH (causing double-counting)?")
    print("="*80)

def test_bvp_residual_sensitivity():
    """Test: Does changing u affect the BVP residual? How?"""
    print("\n" + "="*80)
    print("TEST 2: Does u affect BVP residual r(mL, nL; u)?")
    print("="*80)

    n_act = 1
    x, u, dt, params_dict, n_act = create_test_state(n_act)

    x_coil = x[:n_act*18].reshape(n_act, 18)
    xf = x[n_act*18:]

    fwd_result = crm_diff_py.true_legacy_step_forward(
        x_coil, xf, u, dt, params_dict
    )

    print(f"\nBVP solution at u = {u.flatten()}:")
    print(f"  mL (interface moments): {fwd_result['mL_next'].flatten()}")
    print(f"  nL (interface forces): {fwd_result['nL_next'].flatten()}")

    print(f"\nPerturbing u and checking how mL, nL change:")
    eps = 1e-5
    for i in range(3):
        u_pert = u.copy()
        u_pert[0, i] += eps
        fwd_pert = crm_diff_py.true_legacy_step_forward(
            x_coil, xf, u_pert, dt, params_dict
        )
        dmL_du = (fwd_pert['mL_next'] - fwd_result['mL_next']) / eps
        dnL_du = (fwd_pert['nL_next'] - fwd_result['nL_next']) / eps

        print(f"\n  Component u[{i}]:")
        print(f"    ∂mL/∂u[{i}]: {dmL_du.flatten()}")
        print(f"    ∂nL/∂u[{i}]: {dnL_du.flatten()}")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("  If ∂(mL, nL)/∂u is non-zero, then the BVP solution depends on u.")
    print("  This means J_yu (BVP Jacobian ∂r/∂u) MUST be non-zero.")
    print("  The residual r = 0 at convergence, but ∂r/∂u ≠ 0 in general.")
    print("="*80)

def test_component_1_symmetry():
    """Test: Why is grad_u[1] = 0? Is it due to B field direction or geometry?"""
    print("\n" + "="*80)
    print("TEST 3: Why is component 1 zero?")
    print("="*80)

    n_act = 1
    x, u, dt, params_dict, n_act = create_test_state(n_act)

    x_coil = x[:n_act*18].reshape(n_act, 18)
    xf = x[n_act*18:]

    # Try perturbing u[1] and see if tip_p changes
    fwd_0 = crm_diff_py.true_legacy_step_forward(x_coil, xf, u, dt, params_dict)
    tip_p_0 = fwd_0['tip_p']

    eps_values = [1e-5, 1e-4, 1e-3, 1e-2]
    print(f"\nBase state: u = {u.flatten()}, tip_p = {tip_p_0.flatten()}")
    print(f"\nPerturbing u[1] by different amounts:")

    for eps in eps_values:
        u_pert = u.copy()
        u_pert[0, 1] += eps
        fwd_pert = crm_diff_py.true_legacy_step_forward(x_coil, xf, u_pert, dt, params_dict)
        dtip_p = fwd_pert['tip_p'] - tip_p_0
        sensitivity = np.linalg.norm(dtip_p) / eps
        print(f"  eps={eps:8.1e}: Δtip_p = {dtip_p}, ||Δtip_p||/eps = {sensitivity:.6e}")

    print("\n" + "="*80)
    print("CONCLUSION:")
    print("  If ||Δtip_p||/eps ≈ 0 for all eps, then u[1] truly has no effect (symmetry).")
    print("  If ||Δtip_p||/eps > 0, then VJP should be non-zero but implementation is buggy.")
    print("="*80)

if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# GRADIENT PATHWAY DIAGNOSTIC")
    print("# Goal: Identify double-counting and explain zero components")
    print("#"*80)

    test_magnetic_torque_pathway()
    test_bvp_residual_sensitivity()
    test_component_1_symmetry()

    print("\n" + "#"*80)
    print("# END OF DIAGNOSTIC")
    print("#"*80)
