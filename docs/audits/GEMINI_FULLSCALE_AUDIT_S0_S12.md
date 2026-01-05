# GEMINI FULLSCALE AUDIT S0-S12 (FULLSTATE)

**Date:** 2026-01-05
**Scope:** FULLSTATE forward/VJP/linearization (S0-S12)
**Author:** GEMINI
**Status:** CRITICAL FAILURE

## 1. Executive Summary

The FULLSTATE differentiable simulation stack is **functionally broken** for gradient computation (VJP) and linearization. While the forward pass (`true_legacy_step_forward`) correctly delegates to the authoritative `DynamicsBVP → DYNSolverIVP` chain, the backward pass (`true_legacy_step_backward`) relies on a mismatched and partially heuristic Jacobian implementation that produces garbage gradients.

**Key Findings:**
1.  **Analytic IVP Jacobian Mismatch:** `CRMSolverIVPJacobian` returns derivatives w.r.t. `u` and `ftip`, but the backward pass interprets them as derivatives w.r.t. `nL`, `pL`, and `RL`. This fundamentally invalidates the VJP.
2.  **Broken BVP Jacobians:** The "analytic" `J_yy` and `J_yu` computations use a templated AD path that drops derivatives (via `.val` access) and relies on unstable explicit integrators.
3.  **Heuristic State Sensitivity:** `J_yx` (state sensitivity) is not computed analytically but approximated using a hardcoded `coupling_scale` heuristic.
4.  **Test Failure:** `tests/test_fullstate_vjp_gradcheck_u.py` fails with >100% relative error, confirming the implementation is incorrect.

## 2. Runtime Truth Map

### Forward Pass (Correct)
*   **Entry:** `python/crm_bindings.cpp` -> `py_true_legacy_step_forward`
*   **Wrapper:** `src/CRM_TrueLegacyDynamics.cpp` -> `true_legacy_step_forward`
*   **Core:**
    *   `src/CoilDynamics_Defs.cpp` -> `DynamicsBVP` (Shooting Method)
    *   `src/CoilDynamics_Defs.cpp` -> `DYNSolverIVP` (Integration)

### Backward Pass (Broken)
*   **Entry:** `python/crm_bindings.cpp` -> `py_true_legacy_step_vjp`
*   **Wrapper:** `src/CRM_TrueLegacyDynamics.cpp` -> `true_legacy_step_backward`
*   **Core:**
    *   **IVP Jacobians:** `src/CRM_IVPJacobian.cpp` -> `CRMSolverIVPJacobian` (Analytic)
    *   **BVP Jacobians:** `src/CRM_BVPJacobian.cpp` -> `compute_bvp_jacobians_full_analytic`

## 3. Ranked Bug List

| Rank | Severity | Component | Location | Issue Description |
| :--- | :--- | :--- | :--- | :--- |
| **1** | **CRITICAL** | `CRMSolverIVPJacobian` | `src/CRM_IVPJacobian.cpp:45` | **Return Value Mismatch:** Function returns `{JBVP_p_z, JBVP_ws_z, ...}` where `z` corresponds to `u` (actuation). `true_legacy_step_backward` unpacks this as `[J_u, J_n, J_p, J_R, J_ftip]`. `J_n` (force sensitivity) gets assigned `JBVP_ws_z` (rotation sensitivity to actuation). This renders `v_nL` calculation mathematically meaningless. |
| **2** | **CRITICAL** | `compute_bvp_jacobians_full_analytic` | `src/CRM_BVPJacobian.cpp:337` | **Heuristic Gradient:** `J_yx` (sensitivity of BVP residual to state) is not computed analytically. It uses a hardcoded `coupling_scale = dt * 0.1` and a sparse identity assumption. This is an ad-hoc approximation, not a derivative. |
| **3** | **CRITICAL** | `compute_bvp_jacobians_fmad` | `src/CRM_BVPJacobian.cpp:312` | **Broken AD Path:** The templated `DYNNLEquation_T` path used for `J_yy`/`J_yu` contains multiple points where derivatives are dropped (explicit `.val` access in flexible segments) or exploded (unstable explicit integrator), as documented in `CODEX_AUDIT_DUAL_JYU_VJP.md`. |
| **4** | **HIGH** | `true_legacy_step_backward` | `src/CRM_TrueLegacyDynamics.cpp:241` | **Missing `mL` Sensitivity:** The backward pass assumes `v_mL = 0` (sensitivity to interface moments is zero), despite `mL` being a primary BVP unknown that affects coil dynamics and thus `xf`. |
| **5** | **HIGH** | `true_legacy_step_backward` | `src/CRM_TrueLegacyDynamics.cpp:322` | **Inconsistent `grad_u`:** The single-sample VJP ignores the direct `u -> w_next` (angular acceleration) pathway, assuming it's negligible. The batched version (`true_legacy_step_backward_batched`) explicitly calculates this term (`grad_u_direct`). |
| **6** | **MEDIUM** | `true_legacy_step_forward` | `src/CRM_TrueLegacyDynamics.cpp:108` | **Discontinuity on Failure:** If `DynamicsBVP` fails (`localmin != 0`), the function returns the *previous* state unchanged. This creates a zero-gradient step in the optimization landscape which is not handled in the backward pass (which assumes convergence). |

## 4. Derivative Correctness Audit

*   **FD in Jacobian Paths:** No explicit FD detected in `src/CRM_BVPJacobian.cpp` (uses Dual numbers), but the implementation is broken (Bug #3).
*   **Derivative Drops:** Extensive derivative dropping confirmed in templated headers (`src/CoilDynamics_Defs_Templates*.hpp`) via `.val` usage.
*   **Derivative Clipping:** Code contains explicit clipping of derivatives to `±1e3` to prevent overflow from unstable explicit integration.
*   **Analytic Mismatch:** `CRMSolverIVPJacobian` does not calculate the derivatives required for the FULLSTATE adjoint method (missing derivatives w.r.t `nL`, `pL`, `RL`).

## 5. Parameter Consistency Audit

*   **Inertia/Damping:** Reconstructed in `true_legacy_step_forward` and `true_legacy_step_backward` using identical code.
*   **Risk:** Code duplication between forward and backward passes invites divergence if one is updated without the other. Recommendation: Centralize parameter construction.

## 6. Fix Ordering (Minimal)

To restore functionality for VJP gradcheck:

1.  **Implement `CRMSolverIVPJacobian_Full`:** Create a new or updated function in `src/CRM_IVPJacobian.cpp` that correctly calculates and returns derivatives w.r.t. `nL` (interface force), `pL` (coil position), and `RL` (coil orientation).
2.  **Fix `true_legacy_step_backward` Unpacking:** Update the structured binding to correctly map the new return values to `J_n`, `J_p`, `J_R`.
3.  **Replace Heuristic `J_yx`:** Implement proper analytic `J_yx` computation in `src/CRM_BVPJacobian.cpp`, likely by extending the Dual number AD path to state variables.
4.  **Fix AD Path (`DYNNLEquation_T`):** Refactor templated headers to propagate Dual numbers correctly (remove `.val` access) and use a stable implicit integrator for sensitivity.
5.  **Enable `mL` Sensitivity:** Populate `v_mL` in `true_legacy_step_backward` using the corrected IVP Jacobians.

**Verdict:** The current codebase cannot pass `test_fullstate_vjp_gradcheck_u.py` without significant intervention on items 1-3.
