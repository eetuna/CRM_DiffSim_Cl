# Gemini Audit: Sprint S10 Dual Instability

**Date:** 2026-01-05
**Auditor:** Gemini
**Status:** Diagnosis Complete

## Executive Summary

The numerical instability (explosion of derivatives to $10^{168}$) observed in the Dual-number backward pass is caused by a **critical implementation bug** in `src/CRM_BVPJacobian.cpp`, not by intrinsic physics stiffness or the Dual number implementation itself.

The function `compute_bvp_jacobians_fmad` constructs `ActInertia` using an incorrect heuristic (`Mass * 1e-6`) instead of the correct geometric formula ($I o m r^2$). This results in an inertia value $\sim 10^{-12}$ kg·m², which is **$10^7$ times smaller** than the correct inertia ($\sim 2 \times 10^{-5}$ kg·m²) used in the forward path.

This artificially increases the rotational stiffness ($\lambda \approx -b/I$) by a factor of $10^7$, pushing the time constant into the nanosecond range ($10^{-9}$ s). The explicit integrator ($h \approx 10^{-4}$ s) becomes unconditionally unstable, causing immediate explosion.

The Forward path remains stable because `true_legacy_step_forward` (in `src/CRM_TrueLegacyDynamics.cpp`) calculates `ActInertia` correctly using the geometric formula.

## 1. Divergence Point

*   **File:** `src/CRM_BVPJacobian.cpp`
*   **Line:** ~115 (inside `compute_bvp_jacobians_fmad`)
*   **Code:**
    ```cpp
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 9; ++i) {
            ActInertia[j][i] = (i % 4 == 0) ? params.CathParams->ActMass[j] * 1e-6 : 0.0;
        }
    }
    ```
    *   **Issue:** `ActMass` is $\approx 8 \times 10^{-6}$ kg. The calculation yields $I \approx 8 \times 10^{-12}$ kg·m².
    *   **Contrast:** `true_legacy_step_forward` computes $I \approx 2 \times 10^{-5}$ kg·m² (cylinder formula).

## 2. Call Chain & Failure Propagation

1.  **Entry:** `py_true_legacy_step_vjp` calls `true_legacy_step_backward_batched`.
2.  **Jacobian Prep:** Calls `compute_bvp_jacobians_full_analytic` -> `compute_bvp_jacobians_fmad`.
3.  **Inertia Setup (BUG):** `ActInertia` is initialized to $\sim 10^{-12}$ (Factor $10^7$ error).
4.  **Dual Execution:** `DYNNLEquation_T<Dual>` -> `CoilDynamics_T<Dual>` -> `RK2_coildyn_T<Dual>`.
5.  **Dynamics:**
    *   Stiffness $\lambda_{rot} = -b_{ang} / I \approx -0.03 / 10^{-11} \approx -3 \times 10^9$.
    *   Step size $h_{dual} = 0.0002$.
    *   Stability factor $|1 + h\lambda| \approx 6 \times 10^5$.
6.  **Result:** Derivatives amplify by $\sim 10^5$ *per substep*. Over 10ms (50 substeps), values exceed `double` range ($10^{250}$), resulting in `NaN`/`Inf` or chaos ($10^{168}$). 

## 3. Classification

**Type:** **Implementation Artifact** (Logic Bug / Parameter Mismatch).

The instability is **not** due to:
*   Intrinsic stiffness of the *physical* system (Real $\lambda \approx -1500$, which is stable for $h=0.001$).
*   Dual number implementation overhead.
*   Explicit integration method (RK2 is stable for the correct physics).

## 4. Sensitivity Analysis (SE3)

*   **Current State:** Rotation matrices `R` are treated as `double` (constant) in `CRMIntegrand_dyn_T` and `DYNSE3_TimeSpace_T`.
*   **Impact:** This drops the term $\frac{\partial \mathbf{\tau}_{mag}}{\partial R}$ in the sensitivity analysis.
*   **Stability:** This simplification *improves* stability by removing a feedback loop ($u \to R \to \tau \to \dot{w} \to \dot{R}$). It is **not** the cause of the explosion.
*   **Recommendation:** Once the inertia bug is fixed, the system will be stable. The dropped rotation derivatives affect gradient *accuracy* but fixing them is a secondary task.

## 5. Ranked Fix Directions

### Direction 1: Harmonize Inertia Calculation (Recommended)
Copy the correct geometric inertia calculation logic from `src/CRM_TrueLegacyDynamics.cpp` (lines ~58-75) into `src/CRM_BVPJacobian.cpp` inside `compute_bvp_jacobians_fmad`.

**Pros:**
*   Fixes the root cause (unphysical parameter).
*   Makes Forward and Backward paths physically consistent.
*   Guarantees stability for the given parameters.
*   Low risk (copy-paste existing logic).

**Cons:** None.

### Direction 2: Pass Calculated Inertia from Forward Pass
Modify `compute_bvp_jacobians_fmad` to accept `ActInertia` as an argument, and pass the correct `ActInertia` cached from `fwd_result`.

**Pros:**
*   Ensures exact consistency by design.

**Cons:**
*   Requires signature changes in internal API (`CRM_BVPJacobian.hpp`, `CRM_TrueLegacyDynamics.cpp`).
*   Requires `TrueLegacyStepResult` to cache `ActInertia`.

### Direction 3: Implicit Sensitivity Integration (Not Needed)
Switching to implicit integration for derivatives is **unnecessary** because the system is actually stable with the correct parameters. The stiffness was artificial.

## Conclusion
The Dual instability is an artifact of using a fallback/debug inertia calculation in the BVP Jacobian module that is $10^7$ times too small. Fixing this parameter mismatch will resolve the explosion and likely align the VJP results with Finite Differences.
