# CODEX Execution-Ready Remediation Plan (FULLSTATE VJP + Linearization)

**Scope:** FULLSTATE only (18·N + 15). Forward chain fixed: `DynamicsBVP → DYNSolverIVP`.
**Methods allowed:** Analytic / IFT / forward-mode AD at converged solution only.
**Prohibited:** FD anywhere (incl. tests), torch.autograd, heuristics, placeholders.

## Mandatory Rotation-Sensitivity Decision
**Chosen approach:** Forward-mode AD with `Dual` at the converged solution for **all rotation sensitivities**, including rotation update Jacobians and rotation residuals. No analytic SO(3) Jacobians will be used for gradients. Rotation derivatives propagate only through templated Dual arithmetic in the integrators and residuals.

---

## Dependency-Ordered Steps

### Step 1 — Remove Rotation Residual Singularity (Foundation)
**Issue:** `sqrt(||ΔR||²)` derivative is singular at convergence.

- **Files / functions:**
  - `src/CoilDynamics_Defs_Templates2.hpp`: `DYNNLEquation_T`, `DYNNLEquation_YY_T`
- **Exact algorithm:**
  - Replace rotation residual component `sqrt(v_norm_sq)` with squared norm `0.5 * v_norm_sq` for both double and `Dual` paths.
  - For `Dual`, residual derivative is `0.5 * v_norm_sq.deriv` (no division).
- **Invariants restored:**
  - Residual remains zero at convergence.
  - No divide-by-zero or NaNs in J_yy/J_yu/J_yx at the converged solution.
- **Tests:**
  - `tests/test_fullstate_step_smoke.py` (forward stability).
  - `tests/test_fullstate_vjp_parity.py` (added in Step 8; no NaNs in backward path).

---

### Step 2 — Restore Rotation Derivatives via Dual Propagation (Core Rotation Path)
**Issue:** Rotation sensitivities are dropped by `.val` extraction and double-only SE(3) update.

- **Files / functions:**
  - `src/CoilDynamics_Defs_Templates.hpp`: `DYNSE3_TimeSpace_T`, `CoilDynamics_T` call sites.
  - `src/CRMDYN_Numerical_Integration.hpp`: templated integrator paths that call `SE3_Analytical_Step`.
  - `src/CoilDynamics_Defs_Templates2.hpp`: `DYNNLEquation_T`, `DYNNLEquation_YY_T` (R_L extraction).
- **Exact algorithm:**
  1. **Template rotation state:**
     - Change `DYNSE3_TimeSpace_T` to accept `const T in_R_n[9]` and output `T out_R_np1[9]`.
     - Remove all `.val` extraction for `R_n`, `R_L`, and `twist` in templated paths.
  2. **Dual-safe Rodrigues update:**
     - Implement `Rodrigues_T<T>` in `src/CoilDynamics_Defs_Templates.hpp` using `T` sin/cos operations.
     - Use exact special-case at `theta == 0` (not a threshold) to avoid division by zero: `sinc(0)=1`, `(1-cos)/theta^2 = 0.5`.
     - Multiply `R_n * R_delta` using templated matrix multiply to keep Dual derivatives.
  3. **Integrators:**
     - Update all calls to `DYNSE3_TimeSpace_T` and `SE3_Analytical_Step` in `src/CRMDYN_Numerical_Integration.hpp` and `src/CoilDynamics_Defs_Templates.hpp` to use the templated rotation path.
  4. **Remove derivative drops:**
     - Replace `.val` usage when extracting `R_L` in `src/CoilDynamics_Defs_Templates2.hpp` with direct `T` assignments.
- **Invariants restored:**
  - Rotation sensitivities propagate through coil dynamics and BVP residuals.
  - J_yy/J_yu/J_yx include u→R→residual pathways.
- **Tests:**
  - `tests/test_fullstate_vjp_parity.py` (added in Step 8).
  - `tests/test_fullstate_linearize_consistency.py` (added in Step 8).

---

### Step 3 — Replace Heuristic J_yx with Exact Forward-Mode AD
**Issue:** `J_yx` currently uses an empirical coupling heuristic.

- **Files / functions:**
  - `src/CRM_BVPJacobian.cpp`: `compute_bvp_jacobians_full_analytic`
  - `src/CoilDynamics_Defs_Templates2.hpp`: new templated params support
- **Exact algorithm:**
  1. **Template the BVP params for state inputs:**
     - Introduce `DYNNLEqnParams_T<T>` mirroring `DYNNLEqnParams` but with `v_L_pre`, `w_L_pre`, `p_pre`, `R_pre`, `xf` stored as `T`.
     - Add a conversion helper from double params to `DYNNLEqnParams_T<T>` (value-only copy).
  2. **Dual-seeded J_yx:**
     - In `compute_bvp_jacobians_full_analytic`, for each state component in `x = [x_coil; xf]`, seed that component with `Dual(value, 1)` in `DYNNLEqnParams_T<Dual>`.
     - Call `DYNNLEquation_T<Dual>` with seeded params and extract `out_y_dual[row].deriv` into `J_yx(row, col)`.
  3. **No heuristics:** remove `coupling_scale` and all sparse-identity approximations.
- **Invariants restored:**
  - `J_yx` is the true derivative of the BVP residual with respect to FULLSTATE.
- **Tests:**
  - `tests/test_bvp_jacobian_sanity.py` (add): assert `J_yx.norm() > 0` and `J_yy` solve residual finite (no FD).

---

### Step 4 — Replace Misused CRM IVP Jacobian with Exact DYNSolverIVP Jacobians
**Issue:** `CRMSolverIVPJacobian` returns derivatives w.r.t. actuation/tip inputs but is unpacked as derivatives w.r.t. `nL`, `pL`, `RL`, producing invalid VJP.

- **Files / functions:**
  - `src/CoilDynamics_Defs_Templates2.hpp`: add `DYNSolverIVP_T<T>` (templated IVP for Dual).
  - `src/CoilDynamics_Defs.cpp`: add `DYNSolverIVP_JacobiansFullstate` wrapper.
  - `src/CRM_TrueLegacyDynamics.cpp`: `true_legacy_step_backward`, `true_legacy_step_backward_batched` (use new Jacobians; remove CRMSolverIVPJacobian usage).
- **Exact algorithm:**
  1. **Templated IVP:**
     - Implement `DYNSolverIVP_T<T>` mirroring `DYNSolverIVP` but templated for `T` inputs: `x_coil`, `u0`, `mL`, `nL`, `tau`, `ftip`, and `dt`.
     - Return `x_coil_next` and `xf_next` as `T` so derivatives flow.
  2. **Forward-mode Jacobian extraction (converged solution only):**
     - For each input component in `y = [mL; nL]` and `x = [x_coil; xf]`, seed with `Dual(value, 1)` and run `DYNSolverIVP_T<Dual>`.
     - Extract:
       - `J_xf_y = ∂xf_next/∂y`
       - `J_xf_x = ∂xf_next/∂x`
       - `J_xcoil_y = ∂x_coil_next/∂y` (needed for linearization A block)
  3. **Backward pass wiring:**
     - Set `v_y = J_xf_y^T * v_xf_next` (no zeroing of mL blocks).
     - Use `J_xf_x` to populate direct `grad_x_coil` / `grad_xf` terms.
     - Remove `CRMSolverIVPJacobian` structured binding entirely.
- **Invariants restored:**
  - VJP uses correct ∂xf/∂y and ∂xf/∂x, including mL pathways.
  - No Jacobian interpretation mismatch.
- **Tests:**
  - `tests/test_fullstate_vjp_parity.py`
  - `tests/test_fullstate_linearize_consistency.py`

---

### Step 5 — Fix Adjoint Assembly and Single/Batched Consistency
**Issue:** mL cotangents are zeroed; single-sample VJP misses direct u→w pathway present in batched path.

- **Files / functions:**
  - `src/CRM_TrueLegacyDynamics.cpp`: `true_legacy_step_backward`, `true_legacy_step_backward_batched`
- **Exact algorithm:**
  1. **mL cotangents:**
     - Populate `v_y` using `J_xf_y` from Step 4 (no manual zeroing of mL).
  2. **Direct control pathway parity:**
     - Compute `dTau/du` using the full turn-area matrix (Step 6) and use the same direct gradient term in **both** single and batched paths.
     - In single path: `grad_u = grad_u_direct + grad_u_implicit` where `grad_u_implicit = -(J_yu)^T * lambda`.
- **Invariants restored:**
  - mL block participates in adjoint solve; single and batched VJP are algebraically identical.
- **Tests:**
  - `tests/test_fullstate_vjp_parity.py` (single vs batched equality).

---

### Step 6 — Fix Control Jacobian dTau/du (Full Turn-Area Matrix)
**Issue:** Diagonal assumption for `dTau/du` conflicts with BVP seeding and physical model.

- **Files / functions:**
  - `src/CRM_TrueLegacyDynamics.cpp`: `true_legacy_step_backward_batched`, `true_legacy_step_backward`, `true_legacy_linearize`
- **Exact algorithm:**
  - For each actuator `j`, set `∂MagMoment/∂u[:,i] = CoilAlignmentTurnAreaMatrix[j][:,i]` and compute:
    - `∂tau/∂u[i] = (∂MagMoment/∂u[i]) × B0`.
  - Use this `dTau_du` in both VJP direct term and linearization `G_u`.
- **Invariants restored:**
  - Control Jacobians are consistent across BVP, VJP, and linearization.
- **Tests:**
  - `tests/test_fullstate_vjp_parity.py`
  - `tests/test_fullstate_linearize_consistency.py`

---

### Step 7 — Replace Simplified G_x with Exact IFT Linearization
**Issue:** Linearization uses a simplified, non-physical `G_x`.

- **Files / functions:**
  - `src/CRM_TrueLegacyDynamics.cpp`: `true_legacy_linearize`
- **Exact algorithm:**
  1. **Compute implicit sensitivity:**
     - Use `J_yy`, `J_yx` from Step 3 to compute `dydx = -(J_yy)^{-1} * J_yx`.
  2. **Compute exact `G_x` using IVP Jacobians (Step 4):**
     - `G_x = ∂x_next/∂x + (∂x_next/∂y) * dydx`.
     - Use `J_xcoil_x`, `J_xcoil_y`, `J_xf_x`, `J_xf_y` from `DYNSolverIVP_JacobiansFullstate` for the direct terms.
  3. **Compute `G_u`:**
     - `dydU = -(J_yy)^{-1} * J_yu`.
     - `G_u = (∂x_next/∂y) * dydU + G_u_direct` (direct term from Step 6).
- **Invariants restored:**
  - Linearization matches actual forward dynamics at the converged solution.
- **Tests:**
  - `tests/test_fullstate_linearize_consistency.py`
  - `tests/test_fullstate_linearize_shapes.py` (shape + non-triviality).

---

### Step 8 — Update Tests to be Protocol-Compliant (No FD / No Autograd)
**Issue:** Existing tests use FD and torch.autograd.

- **Files / tests to modify/add:**
  - `tests/test_fullstate_vjp_gradcheck_u.py`: remove FD reference and torch usage.
  - `tests/test_fullstate_linearize_shapes.py`: remove torch.autograd block.
  - Add `tests/test_fullstate_vjp_parity.py`.
  - Add `tests/test_fullstate_linearize_consistency.py`.
  - Add `tests/test_bvp_jacobian_sanity.py`.
- **Exact algorithm for new tests:**
  1. **VJP parity:**
     - Run `true_legacy_step_backward` and `true_legacy_step_backward_batched` with `batch=1`; assert equality within `rtol=1e-12, atol=1e-12` for `grad_x_coil`, `grad_xf`, `grad_u`.
  2. **Linearize vs VJP:**
     - Use `true_legacy_linearize` to get `A` and `B`.
     - Build `v_xf_next` with nonzero tip position cotangent only (`v_tip_p`), zeros elsewhere.
     - Assert `A^T * v_xf_next` matches `grad_x` from `true_legacy_step_backward`, and `B^T * v_xf_next` matches `grad_u`.
  3. **BVP Jacobian sanity:**
     - Compute `J_yy`, `J_yu`, `J_yx` at a converged solution and assert `J_yx.norm() > 0` and `J_yy` solve residual is finite (no NaNs/inf).
- **Invariants restored:**
  - Test suite enforces correctness without FD or autograd.

---

### Step 9 — Python Binding Guard (Forward Correctness)
**Issue:** Python binding always advances IVP even when BVP fails.

- **Files / functions:**
  - `python/crm_bindings.cpp`: `py_true_legacy_step_forward`
- **Exact algorithm:**
  - Mirror the C++ guard: if `out_localmin != 0`, return previous state and `converged=false` without calling `DYNSolverIVP`.
- **Invariants restored:**
  - Python forward matches authoritative C++ behavior and does not propagate unconverged states.
- **Tests:**
  - Add a small pytest that forces BVP non-convergence and checks `converged=false` and unchanged state.

---

## Enforcement Checks (No FD / No Autograd / No Heuristics)
```bash
rg "finite.*diff|FD|fd_eps|fdiff" tests src
rg "torch\.autograd|\.backward\(|grad_fn|requires_grad" tests python
rg "coupling_scale|0\.1\s*\*\s*dt|dt\s*\*\s*0\.1" src/CRM_BVPJacobian.cpp
```

## Execution Order Summary
1. Step 1 (rotation residual singularity)
2. Step 2 (Dual rotation propagation)
3. Step 3 (exact J_yx via forward-mode AD)
4. Step 4 (DYNSolverIVP Jacobians; remove CRMSolverIVPJacobian usage)
5. Step 5 (adjoint assembly + single/batched parity)
6. Step 6 (full `dTau/du`)
7. Step 7 (exact `G_x`, `G_u` linearization)
8. Step 8 (tests: remove FD/autograd, add parity/consistency)
9. Step 9 (Python guard)

**STOP:** This plan is execution-ready. No code changes performed.
