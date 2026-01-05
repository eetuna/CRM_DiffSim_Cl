# GEMINI EXECUTION-READY PLAN: FULLSTATE VJP & LINEARIZATION REMEDIATION

**Date:** 2026-01-05
**Status:** EXECUTION-READY
**Scope:** FULLSTATE (18·N + 15)
**Constraint Enforcement:**
- ✅ NO Finite Differences
- ✅ NO Heuristics / Placeholders
- ✅ NO torch.autograd
- ✅ Fixed Forward Chain: `DynamicsBVP → DYNSolverIVP`

## STRATEGY DECISION: HYBRID ANALYTIC/DUAL
To satisfy the "One Strategy" mandate while respecting codebase realities:
- **Analytic SO(3) & STM:** Used for all **cross-coupling** terms (`J_yx`, `J_yR`, `v_mL`, `G_x`). We will extend the IVP integrator to explicitly return the State Transition Matrix (STM) blocks `∂xf/∂p0` and `∂xf/∂R0`. This bypasses the unstable AD-through-integrator path for long-range sensitivities.
- **Dual Numbers (Forward-Mode AD):** Used **strictly for the BVP internal Jacobian** (`J_yy`) and **residual evaluation**. The existing templated `DYNNLEquation` will be patched to remove derivative drops, ensuring `J_yy` correctness without a full analytic rewrite.

---

## EXECUTION PLAN (Dependency-Ordered)

### 1. Fix Rotation Residual Singularity (Dual Path)
**Goal:** Prevent NaNs in BVP Jacobian when rotation error is zero.
**Files:** `src/CoilDynamics_Defs_Templates2.hpp`
**Function:** `DYNNLEquation_T`
**Algorithm:**
- Replace `sqrt(||ΔR||²)` residual with `||ΔR||²/2` (squared-norm).
- **Invariant:** `∂(||ΔR||²/2)/∂R = ΔR`. Non-singular at `ΔR=0`.
- **Test:** `test_fullstate_step_smoke.py` (Verify convergence maintained).

### 2. Extend IVP Jacobian to Return Full STM (Analytic Path)
**Goal:** Provide `∂xf/∂p_coil` and `∂xf/∂R_coil` for `J_yx` and `G_x`.
**Files:** `src/CRM_IVPJacobian.cpp`, `src/CRM_IVPJacobian.hpp`, `src/CRM_IVP_NumericalIntegrationTemplates.hpp`
**Function:** `CRMSolverIVPJacobian`, `CRMIntegrand`
**Algorithm:**
1.  **Enable Sensitivities:** Uncomment/Enable `_p_p0` and `_ws_w0` computation in `CRMIntegrand` (lines 352-365, 381-384).
2.  **Initialize:** In `CRMSolverIVP_CoreWithJacobian`, initialize `_p_p0` to Identity and `_ws_w0` to `R0`.
3.  **Return:** Update `CRMSolverIVPJacobian` signature to return `MatrixXd J_p0` (3x3) and `MatrixXd J_w0` (3x3) blocks (accumulated from `x_N`).
**Invariant:** `J_p0 = ∂xf_pos/∂p_coil`, `J_w0 = ∂xf_pos/∂θ_coil`.

### 3. Fix Torque Jacobian Consistency (Analytic Path)
**Goal:** Ensure `dTau/du` matches BVP physics (Full Turn-Area Matrix).
**Files:** `src/CRM_TrueLegacyDynamics.cpp`
**Function:** `true_legacy_step_backward_batched`, `true_legacy_linearize`
**Algorithm:**
- Replace diagonal heuristic with loop over `CoilAlignmentTurnAreaMatrix[j]`.
- Compute `dtau = (dM/du) × B` for each control channel.
**Invariant:** `dTau/du` exactly matches `CRM_BVPJacobian.cpp` seeding.

### 4. Implement Analytic `J_yx` and `J_yR`
**Goal:** Compute sensitivity of BVP Residual to Coil State without heuristics.
**Files:** `src/CRM_BVPJacobian.cpp`
**Function:** `compute_bvp_jacobians_full_analytic`
**Algorithm:**
- **J_yx (State):**
  - Use `J_p0` and `J_w0` from **Step 2**.
  - `∂Residual/∂p_coil` = `(∂Residual/∂xf) * J_p0`.
  - `∂Residual/∂R_coil` = `(∂Residual/∂xf) * J_w0` (plus direct rotation constraint terms).
- **J_yR (Rotation):**
  - Explicitly populate `J_yR` block using `J_w0` and analytic rotation residual derivative `(R_final - R_desired)^T`.
**Invariant:** `J_yx` is derived from exact IVP STM, not `dt*0.1`.

### 5. Enable `mL` Sensitivity via Analytic Chain Rule
**Goal:** Fix `v_mL = 0` bug in VJP.
**Files:** `src/CRM_TrueLegacyDynamics.cpp`
**Function:** `true_legacy_step_backward`
**Algorithm:**
- Use **Analytic SO(3) Left Jacobian** `J_l(w*dt)`.
- Chain: `v_mL = (∂τ/∂mL)^T * (∂w/∂τ)^T * (∂R/∂w)^T * (∂xf/∂R)^T * v_xf`.
- `∂R/∂w`: Use `J_l(w*dt) * dt`.
- `∂xf/∂R`: Use `J_w0` from **Step 2**.
**Invariant:** `v_mL` correctly captures moment-to-tip coupling.

### 6. Fix Linearization `G_x` using Analytic STM
**Goal:** Correct `A` matrix using exact dynamics derivatives.
**Files:** `src/CRM_TrueLegacyDynamics.cpp`
**Function:** `true_legacy_linearize`
**Algorithm:**
- **Implicit Term:** `dydx = -inv(J_yy) * J_yx` (Using fixed `J_yx` from Step 4).
- **Explicit Term (`∂f/∂x`):**
  - `∂p_next/∂p = I`, `∂p_next/∂v = I*dt`.
  - `∂R_next/∂R`: Kronecker product `(exp([w*dt]×)^T ⊗ I)`.
  - `∂R_next/∂w`: `R * J_l(w*dt) * dt`.
  - `∂xf/∂x_coil`: Directly use `J_p0`, `J_w0` from **Step 2**.
**Invariant:** `A` matrix satisfies Taylor expansion test `||error|| ~ O(dx²)`.

### 7. Add BVP Convergence Guard
**Goal:** Prevent invalid forward states from corrupting gradients.
**Files:** `python/crm_bindings.cpp`
**Function:** `py_true_legacy_step_forward`
**Algorithm:**
- Check `out_localmin`. If nonzero, return `x_prev` and `converged=False`.

---

## VERIFICATION (Non-FD)

1.  **`tests/test_fullstate_vjp_parity.py`**:
    - Assert `batched_vjp(batch=1) == single_vjp`.
    - Assert `v_mL` is non-zero.
2.  **`tests/test_fullstate_linearize_correctness.py`**:
    - Taylor Test: `||f(x+h) - (f(x) + A*h)|| < 10 * ||h||²`.
3.  **`tests/test_fullstate_vjp_gradcheck_u.py`**:
    - Modify to skip FD check, focus on internal consistency (shapes, non-NaN).
