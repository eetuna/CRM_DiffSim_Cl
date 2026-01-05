# Sprint S14 - Step 5: Adjoint/VJP Finalization - INCOMPLETE

**Status:** ❌ **BLOCKED** - Critical sign error in control gradients
**Date:** 2026-01-05
**Branch:** `s14-codex-plan-impl-claude`

---

## Objective

Finalize and validate the FULLSTATE adjoint/VJP wiring under PATH A assumptions to ensure correct, stable, and documented VJP for controls (u) and supported state components.

---

## Summary

Sprint S14-Step5 aimed to validate the complete adjoint/VJP architecture. During validation, a **critical gradient sign error** was discovered that blocks completion. The architectural review was successful - PATH A assumptions are correctly implemented, no FD/autograd is used, and the VJP wiring is clean. However, control gradients fail validation with 100%+ relative error and opposite signs compared to finite differences.

---

## Work Completed

### 1. Adjoint Backward Pass Review ✅

**File:** `src/CRM_TrueLegacyDynamics.cpp:173-507`

**Findings:**
- ✅ PATH A architecture correctly implemented
- ✅ `v_y` assembly uses only `v_nL` (line 320-341), as documented
- ✅ No accidental `J_xf_x` dependencies
- ✅ Adjoint solve: `(J_yy)^T λ = v_y` using QR decomposition (line 373-374)
- ✅ Runtime warning for PATH A assumptions (line 196-212)
- ✅ Clear documentation of unpopulated columns (line 442-470)

**Control Gradient Formula (line 490):**
```cpp
Eigen::VectorXd grad_u_vec = -J_yu.transpose() * lambda;
```

**Documentation:**
- Lines 472-488: Clear explanation that control gradients flow **only** through BVP implicit pathway
- No direct pathway through `w_next` (as `w` doesn't appear in IVP Jacobians)

### 2. Batched Backward Pass Bug Fix ✅

**File:** `src/CRM_TrueLegacyDynamics.cpp:702-720`

**Issue Found:**
The batched backward pass (lines 702-759) incorrectly included a "direct" gradient term:
```cpp
// INCORRECT (removed):
grad_u_direct += (∂w_next/∂u)^T * v_w
grad_u_vec = grad_u_direct + grad_u_implicit
```

**Fix Applied:**
```cpp
// CORRECT (line 720):
Eigen::VectorXd grad_u_vec = -J_yu.transpose() * lambda;
```

This matches the single backward pass and the documented architecture.

### 3. Python Bindings Verification ✅

**File:** `python/crm_bindings.cpp:632-746`

**Findings:**
- ✅ `py_true_legacy_step_vjp()` correctly calls `true_legacy_step_backward_batched()` (line 703)
- ✅ Proper forward pass caching (line 684-691)
- ✅ Correct dimension handling for FULLSTATE (18·N + 15)

### 4. FD/Autograd Verification ✅

**Grep Results:**
```bash
grep -r "torch\|autograd\|finite.*diff" src/*.cpp src/*.hpp
```

**Findings:**
- ✅ NO torch.autograd usage
- ✅ NO finite differences in backward pass
- ✅ All Jacobians computed analytically via forward-mode AD with Dual numbers

**Confirmed:**
- `J_yy`: Analytic via `DYNNLEquation_YY_T<Dual>` (CRM_BVPJacobian.cpp:268-291)
- `J_yu`: Analytic via `DYNNLEquation_T<Dual>` (CRM_BVPJacobian.cpp:325-404)
- `J_yx`: Analytic via `DYNNLEquation_X_T<Dual>` (CRM_BVPJacobian.cpp:442-525)
- IVP Jacobians: Analytic via `CRMSolverIVPJacobian()` (CRM_IVPJacobian.cpp)

---

## BLOCKER: Control Gradient Sign Error ❌

### Test Results

**Test:** `tests/test_fullstate_vjp_gradcheck_u.py`
**Loss:** `L = ||tip_p||^2`
**Upstream gradient:** `grad_tip_p = 2 * tip_p`

**Output:**
```
VJP gradient:   [ 16.09109184   2.07034988 288.78084584]
FD gradient:    [ -0.73885312  -2.80071254 132.31784433]
Relative error: 1.189584e+00 (118.96%)

FAILED: Gradient mismatch
```

**Analysis:**
- VJP grad_u[0] = **+16.09** (positive)
- FD grad_u[0] = **-0.74** (negative)
- **Opposite signs** and **wrong magnitudes**
- Ratio: VJP / FD = **-21.78**

### Debug Output

From test run:
```
DEBUG backward: J_yu norm = 57.2361
DEBUG backward: J_yu max = 51.985
DEBUG backward: J_yu sample (0,0) = -16.9589

[Lambda Diagnostics - Sprint S12 - Batched]
  lambda norm: 4624.64
  lambda[mL[0]] norm: 94.9403
  lambda[nL[0]] norm: 4623.67
  mL/nL coupling present: YES
```

**Observations:**
- `J_yu` is non-zero and has reasonable values
- `lambda` is computed successfully (no singularity)
- mL/nL coupling is present (as expected)
- **The sign error is systematic, not numerical noise**

### Possible Root Causes

1. **Residual Sign Convention:**
   - BVP residual: `r = [m_computed - mL; n_computed - nL]`
   - Should be: `r = [mL - m_computed; nL - n_computed]` ?
   - Sign flip would propagate through `J_yu` and `grad_u`

2. **Implicit Differentiation Formula:**
   - Current: `grad_u = -J_yu^T λ`
   - Should be: `grad_u = +J_yu^T λ` ?
   - Need to re-derive from first principles

3. **IVP Jacobian Sign:**
   - If `J_n = ∂xf_next/∂nL` has wrong sign, then `v_y` and `lambda` would be incorrect
   - This would cascade to `grad_u`

4. **Adjoint Equation Setup:**
   - Current: `(J_yy)^T λ = v_y`
   - If residual minimization vs root-finding has different conventions, sign could flip

---

## Architectural Validation ✅

Despite the gradient sign error, the PATH A architecture is correctly implemented:

### No J_xf_x Dependencies ✅

**Grep:**
```bash
grep "J_xf_x" src/CRM_TrueLegacyDynamics.cpp
```

**Results:**
- Line 197: Comment - "adjoint assumes J_xf_x unpopulated columns are zero"
- Line 206: Runtime warning message
- Line 443-470: Documentation of PATH A assumption

**Confirmed:** `J_xf_x` is **never used** in backward pass. Only `J_yx` (transpose perspective) is used.

### PATH A Assumption Documentation ✅

**src/CRM_TrueLegacyDynamics.cpp:442-470:**
```cpp
// SPRINT S14 PATH A: J_xf_x ASSUMPTION
// The implicit gradient uses J_yx, which is COMPLETE.
// J_xf_x is PARTIAL (see Step 4C), with unpopulated columns for R0 and x_coil[v,w].
// ASSUMPTION: Unpopulated J_xf_x columns are STRUCTURALLY ZERO.
// Justification: x→xf_next dependence is captured via x→y→xf_next (J_yx pathway).
// The direct path ∂xf_next/∂R0 and ∂xf_next/∂(v_L_pre,w_L_pre) is negligible.
//
// CONSEQUENCE: VJPs wrt x_coil[0:6] (v,w components) will be INCOMPLETE.
// Only x_coil[6:18] (p,R components) receive correct gradients from IVP Jacobians.
```

### Runtime Warning ✅

**src/CRM_TrueLegacyDynamics.cpp:196-212:**
```cpp
if (grad_tip_p_norm > 1e-6) {
    static bool warning_shown = false;
    if (!warning_shown) {
        std::cerr << "\n[Sprint S14 PATH A - Adjoint Zero-Assumption Active]" << std::endl;
        std::cerr << "  Adjoint computation assumes J_xf_x unpopulated columns = 0" << std::endl;
        std::cerr << "  grad_x_coil[v,w] will be INCOMPLETE (missing direct IVP path)" << std::endl;
        std::cerr << "  grad_x_coil[p,R] and grad_xf will be CORRECT (from IVP Jacobians + J_yx)" << std::endl;
        std::cerr << "  This warning shown once per process." << std::endl;
        warning_shown = true;
    }
}
```

**Status:** Warning fires correctly during test execution.

---

## Supported vs Unsupported Gradient Components

### Supported (Correct under PATH A) ✅

1. **grad_xf** (15 components):
   - Flexible catheter tip state
   - Direct pathway: `v_xf_next` from loss
   - Implicit pathway: `-(J_yx)^T λ` from BVP coupling
   - ✅ Both pathways active

2. **grad_x_coil[p]** (3 components per coil):
   - Coil position
   - Direct pathway: `J_p^T grad_tip_p` from IVP Jacobian
   - Implicit pathway: `-(J_yx)^T λ`
   - ✅ Both pathways active

3. **grad_x_coil[R]** (9 components per coil):
   - Coil orientation (rotation matrix)
   - Direct pathway: `J_R^T grad_tip_p` from IVP Jacobian
   - Implicit pathway: `-(J_yx)^T λ`
   - ✅ Both pathways active

4. **grad_u** (3 components per coil):
   - Control currents
   - **ONLY implicit pathway:** `-(J_yu)^T λ`
   - ⚠️ **BLOCKED** by sign error (architecture correct, values wrong)

### Unsupported (Incomplete under PATH A) ⚠️

1. **grad_x_coil[v]** (3 components per coil):
   - Linear velocity at coil interface
   - Missing: Direct IVP pathway `∂xf_next/∂v_L_pre` (unpopulated in J_xf_x)
   - Available: Implicit pathway via J_yx (minimal contribution)
   - **Status:** INCOMPLETE by design

2. **grad_x_coil[w]** (3 components per coil):
   - Angular velocity at coil interface
   - Missing: Direct IVP pathway `∂xf_next/∂w_L_pre` (unpopulated in J_xf_x)
   - Available: Implicit pathway via J_yx (minimal contribution)
   - **Status:** INCOMPLETE by design

**PATH A Trade-off:**
- Control gradients (`grad_u`) are the primary product
- Velocity gradients (`grad_v`, `grad_w`) are acceptable casualties
- This is documented and intentional for S14

---

## Code Changes

### File: `src/CRM_TrueLegacyDynamics.cpp`

**Lines 702-720:** Removed incorrect direct gradient term from batched backward
```cpp
// BEFORE (incorrect):
grad_u_direct += dt * dTau_du[j](i) / I_xx * v_w[i];
grad_u_vec = grad_u_direct + grad_u_implicit;

// AFTER (correct):
Eigen::VectorXd grad_u_vec = -J_yu.transpose() * lambda;
```

**Justification:**
- Matches documented architecture (lines 484-486)
- Consistent with single backward pass (line 490)
- Control gradients flow **only** through BVP implicit pathway

---

## Next Steps (BLOCKED)

### CRITICAL: Fix Control Gradient Sign Error

**Priority:** P0 (BLOCKER)

**Investigation Required:**
1. Verify residual definition in `DYNNLEquation`
   - Check if `r = m_computed - mL` or `r = mL - m_computed`
   - Review shooting method conventions in `CRMDYN.hpp`

2. Re-derive implicit differentiation formula
   - Start from `r(y, u, x) = 0`
   - Verify adjoint equation sign: `(∂r/∂y)^T λ = ∂L/∂y`
   - Verify gradient formula: `∂L/∂u = -(∂r/∂u)^T λ` or `+(∂r/∂u)^T λ`

3. Check IVP Jacobian sign
   - Verify `J_n = ∂xf_next/∂nL` is computed with correct sign
   - Test with simple analytical case

4. Cross-reference with reduced6d implementation
   - Check if reduced6d VJP has same sign error
   - If reduced6d is correct, compare formulas

### Once Sign Error Fixed:

1. Re-run `test_fullstate_vjp_gradcheck_u.py` and verify pass
2. Add control gradient tests to sanity gate
3. Document final gradient accuracy (expected tolerance: < 1e-3 relative error)
4. Proceed to Step 6: Linearization validation

---

## Files Modified

1. `src/CRM_TrueLegacyDynamics.cpp` - Fixed batched backward grad_u computation
2. `docs/audits/SPRINT_S14_STEP5_ADJOINT_VJP_FINALIZATION.md` - This report

---

## Dependencies Verified

- ✅ Sprint S14-Step3 (J_yx complete)
- ✅ Sprint S14-Step4A (Flex forward template)
- ✅ Sprint S14-Step4B (IVP wrapper Jacobians)
- ✅ Sprint S14-Step4C (Complete IVP Jacobians)
- ✅ Sprint S5 (J_yu derivation and implementation)

---

## Test Coverage

### Passing ✅

- ❌ None (blocked by sign error)

### Failing ❌

- **test_fullstate_vjp_gradcheck_u.py:** Control gradient validation (118% relative error, opposite signs)

### Not Run

- Batched VJP tests (depend on single VJP correctness)
- Multi-RHS adjoint tests
- Controller integration tests

---

## Technical Debt

1. **P0:** Fix control gradient sign error (BLOCKER)
2. **P1:** Remove debug print statements after validation passes
3. **P1:** Add inline comments explaining residual sign convention
4. **P2:** Create unit test for simple analytical case (verifiable by hand)
5. **P2:** Add diagram showing adjoint pathways for documentation

---

## Conclusion

**Sprint S14-Step5 status: INCOMPLETE (BLOCKED)**

The adjoint/VJP architecture is correctly implemented under PATH A assumptions:
- ✅ No `J_xf_x` dependencies
- ✅ Clean separation of supported vs unsupported gradients
- ✅ Analytic Jacobians (no FD/autograd)
- ✅ Proper Python bindings

However, a **critical sign error** in control gradients blocks completion. VJP gradients have opposite signs and wrong magnitudes compared to finite differences (118% relative error). This must be resolved before proceeding to linearization or controller validation.

**Recommendation:** Investigate residual sign convention and implicit differentiation formula. Cross-reference with reduced6d implementation if available. Consider deriving a simple 1-DOF analytical test case to isolate the sign error.

**DO NOT PROCEED** to Step 6 (Linearization) until control gradients pass validation.
