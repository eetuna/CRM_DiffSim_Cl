# Sprint S14-Step5A: Control Gradient Sign Error Investigation - INCOMPLETE

**Status:** ❌ **BLOCKED** - Complex sign/residual convention issue
**Date:** 2026-01-05
**Branch:** `s14-codex-plan-impl-claude`

---

## Objective

Resolve the systematic control gradient sign error identified in Sprint S14-Step5 by auditing residual sign conventions and implicit differentiation formulas.

---

## Summary

Sprint S14-Step5A investigated the control gradient sign error through systematic audit of residual definitions, implicit differentiation formulas, and sign conventions. **The issue is NOT a simple global sign flip**, but rather a more complex problem involving the BVP residual definition or J_yu computation. A simple sign flip produces mixed results: component 0 improves but component 2 worsens, indicating fundamental architectural issues beyond sign convention.

---

## Investigation Completed

### 1. Residual Definition Audit ✅

**File:** `src/CoilDynamics_Defs_Templates2.hpp:559,843`

**Finding:**
```cpp
residual[actno_mn][i] = p_f[i] - p_L[i];
```

**Residual Convention:** `r = computed - state`
- `p_f` = position from flexible segment integration (computed value)
- `p_L` = position at coil interface from BVP unknowns (state value)

**Documented Convention** (`src/CRM_BVPJacobian.cpp:225-226`):
```cpp
// r_m[j] = m_computed[j] - mL[j]  (moment balance)
// r_n[j] = n_computed[j] - nL[j]  (force balance)
```

**Confirmation:** Residual convention is **consistent** across codebase: `r = computed - state`

---

### 2. Implicit Differentiation Formula Verification ✅

**Mathematical Derivation:**

Given BVP residual: `r(y, u, x) = 0` where `y = [mL; nL]`

Implicit Function Theorem:
```
dy/du = -(∂r/∂y)^{-1} * (∂r/∂u)
```

VJP for loss `L(y(u))`:
```
∇_u L = (∂L/∂y) * (dy/du)
      = (∂L/∂y) * [-(∂r/∂y)^{-1} * (∂r/∂u)]
      = -[(∂L/∂y) * (∂r/∂y)^{-1}] * (∂r/∂u)
      = -λ^T * (∂r/∂u)
```

where adjoint: `(∂r/∂y)^T * λ = (∂L/∂y)^T`

**Final Formula:** `grad_u = -(∂r/∂u)^T * λ = -J_yu^T * λ`

**Cross-Reference:** Reduced6d implementation (`src/reduced6d/CRM_DiffDynamics.cpp:327`):
```cpp
// grad_u[i] = -(dr/du_i)^T * lambda
grad_ut(i) = -dr_du_i.dot(lambda);
```

**Confirmation:** Formula `grad_u = -J_yu^T * λ` is **mathematically correct** and **consistent** across FULLSTATE and reduced6d.

---

### 3. Sign Flip Hypothesis Test ❌

**Hypothesis:** Simple global sign flip would resolve the error.

**Test:** Changed formula from `-J_yu^T * λ` to `+J_yu^T * λ`

**Results:**

| Component | Original VJP | Flipped VJP | FD (Truth) | Assessment |
|-----------|-------------|-------------|------------|------------|
| u[0]      | +16.09      | **-16.09**  | -0.74      | Sign improved, magnitude still wrong (22x) |
| u[1]      | +2.07       | **-2.07**   | -2.80      | Both improved |
| u[2]      | +288.78     | **-288.78** | +132.32    | Sign WORSENED! |

**Relative Error:**
- Before flip: 118.96%
- After flip: 318.38% **(WORSE!)**

**Conclusion:** Sign flip is **NOT** the solution. Component 0 improved, but component 2 worsened, indicating the issue is more fundamental than a global sign convention.

**File:** `src/CRM_TrueLegacyDynamics.cpp:490,724` (reverted to original `-` sign)

---

## Root Cause Analysis

### Observation

The fact that components have **inconsistent** sign behavior indicates:

1. **NOT a global sign flip** in the adjoint equation
2. **NOT a simple sign flip** in J_yu extraction
3. **POSSIBLE** issues:
   - Residual definition has component-dependent sign inconsistencies
   - J_yu computation has directional errors (different signs for different control axes)
   - Packing/indexing mismatch between residual components and J_yu
   - Physical model has sign errors in magnetic torque application

### Critical Clue

**J_yu Diagnostics:**
```
DEBUG J_yu: CoilAlignmentTurnAreaMatrix[0] = [1.82371, 0.183738, 0, ...]
DEBUG J_yu: MagMoment[0] = [0.191558, -0.161664, -0.09224]
DEBUG J_yu: Seeding MagMoment[0][0] with deriv = 1.82371
DEBUG J_yu: out_y_dual[0].deriv = -16.9589
DEBUG J_yu: J_yu(0,0) = -16.9589
```

When seeding with **positive** deriv (+1.82), the residual derivative comes out **negative** (-16.96). This means:
- Increasing `u[0]` **decreases** residual component `r[0]`
- Physical interpretation: more current → more magnetic torque → changes coil dynamics → residual changes

### Physical Chain

```
u ↑ → MagMoment ↑ → Tb (mag torque) ↑ → τ_internal changes → wdot changes → R_coil changes → p_L changes → residual changes
```

The question: **Should the residual increase or decrease when u increases?**

This depends on:
1. How the shooting method balances moments/forces
2. Which direction the magnetic torque pushes the system
3. Whether we're solving `r = 0` by minimizing or root-finding

---

## Hypothesis for Further Investigation

### H1: Residual Convention Mismatch

**Theory:** The shooting method residual might need to be `r = state - computed` instead of `r = computed - state` for the implicit differentiation formula to work correctly.

**Test:** Flip residual definition in `DYNNLEquation` templates and recompute J_yu.

**Risk:** High - affects forward physics convergence.

---

### H2: J_yu Magnetic Torque Sign Error

**Theory:** The chain `u → MagMoment → muhat → Tb → residual` has a sign flip somewhere.

**Key Locations:**
- `src/CoilDynamics_Defs.cpp:71-72`: `Tb = muhat * (R^T * B0); tau = Tb - m_L`
- `src/CoilDynamics_Defs_Templates2.hpp:759`: `net_mL = m_L - tau`

**Test:** Add debug prints to trace sign of magnetic torque through the chain.

---

### H3: Indexing/Packing Mismatch

**Theory:** The way residuals are packed into `out_y` might have inconsistent ordering with how J_yu is computed.

**Key Location:**
- `src/CoilDynamics_Defs_Templates2.hpp:626`: `out_y[j + i*6] = residual[i][j] * ...`

**Test:** Verify that J_yu row/column indices match the residual packing format.

---

## Blocked Status

**Cannot proceed** without resolving the sign/convention issue. The problem is **NOT** a simple sign flip, and requires:

1. Deeper understanding of shooting method residual semantics
2. Component-by-component trace of u → residual chain
3. Possible physics-level debugging with simplified test case

**STOP CONDITION MET:** More than one sign flip would be required, indicating fundamental issue beyond simple sign convention.

---

## Files Modified

1. `src/CRM_TrueLegacyDynamics.cpp` - Added investigation comments (NO functional change, reverted to original `-` sign)

---

## Files Audited

1. `src/CRM_BVPJacobian.cpp` - Residual definition, J_yy/J_yu computation
2. `src/CoilDynamics_Defs_Templates2.hpp` - DYNNLEquation residual packing
3. `src/CoilDynamics_Defs.cpp` - Magnetic torque application in coil dynamics
4. `src/CRM_TrueLegacyDynamics.cpp` - Adjoint backward pass
5. `src/reduced6d/CRM_DiffDynamics.cpp` - Cross-reference for sign convention

---

## Next Steps (BLOCKED)

### CRITICAL: Systematic Debug Trace Required

**Recommended Approach:**

1. **Create Minimal Test Case:**
   - Single actuator, single control input
   - Fixed magnetic field, fixed geometry
   - Analytically verify expected sign of ∂r/∂u

2. **Add Component-Level Debug Prints:**
   ```cpp
   u → MagMoment (should be +)
   MagMoment → muhat (check sign)
   muhat → Tb (magnetic torque, should be +)
   Tb → tau (net torque, check sign)
   tau → wdot (angular acceleration, check sign)
   wdot → R_coil (rotation, check direction)
   R_coil → p_L (interface position, check direction)
   p_L → residual (check sign of p_f - p_L)
   ```

3. **Verify J_yu Extraction:**
   - Print all intermediate Dual.deriv values through the chain
   - Confirm signs at each step match expected physics

4. **Cross-Check with Reduced6d:**
   - Does reduced6d have the same sign error?
   - If reduced6d is correct, compare formulas line-by-line

---

## Technical Debt

1. **P0:** Resolve control gradient sign error (BLOCKER for S14 completion)
2. **P1:** Add inline comments documenting residual convention at definition sites
3. **P1:** Create unit test with analytical verification for simple 1-DOF case
4. **P2:** Document magnetic torque sign conventions in physics code

---

## Conclusion

Sprint S14-Step5A successfully audited the residual definitions and implicit differentiation formulas, confirming that:
- ✅ Residual convention `r = computed - state` is consistent
- ✅ IFT formula `grad_u = -J_yu^T * λ` is mathematically correct
- ✅ FULLSTATE and reduced6d use the same formula

However, the control gradient sign error is **NOT** a simple global sign flip. Testing shows mixed results:
- Component 0: sign improves with flip, but magnitude still wrong (22x error)
- Component 2: sign WORSENS with flip

This indicates a **fundamental issue** in either:
1. Residual definition semantics (shooting method interpretation)
2. J_yu computation (magnetic torque chain has sign error)
3. Component indexing/packing mismatch

**RECOMMENDATION:** Do NOT proceed to Step 6 (Linearization) until this is resolved. A systematic debug trace with a minimal analytical test case is required to isolate the root cause.

**STATUS:** INCOMPLETE - BLOCKED by complex sign/residual convention issue requiring deeper investigation.

