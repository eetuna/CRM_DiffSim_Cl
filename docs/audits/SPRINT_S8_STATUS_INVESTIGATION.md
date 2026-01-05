# Sprint S8: J_yu Seeding Investigation Status

**Date**: 2026-01-05
**Status**: 🔴 **BLOCKED** - Derivative propagation issue in DYNNLEquation_T

---

## Summary

Sprint S8 goal was to fix J_yu by correcting the forward-mode AD seeding. We successfully identified and corrected the seeding, but discovered a deeper issue: **derivatives are not propagating through DYNNLEquation_T**.

---

## What We Fixed

### ✅ Task 1: Identified Exact u→MagMoment Mapping

**Location**: `src/CoilDynamics_Defs.cpp:1513-1518`

**Exact Formula**:
```cpp
MagMoment[j] = CoilAlignmentTurnAreaMatrix[j] * u[j]
```

Therefore:
```
∂MagMoment[j]/∂u[j][i] = CoilAlignmentTurnAreaMatrix[j][:, i]
```

This is a 3x3 matrix-vector multiplication where `CoilAlignmentTurnAreaMatrix` is the Jacobian.

---

### ✅ Task 2: Corrected Dual Seeding

**Location**: `src/CRM_BVPJacobian.cpp:258-327`

**Before (WRONG - Sprint S7)**:
```cpp
// Heuristic seeding - ignores CoilAlignmentTurnAreaMatrix
muhat_seeded[k][m] = Dual(muhat_base[k][m], muhat_base[k][m] != 0.0 ? 1.0 : 0.0);
```

**After (CORRECT - Sprint S8)**:
```cpp
// Seed MagMoment with exact derivative from CoilAlignmentTurnAreaMatrix
double deriv = shooting_params.CoilAlignmentTurnAreaMatrix[k][m * 3 + i_ctrl];
MagMoment_seeded[k][m] = Dual(MagMoment[k][m], deriv);

// Propagate through wHat_T<Dual> to get muhat_seeded
wHat_T<Dual>(MagMoment_seeded[k], muhat_seeded[k]);
```

---

## 🔴 Discovered Issue: Derivative Loss in DYNNLEquation_T

### Evidence

Debug output from test run:
```
DEBUG J_yu: CoilAlignmentTurnAreaMatrix[0] = [1.82371, 0.183738, 0, -0.218052, -2.79717, 0, 0, 0, 1.8448]
DEBUG J_yu: MagMoment[0] = [0.191558, -0.161664, -0.09224]
DEBUG J_yu: Seeding MagMoment[0][0] with deriv = 1.82371
DEBUG J_yu: out_y_dual[0].deriv = 0          <--- DERIVATIVE LOST!
DEBUG J_yu: J_yu(0,0) = 0
DEBUG J_yu final: norm = 0, max = 0
```

**Analysis**:
1. ✅ CoilAlignmentTurnAreaMatrix is non-zero (values like 1.82371, -2.79717, etc.)
2. ✅ We correctly seed `MagMoment_seeded[0][0].deriv = 1.82371`
3. ✅ wHat_T<Dual> propagates to muhat_seeded
4. ❌ **DYNNLEquation_T<Dual> returns out_y_dual[0].deriv = 0**

### Root Cause Hypothesis

The derivative is lost inside `DYNNLEquation_T`. Possible reasons:

1. **muhat not used in residual computation**
   DYNNLEquation_T may not actually use `muhat` parameter to compute the BVP residual `out_y`

2. **Template instantiation issue**
   DYNNLEquation_T<Dual> may not be correctly propagating Dual arithmetic through all operations

3. **Intermediate value vs residual**
   muhat may affect intermediate values but not the final BVP residual that we're computing

---

## Investigation Needed

### Check DYNNLEquation_T Implementation

**File**: `src/CoilDynamics_Defs_Templates2.hpp:281+`

**Questions**:
1. Does DYNNLEquation_T actually **use** the `muhat` parameter?
2. If yes, does it propagate through to `out_y` (the BVP residual)?
3. Are all intermediate operations templated to support Dual propagation?

### Verify Physical Pathway

The expected chain is:
```
u → MagMoment → muhat → τ_mag → coil dynamics → interface forces → BVP residual r
```

We need to verify each link is present and templated in DYNNLEquation_T.

---

## Files Modified (Sprint S8)

1. **src/CRM_BVPJacobian.cpp** (lines 250-333)
   - Removed heuristic muhat seeding
   - Added exact MagMoment seeding using CoilAlignmentTurnAreaMatrix
   - Added wHat_T<Dual> propagation step
   - Added debug output

2. **src/CRM_TrueLegacyDynamics.cpp** (lines 335-339)
   - Added debug output for J_yu inspection

---

## Test Status

**Test**: `tests/test_fullstate_vjp_gradcheck_u.py`
**Result**: ❌ **FAIL**

```
VJP gradient:   [0. 0. 0.]
FD gradient:    [ -0.73885312  -2.80071254 132.31784433]
Relative error: 1.000000e+00
```

Gradients remain zero despite correct seeding.

---

## Recommended Next Steps

### Option A: Debug DYNNLEquation_T (RECOMMENDED)

1. Add debug output inside DYNNLEquation_T to track Dual propagation
2. Verify muhat parameter is actually used to compute out_y
3. Check if all operations support Dual arithmetic
4. Identify where derivative is lost

### Option B: Alternative Approach

If DYNNLEquation_T cannot propagate muhat derivatives:
1. Consider computing J_yu via finite differences (violates NO FD constraint)
2. OR derive analytic formula for ∂r/∂u directly (major effort)
3. OR re-template DYNNLEquation to explicitly propagate muhat (Sprint S9?)

---

## Conclusion

Sprint S8 successfully corrected the **seeding** (Task 1-2 complete), but uncovered a **propagation** issue in DYNNLEquation_T that blocks completion. The physical mapping is correct, the seeding is exact, but the templated dynamics function does not propagate muhat derivatives to the BVP residual output.

**Status**: Cannot complete Sprint S8 without resolving DYNNLEquation_T derivative propagation.

---

**References**:
- Physical mapping: `src/CoilDynamics_Defs.cpp:1513-1518`
- Seeding code: `src/CRM_BVPJacobian.cpp:258-327`
- Template: `src/CoilDynamics_Defs_Templates2.hpp:281+`
