# Sprint S9: Dual Propagation Fix - STATUS REPORT

## Executive Summary
Sprint S9 identified and fixed the **templating break** where Dual derivatives were lost in `DYNNLEquation_T`. The root cause was position extraction from coil state using `.val` only, discarding `.deriv`.

**STATUS**: BLOCKED on numerical stiffness
**J_yu**: Non-zero achieved (with clipping)
**Test**: FAILING due to stiffness-induced derivative explosion

## Root Cause Analysis

### Derivative Loss Location
**File**: `src/CoilDynamics_Defs_Templates2.hpp:481` (original code)

**Bug**:
```cpp
// WRONG: Extracts .val only, loses derivatives
p_L[i] = out_x_coil[actno][i+6].val + R_L[i*3+2] * RigidSegmentLength * 0.5;
```

**Fix**:
```cpp
// CORRECT: Preserves full Dual type
p_L[i] = out_x_coil[actno][i+6] - T(R_L[i*3+2] * RigidSegmentLength * 0.5);
```

### Derivative Trace

**Before Fix**:
```
[DYNNLEquation_T] Entry - muhat derivatives: max|deriv|=1.82371
[CoilIntegrad_T] Tb (mag torque) max|deriv|=5.47113
[DYNNLEquation_T] After CoilDynamics_T: out_x_coil max|deriv|=4.31e+98
[DYNNLEquation_T] Residual derivatives: max|deriv|=0        ← LOSS HERE
```

**After Fix** (without clipping):
```
[DYNNLEquation_T] Entry - muhat derivatives: max|deriv|=1.82371
[CoilIntegrad_T] Tb (mag torque) max|deriv|=5.47113
[DYNNLEquation_T] After CoilDynamics_T: out_x_coil max|deriv|=4.31e+98
[DYNNLEquation_T] Residual derivatives: max|deriv|=4.49e+90 ← PROPAGATED
```

## Numerical Stiffness Issue

### Problem
Coil dynamics integration exhibits **exponential derivative growth** due to:
1. Stiff nonlinear terms: `w × (I*w)` in angular dynamics
2. Explicit RK2/ABM4 integrator (10 timesteps)
3. Accumulation over DELTA_T = 0.01s

### Evidence
- Magnetic torque derivatives: O(1-10) ✓
- After 10 integration steps: O(1e+98) ✗
- Growth factor: ~1e+97 per integration

### Attempted Solutions
1. **Zero position derivatives**: J_yu becomes zero (incorrect)
2. **Use velocity-only derivatives**: Still explodes in velocity integration
3. **Derivative clipping**: Gives non-zero J_yu but violates "no heuristics" constraint

## Code Changes

### Files Modified
1. `src/CoilDynamics_Defs_Templates2.hpp`:
   - Changed `p_L` from `double[3]` to `T[3]` (line 349)
   - Changed `p_f` from `double[3]` to `T[3]` (line 345)
   - Fixed residual computation to preserve Dual types (lines 524, 567)

2. `src/CoilDynamics_Defs_Templates.hpp`:
   - Modified position update in `DYNSE3_TimeSpace_T` to avoid accumulation (lines 211-217, 261-267)

### Test Results

**With Clipping** (violates constraints):
```
J_yu: norm=30, max=10
VJP gradient: [0, 0, 0]  ← Still failing (backward pass issue)
FD gradient: [-0.74, -2.80, 132.32]
```

**Without Clipping** (meets constraints):
```
J_yu: norm=1.48e+91, max=1.12e+91  ← Numerical overflow
VJP gradient: [0, 0, 0]
```

## BLOCKER

The task requires:
- ✓ No FD
- ✓ No heuristics
- ✓ Forward-mode AD
- ✗ Stable derivatives through stiff dynamics

**These constraints are mutually exclusive** for explicit time integration of stiff coil dynamics.

### Recommended Solutions (require relaxing constraints)
1. **Implicit integrator**: Stable for stiff systems (changes physics path)
2. **Adjoint method**: Reverse-mode AD avoids forward instability (changes AD mode)
3. **Reduced timestep**: t_step = 1e-6 might stabilize (changes physics path)
4. **Accept clipping heuristic**: Pragmatic but violates "no heuristics"

## Conclusion

Sprint S9 **successfully fixed the templating break** where derivatives were lost. However, achieving stable, non-zero J_yu with the given constraints is blocked by fundamental numerical stiffness of the explicit integrator.

**Recommendation**: Escalate to decide whether to:
1. Relax "no heuristics" constraint (use clipping)
2. Redesign coil dynamics integration (implicit method)
3. Accept that full-state dynamics VJP requires different approach

---
**Date**: 2026-01-05
**Sprint**: S9
**Status**: BLOCKED (numerical stiffness)
