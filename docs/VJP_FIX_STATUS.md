# VJP Fix Status Report

## Executive Summary
The VJP backward pass for FULLSTATE dynamics has a **fundamental issue**: the BVP Jacobian `J_yu` (∂r/∂u) requires knowledge of the BVP residual computation, which is inside the compiled `DYNNLEquation` function.

**Current test result**: Component 2 is ~8x too large, components 0 and 1 are incorrect.

## Root Cause Analysis

### Key Finding 1: grad_u_direct = 0 ALWAYS
- The IVP Jacobians (J_p, J_R) do NOT include ∂xf_next/∂w (angular velocity sensitivity)
- Therefore `grad_x_coil[w] = 0` after direct IVP contribution
- This makes `grad_u_direct = (∂w/∂u)^T × grad_w = 0` always
- **Conclusion**: ALL u gradients must flow through `J_yu`

### Key Finding 2: J_yu is the PRIMARY gradient pathway
- Not a correction term - it's the ONLY pathway
- Must equal ∂r/∂u where r(mL, nL; u) is the BVP residual
- The residual r = [m_computed - mL; n_computed - nL]
- Both m_computed and n_computed depend on u through magnetic torque

### Key Finding 3: Cannot derive J_yu without BVP source
- The BVP residual is computed by `DYNNLEquation` (compiled library)
- We only have: ∂τ_mag/∂u = M × (e_i × B) (magnetic torque derivative)
- But need: ∂(m_computed, n_computed)/∂u (how residual changes with u)
- This requires knowing the internal BVP dynamics equations

## Attempted Fixes

### Fix 1: Reorder operations to avoid double-counting
**Result**: No change (confirmed grad_u_direct = 0 regardless of order)

### Fix 2: Populate J_yu[moment rows] = ∂τ/∂u
**Result**: Zero gradients (because v_y[mL] = 0, moment adjoints don't contribute)

### Fix 3: Populate J_yu[force rows] with empirical coefficient
**Result**: Component 2 nearly correct with coeff=0.122, but components 0,1 wrong

### Fix 4: Populate BOTH moment and force rows = ∂τ/∂u
**Result**: 8x too large (suggests coupling is ~1/8 of torque derivative)

## Correct Solution Path

### Option A: Instrument BVP Solver
1. Add instrumentation to `DYNNLEquation` to output ∂r/∂u
2. Use measured values to populate J_yu correctly
3. Verify against finite differences

### Option B: Autodiff Through BVP
1. Implement forward-mode AD through the entire BVP solve
2. Seed with unit vectors for each u component
3. Extract ∂r/∂u analytically

### Option C: Finite Difference J_yu (temporary)
1. For each column i of J_yu:
   - Perturb u[i] by ε
   - Solve BVP → get (mL', nL')
   - Compute residual r'
   - J_yu[:, i] = (r' - r) / ε
2. Use this to validate the backward pass structure
3. Replace with analytical once BVP internals are understood

## Component 1 Mystery

From diagnostic: u[1] DOES affect tip_p (sensitivity ~0.6)

Possible explanations:
1. B field geometry causes asymmetry (∂τ/∂u[1] has specific direction)
2. Coil orientation relative to field creates coupling
3. Test needs multiple forward/backward calls to accumulate gradient

**Need to verify**: Run diagnostic with longer perturbations to rule out numerical noise.

## Recommendations

1. **Immediate**: Use Option C (FD J_yu) to validate backward pass structure
2. **Short-term**: Instrument DYNNLEquation or provide source code access
3. **Long-term**: Implement full autodiff through BVP solver

## Files Modified

- `src/CRM_TrueLegacyDynamics.cpp`: Backward pass (lines 172-464)
- `src/CRM_BVPJacobian.cpp`: J_yu computation (lines 230-290)
- `docs/GRAD_U_PATHWAY_ANALYSIS.md`: Detailed pathway analysis

## Current Code State

J_yu populated as:
```cpp
// Moment rows (0-2): dtau_du[k]
// Force rows (3-5): dtau_du[k]
```

This gives component 2 ~8x too large, suggesting true coupling is ~dtau_du/8 for forces.

But without BVP source, cannot derive the "8" coefficient analytically.
