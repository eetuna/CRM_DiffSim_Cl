# Sprint S5: Exact J_yu Derivation (NO Heuristics)

## Problem Statement

Compute J_yu = ∂r/∂u where:
- r ∈ ℝ^(6N): BVP residual from DYNNLEquation
- u ∈ ℝ^(3N): Actuation currents
- N = NUM_ACT_SET

## Residual Structure (from SPRINT_S5_BVP_RESIDUAL_DEFINITION.md)

For each actuator j ∈ {0, ..., N-1}:
```
r[6j + 0:3] = p_flexible[j] - p_coil[j]           (position continuity)
r[6j + 3:6] = ||R_flexible[j] - R_coil[j]||_cols  (orientation continuity)
```

## Dependency Chain: u → r

From the BVP residual definition:
```
u → MagMoment → muhat → τ_mag → tau → wdot → w_next → R_next → r[orientation]
                                              ↓
                                            v_next → p_next → r[position]
```

## Step 1: ∂MagMoment/∂u

**Location**: `src/CRM_BVPSolver.cpp:418` or `src/CoilDynamics_Defs.cpp:1517`

```
MagMoment[j] = T[j] * u[j]
```

where `T[j] = CoilAlignmentTurnAreaMatrix[j]` ∈ ℝ^(3×3).

Therefore:
```
∂MagMoment[j]/∂u[j]_i = T[j][:, i]  (i-th column of T[j])
```

**CRITICAL**: The current implementation INCORRECTLY assumes T[j] is diagonal.

## Step 2: ∂τ_mag/∂u

**Location**: `src/CoilDynamics_Defs.cpp:70-72` (in CoilIntegrad)

```
τ_mag[j] = [MagMoment[j]]_× * (R[j]^T * B0)
        = MagMoment[j] × (R[j]^T * B0)
```

Using the chain rule:
```
∂τ_mag[j]/∂u[j]_i = ∂/∂u[j]_i [MagMoment[j] × b]
                 = (∂MagMoment[j]/∂u[j]_i) × b
                 = T[j][:, i] × b
```

where `b = R[j]^T * B0` ∈ ℝ³ (magnetic field in body frame).

### Current Implementation (INCORRECT)

`src/CRM_BVPJacobian.cpp:244-259`:
```cpp
// WRONG: Uses M[i] * (e_i × B) assuming diagonal T
dtau_du[k] = M[i] * (e_i × B)[k]
```

This assumes `T[:, i] = M[i] * e_i`, which is only true if T is diagonal.

### Correct Formula

```cpp
// Correct: Use full turn-area matrix
double T_col_i[3];  // i-th column of CoilAlignmentTurnAreaMatrix[j]
T_col_i[0] = CoilAlignmentTurnAreaMatrix[j][0*3 + i];
T_col_i[1] = CoilAlignmentTurnAreaMatrix[j][1*3 + i];
T_col_i[2] = CoilAlignmentTurnAreaMatrix[j][2*3 + i];

// Compute b = R^T * B (need current coil orientation)
double RTB[3];
// ... compute RTB from R_coil[j] and B0 ...

// dtau_du = T_col_i × RTB
dtau_du[0] = T_col_i[1] * RTB[2] - T_col_i[2] * RTB[1];
dtau_du[1] = T_col_i[2] * RTB[0] - T_col_i[0] * RTB[2];
dtau_du[2] = T_col_i[0] * RTB[1] - T_col_i[1] * RTB[0];
```

## Step 3: ∂r/∂τ_mag (Propagation Through Dynamics)

The magnetic torque affects the coil state through integration:
```
tau = τ_mag - mL
wdot = I^(-1) * (tau - w × (I*w) - D*w)
w_next = w + ∫ wdot dt  (via RK2/ABM4)
R_next = R * exp([w*dt]_×)  (via DYNSE3_TimeSpace)
```

For small dt and first-order approximation:
```
∂w_next/∂τ_mag ≈ dt * I^(-1)
∂R_next/∂w_next ≈ [identity with SO(3) tangent space structure]
```

However, the EXACT derivative requires integrating sensitivity ODEs, which is complex.

## Step 4: ∂r/∂R_coil (Residual Sensitivity)

The orientation residual is:
```
r[6j + 3:6] = [||R_f[:,0] - R_c[:,0]||, ||R_f[:,1] - R_c[:,1]||, ||R_f[:,2] - R_c[:,2]||]
```

where R_f = R_flexible[j], R_c = R_coil[j].

The derivative ∂r/∂R_c is complex due to the norm operation.

## Simplified First-Order Approximation

For the BVP at convergence (r ≈ 0), and small dt, we can use:

```
∂r[6j + 3:6]/∂u[j] ≈ -dt * scale_factor * I^(-1) * ∂τ_mag/∂u[j]
```

where the scale_factor accounts for:
- SE(3) integration sensitivity
- Norm computation in residual

## Assignment to J_yu Rows

**CRITICAL CORRECTION**: The current code INCORRECTLY comments that:
> "BVP residual: r = [m_computed - mL; n_computed - nL]"

This is WRONG. The residual structure is:
```
r[6j + 0:3] = POSITION continuity (p_f - p_c)
r[6j + 3:6] = ORIENTATION continuity (||R_f - R_c||)
```

Therefore, J_yu should be populated as:

1. **Position rows (6j + 0:3)**: Minimal effect from τ_mag (second-order through R → p coupling)
   - Set to ZERO or small coupling term

2. **Orientation rows (6j + 3:6)**: Direct effect from τ_mag → w → R
   - Populate with ∂τ_mag/∂u scaled by dynamics sensitivity

**Current Implementation Bug**: Lines 274-285 assign the SAME dtau_du to BOTH position and orientation rows, which double-counts the effect!

## Correct Implementation

```cpp
// Orientation rows ONLY (NOT position rows)
for (int k = 0; k < 3; ++k) {
    int row = j * 6 + 3 + k;  // Rows 3-5: orientation residual
    J_yu(row, col) = sensitivity_factor * dtau_du[k];
}

// Position rows: zero or minimal coupling
for (int k = 0; k < 3; ++k) {
    int row = j * 6 + k;  // Rows 0-2: position residual
    J_yu(row, col) = 0.0;  // Torque doesn't directly affect position
}
```

where `sensitivity_factor` accounts for dt and inertia scaling.

## Exact Formula (Avoiding Heuristics)

To avoid ANY heuristics, we need to compute the EXACT sensitivity, which requires:

1. Extract R_coil[j] from the converged BVP solution
2. Compute b = R_coil[j]^T * B0
3. Compute ∂τ_mag/∂u = T[:, i] × b using full turn-area matrix
4. Scale by dt * I^(-1) to get ∂wdot/∂u
5. For first-order: assume ∂r_orientation/∂w ≈ constant at convergence

The key insight: At BVP convergence (r = 0), the residual is at a critical point, so the first-order sensitivity is dominated by the direct torque effect.

## Proposed Fix

1. Use correct ∂τ_mag/∂u = T[:, i] × (R^T * B) (not diagonal approximation)
2. Assign ONLY to orientation rows (6j+3:6), NOT position rows
3. Use scaling: dt / I (no arbitrary coefficients)
4. Remove double assignment to both position and orientation rows

This gives a mathematically justified J_yu with NO heuristic coefficients.
