# Sprint S5: Authoritative BVP Residual Definition

## Task 1 Completion: Authoritative DYNNLEquation Implementation

### Location
**File**: `src/CoilDynamics_Defs.cpp`
**Lines**: 427-703
**Function**: `void DYNNLEquation(double in_x[], double out_y[], DYNNLEqnParams& Params, double out_u0[3], double out_tau[NUM_ACT_SET*3])`

### Declaration
**File**: `src/CRMDYN.hpp`
**Line**: 138

### Call Chain
The authoritative DYNNLEquation is called from:
1. `numerical/minpack_DYN_Defs.cpp` → `DYNSolver` (BVP solver using trust-region method)
2. `src/CRM_BVPJacobian.cpp:75` → `compute_bvp_residual` (used in VJP backward pass)

### Evidence
- **Runtime path**: `DynamicsBVP` (src/CoilDynamics_Defs.cpp:1066) → `DYNSolver` (numerical/minpack_DYN_Defs.cpp) → `DYNNLEquation` (src/CoilDynamics_Defs.cpp:427)
- **Jacobian path**: `true_legacy_step_backward` → `compute_bvp_jacobians_full_analytic` → `compute_bvp_residual` → `DYNNLEquation`

### Legacy/Unused Variants
- `legacy_worktree/src/CoilDynamics_Defs.cpp:427` - old copy from before migration
- `legacy_worktree/numerical/minpack_DYN_Defs.cpp` - calls to old version

These are in the `legacy_worktree` directory and are NOT used at runtime.

---

## Task 2: Explicit BVP Residual Definition r(y,u)

### Notation
- **y** = [mL[0]; nL[0]; mL[1]; nL[1]; ...] ∈ ℝ^(6N) where N = NUM_ACT_SET
  - mL[j] ∈ ℝ³: Interface moment (body frame) at actuator j
  - nL[j] ∈ ℝ³: Interface force (body frame) at actuator j
- **u** = [u[0]; u[1]; ...] ∈ ℝ^(3N): Actuation currents (Amperes)
- **r** = [r[0]; r[1]; ...] ∈ ℝ^(6N): BVP residual

### Residual Structure
For each actuator j ∈ {0, ..., N-1}:

```
r[6j + 0:3] = p_flexible[j] - p_coil[j]     (position continuity)
r[6j + 3:6] = ||R_flexible[j] - R_coil[j]||  (orientation continuity, column-wise Frobenius norms)
```

where:
- `p_flexible[j]`, `R_flexible[j]`: Position and orientation at interface from flexible segment integration (backward from tip)
- `p_coil[j]`, `R_coil[j]`: Position and orientation from coil dynamics integration

### Detailed Computation Path

#### Step 1: Magnetic Moment (u → MagMoment)
**Location**: `src/CoilDynamics_Defs.cpp:1517` or `src/CRM_BVPSolver.cpp:418`

```
MagMoment[j] = CoilAlignmentTurnAreaMatrix[j] * u[j]  ∈ ℝ³
```

where `CoilAlignmentTurnAreaMatrix[j]` ∈ ℝ^(3×3) encodes:
- Coil alignment angles (relative to body axes)
- Turn-area product (N·A) for each coil

#### Step 2: Skew-Symmetric Matrix (MagMoment → muhat)
**Location**: `src/CoilDynamics_Defs.cpp:474`

```
muhat[j] = wHat(MagMoment[j])  ∈ ℝ^(3×3)
```

where `wHat` constructs the skew-symmetric matrix:
```
wHat([a, b, c]) = [ 0  -c   b ]
                  [ c   0  -a ]
                  [-b   a   0 ]
```

#### Step 3: Magnetic Torque (muhat → τ_mag)
**Location**: `src/CoilDynamics_Defs.cpp:70-72` (in `CoilIntegrad`)

```
τ_mag[j] = muhat[j] * R[j]^T * B0  ∈ ℝ³
```

where:
- `R[j]` ∈ SO(3): Current orientation of coil j
- `B0` ∈ ℝ³: Magnetic field vector (spatial frame)

Physical meaning: τ_mag = μ × B (magnetic torque in body frame)

#### Step 4: Net Torque (τ_mag, mL → tau)
**Location**: `src/CoilDynamics_Defs.cpp:72`

```
tau[j] = τ_mag[j] - mL[j]  ∈ ℝ³
```

Physical meaning: Net moment applied to coil = magnetic torque - interface moment from flexible segment

#### Step 5: Angular Acceleration (tau → wdot)
**Location**: `src/CoilDynamics_Defs.cpp:75-81` (in `CoilIntegrad`)

```
residual_w = tau - w × (I*w) - D_w * w
wdot = I^(-1) * residual_w
```

where:
- `w` ∈ ℝ³: Angular velocity (body frame)
- `I` ∈ ℝ^(3×3): Moment of inertia (diagonal)
- `D_w` ∈ ℝ³: Angular damping coefficients
- `w × (I*w)`: Gyroscopic term (Euler's equations)

#### Step 6: Coil Dynamics Integration (wdot, vdot → coil state)
**Location**: `src/CoilDynamics_Defs.cpp:106-199` (`CoilDynamics`)

Uses RK2/ABM4 multi-step integration over time interval dt:
```
x_coil[j]_next = integrate(f(x_coil[j], mL[j], nL[j], u[j]), dt)
```

where `x_coil[j] = [v; w; p; R]` ∈ ℝ^18:
- v[3]: Linear velocity (body frame)
- w[3]: Angular velocity (body frame)
- p[3]: Position (spatial frame)
- R[9]: Orientation matrix (SO(3), row-major)

#### Step 7: Geometric Continuity Residual
**Location**: `src/CoilDynamics_Defs.cpp:631-656` (interior interfaces), `664-679` (base interface)

Position residual:
```
r[6j + 0:3] = p_flexible[j] - p_coil[j]
```

Orientation residual (column-wise Frobenius norm):
```
v1 = R_flexible[j][:,0] - R_coil[j][:,0]
v2 = R_flexible[j][:,1] - R_coil[j][:,1]
v3 = R_flexible[j][:,2] - R_coil[j][:,2]

r[6j + 3] = ||v1||
r[6j + 4] = ||v2||
r[6j + 5] = ||v3||
```

### Scaling
**Location**: `src/CoilDynamics_Defs.cpp:690-697`

Final residual output:
```
out_y[6j + 0:3] = RESIDUAL_SCALE_P * r[6j + 0:3]
out_y[6j + 3:6] = RESIDUAL_SCALE_R * r[6j + 3:6]
```

where (from `src/CRMDYN.hpp:41-42`):
```
RESIDUAL_SCALE_P = 1.0
RESIDUAL_SCALE_R = 1.0
```

### Input Scaling
**Location**: `src/CoilDynamics_Defs.cpp:432-437`

The BVP solver passes scaled variables:
```
mL[j] = IVALUE_SCALE_M * in_x[6j + 0:3]
nL[j] = IVALUE_SCALE_N * in_x[6j + 3:6]
```

where (from `src/CRMDYN.hpp:37-38`):
```
IVALUE_SCALE_M = 1.0
IVALUE_SCALE_N = 1.0
```

Current implementation uses no scaling (all scales = 1.0).

---

## Summary: How u Enters the Residual

The actuation current **u** affects the BVP residual **r(y,u)** through the following physics-based pathway:

```
u → MagMoment → muhat → τ_mag → tau → wdot → w_next → R_next → residual
```

Specifically:
1. **Direct pathway**: u → τ_mag(u) via `τ_mag = wHat(CoilAlignmentTurnAreaMatrix * u) * R^T * B0`
2. **Dynamics**: τ_mag affects coil angular acceleration via Euler equations
3. **Integration**: Angular acceleration integrated to angular velocity, then to orientation
4. **Residual**: Final orientation compared to flexible segment orientation

**Key insight**: The residual r(y,u) is NOT a simple algebraic function. It involves:
- Time integration of coil dynamics (ODE solve)
- SE(3) integration (exponential map on SO(3))
- Geometric continuity conditions

This makes ∂r/∂u a complex derivative requiring careful chain-rule application.
