# Physics Audit Report: u=[0,0,0] and p_tip ≈ [0,0,L]

**Date**: 2025-12-31
**Objective**: Determine whether u=[0,0,0] should produce p_tip≈[0,0,L_inserted] in this repo
**Answer**: **NO** (for default parameters) — intrinsic curvature and gravity are present

---

## 1. Where L_inserted is Applied

### 1.1 Parameter Construction
**File**: `src/CRM_BVPSolver.cpp`
**Function**: `CRMConstructShootingMethodParamSet`
**Line**: 327

```cpp
ShootingParams.Li = MIN(MAX(InsertionLength, 0), length);
```

- `L_inserted` is clamped to `[0, total_length]`
- `total_length` = sum of all segment lengths (line 306)
- For default params: total = 19.85 + 18.3 + 159.4 = 197.55 mm

### 1.2 Integration Setup
**File**: `src/CRM_IVPSolver.cpp`
**Function**: `CRMSolverIVP_Prep` (constructor)
**Lines**: 198-217

```cpp
InsertedLength = in_Li;  // Line 198
// If catheter inserted more than total length, clamp it
if (InsertedLength > SegEndLambdas[in_no_segments - 1])
    InsertedLength = SegEndLambdas[in_no_segments - 1];

// Reorder segments proximal-to-distal
SegBounds[in_no_segments] = InsertedLength;  // Line 206
```

- Sets boundary conditions for integration
- Segments are reordered from distal-to-proximal → proximal-to-distal for numerical integration

### 1.3 Arc-length Integration
**File**: `src/CRMDYN_Numerical_Integration.hpp`
**Function**: `CRMIntegrand_dyn`
**Lines**: 19, 41

```cpp
Length = in_Params.Li;  // Line 19
...
double lambda = Length - s;  // Line 41
```

- Integration variable `s` runs from 0 (entry point) to `Li` (tip)
- Internal coordinate `lambda = Li - s` runs from `Li` (entry) to 0 (tip)
- Used for interpolating distributed loads (fcum, magnetic forces)

---

## 2. What the Parameter Files Imply

### 2.1 Geometry (CatheterParameterSet_1_dyn.txt)
```
CatheterConfig: F A F
  - Flexible segment 0 (distal)
  - Actuator segment (rigid)
  - Flexible segment 1 (proximal)

SegmentLengths: 19.85, 18.3, 159.40 mm
  - Total: 197.55 mm
  - L_inserted = 50.0 mm < 197.55 mm ✓

ustarlist (intrinsic curvature):
  - Seg 0: [1.486e-05, 9.445e-05, 0] rad/mm
  - Seg 1: [7.400e-04, -2.292e-04, 0] rad/mm
```

**⚠️ Non-zero intrinsic curvature**: The catheter has built-in curvature (pre-shaped)

### 2.2 Spatial Configuration (CatheterSpatialConfiguration_1.txt)
```
gravity: [0.0, 0.0, 9.81] m/s²  → +z direction
p0: [0, 0, 0]                   → origin
R0: identity                     → no base rotation
```

**⚠️ Gravity present**: 9.81 m/s² acts in +z direction (pulls catheter down/up depending on orientation)

### 2.3 Physical Interpretation
With L_inserted = 50.0 mm:
- Tip is in **flexible segment 0** (length 19.85 mm)
- ~30 mm of **actuator segment** (rigid, 18.3 mm total)
- ~0 mm of **flexible segment 1** (proximal)

At u=[0,0,0] (zero actuation currents):
- Intrinsic curvature (ustar) causes bending
- Gravity causes deflection
- **Therefore p_tip ≠ [0,0,50] is physically correct**

---

## 3. Case A/B Harness Results

### Build & Run
```bash
cd /workspaces/CRM_DiffSim_Cl/build
cmake ..
make test_physics_audit
cd ..
./build/test_physics_audit
```

### Case A: Baseline (As-Is from Parameter Files)
**Configuration**:
- ustar: [1.486e-05, 9.445e-05, 0] and [7.400e-04, -2.292e-04, 0]
- gravity: [0, 0, 9.81] m/s²

**Results**:
```
status (converged): 0
p_tip: [-0.10105682, -0.38921460, 49.99824082]
deltau0: [-4.407e-07, 1.260e-07, -1.676e-14]
```

**Analysis**:
- ❌ p_tip ≠ [0, 0, 50]
- Deviation: [-0.101, -0.389, -0.0018] mm
- **This is EXPECTED** due to intrinsic curvature + gravity

### Case B: Zero Ustar + Zero Gravity
**Configuration**:
- ustar: [0, 0, 0] (both segments)
- gravity: [0, 0, 0]

**Results**:
```
status (converged): 0
p_tip: [0.0, 0.0, 50.000000000000121]
deltau0: [0, 0, 0]
```

**Analysis**:
- ✅ p_tip ≈ [0, 0, 50] (within 1.21e-13 mm — numerical tolerance)
- **Confirms the model is correct**
- At u=0 with no intrinsic curvature and no gravity, the rod is perfectly straight

---

## 4. Conclusion: Is p_tip ≈ [0,0,L] Expected at u=0?

### Answer: **IT DEPENDS ON THE PARAMETERS**

#### ✅ YES — when:
1. **ustar = 0** (zero intrinsic curvature)
2. **g = 0** (zero gravity)
3. Free-tip boundary condition (no external forces)
4. R0 = identity, p0 = origin
5. u = [0,0,0] (zero actuation)

**Evidence**: Case B demonstrates p_tip = [0, 0, 50.0±1e-13]

#### ❌ NO — for the DEFAULT parameter files because:
1. **Non-zero intrinsic curvature** exists in `ustarlist`:
   - Flexible segment 0: curvature magnitude ≈ 9.56e-05 rad/mm
   - Flexible segment 1: curvature magnitude ≈ 7.74e-04 rad/mm
2. **Gravity** is present: 9.81 m/s² in +z direction
3. These physical effects cause bending even at zero actuation

**Evidence**: Case A demonstrates p_tip = [-0.101, -0.389, 49.998]

---

## 5. Code Evidence for Physical Behavior

### 5.1 Intrinsic Curvature Applied
**File**: `src/CRMDYN_Numerical_Integration.hpp`
**Lines**: 23, 83-86

```cpp
auto& ustar = in_Params.ustar;  // Line 23
...
// Line 83-86: Constitutive relation
for (int i = 0; i < 3; i++) {
    m[i] = K[i*3+0]*(u[0]-ustar[0]) + K[i*3+1]*(u[1]-ustar[1]) + K[i*3+2]*(u[2]-ustar[2]);
}
```

- Moment `m = K * (u - ustar)`
- At u=0: `m = -K * ustar` (restoring moment from intrinsic curvature)

### 5.2 Gravity Applied
**File**: `src/CRMDYN_Numerical_Integration.hpp`
**Lines**: 89-91

```cpp
for (int i = 0; i < 3; i++) {
    fcum_weighted[i] = fcum[i] + g_weighted[i];
}
```

- Distributed load `fcum` includes gravity contribution
- Gravity creates bending moment via `n_dot = R^T * fcum`

### 5.3 Equilibrium Equation
**File**: `src/CRMDYN_Numerical_Integration.hpp`
**Lines**: 107-112

```cpp
// Moment balance: m_dot = R^T * fcum + R^T * ftip × (p_tip - p)
for (int i = 0; i < 3; i++) {
    udot[i] = Kinv[i*3+0]*mdot[0] + Kinv[i*3+1]*mdot[1] + Kinv[i*3+2]*mdot[2];
}
```

- Curvature evolution `u_dot = K^{-1} * m_dot`
- With ustar≠0, equilibrium u ≠ 0 even when external actuation is zero

---

## 6. Recommendation

For **smoke tests** or **unit tests** expecting straight-line behavior:

1. Use **Case B parameters** (zero ustar, zero gravity)
2. OR test with **non-zero actuation** designed to counteract intrinsic curvature
3. OR verify expected **deflection** against analytical bending models with ustar and gravity

**The default parameters represent a physically realistic catheter with pre-curvature and gravity effects.**

---

## Appendix: Quick Diagnostic Commands

```bash
# Build harness
cd /workspaces/CRM_DiffSim_Cl/build && cmake .. && make test_physics_audit

# Run audit
cd /workspaces/CRM_DiffSim_Cl && ./build/test_physics_audit

# Expected output:
#   CASE A: p_tip = [-0.101, -0.389, 49.998]  (curved due to ustar + gravity)
#   CASE B: p_tip = [0.0, 0.0, 50.0]         (straight with zero ustar, zero g)
```
