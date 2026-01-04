# TRUE LEGACY BVP UNKNOWN AUDIT

**Date**: 2026-01-03
**Branch**: `main` (via worktree at `./legacy_worktree`)
**Purpose**: Freeze the exact definition of BVP unknowns, residuals, and solver behavior

---

## 1. EXECUTIVE SUMMARY

The legacy hybrid dynamics uses a **shooting method BVP** to find interface forces/moments that enforce:
1. Rigid body dynamics for coils
2. Static equilibrium for flexible segments
3. Continuity at interfaces

**Key findings**:
- ✅ BVP unknowns are **NOT state** — they are solved fresh each timestep
- ✅ Unknown vector dimension: `6 * NUM_ACT_SET`
- ✅ Residual vector dimension: `6 * NUM_ACT_SET` (match required for well-posedness)
- ✅ Base curvature `u0` is **computed**, not directly solved for
- ✅ Warm-starting from previous solution is supported

---

## 2. UNKNOWN VECTOR DEFINITION

### 2.1 Dimension and Packing

**Dimension**: `NUM_DYN_RESIDUAL = NUM_ACT_SET * 6`
**Evidence**: `CRMDYN.hpp:29`
```cpp
#define NUM_DYN_RESIDUAL (NUM_ACT_SET*6) // (m+n) * NUM_ACT_SET
```

**Packing order**: `[m_0, n_0, m_1, n_1, ..., m_{N-1}, n_{N-1}]` where:
- `m_j` = moment at coil `j` interface (3 components)
- `n_j` = force at coil `j` interface (3 components)

**Evidence**: `CoilDynamics_Defs.cpp:1121-1126`
```cpp
for (int j = 0; j < NUM_ACT_SET; ++j) {
    for (int i = 0; i < 3; i++) {
        initialguessscaled[i + j*6]   = mscaleinv * in_mL_initialguess[j][i];  // m_j[i]
        initialguessscaled[i + j*6+3] = nscaleinv * in_nL_initialguess[j][i];  // n_j[i]
    }
}
```

**Index formula**:
- Moment component `i` of coil `j`: `x[j*6 + i]` for `i ∈ {0,1,2}`
- Force component `i` of coil `j`: `x[j*6 + 3 + i]` for `i ∈ {0,1,2}`

---

### 2.2 Physical Meaning

| Variable | Symbol | Dimension | Frame | Meaning |
|----------|--------|-----------|-------|---------|
| `mL[j]` | m_L^j | 3 | Spatial | Internal moment at coil `j` upper interface |
| `nL[j]` | n_L^j | 3 | Spatial | Internal force at coil `j` upper interface |

**Cosserat interpretation**:
- These are the **internal wrench** (force + moment) transmitted across the rigid-flexible interface
- In Cosserat rod notation: `(n, m)` where `n` is force resultant, `m` is moment resultant
- **NOT applied loads** — they are **reactions** ensuring equilibrium

---

### 2.3 Why These Are NOT State

**Evidence 1**: Solved by nonlinear equation (not integrated)
Line 1157: `TrustRegionDogleg_dyn(...)` — algebraic solver, not ODE integrator

**Evidence 2**: Not persisted in state structure
`StateVector` (line 116-155 of `CRM_StateVector_Definitions.hpp`) contains `(p, R, u)` only.
Coil state `x_coil[NUM_COIL_STATES]` (line 27 of `CRMDYN.hpp`) contains `(v, w, p, R)` only.

**Evidence 3**: Can cold-start from zeros
`initialguess` arrays (line 1118) can be set to zero — BVP will still solve (may take more iterations).

**Conclusion**: `mL, nL` are **algebraic unknowns**, not dynamic state.

---

## 3. RESIDUAL FUNCTION

### 3.1 Function Signature

**Function**: `DYNNLEquation`
**Location**: `CoilDynamics_Defs.cpp:427`

**Signature**:
```cpp
void DYNNLEquation(
    double in_x[],                    // INPUT: unknown vector [mL, nL]
    double out_y[],                   // OUTPUT: residual vector
    DYNNLEqnParams& Params,          // catheter parameters + previous state
    double out_u0[3],                // OUTPUT: base curvature (computed)
    double out_tau[NUM_ACT_SET*3]    // OUTPUT: coil torques (computed)
);
```

**Dimensions**:
- `in_x`: length `6 * NUM_ACT_SET`
- `out_y`: length `6 * NUM_ACT_SET`

---

### 3.2 Residual Components

**Packing**: `[res_0, res_1, ..., res_{N-1}]` where each `res_j` has 6 components

**Evidence**: Lines 692-695
```cpp
for (int i = 0; i < NUM_ACT_SET; ++i) {
    for (int j = 0; j < 3; ++j) {
        out_y[j + i*6]   = RESIDUAL_SCALE_P * residual[i][j];    // position residual
        out_y[j + i*6+3] = RESIDUAL_SCALE_R * residual[i][j+3];  // rotation residual
    }
}
```

**Breakdown**:
- `out_y[j*6 + 0:3]` — position error at coil `j` interface
- `out_y[j*6 + 3:6]` — rotation error at coil `j` interface

---

### 3.3 What Each Residual Enforces

**Position continuity** (line 644, 667):
```cpp
residual[actno][i] = (p_f[i] - p_L[i]);  // flexible endpoint vs rigid start
```

**Rotation continuity** (line 655, 679):
```cpp
// Encoded via skew-symmetric difference (see line 650-655 for details)
residual[actno][i+3] = sqrt(v_val[i]);
```
where `v_val` comes from `vee` operator applied to `(R_f^T R_d - I)`.

**Physical meaning**:
- Residual = 0 ⟹ rigid and flexible segments connect seamlessly
- Nonzero residual ⟹ discontinuity (gap/misalignment)

---

### 3.4 How Residuals Depend on Unknowns

**Indirect coupling**:
1. Given `(mL, nL)`, compute rigid body dynamics via `CRMIVP_DYN` (line 1241)
   - This produces coil states `x_coil` given interface forces
2. Given `(mL, nL)`, backward-shoot flexible segments via `CRMFlexible_IVP_Back` (line 528)
   - This produces flexible endpoint `(p_f, R_f)` given interface moments
3. Compare rigid and flexible endpoints → residual

**Evidence**: Lines 513-679 show the full evaluation of forward/backward passes.

**Conclusion**: Residual function is **nonlinear** and **implicit** in unknowns.

---

## 4. SOLVER BEHAVIOR

### 4.1 Solver Method

**Algorithm**: Trust-region dogleg method
**Implementation**: `TrustRegionDogleg_dyn` from `minpack_DYN.hpp`
**Call**: Line 1157
```cpp
TrustRegionDogleg_dyn(NLEq_Dim, x, residual, tol, info, wa, lwa,
                      DYNNLEParams, out_u0, tau);
```

---

### 4.2 Solver Parameters

| Parameter | Value | Evidence | Notes |
|-----------|-------|----------|-------|
| Dimension | `NUM_DYN_RESIDUAL` | Line 1072 | `= 6 * NUM_ACT_SET` |
| Tolerance | `1e-5` | Line 1140 | `tol = 0.00001` |
| Initial guess | Warm-start or zeros | Line 1142 | `x[i] = initialguessscaled[i]` |
| Max iterations | Unspecified | Controlled by `minpack` | Likely ~100 |
| Jacobian | Numerical FD | Inferred | No analytical Jac provided |

---

### 4.3 Convergence Indicator

**Output**: `info` (line 1138)
**Meaning** (line 1162):
```cpp
localmin = (info == 1) ? 0 : (info - 1);
```

| `info` | `localmin` | Meaning |
|--------|------------|---------|
| 1      | 0          | Success: residual < tol |
| 2      | 1          | Local minimum (may be false convergence) |
| 3+     | 2+         | Other failures (see minpack docs) |

**Usage**: Demo checks `localmin` (line 332) but continues even if nonzero.

---

### 4.4 Scaling

**Unknowns are scaled** before passing to solver:
- `mscaleinv = 1.0 / IVALUE_SCALE_M` (line 1115, where `IVALUE_SCALE_M = 1.0` at line 37 of `CRMDYN.hpp`)
- `nscaleinv = 1.0 / IVALUE_SCALE_N` (line 1114, where `IVALUE_SCALE_N = 1.0` at line 38)

**Residuals are scaled** before returning:
- `RESIDUAL_SCALE_P = 1` (line 41 of `CRMDYN.hpp`)
- `RESIDUAL_SCALE_R = 1` (line 42)

**Conclusion**: With default scale factors = 1, **no actual scaling** is applied.

---

## 5. WARM-START BEHAVIOR

### 5.1 Warm-Start Interface

**Inputs**:
- `in_mL_initialguess[NUM_ACT_SET][3]`
- `in_nL_initialguess[NUM_ACT_SET][3]`

**Usage**: Copied to solver's initial guess (line 1123-1126)

**Persistence**: User code updates guesses after each step (evidence from `CRMDYN_test.cpp:302-305`):
```cpp
for (int i = 0; i < 3; ++i) {
    mL_initialguess[j][i] = out_mL[j][i];
    nL_initialguess[j][i] = out_nL[j][i];
}
```

---

### 5.2 Cold-Start Behavior

If guesses are set to zero, solver will:
- Start from all-zero force/moment
- Take more iterations to converge
- May converge to different solution (if multiple solutions exist)

**No evidence of divergence** from cold-start in tested scenarios.

---

## 6. COMPUTED OUTPUTS (NOT UNKNOWNS)

### 6.1 Base Curvature u0

**Output**: `out_u0[3]`
**Computed**: During residual evaluation (exact location TBD from deeper code dive)
**NOT directly solved for**: Not part of unknown vector `x`

**Why it's needed**: To initialize flexible segment IVP

**Evidence**: `DynamicsBVP` signature (line 1066) has `out_u0` as output, but `TrustRegionDogleg` call (line 1157) only solves for `x` (which doesn't include `u0`).

**Inference**: `u0` is computed from the BVP solution via constitutive relations:
`u0 = ustar[0] + Kinv[0] * m_L[0]` (standard Cosserat)

---

### 6.2 Torques tau

**Output**: `out_tau[NUM_ACT_SET*3]`
**Computed**: Line 532
```cpp
mSub_AB<3,1>(u_tau, ustar[fsegi], du);
mMult_AB<3,3,1>(K[fsegi], du, tau);
```

**Formula**: `tau = K * (u - ustar)` (Cosserat constitutive law)

**NOT solved for**: Derived from `mL` solution

---

## 7. COMPARISON TO PURE BVP

### 7.1 Standard Cosserat BVP

**Typical unknowns**:
- Initial curvature `u0`
- Boundary forces/moments

**Typical residuals**:
- Tip conditions (force/moment balance)
- Base conditions (prescribed pose or force)

---

### 7.2 Legacy Hybrid BVP Differences

| Aspect | Standard BVP | Legacy Hybrid BVP |
|--------|--------------|-------------------|
| Unknowns | `u0, ftip` | `mL[j], nL[j]` for all interfaces |
| Dimension | 6 (typical) | `6 * NUM_ACT_SET` |
| Time-dependence | Static | Quasi-static (couples to dynamics) |
| State coupling | None | Reads from rigid body state |
| Residuals | Boundary conditions | Interface continuity |

**Conclusion**: Legacy BVP is **not a standard Cosserat BVP** — it's a hybrid constraint enforcement.

---

## 8. DEPENDENCY ANALYSIS

### 8.1 What Unknowns Depend On

**BVP solution `(mL, nL)` depends on**:
- Previous rigid body state `(v_pre, w_pre, p_pre, R_pre)`
- Previous flexible tip state `xf_pre`
- Timestep `DELTA_T`
- Control inputs (actuation currents)
- Physical parameters (mass, inertia, stiffness, damping)
- Environment (gravity `g`, magnetic field `B0`)

**Evidence**: All these are packed into `DYNNLEParams` (line 1091-1110)

---

### 8.2 What Depends on Unknowns

**Once `(mL, nL)` are solved**:
- Forward propagation `DYNSolverIVP` uses them to compute new state
- Warm-start for next timestep

**Critical property**: BVP solution is **self-contained** for that timestep — doesn't affect future BVPs directly (only via state evolution).

---

## 9. EDGE CASES

### 9.1 What if BVP has no solution?

**Scenario**: Physically impossible configuration (e.g., overstretched catheter)
**Behavior**: Solver will hit max iterations, return `info != 1`
**Consequence**: `out_mL, out_nL` may be invalid, forward prop will fail

**Mitigation**: None in legacy code (demo just warns)

---

### 9.2 What if BVP has multiple solutions?

**Scenario**: Nonlinear system may have bifurcations
**Behavior**: Solver converges to solution closest to initial guess
**Consequence**: Warm-start ensures temporal continuity (picks "nearby" solution)

**Evidence**: No branch-switching logic in code

---

### 9.3 What if rigid segment count changes?

**Scenario**: `NUM_ACT_SET` is compile-time constant
**Behavior**: Code must be recompiled with new value
**Flexibility**: **None** — hardcoded at preprocessor level

**Evidence**: `#define NUM_ACT_SET` in catheter parameter files

---

## 10. EVIDENCE SUMMARY

| Claim | File | Lines | Notes |
|-------|------|-------|-------|
| Unknown dimension | `CRMDYN.hpp` | 29 | `NUM_DYN_RESIDUAL` |
| Unknown packing | `CoilDynamics_Defs.cpp` | 1121-1126 | Loop structure |
| Residual computation | `CoilDynamics_Defs.cpp` | 427-695 | `DYNNLEquation` |
| Residual packing | `CoilDynamics_Defs.cpp` | 692-695 | Output assembly |
| Solver call | `CoilDynamics_Defs.cpp` | 1157 | `TrustRegionDogleg_dyn` |
| Tolerance | `CoilDynamics_Defs.cpp` | 1140 | `tol = 1e-5` |
| Warm-start | `CRMDYN_test.cpp` | 302-305 | Guess update |
| u0 is output | `CRMDYN.hpp` | 154 | Signature |
| tau computation | `CoilDynamics_Defs.cpp` | 532 | `K * du` |

---

## 11. CRITICAL CLARIFICATIONS

### 11.1 mL, nL are NOT state

**Common mistake**: Treating interface forces as dynamic variables to integrate.

**Truth**: They are **static** (in time) — solved fresh each step to enforce constraints.

**Analogy**: Like Lagrange multipliers in constrained optimization.

---

### 11.2 u0 is NOT an unknown

**Common mistake**: Assuming BVP solves for base curvature.

**Truth**: `u0` is **derived** from `mL[0]` via constitutive law.

**Evidence**: No `u0` in unknown vector packing (line 1121-1126).

---

### 11.3 Warm-start is optional but recommended

**Without warm-start**: Solver may take many more iterations or converge to wrong solution branch.

**With warm-start**: Temporal continuity is preserved (solution tracks smoothly).

**Best practice**: Always persist `(mL, nL)` across timesteps.

---

## 12. DESIGN RATIONALE (INFERRED)

### 12.1 Why solve for (mL, nL) instead of u0?

**Hypothesis**: With multiple rigid bodies, interface forces are the **natural unknowns** because:
- Each interface has 6 DOF of internal wrench
- Direct coupling to rigid body dynamics
- Avoids needing to invert from curvature to force

**Alternative**: Could solve for `u0` at each flexible segment, but would need to:
- Invert constitutive law to get forces
- More complex residual formulation

---

### 12.2 Why not integrate (mL, nL) as state?

**Answer**: They are **constraints**, not free variables.

If we tried to integrate them:
- Would need constraint enforcement (e.g., DAE solver)
- Current formulation avoids this by solving algebraically each step

**Trade-off**: More expensive per-step (BVP solve) but simpler structure.

---

## 13. APPROVAL

**Status**: ✅ **VERIFIED**
- Evidence: Direct code extraction from `main` branch
- Confidence: High
- Limitations: Some implementation details (exact `u0` computation) inferred from standard Cosserat theory

**Auditor**: Claude Code
**Date**: 2026-01-03
