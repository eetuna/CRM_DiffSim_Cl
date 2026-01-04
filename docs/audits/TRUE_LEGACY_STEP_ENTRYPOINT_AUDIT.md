# TRUE LEGACY STEP ENTRYPOINT AUDIT

**Date**: 2026-01-03
**Branch**: `main` (via worktree at `./legacy_worktree`)
**Purpose**: Identify and document the canonical stepping function(s) used in legacy dynamics

---

## 1. METHODOLOGY

### 1.1 Search Strategy

Searched for entrypoint candidates using:
```bash
rg -n "step|Advance|simulate|DynamicsStep|RunDynamics" ./legacy_worktree
```

Filtered for:
- Function definitions (not comments/helpers)
- Called from test/demo executables
- Advance simulation from `t_n` to `t_{n+1}`

### 1.2 Exclusions

**NOT canonical entrypoints**:
- `RK2_step`, `ABM4_step` — internal integrator subroutines
- `DYNSE3_TimeSpace` — analytical SE3 update (not full dynamics)
- `CRM_ForwardKinematics` — static kinematics (no time evolution)

---

## 2. CANONICAL STEP ENTRYPOINT

### 2.1 Identification

**Function**: `DynamicsBVP`
**Location**: `legacy_worktree/src/CoilDynamics_Defs.cpp:1066`
**Role**: Solve for BVP unknowns given previous state

**Signature**:
```cpp
void DynamicsBVP(
    CRMShootingMethodParams& in_Params,      // catheter parameters + previous state
    const double xf[NUM_STATES],             // tip state from t_{n-1}
    double in_mL_initialguess[NUM_ACT_SET][3],
    double in_nL_initialguess[NUM_ACT_SET][3],
    double in_ftip_initialguess[3],
    double out_u0[3],                        // OUTPUT: base curvature
    double out_mL[NUM_ACT_SET][3],          // OUTPUT: interface moments
    double out_nL[NUM_ACT_SET][3],          // OUTPUT: interface forces
    double out_tau[NUM_ACT_SET][3],         // OUTPUT: coil torques
    double out_ftip[3],
    int& out_localmin
);
```

---

### 2.2 Why This is the Canonical Step

**Evidence 1: Call chain from demo**
`main/CRMDYN_test.cpp:262` (inside time-stepping loop):
```cpp
for (int k = 0; k < 4; ++k) {
    CRMShootingMethodParams BVPParams = CRMDYNConstructShootingMethodParamSet(...);

    DynamicsBVP(BVPParams, xf_pre, mL_initialguess, nL_initialguess, ftip_initialguess,
                out_u0, out_mL, out_nL, out_tau, ftip_calc, localmin);  // ← BVP SOLVE

    DYNSolverIVP(BVPParams, out_u0, out_mL, out_nL, out_tau, ftip_calc,
                 true, xf, x_coil, ReportedMarkerPos);                   // ← FORWARD PROP

    // State update (lines 291-311)
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        v_L_pre[j][i] = x_coil[j][i];       // persist velocity
        // ... (update all state)
    }
}
```

**Evidence 2: State flow**
- **Input**: Previous state embedded in `BVPParams` (constructed line 254):
  - `v_L_pre, w_L_pre, pL, RL` — rigid body state
  - `xf_pre` — flexible tip state
  - `mL_initialguess, nL_initialguess` — warm-start

- **Output**: BVP solution `(u0, mL, nL, tau)` enabling forward propagation

**Evidence 3: Solves nonlinear system**
Lines 1156-1157:
```cpp
TrustRegionDogleg_dyn(NLEq_Dim, x, residual, tol, info, wa, lwa,
                      DYNNLEParams, out_u0, tau);
```
- This is the **core algebraic solve** enforcing hybrid dynamics

---

### 2.3 Companion Function

**Function**: `DYNSolverIVP`
**Location**: `legacy_worktree/src/CoilDynamics_Defs.cpp:1195`
**Role**: Forward propagation using BVP solution

**Signature**:
```cpp
void DYNSolverIVP(
    CRMShootingMethodParams& in_Params,
    const double in_u0[3],
    const double in_mL[NUM_ACT_SET][3],
    const double in_nL[NUM_ACT_SET][3],
    const double in_tau[NUM_ACT_SET][3],
    const double in_ftip[3],
    bool in_FinalValueOnly,
    double out_x_N[NUM_STATES],              // OUTPUT: new tip state
    double out_coil_state[NUM_ACT_SET][NUM_COIL_STATES],  // OUTPUT: new coil states
    double out_p_atLocMarkers[][3]
);
```

**Why this is NOT the primary step entrypoint**:
- It's a **forward evaluation** given BVP solution
- Cannot be called standalone (requires `mL, nL` from BVP)
- Called **after** `DynamicsBVP` in loop (line 277)

**Conclusion**: `DYNSolverIVP` is a **helper** for forward propagation, not the step entrypoint.

---

## 3. CALL CHAIN ANALYSIS

### 3.1 Top-Level Entry

**Executable**: `CRMDYN_test`
**Source**: `main/CRMDYN_test.cpp`
**Entry point**: `RunDynamicsExample()` (line 49, called from `main()` line 45)

### 3.2 Step Sequence

```
RunDynamicsExample() [CRMDYN_test.cpp:49]
  └─> for (k=0; k<4; k++)  [line 253]
        ├─> CRMDYNConstructShootingMethodParamSet(...)  [line 254]
        │     └─> Packages state into BVPParams
        │
        ├─> DynamicsBVP(...)  [line 262]  ◄─── PRIMARY STEP ENTRYPOINT
        │     ├─> CRMDYNSolverIVP_Prep(...)  [line 1094]
        │     │     └─> Prepare IVP parameters
        │     └─> TrustRegionDogleg_dyn(...)  [line 1157]
        │           ├─> DYNNLEquation(...)  [called internally]
        │           │     ├─> CRMIVP_DYN(...)  [forward eval rigid bodies]
        │           │     └─> CRMFlexible_IVP_Back(...)  [backward shoot flexible]
        │           └─> Solve for (mL, nL)
        │
        ├─> DYNSolverIVP(...)  [line 277]  ◄─── FORWARD PROPAGATOR
        │     └─> CRMIVP_DYN(...)
        │           ├─> CoilDynamics(...) [for each coil]
        │           └─> CRMFlexForward_pass(...) [for flexible segments]
        │
        └─> State update [lines 291-311]
              ├─> v_L_pre ← x_coil[...][0:3]
              ├─> w_L_pre ← x_coil[...][3:6]
              ├─> pL ← x_coil[...][6:9]
              ├─> RL ← x_coil[...][9:18]
              ├─> xf_pre ← xf[...]
              └─> mL_initialguess ← out_mL  (warm-start)
```

---

### 3.3 Data Flow Diagram

```
                Previous State
                ┌─────────────┐
                │ xf_pre      │
                │ v_L_pre     │
                │ w_L_pre     │
                │ pL, RL      │
                │ mL_guess    │
                └──────┬──────┘
                       │
                       ├──> CRMDYNConstructShootingMethodParamSet
                       │
                       v
                ┌──────────────┐
                │ DynamicsBVP  │  ◄─── STEP ENTRYPOINT
                └──────┬───────┘
                       │
              ┌────────┴────────┐
              │ BVP Solution    │
              │ (u0, mL, nL, τ) │
              └────────┬────────┘
                       │
                       v
                ┌──────────────┐
                │ DYNSolverIVP │  ◄─── FORWARD PROP
                └──────┬───────┘
                       │
              ┌────────┴────────┐
              │  New State      │
              │  xf, x_coil     │
              └────────┬────────┘
                       │
                       └──> Update prev state for next step
```

---

## 4. STEPPING CONTRACT

### 4.1 Inputs (State at t_n)

**Embedded in `CRMShootingMethodParams`**:
- `p0, R0` — base pose
- `v_L_pre[NUM_ACT_SET][3]` — coil linear velocities
- `w_L_pre[NUM_ACT_SET][3]` — coil angular velocities
- `p_pre[NUM_ACT_SET][3]` — coil positions
- `R_pre[NUM_ACT_SET][9]` — coil rotations
- `xf[NUM_STATES]` — tip state (p, R, u)

**Also in params** (not state):
- `DELTA_T` — timestep
- `ActuationCurrents` — control inputs
- `g, B0` — environment (gravity, magnetic field)
- Physical parameters (mass, inertia, stiffness, damping)

**Warm-start guesses**:
- `mL_initialguess[NUM_ACT_SET][3]`
- `nL_initialguess[NUM_ACT_SET][3]`

---

### 4.2 Outputs (State at t_{n+1})

**From `DYNSolverIVP`**:
- `out_x_N[NUM_STATES]` — new tip state
- `out_coil_state[NUM_ACT_SET][NUM_COIL_STATES]` — new coil states
  - `[0:3]` → v
  - `[3:6]` → w
  - `[6:9]` → p
  - `[9:18]` → R

**Intermediate BVP solution** (used for forward prop):
- `out_u0[3]` — base curvature
- `out_mL[NUM_ACT_SET][3]` — interface moments
- `out_nL[NUM_ACT_SET][3]` — interface forces
- `out_tau[NUM_ACT_SET][3]` — coil torques

---

### 4.3 Semantics

**What the step does**:
1. **BVP Solve**: Find interface forces/moments such that:
   - Rigid bodies obey dynamics: `m v̇ = F`, `I ω̇ = τ`
   - Flexible segments obey statics: `n' = -ρAg`, `m' = -u × m`
   - Interfaces are continuous: `p_rigid = p_flex`, `R_rigid = R_flex`

2. **Forward Propagation**: Given BVP solution, integrate:
   - Rigid bodies: ABM4 for `(v, w, p, R)` over `DELTA_T`
   - Flexible segments: Spatial IVP for `(p, R, u)` in arclength `s`

3. **State Update**: Persist new configuration for next timestep

**Time advancement**: `t_{n+1} = t_n + DELTA_T` (evidence: line 177 shows `DELTA_T = 0.05`)

---

## 5. EVIDENCE SUMMARY

| Claim | Evidence File | Lines | Notes |
|-------|---------------|-------|-------|
| Canonical step is DynamicsBVP | `CRMDYN_test.cpp` | 262 | Called in time loop |
| Followed by DYNSolverIVP | `CRMDYN_test.cpp` | 277 | Forward propagation |
| State updated after step | `CRMDYN_test.cpp` | 291-311 | Persist for next iteration |
| BVP solves for (mL, nL) | `CoilDynamics_Defs.cpp` | 1157 | TrustRegionDogleg call |
| Residual enforces continuity | `CoilDynamics_Defs.cpp` | 427-695 | DYNNLEquation |
| Coil dynamics via ABM4 | `CoilDynamics_Defs.cpp` | 106-199 | CoilDynamics function |
| Flexible segments via IVP | `CRMDYN.hpp` | 141 | CRMFlexible_IVP_Back |
| Loop confirms 4 timesteps | `CRMDYN_test.cpp` | 253 | k=0..3 |

---

## 6. ALTERNATIVE INTERPRETATIONS (REJECTED)

### 6.1 Could `DYNSolverIVP` be the primary step?

**Hypothesis**: Maybe `DYNSolverIVP` is the step, and `DynamicsBVP` is just setup.

**Rejection**:
- `DYNSolverIVP` requires BVP solution `(u0, mL, nL, tau)` as input (signature line 1195-1196)
- Cannot cold-start from state alone
- `DynamicsBVP` is what **computes** these inputs from state
- Call order proves: BVP first (line 262), then IVP (line 277)

**Conclusion**: No, `DYNSolverIVP` is a **helper**, not the step entrypoint.

---

### 6.2 Could there be a higher-level wrapper?

**Hypothesis**: Maybe there's a combined `Step()` function that calls both.

**Search evidence**:
```bash
rg -n "void.*Step|void.*Advance" ./legacy_worktree/src ./legacy_worktree/main
```
Result: No such function found.

**Conclusion**: No unified wrapper. User code must call both functions explicitly (as shown in `CRMDYN_test.cpp`).

---

## 7. CORNER CASES

### 7.1 What if BVP fails to converge?

**Indicator**: `out_localmin != 0` (line 1162)
**Evidence**: Test checks `localmin` (line 332)
**Action**: Demo prints warning but continues (no abort)

**Implication**: Step may return invalid state if BVP fails.

---

### 7.2 What about static kinematics (no dynamics)?

**Alternative API**: `CRM_ForwardKinematics` (mentioned in commented code, lines 102-126)
**Purpose**: Solve for static equilibrium given currents (no time evolution)
**Conclusion**: Different use case — **not a stepping function**.

---

## 8. IMPLEMENTATION DETAILS

### 8.1 BVP Solver

**Method**: Trust-region dogleg (line 1157)
**Tolerance**: `1e-5` (line 1140)
**Max iterations**: Controlled by `minpack` (not explicit in code)
**Jacobian**: Numerical finite difference (inferred from `minpack_DYN.hpp`)

---

### 8.2 Rigid Body Integrator

**Method**: ABM4 (Adams-Bashforth-Moulton 4th order predictor-corrector)
**Initialization**: RK2 for first 3 substeps (line 155-156)
**Substeps**: `ceil(DELTA_T / 0.001)` (e.g., 50 substeps for `DELTA_T = 0.05`)

---

### 8.3 Flexible Segment Solver

**Method**: Spatial IVP shooting (backward from tip)
**Integrator**: ABM4 in arclength `s`
**Step size**: `IntegrationStepSize = 0.2` mm (line 73)

---

## 9. COMPARISON TO CURRENT BRANCH

**Current branch** (`milestone-a-hybrid-vjp`):
- Has `crm_dynamics_torch.py` — likely a refactor
- May have different entrypoint naming

**Action item**: Generate `LEGACY_VS_CURRENT_DIFFSIM_DELTA.md` to compare.

---

## 10. CONCLUSION

### 10.1 Canonical Step Entrypoint

**Primary**: `DynamicsBVP` (solves for algebraic unknowns)
**Secondary**: `DYNSolverIVP` (propagates forward using BVP solution)

**Together they form**: A complete timestep `t_n → t_{n+1}`.

---

### 10.2 Call Pattern

```cpp
// Setup
CRMShootingMethodParams params = CRMDYNConstructShootingMethodParamSet(
    catheter_params, configuration, previous_state, ...);

// Step
DynamicsBVP(params, xf_prev, mL_guess, nL_guess, ftip_guess,
            out_u0, out_mL, out_nL, out_tau, out_ftip, localmin);

DYNSolverIVP(params, out_u0, out_mL, out_nL, out_tau, out_ftip, true,
             xf_new, x_coil_new, markers);

// Update for next step
xf_prev = xf_new;
x_coil_prev = x_coil_new;
mL_guess = out_mL;  // warm-start
nL_guess = out_nL;
```

---

### 10.3 Approval

**Status**: ✅ **VERIFIED**
- Evidence: Direct code inspection + call chain trace
- Confidence: High (executable demo confirms usage)
- Limitations: Only examined `CRMDYN_test.cpp`; other demos may differ

**Auditor**: Claude Code
**Date**: 2026-01-03
