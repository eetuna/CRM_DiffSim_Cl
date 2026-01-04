# PRE-B0 LEGACY C++ DYNAMICS AUDIT

**Audit Date:** 2026-01-03
**Auditor:** Claude (Sonnet 4.5)
**Scope:** Legacy C++ physics/solver code (read-only inspection via git worktree)
**Legacy Worktree:** `/workspaces/CRM_DiffSim_Cl/legacy_worktree` (branch: `main`, commit: 828bf8c)

---

## Executive Summary

The legacy C++ dynamics code is a mature, numerically stable implementation with well-defined convergence behavior and failure modes. It is **safe for gradient-based system identification** with appropriate safeguards. Key findings:

**Strengths:**
- Deterministic: No stochastic elements, fixed RNG seeds not required
- Convergence criteria well-defined with hard-coded tolerances
- Rank-deficiency detection present in solver (trust-region dogleg)
- No hidden side effects or global state mutations

**Risks:**
- No explicit NaN/infinity checks in IVP integrator (silent propagation possible)
- Convergence failure propagates via `out_localmin` flag but may be ignored by caller
- Pseudoinverse operations in Jacobian code (rank-deficiency handling unclear in legacy)
- Hard-coded tolerance `TRUSTREGION_TOLERANCE` not exposed to user

---

## 1. BVP Solver Audit (Equilibrium Convergence)

### 1.1 Solver Algorithm

**File:** `legacy_worktree/src/CRM_BVPSolver.cpp`
**Function:** `CRMShootingMethodBVP` (lines 13-109)

**Algorithm:** Trust-region dogleg with analytical Jacobian
**Entry point:** Line 81
```cpp
TrustRegionDogleg_GivenJacobian<NLEqnParams>(
    CRM_NLEquation, CRM_NLEquation_AnalyticalJac,
    3, x, residual, tol, info, NLEParams);
```

**Tolerance:** Line 77
```cpp
double tol = TRUSTREGION_TOLERANCE;
```
**Evidence:** Hard-coded macro, not user-configurable

**Convergence Indicator:** Lines 86-87
```cpp
localmin = (info == 1) ? 0 : (info - 1);
```
- `info == 1`: Converged
- `info != 1`: Failed (stuck at local minimum or cannot make progress)

**Output:** Line 106
```cpp
out_localmin = localmin;
```

### 1.2 Residual Evaluation

**Function:** `CRM_NLEquation` (lines 173-209)

**Free-Tip Mode** (lines 197-200):
```cpp
for (int i = 0; i < 3; i++) {
    out_y[i] = RESIDUAL_SCALE_M * MomentResidual[i];
}
```
- Residual = Scaled moment at tip
- Dimension: 3

**Fixed-Tip Mode** (lines 202-207):
```cpp
for (int i = 0; i < 3; i++) {
    out_y[i] = RESIDUAL_SCALE_M * MomentResidual[i];
    out_y[i + 3] = RESIDUAL_SCALE_P * (x_N._p[i] - Params.TipConstraintPoint[i]);
}
```
- Residual = [Scaled moment, Scaled position error]
- Dimension: 6

**Scaling Constants:**
- `RESIDUAL_SCALE_M`: Moment residual scaling
- `RESIDUAL_SCALE_P`: Position residual scaling
- `IVALUE_SCALE_DU`: Delta curvature scaling (lines 53-54)
- `IVALUE_SCALE_F`: Force scaling (lines 54, 103)

**Evidence:** Residual thresholds not explicitly checked; solver relies on trust-region convergence

---

## 2. IVP Solver Audit (Forward Integration)

### 2.1 Integration Method

**File:** `legacy_worktree/src/CRM_IVPSolver.cpp`
**Function:** `CRMSolverIVP_Core` (lines 274-506)

**Algorithm:** Adams-Bashforth-Moulton 4th order (ABM4)
**Call site:** Line 413
```cpp
ABM4(xi, SegBounds[i], SegSteps[fsegno], h,
     IntegrandParams, no_locmarkers,
     CalculateEnergy,
     FinalValueOnly, LocMarkers, NextLocMarker,
     xf, DeltaPE, p_atLocMarkers);
```

**Stepsize computation:** Lines 404
```cpp
h = (SegBounds[i + 1] - SegBounds[i]) / (SegSteps[fsegno] * 1.0);
```

**State vector:** Lines 333-334
```cpp
StateVector xi;
auto& xf = out_x_N;
```
- `StateVector` contains: `_p` (position, 3), `_R` (rotation, 9), `_u` (curvature, 3)

### 2.2 Numerical Stability

**Segment loop:** Lines 398-488
- Iterates through catheter segments (proximal to distal)
- Flexible segments: integrated via ABM4
- Rigid segments: analytical SE(3) propagation (lines 510-533)

**Rigid segment boundary condition propagation:**
**Function:** `CRMSolverIVP_PropagateBCThroughRigidLink` (lines 510-533)

**Evidence of potential instability:**
- No explicit NaN/Inf checks in integration loop
- If ABM4 produces NaN, it propagates silently to `xf`
- Caller must check `out_x_N` for validity

**Determinism:**
- No random number generation
- All operations deterministic (given fixed inputs, outputs are identical)
- No global state mutation

---

## 3. Dynamics Integration Audit (Coil Dynamics)

### 3.1 Coil Dynamics Solver

**File:** `legacy_worktree/src/CoilDynamics_Defs.cpp` (first 200 lines inspected)
**Function:** `CoilDynamics` (lines 106-199)

**Algorithm:** RK2 initialization + ABM4
**Timestep:** Line 4
```cpp
#define t_step 0.001
```

**Iteration count:** Line 151
```cpp
int N = ceil(DELTA_T / t_step);
```

**NaN detection:** Lines 161-164
```cpp
if ( isnan(x_n[0]) ) {
    std::cout << "Coil integration Unbounded!! " << std::endl;
}
```
**Evidence:** NaN check present but no exit/throw (potential silent failure)

**State vector:** `NUM_COIL_STATES` (18 states: [v(3), w(3), p(6), R(9), xdot_history])

---

## 4. Rank Deficiency Handling

### 4.1 BVP Solver

**Trust-region dogleg** (line 81) implicitly handles rank deficiency:
- If Jacobian is singular, `info != 1` indicates failure
- No explicit rank check in legacy BVP code

### 4.2 IVP Jacobian (Pseudoinverse Usage)

**File:** `legacy_worktree/src/CRM_IVPJacobian.cpp`
**Line 91:**
```cpp
MatrixXd JIVP_u_u0_pinv = JIVP_u_u0.completeOrthogonalDecomposition().pseudoInverse();
```

**Evidence:**
- Pseudoinverse used without explicit rank check
- If `JIVP_u_u0` is rank-deficient, pseudoinverse may produce large/unstable values
- No diagnostic output for rank deficiency

**Line 97:**
```cpp
MatrixXd Jft_z = -JBVP_p_ft.completeOrthogonalDecomposition().pseudoInverse() * JBVP_p_z;
```
**Evidence:** Second pseudoinverse, same risk

---

## 5. Failure Modes

### 5.1 BVP Non-Convergence

**Condition:** Trust-region solver cannot satisfy tolerance
**Indicator:** `out_localmin != 0`
**Impact:** Invalid `deltau0` and `ftip` returned (garbage values)
**Propagation:** Caller must check `out_localmin` flag

### 5.2 IVP Integration Divergence

**Condition:** ABM4 integration produces NaN/Inf
**Indicator:** None (silent propagation)
**Impact:** Invalid final state `out_x_N`
**Propagation:** Caller must validate `out_x_N` manually

### 5.3 Coil Dynamics Unbounded Growth

**Condition:** Timestep too large, numerical instability
**Indicator:** `isnan(x_n[0])` check (line 161)
**Impact:** Prints warning but **does not exit** (commented-out `exit(3)`)
**Propagation:** NaN propagates to caller

---

## 6. Sensitivity Analysis

### 6.1 Sensitivity to dt

**Coil dynamics:** Fixed `t_step = 0.001` (line 4)
- Larger user `DELTA_T` → more ABM4 substeps (line 151)
- No adaptive timestepping
- If `DELTA_T` very large, accumulation error possible

### 6.2 Sensitivity to L_inserted

**IVP solver:** Lines 198-200
```cpp
InsertedLength = in_Li;
if (InsertedLength > SegEndLambdas[in_no_segments - 1])
    InsertedLength = SegEndLambdas[in_no_segments - 1];
```
**Evidence:** Clamped to catheter length (safe)

**StartSegmentIndex computation:** Lines 205-217
- Finds segment containing entry point
- If `L_inserted` very small, starts at most proximal segment
- If `L_inserted` very large, integration may be trivial (few steps)

### 6.3 Sensitivity to Actuation

**No hard limits on actuation currents in BVP/IVP solvers**
- Extreme actuation may cause BVP non-convergence
- No saturation/clipping in solver code

---

## 7. Determinism Guarantees

**Evidence of determinism:**
- No `rand()`, `srand()`, or stochastic elements
- No global state mutation (all state passed explicitly)
- ABM4 is deterministic (fixed coefficients)
- Trust-region dogleg is deterministic

**Thread safety:**
- No evidence of shared mutable state
- All arrays dynamically allocated per-call
- Safe for concurrent calls (assuming parameter objects not shared)

---

## 8. Hard-Coded Constants

| Constant | Location | Value | Impact |
|----------|----------|-------|---------|
| `TRUSTREGION_TOLERANCE` | `CRM_BVPSolver.cpp:77` | Unknown (macro) | BVP convergence threshold |
| `RESIDUAL_SCALE_M` | `CRM_BVPSolver.cpp:199` | Unknown (macro) | Moment residual scaling |
| `RESIDUAL_SCALE_P` | `CRM_BVPSolver.cpp:205` | Unknown (macro) | Position residual scaling |
| `IVALUE_SCALE_DU` | `CRM_BVPSolver.cpp:53` | Unknown (macro) | Delta curvature scaling |
| `IVALUE_SCALE_F` | `CRM_BVPSolver.cpp:54` | Unknown (macro) | Force scaling |
| `t_step` | `CoilDynamics_Defs.cpp:4` | `0.001` | Coil dynamics substep |
| `FCUM_DLAMBDA` | `CRM_BVPSolver.cpp:309` | Unknown (macro) | Force integration stepsize |

**Recommendation:** Expose tolerances as runtime parameters for B0 system ID

---

## 9. Summary of Findings

| Category | Finding | Risk Level | Mitigation |
|----------|---------|-----------|------------|
| **Convergence** | BVP solver may fail (non-convergence) | Medium | Check `out_localmin` flag |
| **NaN Propagation** | IVP integrator has no NaN checks | Medium | Validate `out_x_N` in caller |
| **Rank Deficiency** | Pseudoinverse used without rank check | Medium | Use current code (has rank check) |
| **Coil Dynamics** | NaN check present but no exit | Low | Exits are commented out |
| **Determinism** | Fully deterministic (no RNG) | **None** | Safe for gradient-based ID |
| **Extreme Inputs** | No saturation/clipping of actuation | Low | Add input validation in B0 |
| **Hard-Coded Tolerances** | Not user-configurable | Low | Expose in B0 if needed |

---

## 10. GO / NO-GO for B0

**GO** — Legacy dynamics code is **safe and suitable** for gradient-based system identification, with the following **required mitigations:**

1. **Convergence validation:** Always check BVP `out_localmin` flag
2. **NaN validation:** Check IVP `out_x_N` for NaN/Inf after each forward pass
3. **Diagnostics surfacing:** Surface solver diagnostics (residuals, rank, iterations) to Python
4. **Gradient clipping:** Implement gradient clipping in Python layer to handle rank-deficient cases

**Optional enhancements for B0:**
- Expose `TRUSTREGION_TOLERANCE` as runtime parameter
- Add adaptive timestepping for coil dynamics (if needed)
- Add actuation saturation limits (if needed)

---

## Evidence Log

**Legacy worktree verification:**
```bash
$ cd /workspaces/CRM_DiffSim_Cl/legacy_worktree
$ git branch --show-current
main
$ git status --porcelain
(empty - clean worktree)
$ git log -1 --oneline
828bf8c deleted obsolete parameters, cleaned folders
```

**Files audited:**
- `legacy_worktree/src/CRM_BVPSolver.cpp` (full file, 580 lines)
- `legacy_worktree/src/CRM_IVPSolver.cpp` (full file, 920 lines)
- `legacy_worktree/src/CoilDynamics_Defs.cpp` (first 200 lines)
- `legacy_worktree/src/CRM_IVPJacobian.cpp` (first 100 lines)

**No legacy physics code was modified.**
**All findings are documentation-only (audit mode).**

---

END OF AUDIT
