# PRE-B0 LEGACY C++ DYNAMICS AUDIT

**Audit Date:** 2026-01-04
**Auditor:** Claude Code (Sonnet 4.5)
**Scope:** Legacy C++ physics/solver code (Read-Only Inspection)
**Legacy Reference:** `main` branch at `/workspaces/CRM_DiffSim_Cl/legacy_worktree` (commit 828bf8c)
**Authority:** Read-only inspection via git worktree

---

## EXECUTIVE SUMMARY

The legacy C++ dynamics code in `legacy_worktree/src/` implements a hybrid BVP/IVP solver system for catheter dynamics. The code is **deterministic by design** with no random elements, but contains **critical safety gaps** in error handling that could cause crashes during gradient-based system identification.

**Key Findings:**
- ✅ Deterministic (no RNG)
- ✅ Mathematically sound (ABM4, trust-region, SE3-aware)
- ❌ **NaN detection disabled** (commented out at CoilDynamics_Defs.cpp:161)
- ❌ **Singular matrix handling incomplete** (prints warning but continues)
- ⚠️  **No explicit rank deficiency checks** in BVP solver
- ⚠️  Silent failure propagation via `localmin` status codes

---

## 1. SOLVER ARCHITECTURE

### 1.1 Main Physics/Solver Entrypoints

#### Static Equilibrium (BVP Solvers)

**CRMShootingMethodBVP**
Location: `legacy_worktree/src/CRM_BVPSolver.cpp:13-109`

- Main boundary value problem solver for static equilibrium
- Wraps trust-region nonlinear solver from `numerical/minpack.hpp`
- Returns `out_localmin`:
  - `0` = success
  - `>0` = failure (trust-region method returned `info != 1`)
- Conversion logic (line 86): `localmin = (info == 1) ? 0 : (info - 1);`

**CRM_ForwardKinematics**
Location: `legacy_worktree/src/CRM_ForwardKinematics.cpp`

- High-level API for catheter shape calculation
- Supports FREE_TIP and FIXED_TIP contact modes
- Uses incremental actuation (`#define INCREMENTALLY_APPLY_CURRENTS`)
- Preserves last good solution when `localmin == 0` (CRM_CatheterClass.cpp:44, 77)

#### Dynamics (IVP Solvers)

**DynamicsBVP**
Location: `legacy_worktree/src/CoilDynamics_Defs.cpp:1066`

- Dynamics boundary value problem for coil actuators
- Solves for coil interface moments `m_L[N][3]` and forces `n_L[N][3]`
- Returns `localmin` status code

**DYNSolverIVP**
Location: `legacy_worktree/src/CoilDynamics_Defs.cpp:1195`

- Time-stepping dynamics integrator
- Forward propagation through catheter segments
- Uses ABM4 numerical integration (see §1.3)

#### Shared IVP Core

**CRMSolverIVP_Core**
Location: `legacy_worktree/src/CRM_IVPSolver.cpp:274-506`

- Core integrator for shooting method
- Iterates through flexible/rigid segments
- Handles boundary conditions between segments

### 1.2 Convergence Criteria and Tolerances

**Trust Region Method** (Primary Nonlinear Solver)

Location: `legacy_worktree/numerical/minpack.hpp`

**Tolerance:** `TRUSTREGION_TOLERANCE = 1e-5`
Evidence: `legacy_worktree/src/CRM.hpp:32`

**Return codes:**
- `info == 1` → Success
- `info == 2` → Both actual and predicted relative reductions in the sum of squares are at most `ftol`
- `info == 3` → Relative error between two consecutive iterates is at most `xtol`
- `info == 4` → Conditions for `info = 2` and `info = 3` both hold
- `info == 5` → Number of calls to `fcn` reached `maxfev`
- `info == 6` → `ftol` too small
- `info == 7` → `xtol` too small
- `info == 8` → `gtol` too small

**Residual Thresholds**

Static solver residual scaling:
- `RESIDUAL_SCALE_M = 1.0` (moment residuals) — `legacy_worktree/src/CRM.hpp:24`
- `RESIDUAL_SCALE_P = 1.0` (position residuals) — `legacy_worktree/src/CRM.hpp:25`

Dynamics solver residual scaling:
- `RESIDUAL_SCALE_P = 1` (position) — `legacy_worktree/src/CRMDYN.hpp:41`
- `RESIDUAL_SCALE_R = 1` (rotation) — `legacy_worktree/src/CRMDYN.hpp:42`

### 1.3 Numerical Integration Methods

**ABM4 (Adams-Bashforth-Moulton 4th Order)**

Location: `legacy_worktree/src/CRM_IVP_NumericalIntegrationTemplates.hpp`

**Predictor coefficients:** `[55/24, -59/24, 37/24, -9/24]`
**Corrector coefficients:** `[9/24, 19/24, -5/24, 1/24]`

**Initialization:** RK2 (Runge-Kutta 2nd order) for first 3 steps

**Analytical SE3 Step**
Enabled by: `#define ANALYTICAL_SE3_STEP`

- Uses Rodrigues' formula for SO(3) exponential maps
- Avoids numerical drift in rotation matrices
- Evidence: Rotation matrix updates use exact matrix exponentials

**Numerical Jacobian Finite Differences**

Stepsize range: `1e-5` to `1e-2`
Evidence: `legacy_worktree/src/CRM_IVPJacobian.cpp` (various locations)

### 1.4 Stability Checks

**NaN Detection (DISABLED)**

Location: `legacy_worktree/src/CoilDynamics_Defs.cpp:161`

```cpp
if (!finite(v_L_pre[j][k]) || !finite(w_L_pre[j][k])) {
    printf("Coil integration Unbounded!!\n");
    // exit(3);  // <-- COMMENTED OUT
}
```

**CRITICAL ISSUE:** Exit call is commented out. NaN propagation will continue silently.

**Impact:**
- NaNs in coil velocities/angular velocities will propagate through subsequent timesteps
- May cause gradient explosion in backward pass
- No recovery mechanism

---

## 2. RANK DEFICIENCY HANDLING

### 2.1 Singular Matrix Detection (INCOMPLETE)

Location: `legacy_worktree/src/CoilDynamics_Defs.cpp:1579-1587`

```cpp
bool singular = sy < 1e-6;
if (singular) {
    printf("Singular Rotation matrix............\n");
}
// NO RECOVERY MECHANISM - computation continues despite singularity
```

**Issue:** Detection exists but no corrective action taken. Computation continues with potentially invalid rotation matrix.

### 2.2 LU Rank Checking (NOT PRESENT IN LEGACY)

**Finding:** The legacy code does NOT explicitly check for:
- Matrix rank deficiency
- Conditioning numbers (via condition number estimates)
- Jacobian singularities

**Implicit Handling:**
- Trust region method's Powell dogleg algorithm has some robustness to ill-conditioned Jacobians
- Numerical Jacobian finite differences can mask singularities if stepsize is too large

**Evidence:** No calls to `rank()`, `cond()`, or similar condition number checks in legacy C++ code.

---

## 3. FAILURE MODES AND PROPAGATION

### 3.1 BVP Solver Failures

**Exit Points for Unsupported Configurations:**

1. `legacy_worktree/src/CRM_BVPSolver.cpp:91`
   Undefined solver method

2. `legacy_worktree/src/CRM_BVPSolver.cpp:254`
   Unimplemented analytical Jacobian for FIXED_TIP mode

3. `legacy_worktree/src/CoilDynamics_Defs.cpp:1159`
   Undefined dynamics solver type

**Status Code Propagation:**

`localmin` flag returned to caller:
- `localmin == 0` → Success
- `localmin > 0` → Failure (specific error code from trust-region solver)

**CRITICAL GAP:** `localmin` is NOT checked at all intermediate call levels. Failures can propagate silently upward until eventually checked (or not) at top-level caller.

### 3.2 Error Handling Strategy

**NO EXCEPTIONS in numerical code:**
- All commented-out `exit()` calls have been disabled
- Errors return via status codes, not exceptions

**Runtime errors ONLY for I/O failures:**
- `legacy_worktree/src/CRM_SupportFunctions.cpp:34, 47, 55, 58, 157`
  File open/read failures throw runtime errors

**Silent Failures:**
- `localmin` flag propagated upward
- NOT checked at intermediate levels
- Incremental FK preserves last good solution on failure (CRM_CatheterClass.cpp:44, 77)

**No Timeout Mechanism:**
- No maximum iteration bounds exposed to caller
- Trust-region solver has internal `maxfev` but not configurable from outside

**No Partial Result Recovery:**
- If solver fails, caller gets `localmin > 0` but no information about how close the solver got to convergence

---

## 4. DETERMINISM ANALYSIS

### 4.1 Random Number Generation (NONE FOUND)

**Search performed:**
```bash
$ grep -r "random\|seed\|srand\|rand()" legacy_worktree/src
(no results)
```

**Conclusion:** No stochastic elements in legacy physics code.

### 4.2 Deterministic Components

- All numerical integration is deterministic (ABM4, RK2)
- Trust-region solver has no randomization (Powell dogleg is deterministic)
- Initial guesses either fixed or user-provided
- No Monte Carlo methods
- No stochastic ODE solvers

### 4.3 Floating Point Assumptions

**Thresholds assume IEEE 754 behavior:**
- Epsilon value for SE3: `EPS = 1e-12`
- Small magnitude check: `umagsq < EPS` for zero velocity detection
- Rotation matrix singularity: `sy < 1e-6`

**No Explicit Rounding Mode Control:**
- Code assumes default IEEE 754 rounding (round-to-nearest)
- No calls to `fesetround()` or similar FP environment controls

**Potential Non-Determinism Sources:**
1. Uninitialized memory (if constructors fail silently)
2. Compiler optimization differences (e.g., `-ffast-math` changes FP semantics)
3. Platform-dependent floating point modes if not explicitly set by environment

**Recommendation:** For deterministic gradient-based system ID, ensure:
- Same compiler flags across runs
- No `-ffast-math` or similar aggressive FP optimizations
- Explicit FP rounding mode setting if reproducibility is critical

---

## 5. SENSITIVITY TO EXTREME INPUTS

### 5.1 Extreme Timestep (`dt`)

**Large `dt`:** ABM4 has stability region, but not explicitly documented. Expect instability for `dt > critical_value` depending on system stiffness.

**Small `dt`:** Roundoff error accumulation over many timesteps. Trust-region solver may struggle with very small residuals.

**Evidence:** No dt range checks in legacy code. User responsible for choosing appropriate timestep.

### 5.2 Extreme Insertion Depth (`L_inserted`)

**Large `L_inserted`:** More flexible segments, larger state dimension, potentially more BVP unknowns.

**Small `L_inserted`:** Fewer segments, simpler problem, but may hit rigid body limit.

**Evidence:** No bounds checking on `L_inserted` in legacy code.

### 5.3 Zero and High Actuation Currents

**Zero current:** May hit singular Jacobian if equilibrium shape becomes straight (loss of curvature DOF).

**High current:** Large magnetic forces/torques. May exceed small-angle approximations (if any). No saturation or clamping detected in legacy code.

**Evidence:** No actuation bounds in legacy physics (only limited by catheter parameters).

---

## 6. KEY FILES SUMMARY

### 6.1 Core Physics

1. `legacy_worktree/src/CRM_IVPSolver.cpp` — Static IVP integration
2. `legacy_worktree/src/CRM_BVPSolver.cpp` — Static BVP solver
3. `legacy_worktree/src/CoilDynamics_Defs.cpp` — Dynamics equations (BVP + IVP)
4. `legacy_worktree/src/CRM_ForwardKinematics.cpp` — Forward kinematics API

### 6.2 Numerical Methods

5. `legacy_worktree/src/CRM_IVP_NumericalIntegrationTemplates.hpp` — ABM4 templates
6. `legacy_worktree/src/CRMDYN_Numerical_Integration.hpp` — Dynamics integrators
7. `legacy_worktree/numerical/minpack.hpp` — Trust-region solver (Powell dogleg)

### 6.3 Configuration

8. `legacy_worktree/src/CRM.hpp` — Static solver constants (`TRUSTREGION_TOLERANCE`, residual scales)
9. `legacy_worktree/src/CRMDYN.hpp` — Dynamics solver constants
10. `legacy_worktree/src/CRM_BVPIVP_APIDeclarations.hpp` — Shared declarations

---

## 7. CRITICAL FINDINGS SUMMARY

| Finding | Severity | Evidence |
|---------|----------|----------|
| NaN detection commented out | **CRITICAL** | `CoilDynamics_Defs.cpp:161` |
| Singular matrix continues execution | **HIGH** | `CoilDynamics_Defs.cpp:1579` |
| No explicit rank deficiency handling | **MEDIUM** | No rank checks in BVP solver |
| Silent failure propagation | **MEDIUM** | `localmin` not checked at intermediate levels |
| No timeout mechanism | **LOW** | Trust-region `maxfev` internal only |
| Hard-coded tolerance (1e-5) | **LOW** | `CRM.hpp:32` |

---

## 8. SAFETY RECOMMENDATIONS FOR B0

### 8.1 Before System Identification Begins

1. **Re-enable NaN detection** with proper error propagation (not `exit()`, but status return)
2. **Add singular matrix recovery** (return failure status instead of continuing)
3. **Surface convergence diagnostics** to Python layer (iterations, residual norms, rank)
4. **Add input validation** in Python wrapper (dt range, L_inserted bounds, actuation limits)

### 8.2 During Gradient-Based Optimization

1. **Monitor gradient norms** for explosion/vanishing
2. **Check BVP convergence** at every forward step
3. **Validate finite values** in both forward state and backward gradients
4. **Clip gradients** if necessary (document threshold choice)

---

## AUDIT STATUS

**Scope:** Read-only inspection of legacy C++ physics/solver code
**Modifications Made:** NONE (audit-only)
**Legacy Code Modified:** NO
**Git Worktree Used:** YES (`legacy_worktree` on `main` branch)

**Completion:** ✅ ALL LEGACY DYNAMICS AUDITED

**Next Steps:**
- Proceed to Binding Layer Audit
- Then Python Control Layer Audit
- Then Cross-Layer Risk Analysis

---

**END OF LEGACY C++ DYNAMICS AUDIT**
