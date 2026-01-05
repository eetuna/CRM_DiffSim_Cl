# Sprint S14 — Step 4 BLOCKED: IVP Jacobians via Forward-Mode AD

**Date:** 2026-01-05
**Scope:** FULLSTATE only (18·N + 15)
**Objective:** Replace misuse of CRMSolverIVPJacobian with exact DYNSolverIVP Jacobians computed via forward-mode AD (Dual) at the converged solution.

---

## Status: BLOCKED

**Step 4 cannot be completed without massive code duplication.**

---

## Blocker Analysis

### Required Implementation

Per `docs/audits/CODEX_EXECUTION_READY_PLAN.md`, Step 4 requires:

1. **Templated IVP Solver:**
   - Implement `DYNSolverIVP_T<T>` mirroring `DYNSolverIVP` but templated for `T` inputs
   - All state, force, torque, and rotation variables must be type `T`
   - No `.val` extraction anywhere

2. **Jacobian Extraction:**
   - For each input component in `y = [mL; nL]` and `x = [x_coil; xf]`:
     - Seed exactly ONE Dual derivative at a time
     - Run `DYNSolverIVP_T<Dual>` at the converged solution
     - Extract Jacobians: `J_xf_y`, `J_xf_x`, `J_xcoil_y`

### What DYNSolverIVP Does

`DYNSolverIVP` (src/CoilDynamics_Defs.cpp:1195-1257) is a thin wrapper around `CRMIVP_DYN`:

```cpp
void DYNSolverIVP(CRMShootingMethodParams& in_Params,
                  const double in_u0[3],
                  const double in_mL[NUM_ACT_SET][3],
                  const double in_nL[NUM_ACT_SET][3],
                  const double in_tau[NUM_ACT_SET][3],
                  const double in_ftip[3],
                  bool in_FinalValueOnly,
                  double out_x_N[NUM_STATES],
                  double out_coil_state[NUM_ACT_SET][NUM_COIL_STATES],
                  double out_p_atLocMarkers[][3])
```

It:
1. Prepares parameters via `CRMDYNSolverIVP_Prep`
2. Calls `CRMIVP_DYN` (src/CoilDynamics_Defs.cpp:907-1064)
3. Returns final tip state `x_N = [p, R, u]` and coil states

### What CRMIVP_DYN Does

`CRMIVP_DYN` (src/CoilDynamics_Defs.cpp:907-1064) performs **FORWARD integration**:

```cpp
for (int SegmentIndex = 0; SegmentIndex < NUM_SEGMENTS; ++SegmentIndex) {
    if (SegmentIndex % 2 == 0) {
        // Flexible segment
        CRMFlexForward_pass(SegmentIndex, p0, R0, CoreParams, u_0, n_f,
                            u_new, p_new, R_new, p_atLocMarkers);
    } else {
        // Rigid actuator segment
        CoilDynamics(x_coil[actno], net_nL, CoreParams.g, actMass[actno],
                     actInertia[actno], CoreParams.damping[actno],
                     CoreParams.DELTA_T, CoreParams.B0, muhat[actno],
                     net_mL, update_coil_state, out_xdot);
        // ... update states ...
    }
}
```

### Required Templating Cascade

To create `DYNSolverIVP_T<T>`, we must template:

#### 1. CRMIVP_DYN → CRMIVP_DYN_T<T>
   - Template all state variables: `p0`, `R0`, `u_0`, `x_coil`
   - Template inputs: `mL`, `nL`, `tau`, `ftip`
   - Template outputs: `u_new`, `p_new`, `R_new`, `out_coil_state`

#### 2. CRMFlexForward_pass → CRMFlexForward_pass_T<T>
   - **DOES NOT EXIST** (only backward version `CRMFlexible_IVP_Back_T` exists)
   - Requires full forward integration of flexible segments
   - Must template:
     - `CRMIntegrand_dyn` → `CRMIntegrand_dyn_T<T>` ✅ (already exists in Step 3)
     - ABM4_dyn forward integration (Step 3 only has backward)
     - State vector updates with templated R, p, u
     - Marker location tracking (if needed)

#### 3. CoilDynamics → CoilDynamics_T<T>
   - ✅ **Already exists** (src/CoilDynamics_Defs_Templates.hpp:470+)
   - Used in Step 3 for BVP residual evaluation

### Current Templated Infrastructure

**What EXISTS (from Steps 1-3):**
- ✅ `Dual` type with full arithmetic (src/CRM_BVPJacobian.hpp:13-147)
- ✅ `CoilDynamics_T<T>` (src/CoilDynamics_Defs_Templates.hpp)
- ✅ `CRMFlexible_IVP_Back_T<T>` (src/CoilDynamics_Defs_Templates2.hpp:97+)
- ✅ `CRMIntegrand_dyn_T<T>` (via Step 3)
- ✅ `DYNNLEquation_T<T>` (BVP residual, Step 3)
- ✅ Templated Rodrigues, SE(3) updates, RK2/ABM4 coil dynamics

**What is MISSING:**
- ❌ `CRMIVP_DYN_T<T>` (forward IVP integrator)
- ❌ `CRMFlexForward_pass_T<T>` (forward flexible segment integration)
- ❌ Templated forward ABM4 path for flexible segments

### Code Duplication Estimate

Creating the missing infrastructure would require:

1. **CRMFlexForward_pass_T<T>** (~300 lines)
   - Mirror `CRMFlexForward_pass` (src/CoilDynamics_Defs.cpp:809-905)
   - Replace double with T throughout
   - Call `CRMIntegrand_dyn_T<T>` instead of `CRMIntegrand_dyn`
   - Implement forward ABM4 loop with templated state vectors
   - Handle rotation updates with `Rodrigues_T<T>`

2. **CRMIVP_DYN_T<T>** (~200 lines)
   - Mirror `CRMIVP_DYN` (src/CoilDynamics_Defs.cpp:907-1064)
   - Template all state arrays
   - Call `CRMFlexForward_pass_T<T>` and `CoilDynamics_T<T>`
   - Template segment loop logic

3. **DYNSolverIVP_T<T>** (~100 lines)
   - Wrapper around `CRMIVP_DYN_T<T>`
   - Templated parameter preparation

**Total: ~600 lines of duplicated logic**

This violates the STOP condition:
> `DYNSolverIVP_T<T>` cannot be implemented without duplicating logic

---

## Alternative Approaches Considered

### Option 1: Finite Differences for IVP Jacobians
❌ **Prohibited** by plan:
> **Prohibited:** FD anywhere (incl. tests), torch.autograd, heuristics, placeholders.

### Option 2: Keep CRMSolverIVPJacobian, Fix Variable Mapping
❌ **Incorrect** — `CRMSolverIVPJacobian` computes:
- `∂xf/∂u0` (tip state w.r.t. base curvature)
- `∂xf/∂ftip` (tip state w.r.t. tip force)
- `∂xf/∂z` (tip state w.r.t. actuator currents)

We need:
- `∂xf/∂(mL,nL)` (tip state w.r.t. BVP unknowns)
- `∂xf/∂x_coil` (tip state w.r.t. coil states)

These are **fundamentally different Jacobians**.

### Option 3: Refactor to Avoid Duplication
This would require:
1. Extract common integrator logic into a policy-based template
2. Parameterize forward vs backward integration
3. Refactor all existing callers to use new API

**Estimated effort:** 2-3 weeks, high regression risk.

❌ **Out of scope** for Step 4.

---

## Recommendation

**Step 4 should be re-scoped** to one of:

### Recommended Path: Implement Infrastructure Incrementally

1. **Sub-step 4a:** Create `CRMFlexForward_pass_T<T>` (forward flexible segment integration)
   - Estimated: 300 lines, 4-6 hours
   - Blockers: None (all dependencies exist)

2. **Sub-step 4b:** Create `CRMIVP_DYN_T<T>` (forward IVP solver)
   - Estimated: 200 lines, 2-3 hours
   - Dependency: 4a

3. **Sub-step 4c:** Create `DYNSolverIVP_JacobiansFullstate` wrapper
   - Estimated: 150 lines, 2-3 hours
   - Dependency: 4b
   - Extract Jacobians via Dual seeding

4. **Sub-step 4d:** Replace `CRMSolverIVPJacobian` usage in backward paths
   - Estimated: 100 lines, 1-2 hours
   - Dependency: 4c

**Total estimated effort:** 750 lines, 10-15 hours

### Alternative: Defer Step 4, Proceed with Steps 5-7 Using Existing Jacobians

- Use `CRMSolverIVPJacobian` output **as-is** for adjoint assembly
- Document the variable mismatch as a known issue
- Prioritize Steps 5-7 (adjoint assembly, control Jacobians, linearization)
- Revisit Step 4 after forward physics is validated

---

## Files Investigated

### Read (full or partial):
- `docs/audits/CODEX_EXECUTION_READY_PLAN.md` (Step 4 spec)
- `docs/audits/SPRINT_S14_STEP3_JYX_COMPLETION.md` (existing infrastructure)
- `src/CoilDynamics_Defs.cpp` (DYNSolverIVP, CRMIVP_DYN, CRMFlexForward_pass)
- `src/CoilDynamics_Defs_Templates.hpp` (CoilDynamics_T, Rodrigues_T)
- `src/CoilDynamics_Defs_Templates2.hpp` (CRMFlexible_IVP_Back_T, DYNNLEquation_XT)
- `src/CRM_BVPJacobian.hpp` (Dual type definition)
- `src/CRM_IVPJacobian.cpp` (CRMSolverIVPJacobian implementation)
- `src/CRM_TrueLegacyDynamics.cpp` (backward pass usage of CRMSolverIVPJacobian)

### Grepped:
- DYNSolverIVP usages: 63 files
- CRMSolverIVPJacobian usages: 28 files (3 in CRM_TrueLegacyDynamics.cpp)
- Templated infrastructure: CRMFlexible_IVP_Back_T exists, CRMFlexForward_pass_T does NOT

---

## Verification: No Changes Made

```bash
git status
```

**Output:**
```
M build/CRMTest
M build/libCRMCPPLib.a
M src/CRMDYN_Numerical_Integration.hpp
M src/CRM_BVPJacobian.cpp
M src/CRM_StateVector_Definitions.hpp
M src/CoilDynamics_Defs_Templates.hpp
M src/CoilDynamics_Defs_Templates2.hpp
?? docs/audits/SPRINT_S14_CLAUDE_CODEX_PLAN_IMPLEMENTATION.md
?? docs/audits/SPRINT_S14_STEP3_JYX_COMPLETION.md
?? legacy_worktree/
```

✅ **No new modifications** — Only investigation performed.

---

## Conclusion

**Step 4 is BLOCKED pending architectural decision.**

The requirement to create `DYNSolverIVP_T<T>` without code duplication **cannot be met** with the current codebase structure. The missing `CRMFlexForward_pass_T<T>` infrastructure represents a ~300-line implementation that would duplicate the existing forward integration logic.

**Recommended action:**
1. Break Step 4 into sub-steps (4a-4d) to implement incrementally, OR
2. Defer Step 4 and proceed with Steps 5-7 using existing IVP Jacobians

**STOP:** Awaiting clarification on scope before proceeding.

---

## Command Log

```bash
# Investigation commands
rg "DYNSolverIVP" --files-with-matches
rg "CRMSolverIVPJacobian" --files-with-matches
rg "CRMFlexForward_pass_T" /workspaces/CRM_DiffSim_Cl/src/
rg "^void DYNSolverIVP|^template.*DYNSolverIVP" --line-number
rg "^void CRMIVP_DYN\(|^template.*CRMIVP_DYN" --line-number
rg "CRMFlexForward_pass" --line-number src/CoilDynamics_Defs.cpp
```

All commands executed successfully; no compilation attempted.

---

**END OF REPORT**
