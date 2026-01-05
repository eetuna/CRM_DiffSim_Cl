# PRE-B0 CORRECTION NOTE

**Date:** 2026-01-03
**Corrector:** Claude (Sonnet 4.5)
**Authority:** `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md`

---

## EXECUTIVE SUMMARY

The previously generated Pre-B0 audit documents **incorrectly identified the state contract** by conflating:
- **6D reduced state** (u_0, v_0) with "legacy state"
- **TRUE legacy state** (18·N + 15) as if it were the same as the 6D path

**This correction supersedes:**
1. `docs/audits/PRE_B0_PYTHON_CONTROL_AUDIT.md`
2. `docs/audits/PRE_B0_COMPLETION_REPORT.md`
3. `docs/audits/PRE_B0_RISKS_AND_MITIGATIONS.md`

---

## WHAT WAS WRONG

### Incorrect Claim 1: "6D is legacy state"

**Incorrect statement (line 40, PRE_B0_PYTHON_CONTROL_AUDIT.md):**
```
STATE_DIM_LEGACY = 6  # Total state dimension
```

**Correct statement:**
```
STATE_DIM_LEGACY = 6  # REDUCED state dimension (NOT TRUE legacy)
```

**Evidence:**
- TRUE legacy state is **18·N + 15** per `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md:142`
- 6D state is a **reduced projection** created for CP2 milestone
- The authoritative state contract at `legacy_worktree/main/CRMDYN_test.cpp:291-311` persists **18·N + 15** scalars

### Incorrect Claim 2: "Legacy contract uses 6D"

**Incorrect statement (line 53, PRE_B0_PYTHON_CONTROL_AUDIT.md):**
```
**Evidence:** State is exactly 6D (u_0 + v_0), matches TRUE legacy hybrid reduction
```

**Correct statement:**
```
**Evidence:** State is exactly 6D (u_0 + v_0), which is a REDUCED representation.
TRUE legacy hybrid state is 18·N + 15 per the authoritative contract.
```

### Incorrect Claim 3: GO/NO-GO decision based on 6D path

**Incorrect statement (line 437, PRE_B0_COMPLETION_REPORT.md):**
```
**GO** — Python control layer is **safe and suitable** for B0 system identification
```

**Correct statement:**
```
**GO** — Python control layer is safe IF using reduced 6D path.
**NO-GO** — If B0 requires TRUE legacy (18·N+15) for parity with main branch system ID.
```

---

## WHAT CHANGED

### 1. State Contract Classification

**Corrected classification:**

| Path | State Dim | Label | Used for B0? | Evidence |
|------|-----------|-------|--------------|----------|
| `true_legacy_step.py` | **18·N + 15** | **TRUE legacy** | ❌ NO (too high-dim) | `python/control/true_legacy_step.py:36` |
| `step_legacy_contract.py` | **6D** | **REDUCED (non-legacy)** | ✅ YES (current path) | `python/control/legacy_state.py:19` |
| `step_hybrid_legacy_contract.py` | **9D** | **REDUCED + observable** | ❌ NO (prototype) | `python/control/hybrid_state_contract.py:36` |

**Authoritative reference:**
- `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md:142`
  > **Total**: `18N + 15` core state variables.

### 2. Runtime Entrypoint Classification

**TRUE legacy entrypoints (18·N+15):**

| Component | File:Line | Function | State Type |
|-----------|-----------|----------|------------|
| Python API | `python/control/true_legacy_step.py:23` | `true_legacy_step` | 18·N+15 |
| Autograd | `python/control/true_legacy_step_autograd.py:22` | `TrueLegacyStepFn` | 18·N+15 |
| Binding | `python/crm_bindings.cpp:951` | `py_true_legacy_step_forward` | 18·N+15 |
| C++ wrapper | `src/CRM_TrueLegacyDynamics.cpp:12` | `TrueLegacyDynamics_Forward` | 18·N+15 |
| Legacy core | `legacy_worktree/src/CoilDynamics_Defs.cpp:1066` | `DynamicsBVP` | 18·N+15 |
| Legacy core | `legacy_worktree/src/CoilDynamics_Defs.cpp:1195` | `DYNSolverIVP` | 18·N+15 |

**Reduced state entrypoints (6D - MISLABELED as "legacy"):**

| Component | File:Line | Function | State Type |
|-----------|-----------|----------|------------|
| Python API | `python/control/step_legacy_contract.py:114` | `step_legacy_contract` | 6D **REDUCED** |
| State class | `python/control/legacy_state.py:25` | `LegacyState` | 6D **REDUCED** |
| Binding | `python/crm_bindings.cpp:250` | `py_dynamics_forward` | 6D **REDUCED** |
| C++ wrapper | `src/CRM_DiffDynamics.cpp` | `DynamicsForward` | 6D **REDUCED** |

**Evidence from control layer:**

```bash
$ rg -n "A0.*TRUE legacy|A1.*Legacy 6D" python/control/__init__.py
7:# - A1: Legacy 6D state adapter (contract-exact)
25:# A1 + A2: Legacy 6D state (contract-exact) with implicit VJP
45:# A0: TRUE legacy state adapter (18*N+15, no reduction)
```

**Line 45 explicitly labels 18·N+15 as "TRUE legacy"**
**Line 7, 25 label 6D as "Legacy 6D" but this is MISLEADING — should say "REDUCED 6D"**

### 3. GO/NO-GO Decision Logic

**Old decision (INCORRECT):**
```
GO — Python control layer (6D) is safe for B0 system identification
```

**New decision (CORRECT):**
```
IF B0 system ID is intended to match TRUE legacy dynamics (DynamicsBVP → DYNSolverIVP):
  → NO-GO (current stack uses 6D REDUCED, not 18·N+15 TRUE legacy)

IF B0 system ID is acceptable with 6D reduced dynamics:
  → GO (current stack is safe for reduced 6D path)
```

**Critical question to resolve:**
> Does B0 require parity with the TRUE legacy stepping function (DynamicsBVP → DYNSolverIVP)
> used in the original MATLAB/C++ system ID, or is the reduced 6D path acceptable?

---

## CORRECTED EVIDENCE LOG

### Legacy Worktree Verification (UNCHANGED)

```bash
$ git worktree list
/workspaces/CRM_DiffSim_Cl                  c764edb [milestone-a-hybrid-vjp]
/workspaces/CRM_DiffSim_Cl/legacy_worktree  828bf8c [main]

$ cd legacy_worktree
$ git branch --show-current
main
$ git status --porcelain
(empty - clean)
$ git log -1 --oneline
828bf8c deleted obsolete parameters, cleaned folders
```

### State Dimension Evidence (CORRECTED)

**TRUE legacy state (18·N+15):**

```bash
$ rg -n "18\*N|18\*NUM_ACT|COIL_STATE_DIM.*18|TIP_STATE_DIM.*15" python/control
python/control/true_legacy_state_adapter.py:20:COIL_STATE_DIM = 18  # v[3], w[3], p[3], R[9]
python/control/true_legacy_state_adapter.py:21:TIP_STATE_DIM = 15   # p_tip[3], R_tip[9], u_tip[3]
python/control/true_legacy_state_adapter.py:32:        State dimension: 18·N + 15
python/control/true_legacy_state_adapter.py:34:    return COIL_STATE_DIM * n_act + TIP_STATE_DIM
python/control/true_legacy_state_adapter.py:42:            - Unbatched: [18*n_act + 15]
python/control/true_legacy_state_adapter.py:43:            - Batched: [B, 18*n_act + 15]
```

**Reduced state (6D - MISLABELED "legacy"):**

```bash
$ rg -n "STATE_DIM_LEGACY.*6|u_0.*v_0" python/control/legacy_state.py
python/control/legacy_state.py:7:State layout (6D):
python/control/legacy_state.py:8:  x[0:3] = u_0 (base curvature, 1/mm)
python/control/legacy_state.py:9:  x[3:6] = v_0 (base curvature velocity, 1/mm/s)
python/control/legacy_state.py:19:STATE_DIM_LEGACY = 6
python/control/legacy_state.py:34:        u_0: Base curvature (3,) in 1/mm
python/control/legacy_state.py:35:        v_0: Base curvature velocity (3,) in 1/mm/s
python/control/legacy_state.py:40:    u_0: np.ndarray  # (3,) float64
python/control/legacy_state.py:41:    v_0: np.ndarray  # (3,) float64
```

**Authoritative contract reference:**

```bash
$ grep -n "18N + 15\|18.*N.*15" docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md
142:**Total**: `18N + 15` core state variables.
299:1. **State dimension**: `18 * NUM_ACT_SET + 15` (excluding warm-start)
```

### Call Chain Evidence (DynamicsBVP → DYNSolverIVP)

**TRUE legacy call chain:**

```bash
$ rg -n "DynamicsBVP|DYNSolverIVP" python/control python/crm_bindings.cpp src/CRM_TrueLegacyDynamics.cpp
python/control/true_legacy_step.py:5:and the canonical stepping sequence: DynamicsBVP → DYNSolverIVP
python/control/true_legacy_step.py:34:    One-step forward using TRUE legacy dynamics (DynamicsBVP → DYNSolverIVP).
python/crm_bindings.cpp:925:    // A0: TRUE legacy stepping (DynamicsBVP → DYNSolverIVP)
python/crm_bindings.cpp:930:          "A0: TRUE legacy step (DynamicsBVP → DYNSolverIVP). Returns next state and observables.");
python/crm_bindings.cpp:1113:    DynamicsBVP(params, xf_data, mL_guess, nL_guess, ftip_guess,
python/crm_bindings.cpp:1122:    DYNSolverIVP(params, out_u0, out_mL, out_nL, out_tau, out_ftip,
src/CRM_TrueLegacyDynamics.cpp:12:// Forward pass: wrapper around existing DynamicsBVP → DYNSolverIVP
src/CRM_TrueLegacyDynamics.cpp:104:    // Call DynamicsBVP
src/CRM_TrueLegacyDynamics.cpp:105:    DynamicsBVP(shooting_params, xf, mL_guess_arr, nL_guess_arr, ftip_guess,
src/CRM_TrueLegacyDynamics.cpp:110:    // Call DYNSolverIVP
src/CRM_TrueLegacyDynamics.cpp:112:    DYNSolverIVP(shooting_params, out.u0, out.mL, out.nL, out.tau, out.ftip,
```

**Evidence:** TRUE legacy path directly calls `DynamicsBVP` and `DYNSolverIVP` from legacy C++

**Reduced 6D path does NOT use DynamicsBVP/DYNSolverIVP:**

```bash
$ rg -n "DynamicsBVP|DYNSolverIVP" src/CRM_DiffDynamics.cpp src/CRM_DiffEquilibrium.cpp
(no matches)
```

**Evidence:** Reduced 6D path uses new wrappers (`CRM_DiffDynamics`, `CRM_DiffEquilibrium`), NOT the TRUE legacy solvers

---

## CORRECTED FINDINGS

### Python Control Layer Classification

**TRUE legacy paths (18·N+15):**

| Module | Purpose | State Contract | Status |
|--------|---------|----------------|--------|
| `true_legacy_state_adapter.py` | State packing/unpacking | 18·N+15 | ✅ Correct |
| `true_legacy_step.py` | Forward pass | 18·N+15 | ✅ Correct |
| `true_legacy_step_autograd.py` | PyTorch autograd | 18·N+15 | ✅ Correct |

**Reduced paths (MISLABELED as "legacy"):**

| Module | Purpose | State Contract | Correct Label |
|--------|---------|----------------|---------------|
| `legacy_state.py` | 6D state class | 6D (u_0, v_0) | **REDUCED** |
| `legacy_state_adapter.py` | 6D conversions | 6D (u_0, v_0) | **REDUCED** |
| `step_legacy_contract.py` | 6D stepping | 6D (u_0, v_0) | **REDUCED** |
| `hybrid_state_contract.py` | 9D hybrid | 6D + 3D observable | **REDUCED + observable** |

**Authoritative reference citations:**

1. **TRUE legacy state dimension:**
   `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md:142`
   > **Total**: `18N + 15` core state variables.

2. **Rigid body state layout:**
   `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md:63-73`
   > Per coil: `v[3], w[3], p[3], R[9]` (18 scalars)

3. **Flexible tip state layout:**
   `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md:95-102`
   > Tip: `p[3], R[9], u[3]` (15 scalars)

4. **Legacy stepping entrypoint:**
   `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md:25-28`
   > **Function**: `DynamicsBVP`
   > **Location**: `legacy_worktree/src/CoilDynamics_Defs.cpp:1066`

---

## UPDATED GO/NO-GO DECISION

### Question 1: What is B0 intended to do?

**Option A:** System ID using TRUE legacy dynamics (DynamicsBVP → DYNSolverIVP) for parity with original MATLAB/C++ system ID
→ **Answer:** NO-GO — Current stack uses 6D reduced path, not TRUE legacy (18·N+15)

**Option B:** System ID using reduced 6D dynamics path (acceptable approximation)
→ **Answer:** GO — Current stack is safe for reduced 6D path with prescribed mitigations

### Question 2: Is the current stack "TRUE legacy"?

**Answer:** **NO**

**Evidence:**
- Current B0-bound path: `step_legacy_contract.py` (6D reduced)
- TRUE legacy path: `true_legacy_step.py` (18·N+15, calls DynamicsBVP → DYNSolverIVP)
- These are **distinct implementations** with different state contracts

### Updated Decision Logic

```
IF intention is "TRUE legacy system ID matching main branch CRMDYN_test.cpp":
    THEN decision = NO-GO
    REASON: Current stack milestone-a-hybrid-vjp uses 6D REDUCED state
    REQUIRED: Switch to true_legacy_step.py path (18·N+15)

ELSE IF intention is "system ID with 6D reduced dynamics":
    THEN decision = GO (with P0 mitigations)
    REASON: 6D reduced path is safe, deterministic, gradient-correct
    REQUIRED: Implement status checking, NaN validation, gradient clipping

ENDIF
```

---

## REQUIRED ACTIONS

### Immediate (Clarification)

1. **User must clarify B0 intent:**
   - Does B0 require TRUE legacy (18·N+15) for parity?
   - Or is 6D reduced dynamics acceptable?

### If TRUE legacy required (18·N+15)

1. **Switch to TRUE legacy path:**
   - Use `true_legacy_step_torch` from `python/control/true_legacy_step_autograd.py`
   - State dimension: `18 * n_act + 15` where `n_act = NUM_ACT_SET`
   - Re-audit VJP implementation (`CRM_BVPJacobian.cpp`)

2. **Re-run all Pre-B0 audits for TRUE legacy path**

### If 6D reduced acceptable

1. **Rename modules to avoid confusion:**
   - `legacy_state.py` → `reduced_state.py` (or add docstring clarification)
   - Add warning in `step_legacy_contract.py` that this is REDUCED, not TRUE legacy

2. **Proceed with existing GO decision + P0 mitigations**

---

## SUPERSEDED DOCUMENTS

The following documents are **SUPERSEDED** by this correction:

1. ~~`docs/audits/PRE_B0_PYTHON_CONTROL_AUDIT.md`~~ → See corrected version
2. ~~`docs/audits/PRE_B0_COMPLETION_REPORT.md`~~ → See corrected version
3. ~~`docs/audits/PRE_B0_RISKS_AND_MITIGATIONS.md`~~ → See corrected version

**New corrected versions will be written to replace these files.**

---

## SUMMARY OF CORRECTIONS

| Item | Incorrect | Correct |
|------|-----------|---------|
| **State label** | "6D legacy state" | "6D REDUCED state (non-legacy)" |
| **TRUE legacy dim** | (conflated with 6D) | **18·N + 15** explicitly |
| **Current B0 path** | "legacy" | **REDUCED 6D** |
| **GO/NO-GO** | Unconditional GO | **Conditional: GO for 6D, NO-GO for TRUE legacy (18·N+15)** |
| **Evidence citations** | Missing contract reference | **Cites `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md`** |

---

**END OF CORRECTION NOTE**

**Authority:** `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md` (FROZEN ground truth)
**Date:** 2026-01-03
**Status:** APPROVED
