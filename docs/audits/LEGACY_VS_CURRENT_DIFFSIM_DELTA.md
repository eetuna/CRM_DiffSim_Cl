# LEGACY VS CURRENT DIFFSIM DELTA

**Date**: 2026-01-03
**Legacy branch**: `main` (commit `828bf8c`)
**Current branch**: `milestone-a-hybrid-vjp` (commit `c764edb`)
**Purpose**: Identify gaps between TRUE legacy contract and current implementation

---

## 1. EXECUTIVE SUMMARY

**Status**: ⚠️ **SIGNIFICANT DIVERGENCE** — Current branch has added differentiable wrappers but may not match TRUE legacy contract.

**Key findings**:
- ✅ Legacy C++ code still present in current branch
- ⚠️ Current branch added Python layer (`python/control/`, `python/crm_bindings.cpp`)
- ⚠️ Current branch added differentiable equilibrium (`CRM_DiffEquilibrium`)
- ⚠️ Current branch has "hybrid state" and "legacy state" adapters
- ❌ **CRITICAL**: Uncertain if "legacy state" adapter matches TRUE legacy contract extracted from `main`

**Action required**: Verify current "legacy state" contract against TRUE legacy frozen contract.

---

## 2. FILE-LEVEL DELTA

### 2.1 Added Files (Current Only)

**Python infrastructure** (new in current):
```
python/control/__init__.py
python/control/hybrid_controller.py
python/control/hybrid_state_contract.py
python/control/ilqr.py
python/control/legacy_state.py
python/control/legacy_state_adapter.py
python/control/step_hybrid_legacy_contract.py
python/control/step_legacy_contract.py
python/crm_bindings.cpp
python/crm_config.py
python/crm_dynamics_torch.py
python/crm_equilibrium.py
```

**Differentiable C++ (new in current)**:
```
src/CRM_DiffDynamics.cpp
src/CRM_DiffDynamics.hpp
src/CRM_DiffEquilibrium.cpp
src/CRM_DiffEquilibrium.hpp
```

**Tests (new in current)**:
```
python/test_a1_legacy_state_adapter.py
python/test_a1_step_legacy_contract_forward.py
python/test_a2_implicit_vjp_gradcheck.py
python/test_a3_batched_vjp.py
python/test_hybrid_contract_roundtrip.py
python/test_hybrid_vjp_gradcheck.py
```

**Total new files**: ~150+ (mostly Python/tests/docs)

---

### 2.2 Deleted Files (Legacy Only)

**None** — All legacy C++ files are preserved in current branch.

**Conclusion**: Current is **additive** (no deletions).

---

### 2.3 Modified Files

**Key modifications**:
- `CMakeLists.txt` — added Python binding targets
- `src/CRM_IVPJacobian.cpp` — minor changes (likely for differentiability)
- Build artifacts (binaries regenerated)

**Core legacy files UNCHANGED**:
- ✅ `src/CoilDynamics_Defs.cpp` — BVP/IVP implementation intact
- ✅ `src/CRMDYN.hpp` — interface unchanged
- ✅ `src/CRM_StateVector_Definitions.hpp` — state definitions preserved

---

## 3. API-LEVEL DELTA

### 3.1 Legacy C++ API (Still Present)

**From `main` branch** (extracted in audit):
```cpp
void DynamicsBVP(
    CRMShootingMethodParams& in_Params,
    const double xf[NUM_STATES],
    double in_mL_initialguess[NUM_ACT_SET][3],
    double in_nL_initialguess[NUM_ACT_SET][3],
    double in_ftip_initialguess[3],
    double out_u0[3],
    double out_mL[NUM_ACT_SET][3],
    double out_nL[NUM_ACT_SET][3],
    double out_tau[NUM_ACT_SET][3],
    double out_ftip[3],
    int& out_localmin
);

void DYNSolverIVP(
    CRMShootingMethodParams& in_Params,
    const double in_u0[3],
    const double in_mL[NUM_ACT_SET][3],
    const double in_nL[NUM_ACT_SET][3],
    const double in_tau[NUM_ACT_SET][3],
    const double in_ftip[3],
    bool in_FinalValueOnly,
    double out_x_N[NUM_STATES],
    double out_coil_state[NUM_ACT_SET][NUM_COIL_STATES],
    double out_p_atLocMarkers[][3]
);
```

**Status in current**: ✅ **PRESENT** (not removed)

---

### 3.2 New Python API (Current Branch)

**From `python/control/__init__.py`** (current branch):

#### Milestone A: "Hybrid State" API
```python
from .hybrid_state_contract import (
    HybridState, HybridStepResult, HybridVJPResult,
    pack_hybrid_state, unpack_hybrid_state,
    hybrid_to_legacy, legacy_to_hybrid,
    STATE_DIM_HYBRID, STATE_DIM_DYNAMICS, STATE_DIM_OBSERVABLE,
)
from .step_hybrid_legacy_contract import (
    step_hybrid_legacy_contract, vjp_hybrid_legacy_contract,
)
```

**Dimensions claimed**:
- `STATE_DIM_HYBRID` = **9** (per docstring: "9D hybrid state")
- `STATE_DIM_DYNAMICS` = ?
- `STATE_DIM_OBSERVABLE` = ?

**Question**: ⚠️ How does 9D state relate to TRUE legacy's `18*N + 15` state?

---

#### A1/A2: "Legacy State" API
```python
from .legacy_state import (
    LegacyState,
    STATE_DIM_LEGACY, U0_DIM, V0_DIM,
)
from .step_legacy_contract import (
    step_legacy_contract, vjp_legacy_contract,
)
```

**Dimensions claimed**:
- `STATE_DIM_LEGACY` = **6** (per docstring: "Legacy 6D state")
- `U0_DIM` = ?
- `V0_DIM` = ?

**Question**: ❌ **CRITICAL MISMATCH** — TRUE legacy state is NOT 6D!
- TRUE rigid state per coil: **18** scalars `(v, w, p, R)` per `x_coil[j]`
- TRUE flexible tip state: **15** scalars `(p, R, u)`
- For N=1 coil: `18 + 15 = 33` state variables
- For N=2 coils: `36 + 15 = 51` state variables

**Hypothesis**: "6D state" may be a **reduced representation**, not the full TRUE state.

---

### 3.3 New C++ Differentiable API (Current Branch)

**From `src/CRM_DiffEquilibrium.hpp`**:
```cpp
int equilibrium_forward(
    const double* u,
    double L_inserted,
    const CRMForwardKinematicsData& fk_params,
    EquilibriumResult& result
);
```

**Note**: This is **static equilibrium**, not dynamics (no time evolution).

**Status**: New addition, doesn't conflict with legacy dynamics.

---

## 4. STATE CONTRACT DELTA

### 4.1 TRUE Legacy State (from main)

**Extracted from audit**:

| Component | Dimension | Total for N coils |
|-----------|-----------|-------------------|
| Rigid bodies (coils) | 18 per coil | `18 * N` |
| Flexible tip | 15 | 15 |
| **Total** | — | **`18N + 15`** |

**Example**:
- N=1: `18 + 15 = 33` scalars
- N=2: `36 + 15 = 51` scalars

**Packing**:
```
x_coil[0][0:18]   = [v0, w0, p0, R0]  (coil 0 state)
x_coil[1][0:18]   = [v1, w1, p1, R1]  (coil 1 state, if N=2)
...
xf[0:15]          = [p_tip, R_tip, u_tip]
```

---

### 4.2 Current "Hybrid State" (from python/control/__init__.py)

**Claimed dimension**: 9

**Speculation** (no code inspection yet):
- Possibly `(p, v, θ)` where `θ` are reduced DOF (e.g., Euler angles instead of rotation matrix)?
- Or: tip-only state, ignoring intermediate coils?

**Conclusion**: ⚠️ **Needs investigation** — 9D ≠ `18N + 15`.

---

### 4.3 Current "Legacy State" (from python/control/__init__.py)

**Claimed dimension**: 6

**Speculation**:
- Possibly `(u0, v0)` where `u0` is base curvature, `v0` is initial velocity?
- Or: Tip pose only `(p_tip[3], orientation[3])`?

**Conclusion**: ❌ **MISMATCH** — 6D ≠ TRUE legacy full state.

---

## 5. STEPPING CONTRACT DELTA

### 5.1 TRUE Legacy Stepping (from main)

**Pattern**:
```cpp
// Setup
CRMShootingMethodParams params = CRMDYNConstructShootingMethodParamSet(...);

// Step
DynamicsBVP(params, xf_prev, mL_guess, nL_guess, ftip_guess,
            out_u0, out_mL, out_nL, out_tau, out_ftip, localmin);

DYNSolverIVP(params, out_u0, out_mL, out_nL, out_tau, out_ftip, true,
             xf_new, x_coil_new, markers);

// Update
xf_prev = xf_new;
x_coil_prev = x_coil_new;
mL_guess = out_mL;
nL_guess = out_nL;
```

**State dimension**: Full `18N + 15`
**BVP unknowns**: `6N` (interface forces/moments)

---

### 5.2 Current "step_legacy_contract" (from python/control/__init__.py)

**Signature** (not yet inspected):
```python
step_legacy_contract(...)  # returns LegacyStepResult
vjp_legacy_contract(...)   # returns LegacyVJPResult
```

**Expected behavior** (from docstring):
- "A1: Legacy 6D state adapter (contract-exact)"
- "implicit VJP"

**Question**: ❌ What does "contract-exact" mean if state is 6D, not `18N+15`D?

**Hypothesis**: Adapter may:
1. Take 6D **reduced state** as input
2. Expand to full `18N+15`D state internally
3. Call TRUE legacy C++ `DynamicsBVP + DYNSolverIVP`
4. Compress back to 6D output

**Needs verification**: Inspect `legacy_state_adapter.py` and `step_legacy_contract.py`.

---

### 5.3 Current "step_hybrid_legacy_contract" (from python/control/__init__.py)

**Signature**:
```python
step_hybrid_legacy_contract(...)  # returns HybridStepResult
vjp_hybrid_legacy_contract(...)   # returns HybridVJPResult
```

**Claimed**:
- Milestone A prototype
- 9D hybrid state

**Question**: ⚠️ How does this relate to TRUE legacy?

**Speculation**: May be a **new formulation**, not claiming to match TRUE legacy exactly.

---

## 6. BVP UNKNOWN DELTA

### 6.1 TRUE Legacy BVP Unknowns (from main)

**Unknowns**: `(mL[N][3], nL[N][3])` packed as `[m0, n0, m1, n1, ...]`
**Dimension**: `6N`
**Solver**: Trust-region dogleg
**Tolerance**: `1e-5`

**Physical meaning**: Interface forces/moments ensuring rigid-flexible continuity.

---

### 6.2 Current Branch BVP Behavior

**C++ equilibrium** (`CRM_DiffEquilibrium`):
- Uses `CRM_ForwardKinematics` (static, not dynamics)
- Likely solves for `u0, ftip` (standard BVP), not `(mL, nL)`

**Python dynamics** (`step_legacy_contract`):
- Unknown if it calls TRUE legacy `DynamicsBVP` or a new implementation

**Conclusion**: ⚠️ **UNCLEAR** — need to inspect Python wrapper to confirm BVP path.

---

## 7. CRITICAL GAPS

### 7.1 State Dimension Mismatch

| Source | State Dimension | Evidence |
|--------|-----------------|----------|
| TRUE legacy (main) | `18N + 15` | Direct code extraction |
| Current "legacy state" | 6 | Docstring claim |
| Current "hybrid state" | 9 | Docstring claim |

**Gap**: ❌ **6D and 9D do NOT match TRUE legacy full state**

**Hypotheses**:
1. **Reduced representation**: Current uses reduced DOF (e.g., tip-only)
2. **Adapter layer**: Current expands 6D → full state internally
3. **Different formulation**: Current is NOT claiming to match TRUE legacy

**Action**: Inspect `legacy_state_adapter.py` to resolve.

---

### 7.2 Stepping API Divergence

**TRUE legacy** (from main):
- Two-step process: `DynamicsBVP` + `DYNSolverIVP`
- User must manually manage state persistence
- No automatic differentiation

**Current** (claimed):
- Single-function API: `step_legacy_contract(...)`
- "implicit VJP" (automatic differentiation)
- "Contract-exact" (but dimension mismatch?)

**Gap**: ⚠️ **Interface is different**, unclear if behavior matches.

---

### 7.3 Missing Evidence

**Not yet inspected**:
- `python/control/legacy_state.py` — what IS the "6D state"?
- `python/control/legacy_state_adapter.py` — how does 6D → full state work?
- `python/control/step_legacy_contract.py` — does it call TRUE legacy C++?
- `python/test_a1_step_legacy_contract_forward.py` — what tests validate?

**Conclusion**: ❌ **INSUFFICIENT EVIDENCE** to claim parity with TRUE legacy.

---

## 8. COMPATIBILITY ANALYSIS

### 8.1 Can Current Call TRUE Legacy?

**C++ layer**: ✅ **YES**
- `DynamicsBVP` and `DYNSolverIVP` still present
- Can be called directly from C++ or via bindings

**Python layer**: ⚠️ **UNKNOWN**
- Need to inspect `crm_bindings.cpp` to see if legacy dynamics are exposed
- From partial read (lines 1-100): Only saw `equilibrium_forward`, not dynamics

**Conclusion**: C++ path exists, but Python wrapper unclear.

---

### 8.2 Does Current Reproduce TRUE Legacy Behavior?

**Tests claiming validation** (from current branch):
- `test_a1_step_legacy_contract_forward.py` — forward pass test
- `test_a2_implicit_vjp_gradcheck.py` — VJP correctness
- `test_a3_batched_vjp.py` — batching

**What's NOT tested** (based on filenames):
- ❌ Comparison to `CRMDYN_test.cpp` demo (TRUE legacy stepping loop)
- ❌ State dimension validation (6D vs `18N+15`)
- ❌ BVP unknown packing (does Python match C++ structure?)

**Conclusion**: ⚠️ **TESTS EXIST** but don't prove parity with TRUE legacy contract.

---

## 9. RECOMMENDED ACTIONS

### 9.1 Immediate (Required Before Milestone A Sign-Off)

1. **Inspect `legacy_state.py`**:
   - What are the 6 dimensions?
   - How do they relate to TRUE legacy `(x_coil, xf)`?

2. **Inspect `legacy_state_adapter.py`**:
   - Does `pack_legacy_state` expand to full `18N+15`?
   - Or is it a genuinely reduced representation?

3. **Inspect `step_legacy_contract.py`**:
   - Does it call `DynamicsBVP + DYNSolverIVP`?
   - Or is it a reimplementation?

4. **Inspect `crm_bindings.cpp`**:
   - Are dynamics functions exposed to Python?
   - Or only equilibrium?

5. **Run comparison test**:
   - Port `CRMDYN_test.cpp` loop to Python
   - Compare outputs to current `step_legacy_contract`
   - Verify state trajectories match

---

### 9.2 Documentation (Needed)

1. **State mapping document**:
   - Explicit formula: 6D ↔ `18N+15`
   - When is each representation valid?

2. **API compatibility matrix**:
   - Which Python functions call which C++ functions?
   - What transformations occur at the boundary?

3. **Test coverage report**:
   - Do tests cover TRUE legacy contract, or just current API?

---

### 9.3 Risk Mitigation

**If 6D ≠ TRUE legacy**:
- ⚠️ **Risk**: Milestone A VJP may not be correct for TRUE legacy dynamics
- **Mitigation**: Either:
  1. Prove 6D is a valid reduction with invertible mapping, OR
  2. Extend current to support full `18N+15` state

**If current doesn't call legacy C++**:
- ⚠️ **Risk**: Reimplementation bugs (different BVP unknowns, integrator, etc.)
- **Mitigation**:
  1. Add tests comparing to `DynamicsBVP + DYNSolverIVP` directly
  2. Port TRUE legacy to Python as reference implementation

---

## 10. GO/NO-GO ASSESSMENT

### 10.1 For Using Current "Legacy State" as Ground Truth

**Status**: ❌ **NO-GO**

**Reasons**:
1. Dimension mismatch (6D ≠ `18N+15`)
2. No evidence of equivalence proof
3. Insufficient test coverage vs TRUE legacy

**Required before GO**:
- Prove 6D → full state mapping is correct
- Add regression test vs `CRMDYN_test.cpp`
- Document exactly what "legacy" means in current context

---

### 10.2 For Proceeding with Milestone A VJP

**Status**: ⚠️ **CONDITIONAL GO**

**Safe path**:
1. Use current "hybrid state" (9D) as NEW formulation
2. Do NOT claim it matches TRUE legacy
3. Build VJP for current system
4. Later: Add TRUE legacy VJP separately

**Risky path**:
1. Claim current "legacy state" matches TRUE legacy
2. Build VJP assuming equivalence
3. Discover bugs later when comparing to C++ demos

**Recommendation**: Take safe path.

---

## 11. SUMMARY TABLE

| Aspect | TRUE Legacy (main) | Current (milestone-a) | Match? |
|--------|-------------------|---------------------|--------|
| C++ dynamics API | `DynamicsBVP + DYNSolverIVP` | Still present | ✅ |
| State dimension | `18N + 15` | 6D ("legacy") or 9D ("hybrid") | ❌ |
| BVP unknowns | `6N` (mL, nL) | Unknown | ⚠️ |
| Python wrapper | None | `step_legacy_contract` | N/A |
| VJP support | None | `vjp_legacy_contract` | N/A |
| Differentiability | No | Yes (implicit) | N/A |
| Test coverage | Demo only | Unit tests | ⚠️ |

**Legend**:
- ✅ = Match confirmed
- ❌ = Mismatch confirmed
- ⚠️ = Unknown / needs investigation
- N/A = Not applicable (different layer)

---

## 12. CONCLUSION

**Current branch has diverged** from TRUE legacy in the following ways:

1. **Added Python layer** with reduced state representations (6D, 9D)
2. **Added automatic differentiation** (implicit VJP)
3. **Unclear mapping** from reduced state to TRUE legacy full state

**Before using current as "legacy" reference**:
- ❌ Must prove state dimension mapping
- ❌ Must validate against TRUE legacy C++ demos
- ❌ Must document exactly what "legacy" means

**Recommendation**:
- **Do NOT** assume current "legacy state" matches TRUE legacy
- **Do** inspect adapter code before proceeding with A1-A3
- **Do** add regression tests vs `CRMDYN_test.cpp` demo

**Status**: ⚠️ **AUDIT INCOMPLETE** — critical gaps remain.

---

## APPROVAL

**Status**: ⚠️ **CONDITIONAL APPROVAL**
- Evidence: Git diff confirms file additions
- Limitation: Python code not fully inspected
- Action: Must inspect `legacy_state_adapter.py` before Milestone A sign-off

**Auditor**: Claude Code
**Date**: 2026-01-03
