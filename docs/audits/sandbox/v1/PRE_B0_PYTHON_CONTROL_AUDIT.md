# PRE-B0 PYTHON CONTROL LAYER AUDIT (CORRECTED)

**Audit Date:** 2026-01-03
**Auditor:** Claude (Sonnet 4.5)
**Scope:** Python control module (`python/control/`)
**Current Branch:** `milestone-a-hybrid-vjp`
**Authority:** `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md`

**CORRECTION:** This document supersedes the previous version which incorrectly labeled 6D state as "legacy"

---

## Executive Summary

The Python control layer implements **THREE DISTINCT STATE CONTRACTS**:

1. **TRUE legacy (18·N+15)** — Full rigid-body + flexible tip state (A0 milestone)
2. **Reduced 6D (u_0, v_0)** — Reduced dynamics state (A1+A2 milestone)
3. **Hybrid 9D (6D + 3D)** — Reduced + observable (Milestone A prototype)

**CRITICAL FINDING:** The current B0-intended path uses **REDUCED 6D** state, **NOT TRUE legacy (18·N+15)**.

**State contract classifications:**

| Path | State Dim | Label | DynamicsBVP/DYNSolverIVP? | Evidence |
|------|-----------|-------|---------------------------|----------|
| `true_legacy_step.py` | **18·N + 15** | **TRUE legacy** | ✅ YES | `python/control/true_legacy_step.py:5,34` |
| `step_legacy_contract.py` | **6D** | **REDUCED (non-legacy)** | ❌ NO | `python/control/legacy_state.py:19` |
| `step_hybrid_legacy_contract.py` | **9D** | **REDUCED + observable** | ❌ NO | `python/control/hybrid_state_contract.py:36` |

**Authoritative reference:** `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md:142`

---

## 1. State Contract Audit (CORRECTED)

### 1.1 TRUE Legacy State Definition (18·N+15)

**Authoritative contract:** `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md`

**State composition:**

**Per actuator coil (N = NUM_ACT_SET):**
- Linear velocity: `v[3]` (body frame)
- Angular velocity: `w[3]` (body frame)
- Position: `p[3]` (spatial frame)
- Rotation: `R[9]` (row-major, spatial frame)
- **Total per coil:** 18 scalars

**Flexible tip:**
- Position: `p_tip[3]` (spatial frame)
- Rotation: `R_tip[9]` (row-major, spatial frame)
- Curvature strain: `u_tip[3]` (body frame)
- **Total tip:** 15 scalars

**Total TRUE legacy state:** `18·N + 15`

**Evidence:**
```
docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md:63-73
docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md:95-102
docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md:142
```

**Python implementation:**

**File:** `python/control/true_legacy_state_adapter.py`

```python
COIL_STATE_DIM = 18  # v[3], w[3], p[3], R[9]
TIP_STATE_DIM = 15   # p_tip[3], R_tip[9], u_tip[3]

def true_legacy_state_dim(n_act: int) -> int:
    """
    Compute TRUE legacy state dimension.
    State dimension: 18·N + 15
    """
    return COIL_STATE_DIM * n_act + TIP_STATE_DIM
```

**Evidence:** `python/control/true_legacy_state_adapter.py:20-34`

**Public API:**

**File:** `python/control/true_legacy_step.py:23-68`

```python
def true_legacy_step(
    x: torch.Tensor,  # [18*n_act + 15] or [B, 18*n_act + 15]
    u: torch.Tensor,  # [n_act, 3] or [B, n_act, 3]
    dt: float,
    *,
    n_act: int,
    catheter_params: Optional[Dict] = None,
    warmstart: Optional[Dict] = None,
    return_orientation: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    """
    One-step forward using TRUE legacy dynamics (DynamicsBVP → DYNSolverIVP).

    State representation: 18·N + 15 (no reduction, full rotation matrices)
    """
```

**Evidence:** `python/control/true_legacy_step.py:23-38`

**C++ binding:**

**File:** `python/crm_bindings.cpp:925-930`

```cpp
// A0: TRUE legacy stepping (DynamicsBVP → DYNSolverIVP)
m.def("true_legacy_step_forward", &py_true_legacy_step_forward,
      py::arg("x_coil"), py::arg("xf"), py::arg("u"),
      py::arg("dt"), py::arg("catheter_params"),
      py::arg("mL_guess") = py::none(), py::arg("nL_guess") = py::none(),
      "A0: TRUE legacy step (DynamicsBVP → DYNSolverIVP). Returns next state and observables.");
```

**C++ call chain to legacy solvers:**

**File:** `src/CRM_TrueLegacyDynamics.cpp:104-112`

```cpp
// Call DynamicsBVP
DynamicsBVP(shooting_params, xf, mL_guess_arr, nL_guess_arr, ftip_guess,
            out.u0, out.mL, out.nL, out.tau, out.ftip,
            out.ftip_calc, out_localmin);

// Call DYNSolverIVP
DYNSolverIVP(shooting_params, out.u0, out.mL, out.nL, out.tau, out.ftip,
             out.ftip_calc, out.x_coil_next, out.xf_next);
```

**Evidence:** `src/CRM_TrueLegacyDynamics.cpp:12,104-112`

**Authoritative legacy reference:**

**File:** `legacy_worktree/src/CoilDynamics_Defs.cpp:1066` (DynamicsBVP)
**File:** `legacy_worktree/src/CoilDynamics_Defs.cpp:1195` (DYNSolverIVP)

**Evidence:** `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md:25-55`

### 1.2 Reduced 6D State Definition (MISLABELED "legacy")

**File:** `python/control/legacy_state.py`

**Dimensions:**
```python
STATE_DIM_LEGACY = 6  # REDUCED state dimension (NOT TRUE legacy)
U0_DIM = 3            # Base curvature (u_0)
V0_DIM = 3            # Base angular velocity (v_0)
```

**Evidence:** `python/control/legacy_state.py:19-21`

**State composition:**
```python
@dataclass
class LegacyState:
    """
    Immutable 6D REDUCED state representation.

    IMPORTANT: This is a REDUCED state, NOT the TRUE legacy (18·N+15).
    The name "LegacyState" is MISLEADING for historical reasons.
    """
    u_0: np.ndarray  # (3,) base curvature vector (1/mm)
    v_0: np.ndarray  # (3,) base curvature velocity (rad/s)
```

**Evidence:** `python/control/legacy_state.py:25-41`

**Observables (NOT in State):**

**File:** `python/control/step_legacy_contract.py:195-199`

```python
observables = {
    'p_tip': p_tip,   # Tip position (3,) mm
    'u_tip': u_tip,   # Tip curvature (3,) 1/mm
}
```

**Evidence:**
- p_tip and u_tip are **observables**, not state variables
- Returned separately in `LegacyStepResult.observables`
- **NO observable leakage into state**

**Contract adherence:** ✅ Strict separation of state (6D) and observables

**CRITICAL DISTINCTION:**
- TRUE legacy: 18·N+15 (includes coil rigid-body states + tip pose/strain)
- Reduced 6D: u_0, v_0 only (base curvature + velocity)

### 1.3 Hybrid 9D State Definition (Milestone A Prototype)

**File:** `python/control/hybrid_state_contract.py:34-36`

```python
STATE_DIM_DYNAMICS = 6    # Reduced CP2 state: [u_0, v_0]
STATE_DIM_OBSERVABLE = 3  # Tip position p_tip
STATE_DIM_HYBRID = 9      # Full hybrid state
```

**State composition:**
```python
@dataclass
class HybridState:
    u_0: np.ndarray    # (3,) base curvature (1/mm)
    v_0: np.ndarray    # (3,) base curvature velocity (1/mm/s)
    p_tip: np.ndarray  # (3,) tip position (mm) - OBSERVABLE in state
```

**Evidence:** `python/control/hybrid_state_contract.py:54-60`

**Purpose:** Prototype for Milestone A, includes observable in state for optimizer convenience

**Status:** NOT used for B0 system ID (prototype only)

---

## 2. Python Control Layer Classification

### 2.1 Module-by-Module Classification

**TRUE legacy paths (18·N+15):**

| Module | File:Line | Purpose | State Contract | Status |
|--------|-----------|---------|----------------|--------|
| State adapter | `true_legacy_state_adapter.py:20-34` | Pack/unpack | 18·N+15 | ✅ Correct |
| Forward step | `true_legacy_step.py:23-180` | Forward pass | 18·N+15 | ✅ Correct |
| Autograd | `true_legacy_step_autograd.py:22-218` | PyTorch autograd | 18·N+15 | ✅ Correct |

**Reduced 6D paths (MISLABELED "legacy"):**

| Module | File:Line | Purpose | State Contract | Correct Label |
|--------|-----------|---------|----------------|---------------|
| State class | `legacy_state.py:25-120` | 6D state | u_0, v_0 (6D) | **REDUCED** |
| State adapter | `legacy_state_adapter.py:15-139` | 6D conversions | u_0, v_0 (6D) | **REDUCED** |
| Forward step | `step_legacy_contract.py:114-215` | Forward pass | u_0, v_0 (6D) | **REDUCED** |
| VJP | `step_legacy_contract.py:218-340` | Implicit VJP | u_0, v_0 (6D) | **REDUCED** |

**Hybrid 9D paths (Milestone A prototype):**

| Module | File:Line | Purpose | State Contract | Correct Label |
|--------|-----------|---------|----------------|---------------|
| Hybrid state | `hybrid_state_contract.py:52-75` | 9D state | 6D + 3D obs | **REDUCED + observable** |
| Hybrid step | `step_hybrid_legacy_contract.py:47-150` | Forward + VJP | 6D + 3D obs | **REDUCED + observable** |

**Control algorithms:**

| Module | File:Line | Purpose | Compatible State | Status |
|--------|-----------|---------|------------------|--------|
| iLQR | `ilqr.py:14-300` | Trajectory opt | 6D or 9D | ✅ Correct |
| MPC | `mpc.py:27-200` | Model predictive | 6D or 9D | ✅ Correct |

### 2.2 Public API Export Classification

**File:** `python/control/__init__.py`

**TRUE legacy exports (A0):**
```python
# A0: TRUE legacy state adapter (18*N+15, no reduction)
from .true_legacy_state_adapter import (
    COIL_STATE_DIM, TIP_STATE_DIM,
    true_legacy_state_dim, true_legacy_warmstart_dim,
    pack_true_legacy_state, unpack_true_legacy_state,
)
from .true_legacy_step import (
    true_legacy_step,
)
from .true_legacy_step_autograd import (
    true_legacy_step_torch,
    TrueLegacyStepFn,
)
```

**Evidence:** `python/control/__init__.py:45-58`

**Reduced 6D exports (A1+A2 - MISLABELED "legacy"):**
```python
# A1 + A2: Legacy 6D state (contract-exact) with implicit VJP
# WARNING: "Legacy" here means REDUCED 6D, NOT TRUE legacy (18·N+15)
from .legacy_state import (
    LegacyState,
    STATE_DIM_LEGACY, U0_DIM, V0_DIM,
)
from .step_legacy_contract import (
    LegacyStepResult,
    LegacyVJPResult,
    step_legacy_contract,
    vjp_legacy_contract,
)
```

**Evidence:** `python/control/__init__.py:25-43`

**Comment line 7:**
```python
# - A1: Legacy 6D state adapter (contract-exact)
```

**Evidence:** `python/control/__init__.py:7`

**INCORRECT LABELING:** "Legacy 6D" is misleading — should say "REDUCED 6D (non-legacy)"

---

## 3. Forward Pass Audit by State Contract

### 3.1 TRUE Legacy Forward Pass (18·N+15)

**Function:** `true_legacy_step`
**File:** `python/control/true_legacy_step.py:23-180`

**Signature:**
```python
def true_legacy_step(
    x: torch.Tensor,      # [18*n_act + 15] or [B, 18*n_act + 15]
    u: torch.Tensor,      # [n_act, 3] or [B, n_act, 3]
    dt: float,
    *,
    n_act: int,
    catheter_params: Optional[Dict] = None,
    warmstart: Optional[Dict] = None,
    return_orientation: bool = True,
) -> Tuple[torch.Tensor, Dict]:
```

**Input validation (lines 74-109):**
- State shape: `[18*n_act + 15]` or `[B, 18*n_act + 15]`
- Control shape: `[n_act, 3]` or `[B, n_act, 3]`
- dt > 0
- catheter_params required

**Processing:**
1. Unpack state: `x_coil, xf = unpack_true_legacy_state(x, n_act)`
2. Call C++ binding: `crm_diff_py.true_legacy_step_forward(...)`
3. C++ calls: `DynamicsBVP(...)` then `DYNSolverIVP(...)`
4. Pack next state: `x_next = pack_true_legacy_state(x_coil_next, xf_next)`

**Evidence:** `python/control/true_legacy_step.py:183-266`

**C++ call chain:**

**Binding:** `python/crm_bindings.cpp:951-1200` (py_true_legacy_step_forward)
**Wrapper:** `src/CRM_TrueLegacyDynamics.cpp:12-120` (TrueLegacyDynamics_Forward)
**Legacy core:** `legacy_worktree/src/CoilDynamics_Defs.cpp:1066,1195` (DynamicsBVP, DYNSolverIVP)

**Verification:** ✅ TRUE legacy path directly calls canonical stepping sequence

### 3.2 Reduced 6D Forward Pass

**Function:** `step_legacy_contract`
**File:** `python/control/step_legacy_contract.py:114-215`

**Signature:**
```python
def step_legacy_contract(
    x_t: LegacyState,      # Current state (6D: u_0, v_0)
    u_t: np.ndarray,       # Control (3,) - actuation currents
    dt: float,             # Time step (seconds)
    L_inserted: float,     # Insertion length (mm)
    params: Dict[str, Any], # Catheter parameters
) -> LegacyStepResult:
```

**Input validation (lines 156-171):**
```python
# Validate input state
if not isinstance(x_t, LegacyState):
    raise TypeError(f"x_t must be LegacyState, got {type(x_t)}")

# Validate control
u_t = np.asarray(u_t, dtype=np.float64)
if u_t.shape != (3,):
    raise ValueError(f"u_t must be (3,), got {u_t.shape}")
if not np.all(np.isfinite(u_t)):
    raise ValueError("u_t contains non-finite values (NaN or Inf)")
```

**Evidence:** ✅ Shape, dtype, finiteness validation strict

**C++ binding call (lines 180-184):**
```python
result = crm_diff_py.dynamics_forward(
    x_t_np, u_t_np, float(dt), float(L_inserted), params
)
```

**C++ call chain:**

**Binding:** `python/crm_bindings.cpp:250-340` (py_dynamics_forward)
**Wrapper:** `src/CRM_DiffDynamics.cpp` (DynamicsForward)
**DOES NOT CALL:** DynamicsBVP or DYNSolverIVP

**Evidence:** `rg "DynamicsBVP|DYNSolverIVP" src/CRM_DiffDynamics.cpp` returns empty

**Forward result caching (lines 186-214):**
```python
return LegacyStepResult(
    x_next=x_next,
    observables=observables,
    diagnostics=diagnostics,
    _fwd_cache=result,  # Cache full forward result for VJP
)
```

**Evidence:** ✅ Full forward result cached for implicit VJP

---

## 4. Backward Pass Audit by State Contract

### 4.1 TRUE Legacy VJP (18·N+15)

**Function:** `TrueLegacyStepFn.backward`
**File:** `python/control/true_legacy_step_autograd.py:104-177`

**Signature:**
```python
@staticmethod
def backward(ctx, *grad_outputs):
    """
    Backward pass: Compute VJP using implicit differentiation.

    Returns:
        (grad_x, grad_u, None, None, None, None, None)
    """
```

**C++ VJP binding call (line 158-160):**
```python
vjp_result = crm_diff_py.true_legacy_step_vjp(
    x_coil_np, xf_np, u_np, dt, L_inserted, params_with_L, grad_tip_p_np
)
```

**C++ VJP implementation:**

**Binding:** `python/crm_bindings.cpp:1300-1500` (py_true_legacy_step_vjp)
**Jacobians:** `src/CRM_BVPJacobian.cpp` (BVP adjoint Jacobians)

**Evidence:** TRUE legacy VJP uses BVP Jacobians via implicit differentiation

**Gradients returned:**
```python
grad_x_coil = torch.from_numpy(grad_x_coil_np)  # [n_act, 18]
grad_xf = torch.from_numpy(grad_xf_np)          # [15]
grad_u = torch.from_numpy(grad_u_np)            # [n_act, 3]

grad_x = pack_true_legacy_state(grad_x_coil, grad_xf)  # [18*n_act + 15]
```

**Evidence:** `python/control/true_legacy_step_autograd.py:163-173`

### 4.2 Reduced 6D VJP

**Function:** `vjp_legacy_contract`
**File:** `python/control/step_legacy_contract.py:218-340`

**Signature:**
```python
def vjp_legacy_contract(
    fwd_result: LegacyStepResult,  # Cached forward result
    grad_x_next: np.ndarray,       # Upstream gradient (6,)
    params: Dict[str, Any],        # Same params used in forward
) -> LegacyVJPResult:
```

**Input validation:**
- Checks `fwd_result._fwd_cache` is not None
- Validates `grad_x_next.shape == (6,)`
- Validates `grad_x_next.dtype == float64`

**C++ binding call:**
```python
bwd_result = crm_diff_py.dynamics_backward(
    fwd_result._fwd_cache, grad_x_next, params
)
```

**Evidence:**
- Calls `py_dynamics_backward` in `python/crm_bindings.cpp:370-490`
- Uses cached Jacobians from `fwd_result._fwd_cache`
- **NO physics solver re-execution** (implicit differentiation via linear solve)

**VJP output:**
```python
return LegacyVJPResult(
    grad_x_t=bwd_result['grad_x_t'],  # (6,) ∂L/∂x_t
    grad_u_t=bwd_result['grad_u_t'],  # (3,) ∂L/∂u_t
    diagnostics={
        'status': bwd_result['status'],
        'lu_rank': bwd_result['lu_rank'],
        'rel_residual': bwd_result['rel_residual'],
    }
)
```

**Evidence:** ✅ Gradients returned, diagnostics surfaced, no gradient clipping

---

## 5. Determinism Audit

### 5.1 Randomness Check

**Grep evidence:** No `np.random`, `torch.randn`, or `random.seed` in control layer

**Evidence from control/__init__.py:**
- No imports of `random`, `numpy.random`, or `torch`
- All functions deterministic (given fixed inputs, outputs identical)

**Determinism guarantee:** ✅ Fully deterministic (no stochastic elements)

### 5.2 Caching Correctness

**Forward pass:**
1. Call `crm_diff_py.dynamics_forward(x_t, u_t, dt, L, params)` or `true_legacy_step_forward(...)`
2. Store full `result` dict in `_fwd_cache`

**Backward pass:**
1. Retrieve `_fwd_cache` from `fwd_result`
2. Call `crm_diff_py.dynamics_backward(_fwd_cache, grad_x_next, params)` or `true_legacy_step_vjp(...)`
3. C++ reconstructs Jacobians from cached data
4. Solves adjoint system (no solver re-execution)

**Evidence:** ✅ No hidden forward re-execution in backward

---

## 6. GO / NO-GO for B0 (CORRECTED)

### 6.1 Critical Question

**Does B0 require TRUE legacy (18·N+15) for parity with main branch system ID?**

**Option A: TRUE legacy required**
→ **NO-GO** — Current stack uses 6D reduced path

**Required action:**
- Switch to `true_legacy_step_torch` path
- Re-audit VJP implementation (`CRM_BVPJacobian.cpp`)
- Implement P0 mitigations for 18·N+15 state

**Option B: 6D reduced acceptable**
→ **GO** — Current stack is safe for 6D reduced path

**Required action:**
- Implement P0 mitigations (status checking, NaN validation, gradient clipping)
- Clarify that B0 is NOT using TRUE legacy dynamics

### 6.2 GO Decision (Conditional)

**IF using 6D reduced path:**

**GO** — Python control layer is **safe and suitable** for B0 system identification, with the following **required mitigations:**

1. **Status validation:**
   ```python
   if not result.success:
       raise RuntimeError(f"Step failed: {result.diagnostics}")
   ```

2. **NaN validation:**
   ```python
   if not np.all(np.isfinite(result.x_next.to_numpy())):
       raise ValueError("Forward step produced NaN/Inf")
   ```

3. **Gradient clipping:**
   ```python
   grad_x_t = np.clip(vjp_result.grad_x_t, -1e3, 1e3)
   grad_u_t = np.clip(vjp_result.grad_u_t, -1e3, 1e3)
   ```

4. **Rank-deficiency handling:**
   ```python
   if vjp_result.diagnostics['lu_rank'] < 6:
       # Degrade gracefully
       grad_x_t = np.zeros(6)
       grad_u_t = np.zeros(3)
   ```

**IF using TRUE legacy (18·N+15):**

**REQUIRES ADDITIONAL AUDIT** — TRUE legacy VJP not fully audited in this document

---

## 7. Summary of Findings (CORRECTED)

| Category | Finding | Risk Level | Mitigation |
|----------|---------|-----------|------------|
| **State Contract (6D)** | 6D REDUCED state (u_0, v_0), NOT TRUE legacy | **CRITICAL** | Clarify intent |
| **State Contract (18·N+15)** | TRUE legacy exists, not currently used | **None** | Document clearly |
| **Observable Leakage** | p_tip, u_tip separate (not in state) | **None** | N/A |
| **Caching** | Forward cached, backward reads cache | **None** | N/A |
| **Determinism** | No RNG, fully deterministic | **None** | N/A |
| **Status Checking** | Silent failure if status not checked | **High** | Check `result.success` |
| **NaN Propagation** | No automatic NaN check | **Medium** | Add finiteness check |
| **Gradient Explosion** | No gradient clipping | **Medium** | Clip gradients |
| **LU Rank** | Rank deficiency not handled gracefully | **Medium** | Degrade to zero gradients |

---

## 8. Evidence Log (CORRECTED)

**Authoritative contract:**
- `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md` (FROZEN ground truth)

**Files audited:**
- `python/control/__init__.py:1-97`
- `python/control/true_legacy_state_adapter.py:1-200`
- `python/control/true_legacy_step.py:1-267`
- `python/control/true_legacy_step_autograd.py:1-219`
- `python/control/legacy_state.py:1-126`
- `python/control/step_legacy_contract.py:1-450`
- `python/control/hybrid_state_contract.py:1-240`

**Legacy worktree:**
- `legacy_worktree/src/CoilDynamics_Defs.cpp:1066-1260` (DynamicsBVP, DYNSolverIVP)

**C++ bindings:**
- `python/crm_bindings.cpp:925-1500` (TRUE legacy bindings)
- `python/crm_bindings.cpp:250-490` (Reduced 6D bindings)

**C++ implementations:**
- `src/CRM_TrueLegacyDynamics.cpp:12-120`
- `src/CRM_BVPJacobian.cpp:1-300`
- `src/CRM_DiffDynamics.cpp` (reduced 6D wrapper)

**No Python control code modified legacy physics**

**All findings are documentation-only (audit mode)**

---

**END OF AUDIT (CORRECTED)**

**SUPERSEDES:** Previous version of this document dated 2026-01-03
**AUTHORITY:** `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md`
**CORRECTION NOTE:** `docs/audits/PRE_B0_CORRECTION_NOTE.md`
