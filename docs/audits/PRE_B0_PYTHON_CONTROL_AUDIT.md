# PRE-B0 PYTHON CONTROL LAYER AUDIT

**Audit Date:** 2026-01-04
**Auditor:** Claude Code (Sonnet 4.5)
**Scope:** Python control layer (`python/control/`)
**Current Branch:** `true-legacy-dynamics` (commit 6d86781)
**Authority:** `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md`

---

## EXECUTIVE SUMMARY

The Python control layer implements **both** the TRUE legacy state contract (18·N+15) AND a reduced 6D dynamics interface. The TRUE legacy implementation correctly adheres to the FULLSTATE hybrid contract with no observable leakage, clean separation of concerns, and efficient batched VJP support for trajectory optimization.

**Key Findings:**
- ✅ TRUE legacy state contract (18·N+15) correctly implemented
- ✅ No observable leakage into state
- ✅ Robust packing/unpacking with comprehensive validation
- ✅ Efficient batched VJP for optimization
- ✅ PyTorch autograd integration seamless
- ⚠️ MPC uses seeded random cold-start (seed=42) — deterministic but worth noting
- ✅ Error handling comprehensive with detailed diagnostics

---

## 1. TRUE LEGACY STATE CONTRACT (18·N+15)

### 1.1 Core Implementation

**File:** `python/control/true_legacy_state_adapter.py`

**State Dimensions:**
- **Coil State (per actuator):** 18 elements `[v[3], w[3], p[3], R[9]]`
  - `v[3]`: Linear velocity (body frame)
  - `w[3]`: Angular velocity (body frame)
  - `p[3]`: Position (spatial frame)
  - `R[9]`: Rotation matrix, row-major (spatial frame)

- **Tip State:** 15 elements `[p_tip[3], R_tip[9], u_tip[3]]`
  - `p_tip[3]`: Tip position
  - `R_tip[9]`: Tip rotation matrix, row-major
  - `u_tip[3]`: Tip curvature strain

**Total State Dimension:** `18·N + 15` where `N = NUM_ACT_SET` (number of coils)

**Warm-start Guesses** (optional, separate from state): `6·N` elements
- Per coil: `mL[3], nL[3]` (interface moments and forces from BVP solver)

**Evidence:**
```python
# python/control/true_legacy_state_adapter.py:32
STATE_DIM_PER_COIL = 18  # [v, w, p, R]
STATE_DIM_TIP = 15        # [p, R, u]
# Total: 18*N + 15
```

---

### 1.2 Packing/Unpacking Functions

**`pack_true_legacy_state(x_coil, xf)`**
Location: `python/control/true_legacy_state_adapter.py:50-136`

**Input:**
- `x_coil`: `[N, 18]` or `[B, N, 18]` (batched)
- `xf`: `[15]` or `[B, 15]` (batched)

**Output:** Flat packed state `[18·N + 15]` or `[B, 18·N + 15]`

**Packing order:** Coil states first (all N coils flattened), then tip state

**Memory layout:** `.contiguous()` guaranteed

**Shape validation:**
```python
if x_coil.shape[-1] != 18:
    raise ValueError(f"Expected x_coil[...18], got shape {x_coil.shape}")
if xf.shape[-1] != 15:
    raise ValueError(f"Expected xf[...15], got shape {xf.shape}")
```

---

**`unpack_true_legacy_state(x, n_act)`**
Location: `python/control/true_legacy_state_adapter.py:138-199`

**Input:** Packed state `[18·N + 15]` or `[B, 18·N + 15]`

**Output:** `(x_coil, xf)` tuple

**Dimension validation:**
```python
expected_dim = 18 * n_act + 15
if x.shape[-1] != expected_dim:
    raise ValueError(f"Expected state dim {expected_dim}, got {x.shape[-1]}")
```

---

**`pack_true_legacy_warmstart(mL_guess, nL_guess)`**
Location: `python/control/true_legacy_state_adapter.py:202-289`

Packs BVP warm-start guesses `[N, 3]` each → `[6·N]`

---

**`unpack_true_legacy_warmstart(w, n_act)`**
Location: `python/control/true_legacy_state_adapter.py:292-349`

Unpacks warm-start vector → `(mL_guess, nL_guess)`

---

**Error Handling:** Comprehensive shape validation with clear error messages showing expected vs. actual shapes.

**Batch Dimension Consistency:** Checks that `x_coil` and `xf` have same batch dimension (if batched).

---

## 2. FORWARD STEP FUNCTIONS

### 2.1 Main Entry Point

**`true_legacy_step(x, u, dt, n_act, catheter_params, warmstart, return_orientation)`**
Location: `python/control/true_legacy_step.py:23-180`

**Key Characteristics:**
- **Batching:** Processes batched inputs by looping over batch dimension (lines 125-173)
- **State validation:** Enforces correct shapes and dimensions
- **Dtype preservation:** Maintains original dtype and device
- **Warmstart support:** Optional BVP initial guess for faster convergence

**Caching Strategy:**
- Forward result includes `warmstart_next` dict with updated `mL_guess` and `nL_guess`
- **No automatic caching for VJP**—handled separately in autograd layer

**Single-Step Implementation:** `_true_legacy_step_single()` (lines 183-266)

**Conversion flow:**
1. PyTorch → NumPy (float64, C-contiguous)
2. C++ computation (calls `crm_diff_py.true_legacy_step_forward()`)
3. NumPy → PyTorch (restore dtype/device)

**C++ Binding Called:** `crm_diff_py.true_legacy_step_forward()`
Evidence: `python/crm_bindings.cpp:952-1206`

---

### 2.2 C++ Binding Workflow

**Forward Pass** (`crm_bindings.cpp:952-1206`):
1. Validate inputs (shape checks for `x_coil`, `xf`, `u`)
2. Unpack coil state (extract `v_L_pre`, `w_L_pre`, `p_pre`, `R_pre`)
3. Construct shooting params (`CRMDYNConstructShootingMethodParamSet()`)
4. Call BVP solver: `DynamicsBVP()` (line 1113)
   - Solves for interface forces/moments: `mL`, `nL`
   - Computes base curvature `u0`
5. Propagate dynamics: `DYNSolverIVP()` (line 1122)
   - Integrates rigid body motion with BVP solution
6. Package results (state, observables, warm-start guesses)

**Observable Extraction** (lines 1150-1180):
- `tip_p`: First 3 elements of `xf_next`
- `tip_R`: Elements 3-11 of `xf_next`
- `tip_u`: Elements 12-14 of `xf_next`

**Convergence Check** (line 1202):
```cpp
converged = (out_localmin == 0);
```

---

## 3. VJP AND BATCHED VJP IMPLEMENTATIONS

### 3.1 PyTorch Autograd Integration

**`TrueLegacyStepFn`** (autograd.Function)
Location: `python/control/true_legacy_step_autograd.py:22-177`

**Forward Pass** (lines 31-101):
- **Input requirements:** `x` and `u` must be `float64`
- **Caching:** Saves `(x, u)` and context info for backward
- **Convergence enforcement:** Raises `RuntimeError` if BVP fails (line 82)
- **Returns:** `tip_p` or `(tip_p, x_next)` based on `return_x_next` flag

**Backward Pass** (lines 104-177):
- **Gradient combination:** Handles gradients from both `tip_p` and `x_next` paths
- **Special handling:** `tip_p` appears in both observable and `xf_next[0:3]`
- **C++ VJP call:** `crm_diff_py.true_legacy_step_vjp()` (line 158)
- **Returns:** `(grad_x, grad_u, None, None, None, None, None)`—only differentiable w.r.t. state and control

**Evidence:**
```python
# python/control/true_legacy_step_autograd.py:158
vjp_result = crm_diff_py.true_legacy_step_vjp(
    x_coil_np, xf_np, u_np, dt, L_inserted, params_dict, grad_tip_p_np
)
```

---

### 3.2 C++ VJP Implementation

**Single VJP:** `crm_bindings.cpp:1209-1324`

**Workflow:**
1. Re-run forward (caches BVP solution in `TrueLegacyStepResult`)
2. Call batched backward with `num_rhs=1`

**Design note:** Single VJP delegates to batched code path for consistency.

---

**Batched VJP:** `crm_bindings.cpp:1326-1450`

**Function:** `py_true_legacy_step_vjp_batched()`

**Processes:** K cotangent vectors in one call

**Efficiency:** Amortizes cost of linear system factorization

**Use case:** Trajectory optimization (iLQR/MPC with TRUE legacy state)

**C++ Implementation:** `CRM_TrueLegacyDynamics.hpp::true_legacy_step_backward_batched()` (lines 71-81)

**Method:** Implicit function theorem on BVP solution (no backprop through iterations)

**Input:** `grad_tip_p_batch` `[num_rhs, 3]` (row-major)

**Output:**
- `grad_x_coil_batch` `[num_rhs, NUM_ACT_SET, 18]`
- `grad_xf_batch` `[num_rhs, NUM_STATES]`
- `grad_u_batch` `[num_rhs, NUM_ACT_SET, 3]`

**Cached Data Structure:** `TrueLegacyStepResult` (CRM_TrueLegacyDynamics.hpp:9-36)
- Stores full forward computation: inputs, outputs, BVP solution
- Enables implicit differentiation without re-solving BVP

---

## 4. OBSERVABLE HANDLING (NO LEAKAGE)

### 4.1 Critical Design Principle

**Observables are NOT part of state.**

### 4.2 Observable Flow

1. **Computation:** Derived from state during forward pass
2. **Return mechanism:** Separate `obs` dict in forward result
3. **State separation:** Tip position/orientation in `xf`, but also returned as observables

**Observables Dict** (`true_legacy_step.py:253-261`):
```python
obs = {
    'tip_p': [...],        # Tip position [3]
    'tip_R': [...],        # Tip orientation [9] (optional)
    'tip_u': [...],        # Tip curvature [3]
    'converged': bool,     # BVP solver status
    'warmstart_next': {    # Updated guesses for next step
        'mL_guess': [...],
        'nL_guess': [...]
    }
}
```

### 4.3 No Leakage Verification

**State contains `xf[15]` which includes `[p, R, u]`.**

**Observables are views/copies of state components**, not additional state.

**Gradient paths properly separate observable contributions from state contributions.**

**Evidence:** No code that modifies state based on observable values. Observables are read-only outputs.

---

## 5. DETERMINISM GUARANTEES

### 5.1 Randomness Analysis

**TRUE legacy step:** No randomness—BVP solver is fully deterministic.

**MPC controller:** Uses seeded random initialization (ONLY place found):
```python
# python/control/mpc.py:118
np.random.seed(42)
```

**Purpose:** Cold-start perturbation in trajectory optimization.

**Impact:** Deterministic across runs (fixed seed).

---

### 5.2 Ordering Guarantees

- **State layout:** Fixed row-major order (documented in contract)
- **BVP packing:** Deterministic order `[m_0, n_0, m_1, n_1, ...]`
- **No dict iteration:** All array operations use explicit indexing

---

### 5.3 Numerical Determinism

- **Solver tolerance:** Fixed at `1e-5` (from legacy contract)
- **Linear algebra:** LAPACK/BLAS calls are deterministic given same inputs
- **No parallelism:** Sequential execution in both forward and backward

---

## 6. ERROR HANDLING AND DIAGNOSTIC SURFACING

### 6.1 Python Layer Validation

**Input Validation** (`true_legacy_step.py:74-116`):
- State dimension checks (lines 84-88)
- Control shape validation (lines 91-106)
- Timestep positivity (line 109)
- Catheter params presence (lines 113-116)

**Error Types:**
- `ValueError`: Shape mismatches, invalid dimensions
- `RuntimeError`: Solver convergence failures

---

### 6.2 Diagnostic Information

**Convergence Status** (returned in observables):
- `converged`: Boolean flag (BVP solver success)
- `localmin`: Exit code from trust-region solver

**C++ Diagnostic Surfacing** (`crm_bindings.cpp:1202-1203`):
```cpp
result["converged"] = (out_localmin == 0);
result["localmin"] = out_localmin;
```

**VJP Diagnostics** (`CRM_TrueLegacyDynamics.hpp:63-64`):
- `lu_rank`: Rank of linear system (optional output)
- `rel_residual`: Solve residual norm (optional output)

---

### 6.3 Error Propagation

- C++ returns status codes (0=success)
- Python wrapper checks status and raises exceptions
- Full error context preserved in exception messages

**Example:**
```python
if not obs['converged']:
    raise RuntimeError(f"BVP solver failed: localmin={obs['localmin']}")
```

---

## 7. MAIN API ENTRYPOINTS

### 7.1 Public API

**File:** `python/control/__init__.py:45-58`

**State Adapter Functions:**
- `true_legacy_state_dim(n_act)` → Returns `18·N + 15`
- `pack_true_legacy_state(x_coil, xf)` → Packed state vector
- `unpack_true_legacy_state(x, n_act)` → `(x_coil, xf)` tuple
- `pack_true_legacy_warmstart(mL_guess, nL_guess)` → Packed guesses
- `unpack_true_legacy_warmstart(w, n_act)` → `(mL_guess, nL_guess)`

**Forward Step Functions:**
- `true_legacy_step(x, u, dt, ...)` → `(x_next, obs)` (pure forward, no autograd)
- `true_legacy_step_torch(x, u, dt, ...)` → `tip_p` or `(tip_p, x_next)` (with autograd)

**PyTorch Autograd:**
- `TrueLegacyStepFn` — Low-level `autograd.Function`

---

### 7.2 C++ Binding Relationship

**Python → C++ Call Chain:**
```
true_legacy_step()                        [Python wrapper]
  ↓
_true_legacy_step_single()                [Single-sample handler]
  ↓
crm_diff_py.true_legacy_step_forward()    [pybind11 binding]
  ↓
py_true_legacy_step_forward()             [C++ wrapper, crm_bindings.cpp:952]
  ↓
DynamicsBVP() → DYNSolverIVP()            [Core C++ dynamics]
```

**VJP Call Chain:**
```
TrueLegacyStepFn.backward()               [PyTorch autograd]
  ↓
crm_diff_py.true_legacy_step_vjp()        [pybind11 binding]
  ↓
py_true_legacy_step_vjp()                 [C++ wrapper, crm_bindings.cpp:1209]
  ↓
true_legacy_step_forward()                [Re-cache forward]
  ↓
true_legacy_step_backward_batched()       [Implicit VJP, num_rhs=1]
```

**Binding Constants** (`crm_bindings.cpp:53-57`):
- `CRM_PACKAGE_VERSION = "1.0.0"`
- `CRM_API_VERSION = "1.1.0"`
- Contract versions for equilibrium and dynamics

---

## 8. REDUCED 6D DYNAMICS (SEPARATE INTERFACE)

**Note:** The codebase ALSO implements a reduced 6D dynamics interface (NOT TRUE legacy).

**File:** `python/control/step_legacy_contract.py`

**State:** 6D (u_0[3], v_0[3])

**C++ Binding:** `crm_diff_py.dynamics_forward()`, `dynamics_backward()`, `dynamics_backward_batched()`

**Evidence:** Used extensively in MPC, iLQR, and training scripts (see evidence commands output).

**Correct Label:** REDUCED 6D (non-legacy)

**Relationship to TRUE legacy:**
- Independent implementation
- Different state dimension (6 vs. 18·N+15)
- Different C++ entrypoints (implicit Euler vs. DynamicsBVP/DYNSolverIVP)

---

## 9. KEY FINDINGS SUMMARY

| Finding | Status | Evidence |
|---------|--------|----------|
| TRUE legacy state contract (18·N+15) correctly implemented | ✅ Verified | `true_legacy_state_adapter.py` |
| Packing/unpacking robust with validation | ✅ Verified | Comprehensive shape/dtype checks |
| Forward step calls DynamicsBVP → DYNSolverIVP | ✅ Verified | `crm_bindings.cpp:952-1206` |
| VJP uses implicit differentiation (no FD) | ✅ Verified | `CRM_TrueLegacyDynamics.hpp:71-81` |
| Batched VJP for efficient optimization | ✅ Verified | `crm_bindings.cpp:1326-1450` |
| No observable leakage into state | ✅ Verified | Observables are views/copies only |
| Deterministic (except seeded MPC cold-start) | ✅ Verified | Seed=42 in `mpc.py:118` |
| Error handling comprehensive | ✅ Verified | Input validation + convergence checks |
| PyTorch autograd integration seamless | ✅ Verified | `TrueLegacyStepFn` in `true_legacy_step_autograd.py` |

---

## 10. SAFETY RECOMMENDATIONS FOR B0

### 10.1 Before System Identification

1. **Validate BVP convergence** at every forward step
2. **Check finite values** in state and gradients
3. **Monitor gradient norms** for explosion/vanishing
4. **Clip gradients if necessary** (document threshold)

### 10.2 Runtime Validation

**Example:**
```python
x_next, obs = true_legacy_step(x, u, dt, n_act, params, warmstart)

# Check convergence
if not obs['converged']:
    raise RuntimeError(f"BVP failed: localmin={obs['localmin']}")

# Check finite values
if not torch.all(torch.isfinite(x_next)):
    raise ValueError("Forward step produced NaN/Inf in state")

# Monitor gradient norms (during backward)
if grad_norm > 1e3:
    print(f"WARNING: Large gradient norm {grad_norm}")
```

---

## AUDIT STATUS

**Scope:** Python control layer for TRUE legacy dynamics
**Modifications Made:** NONE (audit-only)
**Code Modified:** NO
**Violations Found:** ZERO

**Completion:** ✅ ALL PYTHON CONTROL LAYER AUDITED

**Next Steps:**
- Proceed to Cross-Layer Risks and Mitigations
- Generate Completion Report

---

**END OF PYTHON CONTROL LAYER AUDIT**
