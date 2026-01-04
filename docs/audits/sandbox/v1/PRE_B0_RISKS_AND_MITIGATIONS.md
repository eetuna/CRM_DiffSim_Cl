# PRE-B0 RISKS AND MITIGATIONS (CORRECTED)

**Audit Date:** 2026-01-03
**Auditor:** Claude (Sonnet 4.5)
**Scope:** Cross-layer gradient & stability analysis
**Current Branch:** `milestone-a-hybrid-vjp`
**Authority:** `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md`

**CORRECTION:** This document supersedes the previous version which incorrectly labeled 6D state as "legacy"

---

## Executive Summary (CORRECTED)

This document consolidates all identified risks for **BOTH** state paths:
1. **TRUE legacy (18·N+15)** — Full rigid-body + flexible tip state
2. **Reduced 6D (u_0, v_0)** — Reduced dynamics state

**CRITICAL FINDING:** The current B0-intended path uses **REDUCED 6D**, NOT TRUE legacy (18·N+15).

**Risk Categories:**
1. **State Contract Mismatch** (NEW RISK - CRITICAL)
2. **Convergence Failures** (BVP / IVP solver non-convergence)
3. **NaN Propagation** (silent numerical instability)
4. **Rank Deficiency** (gradient explosion from singular Jacobians)
5. **Exploding / Vanishing Gradients** (extreme gradient magnitudes)
6. **Solver Residual Sensitivity** (tolerance-dependent behavior)
7. **Extreme Input Regimes** (dt, L_inserted, actuation edge cases)

**Overall Risk Level:** **MEDIUM** (for reduced 6D with mitigations)
**Overall Risk Level:** **UNKNOWN** (for TRUE legacy 18·N+15 - requires full audit)

**GO / NO-GO:** **CONDITIONAL**
- GO for reduced 6D (with prescribed mitigations)
- NO-GO for TRUE legacy (requires additional audit + mitigations)

---

## 1. STATE CONTRACT MISMATCH RISK (NEW - CRITICAL)

### 1.1 Conflation of "Legacy" Label

**Risk:** Using term "legacy" for two distinct state representations causes confusion

**Source:**
- `python/control/__init__.py:7,25` labels 6D as "Legacy 6D state"
- TRUE legacy is 18·N+15 per `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md:142`

**Impact:**
- System ID may use wrong state contract
- Comparison with main branch system ID invalid
- Gradient correctness cannot be verified if state mismatch

**Risk Level:** **CRITICAL**

**Required Mitigation (B0):**
```python
# CLARIFY which path is being used for B0
if B0_INTENT == "TRUE_LEGACY_PARITY":
    # Use true_legacy_step_torch (18·N+15)
    from control import true_legacy_step_torch
    state_dim = true_legacy_state_dim(n_act)  # 18*n_act + 15
else:
    # Use reduced 6D path (explicitly labeled)
    from control import step_legacy_contract  # REDUCED 6D (not TRUE legacy)
    state_dim = STATE_DIM_LEGACY  # 6
```

### 1.2 Parity Validation Risk

**Risk:** If B0 requires parity with main branch system ID but uses wrong state contract

**Source:**
- Main branch system ID uses DynamicsBVP → DYNSolverIVP (18·N+15)
- Current B0 path uses reduced 6D (does NOT call DynamicsBVP/DYNSolverIVP)

**Evidence:**
```bash
$ rg "DynamicsBVP|DYNSolverIVP" src/CRM_DiffDynamics.cpp
(no matches - reduced 6D path uses different wrappers)

$ rg "DynamicsBVP|DYNSolverIVP" src/CRM_TrueLegacyDynamics.cpp
src/CRM_TrueLegacyDynamics.cpp:104:    // Call DynamicsBVP
src/CRM_TrueLegacyDynamics.cpp:105:    DynamicsBVP(shooting_params, xf, mL_guess_arr, ...
src/CRM_TrueLegacyDynamics.cpp:110:    // Call DYNSolverIVP
src/CRM_TrueLegacyDynamics.cpp:112:    DYNSolverIVP(shooting_params, out.u0, ...
```

**Impact:**
- B0 system ID results NOT comparable to main branch
- Identified parameters may be incorrect (state space mismatch)

**Risk Level:** **CRITICAL** (if parity required)

**Required Mitigation (B0):**
1. **Clarify B0 intent:**
   - Does B0 require parity with main branch (TRUE legacy)?
   - Or is reduced 6D approximation acceptable?

2. **IF parity required:**
   - Switch to `true_legacy_step_torch` path
   - Re-audit TRUE legacy VJP implementation
   - Implement mitigations for 18·N+15 state

3. **IF reduced 6D acceptable:**
   - Document explicitly that B0 uses REDUCED 6D (non-legacy)
   - Acknowledge NO parity with main branch system ID

---

## 2. CONVERGENCE FAILURE RISKS

### 2.1 BVP Solver Non-Convergence

**Source:** `legacy_worktree/src/CRM_BVPSolver.cpp:81`

**Failure Mode:**
- Trust-region dogleg solver cannot satisfy `TRUSTREGION_TOLERANCE`
- Returns `out_localmin != 0` (stuck at local minimum)

**Affected Paths:**
- TRUE legacy (18·N+15): **YES** (calls DynamicsBVP directly)
- Reduced 6D: **YES** (calls equilibrium_forward which uses BVP internally)

**Impact:**
- Invalid `deltau0` and `ftip` returned (garbage values)
- Forward pass produces invalid `x_next`

**Risk Level:** **HIGH** (if status not checked)

**Required Mitigation (B0 - BOTH paths):**
```python
result = step_legacy_contract(state, u_t, dt, L, params)  # or true_legacy_step(...)
if not result.success:  # Check status flag
    logging.error(f"Step failed: {result.diagnostics}")
    raise RuntimeError("Forward step failed, cannot continue trajectory")
```

**Optional Mitigation:**
- Adaptive parameter initialization (warm-start from previous step)
- Fallback to relaxed tolerance on failure
- Rejection sampling (retry with perturbed initial guess)

### 2.2 IVP Integration Divergence

**Source:** `legacy_worktree/src/CRM_IVPSolver.cpp:413` (ABM4 integration)

**Failure Mode:**
- Numerical integration produces NaN/Inf (timestep too large, stiff system)
- No explicit NaN check in legacy IVP code

**Affected Paths:**
- TRUE legacy (18·N+15): **YES** (integrates rigid body + flexible dynamics)
- Reduced 6D: **YES** (integrates flexible dynamics for equilibrium)

**Impact:**
- Invalid final state `out_x_N`
- NaN propagates to Python layer

**Risk Level:** **MEDIUM** (uncommon with reasonable dt)

**Required Mitigation (B0 - BOTH paths):**
```python
if not np.all(np.isfinite(result.x_next.to_numpy())):
    raise ValueError(f"Forward step produced NaN/Inf, dt={dt}, L={L_inserted}")
```

---

## 3. NAN PROPAGATION RISKS

### 3.1 Coil Dynamics Unbounded Growth

**Source:** `legacy_worktree/src/CoilDynamics_Defs.cpp:161`

**Failure Mode:**
- RK2 / ABM4 integration produces NaN in rigid body dynamics
- `isnan(x_n[0])` check present but **no exit** (commented out)

**Affected Paths:**
- TRUE legacy (18·N+15): **YES** (integrates rigid body states)
- Reduced 6D: **NO** (does not integrate rigid body states)

**Impact:**
- NaN propagates to caller (affects x_coil component of 18·N+15 state)

**Risk Level:** **LOW** (only affects TRUE legacy path, not used if B0 uses reduced 6D)

**Required Mitigation (B0 - TRUE legacy only):**
```python
# For TRUE legacy path only
if not np.all(np.isfinite(x_coil)):
    raise ValueError("Rigid body state contains NaN/Inf")
```

### 3.2 Silent NaN in Observables

**Source:** `python/crm_bindings.cpp:295-298`

**Failure Mode:**
- `p_tip` or `u_tip` contains NaN (invalid equilibrium solution)
- Python layer receives NaN observables

**Affected Paths:**
- TRUE legacy (18·N+15): **YES**
- Reduced 6D: **YES**

**Impact:**
- Objective function J(x, u) evaluates to NaN
- Optimizer diverges

**Risk Level:** **MEDIUM**

**Required Mitigation (B0 - BOTH paths):**
```python
if not np.all(np.isfinite(result.observables['p_tip'])):
    raise ValueError("Observable p_tip contains NaN/Inf")
if not np.all(np.isfinite(result.observables['u_tip'])):
    raise ValueError("Observable u_tip contains NaN/Inf")
```

---

## 4. RANK DEFICIENCY RISKS

### 4.1 Equilibrium Jacobian Rank Deficiency

**Source:** `src/CRM_IVPJacobian.cpp:95-100` (current code)

**Fixed in Current Code:**
```cpp
Eigen::FullPivLU<Matrix3d> lu_u_u0(JIVP_u_u0);
if (lu_u_u0.rank() < 3) {
    std::cerr << "ERROR: JIVP_u_u0 rank-deficient, rank=" << lu_u_u0.rank() << std::endl;
    // Fail-fast: return zero Jacobians
    MatrixXd zero_pz = MatrixXd::Zero(3, Cs + 1);
    // ...
```

**Affected Paths:**
- TRUE legacy (18·N+15): **YES** (uses equilibrium Jacobians for VJP)
- Reduced 6D: **YES** (uses equilibrium Jacobians for VJP)

**Risk Level:** **MEDIUM** (fixed in current code, but still surfaced)

**Required Mitigation (B0 - BOTH paths):**
```python
if vjp_result.diagnostics['lu_rank'] < 6:
    logging.warning("Rank-deficient Jacobian detected")
    # Degrade gracefully: zero gradients or clip
    grad_x_t = np.zeros(6)  # or np.zeros(18*n_act + 15) for TRUE legacy
    grad_u_t = np.zeros(3)
```

---

## 5. EXPLODING / VANISHING GRADIENT RISKS

### 5.1 Exploding Gradients

**Source:** Implicit differentiation sensitivity to Jacobian condition number

**Failure Mode:**
- Poorly conditioned A matrix (cond(A) » 1)
- Solve A^T λ = v produces large λ
- Gradients scale with ||λ|| · ||B||

**Affected Paths:**
- TRUE legacy (18·N+15): **YES** (higher dimensional state → potentially worse conditioning)
- Reduced 6D: **YES**

**Indicators:**
- `rel_residual` large (> 1e-6)
- `lu_rank` < expected rank (6 for reduced, depends on n_act for TRUE legacy)
- Gradient norm ||grad_x_t|| > 1e3

**Risk Level:** **MEDIUM**

**Required Mitigation (B0 - BOTH paths, adjusted for state dim):**
```python
# Gradient norm clipping
grad_norm_x = np.linalg.norm(vjp_result.grad_x_t)
grad_norm_u = np.linalg.norm(vjp_result.grad_u_t)

CLIP_THRESHOLD = 1e3
if grad_norm_x > CLIP_THRESHOLD:
    logging.warning(f"Clipping grad_x_t, norm={grad_norm_x:.2e}")
    vjp_result.grad_x_t *= CLIP_THRESHOLD / grad_norm_x

if grad_norm_u > CLIP_THRESHOLD:
    logging.warning(f"Clipping grad_u_t, norm={grad_norm_u:.2e}")
    vjp_result.grad_u_t *= CLIP_THRESHOLD / grad_norm_u
```

---

## 6. MITIGATION PRIORITY MATRIX (CORRECTED)

### For Reduced 6D Path (Current B0 Path)

| Risk | Severity | Likelihood | Priority | Required for B0? |
|------|----------|-----------|----------|------------------|
| State contract mismatch | **CRITICAL** | High | **P0** | ✅ Clarify intent |
| BVP Non-Convergence | High | Medium | **P0** | ✅ Yes |
| IVP NaN Propagation | High | Low | **P1** | ✅ Yes |
| Observable NaN | High | Low | **P1** | ✅ Yes |
| Rank-Deficient Jacobian | Medium | Medium | **P1** | ✅ Yes |
| Exploding Gradients | High | Medium | **P0** | ✅ Yes |
| Vanishing Gradients | Low | Low | **P3** | ❌ No |
| Large Solve Residual | Medium | Low | **P2** | ✅ Yes |
| Extreme Actuation | Medium | Medium | **P2** | ✅ Yes |

### For TRUE Legacy (18·N+15) Path (If Used)

| Risk | Severity | Likelihood | Priority | Additional Audit Required? |
|------|----------|-----------|----------|----------------------------|
| State contract mismatch | **CRITICAL** | High | **P0** | ❌ (clarify only) |
| Coil Dynamics NaN | High | Low | **P1** | ✅ Yes (rigid body specific) |
| BVP Non-Convergence | High | Medium | **P0** | ❌ (same as 6D) |
| Observable NaN | High | Low | **P1** | ❌ (same as 6D) |
| Rank-Deficient Jacobian | Medium | Medium | **P1** | ✅ Yes (higher dim) |
| Exploding Gradients | High | **High** | **P0** | ✅ Yes (18·N+15 state) |
| VJP Correctness | **CRITICAL** | Unknown | **P0** | ✅ Yes (BVPJacobian.cpp) |

**P0:** Blocking (must implement before B0)
**P1:** High priority (implement in first B0 iteration)
**P2:** Medium priority (implement before full system ID)
**P3:** Low priority (optional enhancement)

---

## 7. RECOMMENDED MITIGATION IMPLEMENTATION (CORRECTED)

### 7.1 P0 Mitigations (Blocking) - Reduced 6D Path

**File:** `python/control/safe_step.py` (NEW)

```python
import numpy as np
import logging
from .step_legacy_contract import step_legacy_contract, vjp_legacy_contract

def safe_step_legacy_contract(x_t, u_t, dt, L_inserted, params, clip_gradients=True):
    """
    Safe wrapper with status checking and NaN validation for REDUCED 6D path.

    IMPORTANT: This uses REDUCED 6D state (u_0, v_0), NOT TRUE legacy (18·N+15).
    """
    # Saturate actuation
    u_t = np.clip(u_t, -1.0, 1.0)

    # Forward pass
    result = step_legacy_contract(x_t, u_t, dt, L_inserted, params)

    # P0: Status checking
    if not result.success:
        raise RuntimeError(f"Forward step failed: {result.diagnostics}")

    # P0: NaN validation (state)
    if not np.all(np.isfinite(result.x_next.to_numpy())):
        raise ValueError("Forward step produced NaN/Inf in x_next")

    # P0: NaN validation (observables)
    if not np.all(np.isfinite(result.observables['p_tip'])):
        raise ValueError("Forward step produced NaN/Inf in p_tip")

    return result

def safe_vjp_legacy_contract(fwd_result, grad_x_next, params, clip_threshold=1e3):
    """
    Safe VJP wrapper with gradient clipping for REDUCED 6D path.
    """
    vjp_result = vjp_legacy_contract(fwd_result, grad_x_next, params)

    # P0: Gradient clipping
    grad_norm_x = np.linalg.norm(vjp_result.grad_x_t)
    grad_norm_u = np.linalg.norm(vjp_result.grad_u_t)

    if grad_norm_x > clip_threshold:
        logging.warning(f"Clipping grad_x_t, norm={grad_norm_x:.2e}")
        vjp_result.grad_x_t *= clip_threshold / grad_norm_x

    if grad_norm_u > clip_threshold:
        logging.warning(f"Clipping grad_u_t, norm={grad_norm_u:.2e}")
        vjp_result.grad_u_t *= clip_threshold / grad_norm_u

    return vjp_result
```

### 7.2 P0 Mitigations (Blocking) - TRUE Legacy (18·N+15) Path

**File:** `python/control/safe_true_legacy_step.py` (NEW - if using TRUE legacy)

```python
import torch
import numpy as np
import logging
from .true_legacy_step_autograd import true_legacy_step_torch

def safe_true_legacy_step_torch(x, u, dt, *, n_act, catheter_params, L_inserted=100.0,
                                 clip_threshold=1e3):
    """
    Safe wrapper for TRUE legacy (18·N+15) with status checking and NaN validation.

    IMPORTANT: This uses TRUE legacy state (18·N+15), NOT reduced 6D.
    """
    # Validate state dimension
    expected_dim = 18 * n_act + 15
    if x.shape[-1] != expected_dim:
        raise ValueError(f"State dimension must be {expected_dim}, got {x.shape[-1]}")

    # Forward pass with autograd
    result = true_legacy_step_torch(x, u, dt, n_act=n_act,
                                    catheter_params=catheter_params,
                                    L_inserted=L_inserted, return_x_next=True)
    tip_p, x_next = result

    # P0: NaN validation (rigid body + tip state)
    if not torch.all(torch.isfinite(x_next)):
        raise ValueError("TRUE legacy forward step produced NaN/Inf in x_next (18·N+15)")

    # P0: NaN validation (observable)
    if not torch.all(torch.isfinite(tip_p)):
        raise ValueError("TRUE legacy forward step produced NaN/Inf in tip_p")

    # Note: Gradient clipping happens in backward pass (TrueLegacyStepFn.backward)
    # Consider adding gradient norm monitoring here if needed

    return tip_p, x_next
```

---

## 8. TESTING REQUIREMENTS FOR B0 (CORRECTED)

Before starting B0 system identification, the following tests must pass:

### For Reduced 6D Path

1. **Status validation test:** Verify exception raised on BVP failure
2. **NaN validation test:** Verify exception raised on NaN state/observable
3. **Gradient clipping test:** Verify gradients clipped to threshold
4. **Rank-deficiency test:** Verify graceful degradation on rank-deficient Jacobian
5. **Residual monitoring test:** Verify warnings logged for large residuals
6. **Regression test:** Verify backward gradients match finite differences (small dt)

**Test file:** `python/test_b0_safety_gates_reduced6d.py`

### For TRUE Legacy (18·N+15) Path (If Used)

1. **All tests above** (adjusted for 18·N+15 state dimension)
2. **Rigid body NaN test:** Verify exception raised on NaN in coil states
3. **BVP Jacobian correctness:** Verify VJP against finite differences
4. **Parity test:** Verify forward pass matches DynamicsBVP → DYNSolverIVP

**Test file:** `python/test_b0_safety_gates_true_legacy.py`

---

## 9. GO / NO-GO DECISION (CORRECTED)

### Decision Table

| B0 Intent | State Contract | Current Stack | Decision | Mitigations Required |
|-----------|----------------|---------------|----------|---------------------|
| TRUE legacy parity | 18·N+15 | Uses 6D reduced | **NO-GO** | Switch to TRUE legacy path + audit VJP |
| Reduced dynamics OK | 6D (u_0, v_0) | Uses 6D reduced | **GO** | P0 mitigations (status, NaN, clipping) |

### Hard Requirements for GO (Reduced 6D Path)

1. ✅ Clarify that B0 uses **REDUCED 6D**, not TRUE legacy
2. Implement `safe_step_legacy_contract` with status + NaN checks
3. Implement `safe_vjp_legacy_contract` with gradient clipping
4. Pass all 6 safety gate tests for reduced 6D

**Timeline:** 2 days

### Hard Requirements for GO (TRUE Legacy Path)

1. Switch to `true_legacy_step_torch` path
2. Re-audit `src/CRM_BVPJacobian.cpp` (BVP adjoint Jacobians)
3. Implement `safe_true_legacy_step_torch` with status + NaN checks
4. Verify VJP correctness against finite differences
5. Pass all 8 safety gate tests for TRUE legacy

**Timeline:** 5 days

---

## 10. SUMMARY OF CORRECTED RISKS

**NEW RISK (CRITICAL):**
- State contract mismatch (6D labeled "legacy" vs TRUE legacy 18·N+15)

**RISKS APPLYING TO BOTH PATHS:**
- BVP convergence failures
- Observable NaN propagation
- Rank-deficient Jacobians
- Exploding gradients

**RISKS SPECIFIC TO TRUE LEGACY (18·N+15):**
- Rigid body dynamics NaN propagation
- Higher-dimensional gradient explosion
- BVP Jacobian VJP correctness (requires additional audit)

**RECOMMENDED DECISION:**
- **IF B0 requires parity with main branch:** NO-GO → switch to TRUE legacy
- **IF B0 accepts reduced 6D:** GO → implement P0 mitigations for 6D

---

**END OF RISK ASSESSMENT (CORRECTED)**

**SUPERSEDES:** Previous version of this document dated 2026-01-03
**AUTHORITY:** `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md`
**CORRECTION NOTE:** `docs/audits/PRE_B0_CORRECTION_NOTE.md`
