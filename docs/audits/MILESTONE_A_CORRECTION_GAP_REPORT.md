# Milestone A Correction Gap Report

**Document Type**: Gap Analysis & Correction Prescription
**Date**: 2026-01-03
**Status**: A0 — Contract Extraction Complete

---

## Executive Summary

The current "Milestone A Complete" claim is **INVALID**.

The implementation in `python/control/` does not match the legacy hybrid state contract defined in `src/CRM_DiffDynamics.hpp`. This document itemizes the violations and prescribes corrective actions for future work.

**Ground Truth**: `src/CRM_DiffDynamics.hpp` (legacy C++ implementation)
**Current Impl**: `python/control/hybrid_state_contract.py`, `python/control/step_hybrid_legacy_contract.py`

---

## 1. Mismatch Table

| Aspect | Legacy Contract | Current Implementation | Gap |
|--------|-----------------|------------------------|-----|
| **State dimensionality** | 6D | 9D | +3 elements (p_tip wrongly included) |
| **State contents** | [u_0, v_0] only | [u_0, v_0, p_tip] | Observable mixed into state |
| **Dynamic/QS split** | u_0,v_0 dynamic; p_tip,u_tip quasi-static | Conflated in single vector | Violates hybrid semantics |
| **Tip pose handling** | Output observable | Part of state vector | Misclassified |
| **Gradient method** | Implicit (cached Jacobians via `dynamics_backward`) | Finite differences | FD is O(n) slower, less accurate |
| **Batched VJP** | `dynamics_backward_batched` exists | Not used | Missing performance feature |
| **Theta parameter** | Not in legacy | Defined but not implemented | Non-functional claim |

---

## 2. Violations Detail

### V1: State Dimensionality Mismatch

**Legacy**: 6D state `x[6] = [u_0(3), v_0(3)]`
**Current**: 9D state `x_hybrid[9] = [u_0(3), v_0(3), p_tip(3)]`

**Location**: `python/control/hybrid_state_contract.py:33-36`
```python
STATE_DIM_DYNAMICS = 6    # Legacy CP2 state: [u_0, v_0]
STATE_DIM_OBSERVABLE = 3  # Tip position p_tip
STATE_DIM_HYBRID = 9      # Full hybrid state  <-- VIOLATION
```

**Why it's wrong**: The legacy contract explicitly stores observables (p_tip, u_tip) in `DynamicsStepResult` as outputs, not as state elements. Including p_tip in the state vector:
1. Breaks the dynamic/quasi-static split
2. Creates redundancy (p_tip is computable from u_0)
3. Causes gradient shape mismatches

---

### V2: Observable Embedded in State

**Legacy**: `p_tip` is a field of `DynamicsStepResult`, not part of `x_next[6]`
**Current**: `p_tip` is the 7th-9th elements of `x_hybrid`

**Location**: `python/control/hybrid_state_contract.py:49-103` (HybridState class)
```python
@dataclass
class HybridState:
    u_0: np.ndarray    # (3,) float64
    v_0: np.ndarray    # (3,) float64
    p_tip: np.ndarray  # (3,) float64  <-- SHOULD BE OUTPUT ONLY
```

**Evidence from legacy** (`src/CRM_DiffDynamics.hpp:11-16`):
```cpp
double x_next[6];    // Next state: [u_0_{t+1}, v_0_{t+1}]  <-- 6D only
double p_tip[3];     // Tip position at t+1 (mm)  <-- SEPARATE OUTPUT
```

---

### V3: Finite-Difference VJP

**Legacy**: Uses cached Jacobians in `dynamics_backward()` for implicit VJP
**Current**: Uses finite differences with eps=1e-6

**Location**: `python/control/step_hybrid_legacy_contract.py:141-231`
```python
def vjp_hybrid_legacy_contract(..., eps: float = 1e-6):
    # ... uses FD perturbations
    for i in range(STATE_DIM_DYNAMICS):
        x_plus = x_t_hybrid.copy()
        x_plus[i] += eps
        # ... call forward twice
```

**Why it's wrong**:
1. FD is O(n) forward calls per VJP (vs O(1) for implicit)
2. FD has numerical precision issues (eps choice is problem-dependent)
3. Legacy has `dynamics_backward()` that uses cached Jacobians — should use it

**Evidence**: `src/CRM_DiffDynamics.hpp:59-67` shows analytical backward pass exists.

---

### V4: Missing Batched VJP

**Legacy**: `dynamics_backward_batched()` for computing K VJPs efficiently
**Current**: Not exposed, not used

**Location**: Not present in `python/control/step_hybrid_legacy_contract.py`

**Evidence from legacy** (`src/CRM_DiffDynamics.hpp:75-84`):
```cpp
int dynamics_backward_batched(
    const DynamicsStepResult& fwd_result,
    const double* V,     // K adjoint vectors (K×6)
    int K,
    ...
);
```

**Impact**: Jacobian computation is K× slower than necessary for trajectory optimization.

---

### V5: Non-Functional Theta Scaling

**Claim**: `theta[3] = [damping_scale, stiffness_scale, mass_scale]` scales physics matrices
**Reality**: Theta is validated but never applied to M, D, K

**Location**: `python/control/step_hybrid_legacy_contract.py:32-77`
```python
def step_hybrid_legacy_contract(
    x_t_hybrid, u_t, dt, L_inserted, theta, params_dict
):
    # theta is validated (line 65)
    # theta is NEVER PASSED to dynamics_forward
    result = crm_diff_py.dynamics_forward(
        x_t_np, u_t_np, dt, L_inserted, params_dict
    )  # <-- theta not used
```

**Impact**: grad_theta is always ~0 (tests pass trivially, feature doesn't work).

---

### V6: False Parity Claim

**Location**: `docs/audits/MILESTONE_A_HYBRID_CONTRACT_VJP.md`

The document claims "COMPLETE" status with:
- "Hybrid state contract successfully combines 6D dynamics with 3D observable"
- "Implicit VJP support validated"

**Reality**:
- State is 9D (not "6D + 3D observable" as separate output)
- VJP uses finite differences (not implicit differentiation)
- Tests pass because they don't verify the broken features

---

## 3. Prescribed Corrective Actions

These are prescriptions for **future work** (not implemented in A0):

| Violation | File | Action |
|-----------|------|--------|
| V1: 9D state | `hybrid_state_contract.py` | Rename to `hybrid_state_contract_prototype.py`; create true 6D contract |
| V2: p_tip in state | `hybrid_state_contract.py` | Remove p_tip from HybridState; return as separate output |
| V3: FD VJP | `step_hybrid_legacy_contract.py` | Replace with `dynamics_backward()` wrapper |
| V4: No batched VJP | `step_hybrid_legacy_contract.py` | Add `dynamics_backward_batched` Python binding |
| V5: Broken theta | `step_hybrid_legacy_contract.py` | Either implement scaling or remove theta parameter |
| V6: False claim | `MILESTONE_A_HYBRID_CONTRACT_VJP.md` | Retract "COMPLETE"; mark as prototype |

### Gating Recommendation

Until corrected, gate existing code behind flag:
```python
if os.environ.get("HYBRID_PROTOTYPE") == "1":
    from python.control.hybrid_state_contract import *  # prototype
else:
    raise ImportError("Hybrid contract not yet implemented. Set HYBRID_PROTOTYPE=1 for prototype.")
```

---

## 4. Required Commands Output

### Current Branch Patterns
```bash
$ rg -n "Hybrid|BVP|Equilibrium|Dynamics|step|Advance" -S .
# [output omitted for brevity — run to verify]
```

### Legacy Evidence Extraction
```bash
$ rg -n "struct DynamicsStepResult" src/
src/CRM_DiffDynamics.hpp:9:struct DynamicsStepResult {

$ rg -n "double x_next" src/CRM_DiffDynamics.hpp
src/CRM_DiffDynamics.hpp:11:    double x_next[6];

$ rg -n "dynamics_backward" src/
src/CRM_DiffDynamics.hpp:59:int dynamics_backward(
src/CRM_DiffDynamics.hpp:75:int dynamics_backward_batched(
```

---

## 5. Conclusion

**Milestone A is NOT complete.**

The current implementation is a **prototype** that:
- Uses an incorrect 9D state representation
- Uses finite differences instead of implicit differentiation
- Has non-functional theta scaling
- Does not use the available batched VJP

This gap report documents these violations without making changes. Future work must address each violation before claiming parity with legacy.

---

## Appendix: File Locations

| Purpose | File | Lines |
|---------|------|-------|
| Legacy contract (ground truth) | `src/CRM_DiffDynamics.hpp` | 1-88 |
| Equilibrium contract | `src/CRM_DiffEquilibrium.hpp` | 1-50 |
| Current 9D state (violation) | `python/control/hybrid_state_contract.py` | 33-103 |
| Current FD VJP (violation) | `python/control/step_hybrid_legacy_contract.py` | 141-231 |
| False completion claim | `docs/audits/MILESTONE_A_HYBRID_CONTRACT_VJP.md` | entire |
