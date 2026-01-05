# Reduced 6D Dynamics Archive Notice

**Date Archived:** 2026-01-04
**Status:** ARCHIVED (Non-Legacy)

---

## Overview

The reduced 6D dynamics implementation has been **ARCHIVED** as it is **NOT** the TRUE legacy implementation.

This archive preserves the reduced 6D code for historical reference and backward compatibility, but it should not be used for new development.

---

## Location

- **Python Code:** `python/control/reduced6d/`
- **Tests:** `python/test/reduced6d/`
- **Documentation:** `docs/archive/reduced6d/`

---

## Why Archived?

The 6D state (u_0[3], v_0[3]) is a **simplified approximation**, not the TRUE legacy dynamics used in the main branch.

### Differences

| Aspect | Reduced 6D (Archived) | TRUE Legacy (Current) |
|--------|----------------------|----------------------|
| State dimension | 6 (u_0[3], v_0[3]) | 18·N+15 (full rigid body) |
| Physics | Implicit Euler, simplified | DynamicsBVP → DYNSolverIVP |
| Rotation | Curvature-based | Full rotation matrices |
| Coil dynamics | Reduced to base | Explicit per-coil state |
| Solver | Simplified linearization | Full BVP/IVP hybrid solver |

**TRUE legacy** uses the full 18·N+15 state from the hybrid BVP/IVP solver pipeline, matching the `main` branch implementation.

---

## What Was Moved

### Python Files
- `python/control/legacy_state.py` → `python/control/reduced6d/legacy_state.py`
- `python/control/legacy_state_adapter.py` → `python/control/reduced6d/legacy_state_adapter.py`
- `python/control/step_legacy_contract.py` → `python/control/reduced6d/step_legacy_contract.py`

### Test Files
- `python/test_a1_legacy_state_adapter.py` → `python/test/reduced6d/test_a1_legacy_state_adapter.py`
- `python/test_a1_step_legacy_contract_forward.py` → `python/test/reduced6d/test_a1_step_legacy_contract_forward.py`
- `python/test_a2_implicit_vjp_gradcheck.py` → `python/test/reduced6d/test_a2_implicit_vjp_gradcheck.py`
- `python/test_a3_batched_vjp.py` → `python/test/reduced6d/test_a3_batched_vjp.py`

### Documentation
- `docs/contracts/LEGACY_HYBRID_STATE_CONTRACT.md` → `docs/archive/reduced6d/contracts/LEGACY_HYBRID_STATE_CONTRACT.md`
- `docs/audits/LEGACY_CONTRACT_EXTRACTION_COMMAND_LOG.md` → `docs/archive/reduced6d/audits/LEGACY_CONTRACT_EXTRACTION_COMMAND_LOG.md`
- `docs/audits/LEGACY_VS_CURRENT_DIFFSIM_DELTA.md` → `docs/archive/reduced6d/audits/LEGACY_VS_CURRENT_DIFFSIM_DELTA.md`

---

## Accessing Archived Code

The reduced 6D code is still functional and can be imported:

```python
# Emits DeprecationWarning
from python.control.reduced6d import (
    LegacyState,
    pack_legacy_state,
    unpack_legacy_state,
    step_legacy_contract,
    vjp_legacy_contract
)
```

**Warning:** You will receive a `DeprecationWarning` when importing from `reduced6d`:
```
DeprecationWarning: Reduced6D dynamics is NON-legacy and archived.
Use true_legacy_step for TRUE legacy (18·N+15) dynamics.
```

---

## Migration Path

**For new code:**
Use TRUE legacy dynamics from `python/control/true_legacy_step.py`.

**Migration guide:**
See `docs/migrations/MIGRATE_TO_TRUE_LEGACY_MPC_ILQR.md`

**Example migration:**

**OLD (6D):**
```python
from python.control import step_legacy_contract, LegacyState

x0 = np.array([0, 0, 0, 0, 0, 0])  # 6D state
x_next, obs = step_legacy_contract(x0, u, dt, L_inserted, params)
```

**NEW (TRUE Legacy):**
```python
from python.control import true_legacy_step, pack_true_legacy_state

n_act = 1
x_coil = np.zeros((n_act, 18))  # Per-coil: [v, w, p, R]
xf = np.zeros(15)                # Tip: [p, R, u]
x0 = pack_true_legacy_state(x_coil, xf)  # 18*n_act+15 state

x_next, obs = true_legacy_step(
    torch.from_numpy(x0),
    torch.from_numpy(u),
    dt, n_act=n_act,
    catheter_params=params,
    L_inserted=L_inserted
)
```

---

## Testing Archived Code

The archived tests still pass:

```bash
PYTHONPATH=build:python pytest python/test/reduced6d/ -v
```

This ensures the archived implementation remains functional for backward compatibility.

---

## Historical Context

The reduced 6D implementation was created as a simplified dynamics model for faster computation during early development. It served its purpose well for prototyping MPC and iLQR controllers.

However, it was incorrectly labeled as "legacy" when in fact the TRUE legacy implementation (18·N+15) from the `main` branch is the authoritative ground truth.

This archive corrects that naming and establishes TRUE legacy as the default.

---

## Related Documentation

- **TRUE Legacy Contract:** `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md`
- **Migration Guide:** `docs/migrations/MIGRATE_TO_TRUE_LEGACY_MPC_ILQR.md`
- **Archive README:** `docs/archive/reduced6d/README.md`
- **PRE-B0 Audit:** `docs/audits/PRE_B0_COMPLETION_REPORT.md`

---

**Archive Date:** 2026-01-04
**Archived By:** Migration to TRUE Legacy Dynamics
**Status:** Preserved for historical reference and backward compatibility
