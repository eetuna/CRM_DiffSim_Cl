# Reduced 6D Dynamics (ARCHIVED - NON-LEGACY)

This directory contains the archived **reduced 6D state dynamics** implementation.

**IMPORTANT:** This is NOT the TRUE legacy implementation.

- **State:** 6D (u_0[3], v_0[3])
- **Physics:** Implicit Euler with simplified model
- **Status:** ARCHIVED - do not use for new code

## TRUE Legacy Implementation

For TRUE legacy dynamics (18·N+15 state), use:
- `python/control/true_legacy_step.py`
- `python/control/true_legacy_state_adapter.py`
- Contract: `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md`

## Contents

- `contracts/` - 6D state contract specification
- `audits/` - Historical audit documents

## Why Archived?

The 6D state (u_0[3], v_0[3]) is a **simplified approximation**, not the TRUE legacy dynamics used in the main branch.

**TRUE legacy** uses the full 18·N+15 state from the hybrid BVP/IVP solver pipeline (DynamicsBVP → DYNSolverIVP).

## Migration

See: `docs/migrations/MIGRATE_TO_TRUE_LEGACY_MPC_ILQR.md`

## Accessing Archived Code

```python
# Emits DeprecationWarning
from python.control.reduced6d import step_legacy_contract, LegacyState
```

## Date Archived

2026-01-04
