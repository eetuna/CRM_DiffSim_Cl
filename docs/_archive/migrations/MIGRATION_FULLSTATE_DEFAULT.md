# FULLSTATE Default Migration Guide

**Date**: 2026-01-04
**Objective**: Make FULLSTATE (18·N+15) the default for all controllers (iLQR/MPC/LQR/Hybrid)

---

## Overview

This migration replaces the Reduced6D dynamics (6-dimensional `[u_0, v_0]` state) with FULLSTATE TRUE legacy dynamics (18·N+15 dimensional state) as the **default and only** option for all Python controllers.

### Key Changes

1. **C++ Backend**: Removed `CRM_DiffDynamics.{cpp,hpp}` from default build → archived to `src/reduced6d/`
2. **Python Bindings**: Removed `dynamics_forward()` and related Reduced6D bindings → ONLY `true_legacy_step_forward()` exposed
3. **Controllers**: Replaced all default controllers with FULLSTATE implementations:
   - `python/control/ilqr.py` → FULLSTATE iLQR (18·N+15)
   - `python/control/mpc.py` → FULLSTATE MPC
   - `python/control/lqr.py` → FULLSTATE LQR
   - `python/control/hybrid_controller.py` → FULLSTATE Hybrid
4. **Old Controllers**: Preserved in `python/control/reduced6d/` with `_6d` suffix (NOT imported by default)

---

## State Dimension Changes

| Aspect | Reduced6D (OLD) | FULLSTATE (NEW) |
|--------|-----------------|-----------------|
| **State Dimension** | 6 | 18·N + 15 (33 for N=1) |
| **State Components** | `[u_0[3], v_0[3]]` | `x_coil[N,18], xf[15]` |
| **Control Dimension** | 3 | 3·N (3 for N=1) |
| **Dynamics Function** | `dynamics_forward()` | `true_legacy_step_forward()` |
| **Linearization** | `dynamics_linearize()` | `true_legacy_linearize()` |
| **PyTorch Autograd** | `dynamics_step()` | `true_legacy_step_torch()` |

For N=1 actuator set:
- OLD: `x ∈ ℝ⁶`, `u ∈ ℝ³`
- NEW: `x ∈ ℝ³³`, `u ∈ ℝ³`

---

## Migration Instructions

### For Users of iLQR/MPC/LQR/Hybrid Controllers

**Before** (Reduced6D - NO LONGER WORKS):
```python
from control import iLQRSolver
import numpy as np

# OLD: 6D state
x0 = np.zeros(6)  # [u_0, v_0]
Q = np.eye(6)
R = np.eye(3)

solver = iLQRSolver(
    dt=0.01, L_inserted=100.0, params_dict=params,
    horizon=50, Q=Q, R=R, p_target=np.array([10, 0, 0])
)
X, U, converged = solver.solve(x0)  # FAILS - wrong state dimension
```

**After** (FULLSTATE):
```python
from control import iLQRSolver, pack_true_legacy_state, STATE_DIM_FULL
import numpy as np

# NEW: FULLSTATE (18*N+15)
n_act = 1
state_dim = STATE_DIM_FULL(n_act)  # 33

# Initialize from coil + tip state
x_coil = np.zeros((n_act, 18))  # Coil state: v[3], w[3], p[3], R[9]
xf = np.zeros(15)               # Tip state: p_tip[3], R_tip[9], u_tip[3]
x0 = pack_true_legacy_state(x_coil, xf, n_act)

# Cost matrices: FULLSTATE dimensions
Q = np.zeros((state_dim, state_dim))  # 33x33
R = 0.1 * np.eye(3)                   # 3x3

solver = iLQRSolver(
    dt=0.01, L_inserted=100.0, params_dict=params,
    horizon=50, n_act=n_act, Q=Q, R=R, p_target=np.array([10, 0, 0])
)
X, U, converged = solver.solve(x0)  # WORKS
```

### Key API Changes

#### State Initialization

```python
# OLD (Reduced6D - NO LONGER AVAILABLE)
x0 = np.zeros(6)

# NEW (FULLSTATE)
from control import pack_true_legacy_state
x_coil = np.zeros((1, 18))
xf = np.zeros(15)
x0 = pack_true_legacy_state(x_coil, xf, n_act=1)
```

#### Cost Matrix Dimensions

```python
# OLD
Q = np.eye(6)   # State cost
R = np.eye(3)   # Control cost

# NEW
Q = np.zeros((33, 33))  # State cost (18*1+15 = 33)
R = 0.1 * np.eye(3)     # Control cost (unchanged for n_act=1)
```

#### Controller Initialization

```python
# OLD - NO n_act parameter
solver = iLQRSolver(dt, L, params, horizon, Q=Q, R=R, ...)

# NEW - MUST specify n_act
solver = iLQRSolver(dt, L, params, horizon, n_act=1, Q=Q, R=R, ...)
```

---

## Breaking Changes

### Removed Imports (Will Cause ImportError)

```python
# These NO LONGER WORK:
from crm_diff_py import dynamics_forward            # REMOVED
from crm_diff_py import dynamics_backward           # REMOVED
from crm_diff_py import dynamics_linearize          # REMOVED
from crm_diff_py import dynamics_linearize_batched  # REMOVED
from crm_dynamics_torch import dynamics_step        # REMOVED
```

### Removed Python Modules (Will Cause ImportError)

Old 6D controllers at default paths have been **moved** (not aliased):

```python
# These NO LONGER WORK:
from control.ilqr import iLQRSolver                  # Now FULLSTATE
from control.mpc import MPCController                 # Now FULLSTATE
from control.lqr import finite_horizon_lqr           # Now FULLSTATE
from control.hybrid_controller import HybridController  # Now FULLSTATE
from control import hybrid_state_contract            # MOVED to reduced6d/
from control import step_hybrid_legacy_contract      # MOVED to reduced6d/
```

---

## Accessing Reduced6D (If Absolutely Necessary)

**WARNING**: Reduced6D is ARCHIVED and NOT the TRUE legacy implementation.

If you must access old 6D controllers (for comparison only):

```python
# Explicit import from archived location
from python.control.reduced6d.ilqr_6d import iLQRSolver as iLQRSolver6D
from python.control.reduced6d.legacy_state import LegacyState
```

**Do NOT use** for production - Reduced6D is not the correct physics.

---

## Default Dynamics: TRUE Legacy

The default dynamics is now **TRUE legacy** (`DynamicsBVP → DYNSolverIVP`):

```python
# Default binding (FULLSTATE)
from crm_diff_py import true_legacy_step_forward

# Usage
result = true_legacy_step_forward(x_coil, xf, u, dt, params_dict)
x_coil_next = result['x_coil_next']  # (N, 18)
xf_next = result['xf_next']          # (15,)
```

Reduced6D bindings (`dynamics_forward()`) have been **removed entirely** from the default namespace.

---

## Summary

- **FULLSTATE (18·N+15)** is now the **ONLY** default
- All controllers use `true_legacy_step()` internally
- Cost matrices must be resized to FULLSTATE dimensions
- Reduced6D archived under `python/control/reduced6d/` (opt-in only)
- **NO backward compatibility shims** - old imports will fail with `ImportError`

For questions, see `docs/migrations/ARCHIVE_REDUCED6D_9D.md` or `MIGRATION_COMPLETION_REPORT_FULLSTATE.md`.
