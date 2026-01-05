# Migration Guide: MPC/iLQR to TRUE Legacy Dynamics

**Date:** 2026-01-04
**Status:** Infrastructure Complete - Controllers Need Implementation

---

## Overview

This guide explains how to migrate from **reduced 6D dynamics** to **TRUE legacy dynamics (18·N+15)**.

**Migration Status:**
- ✅ Reduced 6D archived in `python/control/reduced6d/`
- ✅ TRUE legacy linearization functions added
- ✅ TRUE legacy is now the default in `python/control/__init__.py`
- ⏳ Full iLQR/MPC controllers (user implementation needed)

---

## What Changed

### Old (Reduced 6D)
- **State dimension:** 6 (u_0[3], v_0[3])
- **Physics:** Implicit Euler with simplified model
- **Files:** `python/control/step_legacy_contract.py` (ARCHIVED)
- **Status:** NON-LEGACY, moved to `python/control/reduced6d/`

### New (TRUE Legacy 18·N+15)
- **State dimension:** 18·N + 15 (full rigid body)
- **Physics:** DynamicsBVP → DYNSolverIVP
- **Files:** `python/control/true_legacy_step.py`
- **Status:** DEFAULT dynamics interface

---

## Available TRUE Legacy Functions

### 1. Forward Dynamics

```python
from python.control import true_legacy_step, true_legacy_step_torch
import torch

# Non-differentiable forward step
x_next, obs = true_legacy_step(
    x, u, dt, n_act=1,
    catheter_params=params,
    L_inserted=100.0,
    warmstart=None
)

# Differentiable forward step (PyTorch autograd)
tip_p, x_next = true_legacy_step_torch(
    x_torch, u_torch, dt,
    n_act=1,
    catheter_params=params,
    L_inserted=100.0,
    return_x_next=True
)
```

### 2. Linearization (NEW)

```python
from python.control import true_legacy_linearize

# Compute Jacobians A, B
A, B = true_legacy_linearize(
    x, u, dt,
    n_act=1,
    catheter_params=params,
    L_inserted=100.0,
    method="torch_autograd"  # or "finite_diff"
)

# A: [state_dim, state_dim] = ∂x_next/∂x
# B: [state_dim, n_act*3] = ∂x_next/∂u
```

### 3. Tip Position Jacobian (NEW)

```python
from python.control import true_legacy_tip_jacobian

# Compute ∂p_tip/∂x
J_p = true_legacy_tip_jacobian(
    x, n_act=1,
    method="analytic"  # or "autograd"
)

# J_p: [3, state_dim]
```

---

## State Initialization

### Old (6D)
```python
x0 = np.array([0, 0, 0,  # u_0
               0, 0, 0]) # v_0
```

### New (18·N+15)
```python
from python.control import pack_true_legacy_state

n_act = 1  # Number of coils
x_coil = np.zeros((n_act, 18))  # Per-coil state: [v, w, p, R]
xf = np.zeros(15)                # Tip state: [p, R, u]
x0 = pack_true_legacy_state(x_coil, xf)  # Flatten to [18*n_act+15]
```

---

## Implementing TRUE Legacy iLQR

### Key Modifications Needed

1. **Update state dimension**
   ```python
   self.state_dim = 18 * n_act + 15  # Instead of 6
   self.control_dim = n_act * 3
   ```

2. **Use TRUE legacy forward rollout**
   ```python
   def rollout(self, x0, U, warmstart=None):
       X = np.zeros((self.horizon + 1, self.state_dim))
       X[0] = x0
       current_warmstart = warmstart

       for t in range(self.horizon):
           x_next, obs = true_legacy_step(
               torch.from_numpy(X[t]),
               torch.from_numpy(U[t]),
               self.dt,
               n_act=self.n_act,
               catheter_params=self.params_dict,
               L_inserted=self.L_inserted,
               warmstart=current_warmstart
           )
           X[t+1] = x_next.numpy()
           current_warmstart = obs['warmstart_next']

       return X
   ```

3. **Use TRUE legacy linearization**
   ```python
   def extract_jacobians(self, x_t, u_t):
       return true_legacy_linearize(
           x_t, u_t, self.dt,
           n_act=self.n_act,
           catheter_params=self.params_dict,
           L_inserted=self.L_inserted,
           method="torch_autograd"
       )
   ```

4. **Update cost function**
   ```python
   def compute_tip_jacobian(self, x):
       return true_legacy_tip_jacobian(x, self.n_act, method="analytic")
   ```

### Minimal iLQR Example

```python
from python.control import (
    true_legacy_step, true_legacy_linearize,
    true_legacy_tip_jacobian, pack_true_legacy_state
)
import numpy as np
import torch

class SimpleTrueLegacyiLQR:
    def __init__(self, dt, L_inserted, params_dict, n_act, horizon):
        self.dt = dt
        self.L_inserted = L_inserted
        self.params_dict = params_dict
        self.n_act = n_act
        self.horizon = horizon
        self.state_dim = 18 * n_act + 15

    def solve(self, x0, x_goal):
        # Initialize control sequence
        U = np.zeros((self.horizon, self.n_act, 3))

        for iteration in range(10):  # Simple fixed iterations
            # Forward rollout
            X = self.rollout(x0, U)

            # Backward pass (compute gains)
            # ... (implement standard iLQR backward pass)

            # Forward pass (line search)
            # ... (implement line search with new controls)

        return U, X
```

---

## Implementing TRUE Legacy MPC

### Key Modifications

```python
from python.control import true_legacy_step

class SimpleTrueLegacyMPC:
    def __init__(self, dt, L_inserted, params_dict, n_act, horizon=10):
        self.dt = dt
        self.L_inserted = L_inserted
        self.params_dict = params_dict
        self.n_act = n_act
        self.horizon = horizon
        self.state_dim = 18 * n_act + 15

    def get_control(self, x_current, x_goal):
        # Solve iLQR for current state
        # ... (use SimpleTrueLegacyiLQR or equivalent)

        # Return first control
        return u_mpc
```

---

## Accessing Reduced 6D (if needed)

The reduced 6D implementation is ARCHIVED but still functional:

```python
# Emits DeprecationWarning
from python.control.reduced6d import (
    LegacyState,
    step_legacy_contract,
    vjp_legacy_contract
)
```

**Location:**
- **Python:** `python/control/reduced6d/`
- **Tests:** `python/test/reduced6d/`
- **Docs:** `docs/archive/reduced6d/`

---

## Running Tests

### TRUE Legacy Tests
```bash
PYTHONPATH=build:python pytest python/test_true_legacy_step.py -v
PYTHONPATH=build:python pytest python/test_true_legacy_step_vjp.py -v
```

### Archived Reduced 6D Tests
```bash
PYTHONPATH=build:python pytest python/test/reduced6d/ -v
```

---

## Next Steps

1. **Implement full iLQR solver** using TRUE legacy linearization
2. **Implement MPC controller** wrapping iLQR
3. **Write comprehensive tests** for controllers
4. **Benchmark performance** vs reduced 6D (expect ~3x slower due to state dimension)
5. **Tune solver parameters** (trust-region tolerance, warm-start strategy)

---

## Reference Documentation

- **TRUE Legacy Contract:** `docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md`
- **Reduced 6D Archive:** `docs/archive/reduced6d/README.md`
- **PRE-B0 Audit:** `docs/audits/PRE_B0_COMPLETION_REPORT.md`

---

## Support

For questions or issues:
1. Check existing audit documents in `docs/audits/`
2. Review TRUE legacy contract specification
3. Examine working examples in `python/test_true_legacy_*.py`

**Migration Date:** 2026-01-04
**Migration Status:** Partial (infrastructure complete, controllers pending)
