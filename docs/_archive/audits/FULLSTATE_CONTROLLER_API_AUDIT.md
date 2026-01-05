# FULLSTATE Controller API Audit

**Date**: 2026-01-04
**Objective**: Verify that MPC, iLQR, LQR, and Hybrid controllers use FULLSTATE (18·N+15) dynamics exclusively.
**Verdict**: ✅ **PASS** - All controllers migrated to FULLSTATE.

---

## 1. iLQR Controller

**File**: `python/control/ilqr.py`

### State Dimension
```python
# python/control/ilqr.py:78
self.state_dim = true_legacy_state_dim(n_act)  # 18*n_act + 15
```
✅ Uses `true_legacy_state_dim()` helper → **18·N+15**

### Forward Step
```python
# python/control/ilqr.py:144-146
result = crm_diff_py.true_legacy_step_forward(
    x_coil_t, xf_t, u_t, self.dt, self.params_dict
)
```
✅ Calls `crm_diff_py.true_legacy_step_forward` (FULLSTATE C++ backend)

### Linearization (PyTorch Mode)
```python
# python/control/ilqr.py:177-178
A = torch.autograd.functional.jacobian(
    lambda x: true_legacy_step_torch(x, u_t_torch, self.dt, self.L_inserted, self.params_dict, self.n_act),
    x_t_torch
).numpy()
```
✅ Uses `true_legacy_step_torch` (FULLSTATE autograd wrapper)

### Linearization (C++ Mode)
```python
# python/control/ilqr.py:201-203
result = true_legacy_linearize(
    x_t, u_t, self.dt, self.L_inserted, self.params_dict, self.n_act
)
return result['A'], result['B']
```
✅ Uses `true_legacy_linearize` from `python/control/true_legacy_step.py`

**API imports**:
```python
# python/control/ilqr.py:24,28
from control.true_legacy_step_autograd import true_legacy_step_torch
from control.true_legacy_step import true_legacy_linearize, true_legacy_tip_jacobian
```

### Jacobian Dimensions Check
- **A** (state Jacobian): `∂x_next/∂x` → **(18·N+15) × (18·N+15)**
- **B** (control Jacobian): `∂x_next/∂u` → **(18·N+15) × (3·N)**

✅ Dimensions match FULLSTATE

---

## 2. MPC Controller

**File**: `python/control/mpc.py`

### State Dimension
```python
# python/control/mpc.py:58
self.state_dim = true_legacy_state_dim(n_act)  # 18*n_act + 15
```
✅ Uses `true_legacy_state_dim()` helper → **18·N+15**

### Forward Step
```python
# python/control/mpc.py:223-224
result = crm_diff_py.true_legacy_step_forward(
    x_coil_t, xf_t, u_mpc, dt, controller.params_dict
)
```
✅ Calls `crm_diff_py.true_legacy_step_forward` (FULLSTATE C++ backend)

### Underlying Solver
```python
# python/control/mpc.py:18
from control.ilqr import iLQRSolver
```
✅ MPC uses iLQRSolver internally, which is already verified to be FULLSTATE

**MPC does NOT implement its own dynamics** - it delegates to iLQR solver at line 107-122:
```python
# python/control/mpc.py:107-122
solver = iLQRSolver(
    dt=self.dt,
    L_inserted=self.L_inserted,
    params_dict=self.params_dict,
    horizon=self.horizon,
    n_act=self.n_act,
    Q=Q,
    R=R,
    p_target=p_target,
    terminal_weight=self.Q_tip,
    max_iters=self.max_ilqr_iters,
    tol=1e-2,
    reg_init=1e-3,
    reg_scale=10.0,
    line_search_alphas=[1.0, 0.5, 0.25, 0.1]
)
```

✅ MPC is FULLSTATE (inherits from iLQR)

---

## 3. LQR Controller

**File**: `python/control/lqr.py`

### State Dimension
```python
# python/control/lqr.py:54
state_dim = true_legacy_state_dim(n_act)  # 18*n_act + 15
```
✅ Uses `true_legacy_state_dim()` helper → **18·N+15**

### Forward Step
```python
# python/control/lqr.py:103-104
result = crm_diff_py.true_legacy_step_forward(
    x_coil_t, xf_t, u_t, dt, params_dict
)
```
✅ Calls `crm_diff_py.true_legacy_step_forward` (FULLSTATE C++ backend)

### Linearization
```python
# python/control/lqr.py:117-120
A_t = torch.autograd.functional.jacobian(
    lambda x: true_legacy_step_torch(x, u_t_torch, dt, L_inserted, params_dict, n_act),
    x_t_torch
).numpy()
```
✅ Uses `true_legacy_step_torch` (FULLSTATE autograd wrapper)

**API imports**:
```python
# python/control/lqr.py:20
from control.true_legacy_step_autograd import true_legacy_step_torch
```

### Jacobian Dimensions Check
- **A** (state Jacobian): `∂x_next/∂x` → **(18·N+15) × (18·N+15)**
- **B** (control Jacobian): `∂x_next/∂u` → **(18·N+15) × (3·N)**

✅ Dimensions match FULLSTATE

---

## 4. Hybrid Controller

**File**: `python/control/hybrid_controller.py`

### State Dimension
```python
# python/control/hybrid_controller.py:89
self.state_dim = true_legacy_state_dim(n_act)  # 18*n_act + 15
```
✅ Uses `true_legacy_state_dim()` helper → **18·N+15**

### Forward Step (Safety Check)
```python
# python/control/hybrid_controller.py:209-210
result = crm_diff_py.true_legacy_step_forward(
    x_coil_t, xf_t, u_t, self.dt, self.params_dict
)
```
✅ Calls `crm_diff_py.true_legacy_step_forward` (FULLSTATE C++ backend)

### MPC Backend
```python
# python/control/hybrid_controller.py:33
from control.ilqr import iLQRSolver
```

```python
# python/control/hybrid_controller.py:282-295
solver = iLQRSolver(
    dt=self.dt,
    L_inserted=self.L_inserted,
    params_dict=self.params_dict,
    horizon=self.mpc_horizon,
    n_act=self.n_act,
    Q=np.zeros((self.state_dim, self.state_dim)),  # No state cost
    R=0.01 * np.eye(self.control_dim),  # Control regularization
    p_target=p_ref_mpc[0],  # Terminal target
    terminal_weight=1.0,
    max_iters=self.mpc_max_iters,
    tol=self.mpc_tol,
    jacobian_mode="cpp"  # Use fast Jacobians
)
```

✅ Uses iLQRSolver (already verified FULLSTATE)
✅ Explicitly requests `jacobian_mode="cpp"` → uses `true_legacy_linearize`

**Hybrid controller combines**:
- Ensemble policy (learned, fast)
- iLQR-based MPC (FULLSTATE, safe backup)

✅ Hybrid controller is FULLSTATE when MPC is invoked

---

## 5. Summary Table

| Controller | File | State Dim | Forward Step | Linearization | Verdict |
|------------|------|-----------|--------------|---------------|---------|
| **iLQR** | `ilqr.py:78` | `18·N+15` | `true_legacy_step_forward`:144 | `true_legacy_linearize`:201 or PyTorch:177 | ✅ FULLSTATE |
| **MPC** | `mpc.py:58` | `18·N+15` | `true_legacy_step_forward`:223 | Delegates to iLQR:107-122 | ✅ FULLSTATE |
| **LQR** | `lqr.py:54` | `18·N+15` | `true_legacy_step_forward`:103 | PyTorch:117 | ✅ FULLSTATE |
| **Hybrid** | `hybrid_controller.py:89` | `18·N+15` | `true_legacy_step_forward`:209 | Delegates to iLQR:282-295 | ✅ FULLSTATE |

---

## 6. FULLSTATE API Primitives Checklist

All controllers require these primitives for FULLSTATE operation:

| Primitive | Provider | Status |
|-----------|----------|--------|
| Forward step | `crm_diff_py.true_legacy_step_forward` | ✅ Available (C++) |
| VJP (single) | `crm_diff_py.true_legacy_step_vjp` | ✅ Available (C++) |
| VJP (batched) | `crm_diff_py.true_legacy_step_vjp_batched` | ✅ Available (C++) |
| Linearization | `python/control/true_legacy_step.py:true_legacy_linearize` | ✅ Available (Python wrapper using PyTorch/FD) |
| Tip Jacobian | `python/control/true_legacy_step.py:true_legacy_tip_jacobian` | ✅ Available (Python wrapper) |

**Linearization implementation**:
```python
# python/control/true_legacy_step.py:269-308
def true_legacy_linearize(
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    *,
    n_act: int,
    catheter_params: Dict,
    L_inserted: float = 100.0,
    method: str = "torch_autograd",  # or "finite_diff"
    eps: float = 1e-7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute linearization Jacobians A, B for TRUE legacy dynamics.

    A = ∂x_next/∂x  (state Jacobian)
    B = ∂x_next/∂u  (control Jacobian)

    Returns:
        A: State Jacobian [state_dim, state_dim]  # (18·N+15) × (18·N+15)
        B: Control Jacobian [state_dim, n_act*3]  # (18·N+15) × (3·N)
    """
```

✅ All required primitives are available for FULLSTATE operation.

---

## 7. Verdict

**✅ PASS**: All controllers (iLQR, MPC, LQR, Hybrid) use FULLSTATE (18·N+15) exclusively.

**Key findings**:
- ✅ All controllers compute `state_dim = 18·N+15` using `true_legacy_state_dim(n_act)`
- ✅ All controllers call `crm_diff_py.true_legacy_step_forward` for dynamics
- ✅ Linearization uses either PyTorch autograd or `true_legacy_linearize` wrapper
- ✅ No imports from `control.reduced6d` or legacy reduced-state modules
- ✅ All Jacobian dimensions match FULLSTATE (A: DxD, B: Dx3N where D=18N+15)

**Migration complete**: MPC/iLQR are fully migrated to FULLSTATE default path.
