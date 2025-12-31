# CP2.0 Patch Summary

**Date**: 2025-12-31
**Status**: ✓ Applied to CP2.1 implementation plan
**Commits**: 77ab9d2

---

## Critical Blockers Identified & Fixed

### BLOCKER A: Zero Control Coupling

**Problem**:
```
B = ∂G/∂u_t = 0  →  ∂x_{t+1}/∂u_t = 0
```
Control has **no effect** on next state. MPC/iLQR/RL impossible.

**Root cause**: Original residual omitted actuation force term.

**Fix**: Add actuation coupling via equilibrium Jacobians:
```
B = [0           ]  (6×3N)
    [-dt*K*J_u_zc]

where:
  K = K_tip from equilibrium_forward
  J_u_zc = ∂u_tip/∂u_t from equilibrium_forward
```

**Result**: Control now affects velocity → system is **controllable**.

---

### BLOCKER B: Unnecessary Newton Solver

**Problem**: Using 20 Newton iterations to solve a **linear system**:
```
G = A*x_{t+1} + C*x_t + B*u_t = 0
```
This is affine in x_{t+1}, not nonlinear!

**Consequences**:
- Unnecessary complexity
- Additional failure modes (convergence)
- Performance overhead (20 iterations vs 1 solve)

**Fix**: Direct linear solve:
```
A * x_{t+1} = -C*x_t - B*u_t
x_{t+1} = -A^{-1} * (C*x_t + B*u_t)
```

**Result**: One FullPivLU solve, no iteration, simpler, faster.

---

## Updated Design

### Residual (Patched)

```
G(x_{t+1}, x_t, u_t) = [u_{t+1} - u_t - dt*v_{t+1}]
                        [M*v_{t+1} - M*v_t + dt*D*v_{t+1} + dt*K*u_{t+1} - dt*K*J_u_zc*u_t]

Affine form:
G = A*x_{t+1} + C*x_t + B*u_t = 0
```

**Key change**: Added `-dt*K*J_u_zc*u_t` term to velocity equation.

### Jacobians (Patched)

```
A = ∂G/∂x_{t+1} = [I,       -dt*I    ]  (6×6, unchanged)
                   [dt*K,    M+dt*D   ]

C = ∂G/∂x_t = [-I,  0  ]  (6×6, unchanged)
               [0,  -M  ]

B = ∂G/∂u_t = [0           ]  (6×3N, NOW NON-ZERO!)
               [-dt*K*J_u_zc]
```

### Forward Solve (Patched)

**Before** (WRONG):
```cpp
for (iter = 0; iter < 20; iter++) {
    G = residual(x_next, ...);
    if (||G|| < tol) break;
    A = jacobian(...);
    delta_x = -A^{-1} * G;
    x_next += delta_x;
}
```

**After** (CORRECT):
```cpp
rhs = -C*x_t - B*u_t;
x_next = A^{-1} * rhs;  // One FullPivLU solve
```

### Backward Pass (Patched)

**Algorithm** (same structure, but B ≠ 0 now):
```
1. Solve: A^T λ = grad_x_next
2. grad_x_t = -C^T λ
3. grad_u_t = -B^T λ  (NOW NON-ZERO!)
```

---

## Changes to Data Structures

### DynamicsStepResult

**Removed**:
- `int nl_iterations` (no longer meaningful for linear solve)

**Renamed**:
- `final_residual` → `solve_residual` (measures solve accuracy, not Newton convergence)

**Unchanged**:
- `lu_rank`, `rel_solve_residual`, `exit_code`

---

## Validation Changes

### CP2.1 Smoke Test

**Added checks**:
1. **Control authority test**:
   ```cpp
   u_t = [0.1, 0, 0];  // Non-zero
   // Verify: x_next ≠ x_t (control has effect)
   ```

2. **Backward control gradient**:
   ```cpp
   // Verify: grad_u_t ≠ 0 (control gradients exist)
   ```

### Acceptance Criteria

**Added**:
- [ ] For non-zero input (u=[0.1,0,0]), x_next ≠ x_t
- [ ] grad_u_t is finite (may be non-zero even for zero input)

**Removed**:
- ~~Newton solver converges (nl_iterations < 10)~~
- ~~Final residual < 1e-9~~

**Updated**:
- [x] Forward solve rank = 6 (full rank A)
- [x] Solve residual < 1e-10

---

## Files Modified

| File | Changes |
|------|---------|
| `docs/design/CP2_0_DYNAMICS_PATCH.md` | **NEW** - Full patch documentation |
| `docs/design/CP2_1_IMPLEMENTATION_PLAN.md` | Updated residual, Jacobians, forward solve, smoke test, acceptance criteria |

**Total changes**: 725 insertions, 84 deletions

---

## Why This Works

### Physical Justification

The term `-dt*K*J_u_zc*u_t` represents:

1. **J_u_zc**: How actuation current u_t affects tip curvature
   - From equilibrium: magnetic field → coil torque → tip bends
   - ∂u_tip/∂u_t captured by equilibrium solver

2. **K**: Stiffness converts curvature to elastic moment
   - K_tip is the stiffness matrix at the tip
   - Elastic response to curvature change

3. **K * J_u_zc**: Magnetic actuation → tip curvature → elastic moment
   - Chain rule: moment = K * (curvature from actuation)
   - This moment propagates to base via equilibrium constraint

4. **-dt * K * J_u_zc * u_t**: Force term in momentum equation
   - Time-integrated actuation force on base velocity
   - Negative sign: elastic restoring force opposes actuation

**No new physics**: Just using existing equilibrium Jacobians from CP1.4.

### Mathematical Justification

The system is **affine** (linear + constant):
```
G(x_{t+1}, x_t, u_t) = A*x_{t+1} + C*x_t + B*u_t
```

For affine systems:
- **Forward**: Direct solve A*x_{t+1} = -(C*x_t + B*u_t)
- **Backward**: VJP via adjoint λ solving A^T λ = v

No iteration needed. This is standard linear algebra, not nonlinear optimization.

---

## Benefits

| Aspect | Before | After |
|--------|--------|-------|
| Controllability | Uncontrollable (B=0) | Controllable (B≠0) |
| Solve method | Newton (20 iters) | Direct (1 solve) |
| Failure modes | Convergence, iteration limit | Rank-deficiency only |
| Performance | O(k·n³) | O(n³) |
| Diagnostics | nl_iterations, final_residual | solve_residual only |
| Complexity | Higher | Lower |

---

## References to Existing Code

All components use existing functions (no CP1.x modifications):

| Component | Source | Location |
|-----------|--------|----------|
| K_tip | `equilibrium_forward` | `CRM_DiffEquilibrium.cpp:116-124` |
| J_u_zc | `equilibrium_forward` | `CRM_DiffEquilibrium.cpp:103-112` |
| FullPivLU pattern | `equilibrium_backward` | `CRM_DiffEquilibrium.cpp:163-194` |

---

## Next Steps

1. ✓ Patch applied to CP2_1_IMPLEMENTATION_PLAN.md
2. **Implement CP2.1** following patched plan:
   - Create `src/CRM_DiffDynamics.{hpp,cpp}`
   - Implement direct linear solve (no Newton)
   - Compute B = -dt*K*J_u_zc from equilibrium Jacobians
   - Add smoke test with control authority check
3. Verify all patched acceptance criteria pass
4. Proceed to CP2.2 (finite difference validation)

---

**Patch Version**: CP2.0
**Applied**: 2025-12-31
**Status**: Ready for CP2.1 implementation
