# CP2.0 Design Patch: Fix Control Coupling & Remove Newton

**Date**: 2025-12-31
**Status**: Critical design fix before CP2.1 implementation
**Affects**: `DYNAMICS_V1_1_DESIGN.md`, `CP2_1_IMPLEMENTATION_PLAN.md`

---

## Issue Summary

Two critical blockers identified in the current v1.1 dynamics design:

**BLOCKER A**: Control has no effect on next state (B = 0)
**BLOCKER B**: Linear residual doesn't need Newton solver

---

## BLOCKER A: Zero Control Coupling

### Current Design Problem

**Reference**: `DYNAMICS_V1_1_DESIGN.md` Decision Log, equation for B:

```
B = ∂G/∂u_t = [0]  (6×3)
               [0]
```

**Reference**: `CP2_1_IMPLEMENTATION_PLAN.md` Section 3.3:

> For v1.1 simplified model (no direct control coupling in residual):
> B = 0 (6×3)

### Why This Breaks MPC/iLQR/RL

If B = 0, then:
```
∂x_{t+1}/∂u_t = -A^{-1} B = 0
```

**Consequence**: Changing control u_t has **zero effect** on the next state x_{t+1}. This makes:
- **MPC impossible**: No control authority
- **iLQR impossible**: Zero control-state coupling
- **RL impossible**: Policy gradient is always zero

The dynamics become **uncontrollable** in the control-theoretic sense.

### Root Cause

The simplified residual in Decision Log omits actuation forces:
```
G_v = M v_{t+1} - M v_t + dt * (D v_{t+1} + K u_{t+1})
```

No term depending on u_t → B = 0.

---

## BLOCKER B: Unnecessary Newton Solver

### Current Design Problem

**Reference**: `CP2_1_IMPLEMENTATION_PLAN.md` Section 5:

> Solve G(x_{t+1}, x_t, u_t) = 0 using Newton's method
> (20 iterations max, tolerance 1e-10)

### Why Newton is Unnecessary

The residual is **affine** in x_{t+1}:
```
G = A * x_{t+1} + C * x_t + B * u_t
```

where A, C, B are constant matrices (independent of x_{t+1}).

**This is a linear system**, not a nonlinear one!

### Consequences of Using Newton

1. **Unnecessary complexity**: 20 iterations for a system solvable in 1 step
2. **Additional failure modes**: convergence failures, iteration limits
3. **Performance overhead**: Multiple FullPivLU solves instead of one
4. **Wasted diagnostics**: nl_iterations, final_residual meaningless for linear system

### Correct Solution

**Direct solve**:
```
A * x_{t+1} = -C * x_t - B * u_t
x_{t+1} = -A^{-1} * (C * x_t + B * u_t)
```

**One FullPivLU solve**, no iteration needed.

---

## PATCH: Minimal Actuation Coupling

### Key Insight: Use Existing Equilibrium Jacobians

From `equilibrium_forward`, we already have:
- **J_u_zc**: ∂u_tip/∂u_t (3×3N) - how actuation affects tip curvature
- **K_tip**: stiffness matrix (3×3)

**Physical interpretation**:
- Magnetic actuation u_t creates tip curvature u_tip
- Tip curvature creates elastic moment τ = K_tip * u_tip
- This moment propagates to base via equilibrium constraint

### Actuation Force Term

Add magnetic torque to momentum equation:
```
f_actuation(u_t) = K_tip * J_u_zc * u_t
```

**Units check**:
- K_tip: N·mm (stiffness)
- J_u_zc: (1/mm) / A (curvature per ampere)
- u_t: A (amperes)
- Result: N (force) ✓

**Why this works**:
- J_u_zc captures equilibrium sensitivity: ∂u_tip/∂u_t
- K_tip converts curvature to moment
- Net effect: actuation current → tip curvature → elastic moment → base force

**No new physics**: Just using existing equilibrium Jacobians computed in CP1.4.

---

## PATCHED DESIGN

### Updated Residual (Linear in x_{t+1})

```
State: x = [u_0, v_0] ∈ R^6

Residual G(x_{t+1}, x_t, u_t) ∈ R^6:
  G_u = u_{t+1} - u_t - dt * v_{t+1}                                    (3×1)
  G_v = M v_{t+1} - M v_t + dt*D v_{t+1} + dt*K u_{t+1} - dt*K*J_u_zc*u_t (3×1)

Affine form:
  G = A * x_{t+1} + C * x_t + B * u_t = 0
```

**Key change**: Added `-dt*K*J_u_zc*u_t` term to G_v.

### Updated Jacobians

#### A = ∂G/∂x_{t+1} (unchanged)

```
A = [I,       -dt*I    ]  (6×6)
    [dt*K,    M+dt*D   ]
```

**Block structure**:
- A[0:3, 0:3] = I (3×3)
- A[0:3, 3:6] = -dt*I (3×3)
- A[3:6, 0:3] = dt*K (3×3)
- A[3:6, 3:6] = M + dt*D (3×3)

#### C = ∂G/∂x_t (unchanged)

```
C = [-I,  0  ]  (6×6)
    [0,  -M  ]
```

#### B = ∂G/∂u_t (NEW - non-zero!)

```
B = [0           ]  (3×3N)
    [-dt*K*J_u_zc]  (3×3N)

where:
  J_u_zc: from equilibrium_forward (3×3N, row-major)
  K: from equilibrium_forward K_tip (3×3, row-major)
  N: NUM_ACT_SET (typically 1)
```

**Dimensions**:
- B is 6×3N (for N=1, this is 6×3)
- Top block (3×3N): zeros
- Bottom block (3×3N): -dt*K*J_u_zc (matrix product, 3×3 times 3×3N = 3×3N)

**Why this fixes BLOCKER A**:
```
∂x_{t+1}/∂u_t = -A^{-1} B ≠ 0

Specifically:
∂v_{t+1}/∂u_t = [bottom 3 rows of -A^{-1} B]
              = [M + dt*D]^{-1} * K * J_u_zc * dt
              ≠ 0 (in general)
```

Control now directly affects velocity (and indirectly affects position via coupling).

---

## PATCHED FORWARD PASS

### Direct Linear Solve (No Newton)

**Reference**: `CP2_1_IMPLEMENTATION_PLAN.md` Section 5 (REPLACED)

**Old approach** (WRONG):
```cpp
// Newton iteration (unnecessary!)
for (int iter = 0; iter < 20; iter++) {
    compute_residual(...);
    compute_jacobian(...);
    solve A * delta_x = -G;
    x_next += delta_x;
}
```

**New approach** (CORRECT):
```cpp
// Direct solve (one step)
// Solve: A * x_{t+1} = -C * x_t - B * u_t

// 1. Compute RHS
double rhs[6];
// rhs = -C * x_t - B * u_t
compute_rhs(C, x_t, B, u_t, rhs);

// 2. Solve A * x_next = rhs using FullPivLU
Map<Matrix<double, 6, 6, RowMajor>> A_map(A);
Map<VectorXd> rhs_map(rhs, 6);

FullPivLU<MatrixXd> lu(A_map);
VectorXd x_next_vec = lu.solve(rhs_map);

// 3. Extract solution
for (int i = 0; i < 6; i++) {
    out.x_next[i] = x_next_vec[i];
}

// 4. Check solve accuracy (NOT convergence!)
VectorXd residual_check = A_map * x_next_vec - rhs_map;
double rel_res = residual_check.norm() / std::max(rhs_map.norm(), 1.0);

if (rel_res > 1e-10) {
    return 2;  // Solve inaccurate
}
```

**Diagnostics changes**:
- ~~nl_iterations~~ → **REMOVED** (always 1 for linear solve)
- ~~final_residual~~ → **RENAMED** to `solve_residual` (measures solve accuracy, not convergence)
- Keep: lu_rank, rel_solve_residual, exit_code

**Why this fixes BLOCKER B**:
- One FullPivLU solve instead of 20 Newton iterations
- No convergence failures
- No iteration limits
- Simpler, faster, more robust

---

## PATCHED BACKWARD PASS

### VJP with Non-Zero B

**Algorithm** (same structure, but now B ≠ 0):

```
Given: upstream gradient v = ∂L/∂x_{t+1} ∈ R^6

1. Solve:  A^T λ = v          // λ ∈ R^6 (adjoint variables)
2. Compute: ∂L/∂x_t = -C^T λ  // (6×6)^T * (6×1) → 6×1
3. Compute: ∂L/∂u_t = -B^T λ  // (6×3N)^T * (6×1) → 3N×1
```

**Implementation** (same FullPivLU pattern as equilibrium_backward):

```cpp
int dynamics_backward(
    const DynamicsStepResult& fwd_result,
    const double grad_x_next[6],
    double grad_x_t[6],
    double grad_u_t[NUM_ACT_SET*3],
    int* lu_rank,
    double* rel_residual
) {
    // Extract cached Jacobians
    Map<const Matrix<double, 6, 6, RowMajor>> A_map(fwd_result.J_G_xnext);
    Map<const Matrix<double, 6, 6, RowMajor>> C_map(fwd_result.J_G_xt);
    Map<const Matrix<double, 6, Dynamic, RowMajor>> B_map(
        fwd_result.J_G_ut, 6, 3*NUM_ACT_SET);
    Map<const VectorXd> v(grad_x_next, 6);

    // Solve A^T λ = v using FullPivLU
    FullPivLU<MatrixXd> lu(A_map.transpose());

    int rank = lu.rank();
    if (lu_rank) *lu_rank = rank;
    if (rank < 6) return 1;  // rank-deficient

    VectorXd lambda = lu.solve(v);

    // Check solve accuracy
    double residual_norm = (A_map.transpose() * lambda - v).norm();
    double rel_res = residual_norm / std::max(v.norm(), 1.0);
    if (rel_residual) *rel_residual = rel_res;
    if (rel_res > 1e-10) return 2;  // inaccurate

    // Compute gradients
    Map<VectorXd> grad_xt(grad_x_t, 6);
    grad_xt = -C_map.transpose() * lambda;

    Map<VectorXd> grad_ut(grad_u_t, 3*NUM_ACT_SET);
    grad_ut = -B_map.transpose() * lambda;  // NOW NON-ZERO!

    return 0;
}
```

**Key change**: `grad_u_t` is now computed via `-B^T λ` where B ≠ 0.

---

## PATCHED SMOKE TEST

### Updated Expectations

**Reference**: `CP2_1_IMPLEMENTATION_PLAN.md` Section 6

**Input** (unchanged):
```cpp
x_t = [0, 0, 0, 0, 0, 0];
u_t = [0, 0, 0];
dt = 0.01;
Li = 50.0;
```

**Expected output** (UPDATED):

Forward pass:
- status = 0
- ~~nl_iterations = 1~~ → **REMOVED** (no iterations for linear solve)
- solve_residual < 1e-10 (solve accuracy, not convergence)
- x_next ≈ [0, 0, 0, 0, 0, 0] (within 1e-6)

Backward pass:
- bwd_status = 0
- lu_rank = 6
- rel_residual < 1e-10
- grad_x_t finite and non-zero
- **grad_u_t may be non-zero** (even for zero input, due to B ≠ 0)

**New test case** (verify control authority):

```cpp
// Test: non-zero control should change state
x_t = [0, 0, 0, 0, 0, 0];
u_t = [0.1, 0, 0];  // Non-zero actuation
dt = 0.01;

// Expect: x_next ≠ x_t (control has effect)
// Specifically: v_{t+1} ≠ 0 due to actuation force
```

---

## UPDATED IMPLEMENTATION REQUIREMENTS

### Computing B Matrix

**In `dynamics_forward`**:

```cpp
// After calling equilibrium_forward to get K_tip and J_u_zc:
EquilibriumResult eq_result;
equilibrium_forward(u_t, L_inserted, params, eq_result);

// Extract K and J_u_zc
Map<const Matrix3d> K_map(eq_result.K_tip);  // 3×3
Map<const Matrix<double, 3, Dynamic, RowMajor>> J_u_zc_map(
    eq_result.J_u_zc, 3, 3*NUM_ACT_SET);

// Compute K * J_u_zc (3×3N matrix product)
MatrixXd K_J_u_zc = K_map * J_u_zc_map;  // 3×3N

// Assemble B (6×3N)
// B[0:3, :] = 0 (top block)
// B[3:6, :] = -dt * K_J_u_zc (bottom block)

for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3*NUM_ACT_SET; j++) {
        out.J_G_ut[i * (3*NUM_ACT_SET) + j] = 0.0;  // top block
        out.J_G_ut[(3+i) * (3*NUM_ACT_SET) + j] = -dt * K_J_u_zc(i, j);  // bottom block
    }
}
```

### Computing RHS = -C*x_t - B*u_t

```cpp
void compute_rhs(
    const double C[36],      // 6×6
    const double x_t[6],
    const double B[18],      // 6×3 (for NUM_ACT_SET=1)
    const double u_t[3],
    double rhs[6]
) {
    Map<const Matrix<double, 6, 6, RowMajor>> C_map(C);
    Map<const Matrix<double, 6, 3, RowMajor>> B_map(B);
    Map<const VectorXd> x_t_map(x_t, 6);
    Map<const VectorXd> u_t_map(u_t, 3);
    Map<VectorXd> rhs_map(rhs, 6);

    rhs_map = -C_map * x_t_map - B_map * u_t_map;
}
```

---

## CHANGES TO DATA STRUCTURES

### DynamicsStepResult (UPDATED)

```cpp
struct DynamicsStepResult {
    double x_next[6];
    int converged;  // 0=success, >0=failed

    double p_tip[3];
    double u_tip[3];

    // Cached Jacobians
    double J_G_xnext[36];            // A: 6×6
    double J_G_xt[36];               // C: 6×6
    double J_G_ut[6*NUM_ACT_SET*3];  // B: 6×3N (NOW NON-ZERO)

    double J_p_u0[9];
    double J_p_ut[3*NUM_ACT_SET*3];

    double M[9], D[9], K[9];

    // Diagnostics (UPDATED)
    // REMOVED: int nl_iterations (no longer needed)
    double solve_residual;       // RENAMED from final_residual
    int lu_rank;
    double rel_solve_residual;
    int exit_code;
};
```

**Changes**:
- `nl_iterations` → REMOVED (always 1 for linear solve)
- `final_residual` → `solve_residual` (measures FullPivLU accuracy, not Newton convergence)
- `J_G_ut` now populated with non-zero values

---

## VALIDATION REQUIREMENTS

### CP2.1 Updated Acceptance Criteria

**Functional Phase** (UPDATED):
- [ ] Forward pass: status = 0, solve_residual < 1e-10
- [ ] Zero input: ||x_next - x_t|| < 1e-6
- [ ] **Non-zero input: ||x_next - x_t|| > 1e-8** (control has effect!)
- [ ] Backward pass: status = 0, rank = 6, rel_residual < 1e-10
- [ ] **grad_u_t ≠ 0 for generic inputs** (control gradients exist!)

### New Test Cases

**Test 1: Control authority**
```cpp
x_t = [0, 0, 0, 0, 0, 0];
u_t = [0.1, 0, 0];
// Verify: x_next[3:5] ≠ 0 (velocity changed by actuation)
```

**Test 2: Backward control gradient**
```cpp
// After forward pass with u_t = [0.1, 0, 0]:
grad_x_next = [0, 0, 0, 1, 0, 0];  // gradient on v_x
dynamics_backward(...);
// Verify: grad_u_t ≠ 0 (control affects velocity)
```

---

## SUMMARY OF CHANGES

### BLOCKER A FIX: Add Actuation Coupling

**Before**:
```
B = 0  →  ∂x_{t+1}/∂u_t = 0  (uncontrollable)
```

**After**:
```
B = [0           ]
    [-dt*K*J_u_zc]

where K*J_u_zc comes from equilibrium Jacobians.

Result: ∂x_{t+1}/∂u_t ≠ 0 (controllable)
```

### BLOCKER B FIX: Remove Newton Solver

**Before**:
```
Newton iteration: 20 loops, convergence checks, failure modes
```

**After**:
```
Direct solve: A * x_{t+1} = -C*x_t - B*u_t
One FullPivLU solve, no iteration
```

### Key Benefits

1. **Controllable**: MPC/iLQR/RL now possible
2. **Simpler**: No Newton iteration, one linear solve
3. **Faster**: O(1) solve vs O(k) iterations
4. **Robust**: No convergence failures
5. **Minimal**: Uses existing equilibrium Jacobians, no new physics

---

## REFERENCES

### Existing Repo Functions Used

| Component | Source | Lines |
|-----------|--------|-------|
| J_u_zc | `equilibrium_forward` | `CRM_DiffEquilibrium.cpp:103-112` |
| K_tip | `equilibrium_forward` | `CRM_DiffEquilibrium.cpp:116-124` |
| FullPivLU pattern | `equilibrium_backward` | `CRM_DiffEquilibrium.cpp:163-194` |

**No modifications to CP1.x equilibrium solver required.**

---

## ACTION ITEMS

1. Update `DYNAMICS_V1_1_DESIGN.md` Decision Log with this patch
2. Update `CP2_1_IMPLEMENTATION_PLAN.md`:
   - Replace Section 5 (Newton) with direct solve
   - Update Section 3.3 (B matrix)
   - Update Section 6 (smoke test expectations)
   - Update Section 9 (acceptance criteria)
3. Proceed with CP2.1 implementation using patched design

---

**Document Version**: CP2.0 Patch
**Created**: 2025-12-31
**Status**: Ready to apply to v1.1 design
