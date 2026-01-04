# Legacy Hybrid State Contract

**Document Type**: Authoritative Reference
**Source**: CRM_DiffSim_Cl/main (legacy ground truth)
**Date**: 2026-01-03

---

## 1. Step Signature

### Forward Pass

```cpp
int dynamics_forward(
    const double x_t[6],                    // Current state [u_0, v_0]
    const double u_t[NUM_ACT_SET*3],        // Actuation currents (Amperes)
    double dt,                              // Time step (seconds)
    double L_inserted,                      // Insertion length (mm)
    const CRMForwardKinematicsData& params, // Fixed physics parameters
    DynamicsStepResult& out                 // Output: x_next + observables + Jacobians
);
```

**Evidence**: `src/CRM_DiffDynamics.hpp:48-55`

### Backward Pass (Implicit VJP)

```cpp
int dynamics_backward(
    const DynamicsStepResult& fwd_result,   // Cached forward result
    const double grad_x_next[6],            // Upstream gradient dL/dx_{t+1}
    const CRMForwardKinematicsData& params,
    double grad_x_t[6],                     // Output: dL/dx_t
    double grad_u_t[NUM_ACT_SET*3],         // Output: dL/du_t
    int* lu_rank = nullptr,                 // Diagnostic: rank from FullPivLU
    double* rel_residual = nullptr          // Diagnostic: solve residual
);
```

**Evidence**: `src/CRM_DiffDynamics.hpp:59-67`

---

## 2. Exact Hybrid State Vector (6D)

| Index | Field | Meaning | Shape | Units | Frame | Classification |
|-------|-------|---------|-------|-------|-------|----------------|
| 0:3 | `u_0` | Base curvature | (3,) | 1/mm | Body | **DYNAMIC** |
| 3:6 | `v_0` | Base curvature velocity | (3,) | 1/mm/s | Body | **DYNAMIC** |

**Evidence**: `src/CRM_DiffDynamics.hpp:11` — `double x_next[6]; // Next state: [u_0_{t+1}, v_0_{t+1}]`

---

## 3. Observables (NOT State)

| Field | Meaning | Shape | Units | Classification |
|-------|---------|-------|-------|----------------|
| `p_tip` | Tip position | (3,) | mm | **QUASI-STATIC** (equilibrium output) |
| `u_tip` | Tip curvature | (3,) | 1/mm | **QUASI-STATIC** (equilibrium output) |

**Evidence**: `src/CRM_DiffDynamics.hpp:15-16`

```cpp
double p_tip[3];  // Tip position at t+1 (mm)
double u_tip[3];  // Tip curvature at t+1 (1/mm)
```

---

## 4. Hybrid Meaning

### Dynamic Variables
- `u_0`, `v_0` are time-integrated via implicit Euler scheme
- Updated each timestep based on physics ODE

### Quasi-Static Variables
- `p_tip`, `u_tip` are solved via BVP equilibrium each step
- Depend on `u_0` through equilibrium relationship

### Equilibrium Solver
- **Residual function**: `CRMShootingMethodBVP`
- **Unknowns**: Base curvature perturbation `deltau0[3]`
- **Evidence**: `src/CRM_BVPIVP_APIDeclarations.hpp:345-347`

---

## 5. Diagnostics Contract

| Field | Type | Meaning | Values |
|-------|------|---------|--------|
| `converged` | int | Solver status | 0=success, >0=failed |
| `lu_rank` | int | Matrix rank from FullPivLU | Expected: 6 |
| `solve_residual` | double | `\|\|Ax - b\|\| / \|\|b\|\|` | Should be < 1e-10 |
| `exit_code` | int | Exit classification | 0=OK, 1=rank-deficient, 2=residual large |

**Evidence**: `src/CRM_DiffDynamics.hpp:39-43`

---

## 6. Cached Jacobians (for Implicit Backward)

| Field | Shape | Meaning |
|-------|-------|---------|
| `J_G_xnext` | 6x6 | dG/dx_{t+1} |
| `J_G_xt` | 6x6 | dG/dx_t |
| `J_G_ut` | 6x3N | dG/du_t |
| `J_p_u0` | 3x3 | dp_tip/du_0 |
| `J_p_ut` | 3x3N | dp_tip/du_t |

**Evidence**: `src/CRM_DiffDynamics.hpp:18-25`

---

## 7. Evidence Summary

| Claim | File | Lines | Symbol |
|-------|------|-------|--------|
| State is 6D | `src/CRM_DiffDynamics.hpp` | 11 | `x_next[6]` |
| p_tip is observable | `src/CRM_DiffDynamics.hpp` | 15 | `p_tip[3]` |
| Forward signature | `src/CRM_DiffDynamics.hpp` | 48-55 | `dynamics_forward` |
| Backward signature | `src/CRM_DiffDynamics.hpp` | 59-67 | `dynamics_backward` |
| Batched backward | `src/CRM_DiffDynamics.hpp` | 75-84 | `dynamics_backward_batched` |
| Equilibrium result | `src/CRM_DiffEquilibrium.hpp` | 9-28 | `EquilibriumResult` |
| BVP solver | `src/CRM_BVPIVP_APIDeclarations.hpp` | 345-347 | `CRMShootingMethodBVP` |

---

## Acceptance Criteria

Another engineer can open legacy code and confirm every variable:
1. State vector is exactly 6D = [u_0(3), v_0(3)]
2. p_tip and u_tip are outputs, not state elements
3. Jacobians are cached in forward for use in backward
4. Implicit backward uses cached Jacobians, not finite differences
