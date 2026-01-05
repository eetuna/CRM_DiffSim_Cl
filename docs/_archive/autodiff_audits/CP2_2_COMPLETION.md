# CP2.2 Completion Summary

**Checkpoint**: CP2.2 — Dynamics Finite Difference Validation
**Status**: ✅ PASS
**Date**: 2025-12-31
**Commit**: `2171d4a` "CP2.2: Fix dynamics_backward to include matrix-dependence via FD dA/du, dB/du"

---

## What Was Built

A finite difference validation test (`test_cp22_dynamics_fd.cpp`) that validates the analytical gradients from `dynamics_backward` against numerical finite differences at three operating points:

- **OP1 (Rest)**: x_t = [0,0,0,0,0,0], u_t = [0,0,0]
- **OP2 (Actuated)**: x_t = [0,0,0,0,0,0], u_t = [0.1,0,0]
- **OP3 (Moving)**: x_t = [0.01,0,0,0.1,0,0], u_t = [0.1,0,0]

The test computes:
- **J_fd_x** = ∂x_next/∂x_t (6×6) via forward finite differences (ε=1e-6)
- **J_fd_u** = ∂x_next/∂u_t (6×3) via forward finite differences (ε=1e-6)
- **J_analytical** = Jacobians assembled from VJP (backward mode)

Acceptance criterion: `rel_err < 1e-4` for both gradients at all operating points.

---

## Problem Discovered

**Initial result**: FAIL at OP2 and OP3 with **rel_err_u ≈ 0.139** (~14% error)

**Root cause**: The dynamics residual is:
```
r(x_next, x_t, u_t) = A(u_t)·x_next + C·x_t + B(u_t)·u_t = 0
```

where:
- **A(u_t)** depends on u_t through K_tip(u_t) from equilibrium
- **B(u_t)** depends on u_t through K_tip(u_t) and J_u_zc(u_t) from equilibrium

The original `dynamics_backward` implementation only computed:
```cpp
grad_ut = -B^T * lambda
```

This captures the **direct** dependency ∂r/∂u = B, but **ignores** that A and B themselves depend on u.

---

## Fix Implemented

Updated `dynamics_backward` to include **matrix-dependence** terms. The complete adjoint formula is:

```
∂r/∂u_i = (∂A/∂u_i)·x_next + (∂B/∂u_i)·u_t + B[:,i]
```

Then:
```
grad_u_i = -(∂r/∂u_i)^T · λ
```

where λ solves `A^T λ = grad_x_next`.

### Implementation Details

For each control dimension i ∈ {0,1,2}:

1. **Perturb** u_t[i] by ε = 1e-6
2. **Call** `equilibrium_forward(u_t + ε·e_i)` to get K_tip_pert and J_u_zc_pert
3. **Recompute** A_pert and B_pert from perturbed physics matrices
4. **Finite difference**:
   - dA/du_i ≈ (A_pert - A_base) / ε
   - dB/du_i ≈ (B_pert - B_base) / ε
5. **Assemble** dr/du_i = dA/du_i · x_next + dB/du_i · u_t + B[:,i]
6. **Compute** grad_u[i] = -dr/du_i^T · λ

### Caching Strategy

Added to `DynamicsStepResult`:
```cpp
double u_t_cached[3];           // Control input
double dt_cached;               // Time step
double L_inserted_cached;       // Insertion length
double K_tip_cached[9];         // K_tip from base equilibrium
double J_u_zc_cached[9];        // J_u_zc from base equilibrium
```

This allows `dynamics_backward` to recompute perturbed matrices without requiring the user to pass extra arguments.

### Signature Change

**Before**:
```cpp
int dynamics_backward(
    const DynamicsStepResult& fwd_result,
    const double grad_x_next[6],
    double grad_x_t[6],
    double grad_u_t[3],
    int* lu_rank = nullptr,
    double* rel_residual = nullptr
);
```

**After**:
```cpp
int dynamics_backward(
    const DynamicsStepResult& fwd_result,
    const double grad_x_next[6],
    const CRMForwardKinematicsData& params,  // ← Added for equilibrium calls
    double grad_x_t[6],
    double grad_u_t[3],
    int* lu_rank = nullptr,
    double* rel_residual = nullptr
);
```

---

## Key Equations

### Dynamics Residual
```
r = A(u)·x_next + C·x_t + B(u)·u = 0
```

### Adjoint System
```
A(u)^T · λ = grad_x_next
```

### Complete Gradients
```
grad_x_t = -C^T · λ
grad_u_i = -[(∂A/∂u_i)·x_next + (∂B/∂u_i)·u + B[:,i]]^T · λ
```

### Matrix Dependencies
```
A(u) = [ I,      -dt·I     ]
       [ dt·K(u), M(u)+dt·D(u) ]

B(u) = [     0          ]
       [ -dt·K(u)·J_u_zc(u) ]

where K(u) = K_tip(u) from equilibrium_forward(u, Li, params)
      J_u_zc(u) from equilibrium_forward(u, Li, params)
```

---

## Final Results

```
Operating Point: OP1 (Rest)
  rel_err_x = 1.19e-16   ✓ PASS
  rel_err_u = 7.76e-07   ✓ PASS

Operating Point: OP2 (Actuated)
  rel_err_x = 1.48e-12   ✓ PASS
  rel_err_u = 3.43e-06   ✓ PASS  [Fixed: was 0.139]

Operating Point: OP3 (Moving)
  rel_err_x = 1.51e-12   ✓ PASS
  rel_err_u = 3.43e-06   ✓ PASS  [Fixed: was 0.139]

Verdict: PASS — All operating points validated successfully
```

**Improvement**: Reduced control gradient error from **13.9%** to **0.0003%** (40,000× improvement).

---

## Build and Test Commands

```bash
# From repository root
cd /workspaces/CRM_DiffSim_Cl/build
cmake ..
make test_cp22_dynamics_fd

# Run test directly
cd /workspaces/CRM_DiffSim_Cl
./build/test_cp22_dynamics_fd

# Run via CTest
cd /workspaces/CRM_DiffSim_Cl/build
ctest -R dynamics_fd_cp22 --output-on-failure

# Verify all CP2 tests pass
ctest -R dynamics --output-on-failure
```

---

## Files Modified

| File | Change |
|------|--------|
| `src/CRM_DiffDynamics.hpp` | Added cache fields (u_t, dt, Li, K_tip, J_u_zc); updated backward signature |
| `src/CRM_DiffDynamics.cpp` | Implemented matrix-dependence via FD in backward pass |
| `test_cp22_dynamics_fd.cpp` | New FD validation test (322 lines) |
| `test_cp21_dynamics_smoke.cpp` | Updated backward call to include params argument |
| `CMakeLists.txt` | Added test_cp22_dynamics_fd target and CTest entry |
| `docs/autodiff_audits/CP2_2_report.md` | Full diagnostic report |

---

## Impact on Existing Code

- **CP2.1 smoke test**: Still passes (updated to new backward signature)
- **Equilibrium code**: No changes
- **Forward dynamics**: No changes (only caching added)
- **Python bindings**: Not yet implemented (CP2.3)

---

## Architectural Notes

### Why Finite Differences for ∂A/∂u and ∂B/∂u?

The matrices A and B depend on u through the equilibrium solve:
```
u → equilibrium_forward → (K_tip, J_u_zc) → compute_jacobians → (A, B)
```

To get ∂A/∂u analytically would require:
1. Implementing backward mode for `equilibrium_forward` ✗ (not yet available)
2. Implementing backward mode for `compute_jacobians` ✗ (matrix construction)
3. Chain rule through both ✗ (complex)

Instead, we use **forward-mode finite differences** on the equilibrium outputs:
- Pro: Simple, robust, reuses existing equilibrium_forward
- Pro: Numerically accurate with ε=1e-6
- Con: Costs 3 extra equilibrium solves per backward call (one per control dimension)
- Con: Not as efficient as full analytical derivatives

This is a **pragmatic hybrid**: analytical adjoint for the main solve (A^T λ = v), numerical for matrix dependencies.

### Performance Characteristics

- **Forward pass**: ~1 equilibrium solve + 1 linear solve (FullPivLU 6×6)
- **Backward pass**: ~3 equilibrium solves (FD) + 1 adjoint solve (FullPivLU 6×6)
- **Total cost**: ~4 equilibrium solves per forward-backward pair

For NUM_ACT_SET=1 (3 controls), this is acceptable. Future optimization could cache equilibrium Jacobians to enable analytical ∂K/∂u and ∂J_u_zc/∂u.

---

## Lessons Learned

1. **Matrix-valued dependencies require care**: When intermediate matrices depend on inputs, the adjoint must account for these dependencies.

2. **FD validation is essential**: Without CP2.2, the missing terms would have propagated silently through optimization, causing incorrect gradients.

3. **Hybrid approaches work**: Analytical adjoints + numerical matrix derivatives = practical and verifiable.

4. **Test at multiple operating points**: OP1 (rest) passed even with incomplete gradients because the matrix-dependence was negligible. OP2/OP3 revealed the bug.

---

## Status for Next Checkpoint

✅ **CP2.1**: Dynamics primitive implemented and passing
✅ **CP2.2**: FD validation implemented and passing
⬜ **CP2.3**: Python bindings for dynamics (next)
⬜ **CP2.4**: End-to-end trajectory optimization test (future)

---

**Completion Date**: 2025-12-31
**Verified By**: CTest (dynamics_smoke_cp21, dynamics_fd_cp22)
**Documentation**: Complete
**Ready for**: CP2.3 (Python bindings)
