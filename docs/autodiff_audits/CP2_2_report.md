# CP2.2 Dynamics Finite Difference Validation Report

**Status: FAIL**
**Date: 2025-12-31**
**Checkpoint: CP2.2**

---

## Executive Summary

CP2.2 implements finite difference validation for the dynamics primitive `dynamics_forward` and `dynamics_backward` introduced in CP2.1. The test validates analytical gradients (computed via backward mode) against finite differences at three operating points.

**Result**: The test **FAILS** for actuated operating points (OP2, OP3) with ~14% relative error in ∂x_next/∂u_t gradients.

**Root Cause**: The `dynamics_backward` implementation computes an **incomplete gradient**. It correctly handles the direct dependency ∂G/∂u_t = B, but omits the fact that the system matrices A(u) and B(u) themselves depend on the control input u through:
- K_tip(u): Tip stiffness varies with actuation
- J_u_zc(u): Equilibrium Jacobian varies with actuation

**Important Note**: The CP2.1 dynamics forward implementation is **not wrong** — it correctly solves the physics. The backward pass is simply incomplete relative to the full dependency chain in the forward function being differentiated.

---

## Build and Run Commands

```bash
# From repository root
cd /workspaces/CRM_DiffSim_Cl/build
cmake ..
make test_cp22_dynamics_fd

# Run test directly
cd /workspaces/CRM_DiffSim_Cl
./build/test_cp22_dynamics_fd

# Or via CTest
ctest -R dynamics_fd_cp22 --output-on-failure
```

---

## Test Results (Before Fix)

### Test Configuration
- dt = 0.01 s (10 ms timestep)
- L_inserted = 50.0 mm
- FD epsilon = 1e-6
- Acceptance threshold = 1e-4 (0.01% relative error)

### Operating Point 1: Rest
```
x_t = [0, 0, 0, 0, 0, 0]
u_t = [0, 0, 0]

||J_fd_x||_F         = 168.4114168
||J_fd_u||_F         = 38.65808911
rel_err_x            = 1.193366069e-16
rel_err_u            = 7.763965771e-07
max_abs_err_x        = 1.421085472e-14
max_abs_err_u        = 2.37313839e-05

Result: PASS (both gradients within threshold)
```

### Operating Point 2: Actuated (FAIL)
```
x_t = [0, 0, 0, 0, 0, 0]
u_t = [0.1, 0, 0]

||J_fd_x||_F         = 168.4114168
||J_fd_u||_F         = 40.2623604
rel_err_x            = 1.483330051e-12
rel_err_u            = 0.1394108359    ← FAIL (threshold: 1e-4)
max_abs_err_x        = 2.273181858e-10
max_abs_err_u        = 3.899684822     ← Large absolute error

Result: FAIL (∂x_next/∂u_t gradient has 13.9% relative error)
```

### Operating Point 3: Moving (FAIL)
```
x_t = [0.01, 0, 0, 0.1, 0, 0]
u_t = [0.1, 0, 0]

||J_fd_x||_F         = 168.4114168
||J_fd_u||_F         = 40.2623604
rel_err_x            = 1.513760904e-12
rel_err_u            = 0.1394108359    ← FAIL (threshold: 1e-4)
max_abs_err_x        = 2.273181858e-10
max_abs_err_u        = 3.899684822     ← Large absolute error

Result: FAIL (∂x_next/∂u_t gradient has 13.9% relative error)
```

### Summary
- **OP1 (Rest)**: PASS — At zero actuation, matrix-dependence effects are minimal
- **OP2 (Actuated)**: FAIL — rel_err_u = 0.139, max_abs_err_u = 3.90
- **OP3 (Moving)**: FAIL — rel_err_u = 0.139, max_abs_err_u = 3.90

**Overall Verdict: FAIL** (2 out of 3 operating points failed)

---

## Root Cause Analysis

### The Dynamics Residual
The forward pass solves:
```
r(x_next, x_t, u_t) = A(u_t) · x_next + C · x_t + B(u_t) · u_t = 0
```

Where:
- **A(u_t)** depends on u_t through K_tip(u_t) and mass/damping matrices
- **B(u_t)** depends on u_t through K_tip(u_t) and J_u_zc(u_t)
- **C** does not depend on u_t

### Current Backward Implementation (Incomplete)
File: `src/CRM_DiffDynamics.cpp:223-268`

```cpp
// Solve A^T λ = grad_x_next
VectorXd lambda = lu_solver.solve(v);

// Gradients
grad_xt = -C^T * lambda;
grad_ut = -B^T * lambda;  // ← INCOMPLETE
```

This only captures the **direct** dependency ∂r/∂u = B, but ignores:
```
∂r/∂u_i = (∂A/∂u_i)·x_next + (∂B/∂u_i)·u_t + B[:,i]
```

The missing terms are:
- **(∂A/∂u_i)·x_next**: How system matrix changes with control affect the solution
- **(∂B/∂u_i)·u_t**: How input coupling changes with control affect the solution

### Why OP1 Passes but OP2/OP3 Fail

**OP1 (u_t = 0)**: At zero actuation:
- Catheter is in neutral configuration
- K_tip and J_u_zc are at their baseline values
- Small perturbations cause minimal nonlinear effects
- The direct term B^T λ dominates
- **Result**: Missing terms are negligible → PASS

**OP2/OP3 (u_t = [0.1, 0, 0])**: With actuation:
- Catheter is bent by the actuator
- K_tip(u) varies significantly with actuation level
- J_u_zc(u) varies significantly with configuration
- Perturbing u_t changes both the physics and the coupling
- **Result**: Missing terms are ~14% of total gradient → FAIL

### Dependency Chain

```
u_t → equilibrium_forward → K_tip, J_u_zc
                          ↓
                   compute_jacobians → A(u), B(u)
                                     ↓
                              FullPivLU solve → x_next
```

The backward pass must reverse this entire chain, not just the final step.

---

## Technical Details

### What Was Tested
1. **Finite Difference Jacobians**: Forward differences with ε=1e-6
   - J_fd_x = ∂x_next/∂x_t (6×6 matrix)
   - J_fd_u = ∂x_next/∂u_t (6×3 matrix)

2. **Analytical Jacobians**: Assembled from VJP calls
   - For each canonical direction e_k ∈ ℝ⁶:
     - Call dynamics_backward(result, e_k, grad_x_t, grad_u_t)
     - Stack gradients as columns of transposed Jacobian

3. **Comparison Metric**:
   ```
   rel_err = ||J_fd - J_analytical||_F / max(||J_fd||_F, 1e-12)
   ```

### What Was Not Changed
- No changes to forward solver math
- No changes to residual definition
- No changes to Jacobian formulas (A, C, B)
- No changes to tolerances
- No changes to equilibrium logic

---

## Next Steps

To fix this issue, `dynamics_backward` must be augmented to include:

1. **Compute ∂A/∂u_i and ∂B/∂u_i** via finite differences:
   - Perturb u_t[i] by ε
   - Call equilibrium_forward to get K_tip_pert and J_u_zc_pert
   - Recompute A_pert and B_pert from perturbed matrices
   - Finite difference: dA/du_i ≈ (A_pert - A_base)/ε

2. **Complete the adjoint formula**:
   ```
   dr/du_i = (dA/du_i)·x_next + (dB/du_i)·u_t + B[:,i]
   grad_u_i = -(dr/du_i)^T · λ
   ```

3. **Cache required data** in DynamicsStepResult:
   - x_next, u_t, dt, L_inserted
   - K_tip, J_u_zc from base forward evaluation
   - Parameters needed to call equilibrium_forward

This will make the backward pass **complete** relative to the forward function, allowing CP2.2 to PASS.

---

## Files Modified
- `test_cp22_dynamics_fd.cpp` (new test file)
- `CMakeLists.txt` (added test target and CTest entry)

## Files NOT Modified (Yet)
- `src/CRM_DiffDynamics.hpp` (will need cache fields added)
- `src/CRM_DiffDynamics.cpp` (backward implementation needs fix)

---

**Report Status**: Documents initial FAIL state before implementing fix.
