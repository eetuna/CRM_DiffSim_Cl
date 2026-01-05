# Gradient Pathway Analysis for grad_u

## Goal
Identify if `grad_u_direct` and `grad_u_implicit` double-count the magnetic torque pathway.

## Backward Pass Order (CRM_TrueLegacyDynamics.cpp:172-464)

### Step 1: Initialize outputs (lines 184-194)
```cpp
grad_x_coil[j][i] = 0.0  // All initialized to zero
grad_xf[i] = 0.0
grad_u[j][i] = 0.0
```

### Step 2: Build cotangent on xf_next (lines 196-201)
```cpp
v_xf_next[0:3] = grad_tip_p[0:3]  // Upstream gradient ∂L/∂tip_p
v_xf_next[3:15] = 0.0
```

### Step 3: Compute IVP Jacobians (lines 203-290)
Returns: `J_u, J_n, J_p, J_R, J_ftip`
- `J_u`: (15 × 3) = ∂xf_next/∂u0 (base curvature)
- `J_n`: (15 × 3N) = ∂xf_next/∂nL (interface forces)
- `J_p`: (15 × 3N) = ∂xf_next/∂p_coil (coil positions)
- `J_R`: (15 × 9N) = ∂xf_next/∂R_coil (coil orientations)

### Step 4: Push cotangent through IVP Jacobians (lines 292-299)
```cpp
v_u0 = J_u^T * v_xf_next
v_nL = J_n^T * v_xf_next
v_p_coil = J_p^T * v_xf_next
v_R_coil = J_R^T * v_xf_next
```

### Step 5: Assemble BVP cotangent v_y (lines 301-323)
```cpp
v_y[mL components] = 0.0  // Moment cotangents (set to zero!)
v_y[nL components] = v_nL  // Force cotangents
```
**NOTE**: Moment components are zeroed! This loses gradient information.

### Step 6: Compute BVP Jacobians (lines 325-333)
```cpp
compute_bvp_jacobians_full_analytic(..., J_yy, J_yu, J_yx)
```
Where:
- `J_yy`: (6N × 6N) = ∂r/∂y where y = [mL; nL]
- `J_yu`: (6N × 3N) = ∂r/∂u
- `J_yx`: (6N × (18N+15)) = ∂r/∂x_t

### Step 7: Solve adjoint system (lines 335-346)
```cpp
lambda = solve((J_yy)^T, v_y)  // QR decomposition
```

### Step 8: Direct gradients for xf (lines 349-352)
```cpp
grad_xf[i] = v_xf_next[i]  // Direct passthrough
```

### Step 9: Direct gradients for x_coil (lines 354-370)
```cpp
// FROM IVP Jacobians ONLY (J_p, J_R)
grad_x_coil[j][6:9] = v_p_coil[j*3:(j+1)*3]  // Position
grad_x_coil[j][9:18] = v_R_coil[j*9:(j+1)*9]  // Orientation
grad_x_coil[j][0:6] = 0.0  // Velocity and angular velocity
```

### Step 10: Add implicit state gradients (lines 372-386)
```cpp
implicit_grad_x = J_yx^T * lambda

grad_xf[i] -= implicit_grad_x[18N + i]
grad_x_coil[j][i] -= implicit_grad_x[j*18 + i]  // ALL components, including w (indices 3-5)!
```
**CRITICAL**: After this, `grad_x_coil[j][3:6]` contains BOTH:
- Direct contribution (was 0.0 from Step 9)
- Implicit contribution from BVP adjoint

### Step 11: Compute grad_u_direct from magnetic torque (lines 388-442)
```cpp
// Compute ∂τ/∂u (lines 393-417)
dTau_du[j](k, i) = magnetic torque derivative

// Initialize (lines 420)
grad_u_direct = zeros(3N)

// Loop over coils (lines 423-442)
for j in actuators:
    v_w[k] = grad_x_coil[j][3 + k]  // Uses MODIFIED grad_x_coil from Step 10!

    for i in [0,1,2]:
        grad_u_direct[j*3 + i] += dt * dTau_du[j](k, i) / I_k * v_w[k]  // For k=0,1,2
```
**ISSUE**: `v_w` here includes implicit contribution from Step 10!

### Step 12: Compute grad_u_implicit (lines 444-445)
```cpp
grad_u_implicit = -J_yu^T * lambda
```

### Step 13: Combine gradients (lines 447-454)
```cpp
grad_u_vec = grad_u_direct + grad_u_implicit

grad_u[j][i] = grad_u_vec[j*3 + i]
```

## Double-Counting Analysis

### Question: Does magnetic torque pathway appear twice?

**Path 1 (Direct)**:
```
u → τ_mag → w_next → xf_next → L
```
Captured by: `grad_u_direct` (Step 11)

**Path 2 (Implicit via BVP)**:
```
u → (affects BVP residual) → (mL, nL) → ... → xf_next → L
```
Captured by: `grad_u_implicit` (Step 12)

### The Problem

At Step 11, `v_w = grad_x_coil[3:6]` includes:
1. Direct contribution: 0.0 (from Step 9)
2. **Implicit contribution**: -J_yx[w components, :]^T * lambda (from Step 10)

The implicit contribution represents: (∂L/∂(mL,nL))^T × (∂(mL,nL)/∂w)

Then at Step 12, `grad_u_implicit = -J_yu^T * lambda` represents: (∂L/∂(mL,nL))^T × (∂(mL,nL)/∂u)

**If J_yu includes moment rows**, then:
- `grad_u_direct` gets: (∂L/∂(mL))^T × (∂(mL)/∂w) × (∂w/∂u)
- `grad_u_implicit` gets: (∂L/∂(mL))^T × (∂(mL)/∂u)

Since (∂(mL)/∂u) = (∂(mL)/∂w) × (∂w/∂u) by chain rule, **THIS IS DOUBLE-COUNTING**.

## Solution

**Option 1**: Compute `grad_u_direct` BEFORE Step 10 (using v_w with only direct contribution)

**Option 2**: Set `J_yu[moment rows, :] = 0` to avoid BVP-mediated moment pathway

**Option 3**: Don't add implicit w contribution in Step 10 (modify J_yx to exclude w rows)

## Current Implementation Status

The current code has `J_yu[moment rows, :] = 0` (our recent "fix"), which is **Option 2**.

But we still have issues because:
1. `v_y[moment components] = 0` (Step 5) loses gradient information
2. `J_yu[force rows, :] != 0` but with heuristic coefficient 0.122

## Next Steps

1. Verify `v_y[moment components]` should be non-zero
2. Derive correct `J_yu[force rows, :]` from BVP residual definition (no heuristics)
3. Ensure no other double-counting pathways
