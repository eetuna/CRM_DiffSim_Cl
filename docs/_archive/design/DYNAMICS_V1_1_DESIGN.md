# Design: Differentiable Dynamics Primitive (v1.1)

**Goal**: Add a differentiable one-step dynamics primitive `x_{t+1} = step(x_t, u_t; θ_fixed)` suitable for MPC/iLQR/RL, with stable backward pass using implicit differentiation.

**Scope**: NO θ-learning yet. Parameters θ (material properties, geometry) are fixed. This design focuses on ∂x_{t+1}/∂x_t and ∂x_{t+1}/∂u_t.

---

## 1. Minimal State Vector x_t

### Design Decision: Reduced Coordinates

For MPC/iLQR over receding horizon (10-50 steps), we need a **low-dimensional Markovian state**. The full distributed catheter configuration is too high-dimensional (100s of DOFs).

### State Definition

```
x_t ∈ R^6:
  x_t = [u_0,    // Base curvature (3D): Δu_0 at catheter base
         v_0]    // Base curvature rate (3D): d(Δu_0)/dt

where:
  u_0 ∈ R^3  // [u_0x, u_0y, u_0z] - base curvature vector
  v_0 ∈ R^3  // [v_0x, v_0y, v_0z] - time derivative of u_0
```

**Rationale**:
- **u_0**: Captures the boundary condition that determines the entire catheter shape (via equilibrium solve at each timestep)
- **v_0**: Provides velocity information for second-order dynamics (inertia + damping)
- **Markovian**: This pair (u_0, v_0) contains sufficient information to predict the next state given control u_t
- **Observable**: Both can be estimated from sensors (strain gauges at base, or differentiated from u_0 measurements)

**What is NOT in the state**:
- Tip position p_tip: Computed on-demand via equilibrium_forward(u_0, u_t, Li) → this keeps state minimal
- Distributed shape: Recomputed each step via equilibrium solver
- Coil currents: Treated as direct control (electrical dynamics assumed fast/quasi-static)

**Extensibility for future θ-learning**:
- When adding ∂/∂θ, no state dimension change needed
- θ parameters (EI, GJ, segment lengths, etc.) are separate from state x_t

---

## 2. Public C++ API

### 2.1 Forward Pass

```cpp
namespace CRMCatheterModel {

// Result structure for one-step dynamics forward pass
struct DynamicsStepResult {
    // Outputs
    double x_next[6];                // Next state: [u_0_{t+1}, v_0_{t+1}]
    int converged;                   // 0=success, >0=failed

    // Observables (computed via equilibrium at t+1)
    double p_tip[3];                 // Tip position at t+1 (mm)
    double u_tip[3];                 // Tip curvature at t+1 (1/mm)

    // Cached Jacobians for backward pass (row-major storage)
    double J_G_xnext[36];            // ∂G/∂x_{t+1} (6×6)
    double J_G_xt[36];               // ∂G/∂x_t (6×6)
    double J_G_ut[6*NUM_ACT_SET*3];  // ∂G/∂u_t (6×3N)

    // Equilibrium Jacobians at t+1 (for computing ∂p_tip/∂x_{t+1})
    double J_p_u0[9];                // ∂p_tip/∂u_0 at t+1 (3×3)
    double J_p_ut[3*NUM_ACT_SET*3];  // ∂p_tip/∂u_t (3×3N)

    // Diagnostics
    int nl_iterations;               // Implicit solve iterations
    double final_residual;           // ||G(x_{t+1})||
    int lu_rank;                     // rank(J_G_xnext) from FullPivLU
    double rel_solve_residual;       // ||A^T λ - rhs|| / max(||rhs||, 1)
    int exit_code;                   // 0=OK, 1=rank-deficient, 2=residual too large
};

// Forward pass: solve for x_{t+1} given (x_t, u_t, dt)
// Returns: 0=success, non-zero=failure
int dynamics_forward(
    const double x_t[6],                    // Current state [u_0, v_0]
    const double u_t[NUM_ACT_SET*3],        // Actuation currents (Amperes)
    double dt,                              // Time step (seconds)
    double L_inserted,                      // Insertion length (mm)
    const CRMForwardKinematicsData& params, // Physics parameters
    DynamicsStepResult& out                 // Output + cached Jacobians
);

} // namespace CRMCatheterModel
```

**Key points**:
- **State x_t = [u_0, v_0]**: 6D vector
- **Control u_t**: Same as equilibrium primitive (3*NUM_ACT_SET actuation currents)
- **Output x_next**: Next state at t+1
- **Cached Jacobians**: All matrices needed for implicit VJP (see Section 4)
- **Observables**: Tip position/curvature computed via equilibrium_forward at t+1
- **Diagnostics**: Same pattern as equilibrium primitive (rank, residual checks, fail-fast)

---

### 2.2 Backward Pass

```cpp
namespace CRMCatheterModel {

// Backward pass: compute VJP (∂L/∂x_t, ∂L/∂u_t) from upstream gradient ∂L/∂x_{t+1}
// Uses implicit function theorem: A^T λ = (∂L/∂x_{t+1})^T, then compute partials
// Returns: 0=success, 1=rank-deficient, 2=residual too large
int dynamics_backward(
    const DynamicsStepResult& fwd_result,  // Cached forward result
    const double grad_x_next[6],           // Upstream gradient ∂L/∂x_{t+1}
    double grad_x_t[6],                    // Output: ∂L/∂x_t
    double grad_u_t[NUM_ACT_SET*3],        // Output: ∂L/∂u_t
    int* lu_rank = nullptr,                // Optional: rank from FullPivLU
    double* rel_residual = nullptr         // Optional: ||A^T λ - rhs|| / max(||rhs||, 1)
);

} // namespace CRMCatheterModel
```

**Key points**:
- **Inputs**: Upstream gradient ∂L/∂x_{t+1}, cached forward result
- **Outputs**: ∂L/∂x_t, ∂L/∂u_t (VJP through one dynamics step)
- **Implicit differentiation**: Solves linear system A^T λ = ∂L/∂x_{t+1} where A = J_G_xnext
- **Diagnostics**: Rank and residual checks (fail-fast if rank-deficient or inaccurate)

---

## 3. Implicit Residual Form

### 3.1 Continuous Dynamics (ODE)

The catheter base curvature follows second-order dynamics:

```
M d²u_0/dt² + D du_0/dt + f(u_0, u_t) = 0
```

where:
- **M**: Inertia matrix (3×3, depends on catheter mass distribution)
- **D**: Damping matrix (3×3, structural damping)
- **f(u_0, u_t)**: Restoring force from elastic deformation + magnetic actuation

**First-order form**: x = [u_0, v_0] where v_0 = du_0/dt
```
du_0/dt = v_0
dv_0/dt = M^{-1} (-D v_0 - f(u_0, u_t))
```

---

### 3.2 Discrete-Time Residual (Implicit Midpoint or Backward Euler)

**Implicit Midpoint Rule** (2nd-order accurate, symplectic):

```
G(x_{t+1}, x_t, u_t) = x_{t+1} - x_t - dt * ẋ_{mid}  = 0  ∈ R^6

where:
  x_{mid} = (x_t + x_{t+1}) / 2
  ẋ_{mid} = dynamics(x_{mid}, u_t)
```

**Backward Euler** (1st-order, simpler, more stable):

```
G(x_{t+1}, x_t, u_t) = x_{t+1} - x_t - dt * ẋ_{t+1}  = 0  ∈ R^6

where:
  ẋ_{t+1} = [v_{t+1},
             M^{-1}(-D v_{t+1} - f(u_{t+1}, u_t))]
```

**Recommended**: Start with **Backward Euler** (simpler, good stability for stiff systems).

---

### 3.3 Jacobians for Implicit Differentiation

Given residual G(x_{t+1}, x_t, u_t) = 0, we need:

```
A = ∂G/∂x_{t+1}  ∈ R^{6×6}
B = ∂G/∂u_t      ∈ R^{6×3N}  (N = NUM_ACT_SET)
C = ∂G/∂x_t      ∈ R^{6×6}
```

**For Backward Euler**:
```
G = x_{t+1} - x_t - dt * [v_{t+1},
                           M^{-1}(-D v_{t+1} - f(u_{t+1}, u_t))]

A = ∂G/∂x_{t+1} = I - dt * [0,        I     ]
                           [-M^{-1}∂f/∂u_0, -M^{-1}D]
                 = [I,              -dt*I                    ]
                   [dt*M^{-1}∂f/∂u_0, I + dt*M^{-1}D        ]

C = ∂G/∂x_t = -I

B = ∂G/∂u_t = -dt * [0,                  ]
                     [M^{-1} ∂f/∂u_t     ]
```

**Key Jacobian**: `∂f/∂u_0` and `∂f/∂u_t`
- These require differentiating the equilibrium solver (already available from equilibrium_backward!)
- Reuse the existing implicit differentiation machinery

---

### 3.4 Computing ∂f/∂u_0 and ∂f/∂u_t

The restoring force f(u_0, u_t) comes from:
1. Solve equilibrium at current u_0, u_t: equilibrium_forward(u_0, u_t, Li) → p_tip, K_tip, Jacobians
2. Extract force on base from equilibrium constraint

**Key insight**: The equilibrium Jacobians J_p_u0, J_u_u0, K_tip already capture sensitivity to u_0 and u_t.

For the dynamics residual:
- **∂f/∂u_0**: Can be approximated via finite differences or derived from K_tip (stiffness at tip projects back to base)
- **∂f/∂u_t**: Comes from ∂(magnetic force)/∂u_t, available from equilibrium Jacobians J_p_zc, J_u_zc

**Practical approach**:
- Call `equilibrium_forward(u_{t+1}, u_t, Li)` to get K_tip and Jacobians
- Use these to populate A and B matrices

---

## 4. Backward Pass Math (VJP)

### 4.1 Implicit Function Theorem

Given residual G(x_{t+1}, x_t, u_t) = 0 solved for x_{t+1}, we have:

```
A dx_{t+1} + C dx_t + B du_t = 0

where:
  A = ∂G/∂x_{t+1}  (6×6)
  C = ∂G/∂x_t      (6×6)
  B = ∂G/∂u_t      (6×3N)
```

Solving for sensitivity:
```
dx_{t+1} = -A^{-1} C dx_t - A^{-1} B du_t
```

Therefore:
```
∂x_{t+1}/∂x_t = -A^{-1} C
∂x_{t+1}/∂u_t = -A^{-1} B
```

---

### 4.2 VJP (Backward Mode AD)

Given upstream gradient v = ∂L/∂x_{t+1} ∈ R^6, compute:
```
∂L/∂x_t = (∂x_{t+1}/∂x_t)^T v = -C^T (A^{-T} v)
∂L/∂u_t = (∂x_{t+1}/∂u_t)^T v = -B^T (A^{-T} v)
```

**Algorithm** (efficient, one linear solve):
```
1. Solve:  A^T λ = v          // λ ∈ R^6 (adjoint variables)
2. Compute: ∂L/∂x_t = -C^T λ  // (6×6)^T * (6×1) → 6×1
3. Compute: ∂L/∂u_t = -B^T λ  // (6×3N)^T * (6×1) → 3N×1
```

**Numerical method**: FullPivLU (same pattern as equilibrium_backward)
- Compute rank(A) to detect singularities
- Check relative residual ||A^T λ - v|| / max(||v||, 1) < tol
- Fail-fast if rank-deficient or inaccurate

---

### 4.3 Pseudocode for dynamics_backward

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
    const double* A = fwd_result.J_G_xnext;  // 6×6
    const double* C = fwd_result.J_G_xt;     // 6×6
    const double* B = fwd_result.J_G_ut;     // 6×3N

    // Map to Eigen (no copy)
    Eigen::Map<const Eigen::Matrix<double, 6, 6>> A_mat(A);
    Eigen::Map<const Eigen::Matrix<double, 6, 6>> C_mat(C);
    Eigen::Map<const Eigen::VectorXd> v(grad_x_next, 6);

    // Solve A^T λ = v using FullPivLU
    Eigen::FullPivLU<Eigen::MatrixXd> lu(A_mat.transpose());

    int rank = lu.rank();
    if (lu_rank) *lu_rank = rank;

    if (rank < 6) {
        // Rank-deficient: dynamics singular
        return 1;
    }

    Eigen::VectorXd lambda = lu.solve(v);

    // Check solve accuracy
    double residual_norm = (A_mat.transpose() * lambda - v).norm();
    double rhs_norm = v.norm();
    double rel_res = residual_norm / std::max(rhs_norm, 1.0);

    if (rel_residual) *rel_residual = rel_res;

    if (rel_res > 1e-10) {
        // Inaccurate solve
        return 2;
    }

    // Compute gradients
    Eigen::Map<Eigen::VectorXd> grad_xt(grad_x_t, 6);
    grad_xt = -C_mat.transpose() * lambda;

    Eigen::Map<Eigen::VectorXd> grad_ut(grad_u_t, 3*NUM_ACT_SET);
    Eigen::Map<const Eigen::MatrixXd> B_mat(B, 6, 3*NUM_ACT_SET);
    grad_ut = -B_mat.transpose() * lambda;

    return 0;  // Success
}
```

---

## 5. Checkpoint Plan (CP2.x)

### CP2.1: Core Dynamics Primitive (C++)

**Goal**: Implement `dynamics_forward` and `dynamics_backward` in C++.

**Tasks**:
1. Create `src/CRM_DiffDynamics.hpp` and `src/CRM_DiffDynamics.cpp`
2. Implement residual G(x_{t+1}, x_t, u_t) using Backward Euler
3. Implement Jacobian computation (A, B, C) via:
   - Analytical derivatives where simple
   - Reuse equilibrium_forward Jacobians for ∂f/∂u_0, ∂f/∂u_t
4. Implement implicit solver for x_{t+1} (Newton's method, reuse minpack patterns)
5. Implement `dynamics_backward` using FullPivLU (same pattern as equilibrium_backward)

**Acceptance gate**:
- Compiles without errors
- Links against CRMCPPLib
- No runtime crashes on smoke case

**Test**: `test_cp21_dynamics_smoke.cpp`
```cpp
// Smoke case: zero initial state, zero control, small dt
x_t = [0, 0, 0, 0, 0, 0];
u_t = [0, 0, 0];
dt = 0.01;
status = dynamics_forward(x_t, u_t, dt, ...);
assert(status == 0);
```

---

### CP2.2: Finite Difference Validation

**Goal**: Verify ∂x_{t+1}/∂u_t and ∂x_{t+1}/∂x_t match finite differences.

**Tasks**:
1. Create `test_cp22_dynamics_fd.cpp`
2. Test at 3 operating points:
   - Rest: x_t = [0, 0, 0, 0, 0, 0], u_t = [0, 0, 0]
   - Actuated: x_t = [0, 0, 0, 0, 0, 0], u_t = [0.1, 0, 0]
   - Moving: x_t = [0.01, 0, 0, 0.1, 0, 0], u_t = [0.1, 0, 0]
3. Compute FD Jacobians with ε = 1e-6:
   - ∂x_{t+1}/∂u_t: perturb each u_t[i], compute (x_{t+1}(u+εe_i) - x_{t+1}(u)) / ε
   - ∂x_{t+1}/∂x_t: perturb each x_t[i], compute (x_{t+1}(x+εe_i) - x_{t+1}(x)) / ε
4. Compute analytical Jacobians via backward pass:
   - For each canonical direction e_i, call dynamics_backward(fwd, e_i, grad_xt, grad_ut)
   - Assemble Jacobian matrices column-by-column
5. Compare: ||J_FD - J_analytical||_F / ||J_FD||_F < 1e-4

**Acceptance gate**:
- All 3 operating points: relative Frobenius error < 1e-4 for both Jacobians
- Print max absolute error per element

**Test output format**:
```
Test 1/3: Rest configuration
  ∂x_{t+1}/∂u_t: rel_err = 2.3e-5, max_abs_err = 1.1e-7  ✓
  ∂x_{t+1}/∂x_t: rel_err = 1.8e-5, max_abs_err = 8.4e-8  ✓

Test 2/3: Actuated configuration
  ∂x_{t+1}/∂u_t: rel_err = 4.7e-5, max_abs_err = 3.2e-7  ✓
  ∂x_{t+1}/∂x_t: rel_err = 3.1e-5, max_abs_err = 1.9e-7  ✓

Test 3/3: Moving configuration
  ∂x_{t+1}/∂u_t: rel_err = 6.2e-5, max_abs_err = 4.8e-7  ✓
  ∂x_{t+1}/∂x_t: rel_err = 5.4e-5, max_abs_err = 3.6e-7  ✓

PASS: All Jacobians match finite differences within tolerance
```

---

### CP2.3: Python Bindings

**Goal**: Expose `dynamics_forward` and `dynamics_backward` to Python.

**Tasks**:
1. Add bindings in `python/crm_bindings.cpp`:
   ```cpp
   m.def("dynamics_forward", &dynamics_forward_wrapper, ...);
   m.def("dynamics_backward", &dynamics_backward_wrapper, ...);
   ```
2. Create wrapper that returns dict with all outputs + cached Jacobians
3. Write Python wrapper class `CRMDynamics` for convenience

**Acceptance gate**:
- Python can import and call both functions
- Return values match C++ (elementwise within 1e-15)

**Test**: `python/test_cp23_dynamics_binding.py`
```python
import crm_diff_py

x_t = np.array([0, 0, 0, 0, 0, 0])
u_t = np.array([0.1, 0, 0])
result = crm_diff_py.dynamics_forward(x_t, u_t, dt=0.01, ...)
assert result['status'] == 0
assert result['x_next'].shape == (6,)
```

---

### CP2.4: PyTorch gradcheck

**Goal**: Verify `dynamics_backward` satisfies PyTorch's gradcheck (double backward test).

**Tasks**:
1. Create PyTorch wrapper in `python/crm_equilibrium.py`:
   ```python
   class DynamicsStep(torch.autograd.Function):
       @staticmethod
       def forward(ctx, x_t, u_t, dt, ...):
           result = crm_diff_py.dynamics_forward(...)
           ctx.save_for_backward(result)
           return result['x_next']

       @staticmethod
       def backward(ctx, grad_x_next):
           result = ctx.saved_tensors[0]
           grads = crm_diff_py.dynamics_backward(result, grad_x_next, ...)
           return grads['grad_x_t'], grads['grad_u_t'], None, ...
   ```
2. Test with `torch.autograd.gradcheck`:
   ```python
   x_t = torch.randn(6, requires_grad=True, dtype=torch.float64)
   u_t = torch.randn(3, requires_grad=True, dtype=torch.float64)

   func = lambda x, u: DynamicsStep.apply(x, u, 0.01, ...)
   torch.autograd.gradcheck(func, (x_t, u_t), eps=1e-6)
   ```

**Acceptance gate**:
- `gradcheck` passes at 3 operating points (same as CP2.2)
- Relative tolerance: 1e-4
- Absolute tolerance: 1e-6

---

### CP2.5: Short Rollout Smoke Test

**Goal**: Verify multi-step rollout stability (no divergence, no NaNs).

**Tasks**:
1. Create `test_cp25_dynamics_rollout.cpp` (C++)
2. Create `python/test_cp25_rollout_smoke.py` (Python)
3. Test configurations:
   - Zero control: u_t = 0 for all t, run 20 steps
   - Constant control: u_t = [0.1, 0, 0], run 20 steps
   - Sinusoidal control: u_t = 0.1*sin(ωt), run 20 steps
4. Check at each step:
   - status == 0
   - ||x_t|| < 100 (state bounded)
   - No NaNs or Infs
   - Backward pass succeeds (status == 0, rank == 6)
5. Compute accumulated loss L = Σ ||p_tip - p_target||² and backprop through entire rollout

**Acceptance gate** (C++):
- All 3 rollouts complete without errors
- Final state is finite and bounded

**Acceptance gate** (Python):
- All 3 rollouts complete
- Backprop through 20-step rollout succeeds
- Gradients ∂L/∂u_t are finite for all timesteps

**Test output format**:
```
Rollout 1/3: Zero control, 20 steps
  Step  1: x_norm = 0.00, p_tip = [0.0, 0.0, 50.0]  ✓
  Step  5: x_norm = 0.00, p_tip = [0.0, 0.0, 50.0]  ✓
  Step 10: x_norm = 0.00, p_tip = [0.0, 0.0, 50.0]  ✓
  Step 20: x_norm = 0.00, p_tip = [0.0, 0.0, 50.0]  ✓
  Backward: grad_u[0:5] finite, ||grad_u|| = 0.0  ✓

Rollout 2/3: Constant control, 20 steps
  Step  1: x_norm = 0.12, p_tip = [0.3, 0.1, 49.9]  ✓
  Step 20: x_norm = 1.34, p_tip = [2.1, 0.8, 48.2]  ✓
  Backward: ||grad_u|| = 12.4  ✓

PASS: All rollouts stable, backprop succeeds
```

---

### CP2.6: CI Integration

**Goal**: Wire dynamics tests into CTest and GitHub Actions.

**Tasks**:
1. Add to `CMakeLists.txt`:
   ```cmake
   add_executable(test_cp22_dynamics_fd test_cp22_dynamics_fd.cpp)
   add_executable(test_cp25_dynamics_rollout test_cp25_dynamics_rollout.cpp)

   add_test(NAME dynamics_fd_check
            COMMAND test_cp22_dynamics_fd
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

   add_test(NAME dynamics_rollout_smoke
            COMMAND test_cp25_dynamics_rollout
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

   add_test(NAME dynamics_python_gradcheck
            COMMAND python3 ${CMAKE_SOURCE_DIR}/python/test_cp24_gradcheck.py
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
   ```

2. Update `.github/workflows/pr_checks.yml`:
   ```yaml
   - name: Run Dynamics FD Check (CP2.2)
     run: ctest --output-on-failure -R dynamics_fd_check

   - name: Run Dynamics Rollout (CP2.5)
     run: ctest --output-on-failure -R dynamics_rollout_smoke

   - name: Run PyTorch Gradcheck (CP2.4)
     run: ctest --output-on-failure -R dynamics_python_gradcheck
   ```

3. Update `.github/workflows/nightly.yml` to include longer rollouts (100 steps)

**Acceptance gate**:
- All dynamics tests run in CI
- PR checks fail if any test fails (hard gate)
- Nightly tests include extended rollouts

---

## 6. Backward Compatibility with Future θ-Learning

### 6.1 What to Cache Now

**Current caching strategy** (for ∂/∂x_t, ∂/∂u_t only):
```cpp
struct DynamicsStepResult {
    double J_G_xnext[36];   // ∂G/∂x_{t+1}
    double J_G_xt[36];      // ∂G/∂x_t
    double J_G_ut[...];     // ∂G/∂u_t

    // Equilibrium Jacobians at t+1
    double J_p_u0[9];       // ∂p_tip/∂u_0
    double J_p_ut[...];     // ∂p_tip/∂u_t
};
```

**Future extension for ∂/∂θ** (v1.2):
```cpp
struct DynamicsStepResult {
    // ... existing fields ...

    // NEW: Parameter Jacobians (added in v1.2)
    double J_G_theta[6*N_PARAMS];  // ∂G/∂θ (6×N_params)
    double J_p_theta[3*N_PARAMS];  // ∂p_tip/∂θ (3×N_params)

    // NEW: Metadata for θ
    int n_params;                  // Number of learnable parameters
};
```

### 6.2 API Evolution Path

**v1.1 (this design)**: Only ∂/∂x_t, ∂/∂u_t
```cpp
int dynamics_backward(
    const DynamicsStepResult& fwd,
    const double grad_x_next[6],
    double grad_x_t[6],
    double grad_u_t[3*NUM_ACT_SET],
    ...
);
```

**v1.2 (future θ-learning)**: Add ∂/∂θ
```cpp
int dynamics_backward_with_params(
    const DynamicsStepResult& fwd,
    const double grad_x_next[6],
    double grad_x_t[6],
    double grad_u_t[3*NUM_ACT_SET],
    double grad_theta[N_PARAMS],    // NEW: parameter gradients
    ...
);
```

**Backward compatibility**: Keep `dynamics_backward` unchanged, add new function for θ-learning.

### 6.3 Parameter Vector θ (Future)

Likely learnable parameters for v1.2:
```
θ ∈ R^N_params:
  - EI[i]: Bending stiffness per segment (N_seg values)
  - GJ[i]: Torsional stiffness per segment (N_seg values)
  - m[i]: Mass per segment (N_seg values)
  - d[i]: Damping coefficients (N_seg values)
  - L[i]: Segment lengths (N_seg values)
  - coil_area[i]: Coil turn area (N_coil values)

Total: ~5*N_seg + N_coil parameters
For typical catheter: ~30-50 parameters
```

**Caching strategy**: During forward pass at t+1, also compute:
- ∂G/∂θ: Requires differentiating residual w.r.t. M, D, K matrices
- ∂equilibrium/∂θ: Requires extending equilibrium_backward to include θ

**Implementation note**: Finite differences for ∂G/∂θ may be acceptable initially (parameters change slowly during meta-learning).

---

## 7. Summary Table

| Phase | Checkpoint | Test Type | Acceptance Criterion |
|-------|-----------|-----------|---------------------|
| CP2.1 | Core dynamics | C++ smoke | Compiles, runs without crash |
| CP2.2 | FD validation | C++ + FD | ||J_FD - J_analytical||_F / ||J_FD||_F < 1e-4 at 3 points |
| CP2.3 | Python bindings | Binding test | Python results match C++ to 1e-15 |
| CP2.4 | PyTorch gradcheck | Double backward | torch.autograd.gradcheck passes (tol=1e-4) |
| CP2.5 | Short rollout | 20-step rollout | No divergence, backprop succeeds |
| CP2.6 | CI integration | CTest + GHA | All dynamics tests in CI, hard gate on PR |

---

## 8. Risk Mitigation

### Risk 1: Jacobian accuracy for stiff dynamics
**Mitigation**:
- Start with small dt (0.001-0.01s) in CP2.2
- Use Backward Euler (A-stable)
- Check condition number of A matrix
- Fail-fast if rank-deficient

### Risk 2: Equilibrium solve in inner loop (performance)
**Mitigation**:
- Cache equilibrium results if u_t doesn't change much
- Use warm-start from previous timestep
- Profile and optimize hotspots after CP2.5

### Risk 3: State dimension too small (not Markovian)
**Mitigation**:
- Validate in CP2.5 that dynamics are reproducible
- If needed, expand state to include more internal variables
- Test with different dt to ensure convergence

### Risk 4: Backward pass numerical instability
**Mitigation**:
- Use FullPivLU with rank checking (already in equilibrium_backward)
- Check relative residual < 1e-10
- Add diagonal regularization if needed (A → A + εI)

---

## 9. Files to Create

### C++ Implementation
- `src/CRM_DiffDynamics.hpp` - API declarations
- `src/CRM_DiffDynamics.cpp` - Implementation

### C++ Tests
- `test_cp21_dynamics_smoke.cpp` - Smoke test
- `test_cp22_dynamics_fd.cpp` - Finite difference validation
- `test_cp25_dynamics_rollout.cpp` - Short rollout test

### Python Tests
- `python/test_cp23_dynamics_binding.py` - Binding test
- `python/test_cp24_gradcheck.py` - PyTorch gradcheck
- `python/test_cp25_rollout_smoke.py` - Python rollout

### Documentation
- `docs/DYNAMICS_API.md` - API documentation
- `docs/CP2_CHECKPOINTS.md` - Checkpoint plan details

---

## 10. Open Design Questions (Resolve in CP2.1)

1. **Inertia matrix M**: How to compute from catheter geometry?
   - Option A: Analytical from segment masses
   - Option B: Estimate from experiments
   - **Decision needed before CP2.1**

2. **Damping matrix D**: Structural damping model?
   - Option A: Rayleigh damping (D = αM + βK)
   - Option B: Constant diagonal
   - **Decision needed before CP2.1**

3. **Force model f(u_0, u_t)**: How to couple equilibrium to dynamics?
   - Option A: f = -K_base * (u_0 - u_star) + tau_magnetic(u_t)
   - Option B: Run mini-equilibrium solve to get force residual
   - **Decision needed before CP2.1**

4. **Time step dt**: Adaptive or fixed?
   - Option A: Fixed dt (simpler for MPC)
   - Option B: Adaptive (better accuracy)
   - **Recommend fixed dt for v1.1**

---

## Decision Log

### 2025-12-31: Section 10 Resolved

**Decision 1: Inertia Matrix M**
- **Choice**: Diagonal inertia based on actuator mass
- **Rationale**: From `CatheterParameterSet_1_dyn.txt`, `ActMass = 8.2859e-06` kg is the only mass parameter available. For v1.1 (no θ-learning), use simplified diagonal inertia:
  ```
  M = diag(m_eff, m_eff, m_eff)  where m_eff = ActMass / L_seg
  ```
  where L_seg is the characteristic length scale (~20mm from segment lengths).
- **Units**: kg/mm (mass per unit curvature)
- **Implementation**: Compute from CathParams.ActMass[0]
- **Future**: In v1.2 θ-learning, M can become learnable

**Decision 2: Damping Matrix D**
- **Choice**: Diagonal damping (critically damped approximation)
- **Rationale**: From `CRMDYN.hpp` line 94, existing dynamics code uses `damping[NUM_ACT_SET][6]`. For v1.1, use simplified model:
  ```
  D = diag(d, d, d)  where d = 2 * sqrt(m_eff * k_eff)
  ```
  where k_eff is estimated from K_tip diagonal elements.
- **Units**: (kg·mm)/s (damping per unit curvature rate)
- **Implementation**: Compute as 2*sqrt(M[0,0] * K_tip[0,0]) for critical damping
- **Alternative**: If K_tip varies significantly, use constant d = 0.01 (empirical tuning parameter)

**Decision 3: Force Model f(u_0, u_t)**
- **Choice**: Force from equilibrium constraint violation
- **Rationale**: The equilibrium solver `equilibrium_forward(u_0, u_t, Li)` solves for deltau0 such that the catheter is in equilibrium. When u_0 ≠ deltau0 (from dynamics), there is a restoring force. From `equilibrium_backward`, the key matrix is `K_tip * J_u_u0`. The force is:
  ```
  f(u_0, u_t) = -K_tip * (u_0 - u_eq)
  ```
  where u_eq = deltau0 from `equilibrium_forward(u_t, Li)` with u_0=0 baseline.
- **Practical implementation**:
  1. Call `equilibrium_forward(u_0, u_t, Li)` → get deltau0, K_tip, Jacobians
  2. Compute residual: `r = u_0 - deltau0` (curvature mismatch)
  3. Force: `f = -K_tip * r`
- **Jacobians**:
  - `∂f/∂u_0 = -K_tip` (direct from stiffness)
  - `∂f/∂u_t` comes from `∂deltau0/∂u_t`, which requires implicit differentiation through equilibrium_backward

**Decision 4: Time Step dt**
- **Choice**: Fixed dt
- **Rationale**: MPC/iLQR requires fixed-interval discretization. Start with dt = 0.01s (10ms).
- **Implementation**: Pass dt as parameter to `dynamics_forward`
- **Validation**: Check in CP2.2 FD tests that results are dt-invariant (within tolerance)

**Key Simplification for v1.1**:
Instead of the full force model, use a **linearized quasi-static approximation**:
```
f(u_0, u_t) ≈ -K_tip * u_0 + f_actuator(u_t)
```
where:
- `-K_tip * u_0` is the elastic restoring force (stiffness times curvature)
- `f_actuator(u_t)` is the magnetic actuation force, already captured in equilibrium Jacobians J_p_zc, J_u_zc

This avoids nested equilibrium solves in the dynamics residual while maintaining differentiability.

**Revised Residual** (Backward Euler with quasi-static force):
```
G(x_{t+1}, x_t, u_t) = [u_{t+1} - u_t - dt * v_{t+1}]
                        [M v_{t+1} - M v_t + dt * (D v_{t+1} + K_tip * u_{t+1})]
                     = 0
```
where the second block is the momentum equation with implicit damping and stiffness.

**Jacobians**:
```
A = ∂G/∂x_{t+1} = [I,         -dt*I              ]
                   [dt*K_tip,  M + dt*D           ]  (6×6)

C = ∂G/∂x_t = [-I,  0  ]
               [0,  -M  ]  (6×6)

B = ∂G/∂u_t = [0]
               [0]  (6×3) for now (no direct control coupling)
```

**Note**: This simplified model decouples equilibrium from dynamics for v1.1. Magnetic actuation enters indirectly through initial conditions. For v1.2+, add explicit control coupling via equilibrium Jacobians.

---

**Document Version**: 1.1
**Created**: 2025-12-31
**Updated**: 2025-12-31
**Status**: Section 10 Resolved - Ready for CP2.1
