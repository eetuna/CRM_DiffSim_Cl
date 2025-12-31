# CP2.1 Implementation Plan: Core Dynamics Primitive

**Status**: Ready for implementation (PATCHED)
**Date**: 2025-12-31 (Updated with CP2.0 patch)
**Dependencies**: CP1.1–CP1.8 complete, Section 10 resolved, **CP2.0 patch applied**

**CRITICAL**: This plan incorporates **CP2.0_DYNAMICS_PATCH.md** which fixes:
- BLOCKER A: Added actuation coupling (B ≠ 0)
- BLOCKER B: Removed Newton solver (direct linear solve)

---

## Overview

Implement `dynamics_forward` and `dynamics_backward` in C++ following the v1.1 design with **CP2.0 patch applied**.

**Key Features**:
1. **Linear residual**: G = A*x_{t+1} + C*x_t + B*u_t (affine in all arguments)
2. **Direct solve**: One FullPivLU solve, no Newton iteration
3. **Actuation coupling**: B ≠ 0 via K*J_u_zc from equilibrium Jacobians
4. **Controllable dynamics**: ∂x_{t+1}/∂u_t ≠ 0 (enables MPC/iLQR/RL)

---

## 1. New Files to Create

### 1.1 Header: `src/CRM_DiffDynamics.hpp`

```cpp
#ifndef CRM_DIFF_DYNAMICS_HPP
#define CRM_DIFF_DYNAMICS_HPP

#include "CRM.hpp"

namespace CRMCatheterModel {

// Result structure for one-step dynamics forward pass
struct DynamicsStepResult {
    // State outputs
    double x_next[6];                // Next state: [u_0_{t+1}, v_0_{t+1}]
    int converged;                   // 0=success, >0=failed

    // Observables (via equilibrium at t+1)
    double p_tip[3];                 // Tip position at t+1 (mm)
    double u_tip[3];                 // Tip curvature at t+1 (1/mm)

    // Cached Jacobians for backward pass (row-major storage)
    double J_G_xnext[36];            // ∂G/∂x_{t+1} (6×6)
    double J_G_xt[36];               // ∂G/∂x_t (6×6)
    double J_G_ut[6*NUM_ACT_SET*3];  // ∂G/∂u_t (6×3N) - currently zeros

    // Equilibrium Jacobians at t+1
    double J_p_u0[9];                // ∂p_tip/∂u_0 at t+1 (3×3)
    double J_p_ut[3*NUM_ACT_SET*3];  // ∂p_tip/∂u_t (3×3N)

    // Physics matrices (computed from catheter params)
    double M[9];                     // Inertia matrix (3×3, diagonal)
    double D[9];                     // Damping matrix (3×3, diagonal)
    double K[9];                     // Stiffness matrix (3×3, from K_tip)

    // Diagnostics
    double solve_residual;           // ||A*x_{t+1} - rhs|| / ||rhs|| (solve accuracy)
    int lu_rank;                     // rank(A) from FullPivLU in forward
    double rel_solve_residual;       // Relative residual from forward solve
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

// Backward pass: compute VJP (∂L/∂x_t, ∂L/∂u_t) from upstream gradient
// Returns: 0=success, 1=rank-deficient, 2=residual too large
int dynamics_backward(
    const DynamicsStepResult& fwd_result,  // Cached forward result
    const double grad_x_next[6],           // Upstream gradient ∂L/∂x_{t+1}
    double grad_x_t[6],                    // Output: ∂L/∂x_t
    double grad_u_t[NUM_ACT_SET*3],        // Output: ∂L/∂u_t
    int* lu_rank = nullptr,                // Optional: rank from FullPivLU
    double* rel_residual = nullptr         // Optional: solve residual
);

} // namespace CRMCatheterModel

#endif // CRM_DIFF_DYNAMICS_HPP
```

### 1.2 Implementation: `src/CRM_DiffDynamics.cpp`

**Outline** (full code not shown, but structure):

```cpp
#include "CRM_DiffDynamics.hpp"
#include "CRM_DiffEquilibrium.hpp"
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <cstring>

using namespace Eigen;

namespace CRMCatheterModel {

// Helper: Compute M, D, K from catheter parameters
static void compute_physics_matrices(
    const CRMCatheterModelParams& CathParams,
    double K_tip[9],  // from equilibrium
    double M_out[9],  // output: inertia
    double D_out[9]   // output: damping
);

// Helper: Residual G(x_{t+1}, x_t, u_t) for given x_{t+1}
static void compute_residual(
    const double x_next[6],
    const double x_t[6],
    const double M[9],
    const double D[9],
    const double K[9],
    double dt,
    double G_out[6]
);

// Helper: Jacobians A = ∂G/∂x_{t+1}, C = ∂G/∂x_t
static void compute_jacobians(
    const double M[9],
    const double D[9],
    const double K[9],
    double dt,
    double A_out[36],  // 6×6
    double C_out[36]   // 6×6
);

int dynamics_forward(
    const double x_t[6],
    const double u_t[NUM_ACT_SET*3],
    double dt,
    double L_inserted,
    const CRMForwardKinematicsData& params,
    DynamicsStepResult& out
) {
    // 1. Get K_tip from equilibrium (call with u_t to get current stiffness)
    //    Use x_t[0:2] as initial guess for deltau0
    // 2. Compute M, D from catheter parameters
    // 3. Solve implicit residual G(x_{t+1}, x_t, u_t) = 0 using Newton
    // 4. Call equilibrium_forward(x_{t+1}[0:2], u_t, Li) to get observables
    // 5. Compute and cache Jacobians A, C, B
    // 6. Return status
}

int dynamics_backward(
    const DynamicsStepResult& fwd_result,
    const double grad_x_next[6],
    double grad_x_t[6],
    double grad_u_t[NUM_ACT_SET*3],
    int* lu_rank,
    double* rel_residual
) {
    // 1. Extract cached A, C, B
    // 2. Solve A^T λ = grad_x_next using FullPivLU
    // 3. Check rank and residual
    // 4. Compute grad_x_t = -C^T λ
    // 5. Compute grad_u_t = -B^T λ (currently zeros)
    // 6. Return status
}

} // namespace CRMCatheterModel
```

---

## 2. Exact Residual Formula

Based on Section 10 decisions, the residual is:

```
State: x = [u_0, v_0] ∈ R^6
  u_0 ∈ R^3: base curvature
  v_0 ∈ R^3: base curvature rate

Dynamics (Backward Euler with actuation coupling):
  u_{t+1} = u_t + dt * v_{t+1}
  M v_{t+1} = M v_t - dt * (D v_{t+1} + K u_{t+1}) + dt * K * J_u_zc * u_t

Residual G(x_{t+1}, x_t, u_t) ∈ R^6:
  G_u = u_{t+1} - u_t - dt * v_{t+1}                                         (3×1)
  G_v = M v_{t+1} - M v_t + dt*D v_{t+1} + dt*K u_{t+1} - dt*K*J_u_zc*u_t   (3×1)

Combined:
  G = [G_u] = [u_{t+1} - u_t - dt * v_{t+1}                             ]
      [G_v]   [M v_{t+1} - M v_t + dt*D v_{t+1} + dt*K u_{t+1} - dt*K*J_u_zc*u_t]
```

**Affine form** (linear residual):
```
G = A * x_{t+1} + C * x_t + B * u_t = 0
```

where:
```
A = [I,       -dt*I    ]  (6×6)
    [dt*K,    M+dt*D   ]

C = [-I,  0 ]  (6×6)
    [0,  -M ]

B = [0           ]  (6×3N, N=NUM_ACT_SET)
    [-dt*K*J_u_zc]
```

---

## 3. Exact Jacobians

### 3.1 Jacobian A = ∂G/∂x_{t+1}

```
A = [∂G_u/∂u_{t+1},  ∂G_u/∂v_{t+1}]
    [∂G_v/∂u_{t+1},  ∂G_v/∂v_{t+1}]

  = [I,       -dt*I      ]  (6×6)
    [dt*K,    M + dt*D   ]
```

**Block structure**:
- Top-left (3×3): I
- Top-right (3×3): -dt*I
- Bottom-left (3×3): dt*K (where K = K_tip from equilibrium)
- Bottom-right (3×3): M + dt*D

**Implementation**:
```cpp
void compute_jacobians(...) {
    // A is 6×6 row-major
    // Top-left: I
    A[0] = 1; A[1] = 0; A[2] = 0;
    A[6] = 0; A[7] = 1; A[8] = 0;
    A[12] = 0; A[13] = 0; A[14] = 1;

    // Top-right: -dt*I
    A[3] = -dt; A[4] = 0; A[5] = 0;
    A[9] = 0; A[10] = -dt; A[11] = 0;
    A[15] = 0; A[16] = 0; A[17] = -dt;

    // Bottom-left: dt*K (K is 3×3 row-major)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A[(3+i)*6 + j] = dt * K[i*3 + j];
        }
    }

    // Bottom-right: M + dt*D (both 3×3 diagonal)
    for (int i = 0; i < 3; i++) {
        A[(3+i)*6 + (3+i)] = M[i*3 + i] + dt * D[i*3 + i];
        // off-diagonals are zero for diagonal M, D
    }
}
```

### 3.2 Jacobian C = ∂G/∂x_t

```
C = [∂G_u/∂u_t,  ∂G_u/∂v_t]
    [∂G_v/∂u_t,  ∂G_v/∂v_t]

  = [-I,  0  ]  (6×6)
    [0,  -M  ]
```

**Implementation**:
```cpp
// C is 6×6 row-major
// Top-left: -I
C[0] = -1; C[1] = 0; C[2] = 0;
C[6] = 0; C[7] = -1; C[8] = 0;
C[12] = 0; C[13] = 0; C[14] = -1;

// Top-right: 0 (already initialized to zero)

// Bottom-left: 0 (already initialized to zero)

// Bottom-right: -M (diagonal)
for (int i = 0; i < 3; i++) {
    C[(3+i)*6 + (3+i)] = -M[i*3 + i];
}
```

### 3.3 Jacobian B = ∂G/∂u_t (PATCHED - non-zero!)

**CP2.0 Patch**: Add actuation coupling via equilibrium Jacobians:

```
B = ∂G/∂u_t = [0           ]  (6×3N, N=NUM_ACT_SET)
               [-dt*K*J_u_zc]

where:
  K = K_tip from equilibrium_forward (3×3)
  J_u_zc from equilibrium_forward (3×3N)
  K * J_u_zc: matrix product (3×3N)
```

**Dimensions** (for NUM_ACT_SET=1):
- B is 6×3
- Top block (3×3): zeros
- Bottom block (3×3): -dt*K*J_u_zc

**Physical interpretation**:
- J_u_zc: how actuation affects tip curvature (∂u_tip/∂u_t)
- K: stiffness converts curvature to moment
- K*J_u_zc: how actuation creates elastic moment at base
- -dt*K*J_u_zc: actuation force term in velocity equation

**Implementation**:
```cpp
// Extract K and J_u_zc from equilibrium result
Map<const Matrix3d> K_map(eq_result.K_tip);
Map<const Matrix<double, 3, Dynamic, RowMajor>> J_u_zc_map(
    eq_result.J_u_zc, 3, 3*NUM_ACT_SET);

// Compute K * J_u_zc
MatrixXd K_J_u_zc = K_map * J_u_zc_map;  // 3×3N

// Assemble B (6×3N row-major)
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3*NUM_ACT_SET; j++) {
        B[i * (3*NUM_ACT_SET) + j] = 0.0;                    // top block
        B[(3+i) * (3*NUM_ACT_SET) + j] = -dt * K_J_u_zc(i, j); // bottom block
    }
}
```

**Why this enables MPC/iLQR/RL**:
```
∂x_{t+1}/∂u_t = -A^{-1} B ≠ 0
```
Control now affects the next state → system is controllable.

---

## 4. Computing M, D, K Matrices

### 4.1 Inertia Matrix M

From Decision 1:
```cpp
void compute_physics_matrices(...) {
    // M = diag(m_eff, m_eff, m_eff)
    // where m_eff = ActMass / L_seg

    double ActMass = CathParams.ActMass[0];  // kg
    double L_seg = CathParams.SegLengths[0]; // mm (first actuator segment)
    double m_eff = ActMass / L_seg;          // kg/mm

    // M is 3×3 diagonal, stored row-major
    std::memset(M_out, 0, 9 * sizeof(double));
    M_out[0] = M_out[4] = M_out[8] = m_eff;
}
```

### 4.2 Damping Matrix D

From Decision 2:
```cpp
void compute_physics_matrices(...) {
    // D = diag(d, d, d)
    // where d = 2 * sqrt(m_eff * k_eff) for critical damping

    // k_eff from K_tip diagonal average
    double k_eff = (K_tip[0] + K_tip[4] + K_tip[8]) / 3.0;
    double d = 2.0 * std::sqrt(m_eff * k_eff);

    // D is 3×3 diagonal
    std::memset(D_out, 0, 9 * sizeof(double));
    D_out[0] = D_out[4] = D_out[8] = d;
}
```

### 4.3 Stiffness Matrix K

From equilibrium:
```cpp
// In dynamics_forward:
EquilibriumResult eq_result;
equilibrium_forward(u_t, L_inserted, params, eq_result);

// K = K_tip from equilibrium (already 3×3 row-major)
std::memcpy(K, eq_result.K_tip, 9 * sizeof(double));
```

---

## 5. Direct Linear Solve for x_{t+1} (PATCHED - no Newton!)

**CP2.0 Patch**: The residual is **linear** in x_{t+1}, so we solve directly without iteration:

```
G = A * x_{t+1} + C * x_t + B * u_t = 0

Rearrange:
A * x_{t+1} = -C * x_t - B * u_t
x_{t+1} = -A^{-1} * (C * x_t + B * u_t)
```

**One FullPivLU solve, no Newton iteration needed.**

```cpp
// 1. Compute RHS = -C*x_t - B*u_t
double rhs[6];
Map<const Matrix<double, 6, 6, RowMajor>> C_map(C);
Map<const Matrix<double, 6, Dynamic, RowMajor>> B_map(B, 6, 3*NUM_ACT_SET);
Map<const VectorXd> x_t_map(x_t, 6);
Map<const VectorXd> u_t_map(u_t, 3*NUM_ACT_SET);
Map<VectorXd> rhs_map(rhs, 6);

rhs_map = -C_map * x_t_map - B_map * u_t_map;

// 2. Solve A * x_{t+1} = rhs using FullPivLU
Map<Matrix<double, 6, 6, RowMajor>> A_map(A);

FullPivLU<MatrixXd> lu(A_map);

// Check rank
int rank = lu.rank();
out.lu_rank = rank;
if (rank < 6) {
    return 1;  // Rank-deficient (singular A matrix)
}

VectorXd x_next_vec = lu.solve(rhs_map);

// 3. Extract solution
for (int i = 0; i < 6; i++) {
    out.x_next[i] = x_next_vec[i];
}

// 4. Check solve accuracy (not convergence!)
VectorXd residual_check = A_map * x_next_vec - rhs_map;
double residual_norm = residual_check.norm();
double rhs_norm = rhs_map.norm();
double rel_res = residual_norm / std::max(rhs_norm, 1.0);

out.solve_residual = residual_norm;
out.rel_solve_residual = rel_res;

if (rel_res > 1e-10) {
    return 2;  // Solve inaccurate
}

out.converged = 0;
out.exit_code = 0;
return 0;  // Success
```

**Key differences from Newton**:
- **No iteration**: One solve, always
- **No convergence check**: Only solve accuracy
- **Simpler diagnostics**: solve_residual instead of final_residual
- **Removed**: nl_iterations (always 1 for linear solve)
- **Faster**: O(n³) one time vs O(k·n³) for k Newton iterations

---

## 6. Smoke Test Specification

### 6.1 Test File: `test_cp21_dynamics_smoke.cpp`

```cpp
#include "CRM.hpp"
#include "CRM_DiffDynamics.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>

using namespace CRMCatheterModel;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "CP2.1 Dynamics Smoke Test" << std::endl;
    std::cout << "==========================" << std::endl << std::endl;

    // Load parameters
    const char* param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt";
    const char* config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt";

    CRMCatheterModelParams CathParams = Load_CRMCatheterModelParams(param_file);
    CatheterConfiguration CathConfig = Load_CatheterConfiguration(config_file);

    // Setup FK params
    CRMForwardKinematicsData FKParams;
    FKParams.CathParams = &CathParams;
    FKParams.CathConfig = &CathConfig;
    FKParams.ContactMode = ContactModeType::FREE_TIP;
    FKParams.TipForce[0] = 0.0;
    FKParams.TipForce[1] = 0.0;
    FKParams.TipForce[2] = 0.0;
    FKParams.deltau0_initialguess[0] = 0.0;
    FKParams.deltau0_initialguess[1] = 0.0;
    FKParams.deltau0_initialguess[2] = 0.0;
    FKParams.IntegrationStepSize = 0.5;
    FKParams.FinalValueOnly = true;

    // Smoke case: zero initial state, zero control, small dt
    double x_t[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double u_t[3] = {0.0, 0.0, 0.0};
    double dt = 0.01;  // 10ms
    double L_inserted = 50.0;

    std::cout << "Test: x_t = [0,0,0,0,0,0], u_t = [0,0,0], dt = " << dt << std::endl;
    std::cout << std::endl;

    DynamicsStepResult result;
    std::memset(&result, 0, sizeof(DynamicsStepResult));

    int status = dynamics_forward(x_t, u_t, dt, L_inserted, FKParams, result);

    std::cout << "Forward pass:" << std::endl;
    std::cout << "  status = " << status << std::endl;
    std::cout << "  converged = " << result.converged << std::endl;
    std::cout << "  lu_rank = " << result.lu_rank << std::endl;
    std::cout << "  solve_residual = " << result.solve_residual << std::endl;
    std::cout << "  rel_solve_residual = " << result.rel_solve_residual << std::endl;
    std::cout << std::endl;

    std::cout << "  x_next = [";
    for (int i = 0; i < 6; i++) {
        std::cout << result.x_next[i];
        if (i < 5) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << std::endl;

    std::cout << "  p_tip = [" << result.p_tip[0] << ", "
              << result.p_tip[1] << ", "
              << result.p_tip[2] << "]" << std::endl;
    std::cout << std::endl;

    // Sanity checks
    bool pass = true;

    if (status != 0) {
        std::cerr << "FAIL: status != 0" << std::endl;
        pass = false;
    }

    if (result.lu_rank < 6) {
        std::cerr << "FAIL: rank-deficient A matrix" << std::endl;
        pass = false;
    }

    if (result.rel_solve_residual >= 1e-10) {
        std::cerr << "FAIL: solve residual too large" << std::endl;
        pass = false;
    }

    // For zero input, expect x_next ≈ x_t (no change)
    double norm_change = 0.0;
    for (int i = 0; i < 6; i++) {
        double diff = result.x_next[i] - x_t[i];
        norm_change += diff * diff;
    }
    norm_change = std::sqrt(norm_change);

    if (norm_change > 1e-6) {
        std::cerr << "FAIL: x_next changed significantly for zero input (norm = "
                  << norm_change << ")" << std::endl;
        pass = false;
    }

    // Check backward pass
    std::cout << "Backward pass test:" << std::endl;
    double grad_x_next[6] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double grad_x_t[6], grad_u_t[3];
    int lu_rank;
    double rel_residual;

    int bwd_status = dynamics_backward(result, grad_x_next, grad_x_t, grad_u_t,
                                       &lu_rank, &rel_residual);

    std::cout << "  bwd_status = " << bwd_status << std::endl;
    std::cout << "  lu_rank = " << lu_rank << std::endl;
    std::cout << "  rel_residual = " << rel_residual << std::endl;
    std::cout << "  grad_x_t = [";
    for (int i = 0; i < 6; i++) {
        std::cout << grad_x_t[i];
        if (i < 5) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << std::endl;

    if (bwd_status != 0) {
        std::cerr << "FAIL: backward pass status != 0" << std::endl;
        pass = false;
    }

    if (lu_rank < 6) {
        std::cerr << "FAIL: rank-deficient Jacobian" << std::endl;
        pass = false;
    }

    if (rel_residual >= 1e-10) {
        std::cerr << "FAIL: backward solve residual too large" << std::endl;
        pass = false;
    }

    std::cout << "==========================" << std::endl;
    if (pass) {
        std::cout << "PASS: Smoke test succeeded" << std::endl;
        return 0;
    } else {
        std::cout << "FAIL: Smoke test failed" << std::endl;
        return 1;
    }
}
```

### 6.2 Expected Behavior (PATCHED)

**Input**:
- x_t = [0, 0, 0, 0, 0, 0] (zero state)
- u_t = [0, 0, 0] (zero control)
- dt = 0.01

**Expected output**:
- status = 0 (success)
- converged = 0
- lu_rank = 6 (full rank)
- solve_residual < 1e-10 (solve accuracy, not convergence)
- rel_solve_residual < 1e-10
- x_next ≈ [0, 0, 0, 0, 0, 0] (no change for zero input, zero velocity)
- p_tip ≈ [~0, ~0, 50.0] (from equilibrium at zero curvature)

**Backward pass**:
- bwd_status = 0
- lu_rank = 6
- rel_residual < 1e-10
- grad_x_t finite and non-zero (identity propagation)
- **grad_u_t finite** (may be non-zero even for zero input due to B ≠ 0)

**Additional test** (verify control authority):
```cpp
// Test: non-zero control should change state
x_t = [0, 0, 0, 0, 0, 0];
u_t = [0.1, 0, 0];  // Non-zero actuation
dt = 0.01;

// Expect: x_next ≠ x_t (specifically, v_{t+1} ≠ 0)
// This verifies B ≠ 0 and control has effect
```

---

## 7. CMakeLists.txt Changes

Add to `CMakeLists.txt`:

```cmake
# Add CRM_DiffDynamics.cpp to CRMCPPLib sources
set(CRMSOURCES
    ${CRMSOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_DiffDynamics.cpp
)

# Rebuild library (no explicit add needed, already included in CRMCPPLib)

# Add smoke test executable
add_executable(test_cp21_dynamics_smoke
    test_cp21_dynamics_smoke.cpp
)

target_link_libraries(test_cp21_dynamics_smoke PUBLIC
    CRMCPPLib
)

target_include_directories(test_cp21_dynamics_smoke PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/numerical"
)

# Add CTest entry
add_test(NAME dynamics_smoke_cp21
         COMMAND test_cp21_dynamics_smoke
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
```

---

## 8. Build and Test Commands

### 8.1 Build

```bash
cd /workspaces/CRM_DiffSim_Cl
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4 CRMCPPLib test_cp21_dynamics_smoke
```

### 8.2 Run Test

```bash
# Direct execution
./build/test_cp21_dynamics_smoke

# Via CTest
cd build
ctest --output-on-failure -R dynamics_smoke_cp21
```

### 8.3 Expected Output (PATCHED)

```
CP2.1 Dynamics Smoke Test
==========================

Test: x_t = [0,0,0,0,0,0], u_t = [0,0,0], dt = 0.01

Forward pass:
  status = 0
  converged = 0
  lu_rank = 6
  solve_residual = 3.2e-15
  rel_solve_residual = 2.1e-16

  x_next = [0, 0, 0, 0, 0, 0]

  p_tip = [-0.101, -0.389, 49.998]

Backward pass test:
  bwd_status = 0
  lu_rank = 6
  rel_residual = 1.5e-16
  grad_x_t = [-1, 0, 0, 0, 0, 0]

==========================
PASS: Smoke test succeeded
```

---

## 9. CP2.1 Acceptance Criteria Checklist

### Build Phase
- [ ] `src/CRM_DiffDynamics.hpp` created with exact API signatures
- [ ] `src/CRM_DiffDynamics.cpp` implemented
- [ ] `test_cp21_dynamics_smoke.cpp` created
- [ ] CMakeLists.txt updated
- [ ] Code compiles without errors
- [ ] Code links against CRMCPPLib

### Functional Phase (PATCHED)
- [ ] Smoke test runs without crashes
- [ ] `dynamics_forward` returns status = 0
- [ ] Forward solve rank = 6 (full rank A matrix)
- [ ] Solve residual < 1e-10 (rel_solve_residual < 1e-10)
- [ ] For zero input, x_next ≈ x_t (within 1e-6)
- [ ] **For non-zero input (u=[0.1,0,0]), x_next ≠ x_t** (control has effect)
- [ ] `dynamics_backward` returns status = 0
- [ ] Backward pass rank = 6 (full rank)
- [ ] Backward solve residual < 1e-10
- [ ] grad_x_t is finite and non-zero
- [ ] **grad_u_t is finite** (may be non-zero even for zero input)

### Diagnostics Phase
- [ ] M, D, K matrices are positive definite (diagonal elements > 0)
- [ ] Jacobian A is non-singular (det(A) != 0)
- [ ] Cached Jacobians A, C are correct shape (6×6)
- [ ] p_tip from equilibrium is reasonable (~[0, 0, 50])

### CTest Integration
- [ ] `ctest -R dynamics_smoke_cp21` passes
- [ ] Test appears in `ctest -N` list
- [ ] Test fails correctly if code is broken (inject failure to verify)

---

## 10. Next Steps After CP2.1

Once all acceptance criteria pass:

1. **Commit CP2.1**:
   ```bash
   git add src/CRM_DiffDynamics.* test_cp21_dynamics_smoke.cpp CMakeLists.txt
   git commit -m "CP2.1: Core dynamics primitive (smoke test only)"
   ```

2. **Proceed to CP2.2**: Finite difference validation at 3 operating points

3. **Update design doc**: Mark CP2.1 as complete in Section 11 decision log

---

**END OF CP2.1 IMPLEMENTATION PLAN**
