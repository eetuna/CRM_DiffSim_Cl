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
    double J_G_ut[6*NUM_ACT_SET*3];  // ∂G/∂u_t (6×3N)

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
