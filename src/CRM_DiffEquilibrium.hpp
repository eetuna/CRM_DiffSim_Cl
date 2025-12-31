#ifndef CRM_DIFF_EQUILIBRIUM_HPP
#define CRM_DIFF_EQUILIBRIUM_HPP

#include "CRM.hpp"

namespace CRMCatheterModel {

// Result structure for differentiable equilibrium primitive
struct EquilibriumResult {
    // Outputs
    double p_tip[3];                      // Tip position (mm)
    double deltau0[3];                    // Solved base curvature
    int converged;                        // 0=success, >0=local min

    // Cached Jacobians for backward (row-major storage)
    double J_p_u0[9];                     // ∂p_tip/∂Δu₀ (3×3)
    double J_u_u0[9];                     // ∂u_tip/∂Δu₀ (3×3)
    double J_p_zc[3*NUM_ACT_SET*3];      // ∂p_tip/∂zc (3×3N)
    double J_u_zc[3*NUM_ACT_SET*3];      // ∂u_tip/∂zc (3×3N)
    double K_tip[9];                      // K matrix at tip segment (3×3 diagonal)

    // Logging
    int nl_iterations;                    // Trust-region iterations
    double final_residual;                // ||F(x*)||
    int lu_rank;                          // rank(A^T) from FullPivLU
    double rel_solve_residual;            // ||A^T λ - rhs|| / max(||rhs||, 1)
    int exit_code;                        // 0=OK, 1=rank-deficient, 2=residual too large
};

// Forward pass: compute tip position and cache Jacobians
int equilibrium_forward(
    const double u[NUM_ACT_SET*3],       // Actuation currents (Amperes)
    double L_inserted,                    // Insertion length (mm)
    const CRMForwardKinematicsData& params,
    EquilibriumResult& out
);

// Backward pass: compute gradient w.r.t. actuation currents using implicit differentiation
// Returns: 0=success, 1=rank-deficient, 2=residual too large
int equilibrium_backward(
    const EquilibriumResult& fwd_result, // Cached forward result
    const double grad_p_tip[3],          // Upstream gradient ∂L/∂p_tip
    double grad_u[NUM_ACT_SET*3],        // Output: ∂L/∂u
    int* lu_rank = nullptr,              // Optional: rank from FullPivLU
    double* rel_residual = nullptr       // Optional: ||A^T λ - rhs|| / max(||rhs||, 1)
);

} // namespace CRMCatheterModel

#endif // CRM_DIFF_EQUILIBRIUM_HPP
