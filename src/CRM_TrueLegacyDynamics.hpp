#ifndef CRM_TRUE_LEGACY_DYNAMICS_HPP
#define CRM_TRUE_LEGACY_DYNAMICS_HPP

#include "CRM.hpp"
#include "CRMDYN.hpp"

namespace CRMCatheterModel {

// Result structure for TRUE legacy one-step forward pass (caches data for backward)
struct TrueLegacyStepResult {
    // Next state outputs
    double xf_next[NUM_STATES];              // Tip state: [p[3], R[9], u[3]]
    double x_coil_next[NUM_ACT_SET][18];     // Coil states: [v[3], w[3], p[3], R[9]] per coil

    // Observables
    double tip_p[3];                         // Tip position
    double tip_R[9];                         // Tip orientation
    double tip_u[3];                         // Tip curvature

    // BVP solution (cached for backward)
    double u0[3];                            // Base curvature
    double mL[NUM_ACT_SET][3];               // Interface moments
    double nL[NUM_ACT_SET][3];               // Interface forces
    double tau[NUM_ACT_SET][3];              // Magnetic torques
    double ftip[3];                          // Tip force

    // Input state (cached for backward)
    double xf[NUM_STATES];                   // Input tip state
    double x_coil[NUM_ACT_SET][18];          // Input coil states
    double u[NUM_ACT_SET][3];                // Input currents
    double dt;                               // Time step
    double L_inserted;                       // Insertion length (cached for backward)

    // Convergence status
    int converged;                           // 0=success, >0=failed
    int localmin;                            // BVP solver exit code
};

// Forward pass: compute next state using DynamicsBVP → DYNSolverIVP
// Returns: 0=success, non-zero=failure
int true_legacy_step_forward(
    const double x_coil[NUM_ACT_SET][18],    // Coil states
    const double xf[NUM_STATES],             // Tip state
    const double u[NUM_ACT_SET][3],          // Actuation currents
    double dt,                               // Time step
    double L_inserted,                       // Insertion length (mm)
    const CRMForwardKinematicsData& params,  // Catheter parameters
    const double* mL_guess,                  // Optional warm-start for mL (can be nullptr)
    const double* nL_guess,                  // Optional warm-start for nL (can be nullptr)
    TrueLegacyStepResult& out                // Output + cached data
);

// Backward pass: compute VJP using implicit differentiation
// Given upstream gradient dL/dtip_p, computes dL/dx and dL/du
// Uses implicit function theorem on BVP solution (no backprop through iterations)
// Returns: 0=success, non-zero=failure
int true_legacy_step_backward(
    const TrueLegacyStepResult& fwd_result,  // Cached forward result
    const double grad_tip_p[3],              // Upstream gradient ∂L/∂tip_p
    const CRMForwardKinematicsData& params,  // Catheter parameters
    double grad_x_coil[NUM_ACT_SET][18],     // Output: ∂L/∂x_coil
    double grad_xf[NUM_STATES],              // Output: ∂L/∂xf
    double grad_u[NUM_ACT_SET][3],           // Output: ∂L/∂u
    int* lu_rank = nullptr,                  // Optional: rank from linear solve
    double* rel_residual = nullptr           // Optional: solve residual
);

// A3.5: Batched backward pass with multi-RHS solve
// Computes VJPs for multiple upstream cotangents using one factorization per sample
// This is more efficient than looping when num_rhs > 1
// Returns: 0=success, non-zero=failure
int true_legacy_step_backward_batched(
    const TrueLegacyStepResult& fwd_result,  // Cached forward result
    int num_rhs,                             // Number of RHS vectors (cotangents)
    const double* grad_tip_p_batch,          // Upstream gradients [num_rhs, 3] row-major
    const CRMForwardKinematicsData& params,  // Catheter parameters
    double* grad_x_coil_batch,               // Output: [num_rhs, NUM_ACT_SET, 18] row-major
    double* grad_xf_batch,                   // Output: [num_rhs, NUM_STATES] row-major
    double* grad_u_batch,                    // Output: [num_rhs, NUM_ACT_SET, 3] row-major
    int* lu_rank = nullptr,                  // Optional: rank from linear solve
    double* rel_residual = nullptr           // Optional: solve residual
);

} // namespace CRMCatheterModel

#endif // CRM_TRUE_LEGACY_DYNAMICS_HPP
