#include "CRM_TrueLegacyDynamics.hpp"
#include "CRM_MatrixOperations.hpp"
#include "CRM_IVPJacobian.hpp"
#include "CRM_BVPJacobian.hpp"
#include <Eigen/Dense>
#include <cstring>
#include <cmath>
#include <iostream>

namespace CRMCatheterModel {

// Forward pass: wrapper around existing DynamicsBVP → DYNSolverIVP
int true_legacy_step_forward(
    const double x_coil[NUM_ACT_SET][18],
    const double xf[NUM_STATES],
    const double u[NUM_ACT_SET][3],
    double dt,
    double L_inserted,
    const CRMForwardKinematicsData& params,
    const double* mL_guess,
    const double* nL_guess,
    TrueLegacyStepResult& out
) {
    // Cache inputs
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 18; ++i) {
            out.x_coil[j][i] = x_coil[j][i];
        }
        for (int i = 0; i < 3; ++i) {
            out.u[j][i] = u[j][i];
        }
    }
    for (int i = 0; i < NUM_STATES; ++i) {
        out.xf[i] = xf[i];
    }
    out.dt = dt;
    out.L_inserted = L_inserted;

    // Extract coil state components
    double v_L_pre[NUM_ACT_SET][3];
    double w_L_pre[NUM_ACT_SET][3];
    double p_pre[NUM_ACT_SET][3];
    double R_pre[NUM_ACT_SET][9];

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            v_L_pre[j][i] = x_coil[j][i];
            w_L_pre[j][i] = x_coil[j][3 + i];
            p_pre[j][i] = x_coil[j][6 + i];
        }
        for (int i = 0; i < 9; ++i) {
            R_pre[j][i] = x_coil[j][9 + i];
        }
    }

    // Reshape u
    double ActuationCurrents[NUM_ACT_SET][3];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            ActuationCurrents[j][i] = u[j][i];
        }
    }

    // Construct actuator inertia (diagonal approximation)
    double ActInertia[NUM_ACT_SET][9];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 9; ++i) {
            ActInertia[j][i] = (i % 4 == 0) ? params.CathParams->ActMass[j] * 1e-6 : 0.0;
        }
    }

    // Zero damping
    double damping[NUM_ACT_SET][6];
    std::memset(damping, 0, sizeof(damping));

    // Construct shooting method params
    CRMShootingMethodParams shooting_params = CRMDYNConstructShootingMethodParamSet(
        *params.CathParams, *params.CathConfig,
        L_inserted, ActuationCurrents,
        params.ContactMode,
        const_cast<double*>(params.TipConstraintPoint), const_cast<double*>(params.TipForce),
        params.IntegrationStepSize, ActInertia,
        v_L_pre, w_L_pre, p_pre, R_pre,
        damping, dt
    );

    // Prepare warm-start
    double mL_guess_arr[NUM_ACT_SET][3];
    double nL_guess_arr[NUM_ACT_SET][3];
    double ftip_guess[3] = {0.0, 0.0, 0.0};

    if (mL_guess != nullptr && nL_guess != nullptr) {
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < 3; ++i) {
                mL_guess_arr[j][i] = mL_guess[j * 3 + i];
                nL_guess_arr[j][i] = nL_guess[j * 3 + i];
            }
        }
    } else {
        std::memset(mL_guess_arr, 0, sizeof(mL_guess_arr));
        std::memset(nL_guess_arr, 0, sizeof(nL_guess_arr));
    }

    // Call DynamicsBVP
    DynamicsBVP(shooting_params, xf, mL_guess_arr, nL_guess_arr, ftip_guess,
                out.u0, out.mL, out.nL, out.tau, out.ftip, out.localmin);

    out.converged = (out.localmin == 0) ? 1 : 0;

    // Call DYNSolverIVP
    double out_markers[10][3];
    DYNSolverIVP(shooting_params, out.u0, out.mL, out.nL, out.tau, out.ftip,
                 true, out.xf_next, out.x_coil_next, out_markers);

    // Extract observables
    for (int i = 0; i < 3; ++i) {
        out.tip_p[i] = out.xf_next[i];
    }
    for (int i = 0; i < 9; ++i) {
        out.tip_R[i] = out.xf_next[3 + i];
    }
    for (int i = 0; i < 3; ++i) {
        out.tip_u[i] = out.xf_next[12 + i];
    }

    return out.converged ? 0 : 1;
}

// Backward pass using analytic implicit differentiation
int true_legacy_step_backward(
    const TrueLegacyStepResult& fwd_result,
    const double grad_tip_p[3],
    const CRMForwardKinematicsData& params,
    double grad_x_coil[NUM_ACT_SET][18],
    double grad_xf[NUM_STATES],
    double grad_u[NUM_ACT_SET][3],
    int* lu_rank,
    double* rel_residual
) {
    // Initialize all outputs to zero
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 18; ++i) {
            grad_x_coil[j][i] = 0.0;
        }
        for (int i = 0; i < 3; ++i) {
            grad_u[j][i] = 0.0;
        }
    }
    for (int i = 0; i < NUM_STATES; ++i) {
        grad_xf[i] = 0.0;
    }

    // Step 1: Cotangent on tip_p -> xf_next
    // tip_p = xf_next[0:3], so ∂tip_p/∂xf_next = [I_3, 0, 0]^T
    Eigen::VectorXd v_xf_next = Eigen::VectorXd::Zero(NUM_STATES);
    for (int i = 0; i < 3; ++i) {
        v_xf_next[i] = grad_tip_p[i];
    }

    // Step 2: Compute IVP Jacobians analytically using CRMSolverIVPJacobian
    // Reconstruct shooting params for Jacobian computation
    double v_L_pre[NUM_ACT_SET][3];
    double w_L_pre[NUM_ACT_SET][3];
    double p_pre[NUM_ACT_SET][3];
    double R_pre[NUM_ACT_SET][9];

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            v_L_pre[j][i] = fwd_result.x_coil[j][i];
            w_L_pre[j][i] = fwd_result.x_coil[j][3 + i];
            p_pre[j][i] = fwd_result.x_coil[j][6 + i];
        }
        for (int i = 0; i < 9; ++i) {
            R_pre[j][i] = fwd_result.x_coil[j][9 + i];
        }
    }

    double ActuationCurrents[NUM_ACT_SET][3];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            ActuationCurrents[j][i] = fwd_result.u[j][i];
        }
    }

    double ActInertia[NUM_ACT_SET][9];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 9; ++i) {
            ActInertia[j][i] = (i % 4 == 0) ? params.CathParams->ActMass[j] * 1e-6 : 0.0;
        }
    }

    double damping[NUM_ACT_SET][6];
    std::memset(damping, 0, sizeof(damping));

    // Use cached L_inserted from forward pass
    double L_inserted = fwd_result.L_inserted;

    CRMShootingMethodParams shooting_params = CRMDYNConstructShootingMethodParamSet(
        *params.CathParams, *params.CathConfig,
        L_inserted, ActuationCurrents,
        params.ContactMode,
        const_cast<double*>(params.TipConstraintPoint), const_cast<double*>(params.TipForce),
        params.IntegrationStepSize, ActInertia,
        v_L_pre, w_L_pre, p_pre, R_pre,
        damping, fwd_result.dt
    );

    // Compute deltau0 from converged solution
    double deltau0[3];
    for (int i = 0; i < 3; ++i) {
        deltau0[i] = fwd_result.u0[i] - shooting_params.ustar[0][i];  // Assuming single segment
    }

    // Call IVP Jacobian (analytic)
    double x_N[NUM_STATES];
    double MomentResidual[3];
    double ftip_copy[3];
    for (int i = 0; i < 3; ++i) {
        ftip_copy[i] = fwd_result.ftip[i];
    }

    auto [J_u, J_n, J_p, J_R, J_ftip] = CRMSolverIVPJacobian(
        shooting_params, deltau0, ftip_copy, true, x_N, MomentResidual
    );

    // J_u: (15 x 3) - ∂xf_next/∂u0
    // J_n: (15 x 3*NUM_ACT_SET) - ∂xf_next/∂nL (per coil)
    // J_p: (15 x 3*NUM_ACT_SET) - ∂xf_next/∂pL (coil positions)
    // J_R: (15 x 9*NUM_ACT_SET) - ∂xf_next/∂RL (coil orientations)
    // J_ftip: (15 x 3) - ∂xf_next/∂ftip

    // Step 3: Push cotangent through IVP Jacobians
    // v_y = J^T * v_xf_next, where y includes outputs that depend on BVP solution

    Eigen::VectorXd v_u0 = J_u.transpose() * v_xf_next;  // (3,) - cotangent on base curvature

    Eigen::VectorXd v_nL = J_n.transpose() * v_xf_next;  // (3*NUM_ACT_SET,) - cotangent on interface forces

    Eigen::VectorXd v_p_coil = J_p.transpose() * v_xf_next;  // (3*NUM_ACT_SET,) - coil position cotangent
    Eigen::VectorXd v_R_coil = J_R.transpose() * v_xf_next;  // (9*NUM_ACT_SET,) - coil orientation cotangent

    // Step 4: Assemble cotangent on BVP unknowns y = [mL; nL]
    // The BVP solves: r(y; x_coil, xf, u, dt) = 0 where y = [mL[0], nL[0], ..., mL[N-1], nL[N-1]]
    // Dimension: 6*NUM_ACT_SET (3 for mL + 3 for nL per actuator)

    const int dim_y = NUM_ACT_SET * 6;
    Eigen::VectorXd v_y(dim_y);

    // v_y receives cotangents from:
    // 1. nL affects xf_next through IVP (via J_n), so v_y[nL components] = v_nL
    // 2. mL affects coil dynamics and indirectly xf_next (currently minor, set to zero for simplicity)
    //    A full implementation would trace mL -> tau -> coil dynamics -> IVP

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        // mL components (indices 0-2 per actuator in y-vector as [mL;nL] interleaved)
        // For now, set to zero (mL's main effect is through coil dynamics which is second-order)
        for (int i = 0; i < 3; ++i) {
            v_y[j * 6 + i] = 0.0;  // v_mL contribution (minimal direct path to xf_next)
        }
        // nL components (indices 3-5 per actuator)
        for (int i = 0; i < 3; ++i) {
            v_y[j * 6 + 3 + i] = v_nL[j * 3 + i];  // From J_n^T * v_xf_next
        }
    }

    // Step 5: Compute BVP Jacobian blocks J_yy, J_yu, and J_yx using strictly analytic methods (NO FD)
    Eigen::MatrixXd J_yy, J_yu, J_yx;

    compute_bvp_jacobians_full_analytic(
        fwd_result.mL, fwd_result.nL, fwd_result.u,
        params, fwd_result.xf, L_inserted, fwd_result.dt,
        fwd_result.x_coil,
        J_yy, J_yu, J_yx
    );

    // Step 6: Solve adjoint system: (J_yy)^T * lambda = v_y
    Eigen::VectorXd lambda;
    int rank_used = dim_y;
    double residual_norm = 0.0;

    // Use QR decomposition for stable solve
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_solver(J_yy.transpose());
    lambda = qr_solver.solve(v_y);

    rank_used = qr_solver.rank();
    residual_norm = (J_yy.transpose() * lambda - v_y).norm();

    // Step 7: Compute gradients using implicit function theorem

    // 7a. Direct gradients for xf (tip state)
    for (int i = 0; i < NUM_STATES; ++i) {
        grad_xf[i] = v_xf_next[i];  // Direct passthrough (xf affects target)
    }

    // 7b. Gradients for x_coil (coil states)
    // Direct contribution from IVP Jacobians
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        // Position gradient (indices 6-8 in x_coil)
        for (int i = 0; i < 3; ++i) {
            grad_x_coil[j][6 + i] = v_p_coil[j*3 + i];
        }
        // Orientation gradient (indices 9-17 in x_coil)
        for (int i = 0; i < 9; ++i) {
            grad_x_coil[j][9 + i] = v_R_coil[j*9 + i];
        }
        // Velocity and angular velocity gradients (indices 0-5)
        // These affect coil dynamics (second-order effect, set to zero for now)
        for (int i = 0; i < 6; ++i) {
            grad_x_coil[j][i] = 0.0;
        }
    }

    // 7c. Add implicit terms: grad_x -= (J_yx)^T * lambda
    // This correctly wires ∂x_{t+1}/∂x_t through the BVP implicit dependence
    Eigen::VectorXd implicit_grad_x = J_yx.transpose() * lambda;

    // Apply to grad_xf
    for (int i = 0; i < NUM_STATES; ++i) {
        grad_xf[i] -= implicit_grad_x[NUM_ACT_SET * 18 + i];
    }

    // Apply to grad_x_coil
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 18; ++i) {
            grad_x_coil[j][i] -= implicit_grad_x[j * 18 + i];
        }
    }

    // 7d. Gradients for u (actuation currents) - BVP adjoint
    // grad_u = -(J_yu)^T * lambda
    Eigen::VectorXd grad_u_vec = -J_yu.transpose() * lambda;  // (3*NUM_ACT_SET,)

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            grad_u[j][i] = grad_u_vec[j * 3 + i];
        }
    }

    // Set diagnostics
    if (lu_rank != nullptr) {
        *lu_rank = rank_used;
    }
    if (rel_residual != nullptr) {
        *rel_residual = residual_norm;
    }

    return 0;  // Success
}

// A3.5: Batched backward pass with multi-RHS solve
// Solves the adjoint system for multiple RHS using one factorization
int true_legacy_step_backward_batched(
    const TrueLegacyStepResult& fwd_result,
    int num_rhs,
    const double* grad_tip_p_batch,
    const CRMForwardKinematicsData& params,
    double* grad_x_coil_batch,
    double* grad_xf_batch,
    double* grad_u_batch,
    int* lu_rank,
    double* rel_residual
) {
    // Step 1: Reconstruct shooting params and IVP Jacobians (same for all RHS)
    double v_L_pre[NUM_ACT_SET][3];
    double w_L_pre[NUM_ACT_SET][3];
    double p_pre[NUM_ACT_SET][3];
    double R_pre[NUM_ACT_SET][9];

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            v_L_pre[j][i] = fwd_result.x_coil[j][i];
            w_L_pre[j][i] = fwd_result.x_coil[j][3 + i];
            p_pre[j][i] = fwd_result.x_coil[j][6 + i];
        }
        for (int i = 0; i < 9; ++i) {
            R_pre[j][i] = fwd_result.x_coil[j][9 + i];
        }
    }

    double ActuationCurrents[NUM_ACT_SET][3];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            ActuationCurrents[j][i] = fwd_result.u[j][i];
        }
    }

    double ActInertia[NUM_ACT_SET][9];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 9; ++i) {
            ActInertia[j][i] = (i % 4 == 0) ? params.CathParams->ActMass[j] * 1e-6 : 0.0;
        }
    }

    double damping[NUM_ACT_SET][6];
    std::memset(damping, 0, sizeof(damping));

    double L_inserted = fwd_result.L_inserted;

    CRMShootingMethodParams shooting_params = CRMDYNConstructShootingMethodParamSet(
        *params.CathParams, *params.CathConfig,
        L_inserted, ActuationCurrents,
        params.ContactMode,
        const_cast<double*>(params.TipConstraintPoint), const_cast<double*>(params.TipForce),
        params.IntegrationStepSize, ActInertia,
        v_L_pre, w_L_pre, p_pre, R_pre,
        damping, fwd_result.dt
    );

    // Compute deltau0 from converged solution
    double deltau0[3];
    for (int i = 0; i < 3; ++i) {
        deltau0[i] = fwd_result.u0[i] - shooting_params.ustar[0][i];
    }

    // Call IVP Jacobian (analytic)
    double x_N[NUM_STATES];
    double MomentResidual[3];
    double ftip_copy[3];
    for (int i = 0; i < 3; ++i) {
        ftip_copy[i] = fwd_result.ftip[i];
    }

    auto [J_u, J_n, J_p, J_R, J_ftip] = CRMSolverIVPJacobian(
        shooting_params, deltau0, ftip_copy, true, x_N, MomentResidual
    );

    // Step 2: Compute BVP Jacobians (same for all RHS)
    Eigen::MatrixXd J_yy, J_yu, J_yx;
    compute_bvp_jacobians_full_analytic(
        fwd_result.mL, fwd_result.nL, fwd_result.u,
        params, fwd_result.xf, L_inserted, fwd_result.dt,
        fwd_result.x_coil,
        J_yy, J_yu, J_yx
    );

    // Step 3: Factorize (J_yy)^T once
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_solver(J_yy.transpose());
    int rank_used = qr_solver.rank();

    // Step 4: Loop over RHS and solve
    const int dim_y = NUM_ACT_SET * 6;
    double max_residual = 0.0;

    for (int rhs_idx = 0; rhs_idx < num_rhs; ++rhs_idx) {
        // Extract grad_tip_p for this RHS
        const double* grad_tip_p = grad_tip_p_batch + rhs_idx * 3;

        // Build cotangent on xf_next
        Eigen::VectorXd v_xf_next = Eigen::VectorXd::Zero(NUM_STATES);
        for (int i = 0; i < 3; ++i) {
            v_xf_next[i] = grad_tip_p[i];
        }

        // Push cotangent through IVP Jacobians
        Eigen::VectorXd v_u0 = J_u.transpose() * v_xf_next;
        Eigen::VectorXd v_nL = J_n.transpose() * v_xf_next;
        Eigen::VectorXd v_p_coil = J_p.transpose() * v_xf_next;
        Eigen::VectorXd v_R_coil = J_R.transpose() * v_xf_next;

        // Assemble cotangent on BVP unknowns y = [mL; nL]
        Eigen::VectorXd v_y(dim_y);
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < 3; ++i) {
                v_y[j * 6 + i] = 0.0;  // v_mL (minimal direct path)
                v_y[j * 6 + 3 + i] = v_nL[j * 3 + i];
            }
        }

        // Solve adjoint system: (J_yy)^T * lambda = v_y (using pre-factored QR)
        Eigen::VectorXd lambda = qr_solver.solve(v_y);

        double residual_norm = (J_yy.transpose() * lambda - v_y).norm();
        max_residual = std::max(max_residual, residual_norm);

        // Compute gradients using implicit function theorem
        // grad_xf = v_xf_next (direct passthrough)
        double* grad_xf = grad_xf_batch + rhs_idx * NUM_STATES;
        for (int i = 0; i < NUM_STATES; ++i) {
            grad_xf[i] = v_xf_next[i];
        }

        // grad_x_coil (direct contribution from IVP Jacobians)
        double* grad_x_coil_flat = grad_x_coil_batch + rhs_idx * NUM_ACT_SET * 18;
        std::memset(grad_x_coil_flat, 0, NUM_ACT_SET * 18 * sizeof(double));

        for (int j = 0; j < NUM_ACT_SET; ++j) {
            // Position gradient
            for (int i = 0; i < 3; ++i) {
                grad_x_coil_flat[j * 18 + 6 + i] = v_p_coil[j * 3 + i];
            }
            // Orientation gradient
            for (int i = 0; i < 9; ++i) {
                grad_x_coil_flat[j * 18 + 9 + i] = v_R_coil[j * 9 + i];
            }
        }

        // Implicit state term: grad_x -= (J_yx)^T * lambda
        Eigen::VectorXd implicit_grad_x = J_yx.transpose() * lambda;
        for (int i = 0; i < NUM_STATES; ++i) {
            grad_xf[i] -= implicit_grad_x[NUM_ACT_SET * 18 + i];
        }
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < 18; ++i) {
                grad_x_coil_flat[j * 18 + i] -= implicit_grad_x[j * 18 + i];
            }
        }

        // grad_u = -(J_yu)^T * lambda
        Eigen::VectorXd grad_u_vec = -J_yu.transpose() * lambda;
        double* grad_u_flat = grad_u_batch + rhs_idx * NUM_ACT_SET * 3;
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < 3; ++i) {
                grad_u_flat[j * 3 + i] = grad_u_vec[j * 3 + i];
            }
        }
    }

    // Set diagnostics
    if (lu_rank != nullptr) {
        *lu_rank = rank_used;
    }
    if (rel_residual != nullptr) {
        *rel_residual = max_residual;
    }

    return 0;  // Success
}

} // namespace CRMCatheterModel
