#include "CRM_TrueLegacyDynamics.hpp"
#include "CRM_MatrixOperations.hpp"
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

    // Construct actuator inertia using hollow cylinder formula
    // Units: kg * mm^2 (mass in kg, radii and lengths in mm)
    double ActInertia[NUM_ACT_SET][9];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        double mass = params.CathParams->ActMass[j];
        double r_outer = params.CathParams->OuterRadius[0];  // Use first flexible segment radii
        double r_inner = params.CathParams->InnerRadius[0];
        double seg_length = params.CathParams->SegLengths[2*j + 1];  // Actuator segment length

        double r_sum_sq = r_outer * r_outer + r_inner * r_inner;
        double I_zz = 0.5 * mass * r_sum_sq;  // Moment about cylinder axis
        double I_xx = 0.25 * mass * r_sum_sq + (1.0/12.0) * mass * seg_length * seg_length;  // Perpendicular

        ActInertia[j][0] = I_xx;  ActInertia[j][1] = 0.0;   ActInertia[j][2] = 0.0;
        ActInertia[j][3] = 0.0;   ActInertia[j][4] = I_xx;  ActInertia[j][5] = 0.0;
        ActInertia[j][6] = 0.0;   ActInertia[j][7] = 0.0;   ActInertia[j][8] = I_zz;
    }

    // Load damping coefficients from catheter parameters
    double damping[NUM_ACT_SET][6];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 6; ++i) {
            damping[j][i] = params.CathParams->ActDamping[j][i];
        }
    }

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

    // Check BVP convergence (localmin == 0 means success)
    out.converged = (out.localmin == 0) ? 1 : 0;

    // CRITICAL: Only call IVP if BVP converged
    // If BVP failed, return previous state unchanged (do NOT advance)
    if (out.localmin != 0) {
        // BVP did not converge - reject this step
        // Return previous state unchanged
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < 18; ++i) {
                out.x_coil_next[j][i] = x_coil[j][i];
            }
        }
        for (int i = 0; i < NUM_STATES; ++i) {
            out.xf_next[i] = xf[i];
        }

        // Extract observables from unchanged state
        for (int i = 0; i < 3; ++i) {
            out.tip_p[i] = xf[i];
        }
        for (int i = 0; i < 9; ++i) {
            out.tip_R[i] = xf[3 + i];
        }
        for (int i = 0; i < 3; ++i) {
            out.tip_u[i] = xf[12 + i];
        }

        return 1;  // Signal failure
    }

    // BVP converged successfully - proceed with IVP integration
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

    return 0;  // Success
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

    // Step 2: Compute IVP Jacobians analytically using DYNSolverIVP_JacobiansFullstate
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

    // (debug lambda removed)

    // (debug lambda removed)


    // Construct actuator inertia using hollow cylinder formula
    // Units: kg * mm^2 (mass in kg, radii and lengths in mm)
    double ActInertia[NUM_ACT_SET][9];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        double mass = params.CathParams->ActMass[j];
        double r_outer = params.CathParams->OuterRadius[0];
        double r_inner = params.CathParams->InnerRadius[0];
        double seg_length = params.CathParams->SegLengths[2*j + 1];

        double r_sum_sq = r_outer * r_outer + r_inner * r_inner;
        double I_zz = 0.5 * mass * r_sum_sq;
        double I_xx = 0.25 * mass * r_sum_sq + (1.0/12.0) * mass * seg_length * seg_length;

        ActInertia[j][0] = I_xx;  ActInertia[j][1] = 0.0;   ActInertia[j][2] = 0.0;
        ActInertia[j][3] = 0.0;   ActInertia[j][4] = I_xx;  ActInertia[j][5] = 0.0;
        ActInertia[j][6] = 0.0;   ActInertia[j][7] = 0.0;   ActInertia[j][8] = I_zz;
    }

    // Load damping coefficients from catheter parameters
    double damping[NUM_ACT_SET][6];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 6; ++i) {
            damping[j][i] = params.CathParams->ActDamping[j][i];
        }
    }

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

    Eigen::MatrixXd J_xf_y, J_xf_x, J_xcoil_y, J_xcoil_x;
    DYNSolverIVP_JacobiansFullstate(
        shooting_params,
        fwd_result.u0,
        fwd_result.mL,
        fwd_result.nL,
        fwd_result.tau,
        fwd_result.ftip,
        fwd_result.x_coil,
        fwd_result.xf,
        J_xf_y,
        J_xf_x,
        J_xcoil_y,
        J_xcoil_x
    );
    // Step 3: Push cotangent through IVP Jacobians
    // v_y = J^T * v_xf_next, where y includes outputs that depend on BVP solution

    Eigen::VectorXd v_y = J_xf_y.transpose() * v_xf_next;

    // Step 4: Assemble cotangent on BVP unknowns y = [mL; nL]
    // The BVP solves: r(y; x_coil, xf, u, dt) = 0 where y = [mL[0], nL[0], ..., mL[N-1], nL[N-1]]
    // Dimension: 6*NUM_ACT_SET (3 for mL + 3 for nL per actuator)

    const int dim_y = NUM_ACT_SET * 6;

    // Step 5: Compute BVP Jacobian blocks J_yy, J_yu, and J_yx using strictly analytic methods
    Eigen::MatrixXd J_yy, J_yu, J_yx;

    compute_bvp_jacobians_full_analytic(
        fwd_result.mL, fwd_result.nL, fwd_result.u,
        params, fwd_result.xf, L_inserted, fwd_result.dt,
        fwd_result.x_coil,
        J_yy, J_yu, J_yx
    );

    // Step 6: Precompute implicit sensitivities from J_yy.
    int rank_used = dim_y;
    double residual_norm = 0.0;

    // Use QR decomposition for stable solve
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_solver(J_yy);
    Eigen::MatrixXd S_x = qr_solver.solve(J_yx);
    Eigen::MatrixXd S_u = qr_solver.solve(J_yu);

    rank_used = qr_solver.rank();
    residual_norm = std::max(
        (J_yy * S_x - J_yx).norm(),
        (J_yy * S_u - J_yu).norm()
    );

    // Step 7: Compute gradients using implicit function theorem

    Eigen::VectorXd grad_x_direct = J_xf_x.transpose() * v_xf_next;

    for (int i = 0; i < NUM_STATES; ++i) {
        grad_xf[i] = grad_x_direct[NUM_ACT_SET * 18 + i];
    }

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 18; ++i) {
            grad_x_coil[j][i] = grad_x_direct[j * 18 + i];
        }
    }

    // 7c. Add implicit terms: grad_x -= S_x^T * v_y
    // This correctly wires ∂x_{t+1}/∂x_t through the BVP implicit dependence
    Eigen::VectorXd implicit_grad_x = S_x.transpose() * v_y;

    // Apply to grad_xf
    for (int i = 0; i < NUM_STATES; ++i) {
        grad_xf[i] -= implicit_grad_x[NUM_ACT_SET * 18 + i];
    }

    // Apply to grad_x_coil (including w components)
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 18; ++i) {
            grad_x_coil[j][i] -= implicit_grad_x[j * 18 + i];
        }
    }

    Eigen::VectorXd grad_u_vec = -S_u.transpose() * v_y;

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

    // Construct actuator inertia using hollow cylinder formula
    // Units: kg * mm^2 (mass in kg, radii and lengths in mm)
    double ActInertia[NUM_ACT_SET][9];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        double mass = params.CathParams->ActMass[j];
        double r_outer = params.CathParams->OuterRadius[0];
        double r_inner = params.CathParams->InnerRadius[0];
        double seg_length = params.CathParams->SegLengths[2*j + 1];

        double r_sum_sq = r_outer * r_outer + r_inner * r_inner;
        double I_zz = 0.5 * mass * r_sum_sq;
        double I_xx = 0.25 * mass * r_sum_sq + (1.0/12.0) * mass * seg_length * seg_length;

        ActInertia[j][0] = I_xx;  ActInertia[j][1] = 0.0;   ActInertia[j][2] = 0.0;
        ActInertia[j][3] = 0.0;   ActInertia[j][4] = I_xx;  ActInertia[j][5] = 0.0;
        ActInertia[j][6] = 0.0;   ActInertia[j][7] = 0.0;   ActInertia[j][8] = I_zz;
    }

    // Load damping coefficients from catheter parameters
    double damping[NUM_ACT_SET][6];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 6; ++i) {
            damping[j][i] = params.CathParams->ActDamping[j][i];
        }
    }

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

    Eigen::MatrixXd J_xf_y, J_xf_x, J_xcoil_y, J_xcoil_x;
    DYNSolverIVP_JacobiansFullstate(
        shooting_params,
        fwd_result.u0,
        fwd_result.mL,
        fwd_result.nL,
        fwd_result.tau,
        fwd_result.ftip,
        fwd_result.x_coil,
        fwd_result.xf,
        J_xf_y,
        J_xf_x,
        J_xcoil_y,
        J_xcoil_x
    );

    // Step 2: Compute BVP Jacobians (same for all RHS)
    Eigen::MatrixXd J_yy, J_yu, J_yx;
    compute_bvp_jacobians_full_analytic(
        fwd_result.mL, fwd_result.nL, fwd_result.u,
        params, fwd_result.xf, L_inserted, fwd_result.dt,
        fwd_result.x_coil,
        J_yy, J_yu, J_yx
    );

    // Step 3: Factorize J_yy once and precompute sensitivities
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_solver(J_yy);
    Eigen::MatrixXd S_x = qr_solver.solve(J_yx);
    Eigen::MatrixXd S_u = qr_solver.solve(J_yu);
    int rank_used = qr_solver.rank();

    // Step 4: Loop over RHS and apply v_y
    double max_residual = std::max(
        (J_yy * S_x - J_yx).norm(),
        (J_yy * S_u - J_yu).norm()
    );

    for (int rhs_idx = 0; rhs_idx < num_rhs; ++rhs_idx) {
        // Extract grad_tip_p for this RHS
        const double* grad_tip_p = grad_tip_p_batch + rhs_idx * 3;

        // Build cotangent on xf_next
        Eigen::VectorXd v_xf_next = Eigen::VectorXd::Zero(NUM_STATES);
        for (int i = 0; i < 3; ++i) {
            v_xf_next[i] = grad_tip_p[i];
        }

        Eigen::VectorXd v_y = J_xf_y.transpose() * v_xf_next;

        Eigen::VectorXd grad_x_direct = J_xf_x.transpose() * v_xf_next;
        const int dim_xcoil = NUM_ACT_SET * 18;

        double* grad_xf = grad_xf_batch + rhs_idx * NUM_STATES;
        for (int i = 0; i < NUM_STATES; ++i) {
            grad_xf[i] = grad_x_direct[dim_xcoil + i];
        }

        double* grad_x_coil_flat = grad_x_coil_batch + rhs_idx * NUM_ACT_SET * 18;
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < 18; ++i) {
                grad_x_coil_flat[j * 18 + i] = grad_x_direct[j * 18 + i];
            }
        }

        // Implicit state term: grad_x -= S_x^T * v_y
        Eigen::VectorXd implicit_grad_x = S_x.transpose() * v_y;
        for (int i = 0; i < NUM_STATES; ++i) {
            grad_xf[i] -= implicit_grad_x[NUM_ACT_SET * 18 + i];
        }
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < 18; ++i) {
                grad_x_coil_flat[j * 18 + i] -= implicit_grad_x[j * 18 + i];
            }
        }

        Eigen::VectorXd grad_u_vec = -S_u.transpose() * v_y;

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

// Linearization: Compute A, B matrices using implicit function theorem
// A = ∂x_next/∂x_t, B = ∂x_next/∂u_t
// Uses formulas: A = G_x - G_y * (R_y^{-1} * R_x), B = G_u - G_y * (R_y^{-1} * R_u)
int true_legacy_linearize_implicit(
    const TrueLegacyStepResult& fwd_result,
    const CRMForwardKinematicsData& params,
    double L_inserted,
    Eigen::MatrixXd& A_out,
    Eigen::MatrixXd& B_out,
    int* qr_rank,
    double* rel_residual
) {
    const int state_dim = NUM_ACT_SET * 18 + NUM_STATES;  // 18*N + 15
    const int control_dim = NUM_ACT_SET * 3;              // 3*N
    const int dim_y = NUM_ACT_SET * 6;                    // 6*N (BVP unknowns)

    // Initialize output matrices
    A_out = Eigen::MatrixXd::Zero(state_dim, state_dim);
    B_out = Eigen::MatrixXd::Zero(state_dim, control_dim);

    // Step 1: Reconstruct shooting params (same as backward pass)
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

    // Construct actuator inertia using hollow cylinder formula
    // Units: kg * mm^2 (mass in kg, radii and lengths in mm)
    double ActInertia[NUM_ACT_SET][9];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        double mass = params.CathParams->ActMass[j];
        double r_outer = params.CathParams->OuterRadius[0];
        double r_inner = params.CathParams->InnerRadius[0];
        double seg_length = params.CathParams->SegLengths[2*j + 1];

        double r_sum_sq = r_outer * r_outer + r_inner * r_inner;
        double I_zz = 0.5 * mass * r_sum_sq;
        double I_xx = 0.25 * mass * r_sum_sq + (1.0/12.0) * mass * seg_length * seg_length;

        ActInertia[j][0] = I_xx;  ActInertia[j][1] = 0.0;   ActInertia[j][2] = 0.0;
        ActInertia[j][3] = 0.0;   ActInertia[j][4] = I_xx;  ActInertia[j][5] = 0.0;
        ActInertia[j][6] = 0.0;   ActInertia[j][7] = 0.0;   ActInertia[j][8] = I_zz;
    }

    // Load damping coefficients from catheter parameters
    double damping[NUM_ACT_SET][6];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 6; ++i) {
            damping[j][i] = params.CathParams->ActDamping[j][i];
        }
    }

    CRMShootingMethodParams shooting_params = CRMDYNConstructShootingMethodParamSet(
        *params.CathParams, *params.CathConfig,
        L_inserted, ActuationCurrents,
        params.ContactMode,
        const_cast<double*>(params.TipConstraintPoint), const_cast<double*>(params.TipForce),
        params.IntegrationStepSize, ActInertia,
        v_L_pre, w_L_pre, p_pre, R_pre,
        damping, fwd_result.dt
    );

    // Step 2: Compute IVP Jacobians
    Eigen::MatrixXd J_xf_y, J_xf_x, J_xcoil_y, J_xcoil_x;
    DYNSolverIVP_JacobiansFullstate(
        shooting_params,
        fwd_result.u0,
        fwd_result.mL,
        fwd_result.nL,
        fwd_result.tau,
        fwd_result.ftip,
        fwd_result.x_coil,
        fwd_result.xf,
        J_xf_y,
        J_xf_x,
        J_xcoil_y,
        J_xcoil_x
    );

    // Step 3: Compute BVP Jacobians
    Eigen::MatrixXd J_yy, J_yu, J_yx;

    compute_bvp_jacobians_full_analytic(
        fwd_result.mL, fwd_result.nL, fwd_result.u,
        params, fwd_result.xf, L_inserted, fwd_result.dt,
        fwd_result.x_coil,
        J_yy, J_yu, J_yx
    );

    // Step 4: Solve R_y^{-1} * R_x and R_y^{-1} * R_u
    // Use full-pivot LU for robustness with potentially ill-conditioned J_yy.
    Eigen::FullPivLU<Eigen::MatrixXd> lu_solver(J_yy);

    Eigen::MatrixXd S_x = lu_solver.solve(J_yx);  // (6N x (18N+15))
    Eigen::MatrixXd S_u = lu_solver.solve(J_yu);  // (6N x 3N)

    int rank_used = lu_solver.rank();
    double residual_x = (J_yy * S_x - J_yx).norm();
    double residual_u = (J_yy * S_u - J_yu).norm();

    // Step 5: Construct G_x, G_u, G_y (IVP full-state Jacobians)

    // G_x: ∂x_next/∂x_t (state_dim x state_dim)
    // G_u: ∂x_next/∂u_t (state_dim x control_dim)
    // G_y: ∂x_next/∂y (state_dim x dim_y) where y = [mL; nL]

    Eigen::MatrixXd G_x = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd G_u = Eigen::MatrixXd::Zero(state_dim, control_dim);
    Eigen::MatrixXd G_y = Eigen::MatrixXd::Zero(state_dim, dim_y);

    const int dim_xcoil = NUM_ACT_SET * 18;
    G_x.topRows(dim_xcoil) = J_xcoil_x;
    G_x.bottomRows(NUM_STATES) = J_xf_x;

    G_y.topRows(dim_xcoil) = J_xcoil_y;
    G_y.bottomRows(NUM_STATES) = J_xf_y;

    // G_u construction: ∂x_next/∂u
    // Coil angular velocities are directly affected by magnetic torques
    // Tip state is indirectly affected through BVP solution
    //
    // Compute magnetic torque derivatives (full turn-area matrix)
    std::vector<Eigen::Matrix3d> dTau_du(NUM_ACT_SET);
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        const double* B = shooting_params.B0;
        const double* A = shooting_params.CoilAlignmentTurnAreaMatrix[j];
        dTau_du[j].setZero();
        for (int i = 0; i < 3; ++i) {
            double dM[3] = {A[0 * 3 + i], A[1 * 3 + i], A[2 * 3 + i]};
            dTau_du[j](0, i) = dM[1] * B[2] - dM[2] * B[1];
            dTau_du[j](1, i) = dM[2] * B[0] - dM[0] * B[2];
            dTau_du[j](2, i) = dM[0] * B[1] - dM[1] * B[0];
        }
    }

    // Populate G_u with direct magnetic torque effect on angular velocities
    // ∂w_next/∂u = dt * ∂wdot/∂u = dt * (∂τ_mag/∂u) / I
    double dt = fwd_result.dt;

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        // Get coil inertia (diagonal 3x3 matrix stored as 9 elements)
        // Diagonal elements at indices: 0, 4, 8
        const double* actInertia = shooting_params.actInertia[j];
        double I_xx = actInertia[0];
        double I_yy = actInertia[4];
        double I_zz = actInertia[8];

        // Populate G_u for angular velocity rows
        for (int i = 0; i < 3; ++i) {  // Loop over control components
            int col = j * 3 + i;  // Control column index

            // Angular velocity rows (indices 3-5 in each coil state)
            int row_w0 = j * 18 + 3;
            int row_w1 = j * 18 + 4;
            int row_w2 = j * 18 + 5;

            G_u(row_w0, col) = dt * dTau_du[j](0, i) / I_xx;
            G_u(row_w1, col) = dt * dTau_du[j](1, i) / I_yy;
            G_u(row_w2, col) = dt * dTau_du[j](2, i) / I_zz;
        }
    }

    // Note: Other state components have zero direct control influence:
    // - ∂v_next/∂u ≈ 0 (magnetic force negligible for uniform B field)
    // - ∂p_next/∂u = 0 (p_next = p + v*dt, no direct u dependence)
    // - ∂R_next/∂u = 0 (rotation doesn't directly depend on u in first order)
    // The indirect pathway through BVP is captured by G_y * S_u term

    // G_y already populated from IVP Jacobians (J_xcoil_y, J_xf_y).

    // Step 6: Apply implicit function theorem
    // A = G_x - G_y * S_x
    // B = G_u - G_y * S_u

    A_out = G_x - G_y * S_x;
    B_out = G_u - G_y * S_u;

    // Set diagnostics
    if (qr_rank != nullptr) {
        *qr_rank = rank_used;
    }
    if (rel_residual != nullptr) {
        *rel_residual = std::max(residual_x, residual_u);
    }

    return 0;  // Success
}

} // namespace CRMCatheterModel
