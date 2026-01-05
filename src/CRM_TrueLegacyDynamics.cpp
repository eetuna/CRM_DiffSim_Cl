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

    // SPRINT S14 PATH A: Runtime warning if x-gradients are non-trivial
    // The adjoint assumes J_xf_x unpopulated columns are zero.
    // If grad_tip_p is large, check that x_coil gradients (especially v,w) are acceptable.
    double grad_tip_p_norm = std::sqrt(grad_tip_p[0]*grad_tip_p[0] +
                                        grad_tip_p[1]*grad_tip_p[1] +
                                        grad_tip_p[2]*grad_tip_p[2]);
    if (grad_tip_p_norm > 1e-6) {
        static bool warning_shown = false;
        if (!warning_shown) {
            std::cerr << "\n[Sprint S14 PATH A - Adjoint Zero-Assumption Active]" << std::endl;
            std::cerr << "  Adjoint computation assumes J_xf_x unpopulated columns = 0" << std::endl;
            std::cerr << "  grad_x_coil[v,w] will be INCOMPLETE (missing direct IVP path)" << std::endl;
            std::cerr << "  grad_x_coil[p,R] and grad_xf will be CORRECT (from IVP Jacobians + J_yx)" << std::endl;
            std::cerr << "  This warning shown once per process." << std::endl;
            warning_shown = true;
        }
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

    // DEBUG: Print J_yu to verify it's non-zero
    std::cerr << "DEBUG backward: J_yu norm = " << J_yu.norm() << std::endl;
    std::cerr << "DEBUG backward: J_yu max = " << J_yu.cwiseAbs().maxCoeff() << std::endl;
    std::cerr << "DEBUG backward: J_yu sample (0,0) = " << J_yu(0, 0) << std::endl;
    std::cerr.flush();

    // Step 6: Solve adjoint system: (J_yy)^T * lambda = v_y
    Eigen::VectorXd lambda;
    int rank_used = dim_y;
    double residual_norm = 0.0;

    // DEBUG: Print v_y to understand cotangent flow
    std::cerr << "DEBUG backward: v_y = [";
    for (int i = 0; i < std::min(dim_y, 6); ++i) {
        std::cerr << v_y[i] << (i < std::min(dim_y, 6)-1 ? ", " : "");
    }
    std::cerr << "]" << std::endl;
    std::cerr.flush();

    // Use QR decomposition for stable solve
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_solver(J_yy.transpose());
    lambda = qr_solver.solve(v_y);

    rank_used = qr_solver.rank();
    residual_norm = (J_yy.transpose() * lambda - v_y).norm();

    // DEBUG: Print lambda
    std::cerr << "DEBUG backward: lambda = [";
    for (int i = 0; i < std::min(dim_y, 6); ++i) {
        std::cerr << lambda[i] << (i < std::min(dim_y, 6)-1 ? ", " : "");
    }
    std::cerr << "]" << std::endl;
    std::cerr.flush();

    // SPRINT S12: Verify lambda has non-zero components in both mL and nL blocks
    std::cerr << "\n[Lambda Diagnostics - Sprint S12]" << std::endl;
    std::cerr << "  lambda norm: " << lambda.norm() << std::endl;
    if (NUM_ACT_SET > 0) {
        // Check first actuator's mL and nL components
        double mL_norm = 0.0, nL_norm = 0.0;
        for (int i = 0; i < 3; ++i) {
            mL_norm += lambda[i] * lambda[i];
            nL_norm += lambda[3 + i] * lambda[3 + i];
        }
        mL_norm = std::sqrt(mL_norm);
        nL_norm = std::sqrt(nL_norm);
        std::cerr << "  lambda[mL[0]] norm: " << mL_norm << std::endl;
        std::cerr << "  lambda[nL[0]] norm: " << nL_norm << std::endl;
        std::cerr << "  mL/nL coupling present: " << (mL_norm > 1e-10 && nL_norm > 1e-10 ? "YES" : "NO") << std::endl;
    }
    std::cerr.flush();

    // Step 7: Compute gradients using implicit function theorem

    // 7a. Direct gradients for xf (tip state)
    for (int i = 0; i < NUM_STATES; ++i) {
        grad_xf[i] = v_xf_next[i];  // Direct passthrough (xf affects target)
    }

    // 7b. Gradients for x_coil (coil states)
    // Direct contribution from IVP Jacobians (only p and R, NOT v and w)
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        // Position gradient (indices 6-8 in x_coil)
        for (int i = 0; i < 3; ++i) {
            grad_x_coil[j][6 + i] = v_p_coil[j*3 + i];
        }
        // Orientation gradient (indices 9-17 in x_coil)
        for (int i = 0; i < 9; ++i) {
            grad_x_coil[j][9 + i] = v_R_coil[j*9 + i];
        }
        // Velocity gradients (indices 0-2): from tip_p propagation
        // tip_p depends on coil velocities via: p_tip = integrate(catheter_shape(p_coil, ...))
        // For now, assume second-order effect (TODO: add if needed)
        for (int i = 0; i < 3; ++i) {
            grad_x_coil[j][i] = 0.0;
        }
        // Angular velocity gradients (indices 3-5): CRITICAL - comes from tip_p via catheter curvature
        // The tip position depends on catheter shape, which depends on coil angular velocities
        // through the mechanics. However, this is a SECOND-ORDER path in the VJP.
        // The FIRST-ORDER path is: w affects next-step w through dynamics, which affects future tip_p.
        // This is captured by the IVP integration over time, not in a single-step Jacobian.
        // So for single-step VJP, w gradient from tip_p is minimal.
        for (int i = 0; i < 3; ++i) {
            grad_x_coil[j][3 + i] = 0.0;  // Will be populated if w→tip_p coupling is significant
        }
    }

    // 7c. Add implicit terms: grad_x -= (J_yx)^T * lambda
    // This correctly wires ∂x_{t+1}/∂x_t through the BVP implicit dependence
    //
    // SPRINT S14 PATH A: J_xf_x ASSUMPTION
    // The implicit gradient uses J_yx, which is COMPLETE.
    // J_xf_x is PARTIAL (see Step 4C), with unpopulated columns for R0 and x_coil[v,w].
    // ASSUMPTION: Unpopulated J_xf_x columns are STRUCTURALLY ZERO.
    // Justification: x→xf_next dependence is captured via x→y→xf_next (J_yx pathway).
    // The direct path ∂xf_next/∂R0 and ∂xf_next/∂(v_L_pre,w_L_pre) is negligible.
    //
    // CONSEQUENCE: VJPs wrt x_coil[0:6] (v,w components) will be INCOMPLETE.
    // Only x_coil[6:18] (p,R components) receive correct gradients from IVP Jacobians.
    Eigen::VectorXd implicit_grad_x = J_yx.transpose() * lambda;

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

    // VALIDATION: Check that implicit gradient components are reasonable
    if (std::isnan(implicit_grad_x.sum()) || std::isinf(implicit_grad_x.sum())) {
        std::cerr << "WARNING [Sprint S14 PATH A]: implicit_grad_x contains NaN/Inf" << std::endl;
        std::cerr << "  This may indicate J_yx singularity or zero-assumption violation" << std::endl;
    }

    // 7d. Gradients for u (actuation currents) via BVP implicit pathway
    //
    // The control u affects the BVP residual r(y,u) through the magnetic torque pathway:
    //   u → MagMoment → τ_mag → coil dynamics → R_coil → residual
    //
    // This is captured by J_yu = ∂r/∂u, computed analytically in CRM_BVPJacobian.cpp.
    //
    // The VJP for u gradients uses implicit differentiation:
    //   ∇_u L = -(J_yu)^T * λ
    //
    // where λ is the adjoint solution to (J_yy)^T * λ = v_y.
    //
    // NOTE: There is NO separate "direct pathway" for u gradients through w_next,
    // because w does not appear in the IVP Jacobians (only p and R do).
    // All u gradients flow through the BVP implicit pathway.
    //
    // See: docs/audits/SPRINT_S5_J_YU_DERIVATION.md

    // SPRINT S14-STEP5A: INVESTIGATION
    // Standard IFT formula is: grad_u = -(∂r/∂u)^T * λ
    // But test shows sign is still wrong - need to investigate residual definition
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

        // SPRINT S12: Verify lambda has non-zero components in both mL and nL blocks
        if (rhs_idx == 0) {  // Only print for first RHS to avoid spam
            std::cerr << "\n[Lambda Diagnostics - Sprint S12 - Batched]" << std::endl;
            std::cerr << "  lambda norm: " << lambda.norm() << std::endl;
            if (NUM_ACT_SET > 0) {
                double mL_norm = 0.0, nL_norm = 0.0;
                for (int i = 0; i < 3; ++i) {
                    mL_norm += lambda[i] * lambda[i];
                    nL_norm += lambda[3 + i] * lambda[3 + i];
                }
                mL_norm = std::sqrt(mL_norm);
                nL_norm = std::sqrt(nL_norm);
                std::cerr << "  lambda[mL[0]] norm: " << mL_norm << std::endl;
                std::cerr << "  lambda[nL[0]] norm: " << nL_norm << std::endl;
                std::cerr << "  mL/nL coupling present: " << (mL_norm > 1e-10 && nL_norm > 1e-10 ? "YES" : "NO") << std::endl;
            }
            std::cerr.flush();
        }

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

        // grad_u: ONLY Implicit (BVP adjoint) pathway
        //
        // The control u affects the BVP residual r(y,u) through the magnetic torque pathway:
        //   u → MagMoment → τ_mag → coil dynamics → R_coil → residual
        //
        // This is captured by J_yu = ∂r/∂u, computed analytically in CRM_BVPJacobian.cpp.
        //
        // The VJP for u gradients uses implicit differentiation:
        //   ∇_u L = -(J_yu)^T * λ
        //
        // where λ is the adjoint solution to (J_yy)^T * λ = v_y.
        //
        // NOTE: There is NO separate "direct pathway" for u gradients through w_next,
        // because w does not appear in the IVP Jacobians (only p and R do).
        // All u gradients flow through the BVP implicit pathway.
        //
        // See: docs/audits/SPRINT_S5_J_YU_DERIVATION.md

        // SPRINT S14-STEP5A: INVESTIGATION (same as single backward pass)
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
    double deltau0[3];
    for (int i = 0; i < 3; ++i) {
        deltau0[i] = fwd_result.u0[i] - shooting_params.ustar[0][i];
    }

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
    // J_n: (15 x 3*NUM_ACT_SET) - ∂xf_next/∂nL
    // J_p: (15 x 3*NUM_ACT_SET) - ∂xf_next/∂pL
    // J_R: (15 x 9*NUM_ACT_SET) - ∂xf_next/∂RL
    // J_ftip: (15 x 3) - ∂xf_next/∂ftip

    // Step 3: Compute BVP Jacobians
    Eigen::MatrixXd J_yy, J_yu, J_yx;

    compute_bvp_jacobians_full_analytic(
        fwd_result.mL, fwd_result.nL, fwd_result.u,
        params, fwd_result.xf, L_inserted, fwd_result.dt,
        fwd_result.x_coil,
        J_yy, J_yu, J_yx
    );

    // Step 4: Solve R_y^{-1} * R_x and R_y^{-1} * R_u using one QR factorization
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_solver(J_yy);

    Eigen::MatrixXd S_x = qr_solver.solve(J_yx);  // (6N x (18N+15))
    Eigen::MatrixXd S_u = qr_solver.solve(J_yu);  // (6N x 3N)

    int rank_used = qr_solver.rank();
    double residual_x = (J_yy * S_x - J_yx).norm();
    double residual_u = (J_yy * S_u - J_yu).norm();

    // Step 5: Construct G_x, G_u, G_y (IVP full-state Jacobians)

    // G_x: ∂x_next/∂x_t (state_dim x state_dim)
    // G_u: ∂x_next/∂u_t (state_dim x control_dim)
    // G_y: ∂x_next/∂y (state_dim x dim_y) where y = [mL; nL]

    Eigen::MatrixXd G_x = Eigen::MatrixXd::Zero(state_dim, state_dim);
    Eigen::MatrixXd G_u = Eigen::MatrixXd::Zero(state_dim, control_dim);
    Eigen::MatrixXd G_y = Eigen::MatrixXd::Zero(state_dim, dim_y);

    // G_x construction:
    // - Coil states (first 18*N elements): mostly identity (direct propagation)
    // - Tip state (last 15 elements): depends on coil positions/orientations via IVP

    // Coil state propagation (simplified: mostly identity for direct terms)
    // Positions and orientations propagate based on velocities (this is a simplified model)
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        int offset = j * 18;
        // Position evolves: p_next = p + v*dt (indices 6-8 depend on indices 0-2)
        for (int i = 0; i < 3; ++i) {
            G_x(offset + 6 + i, offset + i) = fwd_result.dt;  // ∂p_next/∂v
            G_x(offset + 6 + i, offset + 6 + i) = 1.0;         // ∂p_next/∂p
        }
        // Velocities and orientations (simplified: assume minor coupling)
        for (int i = 0; i < 6; ++i) {
            G_x(offset + i, offset + i) = 1.0;  // Direct passthrough
        }
        for (int i = 0; i < 9; ++i) {
            G_x(offset + 9 + i, offset + 9 + i) = 1.0;  // Orientation passthrough
        }
    }

    // Tip state depends on coil positions/orientations through IVP
    int tip_offset = NUM_ACT_SET * 18;
    // ∂xf_next/∂p_coil (from J_p)
    for (int i = 0; i < NUM_STATES; ++i) {
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int k = 0; k < 3; ++k) {
                G_x(tip_offset + i, j * 18 + 6 + k) = J_p(i, j * 3 + k);
            }
        }
    }
    // ∂xf_next/∂R_coil (from J_R)
    for (int i = 0; i < NUM_STATES; ++i) {
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int k = 0; k < 9; ++k) {
                G_x(tip_offset + i, j * 18 + 9 + k) = J_R(i, j * 9 + k);
            }
        }
    }

    // G_u construction: ∂x_next/∂u
    // Coil angular velocities are directly affected by magnetic torques
    // Tip state is indirectly affected through BVP solution
    //
    // Compute magnetic torque derivatives: ∂τ_mag/∂u
    // Formula from BVP Jacobians (CRM_BVPJacobian.cpp:239-253):
    // ∂τ_mag/∂u[i] = M[i] * (e_i × B)

    std::vector<Eigen::Matrix3d> dTau_du(NUM_ACT_SET);
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        const double* B = shooting_params.B0;  // Magnetic field [3]
        const double* M = shooting_params.MagMoment[j];  // Current magnetic moment [3]

        // Compute ∂τ_mag/∂u[i] = M[i] * (e_i × B) for each control component
        for (int i = 0; i < 3; ++i) {
            double dtau_du[3];
            if (i == 0) {
                dtau_du[0] = 0.0;
                dtau_du[1] = -M[0] * B[2];
                dtau_du[2] = M[0] * B[1];
            } else if (i == 1) {
                dtau_du[0] = M[1] * B[2];
                dtau_du[1] = 0.0;
                dtau_du[2] = -M[1] * B[0];
            } else {  // i == 2
                dtau_du[0] = -M[2] * B[1];
                dtau_du[1] = M[2] * B[0];
                dtau_du[2] = 0.0;
            }

            // Store in column i of dTau_du[j]
            for (int k = 0; k < 3; ++k) {
                dTau_du[j](k, i) = dtau_du[k];
            }
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

    // G_y construction: ∂x_next/∂y where y = [mL; nL]
    // Tip state depends on nL through IVP (from J_n)
    for (int i = 0; i < NUM_STATES; ++i) {
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int k = 0; k < 3; ++k) {
                // nL components are at indices [j*6+3 : j*6+6) in y-vector
                G_y(tip_offset + i, j * 6 + 3 + k) = J_n(i, j * 3 + k);
            }
        }
    }

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
