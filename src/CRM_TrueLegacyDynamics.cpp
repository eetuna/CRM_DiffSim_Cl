#include "CRM_TrueLegacyDynamics.hpp"
#include "CRM_MatrixOperations.hpp"
#include "CRM_IVPJacobian.hpp"
#include "CRM_BVPJacobian.hpp"
#include <Eigen/Dense>
#include <cstring>
#include <cmath>
#include <iostream>
#include <vector>

namespace CRMCatheterModel {

// Forward declaration of the exact Jacobian computer
void DYNSolverIVP_JacobiansFullstate(
    const CRMShootingMethodParams& params,
    const double in_u0[3],
    const double in_mL[NUM_ACT_SET][3],
    const double in_nL[NUM_ACT_SET][3],
    const double in_tau[NUM_ACT_SET][3],
    const double in_ftip[3],
    double out_x_N[NUM_STATES],
    double out_coil_state[NUM_ACT_SET][NUM_COIL_STATES],
    Eigen::MatrixXd& J_xf_y,
    Eigen::MatrixXd& J_xf_x,
    Eigen::MatrixXd& J_xcoil_y,
    Eigen::MatrixXd& J_xcoil_x
);

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

    // Construct actuator inertia
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

    // Load damping coefficients
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

    // Check BVP convergence
    out.converged = (out.localmin == 0) ? 1 : 0;

    // Only call IVP if BVP converged
    if (out.localmin != 0) {
        // BVP did not converge - reject this step
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
        return 1;
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

// Backward pass using analytic implicit differentiation and Exact Forward-Mode AD
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
    // Initialize outputs
    for(int j=0; j<NUM_ACT_SET; ++j) {
        for(int i=0; i<18; ++i) grad_x_coil[j][i] = 0.0;
        for(int i=0; i<3; ++i) grad_u[j][i] = 0.0;
    }
    for(int i=0; i<NUM_STATES; ++i) grad_xf[i] = 0.0;

    // 1. Setup VJP inputs
    Eigen::VectorXd v_xf_next = Eigen::VectorXd::Zero(NUM_STATES);
    for(int i=0; i<3; ++i) v_xf_next[i] = grad_tip_p[i];

    // 2. Reconstruct Params
    // ... (Code duplication for params reconstruction - unavoidable without major refactor)
    // We reuse logic from forward pass prep
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
    for (int j = 0; j < NUM_ACT_SET; ++j) for (int i = 0; i < 3; ++i) ActuationCurrents[j][i] = fwd_result.u[j][i];
    
    double ActInertia[NUM_ACT_SET][9];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        double mass = params.CathParams->ActMass[j];
        double r_outer = params.CathParams->OuterRadius[0];
        double r_inner = params.CathParams->InnerRadius[0];
        double seg_length = params.CathParams->SegLengths[2*j + 1];
        double r_sum_sq = r_outer * r_outer + r_inner * r_inner;
        double I_zz = 0.5 * mass * r_sum_sq;
        double I_xx = 0.25 * mass * r_sum_sq + (1.0/12.0) * mass * seg_length * seg_length;
        ActInertia[j][0] = I_xx;  ActInertia[j][4] = I_xx;  ActInertia[j][8] = I_zz;
        ActInertia[j][1]=ActInertia[j][2]=ActInertia[j][3]=ActInertia[j][5]=ActInertia[j][6]=ActInertia[j][7]=0.0;
    }
    double damping[NUM_ACT_SET][6];
    for (int j = 0; j < NUM_ACT_SET; ++j) for (int i = 0; i < 6; ++i) damping[j][i] = params.CathParams->ActDamping[j][i];

    CRMShootingMethodParams shooting_params = CRMDYNConstructShootingMethodParamSet(
        *params.CathParams, *params.CathConfig,
        fwd_result.L_inserted, ActuationCurrents,
        params.ContactMode,
        const_cast<double*>(params.TipConstraintPoint), const_cast<double*>(params.TipForce),
        params.IntegrationStepSize, ActInertia,
        v_L_pre, w_L_pre, p_pre, R_pre,
        damping, fwd_result.dt
    );

    // 3. Compute Exact IVP Jacobians (Forward-Mode AD)
    Eigen::MatrixXd J_xf_y, J_xf_x, J_xcoil_y, J_xcoil_x;
    double dummy_xN[NUM_STATES], dummy_xcoil[NUM_ACT_SET][NUM_COIL_STATES]; // Outputs not needed
    
    DYNSolverIVP_JacobiansFullstate(
        shooting_params, fwd_result.u0, fwd_result.mL, fwd_result.nL, fwd_result.tau, fwd_result.ftip,
        dummy_xN, dummy_xcoil, J_xf_y, J_xf_x, J_xcoil_y, J_xcoil_x
    );

    // 4. Compute BVP Jacobians (Analytic Forward-Mode AD)
    Eigen::MatrixXd J_yy, J_yu, J_yx;
    compute_bvp_jacobians_full_analytic(
        fwd_result.mL, fwd_result.nL, fwd_result.u,
        params, fwd_result.xf, fwd_result.L_inserted, fwd_result.dt,
        fwd_result.x_coil,
        J_yy, J_yu, J_yx
    );

    // 5. Assemble Adjoint System RHS (v_y)
    // y = [mL; nL] (size 6N). IVP y input includes tau, ftip, u0.
    // J_xf_y cols: mL(0..3N), nL(3N..6N), tau(6N..9N), ftip(9N..9N+3), u0(9N+3..9N+6)
    // We map J_xf_y cols to v_y and v_direct terms.
    
    Eigen::VectorXd v_ivp_y = J_xf_y.transpose() * v_xf_next; // Cotangents on IVP inputs
    
    int dim_y_bvp = NUM_ACT_SET * 6;
    Eigen::VectorXd v_y = Eigen::VectorXd::Zero(dim_y_bvp);
    
    // Fill v_y from v_ivp_y (mL and nL parts)
    for(int j=0; j<NUM_ACT_SET; ++j) {
        // mL
        for(int k=0; k<3; ++k) v_y[j*6 + k] = v_ivp_y[j*3 + k];
        // nL
        for(int k=0; k<3; ++k) v_y[j*6 + 3 + k] = v_ivp_y[NUM_ACT_SET*3 + j*3 + k];
    }

    // 6. Solve Adjoint System (J_yy^T * lambda = v_y)
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_solver(J_yy.transpose());
    Eigen::VectorXd lambda = qr_solver.solve(v_y);
    
    if (lu_rank) *lu_rank = qr_solver.rank();
    if (rel_residual) *rel_residual = (J_yy.transpose() * lambda - v_y).norm();

    // 7. Compute Gradients

    // grad_x (implicit + explicit)
    // Explicit: J_xf_x^T * v_xf_next (direct dependence of xf_next on x_coil)
    Eigen::VectorXd grad_x_explicit = J_xf_x.transpose() * v_xf_next;
    
    // Implicit: -J_yx^T * lambda
    Eigen::VectorXd grad_x_implicit = -J_yx.transpose() * lambda;
    
    Eigen::VectorXd grad_x_total = grad_x_explicit + grad_x_implicit;
    
    // Distribute grad_x_total to grad_x_coil and grad_xf
    // x vector structure: x_coil (18N), xf (15)
    for(int j=0; j<NUM_ACT_SET; ++j) {
        for(int k=0; k<18; ++k) grad_x_coil[j][k] = grad_x_total[j*18 + k];
    }
    for(int k=0; k<NUM_STATES; ++k) grad_xf[k] = grad_x_total[NUM_ACT_SET*18 + k];

    // grad_u (implicit + explicit)
    // Explicit: dTau/du terms via tau input to IVP
    // tau is at indices 6N..9N in IVP inputs.
    // v_tau = v_ivp_y[6N..9N]
    
    Eigen::VectorXd grad_u_direct = Eigen::VectorXd::Zero(NUM_ACT_SET * 3);
    
    // dTau/du blocks
    for(int j=0; j<NUM_ACT_SET; ++j) {
        const double* B = shooting_params.B0;
        const double* M = shooting_params.MagMoment[j];
        // dTau/du is skew-symmetric-like logic (M x B)
        // tau = (C * u) x B.
        // dtau/du[i] = (dM/du[i]) x B.
        // dM/du[i] is column i of CoilAlignmentTurnAreaMatrix[j]
        
        Eigen::Matrix3d dM_du;
        for(int row=0; row<3; ++row) for(int col=0; col<3; ++col) {
            dM_du(row, col) = shooting_params.CoilAlignmentTurnAreaMatrix[j][row*3 + col];
        }
        
        // v_tau for this actuator
        Eigen::Vector3d v_tau_j;
        for(int k=0; k<3; ++k) v_tau_j[k] = v_ivp_y[NUM_ACT_SET*6 + j*3 + k];
        
        // grad_u_j += dtau_du^T * v_tau_j
        // dtau_k / du_i = (dM/du_i x B)_k
        for(int i=0; i<3; ++i) { // input u_i
            Eigen::Vector3d dM_du_i = dM_du.col(i);
            Eigen::Vector3d dtau_du_i = dM_du_i.cross(Eigen::Vector3d(B[0], B[1], B[2]));
            grad_u_direct[j*3 + i] += dtau_du_i.dot(v_tau_j);
        }
    }
    
    // Implicit: -J_yu^T * lambda
    Eigen::VectorXd grad_u_implicit = -J_yu.transpose() * lambda;
    
    Eigen::VectorXd grad_u_total = grad_u_direct + grad_u_implicit;
    
    for(int j=0; j<NUM_ACT_SET; ++j) {
        for(int k=0; k<3; ++k) grad_u[j][k] = grad_u_total[j*3 + k];
    }

    return 0;
}

// Batched backward pass
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
    // 1. Reconstruct Params & Jacobians (Single pass)
    // Reuse params reconstruction code...
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
    for (int j = 0; j < NUM_ACT_SET; ++j) for (int i = 0; i < 3; ++i) ActuationCurrents[j][i] = fwd_result.u[j][i];
    
    double ActInertia[NUM_ACT_SET][9];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        double mass = params.CathParams->ActMass[j];
        double r_outer = params.CathParams->OuterRadius[0];
        double r_inner = params.CathParams->InnerRadius[0];
        double seg_length = params.CathParams->SegLengths[2*j + 1];
        double r_sum_sq = r_outer * r_outer + r_inner * r_inner;
        double I_zz = 0.5 * mass * r_sum_sq;
        double I_xx = 0.25 * mass * r_sum_sq + (1.0/12.0) * mass * seg_length * seg_length;
        ActInertia[j][0] = I_xx;  ActInertia[j][4] = I_xx;  ActInertia[j][8] = I_zz;
        ActInertia[j][1]=ActInertia[j][2]=ActInertia[j][3]=ActInertia[j][5]=ActInertia[j][6]=ActInertia[j][7]=0.0;
    }
    double damping[NUM_ACT_SET][6];
    for (int j = 0; j < NUM_ACT_SET; ++j) for (int i = 0; i < 6; ++i) damping[j][i] = params.CathParams->ActDamping[j][i];

    CRMShootingMethodParams shooting_params = CRMDYNConstructShootingMethodParamSet(
        *params.CathParams, *params.CathConfig,
        fwd_result.L_inserted, ActuationCurrents,
        params.ContactMode,
        const_cast<double*>(params.TipConstraintPoint), const_cast<double*>(params.TipForce),
        params.IntegrationStepSize, ActInertia,
        v_L_pre, w_L_pre, p_pre, R_pre,
        damping, fwd_result.dt
    );

    Eigen::MatrixXd J_xf_y, J_xf_x, J_xcoil_y, J_xcoil_x;
    double dummy_xN[NUM_STATES], dummy_xcoil[NUM_ACT_SET][NUM_COIL_STATES];
    
    DYNSolverIVP_JacobiansFullstate(
        shooting_params, fwd_result.u0, fwd_result.mL, fwd_result.nL, fwd_result.tau, fwd_result.ftip,
        dummy_xN, dummy_xcoil, J_xf_y, J_xf_x, J_xcoil_y, J_xcoil_x
    );

    Eigen::MatrixXd J_yy, J_yu, J_yx;
    compute_bvp_jacobians_full_analytic(
        fwd_result.mL, fwd_result.nL, fwd_result.u,
        params, fwd_result.xf, fwd_result.L_inserted, fwd_result.dt,
        fwd_result.x_coil,
        J_yy, J_yu, J_yx
    );

    // Pre-factorize
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_solver(J_yy.transpose());
    if (lu_rank) *lu_rank = qr_solver.rank();

    // 2. Loop over RHS
    for (int rhs = 0; rhs < num_rhs; ++rhs) {
        Eigen::VectorXd v_xf_next = Eigen::VectorXd::Zero(NUM_STATES);
        for(int i=0; i<3; ++i) v_xf_next[i] = grad_tip_p_batch[rhs*3 + i];

        Eigen::VectorXd v_ivp_y = J_xf_y.transpose() * v_xf_next;
        
        int dim_y_bvp = NUM_ACT_SET * 6;
        Eigen::VectorXd v_y = Eigen::VectorXd::Zero(dim_y_bvp);
        for(int j=0; j<NUM_ACT_SET; ++j) {
            for(int k=0; k<3; ++k) v_y[j*6 + k] = v_ivp_y[j*3 + k];
            for(int k=0; k<3; ++k) v_y[j*6 + 3 + k] = v_ivp_y[NUM_ACT_SET*3 + j*3 + k];
        }

        Eigen::VectorXd lambda = qr_solver.solve(v_y);

        // Gradients
        Eigen::VectorXd grad_x_explicit = J_xf_x.transpose() * v_xf_next;
        Eigen::VectorXd grad_x_implicit = -J_yx.transpose() * lambda;
        Eigen::VectorXd grad_x_total = grad_x_explicit + grad_x_implicit;

        for(int j=0; j<NUM_ACT_SET; ++j) {
            for(int k=0; k<18; ++k) grad_x_coil_batch[rhs*NUM_ACT_SET*18 + j*18 + k] = grad_x_total[j*18 + k];
        }
        for(int k=0; k<NUM_STATES; ++k) grad_xf_batch[rhs*NUM_STATES + k] = grad_x_total[NUM_ACT_SET*18 + k];

        // u gradient
        Eigen::VectorXd grad_u_direct = Eigen::VectorXd::Zero(NUM_ACT_SET * 3);
        for(int j=0; j<NUM_ACT_SET; ++j) {
            const double* B = shooting_params.B0;
            Eigen::Matrix3d dM_du;
            for(int row=0; row<3; ++row) for(int col=0; col<3; ++col) {
                dM_du(row, col) = shooting_params.CoilAlignmentTurnAreaMatrix[j][row*3 + col];
            }
            Eigen::Vector3d v_tau_j;
            for(int k=0; k<3; ++k) v_tau_j[k] = v_ivp_y[NUM_ACT_SET*6 + j*3 + k];
            
            for(int i=0; i<3; ++i) {
                Eigen::Vector3d dM_du_i = dM_du.col(i);
                Eigen::Vector3d dtau_du_i = dM_du_i.cross(Eigen::Vector3d(B[0], B[1], B[2]));
                grad_u_direct[j*3 + i] += dtau_du_i.dot(v_tau_j);
            }
        }
        
        Eigen::VectorXd grad_u_implicit = -J_yu.transpose() * lambda;
        Eigen::VectorXd grad_u_total = grad_u_direct + grad_u_implicit;

        for(int j=0; j<NUM_ACT_SET; ++j) {
            for(int k=0; k<3; ++k) grad_u_batch[rhs*NUM_ACT_SET*3 + j*3 + k] = grad_u_total[j*3 + k];
        }
    }

    if (rel_residual) *rel_residual = 0.0; // Approximation
    return 0;
}

// Linearization: Compute A, B matrices
int true_legacy_linearize_implicit(
    const TrueLegacyStepResult& fwd_result,
    const CRMForwardKinematicsData& params,
    double L_inserted,
    Eigen::MatrixXd& A_out,
    Eigen::MatrixXd& B_out,
    int* qr_rank,
    double* rel_residual
) {
    // Reconstruct params (reuse code from backward)
    // ... (Code duplication)
    double v_L_pre[NUM_ACT_SET][3]; double w_L_pre[NUM_ACT_SET][3]; double p_pre[NUM_ACT_SET][3]; double R_pre[NUM_ACT_SET][9];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            v_L_pre[j][i] = fwd_result.x_coil[j][i];
            w_L_pre[j][i] = fwd_result.x_coil[j][3 + i];
            p_pre[j][i] = fwd_result.x_coil[j][6 + i];
        }
        for (int i = 0; i < 9; ++i) R_pre[j][i] = fwd_result.x_coil[j][9 + i];
    }
    double ActuationCurrents[NUM_ACT_SET][3];
    for (int j = 0; j < NUM_ACT_SET; ++j) for (int i = 0; i < 3; ++i) ActuationCurrents[j][i] = fwd_result.u[j][i];
    
    double ActInertia[NUM_ACT_SET][9];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        double mass = params.CathParams->ActMass[j];
        double r_outer = params.CathParams->OuterRadius[0]; double r_inner = params.CathParams->InnerRadius[0]; double seg_length = params.CathParams->SegLengths[2*j + 1];
        double r_sum_sq = r_outer * r_outer + r_inner * r_inner;
        double I_zz = 0.5 * mass * r_sum_sq; double I_xx = 0.25 * mass * r_sum_sq + (1.0/12.0) * mass * seg_length * seg_length;
        ActInertia[j][0] = I_xx;  ActInertia[j][4] = I_xx;  ActInertia[j][8] = I_zz;
        ActInertia[j][1]=ActInertia[j][2]=ActInertia[j][3]=ActInertia[j][5]=ActInertia[j][6]=ActInertia[j][7]=0.0;
    }
    double damping[NUM_ACT_SET][6];
    for (int j = 0; j < NUM_ACT_SET; ++j) for (int i = 0; i < 6; ++i) damping[j][i] = params.CathParams->ActDamping[j][i];

    CRMShootingMethodParams shooting_params = CRMDYNConstructShootingMethodParamSet(
        *params.CathParams, *params.CathConfig,
        L_inserted, ActuationCurrents,
        params.ContactMode,
        const_cast<double*>(params.TipConstraintPoint), const_cast<double*>(params.TipForce),
        params.IntegrationStepSize, ActInertia,
        v_L_pre, w_L_pre, p_pre, R_pre,
        damping, fwd_result.dt
    );

    // Compute Jacobians
    Eigen::MatrixXd J_xf_y, J_xf_x, J_xcoil_y, J_xcoil_x;
    double dummy_xN[NUM_STATES], dummy_xcoil[NUM_ACT_SET][NUM_COIL_STATES];
    
    DYNSolverIVP_JacobiansFullstate(
        shooting_params, fwd_result.u0, fwd_result.mL, fwd_result.nL, fwd_result.tau, fwd_result.ftip,
        dummy_xN, dummy_xcoil, J_xf_y, J_xf_x, J_xcoil_y, J_xcoil_x
    );

    Eigen::MatrixXd J_yy, J_yu, J_yx;
    compute_bvp_jacobians_full_analytic(
        fwd_result.mL, fwd_result.nL, fwd_result.u,
        params, fwd_result.xf, L_inserted, fwd_result.dt,
        fwd_result.x_coil,
        J_yy, J_yu, J_yx
    );

    // Compute Sensitivity: dydx = -J_yy^-1 * J_yx
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_solver(J_yy);
    Eigen::MatrixXd dydx = -qr_solver.solve(J_yx);
    Eigen::MatrixXd dydu = -qr_solver.solve(J_yu);

    // A = G_x + G_y * dydx
    // B = G_u + G_y * dydu
    // Where G_x = [J_xcoil_x; J_xf_x], G_y = [J_xcoil_y; J_xf_y] subset (mL, nL cols)
    
    // Construct G_x, G_y, G_u
    // G matrices map to [x_coil_next; xf_next]
    
    int dim_state = NUM_ACT_SET * 18 + NUM_STATES;
    int dim_input = NUM_ACT_SET * 3;
    int dim_y_bvp = NUM_ACT_SET * 6;

    Eigen::MatrixXd G_x = Eigen::MatrixXd::Zero(dim_state, dim_state);
    Eigen::MatrixXd G_y = Eigen::MatrixXd::Zero(dim_state, dim_y_bvp);
    Eigen::MatrixXd G_u = Eigen::MatrixXd::Zero(dim_state, dim_input);

    // Fill G_x
    G_x.topRows(NUM_ACT_SET * 18) = J_xcoil_x;
    G_x.bottomRows(NUM_STATES) = J_xf_x;

    // Fill G_y (columns corresponding to mL, nL)
    // J_xcoil_y cols: mL, nL, tau, ...
    for(int j=0; j<NUM_ACT_SET; ++j) {
        // mL
        G_y.col(j*6).topRows(NUM_ACT_SET * 18) = J_xcoil_y.col(j*3);
        G_y.col(j*6).bottomRows(NUM_STATES) = J_xf_y.col(j*3);
        G_y.col(j*6+1).topRows(NUM_ACT_SET * 18) = J_xcoil_y.col(j*3+1);
        G_y.col(j*6+1).bottomRows(NUM_STATES) = J_xf_y.col(j*3+1);
        G_y.col(j*6+2).topRows(NUM_ACT_SET * 18) = J_xcoil_y.col(j*3+2);
        G_y.col(j*6+2).bottomRows(NUM_STATES) = J_xf_y.col(j*3+2);
        
        // nL
        G_y.col(j*6+3).topRows(NUM_ACT_SET * 18) = J_xcoil_y.col(NUM_ACT_SET*3 + j*3);
        G_y.col(j*6+3).bottomRows(NUM_STATES) = J_xf_y.col(NUM_ACT_SET*3 + j*3);
        G_y.col(j*6+4).topRows(NUM_ACT_SET * 18) = J_xcoil_y.col(NUM_ACT_SET*3 + j*3+1);
        G_y.col(j*6+4).bottomRows(NUM_STATES) = J_xf_y.col(NUM_ACT_SET*3 + j*3+1);
        G_y.col(j*6+5).topRows(NUM_ACT_SET * 18) = J_xcoil_y.col(NUM_ACT_SET*3 + j*3+2);
        G_y.col(j*6+5).bottomRows(NUM_STATES) = J_xf_y.col(NUM_ACT_SET*3 + j*3+2);
    }

    // Fill G_u (Direct contribution from tau)
    // dTau/du logic
    for(int j=0; j<NUM_ACT_SET; ++j) {
        const double* B = shooting_params.B0;
        Eigen::Matrix3d dM_du;
        for(int row=0; row<3; ++row) for(int col=0; col<3; ++col) {
            dM_du(row, col) = shooting_params.CoilAlignmentTurnAreaMatrix[j][row*3 + col];
        }
        
        // For each u_k
        for(int k=0; k<3; ++k) {
            Eigen::Vector3d dtau_du_k = dM_du.col(k).cross(Eigen::Vector3d(B[0], B[1], B[2]));
            
            // Add dtau contribution to G_u
            // G_u_col = J_..._y(tau_cols) * dtau_du_k
            
            for(int dim=0; dim<3; ++dim) { // components of tau
                G_u.col(j*3 + k).topRows(NUM_ACT_SET * 18) += J_xcoil_y.col(NUM_ACT_SET*6 + j*3 + dim) * dtau_du_k[dim];
                G_u.col(j*3 + k).bottomRows(NUM_STATES) += J_xf_y.col(NUM_ACT_SET*6 + j*3 + dim) * dtau_du_k[dim];
            }
        }
    }

    A_out = G_x + G_y * dydx;
    B_out = G_u + G_y * dydu;

    if(qr_rank) *qr_rank = qr_solver.rank();
    if(rel_residual) *rel_residual = 0.0;

    return 0;
}

} // namespace CRMCatheterModel