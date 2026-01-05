#include "CRM_BVPJacobian.hpp"
#include "CRM_MatrixOperations.hpp"
#include "CRM_MatrixOperations_Templates.hpp"
#include "CoilDynamics_Defs_Templates.hpp"
#include "CoilDynamics_Defs_Templates2.hpp"
#include "CRM_IVPJacobian.hpp"
#include <cstring>

namespace CRMCatheterModel {

// Evaluate BVP residual: r(mL, nL; x_coil, xf, u, dt) = 0
void compute_bvp_residual(
    const double mL[NUM_ACT_SET][3],
    const double nL[NUM_ACT_SET][3],
    const CRMShootingMethodParams& params,
    const double xf[NUM_STATES],
    double residual[NUM_ACT_SET * 6]
) {
    // Pack mL and nL into the format expected by DYNNLEquation (unscaled)
    double in_x[NUM_ACT_SET * 6];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            in_x[j * 6 + i] = mL[j][i] / IVALUE_SCALE_M;
            in_x[j * 6 + 3 + i] = nL[j][i] / IVALUE_SCALE_N;
        }
    }

    // Create DYNNLEqnParams
    double x_0[NUM_STATES];
    for (int i = 0; i < NUM_STATES; i++) {
        if (i < 3) {
            x_0[i] = params.p0[i];
        } else if (i < 12) {
            x_0[i] = params.R0[i - 3];
        } else {
            x_0[i] = 0.0;
        }
    }

    bool FinalValueOnly = true;
    DYNNLEqnParams eqn_params(params.no_flex_seg, params.no_rigid_seg, params.no_act_set,
                               params.no_locmarkers, params.no_fcum_steps);

    double mL_dummy[NUM_ACT_SET][3];
    double nL_dummy[NUM_ACT_SET][3];
    std::memcpy(mL_dummy, mL, sizeof(mL_dummy));
    std::memcpy(nL_dummy, nL, sizeof(nL_dummy));

    CRMDYNSolverIVP_Prep(
        params.no_flex_seg, params.no_rigid_seg, params.no_act_set,
        params.no_locmarkers, params.no_fcum_steps,
        x_0, params.IntegrationStepSize,
        params.Li, params.dlambdainv, const_cast<double*>(params.rho),
        const_cast<CatheterSegmentType*>(params.SegmentTypes),
        const_cast<double*>(params.SegEndLambdas), const_cast<double*>(params.LocMarkerLambdas),
        const_cast<double(*)[9]>(params.K), const_cast<double(*)[9]>(params.Kinv),
        const_cast<double(*)[3]>(params.ustar),
        const_cast<double(*)[3]>(params.MagMoment), const_cast<double(*)[3]>(params.fcumlambda),
        const_cast<double(*)[9]>(params.CoilAlignmentTurnAreaMatrix),
        const_cast<double*>(params.B0), const_cast<double*>(params.g),
        const_cast<double*>(params.ActMass), const_cast<double(*)[9]>(params.actInertia),
        const_cast<double(*)[6]>(params.damping), params.DELTA_T,
        const_cast<double(*)[3]>(params.v_L_pre), const_cast<double(*)[3]>(params.w_L_pre),
        const_cast<double(*)[3]>(params.p_pre), const_cast<double(*)[9]>(params.R_pre),
        mL_dummy, nL_dummy, FinalValueOnly, eqn_params
    );

    eqn_params.ContactMode = params.ContactMode;
    mCopy_AB<3>(params.TipConstraintPoint, eqn_params.TipConstraintPoint);
    mCopy_AB<3>(params.TipForce, eqn_params.TipForce);

    for (int i = 0; i < NUM_STATES; ++i) {
        eqn_params.xf[i] = xf[i];
    }

    double out_u0[3];
    double out_tau[NUM_ACT_SET * 3];
    DYNNLEquation(in_x, residual, eqn_params, out_u0, out_tau);
}

// Compute BVP Jacobians using strictly analytic differentiation
// NO finite differences - uses analytic IVP Jacobians and chain rule
//
// IMPLEMENTATION:
// The BVP residual r(mL, nL, u) is computed by forward integration (IVP).
// We use the analytic IVP Jacobians from CRMSolverIVPJacobian and chain rule
// to obtain BVP Jacobians without any numerical differentiation.
void compute_bvp_jacobians_fmad(
    const double mL[NUM_ACT_SET][3],
    const double nL[NUM_ACT_SET][3],
    const double u[NUM_ACT_SET][3],
    const CRMForwardKinematicsData& params,
    const double xf[NUM_STATES],
    double L_inserted,
    double dt,
    const double x_coil[NUM_ACT_SET][18],
    Eigen::MatrixXd& J_yy,
    Eigen::MatrixXd& J_yu
) {
    const int dim_y = NUM_ACT_SET * 6;  // [mL; nL]
    const int dim_u = NUM_ACT_SET * 3;

    J_yy.resize(dim_y, dim_y);
    J_yu.resize(dim_y, dim_u);

    // Construct parameters for analytic Jacobian computation
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

    double ActuationCurrents[NUM_ACT_SET][3];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            ActuationCurrents[j][i] = u[j][i];
        }
    }

    // Construct actuator inertia using hollow cylinder formula (MUST match forward path exactly)
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

    // Load damping coefficients from catheter parameters (MUST match forward path)
    double damping[NUM_ACT_SET][6];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 6; ++i) {
            damping[j][i] = params.CathParams->ActDamping[j][i];
        }
    }

    // Construct shooting params
    CRMShootingMethodParams shooting_params = CRMDYNConstructShootingMethodParamSet(
        *params.CathParams, *params.CathConfig,
        L_inserted, ActuationCurrents,
        params.ContactMode,
        const_cast<double*>(params.TipConstraintPoint), const_cast<double*>(params.TipForce),
        params.IntegrationStepSize, ActInertia,
        v_L_pre, w_L_pre, p_pre, R_pre,
        damping, dt
    );

    // Setup parameters for IVP Jacobian computation
    double x_0[NUM_STATES];
    for (int i = 0; i < NUM_STATES; i++) {
        if (i < 3) {
            x_0[i] = shooting_params.p0[i];
        } else if (i < 12) {
            x_0[i] = shooting_params.R0[i - 3];
        } else {
            x_0[i] = 0.0;
        }
    }

    bool FinalValueOnly = true;
    DYNNLEqnParams eqn_params(shooting_params.no_flex_seg, shooting_params.no_rigid_seg,
                              shooting_params.no_act_set, shooting_params.no_locmarkers,
                              shooting_params.no_fcum_steps);

    double mL_dummy[NUM_ACT_SET][3], nL_dummy[NUM_ACT_SET][3];
    std::memcpy(mL_dummy, mL, sizeof(mL_dummy));
    std::memcpy(nL_dummy, nL, sizeof(nL_dummy));

    CRMDYNSolverIVP_Prep(
        shooting_params.no_flex_seg, shooting_params.no_rigid_seg, shooting_params.no_act_set,
        shooting_params.no_locmarkers, shooting_params.no_fcum_steps,
        x_0, shooting_params.IntegrationStepSize,
        shooting_params.Li, shooting_params.dlambdainv, const_cast<double*>(shooting_params.rho),
        const_cast<CatheterSegmentType*>(shooting_params.SegmentTypes),
        const_cast<double*>(shooting_params.SegEndLambdas),
        const_cast<double*>(shooting_params.LocMarkerLambdas),
        const_cast<double(*)[9]>(shooting_params.K), const_cast<double(*)[9]>(shooting_params.Kinv),
        const_cast<double(*)[3]>(shooting_params.ustar),
        const_cast<double(*)[3]>(shooting_params.MagMoment),
        const_cast<double(*)[3]>(shooting_params.fcumlambda),
        const_cast<double(*)[9]>(shooting_params.CoilAlignmentTurnAreaMatrix),
        const_cast<double*>(shooting_params.B0), const_cast<double*>(shooting_params.g),
        const_cast<double*>(shooting_params.ActMass), const_cast<double(*)[9]>(shooting_params.actInertia),
        const_cast<double(*)[6]>(shooting_params.damping), shooting_params.DELTA_T,
        const_cast<double(*)[3]>(shooting_params.v_L_pre), const_cast<double(*)[3]>(shooting_params.w_L_pre),
        const_cast<double(*)[3]>(shooting_params.p_pre), const_cast<double(*)[9]>(shooting_params.R_pre),
        mL_dummy, nL_dummy, FinalValueOnly, eqn_params
    );

    eqn_params.ContactMode = shooting_params.ContactMode;
    mCopy_AB<3>(shooting_params.TipConstraintPoint, eqn_params.TipConstraintPoint);
    mCopy_AB<3>(shooting_params.TipForce, eqn_params.TipForce);
    for (int i = 0; i < NUM_STATES; ++i) {
        eqn_params.xf[i] = xf[i];
    }

    // Compute BVP Jacobians using forward-mode AD with Dual numbers (NO FD)
    //
    // The shooting method BVP has structure:
    // residual r(mL, nL) = [moment_balance; force_balance] at each interface
    //
    // For a converged BVP solution, r = 0. The Jacobian ∂r/∂(mL, nL) describes
    // how perturbations in boundary values affect the residual.
    //
    // Key insight: The shooting method residual has the form:
    // r_m[j] = m_computed[j] - mL[j]  (moment balance)
    // r_n[j] = n_computed[j] - nL[j]  (force balance)
    //
    // where m_computed and n_computed come from integrating the IVP.
    //
    // The Jacobian structure is:
    // ∂r_m/∂mL includes both the direct term -I and indirect IVP coupling
    // ∂r_n/∂nL includes both the direct term -I and indirect IVP coupling
    //
    // For the converged solution, the dominant term is the direct dependence.
    // We use forward-mode AD to capture this structure with Dual numbers.

    // Compute J_yy = ∂r/∂y where y = [mL; nL] using forward-mode AD (NO FD)
    // CRITICAL: Must capture mL↔nL coupling for correct implicit gradients
    //
    // Sprint S12: Use DYNNLEquation_YY_T with Dual numbers to compute exact J_yy
    // by seeding each component of y = [mL; nL] with derivative = 1
    J_yy.resize(dim_y, dim_y);
    J_yy.setZero();

    // Pack base mL and nL into input format (values only, no derivatives yet)
    double in_x_base[NUM_ACT_SET * 6];
    for (int j_pack = 0; j_pack < NUM_ACT_SET; ++j_pack) {
        for (int i_pack = 0; i_pack < 3; ++i_pack) {
            in_x_base[i_pack + j_pack*6] = mL[j_pack][i_pack];
            in_x_base[i_pack + j_pack*6 + 3] = nL[j_pack][i_pack];
        }
    }

    // Compute base MagMoment for each actuator (stays double, no derivatives)
    double MagMoment[NUM_ACT_SET][3];
    for (int j_mag = 0; j_mag < NUM_ACT_SET; ++j_mag) {
        for (int i_mag = 0; i_mag < 3; ++i_mag) {
            MagMoment[j_mag][i_mag] = shooting_params.MagMoment[j_mag][i_mag];
        }
    }

    // Compute muhat from MagMoment (for value path, derivatives handled separately)
    double muhat_double[NUM_ACT_SET][9];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        wHat(MagMoment[j], muhat_double[j]);
    }

    // Compute J_yy using forward-mode AD with Dual numbers
    // For each column j_y, seed y[j_y] with derivative = 1 and extract ∂r/∂y[j_y]
    for (int j_y = 0; j_y < dim_y; ++j_y) {
        // Create Dual-seeded input: in_x[i] = Dual(value, deriv)
        // where deriv = 1 if i == j_y, else 0
        Dual in_x_dual[NUM_ACT_SET * 6];
        for (int i = 0; i < dim_y; ++i) {
            double deriv = (i == j_y) ? 1.0 : 0.0;
            in_x_dual[i] = Dual(in_x_base[i], deriv);
        }

        // Evaluate residual with seeded y using templated DYNNLEquation_YY_T
        Dual out_y_dual[NUM_ACT_SET * 6];
        Dual out_u0_dual[3];
        Dual out_tau_dual[NUM_ACT_SET * 3];

        DYNNLEquation_YY_T<Dual>(in_x_dual, out_y_dual, eqn_params, muhat_double,
                                  out_u0_dual, out_tau_dual);

        // Extract derivatives ∂r/∂y[j_y] from Dual numbers (column j_y of J_yy)
        for (int row = 0; row < dim_y; ++row) {
            J_yy(row, j_y) = out_y_dual[row].deriv;
        }
    }

    // DEBUG: Print J_yy statistics
    std::cerr << "\n[J_yy Diagnostics]" << std::endl;
    std::cerr << "  J_yy shape: " << J_yy.rows() << " x " << J_yy.cols() << std::endl;
    std::cerr << "  J_yy norm: " << J_yy.norm() << std::endl;
    std::cerr << "  J_yy max abs: " << J_yy.cwiseAbs().maxCoeff() << std::endl;

    // Check diagonal dominance
    double diag_norm = 0.0;
    double offdiag_norm = 0.0;
    for (int i = 0; i < dim_y; ++i) {
        diag_norm += J_yy(i, i) * J_yy(i, i);
        for (int j = 0; j < dim_y; ++j) {
            if (i != j) {
                offdiag_norm += J_yy(i, j) * J_yy(i, j);
            }
        }
    }
    std::cerr << "  J_yy diagonal norm: " << std::sqrt(diag_norm) << std::endl;
    std::cerr << "  J_yy off-diagonal norm: " << std::sqrt(offdiag_norm) << std::endl;

    // Check mL-nL coupling
    if (NUM_ACT_SET > 0) {
        double mL_nL_coupling = 0.0;
        for (int i = 0; i < 3; ++i) {
            for (int j = 3; j < 6; ++j) {
                mL_nL_coupling += std::abs(J_yy(i, j)) + std::abs(J_yy(j, i));
            }
        }
        std::cerr << "  J_yy mL-nL coupling (sum |J(mL,nL)| + |J(nL,mL)|): " << mL_nL_coupling << std::endl;
    }
    std::cerr.flush();

    // Compute J_yu using forward-mode automatic differentiation (NO FD, NO heuristics)
    //
    // Sprint S8: Exact J_yu via correct physical seeding of the u → MagMoment → muhat chain.
    // Physical mapping: MagMoment[j] = CoilAlignmentTurnAreaMatrix[j] * u[j]
    // For each control input u[j][i], we seed MagMoment[j] with ∂MagMoment/∂u[j][i],
    // propagate through wHat_T to get seeded muhat, then evaluate the residual with
    // templated DYNNLEquation_T to extract ∂r/∂u[j][i] from the Dual derivatives.

    J_yu.setZero();

    // Note: in_x_base and MagMoment are already declared and initialized above for J_yy computation

    // For each control input u[j][i], compute ∂r/∂u[j][i] via forward-mode AD
    // Physical mapping: MagMoment[j] = CoilAlignmentTurnAreaMatrix[j] * u[j]
    // Therefore: ∂MagMoment[j]/∂u[j][i] = CoilAlignmentTurnAreaMatrix[j][:, i]

    // DEBUG: Print CoilAlignmentTurnAreaMatrix for first actuator
    if (NUM_ACT_SET > 0) {
        std::cerr << "DEBUG J_yu: CoilAlignmentTurnAreaMatrix[0] = [";
        for (int i = 0; i < 9; ++i) {
            std::cerr << shooting_params.CoilAlignmentTurnAreaMatrix[0][i] << (i < 8 ? ", " : "");
        }
        std::cerr << "]" << std::endl;
        std::cerr << "DEBUG J_yu: MagMoment[0] = [" << MagMoment[0][0] << ", " << MagMoment[0][1] << ", " << MagMoment[0][2] << "]" << std::endl;
        std::cerr.flush();
    }

    for (int j_ctrl = 0; j_ctrl < NUM_ACT_SET; ++j_ctrl) {
        for (int i_ctrl = 0; i_ctrl < 3; ++i_ctrl) {
            // Seed MagMoment[j_ctrl] with exact derivative ∂MagMoment/∂u[j_ctrl][i_ctrl]
            // = CoilAlignmentTurnAreaMatrix[j_ctrl][:, i_ctrl]
            Dual MagMoment_seeded[NUM_ACT_SET][3];
            for (int k = 0; k < NUM_ACT_SET; ++k) {
                for (int m = 0; m < 3; ++m) {
                    if (k == j_ctrl) {
                        // Extract ∂MagMoment[k][m]/∂u[k][i_ctrl] from CoilAlignmentTurnAreaMatrix
                        // Matrix is stored in row-major order: [row][col] = [row*3 + col]
                        double deriv = shooting_params.CoilAlignmentTurnAreaMatrix[k][m * 3 + i_ctrl];
                        MagMoment_seeded[k][m] = Dual(MagMoment[k][m], deriv);

                        // DEBUG: Print seeding for first column
                        if (j_ctrl == 0 && i_ctrl == 0 && m == 0) {
                            std::cerr << "DEBUG J_yu: Seeding MagMoment[" << k << "][" << m << "] with deriv = " << deriv << std::endl;
                            std::cerr.flush();
                        }
                    } else {
                        // Other actuators: no derivative w.r.t. u[j_ctrl][i_ctrl]
                        MagMoment_seeded[k][m] = Dual(MagMoment[k][m], 0.0);
                    }
                }
            }

            // Propagate derivatives through wHat: muhat = wHat(MagMoment)
            // wHat_T is templated and handles Dual number propagation automatically
            Dual muhat_seeded[NUM_ACT_SET][9];
            for (int k = 0; k < NUM_ACT_SET; ++k) {
                wHat_T<Dual>(MagMoment_seeded[k], muhat_seeded[k]);
            }

            // Evaluate BVP residual with seeded muhat using templated DYNNLEquation_T
            Dual out_y_dual[NUM_ACT_SET * 6];
            Dual out_u0_dual[3];
            Dual out_tau_dual[NUM_ACT_SET * 3];

            DYNNLEquation_T<Dual>(in_x_base, out_y_dual, eqn_params, muhat_seeded, out_u0_dual, out_tau_dual);

            // Extract derivatives ∂r/∂u[j][i] from Dual numbers
            int col = j_ctrl * 3 + i_ctrl;
            for (int row = 0; row < dim_y; ++row) {
                J_yu(row, col) = out_y_dual[row].deriv;
            }

            // DEBUG: Print extracted gradients for first column
            if (j_ctrl == 0 && i_ctrl == 0) {
                std::cerr << "DEBUG J_yu: out_y_dual[0].deriv = " << out_y_dual[0].deriv << std::endl;
                std::cerr << "DEBUG J_yu: J_yu(0,0) = " << J_yu(0, 0) << std::endl;
                std::cerr.flush();
            }
        }
    }

    // DEBUG: Print final J_yu statistics
    std::cerr << "DEBUG J_yu final: norm = " << J_yu.norm() << ", max = " << J_yu.cwiseAbs().maxCoeff() << std::endl;
    std::cerr << "DEBUG J_yu final: J_yu(0,0) = " << J_yu(0, 0) << std::endl;
    std::cerr.flush();
}


// A3.5: Compute full BVP Jacobians including state dependencies
// Computes J_yy, J_yu, and J_yx = ∂r/∂x_t analytically using forward-mode AD (NO FD, NO assumptions)
//
// The BVP residual r(y; x_t, u) where y = [mL; nL] and x_t = [x_coil; xf]
// depends on the state through:
// 1. x_coil affects initial conditions (p_pre, R_pre, v_pre, w_pre) for IVP
// 2. xf affects the shooting method target state
//
// Implementation: Use forward-mode AD with Dual numbers to compute J_yx analytically.
// This captures the true sensitivity of the BVP residual to state perturbations.
void compute_bvp_jacobians_full_analytic(
    const double mL[NUM_ACT_SET][3],
    const double nL[NUM_ACT_SET][3],
    const double u[NUM_ACT_SET][3],
    const CRMForwardKinematicsData& params,
    const double xf[NUM_STATES],
    double L_inserted,
    double dt,
    const double x_coil[NUM_ACT_SET][18],
    Eigen::MatrixXd& J_yy,
    Eigen::MatrixXd& J_yu,
    Eigen::MatrixXd& J_yx
) {
    const int dim_y = NUM_ACT_SET * 6;  // [mL; nL]
    const int dim_x = NUM_ACT_SET * 18 + NUM_STATES;  // [x_coil; xf]

    // First compute J_yy and J_yu using existing analytic function
    compute_bvp_jacobians_fmad(mL, nL, u, params, xf, L_inserted, dt, x_coil, J_yy, J_yu);

    // Initialize J_yx
    J_yx.resize(dim_y, dim_x);
    J_yx.setZero();

    // ========================================================================
    // STEP 3: Compute J_yx using exact forward-mode AD (NO heuristics, NO FD)
    // ========================================================================
    //
    // Strategy: Seed each state component x[j] with Dual(value, 1) and evaluate
    // the BVP residual using DYNNLEquation_XT with templated params.
    //
    // State vector x = [x_coil; xf] where:
    //   x_coil[j] = [v_L, w_L, p_L, R_L]  (18 components per actuator)
    //   xf = tip state (NUM_STATES components)
    //
    // For each state component, we:
    //   1. Create DYNNLEqnParams_T<Dual> with that component seeded
    //   2. Call DYNNLEquation_XT<Dual> to evaluate residual
    //   3. Extract .deriv from output to get J_yx column

    // Setup shooting params (same as in compute_bvp_jacobians_fmad)
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

    double ActuationCurrents[NUM_ACT_SET][3];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            ActuationCurrents[j][i] = u[j][i];
        }
    }

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
        damping, dt
    );

    double x_0[NUM_STATES];
    for (int i = 0; i < NUM_STATES; i++) {
        if (i < 3) {
            x_0[i] = shooting_params.p0[i];
        } else if (i < 12) {
            x_0[i] = shooting_params.R0[i - 3];
        } else {
            x_0[i] = 0.0;
        }
    }

    bool FinalValueOnly = true;
    DYNNLEqnParams eqn_params(shooting_params.no_flex_seg, shooting_params.no_rigid_seg,
                              shooting_params.no_act_set, shooting_params.no_locmarkers,
                              shooting_params.no_fcum_steps);

    double mL_dummy[NUM_ACT_SET][3], nL_dummy[NUM_ACT_SET][3];
    std::memcpy(mL_dummy, mL, sizeof(mL_dummy));
    std::memcpy(nL_dummy, nL, sizeof(nL_dummy));

    CRMDYNSolverIVP_Prep(
        shooting_params.no_flex_seg, shooting_params.no_rigid_seg, shooting_params.no_act_set,
        shooting_params.no_locmarkers, shooting_params.no_fcum_steps,
        x_0, shooting_params.IntegrationStepSize,
        shooting_params.Li, shooting_params.dlambdainv, const_cast<double*>(shooting_params.rho),
        const_cast<CatheterSegmentType*>(shooting_params.SegmentTypes),
        const_cast<double*>(shooting_params.SegEndLambdas),
        const_cast<double*>(shooting_params.LocMarkerLambdas),
        const_cast<double(*)[9]>(shooting_params.K), const_cast<double(*)[9]>(shooting_params.Kinv),
        const_cast<double(*)[3]>(shooting_params.ustar),
        const_cast<double(*)[3]>(shooting_params.MagMoment),
        const_cast<double(*)[3]>(shooting_params.fcumlambda),
        const_cast<double(*)[9]>(shooting_params.CoilAlignmentTurnAreaMatrix),
        const_cast<double*>(shooting_params.B0), const_cast<double*>(shooting_params.g),
        const_cast<double*>(shooting_params.ActMass), const_cast<double(*)[9]>(shooting_params.actInertia),
        const_cast<double(*)[6]>(shooting_params.damping), shooting_params.DELTA_T,
        const_cast<double(*)[3]>(shooting_params.v_L_pre), const_cast<double(*)[3]>(shooting_params.w_L_pre),
        const_cast<double(*)[3]>(shooting_params.p_pre), const_cast<double(*)[9]>(shooting_params.R_pre),
        mL_dummy, nL_dummy, FinalValueOnly, eqn_params
    );

    eqn_params.ContactMode = shooting_params.ContactMode;
    mCopy_AB<3>(shooting_params.TipConstraintPoint, eqn_params.TipConstraintPoint);
    mCopy_AB<3>(shooting_params.TipForce, eqn_params.TipForce);
    for (int i = 0; i < NUM_STATES; ++i) {
        eqn_params.xf[i] = xf[i];
    }

    // Pack input mL and nL (values only, no derivatives for J_yx)
    double in_x_base[NUM_ACT_SET * 6];
    for (int j_pack = 0; j_pack < NUM_ACT_SET; ++j_pack) {
        for (int i_pack = 0; i_pack < 3; ++i_pack) {
            in_x_base[i_pack + j_pack*6] = mL[j_pack][i_pack];
            in_x_base[i_pack + j_pack*6 + 3] = nL[j_pack][i_pack];
        }
    }

    // Compute base MagMoment (stays double for J_yx)
    double MagMoment[NUM_ACT_SET][3];
    for (int j_mag = 0; j_mag < NUM_ACT_SET; ++j_mag) {
        for (int i_mag = 0; i_mag < 3; ++i_mag) {
            MagMoment[j_mag][i_mag] = shooting_params.MagMoment[j_mag][i_mag];
        }
    }

    // Compute muhat from MagMoment (double, no derivatives for u-seeding in J_yx)
    double muhat_double[NUM_ACT_SET][9];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        wHat(MagMoment[j], muhat_double[j]);
    }

    // Loop over each state component in x = [x_coil; xf]
    for (int j_x = 0; j_x < dim_x; ++j_x) {
        // Create templated params with state[j_x] seeded with derivative = 1
        DYNNLEqnParams_T<Dual> params_dual(eqn_params);

        // Determine which state component to seed
        if (j_x < NUM_ACT_SET * 18) {
            // Seeding x_coil component
            int actuator_idx = j_x / 18;
            int coil_comp = j_x % 18;

            if (coil_comp < 3) {
                // v_L_pre component
                params_dual.v_L_pre[actuator_idx][coil_comp].deriv = 1.0;
            } else if (coil_comp < 6) {
                // w_L_pre component
                params_dual.w_L_pre[actuator_idx][coil_comp - 3].deriv = 1.0;
            } else if (coil_comp < 9) {
                // p_pre component
                params_dual.p_pre[actuator_idx][coil_comp - 6].deriv = 1.0;
            } else {
                // R_pre component
                params_dual.R_pre[actuator_idx][coil_comp - 9].deriv = 1.0;
            }
        } else {
            // Seeding xf component
            int xf_comp = j_x - NUM_ACT_SET * 18;
            params_dual.xf[xf_comp].deriv = 1.0;
        }

        // Evaluate residual with seeded params
        Dual out_y_dual[NUM_ACT_SET * 6];
        Dual out_u0_dual[3];
        Dual out_tau_dual[NUM_ACT_SET * 3];

        DYNNLEquation_XT<Dual>(in_x_base, out_y_dual, params_dual, muhat_double,
                               out_u0_dual, out_tau_dual);

        // Extract derivatives ∂r/∂x[j_x] from Dual numbers (column j_x of J_yx)
        for (int row = 0; row < dim_y; ++row) {
            J_yx(row, j_x) = out_y_dual[row].deriv;
        }
    }

    // Print J_yx diagnostics
    std::cerr << "\n[J_yx Diagnostics - Step 3]" << std::endl;
    std::cerr << "  J_yx shape: " << J_yx.rows() << " x " << J_yx.cols() << std::endl;
    std::cerr << "  J_yx norm: " << J_yx.norm() << std::endl;
    std::cerr << "  J_yx max abs: " << J_yx.cwiseAbs().maxCoeff() << std::endl;

    // Count non-zeros
    int nnz = 0;
    for (int i = 0; i < dim_y; ++i) {
        for (int j = 0; j < dim_x; ++j) {
            if (std::abs(J_yx(i, j)) > 1e-14) {
                ++nnz;
            }
        }
    }
    std::cerr << "  J_yx non-zeros (abs > 1e-14): " << nnz << std::endl;
    std::cerr.flush();
}


} // namespace CRMCatheterModel
