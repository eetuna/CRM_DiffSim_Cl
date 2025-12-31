#include "CRM_DiffEquilibrium.hpp"
#include "CRM_BVPIVP_APIDeclarations.hpp"
#include "CRM_FK_InternalAPI.hpp"
#include "CRM_IVPJacobian.hpp"

using namespace Eigen;

namespace CRMCatheterModel {

int equilibrium_forward(
    const double u[NUM_ACT_SET*3],
    double L_inserted,
    const CRMForwardKinematicsData& params,
    EquilibriumResult& out
) {
    // Prepare actuation currents
    double ActuationCurrents[NUM_ACT_SET][3];
    for (int i = 0; i < NUM_ACT_SET; i++) {
        for (int j = 0; j < 3; j++) {
            ActuationCurrents[i][j] = u[i*3 + j];
        }
    }

    auto& CathParams = *(params.CathParams);
    auto& CathConfig = *(params.CathConfig);

    // Copy const arrays to non-const (required by API)
    double TipConstraintPoint[3], TipForce[3];
    for (int i = 0; i < 3; i++) {
        TipConstraintPoint[i] = params.TipConstraintPoint[i];
        TipForce[i] = params.TipForce[i];
    }

    // Construct BVP parameters
    CRMShootingMethodParams BVPParams = CRMConstructShootingMethodParamSet(
        CathParams, CathConfig, L_inserted, ActuationCurrents,
        params.ContactMode, TipConstraintPoint, TipForce,
        params.IntegrationStepSize
    );

    // Solve BVP for deltau0 (using existing solver)
    double deltau0_init[3], ftip_init[3];
    for (int i = 0; i < 3; i++) {
        deltau0_init[i] = params.deltau0_initialguess[i];
        ftip_init[i] = params.ftip_initialguess[i];
    }
    double deltau0_calc[3], ftip_calc[3];
    int localmin;
    CRMShootingMethodBVP(BVPParams, deltau0_init, ftip_init,
                         deltau0_calc, ftip_calc, localmin);

    out.converged = localmin;
    for (int i = 0; i < 3; i++) {
        out.deltau0[i] = deltau0_calc[i];
    }

    // Call IVPJacobian solver to extract sensitivity blocks
    double x_N_state[NUM_STATES];
    double MomentResidual[3];
    auto jac_tuple = CRMSolverIVPJacobian(BVPParams, deltau0_calc, ftip_calc,
                                          true, x_N_state, MomentResidual);

    // Extract p_tip from state
    for (int i = 0; i < 3; i++) {
        out.p_tip[i] = x_N_state[i];
    }

    // Access Jacobian blocks from raw augmented state
    // Need to call the lower-level API to get raw Jacobian blocks
    // Actually, let me use the direct CoreWithJacobian approach
    CRMIVPCoreParams CoreParams(BVPParams.no_flex_seg, BVPParams.no_rigid_seg,
                                 BVPParams.no_act_set, BVPParams.no_locmarkers,
                                 BVPParams.no_fcum_steps);
    double x_0[NUM_STATES];
    for (int i = 0; i < 3; i++) x_0[i] = BVPParams.p0[i];
    for (int i = 0; i < 9; i++) x_0[i + 3] = BVPParams.R0[i];
    for (int i = 0; i < 3; i++) x_0[i + 12] = std::nan("0");

    CRMSolverIVP_Prep(
        BVPParams.no_flex_seg, BVPParams.no_rigid_seg, BVPParams.no_act_set,
        BVPParams.no_locmarkers, BVPParams.no_fcum_steps,
        x_0, BVPParams.IntegrationStepSize,
        BVPParams.Li, BVPParams.dlambdainv,
        BVPParams.SegmentTypes,
        BVPParams.SegEndLambdas, BVPParams.LocMarkerLambdas,
        BVPParams.rho,
        BVPParams.K, BVPParams.Kinv, BVPParams.ustar,
        BVPParams.ActMass,
        BVPParams.CoilAlignmentTurnAreaMatrix,
        BVPParams.MagMoment, BVPParams.fcumlambda,
        BVPParams.B0, BVPParams.g,
        false, true, CoreParams);

    AugmentedStateVector<IVPJacobiansFull> x_N;
    CRMSolverIVP_CoreWithJacobian(CoreParams, deltau0_calc, ftip_calc, x_N, MomentResidual);

    // Extract Jacobian blocks (row-major layout)
    for (int i = 0; i < 9; i++) {
        out.J_p_u0[i] = x_N._p_u0[i];
        out.J_u_u0[i] = x_N._u_u0[i];
    }
    // J_p_zc, J_u_zc: reverse actuator order
    for (int act = 0; act < NUM_ACT_SET; act++) {
        int src_act = (NUM_ACT_SET - 1) - act;
        for (int row = 0; row < 3; row++) {
            for (int col = 0; col < 3; col++) {
                out.J_p_zc[row * (3*NUM_ACT_SET) + act*3 + col] =
                    x_N._p_zc[row * (3*NUM_ACT_SET) + src_act*3 + col];
                out.J_u_zc[row * (3*NUM_ACT_SET) + act*3 + col] =
                    x_N._u_zc[row * (3*NUM_ACT_SET) + src_act*3 + col];
            }
        }
    }

    // Extract K_tip
    if (BVPParams.SegmentTypes[BVPParams.no_segments - 1] == CatheterSegmentType::FLEXIBLE) {
        for (int i = 0; i < 9; i++) {
            out.K_tip[i] = BVPParams.K[BVPParams.no_flex_seg - 1][i];
        }
    } else {
        for (int i = 0; i < 9; i++) {
            out.K_tip[i] = (i % 4 == 0) ? 1.0 : 0.0;
        }
    }

    // Logging
    out.nl_iterations = 0;
    out.final_residual = 0.0;
    out.lu_rank = 0;
    out.rel_solve_residual = 0.0;
    out.exit_code = 0;

    return localmin;
}

int equilibrium_backward(
    const EquilibriumResult& fwd_result,
    const double grad_p_tip[3],
    double grad_u[NUM_ACT_SET*3],
    int* lu_rank,
    double* rel_residual
) {
    // Map cached Jacobians to Eigen matrices (row-major as cached)
    typedef Matrix<double, 3, 3, RowMajor> Matrix3dRowMajor;
    Map<const Matrix3dRowMajor> K_tip_map(fwd_result.K_tip);
    Map<const Matrix3dRowMajor> J_p_u0_map(fwd_result.J_p_u0);
    Map<const Matrix3dRowMajor> J_u_u0_map(fwd_result.J_u_u0);
    Map<const Matrix<double, 3, Dynamic, RowMajor>> J_p_zc_map(
        fwd_result.J_p_zc, 3, 3*NUM_ACT_SET);
    Map<const Matrix<double, 3, Dynamic, RowMajor>> J_u_zc_map(
        fwd_result.J_u_zc, 3, 3*NUM_ACT_SET);

    Map<const Vector3d> grad_p_tip_map(grad_p_tip);
    Map<VectorXd> grad_u_map(grad_u, 3*NUM_ACT_SET);

    // Step 1: Compute K_tip * J_u_u0
    Matrix3d K_J_u = K_tip_map * J_u_u0_map;

    // Step 2: Compute RHS = J_p_u0^T * grad_p_tip
    Vector3d rhs = J_p_u0_map.transpose() * grad_p_tip_map;

    // Step 3: Solve (K_tip * J_u_u0)^T * λ = rhs using FullPivLU
    FullPivLU<Matrix3d> lu(K_J_u.transpose());
    int rank = lu.rank();
    if (lu_rank) *lu_rank = rank;

    // Rank check: must be full rank (3)
    if (rank < 3) {
        return 1;  // rank-deficient
    }

    // Solve for adjoint λ
    Vector3d lambda = lu.solve(rhs);

    // Step 4: Relative residual check
    Vector3d residual_vec = K_J_u.transpose() * lambda - rhs;
    double residual_norm = residual_vec.norm();
    double denom = std::max(rhs.norm(), 1.0);
    double rel_res = residual_norm / denom;
    if (rel_residual) *rel_residual = rel_res;

    if (rel_res > 1e-10) {
        return 2;  // residual too large
    }

    // Step 5: Compute K_tip * λ
    Vector3d K_lambda = K_tip_map * lambda;

    // Step 6: Compute gradient w.r.t. currents
    // g_u = J_p_zc^T * grad_p_tip - J_u_zc^T * (K_tip * λ)
    grad_u_map = J_p_zc_map.transpose() * grad_p_tip_map;
    grad_u_map -= J_u_zc_map.transpose() * K_lambda;

    return 0;  // success
}

} // namespace CRMCatheterModel
