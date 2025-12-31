#include "CRM_DiffDynamics.hpp"
#include "CRM_DiffEquilibrium.hpp"
#include <eigen3/Eigen/Dense>
#include <cmath>
#include <cstring>

using namespace Eigen;

namespace CRMCatheterModel {

// Helper: Compute M, D from catheter parameters
static void compute_physics_matrices(
    const CRMCatheterModelParams& CathParams,
    const double K_tip[9],  // from equilibrium
    double M_out[9],  // output: inertia (3×3 diagonal)
    double D_out[9]   // output: damping (3×3 diagonal)
) {
    // M = diag(m_eff, m_eff, m_eff)
    // where m_eff = ActMass / L_seg
    double ActMass = CathParams.ActMass[0];  // kg
    double L_seg = CathParams.SegLengths[0]; // mm (first actuator segment)
    double m_eff = ActMass / L_seg;          // kg/mm

    // M is 3×3 diagonal, stored row-major
    std::memset(M_out, 0, 9 * sizeof(double));
    M_out[0] = M_out[4] = M_out[8] = m_eff;

    // D = diag(d, d, d)
    // where d = 2 * sqrt(m_eff * k_eff) for critical damping
    // k_eff from K_tip diagonal average
    double k_eff = (K_tip[0] + K_tip[4] + K_tip[8]) / 3.0;

    // Handle case where k_eff might be very small or zero
    double d;
    if (k_eff > 1e-10) {
        d = 2.0 * std::sqrt(m_eff * k_eff);
    } else {
        // Fallback: use constant damping
        d = 0.01;
    }

    // D is 3×3 diagonal
    std::memset(D_out, 0, 9 * sizeof(double));
    D_out[0] = D_out[4] = D_out[8] = d;
}

// Helper: Compute Jacobians A, C, B
static void compute_jacobians(
    const double M[9],      // 3×3 diagonal
    const double D[9],      // 3×3 diagonal
    const double K[9],      // 3×3 from K_tip
    const double J_u_zc[3*NUM_ACT_SET*3],  // 3×3N from equilibrium
    double dt,
    double A_out[36],       // 6×6 row-major
    double C_out[36],       // 6×6 row-major
    double B_out[6*NUM_ACT_SET*3]  // 6×3N row-major
) {
    // Initialize all to zero
    std::memset(A_out, 0, 36 * sizeof(double));
    std::memset(C_out, 0, 36 * sizeof(double));
    std::memset(B_out, 0, 6*NUM_ACT_SET*3 * sizeof(double));

    // A = [I,       -dt*I    ]  (6×6)
    //     [dt*K,    M+dt*D   ]

    // Top-left: I (3×3)
    A_out[0] = 1.0; A_out[7] = 1.0; A_out[14] = 1.0;

    // Top-right: -dt*I (3×3)
    A_out[3] = -dt; A_out[10] = -dt; A_out[17] = -dt;

    // Bottom-left: dt*K (3×3)
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            A_out[(3+i)*6 + j] = dt * K[i*3 + j];
        }
    }

    // Bottom-right: M + dt*D (3×3 diagonal)
    for (int i = 0; i < 3; i++) {
        A_out[(3+i)*6 + (3+i)] = M[i*3 + i] + dt * D[i*3 + i];
    }

    // C = [-I,  0  ]  (6×6)
    //     [0,  -M  ]

    // Top-left: -I (3×3)
    C_out[0] = -1.0; C_out[7] = -1.0; C_out[14] = -1.0;

    // Bottom-right: -M (3×3 diagonal)
    for (int i = 0; i < 3; i++) {
        C_out[(3+i)*6 + (3+i)] = -M[i*3 + i];
    }

    // B = [0           ]  (6×3N)
    //     [-dt*K*J_u_zc]

    // Top block: zeros (already initialized)

    // Bottom block: -dt*K*J_u_zc (3×3 times 3×3N = 3×3N)
    // K is 3×3, J_u_zc is 3×3N
    Map<const Matrix<double, 3, 3, RowMajor>> K_map(K);
    Map<const Matrix<double, 3, Dynamic, RowMajor>> J_u_zc_map(J_u_zc, 3, 3*NUM_ACT_SET);

    MatrixXd K_J_u_zc = K_map * J_u_zc_map;  // 3×3N

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3*NUM_ACT_SET; j++) {
            B_out[(3+i) * (3*NUM_ACT_SET) + j] = -dt * K_J_u_zc(i, j);
        }
    }
}

int dynamics_forward(
    const double x_t[6],
    const double u_t[NUM_ACT_SET*3],
    double dt,
    double L_inserted,
    const CRMForwardKinematicsData& params,
    DynamicsStepResult& out
) {
    // Zero-initialize output
    std::memset(&out, 0, sizeof(DynamicsStepResult));

    // Step 1: Get K_tip from equilibrium (call with u_t to get current stiffness)
    // Use x_t[0:2] as base curvature for equilibrium solve
    double u_current[NUM_ACT_SET*3];
    std::memcpy(u_current, x_t, 3 * sizeof(double));  // u_0 from state

    EquilibriumResult eq_result_initial;
    std::memset(&eq_result_initial, 0, sizeof(EquilibriumResult));

    int eq_status = equilibrium_forward(u_t, L_inserted, params, eq_result_initial);
    if (eq_status != 0) {
        out.exit_code = 3;  // Equilibrium failed
        return 3;
    }

    // Extract K from equilibrium
    std::memcpy(out.K, eq_result_initial.K_tip, 9 * sizeof(double));

    // Step 2: Compute M, D from catheter parameters
    compute_physics_matrices(*(params.CathParams), out.K, out.M, out.D);

    // Step 3: Compute Jacobians A, C, B
    double A[36], C[36], B[6*NUM_ACT_SET*3];
    compute_jacobians(out.M, out.D, out.K, eq_result_initial.J_u_zc, dt, A, C, B);

    // Cache Jacobians for backward pass
    std::memcpy(out.J_G_xnext, A, 36 * sizeof(double));
    std::memcpy(out.J_G_xt, C, 36 * sizeof(double));
    std::memcpy(out.J_G_ut, B, 6*NUM_ACT_SET*3 * sizeof(double));

    // Step 4: Solve A*x_{t+1} = -C*x_t - B*u_t using FullPivLU
    Map<Matrix<double, 6, 6, RowMajor>> A_map(A);
    Map<Matrix<double, 6, 6, RowMajor>> C_map(C);
    Map<Matrix<double, 6, Dynamic, RowMajor>> B_map(B, 6, 3*NUM_ACT_SET);
    Map<const VectorXd> x_t_map(x_t, 6);
    Map<const VectorXd> u_t_map(u_t, 3*NUM_ACT_SET);

    // Compute RHS = -C*x_t - B*u_t
    VectorXd rhs = -C_map * x_t_map - B_map * u_t_map;

    // Solve using FullPivLU
    FullPivLU<MatrixXd> lu(A_map);

    int rank = lu.rank();
    out.lu_rank = rank;

    if (rank < 6) {
        out.exit_code = 1;  // Rank-deficient
        return 1;
    }

    VectorXd x_next_vec = lu.solve(rhs);

    // Extract solution
    for (int i = 0; i < 6; i++) {
        out.x_next[i] = x_next_vec[i];
    }

    // Check solve accuracy
    VectorXd residual_check = A_map * x_next_vec - rhs;
    double residual_norm = residual_check.norm();
    double rhs_norm = rhs.norm();
    double rel_res = residual_norm / std::max(rhs_norm, 1.0);

    out.solve_residual = residual_norm;
    out.rel_solve_residual = rel_res;

    if (rel_res > 1e-10) {
        out.exit_code = 2;  // Solve inaccurate
        return 2;
    }

    // Step 5: Call equilibrium_forward at x_{t+1} to get observables
    // Use x_next[0:2] as u_0 for equilibrium
    double u_next[NUM_ACT_SET*3];
    std::memcpy(u_next, out.x_next, 3 * sizeof(double));

    EquilibriumResult eq_result_next;
    std::memset(&eq_result_next, 0, sizeof(EquilibriumResult));

    eq_status = equilibrium_forward(u_t, L_inserted, params, eq_result_next);
    if (eq_status != 0) {
        // Non-fatal: just warn
        out.exit_code = 4;  // Equilibrium at t+1 failed (but dynamics succeeded)
    }

    // Extract observables
    std::memcpy(out.p_tip, eq_result_next.p_tip, 3 * sizeof(double));
    std::memcpy(out.u_tip, eq_result_next.deltau0, 3 * sizeof(double));

    // Cache equilibrium Jacobians
    std::memcpy(out.J_p_u0, eq_result_next.J_p_u0, 9 * sizeof(double));
    std::memcpy(out.J_p_ut, eq_result_next.J_p_zc, 3*NUM_ACT_SET*3 * sizeof(double));

    out.converged = 0;
    out.exit_code = 0;
    return 0;
}

int dynamics_backward(
    const DynamicsStepResult& fwd_result,
    const double grad_x_next[6],
    double grad_x_t[6],
    double grad_u_t[NUM_ACT_SET*3],
    int* lu_rank,
    double* rel_residual
) {
    // Extract cached Jacobians
    Map<const Matrix<double, 6, 6, RowMajor>> A_map(fwd_result.J_G_xnext);
    Map<const Matrix<double, 6, 6, RowMajor>> C_map(fwd_result.J_G_xt);
    Map<const Matrix<double, 6, Dynamic, RowMajor>> B_map(
        fwd_result.J_G_ut, 6, 3*NUM_ACT_SET);
    Map<const VectorXd> v(grad_x_next, 6);

    // Solve A^T λ = v using FullPivLU
    FullPivLU<MatrixXd> lu_solver(A_map.transpose());

    int rank = lu_solver.rank();
    if (lu_rank) *lu_rank = rank;

    if (rank < 6) {
        return 1;  // Rank-deficient
    }

    VectorXd lambda = lu_solver.solve(v);

    // Check solve accuracy
    double residual_norm = (A_map.transpose() * lambda - v).norm();
    double rel_res = residual_norm / std::max(v.norm(), 1.0);

    if (rel_residual) *rel_residual = rel_res;

    if (rel_res > 1e-10) {
        return 2;  // Inaccurate solve
    }

    // Compute gradients
    Map<VectorXd> grad_xt(grad_x_t, 6);
    grad_xt = -C_map.transpose() * lambda;

    Map<VectorXd> grad_ut(grad_u_t, 3*NUM_ACT_SET);
    grad_ut = -B_map.transpose() * lambda;

    return 0;  // Success
}

} // namespace CRMCatheterModel
