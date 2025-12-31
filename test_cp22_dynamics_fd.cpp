#include "CRM.hpp"
#include "CRM_DiffDynamics.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>

using namespace CRMCatheterModel;

// Helper to compute Frobenius norm
double frobenius_norm(const double* matrix, int rows, int cols) {
    double sum = 0.0;
    for (int i = 0; i < rows * cols; i++) {
        sum += matrix[i] * matrix[i];
    }
    return std::sqrt(sum);
}

// Helper to compute max absolute difference
double max_abs_diff(const double* A, const double* B, int rows, int cols) {
    double max_diff = 0.0;
    for (int i = 0; i < rows * cols; i++) {
        double diff = std::fabs(A[i] - B[i]);
        if (diff > max_diff) max_diff = diff;
    }
    return max_diff;
}

// Compute FD Jacobian w.r.t. x_t (6x6)
bool compute_fd_jacobian_x(const double* x_t, const double* u_t, double dt, double L_inserted,
                           const CRMForwardKinematicsData& FKParams, double eps,
                           double* J_fd_x) {
    DynamicsStepResult result_base, result_pert;

    // Base evaluation
    std::memset(&result_base, 0, sizeof(DynamicsStepResult));
    int status_base = dynamics_forward(x_t, u_t, dt, L_inserted, FKParams, result_base);
    if (status_base != 0 || result_base.lu_rank < 6) {
        std::cerr << "ERROR: Base forward pass failed (status=" << status_base
                  << ", rank=" << result_base.lu_rank << ")" << std::endl;
        return false;
    }

    // Perturb each dimension of x_t
    for (int i = 0; i < 6; i++) {
        double x_pert[6];
        std::memcpy(x_pert, x_t, 6 * sizeof(double));
        x_pert[i] += eps;

        std::memset(&result_pert, 0, sizeof(DynamicsStepResult));
        int status_pert = dynamics_forward(x_pert, u_t, dt, L_inserted, FKParams, result_pert);
        if (status_pert != 0 || result_pert.lu_rank < 6) {
            std::cerr << "ERROR: Perturbed forward pass failed for x_t[" << i
                      << "] (status=" << status_pert << ", rank=" << result_pert.lu_rank << ")" << std::endl;
            return false;
        }

        // Finite difference: column i of J_fd_x
        for (int j = 0; j < 6; j++) {
            J_fd_x[j * 6 + i] = (result_pert.x_next[j] - result_base.x_next[j]) / eps;
        }
    }

    return true;
}

// Compute FD Jacobian w.r.t. u_t (6x3)
bool compute_fd_jacobian_u(const double* x_t, const double* u_t, double dt, double L_inserted,
                           const CRMForwardKinematicsData& FKParams, double eps,
                           double* J_fd_u) {
    DynamicsStepResult result_base, result_pert;

    // Base evaluation
    std::memset(&result_base, 0, sizeof(DynamicsStepResult));
    int status_base = dynamics_forward(x_t, u_t, dt, L_inserted, FKParams, result_base);
    if (status_base != 0 || result_base.lu_rank < 6) {
        std::cerr << "ERROR: Base forward pass failed (status=" << status_base
                  << ", rank=" << result_base.lu_rank << ")" << std::endl;
        return false;
    }

    // Perturb each dimension of u_t
    for (int i = 0; i < 3; i++) {
        double u_pert[3];
        std::memcpy(u_pert, u_t, 3 * sizeof(double));
        u_pert[i] += eps;

        std::memset(&result_pert, 0, sizeof(DynamicsStepResult));
        int status_pert = dynamics_forward(x_t, u_pert, dt, L_inserted, FKParams, result_pert);
        if (status_pert != 0 || result_pert.lu_rank < 6) {
            std::cerr << "ERROR: Perturbed forward pass failed for u_t[" << i
                      << "] (status=" << status_pert << ", rank=" << result_pert.lu_rank << ")" << std::endl;
            return false;
        }

        // Finite difference: column i of J_fd_u
        for (int j = 0; j < 6; j++) {
            J_fd_u[j * 3 + i] = (result_pert.x_next[j] - result_base.x_next[j]) / eps;
        }
    }

    return true;
}

// Compute analytical Jacobians via VJP (backward mode)
bool compute_analytical_jacobians(const double* x_t, const double* u_t, double dt, double L_inserted,
                                  const CRMForwardKinematicsData& FKParams,
                                  double* J_analytical_x, double* J_analytical_u) {
    DynamicsStepResult result;

    // Forward pass
    std::memset(&result, 0, sizeof(DynamicsStepResult));
    int status = dynamics_forward(x_t, u_t, dt, L_inserted, FKParams, result);
    if (status != 0 || result.lu_rank < 6) {
        std::cerr << "ERROR: Forward pass failed (status=" << status
                  << ", rank=" << result.lu_rank << ")" << std::endl;
        return false;
    }

    // For each canonical output direction e_k in R^6
    for (int k = 0; k < 6; k++) {
        double grad_x_next[6] = {0.0};
        grad_x_next[k] = 1.0;

        double grad_x_t[6], grad_u_t[3];
        int lu_rank;
        double rel_residual;

        int bwd_status = dynamics_backward(result, grad_x_next, FKParams, grad_x_t, grad_u_t,
                                          &lu_rank, &rel_residual);

        if (bwd_status != 0 || lu_rank < 6) {
            std::cerr << "ERROR: Backward pass failed for e_" << k
                      << " (status=" << bwd_status << ", rank=" << lu_rank << ")" << std::endl;
            return false;
        }

        // VJP gives (∂x_next/∂x_t)^T e_k, which is column k of J^T
        // So grad_x_t is row k of J_analytical_x
        for (int i = 0; i < 6; i++) {
            J_analytical_x[k * 6 + i] = grad_x_t[i];
        }

        // Similarly for u
        for (int i = 0; i < 3; i++) {
            J_analytical_u[k * 3 + i] = grad_u_t[i];
        }
    }

    return true;
}

// Test a single operating point
bool test_operating_point(const char* op_name, const double* x_t, const double* u_t,
                         double dt, double L_inserted, const CRMForwardKinematicsData& FKParams,
                         double eps, double threshold) {
    std::cout << "Operating Point: " << op_name << std::endl;
    std::cout << "  x_t = [";
    for (int i = 0; i < 6; i++) {
        std::cout << x_t[i];
        if (i < 5) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  u_t = [";
    for (int i = 0; i < 3; i++) {
        std::cout << u_t[i];
        if (i < 2) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Compute FD Jacobians
    double J_fd_x[36], J_fd_u[18];
    if (!compute_fd_jacobian_x(x_t, u_t, dt, L_inserted, FKParams, eps, J_fd_x)) {
        return false;
    }
    if (!compute_fd_jacobian_u(x_t, u_t, dt, L_inserted, FKParams, eps, J_fd_u)) {
        return false;
    }

    // Compute analytical Jacobians
    double J_analytical_x[36], J_analytical_u[18];
    if (!compute_analytical_jacobians(x_t, u_t, dt, L_inserted, FKParams,
                                      J_analytical_x, J_analytical_u)) {
        return false;
    }

    // Compute errors
    double norm_fd_x = frobenius_norm(J_fd_x, 6, 6);
    double norm_fd_u = frobenius_norm(J_fd_u, 6, 3);

    double diff_x[36], diff_u[18];
    for (int i = 0; i < 36; i++) {
        diff_x[i] = J_fd_x[i] - J_analytical_x[i];
    }
    for (int i = 0; i < 18; i++) {
        diff_u[i] = J_fd_u[i] - J_analytical_u[i];
    }

    double norm_diff_x = frobenius_norm(diff_x, 6, 6);
    double norm_diff_u = frobenius_norm(diff_u, 6, 3);

    double rel_err_x = norm_diff_x / std::fmax(norm_fd_x, 1e-12);
    double rel_err_u = norm_diff_u / std::fmax(norm_fd_u, 1e-12);

    double max_abs_err_x = max_abs_diff(J_fd_x, J_analytical_x, 6, 6);
    double max_abs_err_u = max_abs_diff(J_fd_u, J_analytical_u, 6, 3);

    std::cout << "  ||J_fd_x||_F         = " << norm_fd_x << std::endl;
    std::cout << "  ||J_fd_u||_F         = " << norm_fd_u << std::endl;
    std::cout << "  rel_err_x            = " << rel_err_x << std::endl;
    std::cout << "  rel_err_u            = " << rel_err_u << std::endl;
    std::cout << "  max_abs_err_x        = " << max_abs_err_x << std::endl;
    std::cout << "  max_abs_err_u        = " << max_abs_err_u << std::endl;

    bool pass = true;
    if (rel_err_x >= threshold) {
        std::cerr << "  FAIL: rel_err_x = " << rel_err_x << " >= " << threshold << std::endl;
        pass = false;
    } else {
        std::cout << "  PASS: rel_err_x < " << threshold << std::endl;
    }

    if (rel_err_u >= threshold) {
        std::cerr << "  FAIL: rel_err_u = " << rel_err_u << " >= " << threshold << std::endl;
        pass = false;
    } else {
        std::cout << "  PASS: rel_err_u < " << threshold << std::endl;
    }

    std::cout << std::endl;
    return pass;
}

int main() {
    std::cout << std::setprecision(10);
    std::cout << "CP2.2 Dynamics Finite Difference Validation" << std::endl;
    std::cout << "============================================" << std::endl << std::endl;

    // Load parameters
    const char* param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt";
    const char* config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt";

    CRMCatheterModelParams CathParams = Load_CRMCatheterModelParams(param_file);
    CatheterConfiguration CathConfig = Load_CatheterConfiguration(config_file);

    // Setup FK params
    CRMForwardKinematicsData FKParams;
    FKParams.CathParams = &CathParams;
    FKParams.CathConfig = &CathConfig;
    FKParams.ContactMode = ContactModeType::FREE_TIP;
    FKParams.TipForce[0] = 0.0;
    FKParams.TipForce[1] = 0.0;
    FKParams.TipForce[2] = 0.0;
    FKParams.deltau0_initialguess[0] = 0.0;
    FKParams.deltau0_initialguess[1] = 0.0;
    FKParams.deltau0_initialguess[2] = 0.0;
    FKParams.IntegrationStepSize = 0.5;
    FKParams.FinalValueOnly = true;

    // Test parameters
    double dt = 0.01;  // 10ms
    double L_inserted = 50.0;  // mm
    double eps = 1e-6;  // FD epsilon
    double threshold = 1e-4;  // acceptance threshold

    std::cout << "Test configuration:" << std::endl;
    std::cout << "  dt = " << dt << " s" << std::endl;
    std::cout << "  L_inserted = " << L_inserted << " mm" << std::endl;
    std::cout << "  FD epsilon = " << eps << std::endl;
    std::cout << "  Threshold = " << threshold << std::endl;
    std::cout << std::endl;

    bool all_pass = true;

    // OP1: Rest
    {
        double x_t[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double u_t[3] = {0.0, 0.0, 0.0};
        if (!test_operating_point("OP1 (Rest)", x_t, u_t, dt, L_inserted, FKParams, eps, threshold)) {
            all_pass = false;
        }
    }

    // OP2: Actuated
    {
        double x_t[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double u_t[3] = {0.1, 0.0, 0.0};
        if (!test_operating_point("OP2 (Actuated)", x_t, u_t, dt, L_inserted, FKParams, eps, threshold)) {
            all_pass = false;
        }
    }

    // OP3: Moving
    {
        double x_t[6] = {0.01, 0.0, 0.0, 0.1, 0.0, 0.0};
        double u_t[3] = {0.1, 0.0, 0.0};
        if (!test_operating_point("OP3 (Moving)", x_t, u_t, dt, L_inserted, FKParams, eps, threshold)) {
            all_pass = false;
        }
    }

    std::cout << "============================================" << std::endl;
    if (all_pass) {
        std::cout << "PASS: All operating points validated successfully" << std::endl;
        std::cout << "============================================" << std::endl;
        return 0;
    } else {
        std::cout << "FAIL: One or more operating points failed validation" << std::endl;
        std::cout << "============================================" << std::endl;
        return 1;
    }
}
