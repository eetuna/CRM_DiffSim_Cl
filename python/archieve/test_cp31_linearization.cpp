#include "CRM.hpp"
#include "CRM_DiffDynamics.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>

using namespace CRMCatheterModel;

// Helper to compute L2 norm
double vector_norm(const double* v, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += v[i] * v[i];
    }
    return std::sqrt(sum);
}

// Extract Jacobian matrices A (6x6) and B (6x3) using VJP
bool extract_jacobians(const double* x_t, const double* u_t, double dt, double L_inserted,
                       const CRMForwardKinematicsData& FKParams,
                       double* A, double* B) {
    DynamicsStepResult result;

    // Forward pass
    std::memset(&result, 0, sizeof(DynamicsStepResult));
    int status = dynamics_forward(x_t, u_t, dt, L_inserted, FKParams, result);
    if (status != 0 || result.lu_rank < 6) {
        std::cerr << "ERROR: Forward pass failed (status=" << status
                  << ", rank=" << result.lu_rank << ")" << std::endl;
        return false;
    }

    // Extract Jacobians via VJP with canonical basis vectors
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

        // VJP gives (∂x_next/∂x_t)^T e_k = row k of A
        // VJP gives (∂x_next/∂u_t)^T e_k = row k of B
        for (int i = 0; i < 6; i++) {
            A[k * 6 + i] = grad_x_t[i];  // Row k of A
        }
        for (int i = 0; i < 3; i++) {
            B[k * 3 + i] = grad_u_t[i];  // Row k of B
        }
    }

    return true;
}

// Test linearization accuracy at a single operating point
bool test_operating_point(const char* name, const double* x_t, const double* u_t,
                          double dt, double L_inserted,
                          const CRMForwardKinematicsData& FKParams) {
    std::cout << "Operating Point: " << name << std::endl;
    std::cout << "  x_t = [";
    for (int i = 0; i < 6; i++) {
        std::cout << x_t[i];
        if (i < 5) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  u_t = [" << u_t[0] << ", " << u_t[1] << ", " << u_t[2] << "]" << std::endl;
    std::cout << std::endl;

    // Extract Jacobians
    double A[36], B[18];
    if (!extract_jacobians(x_t, u_t, dt, L_inserted, FKParams, A, B)) {
        std::cerr << "FAIL: Could not extract Jacobians" << std::endl;
        return false;
    }

    // Define perturbations (very small for highly nonlinear dynamics)
    double dx[6] = {5e-7, 5e-7, 5e-7, 5e-6, 5e-6, 5e-6};  // Very small perturbations
    double du[3] = {5e-5, 5e-5, 5e-5};

    double norm_dx = vector_norm(dx, 6);
    double norm_du = vector_norm(du, 3);

    std::cout << "  Perturbations:" << std::endl;
    std::cout << "    ||dx|| = " << norm_dx << std::endl;
    std::cout << "    ||du|| = " << norm_du << std::endl;
    std::cout << std::endl;

    // Compute f(x_t, u_t) (base point)
    DynamicsStepResult result_base;
    std::memset(&result_base, 0, sizeof(DynamicsStepResult));
    int status_base = dynamics_forward(x_t, u_t, dt, L_inserted, FKParams, result_base);
    if (status_base != 0 || result_base.lu_rank < 6) {
        std::cerr << "FAIL: Base forward pass failed" << std::endl;
        return false;
    }

    // Compute f(x_t + dx, u_t + du) (actual)
    double x_pert[6], u_pert[3];
    for (int i = 0; i < 6; i++) x_pert[i] = x_t[i] + dx[i];
    for (int i = 0; i < 3; i++) u_pert[i] = u_t[i] + du[i];

    DynamicsStepResult result_pert;
    std::memset(&result_pert, 0, sizeof(DynamicsStepResult));
    int status_pert = dynamics_forward(x_pert, u_pert, dt, L_inserted, FKParams, result_pert);
    if (status_pert != 0 || result_pert.lu_rank < 6) {
        std::cerr << "FAIL: Perturbed forward pass failed" << std::endl;
        return false;
    }

    // Linearized prediction: x_next ≈ f(x_t, u_t) + A*dx + B*du
    double x_linear[6];
    for (int i = 0; i < 6; i++) {
        double A_dx = 0.0;
        for (int j = 0; j < 6; j++) {
            A_dx += A[i * 6 + j] * dx[j];
        }
        double B_du = 0.0;
        for (int j = 0; j < 3; j++) {
            B_du += B[i * 3 + j] * du[j];
        }
        x_linear[i] = result_base.x_next[i] + A_dx + B_du;
    }

    // Compute error
    double error_vec[6];
    for (int i = 0; i < 6; i++) {
        error_vec[i] = result_pert.x_next[i] - x_linear[i];
    }
    double error_norm = vector_norm(error_vec, 6);

    // Relative error
    double rel_error = error_norm / norm_dx;

    std::cout << "  Linearization Test:" << std::endl;
    std::cout << "    ||x_actual - x_linear|| = " << error_norm << std::endl;
    std::cout << "    Relative error (||error|| / ||dx||) = " << rel_error << std::endl;
    std::cout << std::endl;

    // Acceptance threshold: relative error < 0.1
    const double threshold = 0.1;
    bool pass = rel_error < threshold;

    if (pass) {
        std::cout << "  PASS (rel_error < " << threshold << ")" << std::endl;
    } else {
        std::cout << "  FAIL (rel_error >= " << threshold << ")" << std::endl;
    }
    std::cout << std::endl;

    return pass;
}

int main() {
    std::cout << std::setprecision(10);
    std::cout << "CP3.1 Linearization Validation Test (C++)" << std::endl;
    std::cout << "==========================================" << std::endl << std::endl;

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

    double dt = 0.01;
    double L_inserted = 50.0;

    // Operating Point 1: Rest
    double x_op1[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double u_op1[3] = {0.0, 0.0, 0.0};

    // Operating Point 2: Actuated
    double x_op2[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double u_op2[3] = {0.1, 0.0, 0.0};

    // Operating Point 3: Moving
    double x_op3[6] = {0.01, 0.0, 0.0, 0.1, 0.0, 0.0};
    double u_op3[3] = {0.1, 0.0, 0.0};

    // Test all operating points
    bool pass1 = test_operating_point("OP1 - Rest", x_op1, u_op1, dt, L_inserted, FKParams);
    bool pass2 = test_operating_point("OP2 - Actuated", x_op2, u_op2, dt, L_inserted, FKParams);
    bool pass3 = test_operating_point("OP3 - Moving", x_op3, u_op3, dt, L_inserted, FKParams);

    // Overall verdict
    std::cout << "========================================" << std::endl;
    std::cout << "Overall Results:" << std::endl;
    std::cout << "  OP1 (Rest):     " << (pass1 ? "PASS" : "FAIL") << std::endl;
    std::cout << "  OP2 (Actuated): " << (pass2 ? "PASS" : "FAIL") << std::endl;
    std::cout << "  OP3 (Moving):   " << (pass3 ? "PASS" : "FAIL") << std::endl;
    std::cout << std::endl;

    bool all_pass = pass1 && pass2 && pass3;
    if (all_pass) {
        std::cout << "CP3.1 PASS: All operating points validated" << std::endl;
        return 0;
    } else {
        std::cout << "CP3.1 FAIL: Some operating points failed" << std::endl;
        return 1;
    }
}
