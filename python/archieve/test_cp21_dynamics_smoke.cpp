#include "CRM.hpp"
#include "CRM_DiffDynamics.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>

using namespace CRMCatheterModel;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "CP2.1 Dynamics Smoke Test" << std::endl;
    std::cout << "==========================" << std::endl << std::endl;

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

    // Smoke case: zero initial state, zero control, small dt
    double x_t[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double u_t[3] = {0.0, 0.0, 0.0};
    double dt = 0.01;  // 10ms
    double L_inserted = 50.0;

    std::cout << "Test: x_t = [0,0,0,0,0,0], u_t = [0,0,0], dt = " << dt << std::endl;
    std::cout << std::endl;

    DynamicsStepResult result;
    std::memset(&result, 0, sizeof(DynamicsStepResult));

    int status = dynamics_forward(x_t, u_t, dt, L_inserted, FKParams, result);

    std::cout << "Forward pass:" << std::endl;
    std::cout << "  status = " << status << std::endl;
    std::cout << "  converged = " << result.converged << std::endl;
    std::cout << "  lu_rank = " << result.lu_rank << std::endl;
    std::cout << "  solve_residual = " << result.solve_residual << std::endl;
    std::cout << "  rel_solve_residual = " << result.rel_solve_residual << std::endl;
    std::cout << "  exit_code = " << result.exit_code << std::endl;
    std::cout << std::endl;

    std::cout << "  x_next = [";
    for (int i = 0; i < 6; i++) {
        std::cout << result.x_next[i];
        if (i < 5) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << std::endl;

    std::cout << "  p_tip = [" << result.p_tip[0] << ", "
              << result.p_tip[1] << ", "
              << result.p_tip[2] << "]" << std::endl;
    std::cout << std::endl;

    // Sanity checks
    bool pass = true;

    if (status != 0) {
        std::cerr << "FAIL: status != 0" << std::endl;
        pass = false;
    }

    if (result.lu_rank < 6) {
        std::cerr << "FAIL: rank-deficient A matrix (rank = " << result.lu_rank << ")" << std::endl;
        pass = false;
    }

    if (result.rel_solve_residual >= 1e-10) {
        std::cerr << "FAIL: solve residual too large (" << result.rel_solve_residual << ")" << std::endl;
        pass = false;
    }

    // For zero input, expect x_next ≈ x_t (no change)
    double norm_change = 0.0;
    for (int i = 0; i < 6; i++) {
        double diff = result.x_next[i] - x_t[i];
        norm_change += diff * diff;
    }
    norm_change = std::sqrt(norm_change);

    std::cout << "  ||x_next - x_t|| = " << norm_change << std::endl;
    std::cout << std::endl;

    if (norm_change > 1e-6) {
        std::cerr << "FAIL: x_next changed significantly for zero input (norm = "
                  << norm_change << ")" << std::endl;
        pass = false;
    }

    // Check for NaN/Inf
    bool has_nan_inf = false;
    for (int i = 0; i < 6; i++) {
        if (!std::isfinite(result.x_next[i])) {
            std::cerr << "FAIL: x_next[" << i << "] is not finite" << std::endl;
            has_nan_inf = true;
            pass = false;
        }
    }

    // Check M, D, K matrices
    std::cout << "Physics matrices:" << std::endl;
    std::cout << "  M_diag = [" << result.M[0] << ", " << result.M[4] << ", " << result.M[8] << "]" << std::endl;
    std::cout << "  D_diag = [" << result.D[0] << ", " << result.D[4] << ", " << result.D[8] << "]" << std::endl;
    std::cout << "  K_diag = [" << result.K[0] << ", " << result.K[4] << ", " << result.K[8] << "]" << std::endl;
    std::cout << std::endl;

    if (result.M[0] <= 0 || result.M[4] <= 0 || result.M[8] <= 0) {
        std::cerr << "FAIL: M matrix has non-positive diagonal elements" << std::endl;
        pass = false;
    }

    if (result.D[0] <= 0 || result.D[4] <= 0 || result.D[8] <= 0) {
        std::cerr << "FAIL: D matrix has non-positive diagonal elements" << std::endl;
        pass = false;
    }

    // Check backward pass
    std::cout << "Backward pass test:" << std::endl;
    double grad_x_next[6] = {1.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    double grad_x_t[6], grad_u_t[3];
    int lu_rank;
    double rel_residual;

    int bwd_status = dynamics_backward(result, grad_x_next, FKParams, grad_x_t, grad_u_t,
                                       &lu_rank, &rel_residual);

    std::cout << "  bwd_status = " << bwd_status << std::endl;
    std::cout << "  lu_rank = " << lu_rank << std::endl;
    std::cout << "  rel_residual = " << rel_residual << std::endl;
    std::cout << "  grad_x_t = [";
    for (int i = 0; i < 6; i++) {
        std::cout << grad_x_t[i];
        if (i < 5) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  grad_u_t = [";
    for (int i = 0; i < 3; i++) {
        std::cout << grad_u_t[i];
        if (i < 2) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << std::endl;

    if (bwd_status != 0) {
        std::cerr << "FAIL: backward pass status != 0" << std::endl;
        pass = false;
    }

    if (lu_rank < 6) {
        std::cerr << "FAIL: backward pass rank-deficient" << std::endl;
        pass = false;
    }

    if (rel_residual >= 1e-10) {
        std::cerr << "FAIL: backward solve residual too large" << std::endl;
        pass = false;
    }

    // Check gradients are finite
    for (int i = 0; i < 6; i++) {
        if (!std::isfinite(grad_x_t[i])) {
            std::cerr << "FAIL: grad_x_t[" << i << "] is not finite" << std::endl;
            pass = false;
        }
    }

    for (int i = 0; i < 3; i++) {
        if (!std::isfinite(grad_u_t[i])) {
            std::cerr << "FAIL: grad_u_t[" << i << "] is not finite" << std::endl;
            pass = false;
        }
    }

    std::cout << "==========================" << std::endl;
    if (pass) {
        std::cout << "PASS: Smoke test succeeded" << std::endl;
        std::cout << "==========================" << std::endl;
        return 0;
    } else {
        std::cout << "FAIL: Smoke test failed" << std::endl;
        std::cout << "==========================" << std::endl;
        return 1;
    }
}
