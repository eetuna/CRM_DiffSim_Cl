#include "CRM.hpp"
#include "CRM_DiffEquilibrium.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>

using namespace CRMCatheterModel;

// Helper: check if value is finite
bool is_finite(double x) {
    return std::isfinite(x);
}

// Helper: check if array is all finite
bool is_all_finite(const double* arr, int n) {
    for (int i = 0; i < n; i++) {
        if (!is_finite(arr[i])) return false;
    }
    return true;
}

// Helper: check if array is not all zero
bool is_not_all_zero(const double* arr, int n, double tol = 1e-14) {
    for (int i = 0; i < n; i++) {
        if (std::abs(arr[i]) > tol) return true;
    }
    return false;
}

int main() {
    std::cout << std::setprecision(17);
    std::cout << "CP1.4 Golden Backward Test" << std::endl;
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

    // Test case: u = [0, 0, 0], L_inserted = 50.0
    double u[3] = {0.0, 0.0, 0.0};
    double L_inserted = 50.0;

    // Forward pass
    std::cout << "[1/2] Running equilibrium_forward..." << std::endl;
    EquilibriumResult fwd_result;
    std::memset(&fwd_result, 0, sizeof(EquilibriumResult));

    int fwd_status = equilibrium_forward(u, L_inserted, FKParams, fwd_result);

    std::cout << "  Forward status = " << fwd_status << std::endl;
    std::cout << "  p_tip = [" << fwd_result.p_tip[0] << ", "
              << fwd_result.p_tip[1] << ", "
              << fwd_result.p_tip[2] << "]" << std::endl << std::endl;

    if (fwd_status != 0) {
        std::cerr << "FAIL: Forward pass failed with status = " << fwd_status << std::endl;
        std::cerr << "fingerprint=\"forward_failed;param=" << param_file
                  << ";config=" << config_file << ";Li=" << L_inserted << "\"" << std::endl;
        return 1;
    }

    // Backward passes for each canonical direction
    std::cout << "[2/2] Running equilibrium_backward for 3 canonical directions..." << std::endl;
    std::cout << std::endl;

    bool all_pass = true;

    const char* direction_names[3] = {"X", "Y", "Z"};

    for (int i = 0; i < 3; i++) {
        std::cout << "Test " << (i+1) << "/3: grad_p_tip = e_" << direction_names[i]
                  << " = [" << (i==0?1:0) << ", " << (i==1?1:0) << ", " << (i==2?1:0) << "]" << std::endl;

        double grad_p_tip[3] = {0.0, 0.0, 0.0};
        grad_p_tip[i] = 1.0;

        double grad_u[3];
        int lu_rank;
        double rel_residual;

        int bwd_status = equilibrium_backward(fwd_result, grad_p_tip, grad_u, &lu_rank, &rel_residual);

        std::cout << "  status = " << bwd_status << std::endl;
        std::cout << "  lu_rank = " << lu_rank << std::endl;
        std::cout << "  rel_residual = " << rel_residual << std::endl;
        std::cout << "  grad_u = [" << grad_u[0] << ", " << grad_u[1] << ", " << grad_u[2] << "]" << std::endl;

        // Assertions
        bool pass = true;

        // Check 1: status == 0
        if (bwd_status != 0) {
            std::cerr << "  FAIL: status = " << bwd_status << " (expected 0)" << std::endl;
            pass = false;
        }

        // Check 2: lu_rank == 3
        if (lu_rank != 3) {
            std::cerr << "  FAIL: lu_rank = " << lu_rank << " (expected 3)" << std::endl;
            pass = false;
        }

        // Check 3: rel_residual < 1e-10
        if (rel_residual >= 1e-10) {
            std::cerr << "  FAIL: rel_residual = " << rel_residual << " >= 1e-10" << std::endl;
            pass = false;
        }

        // Check 4: grad_u is finite
        if (!is_all_finite(grad_u, 3)) {
            std::cerr << "  FAIL: grad_u contains non-finite values" << std::endl;
            pass = false;
        }

        // Check 5: grad_u is not all zero
        if (!is_not_all_zero(grad_u, 3)) {
            std::cerr << "  FAIL: grad_u is all zeros" << std::endl;
            pass = false;
        }

        if (pass) {
            std::cout << "  PASS" << std::endl;
        } else {
            std::cerr << "  fingerprint=\"backward_failed;direction=" << direction_names[i]
                      << ";param=" << param_file << ";config=" << config_file
                      << ";Li=" << L_inserted << "\"" << std::endl;
            all_pass = false;
        }

        std::cout << std::endl;
    }

    std::cout << "==========================" << std::endl;
    if (all_pass) {
        std::cout << "PASS: All backward tests succeeded" << std::endl;
        std::cout << "==========================" << std::endl;
        return 0;
    } else {
        std::cout << "FAIL: One or more backward tests failed" << std::endl;
        std::cout << "==========================" << std::endl;
        return 1;
    }
}
