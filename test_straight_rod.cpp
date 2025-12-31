#include "CRM.hpp"
#include "CRM_DiffEquilibrium.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>

using namespace CRMCatheterModel;

// Helper: check if value is near zero
bool is_near_zero(double x, double tol = 1e-9) {
    return std::abs(x) < tol;
}

int main() {
    std::cout << std::setprecision(17);
    std::cout << "CP1.X Straight-Rod Physics Sanity Test" << std::endl;
    std::cout << "=======================================" << std::endl << std::endl;

    // Load straight-rod parameters (ustar=0, gravity=0)
    const char* param_file = "./catheterdata/CatheterParameterSet_1_straight.txt";
    const char* config_file = "./catheterdata/CatheterSpatialConfiguration_1_straight.txt";

    std::cout << "Loading straight-rod configuration..." << std::endl;
    std::cout << "  param_file:  " << param_file << std::endl;
    std::cout << "  config_file: " << config_file << std::endl;
    std::cout << "  (ustar = 0, gravity = 0)" << std::endl << std::endl;

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

    // Test cases: u = [0, 0, 0], Li ∈ {0, 50, 100}
    double u[3] = {0.0, 0.0, 0.0};
    double Li_values[3] = {0.0, 50.0, 100.0};
    int num_tests = 3;

    bool all_pass = true;
    double tol = 1e-9;

    for (int i = 0; i < num_tests; i++) {
        double Li = Li_values[i];

        std::cout << "Test " << (i+1) << "/" << num_tests << ": Li = " << Li << " mm" << std::endl;

        EquilibriumResult result;
        std::memset(&result, 0, sizeof(EquilibriumResult));

        int status = equilibrium_forward(u, Li, FKParams, result);

        std::cout << "  status = " << status << std::endl;
        std::cout << "  p_tip = [" << result.p_tip[0] << ", "
                  << result.p_tip[1] << ", "
                  << result.p_tip[2] << "]" << std::endl;

        // Expected: p_tip ≈ [0, 0, Li]
        double expected_x = 0.0;
        double expected_y = 0.0;
        double expected_z = Li;

        double err_x = std::abs(result.p_tip[0] - expected_x);
        double err_y = std::abs(result.p_tip[1] - expected_y);
        double err_z = std::abs(result.p_tip[2] - expected_z);

        std::cout << "  Expected: p_tip ≈ [0, 0, " << Li << "]" << std::endl;
        std::cout << "  Errors: |x| = " << err_x << ", |y| = " << err_y
                  << ", |z - Li| = " << err_z << std::endl;

        bool pass = true;

        // Check 1: Forward pass converged
        if (status != 0) {
            std::cerr << "  FAIL: Forward pass failed with status = " << status << std::endl;
            pass = false;
        }

        // Check 2: |x| < tol
        if (err_x >= tol) {
            std::cerr << "  FAIL: |x| = " << err_x << " >= " << tol << std::endl;
            pass = false;
        }

        // Check 3: |y| < tol
        if (err_y >= tol) {
            std::cerr << "  FAIL: |y| = " << err_y << " >= " << tol << std::endl;
            pass = false;
        }

        // Check 4: |z - Li| < tol
        if (err_z >= tol) {
            std::cerr << "  FAIL: |z - Li| = " << err_z << " >= " << tol << std::endl;
            pass = false;
        }

        if (pass) {
            std::cout << "  PASS" << std::endl;
        } else {
            std::cerr << "  fingerprint=\"straight_rod_failed;Li=" << Li
                      << ";param=" << param_file << ";config=" << config_file << "\"" << std::endl;
            all_pass = false;
        }

        std::cout << std::endl;
    }

    std::cout << "=======================================" << std::endl;
    if (all_pass) {
        std::cout << "PASS: All straight-rod tests succeeded" << std::endl;
        std::cout << "  Straight-rod geometry verified for Li ∈ {0, 50, 100} mm" << std::endl;
        std::cout << "  |x|, |y|, |z-Li| < 1e-9" << std::endl;
        std::cout << "=======================================" << std::endl;
        return 0;
    } else {
        std::cout << "FAIL: One or more straight-rod tests failed" << std::endl;
        std::cout << "=======================================" << std::endl;
        return 1;
    }
}
