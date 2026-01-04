#include "CRM.hpp"
#include "CRM_DiffEquilibrium.hpp"
#include <iostream>
#include <cmath>

using namespace CRMCatheterModel;

int main() {
    std::cout << "CP1.3 Test - Jacobian caching" << std::endl << std::endl;

    // Load catheter parameters
    CRMCatheterModelParams CathParams = Load_CRMCatheterModelParams("../catheterdata/CatheterParameterSet_1_new.txt");
    CatheterConfiguration CathConfig = Load_CatheterConfiguration("../catheterdata/CatheterSpatialConfiguration_1.txt");

    // Setup forward kinematics data
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

    // Test case: zero currents
    double u[3] = {0.0, 0.0, 0.0};
    double L_inserted = 50.0;

    EquilibriumResult result;
    int status = equilibrium_forward(u, L_inserted, FKParams, result);

    std::cout << "Status: " << status << std::endl;
    std::cout << "p_tip: [" << result.p_tip[0] << ", " << result.p_tip[1] << ", "
              << result.p_tip[2] << "]" << std::endl;
    std::cout << std::endl;

    // Check J_p_u0 (3×3)
    std::cout << "J_p_u0 (3×3, row-major):" << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << result.J_p_u0[i*3 + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Check J_u_u0 (3×3)
    std::cout << "J_u_u0 (3×3, row-major):" << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << result.J_u_u0[i*3 + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Check K_tip (3×3)
    std::cout << "K_tip (3×3, row-major):" << std::endl;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            std::cout << result.K_tip[i*3 + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // Verify Jacobians are non-zero
    double J_p_u0_norm = 0.0, J_u_u0_norm = 0.0, K_tip_norm = 0.0;
    for (int i = 0; i < 9; i++) {
        J_p_u0_norm += result.J_p_u0[i] * result.J_p_u0[i];
        J_u_u0_norm += result.J_u_u0[i] * result.J_u_u0[i];
        K_tip_norm += result.K_tip[i] * result.K_tip[i];
    }
    J_p_u0_norm = std::sqrt(J_p_u0_norm);
    J_u_u0_norm = std::sqrt(J_u_u0_norm);
    K_tip_norm = std::sqrt(K_tip_norm);

    std::cout << "Frobenius norms:" << std::endl;
    std::cout << "||J_p_u0|| = " << J_p_u0_norm << std::endl;
    std::cout << "||J_u_u0|| = " << J_u_u0_norm << std::endl;
    std::cout << "||K_tip|| = " << K_tip_norm << std::endl;
    std::cout << std::endl;

    bool pass = true;
    if (status != 0) {
        std::cout << "FAIL: status != 0" << std::endl;
        pass = false;
    }
    if (J_p_u0_norm < 1e-6) {
        std::cout << "FAIL: J_p_u0 is zero" << std::endl;
        pass = false;
    }
    if (J_u_u0_norm < 1e-6) {
        std::cout << "FAIL: J_u_u0 is zero" << std::endl;
        pass = false;
    }
    if (K_tip_norm < 1e-6) {
        std::cout << "FAIL: K_tip is zero" << std::endl;
        pass = false;
    }

    if (pass) {
        std::cout << "CP1.3: PASS - Jacobians cached" << std::endl;
        return 0;
    } else {
        std::cout << "CP1.3: FAIL" << std::endl;
        return 1;
    }
}
