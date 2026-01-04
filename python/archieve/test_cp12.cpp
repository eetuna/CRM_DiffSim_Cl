#include "CRM.hpp"
#include "CRM_DiffEquilibrium.hpp"
#include <iostream>

using namespace CRMCatheterModel;

int main() {
    std::cout << "CP1.2 Test - equilibrium_forward() wrapper" << std::endl << std::endl;

    // Load catheter parameters (same as existing CRMTest)
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

    // Test case 1: zero currents, 50mm insertion
    double u[3] = {0.0, 0.0, 0.0};
    double L_inserted = 50.0;

    EquilibriumResult result;
    int status = equilibrium_forward(u, L_inserted, FKParams, result);

    std::cout << "Input: u=[" << u[0] << "," << u[1] << "," << u[2]
              << "], L=" << L_inserted << "mm" << std::endl;
    std::cout << "Status: " << status << " (0=success)" << std::endl;
    std::cout << "Converged: " << result.converged << std::endl;
    std::cout << "p_tip: [" << result.p_tip[0] << ", "
              << result.p_tip[1] << ", "
              << result.p_tip[2] << "]" << std::endl;
    std::cout << "deltau0: [" << result.deltau0[0] << ", "
              << result.deltau0[1] << ", "
              << result.deltau0[2] << "]" << std::endl;
    std::cout << std::endl;

    if (status == 0 && result.converged == 0) {
        std::cout << "CP1.2: PASS - equilibrium_forward() wrapper works" << std::endl;
        return 0;
    } else {
        std::cout << "CP1.2: FAIL - status or converged != 0" << std::endl;
        return 1;
    }
}
