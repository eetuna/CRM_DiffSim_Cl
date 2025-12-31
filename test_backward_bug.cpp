#include "CRM.hpp"
#include "CRM_DiffEquilibrium.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>

using namespace CRMCatheterModel;

int main() {
    std::cout << std::setprecision(17);
    std::cout << "Testing equilibrium_backward directly\n";
    std::cout << "======================================\n\n";

    // Load parameters
    CRMCatheterModelParams CathParams = Load_CRMCatheterModelParams("./catheterdata/CatheterParameterSet_1_dyn.txt");
    CatheterConfiguration CathConfig = Load_CatheterConfiguration("./catheterdata/CatheterSpatialConfiguration_1.txt");

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

    // Test case
    double u[3] = {0.0, 0.0, 0.0};
    double L_inserted = 50.0;

    // Forward pass
    EquilibriumResult fwd_result;
    std::memset(&fwd_result, 0, sizeof(EquilibriumResult));

    int fwd_status = equilibrium_forward(u, L_inserted, FKParams, fwd_result);

    std::cout << "Forward pass:\n";
    std::cout << "  status = " << fwd_status << "\n";
    std::cout << "  p_tip = [" << fwd_result.p_tip[0] << ", "
              << fwd_result.p_tip[1] << ", "
              << fwd_result.p_tip[2] << "]\n\n";

    // Print cached Jacobians
    std::cout << "Cached Jacobians:\n";
    std::cout << "J_p_zc:\n";
    for (int i = 0; i < 3; i++) {
        std::cout << "  [";
        for (int j = 0; j < 3; j++) {
            std::cout << fwd_result.J_p_zc[i*3 + j];
            if (j < 2) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << "\n";

    // Backward pass for each component
    std::cout << "Backward passes:\n";
    std::cout << "================\n\n";

    for (int i = 0; i < 3; i++) {
        double grad_p_tip[3] = {0.0, 0.0, 0.0};
        grad_p_tip[i] = 1.0;

        double grad_u[3];
        int lu_rank;
        double rel_residual;

        int bwd_status = equilibrium_backward(fwd_result, grad_p_tip, grad_u, &lu_rank, &rel_residual);

        std::cout << "Backward pass " << i << ": grad_p_tip = ["
                  << grad_p_tip[0] << ", " << grad_p_tip[1] << ", " << grad_p_tip[2] << "]\n";
        std::cout << "  grad_u = [" << grad_u[0] << ", " << grad_u[1] << ", " << grad_u[2] << "]\n";
        std::cout << "  status = " << bwd_status << "\n";
        std::cout << "  lu_rank = " << lu_rank << "\n";
        std::cout << "  rel_residual = " << rel_residual << "\n\n";
    }

    return 0;
}
