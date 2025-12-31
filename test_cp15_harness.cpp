#include "CRM.hpp"
#include "CRM_DiffEquilibrium.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>

using namespace CRMCatheterModel;

// Helper: compute Frobenius norm of a 3x3 matrix stored row-major
double frobenius_norm_3x3(const double* M) {
    double sum = 0.0;
    for (int i = 0; i < 9; i++) {
        sum += M[i] * M[i];
    }
    return std::sqrt(sum);
}

// Helper: compute Frobenius norm of a 3×N matrix stored row-major
double frobenius_norm_3xN(const double* M, int N) {
    double sum = 0.0;
    for (int i = 0; i < 3 * N; i++) {
        sum += M[i] * M[i];
    }
    return std::sqrt(sum);
}

// Helper: generate fingerprint string
std::string generate_fingerprint(
    const char* param_file,
    const char* config_file,
    double L_inserted,
    const CRMCatheterModelParams& CathParams,
    const CatheterConfiguration& CathConfig
) {
    std::ostringstream oss;
    oss << std::setprecision(17);

    // File paths (absolute or relative)
    oss << "param_file=" << param_file << ";";
    oss << "config_file=" << config_file << ";";

    // L_inserted
    oss << "L_inserted=" << L_inserted << ";";

    // NUM_ACT_SET (compile-time constant)
    oss << "NUM_ACT_SET=" << NUM_ACT_SET << ";";

    // no_segments, no_flex_seg, no_rigid_seg
    oss << "no_segments=" << (CathParams.no_flex_seg + CathParams.no_rigid_seg + CathParams.no_act_set) << ";";
    oss << "no_flex_seg=" << CathParams.no_flex_seg << ";";
    oss << "no_rigid_seg=" << CathParams.no_rigid_seg << ";";

    // Gravity vector
    oss << "gravity=[" << CathConfig.g[0] << "," << CathConfig.g[1] << "," << CathConfig.g[2] << "];";

    // First 3 values of uStarList (from first flexible segment)
    oss << "ustar_0=[" << CathParams.ustar[0][0] << "," << CathParams.ustar[0][1] << "," << CathParams.ustar[0][2] << "];";

    return oss.str();
}

int main() {
    std::cout << std::setprecision(17);
    std::cout << "CP1.5 C++ Reference Harness" << std::endl;
    std::cout << "===========================" << std::endl << std::endl;

    // Load catheter parameters (exact paths specified)
    const char* param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt";
    const char* config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt";

    CRMCatheterModelParams CathParams = Load_CRMCatheterModelParams(param_file);
    CatheterConfiguration CathConfig = Load_CatheterConfiguration(config_file);

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

    // Smoke case: u = [0, 0, 0], L_inserted = 50.0
    double u[3] = {0.0, 0.0, 0.0};
    double L_inserted = 50.0;

    // Zero-initialize result structure
    EquilibriumResult result;
    std::memset(&result, 0, sizeof(EquilibriumResult));

    // Call equilibrium_forward
    int status = equilibrium_forward(u, L_inserted, FKParams, result);

    // Print all outputs
    std::cout << "status=" << status << std::endl;
    std::cout << std::endl;

    std::cout << "p_tip=[" << result.p_tip[0] << ", "
              << result.p_tip[1] << ", "
              << result.p_tip[2] << "]" << std::endl;
    std::cout << std::endl;

    std::cout << "deltau0=[" << result.deltau0[0] << ", "
              << result.deltau0[1] << ", "
              << result.deltau0[2] << "]" << std::endl;
    std::cout << std::endl;

    // J_p_u0 (3×3, row-major)
    std::cout << "J_p_u0 (3x3, row-major):" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "  [";
        for (int j = 0; j < 3; j++) {
            std::cout << result.J_p_u0[i*3 + j];
            if (j < 2) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;

    // J_u_u0 (3×3, row-major)
    std::cout << "J_u_u0 (3x3, row-major):" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "  [";
        for (int j = 0; j < 3; j++) {
            std::cout << result.J_u_u0[i*3 + j];
            if (j < 2) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;

    // J_p_zc (3×3*NUM_ACT_SET, row-major)
    std::cout << "J_p_zc (3x" << 3*NUM_ACT_SET << ", row-major):" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "  [";
        for (int j = 0; j < 3*NUM_ACT_SET; j++) {
            std::cout << result.J_p_zc[i*(3*NUM_ACT_SET) + j];
            if (j < 3*NUM_ACT_SET - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;

    // J_u_zc (3×3*NUM_ACT_SET, row-major)
    std::cout << "J_u_zc (3x" << 3*NUM_ACT_SET << ", row-major):" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "  [";
        for (int j = 0; j < 3*NUM_ACT_SET; j++) {
            std::cout << result.J_u_zc[i*(3*NUM_ACT_SET) + j];
            if (j < 3*NUM_ACT_SET - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;

    // K_tip (3×3, row-major)
    std::cout << "K_tip (3x3, row-major):" << std::endl;
    for (int i = 0; i < 3; i++) {
        std::cout << "  [";
        for (int j = 0; j < 3; j++) {
            std::cout << result.K_tip[i*3 + j];
            if (j < 2) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
    std::cout << std::endl;

    // K_tip diagonal
    std::cout << "K_tip_diag=[" << result.K_tip[0] << ", "
              << result.K_tip[4] << ", "
              << result.K_tip[8] << "]" << std::endl;
    std::cout << std::endl;

    // Frobenius norms
    double J_p_u0_norm = frobenius_norm_3x3(result.J_p_u0);
    double J_u_u0_norm = frobenius_norm_3x3(result.J_u_u0);
    double J_p_zc_norm = frobenius_norm_3xN(result.J_p_zc, 3*NUM_ACT_SET);
    double J_u_zc_norm = frobenius_norm_3xN(result.J_u_zc, 3*NUM_ACT_SET);
    double K_tip_norm = frobenius_norm_3x3(result.K_tip);

    std::cout << "Frobenius norms:" << std::endl;
    std::cout << "  ||J_p_u0|| = " << J_p_u0_norm << std::endl;
    std::cout << "  ||J_u_u0|| = " << J_u_u0_norm << std::endl;
    std::cout << "  ||J_p_zc|| = " << J_p_zc_norm << std::endl;
    std::cout << "  ||J_u_zc|| = " << J_u_zc_norm << std::endl;
    std::cout << "  ||K_tip||  = " << K_tip_norm << std::endl;
    std::cout << std::endl;

    // Fingerprint
    std::string fingerprint = generate_fingerprint(
        param_file, config_file, L_inserted, CathParams, CathConfig
    );
    std::cout << "fingerprint=\"" << fingerprint << "\"" << std::endl;
    std::cout << std::endl;

    // Verify K_tip is diagonal
    bool K_diagonal = true;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (i != j && std::abs(result.K_tip[i*3 + j]) > 1e-12) {
                K_diagonal = false;
            }
        }
    }

    if (!K_diagonal) {
        std::cout << "WARNING: K_tip is not diagonal!" << std::endl;
    }

    std::cout << "===========================" << std::endl;
    std::cout << "C++ Reference Complete" << std::endl;

    return 0;
}
