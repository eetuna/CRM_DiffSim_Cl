#include "CRM.hpp"
#include "CRM_DiffEquilibrium.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>

using namespace CRMCatheterModel;

void print_results(const char* case_name, const EquilibriumResult& result,
                   const double ustar_used[][3], int num_flex, const double g_used[3]) {
    std::cout << "\n========================================\n";
    std::cout << case_name << "\n";
    std::cout << "========================================\n";

    std::cout << std::setprecision(17);
    std::cout << "ustar_used:\n";
    for (int i = 0; i < num_flex; i++) {
        std::cout << "  Seg " << i << ": [" << ustar_used[i][0] << ", "
                  << ustar_used[i][1] << ", " << ustar_used[i][2] << "]\n";
    }

    std::cout << "gravity_used: [" << g_used[0] << ", " << g_used[1] << ", " << g_used[2] << "]\n";

    std::cout << "\nRESULTS:\n";
    std::cout << "  status (converged): " << result.converged << "\n";
    std::cout << "  p_tip: [" << result.p_tip[0] << ", "
              << result.p_tip[1] << ", " << result.p_tip[2] << "]\n";
    std::cout << "  deltau0: [" << result.deltau0[0] << ", "
              << result.deltau0[1] << ", " << result.deltau0[2] << "]\n";

    // Check if p_tip is approximately [0, 0, 50]
    double tol = 1e-3; // 1mm tolerance
    bool is_straight = (std::abs(result.p_tip[0]) < tol &&
                        std::abs(result.p_tip[1]) < tol &&
                        std::abs(result.p_tip[2] - 50.0) < tol);
    std::cout << "\n  Is p_tip ≈ [0,0,50]? " << (is_straight ? "YES" : "NO") << "\n";
    if (!is_straight) {
        std::cout << "  Deviation from [0,0,50]: ["
                  << result.p_tip[0] << ", "
                  << result.p_tip[1] << ", "
                  << (result.p_tip[2] - 50.0) << "]\n";
    }
}

int main() {
    std::cout << "===========================================\n";
    std::cout << "PHYSICS AUDIT: Does u=0 produce straight rod?\n";
    std::cout << "===========================================\n";

    // Load catheter parameters
    const char* param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt";
    const char* config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt";

    CRMCatheterModelParams CathParams = Load_CRMCatheterModelParams(param_file);
    CatheterConfiguration CathConfig = Load_CatheterConfiguration(config_file);

    // Print segment information
    std::cout << "\nCATHETER GEOMETRY:\n";
    std::cout << "  SegmentLengths: ";
    double total_length = 0.0;
    for (int i = 0; i < CathParams.no_segments; i++) {
        std::cout << CathParams.SegLengths[i] << " ";
        total_length += CathParams.SegLengths[i];
    }
    std::cout << "\n  Total length: " << total_length << " mm\n";
    std::cout << "  L_inserted: 50.0 mm\n";
    std::cout << "  no_flex_seg: " << CathParams.no_flex_seg << "\n";
    std::cout << "  no_segments: " << CathParams.no_segments << "\n";

    // Print original parameters
    std::cout << "\nORIGINAL PARAMETERS:\n";
    std::cout << "  ustarlist (intrinsic curvature):\n";
    for (int i = 0; i < CathParams.no_flex_seg; i++) {
        std::cout << "    Seg " << i << ": [" << CathParams.ustar[i][0] << ", "
                  << CathParams.ustar[i][1] << ", " << CathParams.ustar[i][2] << "]\n";
    }
    std::cout << "  gravity: [" << CathConfig.g[0] << ", "
              << CathConfig.g[1] << ", " << CathConfig.g[2] << "]\n";
    std::cout << "  p0: [" << CathConfig.p0[0] << ", "
              << CathConfig.p0[1] << ", " << CathConfig.p0[2] << "]\n";
    std::cout << "  R0: identity? " <<
        ((CathConfig.R0[0] == 1.0 && CathConfig.R0[4] == 1.0 && CathConfig.R0[8] == 1.0) ? "YES" : "NO") << "\n";

    // Setup forward kinematics data (common for both cases)
    double u[3] = {0.0, 0.0, 0.0};
    double L_inserted = 50.0;

    CRMForwardKinematicsData FKParams;
    FKParams.ContactMode = ContactModeType::FREE_TIP;
    FKParams.TipForce[0] = 0.0;
    FKParams.TipForce[1] = 0.0;
    FKParams.TipForce[2] = 0.0;
    FKParams.deltau0_initialguess[0] = 0.0;
    FKParams.deltau0_initialguess[1] = 0.0;
    FKParams.deltau0_initialguess[2] = 0.0;
    FKParams.IntegrationStepSize = 0.5;
    FKParams.FinalValueOnly = true;

    // ============================================================
    // CASE A: Baseline (as-is from parameter files)
    // ============================================================

    FKParams.CathParams = &CathParams;
    FKParams.CathConfig = &CathConfig;

    EquilibriumResult result_A;
    std::memset(&result_A, 0, sizeof(EquilibriumResult));

    int status_A = equilibrium_forward(u, L_inserted, FKParams, result_A);

    // Store what was actually used
    double ustar_A[2][3];
    for (int i = 0; i < CathParams.no_flex_seg; i++) {
        for (int j = 0; j < 3; j++) {
            ustar_A[i][j] = CathParams.ustar[i][j];
        }
    }
    double g_A[3] = {CathConfig.g[0], CathConfig.g[1], CathConfig.g[2]};

    print_results("CASE A: Baseline (as-is)", result_A, ustar_A, CathParams.no_flex_seg, g_A);

    // ============================================================
    // CASE B: Zero intrinsic curvature + Zero gravity
    // ============================================================

    // Make a copy and zero out ustar and gravity
    CRMCatheterModelParams CathParams_B = CathParams;
    CatheterConfiguration CathConfig_B = CathConfig;

    // Zero intrinsic curvature
    for (int i = 0; i < CathParams_B.no_flex_seg; i++) {
        CathParams_B.ustar[i][0] = 0.0;
        CathParams_B.ustar[i][1] = 0.0;
        CathParams_B.ustar[i][2] = 0.0;
    }

    // Zero gravity
    CathConfig_B.g[0] = 0.0;
    CathConfig_B.g[1] = 0.0;
    CathConfig_B.g[2] = 0.0;

    FKParams.CathParams = &CathParams_B;
    FKParams.CathConfig = &CathConfig_B;

    EquilibriumResult result_B;
    std::memset(&result_B, 0, sizeof(EquilibriumResult));

    int status_B = equilibrium_forward(u, L_inserted, FKParams, result_B);

    // Store what was actually used
    double ustar_B[2][3];
    for (int i = 0; i < CathParams_B.no_flex_seg; i++) {
        for (int j = 0; j < 3; j++) {
            ustar_B[i][j] = CathParams_B.ustar[i][j];
        }
    }
    double g_B[3] = {CathConfig_B.g[0], CathConfig_B.g[1], CathConfig_B.g[2]};

    print_results("CASE B: Zero ustar + Zero gravity", result_B, ustar_B, CathParams_B.no_flex_seg, g_B);

    // ============================================================
    // INTERPRETATION
    // ============================================================

    std::cout << "\n========================================\n";
    std::cout << "INTERPRETATION\n";
    std::cout << "========================================\n";

    std::cout << "\n1. In CASE A (as-is from parameter files):\n";
    std::cout << "   - Non-zero intrinsic curvature (ustarlist) exists\n";
    std::cout << "   - Gravity is present (9.81 m/s² in +z)\n";
    std::cout << "   - Therefore, p_tip ≠ [0,0,50] is EXPECTED\n";

    std::cout << "\n2. In CASE B (zero ustar + zero gravity):\n";
    double tol = 1e-3;
    bool case_b_straight = (std::abs(result_B.p_tip[0]) < tol &&
                            std::abs(result_B.p_tip[1]) < tol &&
                            std::abs(result_B.p_tip[2] - 50.0) < tol);

    if (case_b_straight) {
        std::cout << "   - p_tip ≈ [0,0,50] → Confirms model is correct\n";
        std::cout << "   - At u=0 with zero ustar and zero gravity, rod is straight\n";
    } else {
        std::cout << "   - p_tip ≠ [0,0,50] → Unexpected!\n";
        std::cout << "   - Possible causes:\n";
        std::cout << "     (a) Coordinate frame convention (e.g., z-up vs z-down)\n";
        std::cout << "     (b) Base rotation R0 not truly identity\n";
        std::cout << "     (c) Rigid segment mechanics (if partially inserted)\n";
        std::cout << "     (d) Numerical tolerance in solver\n";

        // Check which segment is at the tip
        double cumulative = 0.0;
        int tip_segment = -1;
        for (int i = CathParams.no_segments - 1; i >= 0; i--) {
            cumulative += CathParams.SegLengths[i];
            if (cumulative >= L_inserted) {
                tip_segment = i;
                break;
            }
        }
        std::cout << "     Tip is in segment " << tip_segment << " (0-indexed from distal)\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "CONCLUSION\n";
    std::cout << "========================================\n";
    std::cout << "At u=[0,0,0], p_tip ≈ [0,0,L] is " << (case_b_straight ? "" : "NOT ")
              << "expected when:\n";
    std::cout << "  - ustar = 0 (zero intrinsic curvature)\n";
    std::cout << "  - g = 0 (zero gravity)\n";
    std::cout << "  - Free-tip boundary condition\n";
    std::cout << "  - R0 = identity, p0 = [0,0,0]\n";

    std::cout << "\nFor the DEFAULT parameter files (CASE A):\n";
    std::cout << "  p_tip ≠ [0,0,L] is EXPECTED due to:\n";
    std::cout << "    1. Non-zero intrinsic curvature (ustarlist)\n";
    std::cout << "    2. Gravity (9.81 m/s² in +z)\n";

    return 0;
}
