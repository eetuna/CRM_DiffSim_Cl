#include "CRM.hpp"
#include "CRM_DiffEquilibrium.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <sstream>
#include <vector>

using namespace CRMCatheterModel;

// Generate fingerprint string
std::string generate_fingerprint(
    const char* param_file,
    const char* config_file,
    double L_inserted,
    const CRMCatheterModelParams& CathParams,
    const CatheterConfiguration& CathConfig,
    bool straight_mode
) {
    std::ostringstream oss;
    oss << std::setprecision(17);

    oss << "param_file=" << param_file << ";";
    oss << "config_file=" << config_file << ";";
    oss << "L_inserted=" << L_inserted << ";";
    oss << "NUM_ACT_SET=" << NUM_ACT_SET << ";";
    oss << "no_segments=" << (CathParams.no_flex_seg + CathParams.no_rigid_seg + CathParams.no_act_set) << ";";
    oss << "no_flex_seg=" << CathParams.no_flex_seg << ";";
    oss << "no_rigid_seg=" << CathParams.no_rigid_seg << ";";
    oss << "gravity=[" << CathConfig.g[0] << "," << CathConfig.g[1] << "," << CathConfig.g[2] << "];";

    // ustar from first flex segment
    if (CathParams.no_flex_seg > 0) {
        oss << "ustar_0=[" << CathParams.ustar[0][0] << ","
            << CathParams.ustar[0][1] << "," << CathParams.ustar[0][2] << "];";
    }

    oss << "mode=" << (straight_mode ? "straight" : "baseline") << ";";

    return oss.str();
}

struct SweepResult {
    double Li;
    int status;
    double p_tip[3];
    double deltau0[3];
    double z_error;  // z - Li
    std::string fingerprint;
};

void run_sweep(
    const char* param_file,
    const char* config_file,
    bool straight_mode,
    const char* csv_filename,
    std::vector<SweepResult>& results
) {
    std::cout << "\n========================================\n";
    std::cout << (straight_mode ? "STRAIGHT ROD MODE" : "BASELINE MODE") << "\n";
    std::cout << "========================================\n";

    // Load catheter parameters
    CRMCatheterModelParams CathParams = Load_CRMCatheterModelParams(param_file);
    CatheterConfiguration CathConfig = Load_CatheterConfiguration(config_file);

    // Compute total length
    double total_length = 0.0;
    for (int i = 0; i < CathParams.no_segments; i++) {
        total_length += CathParams.SegLengths[i];
    }

    std::cout << "Total catheter length: " << total_length << " mm\n";

    // If straight mode, override ustar and gravity
    if (straight_mode) {
        std::cout << "Overriding ustar=0 and gravity=0...\n";
        for (int i = 0; i < CathParams.no_flex_seg; i++) {
            CathParams.ustar[i][0] = 0.0;
            CathParams.ustar[i][1] = 0.0;
            CathParams.ustar[i][2] = 0.0;
        }
        CathConfig.g[0] = 0.0;
        CathConfig.g[1] = 0.0;
        CathConfig.g[2] = 0.0;
    }

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

    // Actuation: u = [0, 0, 0]
    double u[3] = {0.0, 0.0, 0.0};

    // Open CSV file
    std::ofstream csv(csv_filename);
    csv << std::setprecision(17);
    csv << "Li,status,p_tip_x,p_tip_y,p_tip_z,deltau0_x,deltau0_y,deltau0_z,z_error,fingerprint\n";

    // Sweep Li from 0 to total_length in steps
    double step = 10.0;  // 10mm steps
    int num_points = 0;
    bool all_pass = true;
    double max_x_error = 0.0, max_y_error = 0.0, max_z_error = 0.0;

    std::cout << "\nRunning Li sweep (step=" << step << " mm)...\n";
    std::cout << std::setprecision(17);

    for (double Li = 0.0; Li <= total_length + 0.1; Li += step) {
        // Clamp to total_length
        if (Li > total_length) Li = total_length;

        EquilibriumResult result;
        std::memset(&result, 0, sizeof(EquilibriumResult));

        int status = equilibrium_forward(u, Li, FKParams, result);

        double z_error = result.p_tip[2] - Li;

        // Generate fingerprint
        std::string fp = generate_fingerprint(
            param_file, config_file, Li, CathParams, CathConfig, straight_mode
        );

        // Store result
        SweepResult sr;
        sr.Li = Li;
        sr.status = status;
        for (int i = 0; i < 3; i++) {
            sr.p_tip[i] = result.p_tip[i];
            sr.deltau0[i] = result.deltau0[i];
        }
        sr.z_error = z_error;
        sr.fingerprint = fp;
        results.push_back(sr);

        // Write to CSV
        csv << Li << ","
            << status << ","
            << result.p_tip[0] << ","
            << result.p_tip[1] << ","
            << result.p_tip[2] << ","
            << result.deltau0[0] << ","
            << result.deltau0[1] << ","
            << result.deltau0[2] << ","
            << z_error << ","
            << "\"" << fp << "\"\n";

        // Print to console (every 5 points)
        if (num_points % 5 == 0 || Li == 0.0 || Li == total_length) {
            std::cout << "Li=" << std::setw(7) << Li
                      << " | status=" << status
                      << " | p_tip=[" << std::setw(12) << result.p_tip[0]
                      << ", " << std::setw(12) << result.p_tip[1]
                      << ", " << std::setw(12) << result.p_tip[2]
                      << "] | z_err=" << std::setw(12) << z_error << "\n";
        }

        // Check straight rod criteria in straight mode
        if (straight_mode) {
            double tol = 1e-9;
            double x_err = std::abs(result.p_tip[0]);
            double y_err = std::abs(result.p_tip[1]);
            double z_err_abs = std::abs(z_error);

            max_x_error = std::max(max_x_error, x_err);
            max_y_error = std::max(max_y_error, y_err);
            max_z_error = std::max(max_z_error, z_err_abs);

            if (x_err > tol || y_err > tol || z_err_abs > tol) {
                all_pass = false;
            }
        }

        num_points++;

        // Stop if we've reached total_length
        if (Li >= total_length - 0.01) break;
    }

    csv.close();
    std::cout << "\nWrote " << num_points << " data points to " << csv_filename << "\n";

    // Straight mode acceptance criteria
    if (straight_mode) {
        std::cout << "\n--- STRAIGHT ROD ACCEPTANCE CRITERIA ---\n";
        std::cout << "Tolerance: 1e-9\n";
        std::cout << "Max errors across all Li:\n";
        std::cout << "  |x|_max   = " << max_x_error << "\n";
        std::cout << "  |y|_max   = " << max_y_error << "\n";
        std::cout << "  |z-Li|_max = " << max_z_error << "\n";

        if (all_pass) {
            std::cout << "\n✓ PASS: All points satisfy p_tip ≈ [0,0,Li] within tolerance\n";
        } else {
            std::cout << "\n✗ FAIL: Some points exceed tolerance\n";
        }
    }
}

int main(int argc, char* argv[]) {
    std::cout << "========================================\n";
    std::cout << "Li SWEEP REGRESSION AUDIT\n";
    std::cout << "========================================\n";

    // Parse mode from command line
    bool baseline_mode = true;
    bool straight_mode = true;

    if (argc > 1) {
        std::string arg1(argv[1]);
        if (arg1 == "--baseline-only") {
            straight_mode = false;
        } else if (arg1 == "--straight-only") {
            baseline_mode = false;
        } else if (arg1 == "--both") {
            // Run both (default)
        } else {
            std::cout << "Usage: " << argv[0] << " [--baseline-only | --straight-only | --both]\n";
            std::cout << "  Default: --both\n";
            return 1;
        }
    }

    const char* param_file = "./catheterdata/CatheterParameterSet_1_dyn.txt";
    const char* config_file = "./catheterdata/CatheterSpatialConfiguration_1.txt";

    std::vector<SweepResult> baseline_results;
    std::vector<SweepResult> straight_results;

    // Run baseline mode
    if (baseline_mode) {
        run_sweep(param_file, config_file, false, "li_sweep_baseline.csv", baseline_results);
    }

    // Run straight mode
    if (straight_mode) {
        run_sweep(param_file, config_file, true, "li_sweep_straight.csv", straight_results);
    }

    // Summary
    std::cout << "\n========================================\n";
    std::cout << "AUDIT SUMMARY\n";
    std::cout << "========================================\n";

    if (baseline_mode) {
        std::cout << "Baseline mode: " << baseline_results.size() << " points\n";
        std::cout << "  CSV: li_sweep_baseline.csv\n";
        std::cout << "  Interpretation: Non-straight due to ustar + gravity\n";
    }

    if (straight_mode) {
        std::cout << "Straight mode: " << straight_results.size() << " points\n";
        std::cout << "  CSV: li_sweep_straight.csv\n";
        std::cout << "  Acceptance: Check for p_tip ≈ [0,0,Li]\n";
    }

    std::cout << "\nParameter files:\n";
    std::cout << "  " << param_file << "\n";
    std::cout << "  " << config_file << "\n";

    return 0;
}
