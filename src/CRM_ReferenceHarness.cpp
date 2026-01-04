#include "CRM_ReferenceHarness.hpp"
#include <cstring>
#include <iostream>

namespace CRMCatheterModel {

CRMDYNReferenceRolloutResult* allocate_rollout_result(int num_steps) {
    CRMDYNReferenceRolloutResult* result = new CRMDYNReferenceRolloutResult();
    result->num_steps = num_steps;

    // Allocate state trajectories
    result->X_traj = new double*[num_steps];
    for (int i = 0; i < num_steps; i++) {
        result->X_traj[i] = new double[NUM_STATES];
    }

    result->X_coil_traj = new double**[num_steps];
    for (int i = 0; i < num_steps; i++) {
        result->X_coil_traj[i] = new double*[NUM_ACT_SET];
        for (int j = 0; j < NUM_ACT_SET; j++) {
            result->X_coil_traj[i][j] = new double[18];
        }
    }

    // Allocate observable trajectories
    result->P_tip_traj = new double*[num_steps];
    for (int i = 0; i < num_steps; i++) {
        result->P_tip_traj[i] = new double[3];
    }

    // Allocate BVP solution trajectories
    result->u0_traj = new double*[num_steps];
    for (int i = 0; i < num_steps; i++) {
        result->u0_traj[i] = new double[3];
    }

    result->nL_traj = new double**[num_steps];
    for (int i = 0; i < num_steps; i++) {
        result->nL_traj[i] = new double*[NUM_ACT_SET];
        for (int j = 0; j < NUM_ACT_SET; j++) {
            result->nL_traj[i][j] = new double[3];
        }
    }

    result->mL_traj = new double**[num_steps];
    for (int i = 0; i < num_steps; i++) {
        result->mL_traj[i] = new double*[NUM_ACT_SET];
        for (int j = 0; j < NUM_ACT_SET; j++) {
            result->mL_traj[i][j] = new double[3];
        }
    }

    result->ftip_traj = new double*[num_steps];
    for (int i = 0; i < num_steps; i++) {
        result->ftip_traj[i] = new double[3];
    }

    // Allocate diagnostics
    result->converged = new int[num_steps];
    result->localmin = new int[num_steps];

    return result;
}

void free_rollout_result(CRMDYNReferenceRolloutResult* result) {
    if (result == nullptr) return;

    // Free state trajectories
    if (result->X_traj != nullptr) {
        for (int i = 0; i < result->num_steps; i++) {
            delete[] result->X_traj[i];
        }
        delete[] result->X_traj;
    }

    if (result->X_coil_traj != nullptr) {
        for (int i = 0; i < result->num_steps; i++) {
            if (result->X_coil_traj[i] != nullptr) {
                for (int j = 0; j < NUM_ACT_SET; j++) {
                    delete[] result->X_coil_traj[i][j];
                }
                delete[] result->X_coil_traj[i];
            }
        }
        delete[] result->X_coil_traj;
    }

    // Free observable trajectories
    if (result->P_tip_traj != nullptr) {
        for (int i = 0; i < result->num_steps; i++) {
            delete[] result->P_tip_traj[i];
        }
        delete[] result->P_tip_traj;
    }

    // Free BVP solution trajectories
    if (result->u0_traj != nullptr) {
        for (int i = 0; i < result->num_steps; i++) {
            delete[] result->u0_traj[i];
        }
        delete[] result->u0_traj;
    }

    if (result->nL_traj != nullptr) {
        for (int i = 0; i < result->num_steps; i++) {
            if (result->nL_traj[i] != nullptr) {
                for (int j = 0; j < NUM_ACT_SET; j++) {
                    delete[] result->nL_traj[i][j];
                }
                delete[] result->nL_traj[i];
            }
        }
        delete[] result->nL_traj;
    }

    if (result->mL_traj != nullptr) {
        for (int i = 0; i < result->num_steps; i++) {
            if (result->mL_traj[i] != nullptr) {
                for (int j = 0; j < NUM_ACT_SET; j++) {
                    delete[] result->mL_traj[i][j];
                }
                delete[] result->mL_traj[i];
            }
        }
        delete[] result->mL_traj;
    }

    if (result->ftip_traj != nullptr) {
        for (int i = 0; i < result->num_steps; i++) {
            delete[] result->ftip_traj[i];
        }
        delete[] result->ftip_traj;
    }

    // Free diagnostics
    delete[] result->converged;
    delete[] result->localmin;

    delete result;
}

int crmdyn_reference_rollout(
    const double x0_coil[NUM_ACT_SET][18],
    const double x0_tip[NUM_STATES],
    const double* u_seq,
    int num_steps,
    double dt,
    const CRMCatheterModelParams& CathParams,
    const CatheterConfiguration& CathConfig,
    double InsertedLength,
    ContactModeType ContactMode,
    const double TipConstraintPoint_in[3],
    const double TipForce_in[3],
    double IntegrationStepSize,
    const double ActInertia_in[NUM_ACT_SET][9],
    const double damping_in[NUM_ACT_SET][6],
    bool use_warmstart,
    CRMDYNReferenceRolloutResult* result
) {
    // Create non-const copies for API compatibility
    double TipConstraintPoint[3];
    double TipForce[3];
    double ActInertia[NUM_ACT_SET][9];
    double damping[NUM_ACT_SET][6];

    std::memcpy(TipConstraintPoint, TipConstraintPoint_in, 3 * sizeof(double));
    std::memcpy(TipForce, TipForce_in, 3 * sizeof(double));
    for (int i = 0; i < NUM_ACT_SET; i++) {
        std::memcpy(ActInertia[i], ActInertia_in[i], 9 * sizeof(double));
        std::memcpy(damping[i], damping_in[i], 6 * sizeof(double));
    }
    // Storage for localization markers (required by IVP solver but not used here)
    double (*ReportedMarkerPos)[3] = new double[CathParams.no_locmarkers][3];

    // Working state variables (will be updated each step)
    double xf_pre[NUM_STATES];
    double v_L_pre[NUM_ACT_SET][3];
    double w_L_pre[NUM_ACT_SET][3];
    double pL[NUM_ACT_SET][3];
    double RL[NUM_ACT_SET][9];

    // Initial guesses for BVP solver (will be updated with warm-start)
    double nL_initialguess[NUM_ACT_SET][3];
    double mL_initialguess[NUM_ACT_SET][3];
    double ftip_initialguess[3] = {0.0, 0.0, 0.0};
    double deltau0_initialguess[3] = {0.0, 0.0, 0.0};

    // Initialize from x0
    std::memcpy(xf_pre, x0_tip, NUM_STATES * sizeof(double));
    for (int j = 0; j < NUM_ACT_SET; j++) {
        for (int i = 0; i < 3; i++) {
            v_L_pre[j][i] = x0_coil[j][i];
            w_L_pre[j][i] = x0_coil[j][i + 3];
            pL[j][i] = x0_coil[j][i + 6];
            nL_initialguess[j][i] = 0.0;
            mL_initialguess[j][i] = 0.0;
        }
        for (int i = 0; i < 9; i++) {
            RL[j][i] = x0_coil[j][i + 9];
        }
    }

    // Store initial state (step 0)
    std::memcpy(result->X_traj[0], x0_tip, NUM_STATES * sizeof(double));
    for (int j = 0; j < NUM_ACT_SET; j++) {
        std::memcpy(result->X_coil_traj[0][j], x0_coil[j], 18 * sizeof(double));
    }
    std::memcpy(result->P_tip_traj[0], &x0_tip[0], 3 * sizeof(double));

    // Initialize diagnostics for step 0
    result->converged[0] = 0;
    result->localmin[0] = 0;
    for (int i = 0; i < 3; i++) {
        result->u0_traj[0][i] = 0.0;
        result->ftip_traj[0][i] = 0.0;
    }
    for (int j = 0; j < NUM_ACT_SET; j++) {
        for (int i = 0; i < 3; i++) {
            result->nL_traj[0][j][i] = 0.0;
            result->mL_traj[0][j][i] = 0.0;
        }
    }

    // Run rollout for num_steps-1 timesteps (to get num_steps total states)
    for (int step = 0; step < num_steps - 1; step++) {
        // Extract actuation for this step
        double ActuationCurrents[NUM_ACT_SET][3];
        for (int j = 0; j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                ActuationCurrents[j][i] = u_seq[(step * NUM_ACT_SET + j) * 3 + i];
            }
        }

        // Construct BVP parameters (exactly as in CRMDYN_test.cpp)
        CRMShootingMethodParams BVPParams = CRMDYNConstructShootingMethodParamSet(
            CathParams, CathConfig, InsertedLength,
            ActuationCurrents, ContactMode,
            TipConstraintPoint, TipForce,
            IntegrationStepSize, ActInertia,
            v_L_pre, w_L_pre, pL, RL, damping,
            dt
        );

        // Output storage for BVP
        double out_u0[3];
        double out_nL[NUM_ACT_SET][3];
        double out_mL[NUM_ACT_SET][3];
        double out_tau[NUM_ACT_SET][3];
        double ftip_calc[3];
        int localmin;

        // Solve BVP (exactly as in CRMDYN_test.cpp line 262)
        DynamicsBVP(BVPParams, xf_pre, mL_initialguess, nL_initialguess, ftip_initialguess,
                    out_u0, out_mL, out_nL, out_tau, ftip_calc, localmin);

        // Output storage for IVP
        double xf[NUM_STATES];
        double x_coil[NUM_ACT_SET][18];

        // Solve IVP (exactly as in CRMDYN_test.cpp line 277)
        DYNSolverIVP(BVPParams, out_u0, out_mL, out_nL, out_tau, ftip_calc,
                     true, xf, x_coil, ReportedMarkerPos);

        // Store results for step+1
        int next_step = step + 1;
        std::memcpy(result->X_traj[next_step], xf, NUM_STATES * sizeof(double));
        for (int j = 0; j < NUM_ACT_SET; j++) {
            std::memcpy(result->X_coil_traj[next_step][j], x_coil[j], 18 * sizeof(double));
        }
        std::memcpy(result->P_tip_traj[next_step], &xf[0], 3 * sizeof(double));

        // Store BVP solution for diagnostics
        std::memcpy(result->u0_traj[next_step], out_u0, 3 * sizeof(double));
        std::memcpy(result->ftip_traj[next_step], ftip_calc, 3 * sizeof(double));
        for (int j = 0; j < NUM_ACT_SET; j++) {
            std::memcpy(result->nL_traj[next_step][j], out_nL[j], 3 * sizeof(double));
            std::memcpy(result->mL_traj[next_step][j], out_mL[j], 3 * sizeof(double));
        }
        result->converged[next_step] = (localmin == 0) ? 0 : 1;
        result->localmin[next_step] = localmin;

        // Update state for next iteration (exactly as in CRMDYN_test.cpp lines 291-311)
        for (int j = 0; j < NUM_ACT_SET; j++) {
            for (int i = 0; i < 3; i++) {
                v_L_pre[j][i] = x_coil[j][i];
                w_L_pre[j][i] = x_coil[j][i + 3];
                pL[j][i] = x_coil[j][i + 6];
            }
            for (int i = 0; i < 9; i++) {
                RL[j][i] = x_coil[j][i + 9];
            }

            // Warm-start for next iteration (if enabled)
            if (use_warmstart) {
                for (int i = 0; i < 3; i++) {
                    mL_initialguess[j][i] = out_mL[j][i];
                    nL_initialguess[j][i] = out_nL[j][i];
                }
            }
        }

        for (int i = 0; i < NUM_STATES; i++) {
            xf_pre[i] = xf[i];
        }
    }

    delete[] ReportedMarkerPos;
    return 0;
}

} // namespace CRMCatheterModel
