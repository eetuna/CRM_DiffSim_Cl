#ifndef CRM_REFERENCE_HARNESS_HPP
#define CRM_REFERENCE_HARNESS_HPP

#include "CRM.hpp"
#include "CRMDYN.hpp"

namespace CRMCatheterModel {

/**
 * Deterministic reference harness for CRMDYN_test.cpp regression testing.
 *
 * Runs multi-step rollout using the exact same logic as CRMDYN_test.cpp
 * to provide ground-truth trajectories for validating Python wrappers.
 */
struct CRMDYNReferenceRolloutResult {
    // Multi-step trajectories (T+1 steps including initial state)
    int num_steps;                              // T+1

    // State trajectories
    double** X_traj;                            // [T+1][NUM_STATES] - tip states
    double*** X_coil_traj;                      // [T+1][NUM_ACT_SET][18] - coil states

    // Observable trajectories
    double** P_tip_traj;                        // [T+1][3] - tip positions

    // BVP solution trajectories (for diagnostics)
    double** u0_traj;                           // [T+1][3] - base curvatures
    double*** nL_traj;                          // [T+1][NUM_ACT_SET][3] - interface forces
    double*** mL_traj;                          // [T+1][NUM_ACT_SET][3] - interface moments
    double** ftip_traj;                         // [T+1][3] - tip forces

    // Convergence diagnostics
    int* converged;                             // [T+1] - 0=success, >0=failed
    int* localmin;                              // [T+1] - BVP solver exit codes
};

/**
 * Allocate memory for rollout result structure.
 */
CRMDYNReferenceRolloutResult* allocate_rollout_result(int num_steps);

/**
 * Free memory for rollout result structure.
 */
void free_rollout_result(CRMDYNReferenceRolloutResult* result);

/**
 * Run deterministic multi-step rollout matching CRMDYN_test.cpp logic.
 *
 * This function replicates the exact stepping sequence from CRMDYN_test.cpp:
 * - DynamicsBVP to solve for u0, nL, mL, tau, ftip
 * - DYNSolverIVP to integrate forward one timestep
 * - Warm-start from previous iteration
 *
 * @param x0_coil Initial coil states [NUM_ACT_SET][18]: [v, w, p, R]
 * @param x0_tip Initial tip state [NUM_STATES]: [p, R, u]
 * @param u_seq Actuation sequence [num_steps][NUM_ACT_SET][3] - currents (A)
 * @param num_steps Number of timesteps to simulate (T)
 * @param dt Timestep (seconds)
 * @param CathParams Catheter physical parameters
 * @param CathConfig Catheter spatial configuration
 * @param InsertedLength Insertion length (mm)
 * @param ContactMode Contact mode (FREE_TIP or FIXED_TIP)
 * @param TipConstraintPoint Tip constraint point [3] (used if FIXED_TIP)
 * @param TipForce External tip force [3] (used if FREE_TIP)
 * @param IntegrationStepSize Integration stepsize (mm)
 * @param ActInertia Actuator inertia matrices [NUM_ACT_SET][9]
 * @param damping Damping coefficients [NUM_ACT_SET][6]
 * @param use_warmstart If true, carry warm-start from previous step
 * @param result Output structure (must be pre-allocated)
 *
 * @return 0 on success, non-zero on failure
 */
int crmdyn_reference_rollout(
    const double x0_coil[NUM_ACT_SET][18],
    const double x0_tip[NUM_STATES],
    const double* u_seq,                        // [num_steps, NUM_ACT_SET, 3] row-major
    int num_steps,
    double dt,
    const CRMCatheterModelParams& CathParams,
    const CatheterConfiguration& CathConfig,
    double InsertedLength,
    ContactModeType ContactMode,
    const double TipConstraintPoint[3],
    const double TipForce[3],
    double IntegrationStepSize,
    const double ActInertia[NUM_ACT_SET][9],
    const double damping[NUM_ACT_SET][6],
    bool use_warmstart,
    CRMDYNReferenceRolloutResult* result
);

} // namespace CRMCatheterModel

#endif // CRM_REFERENCE_HARNESS_HPP
