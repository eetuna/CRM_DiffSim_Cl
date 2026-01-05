#pragma once

#include "CRM.hpp"
#include "CRMDYN.hpp"
#include "CRM_MatrixOperations.hpp"
#include "CRM_MatrixOperations_Templates.hpp"
#include "CRMDYN_Numerical_Integration.hpp"
#include "CoilDynamics_Defs_Templates.hpp"
#include "CRM_BVPJacobian.hpp"
#include <cmath>

namespace CRMCatheterModel {

// ============================================================================
// Templated flexible segment integration for forward-mode AD
// ============================================================================

// Flexible segment backward integration - templated version
template<typename T>
void CRMFlexible_IVP_Back_T(
    int SegmentIndex,
    const T in_p[3],             // Boundary p carries Dual derivatives
    const T in_R[9],             // Boundary R carries Dual derivatives
    DYNNLEqnParams_T<T>& in_params,
    const T in_u[3],             // Curvature carries Dual
    const T in_n_L[3],           // Force carries Dual (from mL, nL)
    T out_u[3],                  // Output curvature carries Dual
    T out_p[3],                  // Output p carries Dual derivatives
    T out_R[9]                   // Output R carries Dual derivatives
) {
    T n_L[3];
    for (int i = 0; i < 3; ++i) {
        n_L[i] = in_n_L[i];
    }

    double h;
    double l_zero[3] = {0.0, 0.0, 0.0};
    int fsegno; // flexible segment no

    // We will copy anything we will access more than once (or write to) to local variables
    int NextLocMarker = in_params.NextLocMarker;

    // For others, we will create aliases
    auto& SegBounds = in_params.SegBounds;
    auto& SegSteps = in_params.SegSteps;
    auto& InsertedLength = in_params.InsertedLength;
    auto& dlambdainv = in_params.dlambdainv;
    auto& K = in_params.K;
    auto& Kinv = in_params.Kinv;
    auto& ustar = in_params.ustar;
    auto& fcumlambda = in_params.fcumlambda;
    auto& FinalValueOnly = in_params.FinalValueOnly;
    auto& LocMarkers = in_params.LocMarkers;
    auto& p_atLocMarkers = in_params.p_atLocMarkers;
    auto& no_locmarkers = in_params.no_locmarkers;

    // Flexible Segment - Prepare the CRMIntegrand Parameters
    fsegno = SegmentIndex >> 1; // i/2, flexible segment no

    // Apply substepping for Dual-number sensitivity propagation stability
    int base_steps = SegSteps[fsegno];
    int N_steps;
    if constexpr (!std::is_same_v<T, double>) {
        N_steps = base_steps * DUAL_SUBSTEP_FACTOR;
    } else {
        N_steps = base_steps;
    }
    h = -1 * (SegBounds[SegmentIndex + 1] - SegBounds[SegmentIndex]) / (N_steps * 1.0);

    StateVector_T<T> xi_statevec;  // Initial value of the state for the next segment to be integrated
    StateVector_T<T> xf_statevec;  // Final value of the state for the next segment to be integrated

    // Copy initial state
    mCopy_AB_T<T, 9>(in_R, xi_statevec._R);  // R is T
    for (int i = 0; i < 3; ++i) {
        xi_statevec._p[i] = T(in_p[i]);  // Convert double p to T (or T to T)
        xi_statevec._u[i] = in_u[i];     // u is already type T
    }

    double DeltaPE = 0.0;
    auto& CalculateEnergy = in_params.CalculateEnergy;
    auto& no_fcum_steps = in_params.no_fcum_steps;
    auto& g = in_params.g;
    auto& rho = in_params.rho;
    double ftip[3] = {0.0, 0.0, 0.0}; // placeholder

    // Define the structure that will be used to pass parameters to the CRMIntegrand
    CRMIntegrandParams IntegrandParams;
    // These parameters are same for all segments
    IntegrandParams.dlambdainv = dlambdainv;
    IntegrandParams.Li = InsertedLength;
    IntegrandParams.l = l_zero;  // We are assuming the distributed moment on the catheter body is zero
    IntegrandParams.no_fcum_steps = no_fcum_steps;
    IntegrandParams.fcumlambda = fcumlambda;
    IntegrandParams.ftip = ftip;
    IntegrandParams.g = g;

    // Assign the parameters that vary from segment to segment
    IntegrandParams.K = K[fsegno];
    IntegrandParams.Kinv = Kinv[fsegno];
    IntegrandParams.ustar = ustar[fsegno];
    IntegrandParams.rho = rho[SegmentIndex];

    // Integrate using templated ABM4_dyn with CRMIntegrand_dyn_T
    // ABM4_dyn is already templated and will use CRMIntegrand_dyn_T internally
    // We need to create a wrapper that calls CRMIntegrand_dyn_T for the templated state vector

    // Call ABM4_dyn with StateVector_T<T> type
    // NOTE: This requires ABM4_dyn to call the correct integrand function
    // Since ABM4_dyn is templated on StVecType, we need to ensure it calls the right integrand

    // For now, we'll manually integrate using the pattern from ABM4_dyn
    // but call CRMIntegrand_dyn_T instead of CRMIntegrand_dyn

    // Actually, looking at ABM4_dyn template, it directly calls the integrand as a hardcoded call
    // We need to make ABM4_dyn flexible to call either CRMIntegrand_dyn or CRMIntegrand_dyn_T
    // For now, let's inline the ABM4 logic with explicit calls to CRMIntegrand_dyn_T

    using _SVT = StateVector_T<T>;
    using _DVT = StateDerivativeVector_T<T>;

    _SVT x_n(xi_statevec);
    _SVT x_np1;
    _SVT x_nm1, x_nm2, x_nm3;
    double t_n = SegBounds[SegmentIndex + 1];
    _DVT xdot_n;
    _DVT xdot_nm1, xdot_nm2, xdot_nm3;

    int N = N_steps;  // Use substepped count for Dual

    // RK2 initialization for first 3 steps
    for (int idx = 0; idx < (N < 3 ? N : 3); idx++) {
        // RK2 step
        _DVT k1, k2;
        _SVT x_mid;

        CRMIntegrand_dyn_T<T>(t_n, x_n, IntegrandParams, n_L, k1);

        x_mid = x_n + (h / 2.0) * k1;
        // R doesn't change in derivative (handled analytically) - x_mid R is just copy of x_n R?
        // Wait, for RK2 midpoint, we should use R at midpoint.
        // DYNSE3_TimeSpace_T calculates R_np1.
        // For midpoint, we need DYNSE3_TimeSpace_T with h/2.
        
        T twist_n[6];
        for(int i=0; i<3; ++i) {
             twist_n[i] = x_n._u[i]; // v part, but we typically assume v=0 or use u as w?
             // In flexible segment, u is curvature (w). v is zero (strain).
             // But here xi_statevec._u is passed as in_u (curvature).
             // Wait, the state vector for flexible segment has u (curvature).
             // The twist v component is 0 for pure bending? Or strain?
             // "u is curvature carries Dual".
             // In standard cosserat rod, v is shear/axial strain.
             // If we assume inextensible and unshearable, v = [0,0,1].
             // The manual code did:
             // twist[i+3] = x_np1._u[i]; // w = u
             // pdot = R * [0;0;1]
             // This implies v = [0,0,1].
        }

        // We need to match the manual logic:
        // pdot = R * [0;0;1]
        // Rdot = R * u_hat
        
        // My DYNSE3_TimeSpace_T assumes twist has v and w.
        // So I need to construct the full twist.
        
        // Let's refine the loop logic.
        
        T twist_n_full[6];
        for(int i=0; i<3; ++i) twist_n_full[i] = T(0.0); // v = 0 ??
        twist_n_full[2] = T(1.0); // v = [0,0,1] for local tangent direction
        
        for(int i=0; i<3; ++i) twist_n_full[i+3] = x_n._u[i]; // w = u

        T R_mid[9], p_mid[3];
        DYNSE3_TimeSpace_T<T>(x_n._R, x_n._p, h/2.0, twist_n_full, R_mid, p_mid);
        
        // Update x_mid R and p
        for(int i=0; i<9; ++i) x_mid._R[i] = R_mid[i];
        for(int i=0; i<3; ++i) x_mid._p[i] = p_mid[i];

        CRMIntegrand_dyn_T<T>(t_n + h / 2.0, x_mid, IntegrandParams, n_L, k2);

        x_np1 = x_n + h * k2;

        // Full step SE(3)
        // We need to use the twist from the first step? Or average?
        // Standard RK2 for Lie groups usually does:
        // x_mid = x_n * exp(h/2 * twist_n)
        // x_np1 = x_n * exp(h * twist_mid)
        // Here we are integrating xdot = f(x).
        // The state includes u. u is integrated via ABM4.
        // R and p are integrated via DYNSE3.
        
        // The manual code used x_np1._u for twist!
        // "twist[i + 3] = x_np1._u[i]; // w = u"
        // And "x_np1 = x_n + h * k2;" (so u is updated)
        // Then it used this updated u to update R and p.
        // This is effectively: u_{n+1} computed first, then R_{n+1}, p_{n+1} computed using u_{n+1}.
        // This is a semi-implicit Euler or symplectic Euler style for R/p?
        
        // "twist[i + 3] = x_np1._u[i];"
        
        T twist_np1_full[6];
        for(int i=0; i<3; ++i) twist_np1_full[i] = T(0.0);
        twist_np1_full[2] = T(1.0);
        for(int i=0; i<3; ++i) twist_np1_full[i+3] = x_np1._u[i]; // Use updated u
        
        DYNSE3_TimeSpace_T<T>(x_n._R, x_n._p, h, twist_np1_full, x_np1._R, x_np1._p);

        xdot_n = k2;

        // Save history...
        if (idx == 0) { xdot_nm3 = xdot_n; x_nm3 = x_n; }
        else if (idx == 1) { xdot_nm2 = xdot_nm3; x_nm2 = x_nm3; xdot_nm3 = xdot_n; x_nm3 = x_n; }
        else if (idx == 2) { xdot_nm1 = xdot_nm2; x_nm1 = x_nm2; xdot_nm2 = xdot_nm3; x_nm2 = x_nm3; xdot_nm3 = xdot_n; x_nm3 = x_n; }

        t_n = t_n + h;
        x_n = x_np1;
    }

    // ABM4 steps for remaining
    for (int idx = 3; idx < N; idx++) {
        // ABM4 predictor
        x_np1 = x_n + (h / 24.0) * (55.0 * xdot_n + (-59.0) * xdot_nm1 + 37.0 * xdot_nm2 + (-9.0) * xdot_nm3);

        // Compute derivative at predicted point
        // We need predicted R and p?
        // The manual code didn't calculate predicted R/p for the derivative evaluation.
        // It just passed x_np1. But x_np1._R and _p were NOT updated in the predictor line (only u was).
        // So it used x_n._R and x_n._p (old values) inside CRMIntegrand_dyn_T?
        // Wait, x_np1 = x_n + ... updates everything in StateVector_T.
        // But R and p are not part of "StateVector" in the same way?
        // StateVector_T operator+ adds _u and _p. _R is just added.
        // But xdot has u derivatives. Does it have p derivatives?
        // CRMIntegrand_dyn_T returns out_xdot which is StateDerivativeVector_T.
        // StateDerivativeVector_T only has _u!
        // So x_np1._p and x_np1._R are NOT updated by the predictor line!
        // They remain copies of x_n._p and x_n._R (because operator+ copies R from 'this', which is x_n).
        
        // So CRMIntegrand_dyn_T was called with old R and p?
        // Manual code:
        // CRMIntegrand_dyn_T<T>(t_n + h, x_np1, IntegrandParams, n_L, xdot_np1_pred);
        // Yes, using old R and p.
        
        _DVT xdot_np1_pred;
        CRMIntegrand_dyn_T<T>(t_n + h, x_np1, IntegrandParams, n_L, xdot_np1_pred);

        // ABM4 corrector
        x_np1 = x_n + (h / 24.0) * (9.0 * xdot_np1_pred + 19.0 * xdot_n + (-5.0) * xdot_nm1 + xdot_nm2);
        
        // Now update R and p using updated u
        T twist_np1_full[6];
        for(int i=0; i<3; ++i) twist_np1_full[i] = T(0.0);
        twist_np1_full[2] = T(1.0);
        for(int i=0; i<3; ++i) twist_np1_full[i+3] = x_np1._u[i];
        
        DYNSE3_TimeSpace_T<T>(x_n._R, x_n._p, h, twist_np1_full, x_np1._R, x_np1._p);

        // Update history
        xdot_nm1 = xdot_n; x_nm1 = x_n;
        xdot_nm2 = xdot_nm1; x_nm2 = x_nm1;
        xdot_nm3 = xdot_nm2; x_nm3 = x_nm2;

        CRMIntegrand_dyn_T<T>(t_n + h, x_np1, IntegrandParams, n_L, xdot_n);

        t_n = t_n + h;
        x_n = x_np1;
    }

    xf_statevec = x_n;

    // Extract outputs - keep Dual derivatives
    for (int j = 0; j < 3; j++) {
        out_u[j] = xf_statevec._u[j];
        out_p[j] = xf_statevec._p[j];
    }
    for (int j = 0; j < 9; ++j) {
        out_R[j] = xf_statevec._R[j];
    }
}

// ============================================================================
// Templated DYNNLEqnParams for forward-mode AD
// ============================================================================
template<typename T>
struct DYNNLEqnParams_T {
    // Parameters that stay double
    const double* SegBounds;
    const int* SegSteps;
    double InsertedLength;
    double dlambdainv;
    const double (*K)[9];
    const double (*Kinv)[9];
    const double (*ustar)[3];
    const double (*fcumlambda)[3];
    bool FinalValueOnly;
    const double* LocMarkers;
    double (*p_atLocMarkers)[3];
    int no_locmarkers;
    int NextLocMarker;
    bool CalculateEnergy;
    int no_fcum_steps;
    const double* g;
    const double* rho;
    const double* TipForce;
    const double* ActMass;
    const double (*actInertia)[9];
    const double (*damping)[6];
    double DELTA_T;
    const double* B0;
    int no_segments;

    // Parameters that need to be T (state inputs for BVP)
    T v_L_pre[NUM_ACT_SET][3];
    T w_L_pre[NUM_ACT_SET][3];
    T p_pre[NUM_ACT_SET][3];
    T R_pre[NUM_ACT_SET][9];
    T xf[12]; // p_f and R_f
    T xi[12]; // p_i and R_i

    // Constructor/Converter
    DYNNLEqnParams_T(DYNNLEqnParams& params) {
        // Copy pointers
        SegBounds = params.SegBounds;
        SegSteps = params.SegSteps;
        InsertedLength = params.InsertedLength;
        dlambdainv = params.dlambdainv;
        K = params.K;
        Kinv = params.Kinv;
        ustar = params.ustar;
        fcumlambda = params.fcumlambda;
        FinalValueOnly = params.FinalValueOnly;
        LocMarkers = params.LocMarkers;
        p_atLocMarkers = params.p_atLocMarkers;
        no_locmarkers = params.no_locmarkers;
        NextLocMarker = params.NextLocMarker;
        CalculateEnergy = params.CalculateEnergy;
        no_fcum_steps = params.no_fcum_steps;
        g = params.g;
        rho = params.rho;
        TipForce = params.TipForce;
        ActMass = params.ActMass;
        actInertia = params.actInertia;
        damping = params.damping;
        DELTA_T = params.DELTA_T;
        B0 = params.B0;
        no_segments = params.no_segments;

        // Convert state inputs to T
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < 3; ++i) {
                v_L_pre[j][i] = T(params.v_L_pre[j][i]);
                w_L_pre[j][i] = T(params.w_L_pre[j][i]);
                p_pre[j][i] = T(params.p_pre[j][i]);
            }
            for (int i = 0; i < 9; ++i) {
                R_pre[j][i] = T(params.R_pre[j][i]);
            }
        }
        for (int i = 0; i < 12; ++i) {
            xf[i] = T(params.xf[i]);
            xi[i] = T(params.xi[i]);
        }
    }
};

// ============================================================================
// Templated DYNNLEquation for forward-mode AD
// ============================================================================

// Forward declarations
template<typename T>
void CoilDynamics_T(const T in_x_n[NUM_COIL_STATES], const T in_n[3], const double g[3],
                    double actMass, const double actInertia[9], const double damping[6],
                    double DELTA_T, const double in_B0[3], const T in_muhat[9],
                    const T in_mL[3], T out_x_np1[NUM_COIL_STATES], T out_xdot[6]);

template<typename T>
void DYNNLEquation_T(const double in_x[], T out_y[], DYNNLEqnParams_T<T>& Params,
                     const T muhat[NUM_ACT_SET][9], T out_u0[3], T out_tau[NUM_ACT_SET*3]);

// Old signature for double-only BVP solver (still needed?)
// Actually we can just update DYNNLEquation_T to take templated Params
// But to avoid breaking existing code in minpack, we might need an adapter.
// However, the task is to replace heuristic J_yx.
// So we will use the templated version for Jacobian computation.
// The BVP solver itself (minpack) uses double.
// We can overload DYNNLEquation_T or just template the params argument.

template<typename T>
void DYNNLEquation_T(const double in_x[], T out_y[], DYNNLEqnParams& Params,
                     const T muhat[NUM_ACT_SET][9], T out_u0[3], T out_tau[NUM_ACT_SET*3]) {
    // Adapter for legacy DYNNLEqnParams (used by double-only calls)
    DYNNLEqnParams_T<T> Params_T(Params);
    DYNNLEquation_T(in_x, out_y, Params_T, muhat, out_u0, out_tau);
}

// The real implementation taking templated params
template<typename T>
void DYNNLEquation_T(const double in_x[], T out_y[], DYNNLEqnParams_T<T>& Params,
                     const T muhat[NUM_ACT_SET][9], T out_u0[3], T out_tau[NUM_ACT_SET*3]) {


    // === DEBUG: Trace muhat derivatives at function entry ===
    if constexpr (!std::is_same_v<T, double>) {
        std::cout << "\n[DYNNLEquation_T] Entry - muhat derivatives:" << std::endl;
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            std::cout << "  Coil " << j << ": ";
            double max_deriv = 0.0;
            for (int i = 0; i < 9; ++i) {
                max_deriv = std::max(max_deriv, std::abs(muhat[j][i].deriv));
            }
            std::cout << "max|deriv|=" << max_deriv << std::endl;
        }
    }

    // output for time advance, not used in BVP, just placeholders
    T m_L[NUM_ACT_SET][3], n_L[NUM_ACT_SET][3];
    T n_0[3];

    // Scale and convert input parameters from double to T
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; i++) {
            m_L[j][i] = T(IVALUE_SCALE_M * in_x[i + j*6]);
            n_L[j][i] = T(IVALUE_SCALE_N * in_x[i + j*6 + 3]);
        }
    }

    for (int i = 0; i < 3; i++) {
        n_0[i] = T(Params.TipForce[i]);
    }

    T x_coil[NUM_ACT_SET][NUM_COIL_STATES], out_x_coil[NUM_ACT_SET][NUM_COIL_STATES];
    double actMass[NUM_ACT_SET], actInertia[NUM_ACT_SET][9];

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            x_coil[j][i] = T(Params.v_L_pre[j][i]);
            x_coil[j][i+3] = T(Params.w_L_pre[j][i]);
            x_coil[j][i+6] = T(Params.p_pre[j][i]);
        }
        for (int i = 0; i < 9; ++i) {
            x_coil[j][i+9] = T(Params.R_pre[j][i]);
        }

        actMass[j] = Params.ActMass[j];
        for (int i = 0; i < 9; ++i) {
            actInertia[j][i] = Params.actInertia[j][i];
        }
    }

    auto& K = Params.K;
    auto& Kinv = Params.Kinv;
    auto& ustar = Params.ustar;
    auto& SegBounds = Params.SegBounds;

    T net_mL[3], tau[3], K2invResidual[3], u_t[3], du[3];

    double tau_0[3] = {0.0, 0.0, 0.0};
    T p_t[3];
    T R_t[9];
    T u_tau[3];
    T p_[3];
    T R_[9];
    int fsegi, actno, actseg, actno_mn;
    T u_L[3], u_f[3];
    T p_f[3];  // Carries Dual derivatives from coil position through flexible segment
    T R_f[9];
    T out_xdot[6];
    T residual[NUM_ACT_SET][6];

    T p_L[3];  // Carries Dual derivatives from coil position
    T R_L[9];
    T v1[3], v2[3], v3[3], v_val[3];

    // root configurations for residual
    T p_d[3], R_d[9];
    for (int i = 0; i < 3; ++i) {
        p_d[i] = Params.xi[i];
    }
    for (int i = 0; i < 9; ++i) {
        R_d[i] = Params.xi[3+i];
    }

    double RigidSegmentLength;
    T net_nL[3];

    auto& NUM_SEGMENTS = Params.no_segments;

    for (int segi = NUM_SEGMENTS-1; segi >= 0; --segi) { // starting from the last segment
        if (segi % 2 == 0) {
            fsegi = segi >> 1;

            if (segi == NUM_SEGMENTS-1) { // last segment is flexible
                // assume free tip no torque tau_0 = [0,0,0]
                mMult_AB_T<T, 3, 3, 1>(Kinv[fsegi], tau_0, K2invResidual);
                mAdd_AB_T<T, 3, 1>(ustar[fsegi], K2invResidual, u_t);

                for (int i = 0; i < 3; ++i) {
                    p_t[i] = T(Params.xf[i]);
                }
                for (int i = 0; i < 9; ++i) {
                    R_t[i] = Params.xf[3+i];
                }

                CRMFlexible_IVP_Back_T<T>(segi, p_t, R_t, Params, u_t, n_0,
                                          u_tau, p_, R_);

                mSub_AB_T<T, 3, 1>(u_tau, ustar[fsegi], du);
                mMult_AB_T<T, 3, 3, 1>(K[fsegi], du, tau); // moment at upper side of the coil

                actno = fsegi - 1;

                mSub_AB_T<T, 3, 1>(m_L[actno], tau, net_mL); // the net torque applied to the downwards coil

                // copy to the output for forward calculation
                for (int i = 0; i < 3; ++i) {
                    out_tau[i+actno*3] = tau[i];
                }

            } else { // the non-free tip flexible segments with coils on top

                actno = fsegi;
                mMult_AB_T<T, 3, 3, 1>(Kinv[fsegi], m_L[actno], K2invResidual);
                mAdd_AB_T<T, 3, 1>(ustar[fsegi], K2invResidual, u_L);

                actseg = segi + 1; // should be the upper actuator

                // calculate starting positions and rotations given last coil states
                RigidSegmentLength = SegBounds[actseg+1] - SegBounds[actseg];

                // Extract R_L (rotation is double, no derivatives)
                for (int i = 0; i < 9; ++i) {
                    if constexpr (std::is_same_v<T, double>) {
                        R_L[i] = out_x_coil[actno][i+9];
                    } else {
                        R_L[i] = out_x_coil[actno][i+9].val;
                    }
                }

                // Extract p_L (position carries derivatives!)
                for (int i = 0; i < 3; ++i) {
                    if constexpr (std::is_same_v<T, double>) {
                        p_L[i] = out_x_coil[actno][i+6] - R_L[i*3+2] * RigidSegmentLength * 0.5;
                    } else {
                        // KEEP DERIVATIVES: p_L should be type T, not double
                        p_L[i] = out_x_coil[actno][i+6] - T(R_L[i*3+2] * RigidSegmentLength * 0.5);
                    }
                }

                // Call flexible integration with full Dual support
                CRMFlexible_IVP_Back_T<T>(segi, p_L, R_L, Params, u_L, n_L[actno],
                                          u_f, p_f, R_f);

                actno = fsegi - 1;

                if (actno > -1) { // if there is a coil linked below, we need to calculate the net torque again
                    mSub_AB_T<T, 3, 1>(u_f, ustar[fsegi], du);
                    mMult_AB_T<T, 3, 3, 1>(K[fsegi], du, tau); // calculate moment at upper side of the coil

                    mSub_AB_T<T, 3, 1>(m_L[actno], tau, net_mL); // net torque applied at the downwards coil

                    for (int i = 0; i < 3; ++i) {
                        out_tau[i+actno*3] = tau[i];
                    }
                }
            }
        }
        else {
            actno = (segi - 1) >> 1;

            if (actno < NUM_ACT_SET-1) {
                mSub_AB_T<T, 3, 1>(n_L[actno], n_L[actno+1], net_nL);
            } else {
                mSub_AB_T<T, 3, 1>(n_L[actno], n_0, net_nL);
            }

            CoilDynamics_T<T>(x_coil[actno], net_nL, Params.g, actMass[actno], actInertia[actno],
                              Params.damping[actno], Params.DELTA_T, Params.B0,
                              muhat[actno], net_mL, out_x_coil[actno], out_xdot);

            // compute the residual against the last flexible segment above
            if (actno < NUM_ACT_SET-1) {
                RigidSegmentLength = SegBounds[segi+1] - SegBounds[segi];

                // indices for the residual
                actno_mn = actno + 1;

                // Extract R_L (rotation is T, carries derivatives)
                for (int i = 0; i < 9; ++i) {
                    R_L[i] = out_x_coil[actno][i+9];
                }

                // Extract p_L (position carries derivatives!)
                for (int i = 0; i < 3; ++i) {
                    p_L[i] = out_x_coil[actno][i+6] + R_L[i*3+2] * RigidSegmentLength * 0.5;
                }

                // === DEBUG: Trace p_L derivatives before residual computation ===
                if constexpr (!std::is_same_v<T, double>) {
                    std::cout << "  [Residual] p_L derivatives for actno=" << actno << ": ";
                    double max_deriv = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        max_deriv = std::max(max_deriv, std::abs(p_L[i].deriv));
                    }
                    std::cout << "max|deriv|=" << max_deriv << std::endl;
                }

                // Compute residual: p_f - p_L (both carry derivatives!)
                for (int i = 0; i < 3; ++i) {
                    residual[actno_mn][i] = p_f[i] - p_L[i];
                }

                // === DEBUG: Trace residual derivatives after computation ===
                if constexpr (!std::is_same_v<T, double>) {
                    std::cout << "  [Residual] residual[" << actno_mn << "] derivatives: ";
                    double max_deriv = 0.0;
                    for (int i = 0; i < 3; ++i) {
                        max_deriv = std::max(max_deriv, std::abs(residual[actno_mn][i].deriv));
                    }
                    std::cout << "max|deriv|=" << max_deriv << std::endl;
                }
                for (int i = 0; i < 3; ++i) {
                    v1[i] = T(R_f[i*3]) - T(R_L[i*3]);
                    v2[i] = T(R_f[1+i*3]) - T(R_L[1+i*3]);
                    v3[i] = T(R_f[2+i*3]) - T(R_L[2+i*3]);
                }

                v_val[0] = vNormSq_T<T, 3>(v1);
                v_val[1] = vNormSq_T<T, 3>(v2);
                v_val[2] = vNormSq_T<T, 3>(v3);
                for (int i = 0; i < 3; ++i) {
                    // CODEX STEP 1: Remove singularity by using 0.5 * squared norm
                    if constexpr (std::is_same_v<T, double>) {
                        residual[actno_mn][i+3] = 0.5 * v_val[i];
                    } else {
                        residual[actno_mn][i+3] = T(0.5) * v_val[i];
                    }
                }
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        out_u0[i] = u_f[i];
    }

    // last segment (root boundary condition)
    actno = 0;
    for (int i = 0; i < 3; ++i) {
        residual[actno][i] = p_f[i] - T(p_d[i]);
    }
    // calculate vector norm of the rotation matrices
    for (int i = 0; i < 3; ++i) {
        v1[i] = T(R_f[i*3]) - T(R_d[i*3]);
        v2[i] = T(R_f[1+i*3]) - T(R_d[1+i*3]);
        v3[i] = T(R_f[2+i*3]) - T(R_d[2+i*3]);
    }

    v_val[0] = vNormSq_T<T, 3>(v1);
    v_val[1] = vNormSq_T<T, 3>(v2);
    v_val[2] = vNormSq_T<T, 3>(v3);
    for (int i = 0; i < 3; ++i) {
        // CODEX STEP 1: Remove singularity by using 0.5 * squared norm
        if constexpr (std::is_same_v<T, double>) {
            residual[actno][i+3] = 0.5 * v_val[i];
        } else {
            residual[actno][i+3] = T(0.5) * v_val[i];
        }
    }

    // === DEBUG: Trace derivatives in residuals before packing ===
    if constexpr (!std::is_same_v<T, double>) {
        std::cout << "[DYNNLEquation_T] Residual derivatives before packing:" << std::endl;
        for (int i = 0; i < NUM_ACT_SET; ++i) {
            std::cout << "  Coil " << i << " residual: ";
            double max_deriv = 0.0;
            for (int j = 0; j < 6; ++j) {
                max_deriv = std::max(max_deriv, std::abs(residual[i][j].deriv));
            }
            std::cout << "max|deriv|=" << max_deriv << std::endl;
        }
    }

    // Pack residuals into output (extract .val for double output)
    for (int i = 0; i < NUM_ACT_SET; ++i) {
        for (int j = 0; j < 6; ++j) {
            out_y[j + i*6] = residual[i][j] * (j < 3 ? RESIDUAL_SCALE_P : RESIDUAL_SCALE_R);
        }
    }

    // === DEBUG: Trace derivatives in out_y after packing ===
    if constexpr (!std::is_same_v<T, double>) {
        std::cout << "[DYNNLEquation_T] out_y derivatives after packing:" << std::endl;
        for (int i = 0; i < NUM_ACT_SET; ++i) {
            std::cout << "  Coil " << i << " out_y: ";
            double max_deriv = 0.0;
            for (int j = 0; j < 6; ++j) {
                max_deriv = std::max(max_deriv, std::abs(out_y[j + i*6].deriv));
            }
            std::cout << "max|deriv|=" << max_deriv << std::endl;
        }
    }
}

// ============================================================================
// DYNNLEquation_YY_T: Fully templated version for J_yy computation
// ============================================================================
// This version accepts templated in_x to enable forward-mode AD seeding of
// y = [mL; nL] for computing J_yy = ∂r/∂y via Dual numbers.
//
// Key difference from DYNNLEquation_T: in_x is templated to preserve derivatives
template<typename T>
void DYNNLEquation_YY_T(const T in_x[], T out_y[], DYNNLEqnParams_T<T>& Params,
                         const double muhat_double[NUM_ACT_SET][9], T out_u0[3], T out_tau[NUM_ACT_SET*3]) {

    // output for time advance, not used in BVP, just placeholders
    T m_L[NUM_ACT_SET][3], n_L[NUM_ACT_SET][3];
    T n_0[3];

    // Scale input parameters - preserve template type T for derivatives
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; i++) {
            m_L[j][i] = in_x[i + j*6] * T(IVALUE_SCALE_M);
            n_L[j][i] = in_x[i + j*6 + 3] * T(IVALUE_SCALE_N);
        }
    }

    for (int i = 0; i < 3; i++) {
        n_0[i] = T(Params.TipForce[i]);
    }

    // Convert muhat to type T
    T muhat[NUM_ACT_SET][9];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 9; ++i) {
            muhat[j][i] = T(muhat_double[j][i]);
        }
    }

    T x_coil[NUM_ACT_SET][NUM_COIL_STATES], out_x_coil[NUM_ACT_SET][NUM_COIL_STATES];
    double actMass[NUM_ACT_SET], actInertia[NUM_ACT_SET][9];

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            x_coil[j][i] = T(Params.v_L_pre[j][i]);
            x_coil[j][i+3] = T(Params.w_L_pre[j][i]);
            x_coil[j][i+6] = T(Params.p_pre[j][i]);
        }
        for (int i = 0; i < 9; ++i) {
            x_coil[j][i+9] = T(Params.R_pre[j][i]);
        }

        actMass[j] = Params.ActMass[j];
        for (int i = 0; i < 9; ++i) {
            actInertia[j][i] = Params.actInertia[j][i];
        }
    }

    auto& K = Params.K;
    auto& Kinv = Params.Kinv;
    auto& ustar = Params.ustar;
    auto& SegBounds = Params.SegBounds;

    T net_mL[3], tau[3], K2invResidual[3], u_t[3], du[3];

    double tau_0[3] = {0.0, 0.0, 0.0};
    T p_t[3];
    T R_t[9];
    T u_tau[3];
    T p_[3];
    T R_[9];
    int fsegi, actno, actseg, actno_mn;
    T u_L[3], u_f[3];
    T p_f[3];
    T R_f[9];
    T out_xdot[6];
    T residual[NUM_ACT_SET][6];

    T p_L[3];
    T R_L[9];
    T v1[3], v2[3], v3[3], v_val[3];

    // root configurations for residual
    T p_d[3], R_d[9];
    for (int i = 0; i < 3; ++i) {
        p_d[i] = Params.xi[i];
    }
    for (int i = 0; i < 9; ++i) {
        R_d[i] = Params.xi[3+i];
    }

    double RigidSegmentLength;
    T net_nL[3];

    auto& NUM_SEGMENTS = Params.no_segments;

    for (int segi = NUM_SEGMENTS-1; segi >= 0; --segi) {
        if (segi % 2 == 0) {
            fsegi = segi >> 1;

            if (segi == NUM_SEGMENTS-1) {
                mMult_AB_T<T, 3, 3, 1>(Kinv[fsegi], tau_0, K2invResidual);
                mAdd_AB_T<T, 3, 1>(ustar[fsegi], K2invResidual, u_t);

                for (int i = 0; i < 3; ++i) {
                    p_t[i] = T(Params.xf[i]);
                }
                for (int i = 0; i < 9; ++i) {
                    R_t[i] = Params.xf[3+i];
                }

                CRMFlexible_IVP_Back_T<T>(segi, p_t, R_t, Params, u_t, n_0,
                                          u_tau, p_, R_);

                mSub_AB_T<T, 3, 1>(u_tau, ustar[fsegi], du);
                mMult_AB_T<T, 3, 3, 1>(K[fsegi], du, tau);

                actno = fsegi - 1;

                mSub_AB_T<T, 3, 1>(m_L[actno], tau, net_mL);

                for (int i = 0; i < 3; ++i) {
                    out_tau[i+actno*3] = tau[i];
                }

            } else {

                actno = fsegi;
                mMult_AB_T<T, 3, 3, 1>(Kinv[fsegi], m_L[actno], K2invResidual);
                mAdd_AB_T<T, 3, 1>(ustar[fsegi], K2invResidual, u_L);

                actseg = segi + 1;

                RigidSegmentLength = SegBounds[actseg+1] - SegBounds[actseg];

                for (int i = 0; i < 9; ++i) {
                    R_L[i] = out_x_coil[actno][i+9];
                }

                for (int i = 0; i < 3; ++i) {
                    p_L[i] = out_x_coil[actno][i+6] - R_L[i*3+2] * RigidSegmentLength * 0.5;
                }

                CRMFlexible_IVP_Back_T<T>(segi, p_L, R_L, Params, u_L, n_L[actno],
                                          u_f, p_f, R_f);

                actno = fsegi - 1;

                if (actno > -1) {
                    mSub_AB_T<T, 3, 1>(u_f, ustar[fsegi], du);
                    mMult_AB_T<T, 3, 3, 1>(K[fsegi], du, tau);

                    mSub_AB_T<T, 3, 1>(m_L[actno], tau, net_mL);

                    for (int i = 0; i < 3; ++i) {
                        out_tau[i+actno*3] = tau[i];
                    }
                }
            }
        }
        else {
            actno = (segi - 1) >> 1;

            if (actno < NUM_ACT_SET-1) {
                mSub_AB_T<T, 3, 1>(n_L[actno], n_L[actno+1], net_nL);
            } else {
                mSub_AB_T<T, 3, 1>(n_L[actno], n_0, net_nL);
            }

            CoilDynamics_T<T>(x_coil[actno], net_nL, Params.g, actMass[actno], actInertia[actno],
                              Params.damping[actno], Params.DELTA_T, Params.B0,
                              muhat[actno], net_mL, out_x_coil[actno], out_xdot);

            if (actno < NUM_ACT_SET-1) {
                RigidSegmentLength = SegBounds[segi+1] - SegBounds[segi];

                actno_mn = actno + 1;

                for (int i = 0; i < 9; ++i) {
                    R_L[i] = out_x_coil[actno][i+9];
                }

                for (int i = 0; i < 3; ++i) {
                    p_L[i] = out_x_coil[actno][i+6] + R_L[i*3+2] * RigidSegmentLength * 0.5;
                }

                for (int i = 0; i < 3; ++i) {
                    residual[actno_mn][i] = p_f[i] - p_L[i];
                }

                for (int i = 0; i < 3; ++i) {
                    v1[i] = T(R_f[i*3]) - T(R_L[i*3]);
                    v2[i] = T(R_f[1+i*3]) - T(R_L[1+i*3]);
                    v3[i] = T(R_f[2+i*3]) - T(R_L[2+i*3]);
                }

                v_val[0] = vNormSq_T<T, 3>(v1);
                v_val[1] = vNormSq_T<T, 3>(v2);
                v_val[2] = vNormSq_T<T, 3>(v3);
                for (int i = 0; i < 3; ++i) {
                    // CODEX STEP 1: Remove singularity by using 0.5 * squared norm
                    if constexpr (std::is_same_v<T, double>) {
                        residual[actno_mn][i+3] = 0.5 * v_val[i];
                    } else {
                        residual[actno_mn][i+3] = T(0.5) * v_val[i];
                    }
                }
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        out_u0[i] = u_f[i];
    }

    // last segment (root boundary condition)
    actno = 0;
    for (int i = 0; i < 3; ++i) {
        residual[actno][i] = p_f[i] - T(p_d[i]);
    }
    for (int i = 0; i < 3; ++i) {
        v1[i] = T(R_f[i*3]) - T(R_d[i*3]);
        v2[i] = T(R_f[1+i*3]) - T(R_d[1+i*3]);
        v3[i] = T(R_f[2+i*3]) - T(R_d[2+i*3]);
    }

    v_val[0] = vNormSq_T<T, 3>(v1);
    v_val[1] = vNormSq_T<T, 3>(v2);
    v_val[2] = vNormSq_T<T, 3>(v3);
    for (int i = 0; i < 3; ++i) {
        // CODEX STEP 1: Remove singularity by using 0.5 * squared norm
        if constexpr (std::is_same_v<T, double>) {
            residual[actno][i+3] = 0.5 * v_val[i];
        } else {
            residual[actno][i+3] = T(0.5) * v_val[i];
        }
    }

    // Pack residuals into output
    for (int i = 0; i < NUM_ACT_SET; ++i) {
        for (int j = 0; j < 6; ++j) {
            out_y[j + i*6] = residual[i][j] * (j < 3 ? RESIDUAL_SCALE_P : RESIDUAL_SCALE_R);
        }
    }
}

} // namespace CRMCatheterModel
