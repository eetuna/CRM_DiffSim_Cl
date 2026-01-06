#pragma once

#include "CRM.hpp"
#include "CRMDYN.hpp"
#include "CRM_MatrixOperations.hpp"
#include "CRM_MatrixOperations_Templates.hpp"
#include "CRM_BVPJacobian.hpp"
#include <cmath>

namespace CRMCatheterModel {

// Templated dynamics functions for forward-mode automatic differentiation
// These functions propagate derivatives from control inputs (u) through the dynamics
// to compute exact J_yu = ∂r/∂u via dual number arithmetic

// Time step for coil dynamics integration (from CoilDynamics_Defs.cpp)
constexpr double t_step = 0.001;
constexpr double EPS = 1e-9;

// Sensitivity propagation stability:
// For Dual-number forward-mode AD, explicit integrators (RK2/ABM4) can be unstable
// for stiff coil dynamics. We use finer substepping for sensitivity evaluation only.
// Criterion: Subdivide by factor k to keep sensitivity stable without changing forward value path.
// Based on rotational dynamics time scale: tau ~ sqrt(I_max / k_magnetic) ~ 0.01s
// With nominal t_step = 0.001s, we use k=5 to balance stability improvement
// against numerical error accumulation from excessive substepping.
constexpr int DUAL_SUBSTEP_FACTOR = 5;

// Forward declarations
template<typename T>
void CoilIntegrad_T(const T in_twist[6], const T in_n[3], const double g[3],
                    const T R[9], double actMass, const double actInertia[9],
                    const double damping[6], const double in_B0[3],
                    const T in_muhat[9], const T in_mL[3], T twistdot[6]);

template<typename T>
void DYNSE3_TimeSpace_T(const T in_R_n[9], const T in_p_n[3], double h,
                        const T in_twist_n[6], T out_R_np1[9], T out_p_np1[3]);

template<typename T>
void RK2_coildyn_T(const T in_x_n[NUM_COIL_STATES], const T in_n[3], const double g[3],
                   double actMass, const double actInertia[9], const double damping[6],
                   const double in_B0[3], const T in_muhat[9], const T in_mL[3],
                   T out_x_np1[NUM_COIL_STATES], T out_xdot_n[6]);

template<typename T>
void ABM4_coildyn_T(const T in_x_n[NUM_COIL_STATES], const T in_xdot_nm1[6],
                    const T in_xdot_nm2[6], const T in_xdot_nm3[6],
                    const T in_x_nm1[NUM_COIL_STATES], const T in_x_nm2[NUM_COIL_STATES],
                    const T in_x_nm3[NUM_COIL_STATES],
                    const T in_n[3], const double g[3], double actMass,
                    const double actInertia[9], const double damping[6],
                    const double in_B0[3], const T in_muhat[9], const T in_mL[3],
                    T out_x_np1[NUM_COIL_STATES], T out_xdot_n[6]);

// ============================================================================
// CoilIntegrad_T: Compute coil dynamics integrand (Euler equations)
// ============================================================================
template<typename T>
void CoilIntegrad_T(const T in_twist[6], const T in_n[3], const double g[3],
                    const T R[9], double actMass, const double actInertia[9],
                    const double damping[6], const double in_B0[3],
                    const T in_muhat[9], const T in_mL[3], T twistdot[6]) {

    T v[3], w[3], n_L[3], m_L[3], B0_T[3], muhat[9];
    T RTg[3];
    T w_v[3], w_hat[9], inertiaw[3], w_inertia_w[3], diff_tau_w[3];
    T vdot[3], wdot[3], Tb[3], tau[3];
    T RscTB0[3];

    // Extract components from twist
    for (int i = 0; i < 3; ++i) {
        v[i] = in_twist[i];
        w[i] = in_twist[i+3];
    }

    // Copy inputs
    for (int i = 0; i < 3; ++i) {
        n_L[i] = in_n[i];
        m_L[i] = in_mL[i];
        B0_T[i] = T(in_B0[i]);  // Convert to T type
    }
    for (int i = 0; i < 9; ++i) {
        muhat[i] = in_muhat[i];
    }

    T g_T[3];
    for (int i = 0; i < 3; ++i) {
        g_T[i] = T(g[i]);
    }
    mMult_ATB_T<T,3,3,1>(R, g_T, RTg);

    // Compute skew-symmetric matrix from w
    wHat_T<T>(w, w_hat);

    // w × v
    mMult_AB_T<T,3,3,1>(w_hat, v, w_v);

    // Linear acceleration: vdot = R^T*g - nL/m - w×v - damping*v
    T damping_vec[3];
    for (int i = 0; i < 3; ++i) {
        damping_vec[i] = T(damping[i]) * v[i];
    }

    for (int i = 0; i < 3; ++i) {
        vdot[i] = RTg[i] - n_L[i] / actMass - w_v[i] - damping_vec[i];
    }

    // Angular acceleration: wdot = I^{-1} * (τ_mag - mL - w×(I*w) - damping*w)

    // Compute I*w (inertia is diagonal)
    T inertiaw_temp[3];
    inertiaw_temp[0] = T(actInertia[0]) * w[0];
    inertiaw_temp[1] = T(actInertia[4]) * w[1];
    inertiaw_temp[2] = T(actInertia[8]) * w[2];

    // For template compatibility, store in 3-element array
    for (int i = 0; i < 3; ++i) {
        inertiaw[i] = inertiaw_temp[i];
    }

    // Compute w × (I*w)
    T w_hat_temp[9];
    wHat_T<T>(w, w_hat_temp);
    mMult_AB_T<T,3,3,1>(w_hat_temp, inertiaw, w_inertia_w);

    // Compute magnetic torque: τ_mag = muhat * R^T * B0
    mMult_ATB_T<T,3,3,1>(R, B0_T, RscTB0);  // R^T * B0

    mMult_AB_T<T,3,3,1>(muhat, RscTB0, Tb);  // Tb = muhat * R^T * B0

    // Net torque: tau = τ_mag - mL
    mSub_AB_T<T,3,1>(Tb, m_L, tau);

    // Angular damping
    T damping_wec[3], residual_w[3];
    for (int i = 0; i < 3; ++i) {
        damping_wec[i] = T(damping[i+3]) * w[i];
    }

    // τ - w×(I*w)
    mSub_AB_T<T,3,1>(tau, w_inertia_w, diff_tau_w);

    // (τ - w×(I*w)) - damping*w
    mSub_AB_T<T,3,1>(diff_tau_w, damping_wec, residual_w);

    // wdot = I^{-1} * residual_w (inertia is diagonal)
    wdot[0] = residual_w[0] / T(actInertia[0]);
    wdot[1] = residual_w[1] / T(actInertia[4]);
    wdot[2] = residual_w[2] / T(actInertia[8]);

    // Pack output
    for (int i = 0; i < 3; ++i) {
        twistdot[i] = vdot[i];
        twistdot[i+3] = wdot[i];
    }
}

template<typename T>
void Rodrigues_T(const T axis[3], const T theta, T out_Rdelta[9]) {
    T axis_hat[9];
    wHat_T(axis, axis_hat);

    T axis_hat2[9];
    mMult_AB_T<T, 3, 3, 3>(axis_hat, axis_hat, axis_hat2);

    T term1[9], term2[9];
    T sin_theta = sin(theta);
    T cos_theta = cos(theta);

    mMult_sA_T<T, 3, 3>(sin_theta, axis_hat, term1);
    mMult_sA_T<T, 3, 3>(T(1.0) - cos_theta, axis_hat2, term2);

    T I[9] = {T(1.0), T(0.0), T(0.0),
              T(0.0), T(1.0), T(0.0),
              T(0.0), T(0.0), T(1.0)};

    mAdd_ABC_T<T, 3, 3>(I, term1, term2, out_Rdelta);
}

// ============================================================================
// DYNSE3_TimeSpace_T: Analytical SE(3) integration
// ============================================================================
template<typename T>
void DYNSE3_TimeSpace_T(const T in_R_n[9], const T in_p_n[3], double h,
                        const T in_twist_n[6], T out_R_np1[9], T out_p_np1[3]) {
    T R_n[9];
    T p_n[3];
    T v_n[3], w_n[3];

    mCopy_AB_T<T, 9>(in_R_n, R_n);
    for (int i = 0; i < 3; ++i) p_n[i] = in_p_n[i];

    for (int i = 0; i < 3; ++i) {
        v_n[i] = in_twist_n[i];
        w_n[i] = in_twist_n[i + 3];
    }

    // Compute p_dot = R_n * v_n
    T p_dot[3];
    mMult_AB_T<T, 3, 3, 1>(R_n, v_n, p_dot);

    // Compute ||w||^2
    T umagsq = vNormSq_T<T, 3>(w_n);

    bool is_zero = false;
    if constexpr (std::is_same_v<T, double>) {
        is_zero = (umagsq == T(0.0));
    } else {
        is_zero = (umagsq.val == 0.0);
    }

    if (is_zero) {
        mCopy_AB_T<T, 9>(R_n, out_R_np1);
    } else {
        T umag = sqrt(umagsq);
        T umagresp = T(1.0) / umag;

        T unorm[3];
        for (int i = 0; i < 3; ++i) {
            unorm[i] = umagresp * w_n[i];
        }

        T delsumag = T(h) * umag;
        T Rdelta[9];
        Rodrigues_T(unorm, delsumag, Rdelta);

        // R_np1 = R_n * Rdelta
        mMult_AB_T<T, 3, 3, 3>(R_n, Rdelta, out_R_np1);
    }

    for (int i = 0; i < 3; ++i) {
        out_p_np1[i] = p_n[i] + p_dot[i] * T(h);
    }
}

// ============================================================================
// RK2_coildyn_T: Second-order Runge-Kutta integrator for coil dynamics
// ============================================================================
template<typename T>
void RK2_coildyn_T(const T in_x_n[NUM_COIL_STATES], const T in_n[3], const double g[3],
                   double actMass, const double actInertia[9], const double damping[6],
                   const double in_B0[3], const T in_muhat[9], const T in_mL[3],
                   T out_x_np1[NUM_COIL_STATES], T out_xdot_n[6], double h = t_step) {

    T nL[3], muhat[9], mL[3], twist_n[6];
    T R_n[9];
    T p_n[3];
    T k1[6], k2oh[6], x_n_p_k1o2[6], xdot_n[6];

    // Extract state components
    for (int i = 0; i < 3; ++i) p_n[i] = in_x_n[i+6];  // Keep position derivatives
    for (int i = 0; i < 9; ++i) R_n[i] = in_x_n[i+9];
    for (int i = 0; i < 6; ++i) twist_n[i] = in_x_n[i];

    for (int i = 0; i < 3; ++i) {
        nL[i] = in_n[i];
        mL[i] = in_mL[i];
    }
    for (int i = 0; i < 9; ++i) {
        muhat[i] = in_muhat[i];
    }

    // RK2 Step 1: Compute k1 = h * f(x_n)
    CoilIntegrad_T<T>(twist_n, nL, g, R_n, actMass, actInertia, damping, in_B0, muhat, mL, xdot_n);

    for (int i = 0; i < 6; ++i) {
        k1[i] = h * xdot_n[i];
        x_n_p_k1o2[i] = twist_n[i] + k1[i] * 0.5;
    }

    // Analytical SE(3) step for R and p at half-step
    T R_np1half[9];
    T p_np1half[3];
    DYNSE3_TimeSpace_T<T>(R_n, p_n, h*0.5, twist_n, R_np1half, p_np1half);

    // RK2 Step 2: Compute k2 = h * f(x_n + k1/2)
    CoilIntegrad_T<T>(x_n_p_k1o2, nL, g, R_np1half, actMass, actInertia, damping, in_B0, muhat, mL, k2oh);

    // Final update: x_{n+1} = x_n + h * k2
    for (int i = 0; i < 6; ++i) {
        out_x_np1[i] = twist_n[i] + h * k2oh[i];
    }

    // Analytical SE(3) step for R and p at full step
    T R_np1[9];
    T p_np1[3];
    DYNSE3_TimeSpace_T<T>(R_n, p_n, h, x_n_p_k1o2, R_np1, p_np1);

    // Pack position and rotation into output state
    for (int i = 0; i < 3; ++i) out_x_np1[i+6] = p_np1[i];
    for (int i = 0; i < 9; ++i) out_x_np1[i+9] = R_np1[i];

    // Return derivative at current state
    for (int i = 0; i < 6; ++i) {
        out_xdot_n[i] = xdot_n[i];
    }
}

// ============================================================================
// ABM4_coildyn_T: Adams-Bashforth-Moulton 4th order integrator
// ============================================================================
template<typename T>
void ABM4_coildyn_T(const T in_x_n[NUM_COIL_STATES], const T in_xdot_nm1[6],
                    const T in_xdot_nm2[6], const T in_xdot_nm3[6],
                    const T in_x_nm1[NUM_COIL_STATES], const T in_x_nm2[NUM_COIL_STATES],
                    const T in_x_nm3[NUM_COIL_STATES],
                    const T in_n[3], const double g[3], double actMass,
                    const double actInertia[9], const double damping[6],
                    const double in_B0[3], const T in_muhat[9], const T in_mL[3],
                    T out_x_np1[NUM_COIL_STATES], T out_xdot_n[6], double h = t_step) {

    const double P_COEFF_N = 55.0/24.0;
    const double P_COEFF_Nm1 = -59.0/24.0;
    const double P_COEFF_Nm2 = 37.0/24.0;
    const double P_COEFF_Nm3 = -9.0/24.0;
    const double C_COEFF_Np1 = 9.0/24.0;
    const double C_COEFF_N = 19.0/24.0;
    const double C_COEFF_Nm1 = -5.0/24.0;
    const double C_COEFF_Nm2 = 1.0/24.0;

    T twist_nm1[6], twist_nm2[6], twist_nm3[6], twist_n[6];
    T R_n[9];
    T p_n[3];
    T x_np1_hat[6], xdot_np1_hat[6], xdot_n[6];
    T xdot_nm1[6], xdot_nm2[6], xdot_nm3[6];

    // Extract twists from states
    for (int i = 0; i < 6; ++i) {
        twist_nm1[i] = in_x_nm1[i];
        twist_nm2[i] = in_x_nm2[i];
        twist_nm3[i] = in_x_nm3[i];
        twist_n[i] = in_x_n[i];
        xdot_nm1[i] = in_xdot_nm1[i];
        xdot_nm2[i] = in_xdot_nm2[i];
        xdot_nm3[i] = in_xdot_nm3[i];
    }

    for (int i = 0; i < 3; ++i) p_n[i] = in_x_n[i+6];  // Keep position derivatives
    for (int i = 0; i < 9; ++i) R_n[i] = in_x_n[i+9];

    T nL[3], muhat[9], mL[3];
    for (int i = 0; i < 3; ++i) {
        nL[i] = in_n[i];
        mL[i] = in_mL[i];
    }
    for (int i = 0; i < 9; ++i) {
        muhat[i] = in_muhat[i];
    }

    // Compute xdot_n
    CoilIntegrad_T<T>(twist_n, nL, g, R_n, actMass, actInertia, damping, in_B0, muhat, mL, xdot_n);

    // AB4 Predictor: x_np1_hat = x_n + h * (sum of weighted derivatives)
    for (int i = 0; i < 6; ++i) {
        x_np1_hat[i] = twist_n[i] + h * (
            P_COEFF_N * xdot_n[i] +
            P_COEFF_Nm1 * xdot_nm1[i] +
            P_COEFF_Nm2 * xdot_nm2[i] +
            P_COEFF_Nm3 * xdot_nm3[i]
        );
    }

    // Analytical SE(3) step for predicted state
    T R_np1[9];
    T p_np1[3];
    DYNSE3_TimeSpace_T<T>(R_n, p_n, h, x_np1_hat, R_np1, p_np1);

    // Evaluate derivative at predicted state
    CoilIntegrad_T<T>(x_np1_hat, nL, g, R_np1, actMass, actInertia, damping, in_B0, muhat, mL, xdot_np1_hat);

    // AM4 Corrector: x_np1 = x_n + h * (sum of weighted derivatives including predicted)
    for (int i = 0; i < 6; ++i) {
        out_x_np1[i] = twist_n[i] + h * (
            C_COEFF_Np1 * xdot_np1_hat[i] +
            C_COEFF_N * xdot_n[i] +
            C_COEFF_Nm1 * xdot_nm1[i] +
            C_COEFF_Nm2 * xdot_nm2[i]
        );
    }

    // Pack position and rotation
    for (int i = 0; i < 3; ++i) out_x_np1[i+6] = p_np1[i];
    for (int i = 0; i < 9; ++i) out_x_np1[i+9] = R_np1[i];

    // Return current derivative
    for (int i = 0; i < 6; ++i) {
        out_xdot_n[i] = xdot_n[i];
    }
}

// ============================================================================
// CoilDynamics_T: Main coil dynamics integrator (RK2 + ABM4)
// ============================================================================
template<typename T>
void CoilDynamics_T(const T in_coil_state[NUM_COIL_STATES], const T in_n[3],
                    const double g[3], double actMass, const double actInertia[9],
                    const double damping[6], double DELTA_T, const double in_B0[3],
                    const T in_muhat[9], const T in_mL[3],
                    T out_coil_state[NUM_COIL_STATES], T out_xdot_n[6]) {

    T x_nm3[NUM_COIL_STATES], x_nm2[NUM_COIL_STATES], x_nm1[NUM_COIL_STATES];
    T x_n[NUM_COIL_STATES], x_np1[NUM_COIL_STATES];
    T xdot_nm3[6], xdot_nm2[6], xdot_nm1[6], xdot_n[6];

    T nL[3], muhat[9], mL[3];

    for (int i = 0; i < 3; ++i) {
        nL[i] = in_n[i];
        mL[i] = in_mL[i];
    }
    for (int i = 0; i < 9; ++i) {
        muhat[i] = in_muhat[i];
    }

    // Initialize
    for (int i = 0; i < NUM_COIL_STATES; ++i) {
        x_n[i] = in_coil_state[i];
    }

    // Determine effective time step and number of steps
    // For Dual-number sensitivity propagation, use finer substepping to stabilize derivatives
    // without changing the forward value path (within numerical tolerance)
    double effective_t_step = t_step;
    int N;
    if constexpr (!std::is_same_v<T, double>) {
        effective_t_step = t_step / DUAL_SUBSTEP_FACTOR;
        N = static_cast<int>(std::ceil(DELTA_T / effective_t_step));
    } else {
        N = static_cast<int>(std::ceil(DELTA_T / t_step));
    }

    for (int idx = 0; idx < N; ++idx) {
        if (idx < 3) {
            // RK2 initialization steps
            RK2_coildyn_T<T>(x_n, nL, g, actMass, actInertia, damping, in_B0, muhat, mL, x_np1, xdot_n, effective_t_step);
        } else {
            // ABM4 steps
            ABM4_coildyn_T<T>(x_n, xdot_nm1, xdot_nm2, xdot_nm3,
                              x_nm1, x_nm2, x_nm3,
                              nL, g, actMass, actInertia, damping, in_B0, muhat, mL,
                              x_np1, xdot_n, effective_t_step);
        }

        // Update iteration history
        for (int i = 0; i < NUM_COIL_STATES; ++i) {
            x_nm3[i] = x_nm2[i];
            x_nm2[i] = x_nm1[i];
            x_nm1[i] = x_n[i];
            x_n[i] = x_np1[i];
        }
        for (int i = 0; i < 6; ++i) {
            xdot_nm3[i] = xdot_nm2[i];
            xdot_nm2[i] = xdot_nm1[i];
            xdot_nm1[i] = xdot_n[i];
        }
    }

    // Copy final state
    for (int i = 0; i < NUM_COIL_STATES; ++i) {
        out_coil_state[i] = x_n[i];
    }
    for (int i = 0; i < 6; ++i) {
        out_xdot_n[i] = xdot_n[i];
    }
}

} // namespace CRMCatheterModel
