#pragma once

#include "CRM.hpp"
#include "CRMDYN.hpp"
#include "CRM_MatrixOperations.hpp"
#include "CRM_MatrixOperations_Templates.hpp"
#include "CRMDYN_Numerical_Integration.hpp"
#include "CoilDynamics_Defs_Templates.hpp"
#include "CRM_BVPJacobian.hpp"
#include <cmath>

// Define constants needed for array sizes
#define MAX_SEGMENTS 50
#define MAX_LOC_MARKERS 20
#define MAX_FCUM_STEPS 2000

namespace CRMCatheterModel {

// ============================================================================
// Templated Parameter Structure for forward-mode AD
// ============================================================================

template <typename T>
struct DYNNLEqnParams_T {
    int32_t no_flex_seg;
    int32_t no_rigid_seg;
    int32_t no_act_set;
    int32_t no_segments;
    int32_t no_locmarkers;
    int32_t no_fcum_steps;
    
    // Intermediate variables (mirrors CRMIVPCoreParams)
    double xi[NUM_STATES];
    double InsertedLength;
    double dlambdainv;
    double B0[3];
    double g[3];
    
    // Pointers
    CatheterSegmentType* SegmentTypes;
    int32_t *FlexActIndex;
    int32_t StartSegmentIndex;
    double *SegBounds;
    int32_t *SegSteps;
    double *rho;
    double (*K)[9];
    double (*Kinv)[9];
    double (*ustar)[3];
    double (*ActMass);
    double (*MagMoment)[3];
    double (*CoilAlignmentTurnAreaMatrix)[9];
    double (*R_atActuators)[9];
    double (*p_atActuators)[3];
    
    bool CalculateEnergy;
    bool FinalValueOnly;
    int NextLocMarker;
    double *LocMarkers;
    double (*p_atLocMarkers)[3];
    double (*fcumlambda)[3];

    // Dynamics parameters (Templated)
    T v_L_pre[NUM_ACT_SET][3];
    T w_L_pre[NUM_ACT_SET][3];
    T p_pre[NUM_ACT_SET][3];
    T R_pre[NUM_ACT_SET][9];
    
    double actInertia[NUM_ACT_SET][9];
    double damping[NUM_ACT_SET][6];
    double DELTA_T;
    double m_L[NUM_ACT_SET][3];
    double n_L[NUM_ACT_SET][3];

    // DYNNLEqnParams specific
    ContactModeType ContactMode;
    double TipConstraintPoint[3];
    double TipForce[3];
    double ftip_initialguess[3];
    T xf[NUM_STATES]; // Templated Tip state
};

template <typename T>
void convert_to_params_t(const DYNNLEqnParams& in_params, DYNNLEqnParams_T<T>& out_params) {
    out_params.no_flex_seg = in_params.no_flex_seg;
    out_params.no_rigid_seg = in_params.no_rigid_seg;
    out_params.no_act_set = in_params.no_act_set;
    out_params.no_segments = in_params.no_segments;
    out_params.no_locmarkers = in_params.no_locmarkers;
    out_params.no_fcum_steps = in_params.no_fcum_steps;

    mCopy_AB<NUM_STATES>(in_params.xi, out_params.xi);
    out_params.InsertedLength = in_params.InsertedLength;
    out_params.dlambdainv = in_params.dlambdainv;
    mCopy_AB<3>(in_params.B0, out_params.B0);
    mCopy_AB<3>(in_params.g, out_params.g);

    out_params.SegmentTypes = in_params.SegmentTypes;
    out_params.FlexActIndex = in_params.FlexActIndex;
    out_params.StartSegmentIndex = in_params.StartSegmentIndex;
    out_params.SegBounds = in_params.SegBounds;
    out_params.SegSteps = in_params.SegSteps;
    out_params.rho = in_params.rho;
    out_params.K = in_params.K;
    out_params.Kinv = in_params.Kinv;
    out_params.ustar = in_params.ustar;
    out_params.ActMass = in_params.ActMass;
    out_params.MagMoment = in_params.MagMoment;
    out_params.CoilAlignmentTurnAreaMatrix = in_params.CoilAlignmentTurnAreaMatrix;
    out_params.R_atActuators = in_params.R_atActuators;
    out_params.p_atActuators = in_params.p_atActuators;

    out_params.CalculateEnergy = in_params.CalculateEnergy;
    out_params.FinalValueOnly = in_params.FinalValueOnly;
    out_params.NextLocMarker = in_params.NextLocMarker;
    out_params.LocMarkers = in_params.LocMarkers;
    out_params.p_atLocMarkers = in_params.p_atLocMarkers;
    out_params.fcumlambda = in_params.fcumlambda;

    // Dynamics parameters (Value copy to T)
    for(int j=0; j<NUM_ACT_SET; ++j) {
        for(int i=0; i<3; ++i) {
            out_params.v_L_pre[j][i] = T(in_params.v_L_pre[j][i]);
            out_params.w_L_pre[j][i] = T(in_params.w_L_pre[j][i]);
            out_params.p_pre[j][i] = T(in_params.p_pre[j][i]);
        }
        for(int i=0; i<9; ++i) {
            out_params.R_pre[j][i] = T(in_params.R_pre[j][i]);
        }
    }

    mCopy_ABm<NUM_ACT_SET, 9>(in_params.actInertia, out_params.actInertia);
    mCopy_ABm<NUM_ACT_SET, 6>(in_params.damping, out_params.damping);
    out_params.DELTA_T = in_params.DELTA_T;
    mCopy_ABm<NUM_ACT_SET, 3>(in_params.m_L, out_params.m_L);
    mCopy_ABm<NUM_ACT_SET, 3>(in_params.n_L, out_params.n_L);

    out_params.ContactMode = in_params.ContactMode;
    mCopy_AB<3>(in_params.TipConstraintPoint, out_params.TipConstraintPoint);
    mCopy_AB<3>(in_params.TipForce, out_params.TipForce);
    mCopy_AB<3>(in_params.ftip_initialguess, out_params.ftip_initialguess);
    
    for(int i=0; i<NUM_STATES; ++i) {
        out_params.xf[i] = T(in_params.xf[i]);
    }
}

// ============================================================================
// Templated flexible segment integration for forward-mode AD
// ============================================================================

template<typename T>
void CRMFlexible_IVP_Back_T(
    int SegmentIndex,
    const T in_p[3],             // Boundary p carries Dual derivatives
    const T in_R[9],             // Boundary R now carries Dual
    DYNNLEqnParams_T<T>& in_params,
    const T in_u[3],             // Curvature carries Dual
    const T in_n_L[3],           // Force carries Dual (from mL, nL)
    T out_u[3],                  // Output curvature carries Dual
    T out_p[3],                  // Output p carries Dual derivatives
    T out_R[9]                   // Output R now carries Dual
) {
    T n_L[3];
    for (int i = 0; i < 3; ++i) {
        n_L[i] = in_n_L[i];
    }

    double h;
    double l_zero[3] = {0.0, 0.0, 0.0};
    int fsegno;

    int NextLocMarker = in_params.NextLocMarker;

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

    fsegno = SegmentIndex >> 1;

    int base_steps = SegSteps[fsegno];
    int N_steps;
    if constexpr (!std::is_same_v<T, double>) {
        N_steps = base_steps * DUAL_SUBSTEP_FACTOR;
    } else {
        N_steps = base_steps;
    }
    h = -1 * (SegBounds[SegmentIndex + 1] - SegBounds[SegmentIndex]) / (N_steps * 1.0);

    StateVector_T<T> xi_statevec;
    StateVector_T<T> xf_statevec;

    for(int i=0; i<9; ++i) xi_statevec._R[i] = in_R[i];
    for(int i=0; i<3; ++i) xi_statevec._p[i] = in_p[i];
    for(int i=0; i<3; ++i) xi_statevec._u[i] = in_u[i];

    double DeltaPE = 0.0;
    auto& CalculateEnergy = in_params.CalculateEnergy;
    auto& no_fcum_steps = in_params.no_fcum_steps;
    auto& g = in_params.g;
    auto& rho = in_params.rho;
    double ftip[3] = {0.0, 0.0, 0.0};

    CRMIntegrandParams IntegrandParams;
    IntegrandParams.dlambdainv = dlambdainv;
    IntegrandParams.Li = InsertedLength;
    IntegrandParams.l = l_zero;
    IntegrandParams.no_fcum_steps = no_fcum_steps;
    IntegrandParams.fcumlambda = fcumlambda;
    IntegrandParams.ftip = ftip;
    IntegrandParams.g = g;

    IntegrandParams.K = K[fsegno];
    IntegrandParams.Kinv = Kinv[fsegno];
    IntegrandParams.ustar = ustar[fsegno];
    IntegrandParams.rho = rho[SegmentIndex];

    using _SVT = StateVector_T<T>;
    using _DVT = StateDerivativeVector_T<T>;

    _SVT x_n(xi_statevec);
    _SVT x_np1;
    _SVT x_nm1, x_nm2, x_nm3;
    double t_n = SegBounds[SegmentIndex + 1];
    _DVT xdot_n;
    _DVT xdot_nm1, xdot_nm2, xdot_nm3;

    int N = N_steps;

    for (int idx = 0; idx < (N < 3 ? N : 3); idx++) {
        _DVT k1, k2;
        _SVT x_mid;

        CRMIntegrand_dyn_T<T>(t_n, x_n, IntegrandParams, n_L, k1);

        x_mid = x_n + (h / 2.0) * k1;
        for (int i_r = 0; i_r < 9; ++i_r) x_mid._R[i_r] = x_n._R[i_r];

        CRMIntegrand_dyn_T<T>(t_n + h / 2.0, x_mid, IntegrandParams, n_L, k2);

        x_np1 = x_n + h * k2;

        T u_n_val[3];
        for(int i=0; i<3; ++i) u_n_val[i] = x_n._u[i];
        
        SE3_Analytical_Step_T<T>(x_n._R, x_n._p, u_n_val, h, x_np1._R, x_np1._p);

        xdot_n = k2;

        if (idx == 0) { xdot_nm3 = xdot_n; x_nm3 = x_n; }
        else if (idx == 1) { xdot_nm2 = xdot_nm3; x_nm2 = x_nm3; xdot_nm3 = xdot_n; x_nm3 = x_n; }
        else if (idx == 2) { xdot_nm1 = xdot_nm2; x_nm1 = x_nm2; xdot_nm2 = xdot_nm3; x_nm2 = x_nm3; xdot_nm3 = xdot_n; x_nm3 = x_n; }

        t_n = t_n + h;
        x_n = x_np1;
    }

    for (int idx = 3; idx < N; idx++) {
        x_np1 = x_n + (h / 24.0) * (55.0 * xdot_n + (-59.0) * xdot_nm1 + 37.0 * xdot_nm2 + (-9.0) * xdot_nm3);

        T u_pred[3];
        for(int i=0; i<3; ++i) 
            u_pred[i] = (55.0/24.0)*x_n._u[i] + (-59.0/24.0)*x_nm1._u[i] + (37.0/24.0)*x_nm2._u[i] + (-9.0/24.0)*x_nm3._u[i];
        
        SE3_Analytical_Step_T<T>(x_n._R, x_n._p, u_pred, h, x_np1._R, x_np1._p);

        _DVT xdot_np1_pred;
        CRMIntegrand_dyn_T<T>(t_n + h, x_np1, IntegrandParams, n_L, xdot_np1_pred);

        x_np1 = x_n + (h / 24.0) * (9.0 * xdot_np1_pred + 19.0 * xdot_n + (-5.0) * xdot_nm1 + xdot_nm2);

        T u_corr[3];
        for(int i=0; i<3; ++i)
            u_corr[i] = (9.0/24.0)*x_np1._u[i] + (19.0/24.0)*x_n._u[i] + (-5.0/24.0)*x_nm1._u[i] + (1.0/24.0)*x_nm2._u[i];
        
        SE3_Analytical_Step_T<T>(x_n._R, x_n._p, u_corr, h, x_np1._R, x_np1._p);

        xdot_nm3 = xdot_nm2; x_nm3 = x_nm2;
        xdot_nm2 = xdot_nm1; x_nm2 = x_nm1;
        xdot_nm1 = xdot_n;   x_nm1 = x_n;

        CRMIntegrand_dyn_T<T>(t_n + h, x_np1, IntegrandParams, n_L, xdot_n);

        t_n = t_n + h;
        x_n = x_np1;
    }

    xf_statevec = x_n;

    for (int j = 0; j < 3; j++) {
        out_u[j] = xf_statevec._u[j];
        out_p[j] = xf_statevec._p[j];
    }
    for (int j = 0; j < 9; ++j) {
        out_R[j] = xf_statevec._R[j];
    }
}

// ============================================================================
// Templated DYNNLEquation for forward-mode AD
// ============================================================================

template<typename T>
void CoilDynamics_T(const T in_x_n[NUM_COIL_STATES], const T in_n[3], const double g[3],
                    double actMass, const double actInertia[9], const double damping[6],
                    double DELTA_T, const double in_B0[3], const T in_muhat[9],
                    const T in_mL[3], T out_x_np1[NUM_COIL_STATES], T out_xdot[6]);

template<typename T>
void DYNNLEquation_T(const double in_x[], T out_y[], DYNNLEqnParams_T<T>& Params,
                     const T muhat[NUM_ACT_SET][9], T out_u0[3], T out_tau[NUM_ACT_SET*3]) {

    T m_L[NUM_ACT_SET][3], n_L[NUM_ACT_SET][3];
    T n_0[3];

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
            x_coil[j][i] = Params.v_L_pre[j][i];
            x_coil[j][i+3] = Params.w_L_pre[j][i];
            x_coil[j][i+6] = Params.p_pre[j][i];
        }
        for (int i = 0; i < 9; ++i) {
            x_coil[j][i+9] = Params.R_pre[j][i];
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

    double p_d[3], R_d[9];
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
                    p_t[i] = Params.xf[i];
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
                    p_L[i] = out_x_coil[actno][i+6] - T(R_L[i*3+2] * RigidSegmentLength * 0.5);
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
                    p_L[i] = out_x_coil[actno][i+6] + T(R_L[i*3+2] * RigidSegmentLength * 0.5);
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
                    residual[actno_mn][i+3] = T(0.5) * v_val[i];
                }
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        out_u0[i] = u_f[i];
    }

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
        residual[actno][i+3] = T(0.5) * v_val[i];
    }

    for (int i = 0; i < NUM_ACT_SET; ++i) {
        for (int j = 0; j < 6; ++j) {
            out_y[j + i*6] = residual[i][j] * (j < 3 ? RESIDUAL_SCALE_P : RESIDUAL_SCALE_R);
        }
    }
}

template<typename T>
void DYNNLEquation_YY_T(const T in_x[], T out_y[], DYNNLEqnParams_T<T>& Params,
                         const double muhat_double[NUM_ACT_SET][9], T out_u0[3], T out_tau[NUM_ACT_SET*3]) {

    T m_L[NUM_ACT_SET][3], n_L[NUM_ACT_SET][3];
    T n_0[3];

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; i++) {
            m_L[j][i] = in_x[i + j*6] * T(IVALUE_SCALE_M);
            n_L[j][i] = in_x[i + j*6 + 3] * T(IVALUE_SCALE_N);
        }
    }

    for (int i = 0; i < 3; i++) {
        n_0[i] = T(Params.TipForce[i]);
    }

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
            x_coil[j][i] = Params.v_L_pre[j][i];
            x_coil[j][i+3] = Params.w_L_pre[j][i];
            x_coil[j][i+6] = Params.p_pre[j][i];
        }
        for (int i = 0; i < 9; ++i) {
            x_coil[j][i+9] = Params.R_pre[j][i];
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

    double p_d[3], R_d[9];
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
                    p_t[i] = Params.xf[i];
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
                    p_L[i] = out_x_coil[actno][i+6] - T(R_L[i*3+2] * RigidSegmentLength * 0.5);
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
                    p_L[i] = out_x_coil[actno][i+6] + T(R_L[i*3+2] * RigidSegmentLength * 0.5);
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
                    residual[actno_mn][i+3] = T(0.5) * v_val[i];
                }
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        out_u0[i] = u_f[i];
    }

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
        residual[actno][i+3] = T(0.5) * v_val[i];
    }

    for (int i = 0; i < NUM_ACT_SET; ++i) {
        for (int j = 0; j < 6; ++j) {
            out_y[j + i*6] = residual[i][j] * (j < 3 ? RESIDUAL_SCALE_P : RESIDUAL_SCALE_R);
        }
    }
}

// ============================================================================
// Templated CRMFlexForward_pass for forward-mode AD
// ============================================================================
template <typename T>
void CRMFlexForward_pass_T(int SegmentIndex, const T in_p[3], const T in_R[9], DYNNLEqnParams_T<T>& in_params,
                           const T in_u[3], const T in_n_L[3],
                           T out_u[3], T out_p[3], T out_R[9], T out_p_atLocMarkers[][3]) {

    T n_L[3];
    for (int i = 0; i < 3; ++i) {
        n_L[i] = in_n_L[i];
    }
    double h;
    int fsegno;

    int NextLocMarker = in_params.NextLocMarker;
    
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

    fsegno = SegmentIndex >> 1;
    
    int base_steps = SegSteps[fsegno];
    int N_steps;
    if constexpr (!std::is_same_v<T, double>) {
        N_steps = base_steps * DUAL_SUBSTEP_FACTOR;
    } else {
        N_steps = base_steps;
    }
    h = (SegBounds[SegmentIndex+1] - SegBounds[SegmentIndex]) / (N_steps * 1.0);

    StateVector_T<T> xi_statevec;
    StateVector_T<T> xf_statevec;

    for(int i=0; i<9; ++i) xi_statevec._R[i] = in_R[i];
    for(int i=0; i<3; ++i) xi_statevec._p[i] = in_p[i];
    for(int i=0; i<3; ++i) xi_statevec._u[i] = in_u[i];

    double DeltaPE = 0.0;
    auto& CalculateEnergy = in_params.CalculateEnergy;
    auto& no_fcum_steps = in_params.no_fcum_steps;
    auto& g = in_params.g;
    auto& rho = in_params.rho;

    double l_zero[3] = {0.0, 0.0, 0.0};
    double ftip_double[3] = {0.0, 0.0, 0.0}; 
    
    CRMIntegrandParams IntegrandParams;
    IntegrandParams.dlambdainv = dlambdainv;
    IntegrandParams.Li = InsertedLength;
    IntegrandParams.l = l_zero;
    IntegrandParams.no_fcum_steps = no_fcum_steps;
    IntegrandParams.fcumlambda = fcumlambda;
    IntegrandParams.ftip = ftip_double;
    IntegrandParams.g = g;
    IntegrandParams.K = K[fsegno];
    IntegrandParams.Kinv = Kinv[fsegno];
    IntegrandParams.ustar = ustar[fsegno];
    IntegrandParams.rho = rho[SegmentIndex];

    using _SVT = StateVector_T<T>;
    using _DVT = StateDerivativeVector_T<T>;

    _SVT x_n(xi_statevec);
    _SVT x_np1;
    _SVT x_nm1, x_nm2, x_nm3;
    double t_n = SegBounds[SegmentIndex];
    _DVT xdot_n;
    _DVT xdot_nm1, xdot_nm2, xdot_nm3;

    int N = N_steps;

    for (int idx = 0; idx < (N < 3 ? N : 3); idx++) {
        _DVT k1, k2;
        _SVT x_mid;

        CRMIntegrand_dyn_T<T>(t_n, x_n, IntegrandParams, n_L, k1);

        x_mid = x_n + (h / 2.0) * k1;
        for (int i_r = 0; i_r < 9; ++i_r) x_mid._R[i_r] = x_n._R[i_r];

        CRMIntegrand_dyn_T<T>(t_n + h / 2.0, x_mid, IntegrandParams, n_L, k2);

        x_np1 = x_n + h * k2;

        T u_n_val[3];
        for(int i=0; i<3; ++i) u_n_val[i] = x_n._u[i]; 
        
        SE3_Analytical_Step_T<T>(x_n._R, x_n._p, u_n_val, h, x_np1._R, x_np1._p);

        xdot_n = k2;

        if (idx == 0) { xdot_nm3 = xdot_n; x_nm3 = x_n; }
        else if (idx == 1) { xdot_nm2 = xdot_nm3; x_nm2 = x_nm3; xdot_nm3 = xdot_n; x_nm3 = x_n; }
        else if (idx == 2) { xdot_nm1 = xdot_nm2; x_nm1 = x_nm2; xdot_nm2 = xdot_nm3; x_nm2 = x_nm3; xdot_nm3 = xdot_n; x_nm3 = x_n; }

        t_n = t_n + h;
        x_n = x_np1;
    }

    for (int idx = 3; idx < N; idx++) {
        x_np1 = x_n + (h / 24.0) * (55.0 * xdot_n + (-59.0) * xdot_nm1 + 37.0 * xdot_nm2 + (-9.0) * xdot_nm3);
        
        T u_pred[3];
        for(int i=0; i<3; ++i) 
            u_pred[i] = (55.0/24.0)*x_n._u[i] + (-59.0/24.0)*x_nm1._u[i] + (37.0/24.0)*x_nm2._u[i] + (-9.0/24.0)*x_nm3._u[i];
        
        SE3_Analytical_Step_T<T>(x_n._R, x_n._p, u_pred, h, x_np1._R, x_np1._p);

        _DVT xdot_np1_pred;
        CRMIntegrand_dyn_T<T>(t_n + h, x_np1, IntegrandParams, n_L, xdot_np1_pred);

        x_np1 = x_n + (h / 24.0) * (9.0 * xdot_np1_pred + 19.0 * xdot_n + (-5.0) * xdot_nm1 + xdot_nm2);

        T u_corr[3];
        for(int i=0; i<3; ++i)
            u_corr[i] = (9.0/24.0)*x_np1._u[i] + (19.0/24.0)*x_n._u[i] + (-5.0/24.0)*x_nm1._u[i] + (1.0/24.0)*x_nm2._u[i];
        
        SE3_Analytical_Step_T<T>(x_n._R, x_n._p, u_corr, h, x_np1._R, x_np1._p);

        xdot_nm3 = xdot_nm2; x_nm3 = x_nm2;
        xdot_nm2 = xdot_nm1; x_nm2 = x_nm1;
        xdot_nm1 = xdot_n;   x_nm1 = x_n;

        CRMIntegrand_dyn_T<T>(t_n + h, x_np1, IntegrandParams, n_L, xdot_n);

        t_n = t_n + h;
        x_n = x_np1;
    }

    xf_statevec = x_n;

    for (int j = 0; j < 3; j++) {
        out_u[j] = xf_statevec._u[j];
        out_p[j] = xf_statevec._p[j];
    }
    for (int j = 0; j < 9; ++j) {
        out_R[j] = xf_statevec._R[j];
    }
}

// ============================================================================
// Templated CRMIVP_DYN for forward-mode AD
// ============================================================================
template <typename T>
void CRMIVP_DYN_T(DYNNLEqnParams_T<T>& CoreParams, const T in_u0[3], const T in_p0[3], const T in_R0[9],
                  const T in_mL[NUM_ACT_SET][3], const T in_nL[NUM_ACT_SET][3], const T in_tau[NUM_ACT_SET][3], const T in_ftip[3],
                  T out_coil_state[NUM_ACT_SET][NUM_COIL_STATES], T out_u_new[3], T out_p_new[3], T out_R_new[9],
                  T out_p_atLocMarkers[][3]) {

    T x_coil[NUM_ACT_SET][NUM_COIL_STATES];
    T MagMoment[NUM_ACT_SET][3], muhat[NUM_ACT_SET][9];
    double actMass[NUM_ACT_SET], actInertia[NUM_ACT_SET][9];

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            x_coil[j][i] = CoreParams.v_L_pre[j][i];
            x_coil[j][i+3] = CoreParams.w_L_pre[j][i];
            x_coil[j][i+6] = CoreParams.p_pre[j][i];
        }
        for (int i = 0; i < 9; ++i) {
            x_coil[j][i+9] = CoreParams.R_pre[j][i];
        }

        actMass[j] = CoreParams.ActMass[j];
        for (int i = 0; i < 9; ++i) {
            actInertia[j][i] = CoreParams.actInertia[j][i];
        }

        for (int i = 0; i < 3; ++i) {
            MagMoment[j][i] = T(CoreParams.MagMoment[j][i]);
        }
        
        wHat_T(MagMoment[j], muhat[j]);
    }

    T u_new[3], p_new[3], R_new[9];
    int actno, fsegip1;
    T RscTB0[3], Tb[3], Residual[3], inertiaw[3];
    auto& Kinv = CoreParams.Kinv;
    auto& ustar = CoreParams.ustar;
    auto& B0 = CoreParams.B0;
    auto& SegBounds = CoreParams.SegBounds;
    
    T w[3], w_dot[3], w_hat[9], w_inertia_w[3], inertia_wdot[3], out_xdot[6], w_terms[3], Tb_ml[3], K2invResidual[3];
    T u_0[3], p0[3], R0[9], n_L[NUM_ACT_SET][3], m_L[NUM_ACT_SET][3], tau[NUM_ACT_SET][3];

    for(int j=0; j<NUM_ACT_SET; ++j) {
        for(int i=0; i<3; ++i) {
            n_L[j][i] = in_nL[j][i];
            m_L[j][i] = in_mL[j][i];
            tau[j][i] = in_tau[j][i];
        }
    }

    for (int i = 0; i < 3; ++i) {
        u_0[i] = in_u0[i];
        p0[i] = in_p0[i];
    }
    for (int i = 0; i < 9; ++i) {
        R0[i] = in_R0[i];
    }

    double RigidSegmentLength;
    T n_f[3], net_nL[3], net_mL[3];
    
    T dummy_p_loc_T[MAX_LOC_MARKERS][3];

    auto& NUM_SEGMENTS = CoreParams.no_segments;
    T update_coil_state[NUM_COIL_STATES];

    for (int SegmentIndex = 0; SegmentIndex < NUM_SEGMENTS; ++SegmentIndex) {
        if (SegmentIndex % 2 == 0) { // Flexible
            actno = SegmentIndex >> 1;
            if (SegmentIndex == NUM_SEGMENTS - 1) {
                for (int i = 0; i < 3; ++i) n_f[i] = in_ftip[i];
            } else {
                for (int i = 0; i < 3; ++i) n_f[i] = n_L[actno][i];
            }
            CRMFlexForward_pass_T(SegmentIndex, p0, R0, CoreParams, u_0, n_f, u_new, p_new, R_new, dummy_p_loc_T);
        } else { // Rigid/Coil
            RigidSegmentLength = (SegBounds[SegmentIndex+1] - SegBounds[SegmentIndex]);
            actno = (SegmentIndex - 1) >> 1;
            fsegip1 = actno + 1;

            if (actno < NUM_ACT_SET - 1) {
                mSub_AB_T<T, 3, 1>(n_L[actno], n_L[actno+1], net_nL);
            } else {
                mSub_AB_T<T, 3, 1>(n_L[actno], in_ftip, net_nL);
            }
            mSub_AB_T<T, 3, 1>(m_L[actno], tau[actno], net_mL);

            CoilDynamics_T<T>(x_coil[actno], net_nL, CoreParams.g, actMass[actno], actInertia[actno], CoreParams.damping[actno], CoreParams.DELTA_T,
                              CoreParams.B0, muhat[actno], net_mL, update_coil_state, out_xdot);

            for (int i = 0; i < 3; ++i) {
                w[i] = update_coil_state[i+3];
                w_dot[i] = out_xdot[i+3];
            }
            wHat_T(w, w_hat);

            T R_coil[9];
            for(int i=0; i<9; ++i) R_coil[i] = update_coil_state[i+9];
            T B0_T[3] = {T(B0[0]), T(B0[1]), T(B0[2])};

            mMult_ATB_T<T, 3, 3, 1>(R_coil, B0_T, RscTB0);
            mMult_AB_T<T, 3, 3, 1>(muhat[actno], RscTB0, Tb);

            T inertia_T[9];
            for(int i=0; i<9; ++i) inertia_T[i] = T(actInertia[actno][i]);

            mMult_AB_T<T, 3, 3, 1>(inertia_T, w, inertiaw);
            mMult_AB_T<T, 3, 3, 1>(w_hat, inertiaw, w_inertia_w);
            mMult_AB_T<T, 3, 3, 1>(inertia_T, w_dot, inertia_wdot);
            mAdd_AB_T<T, 3, 1>(inertia_wdot, w_inertia_w, w_terms);

            mSub_AB_T<T, 3, 1>(Tb, m_L[actno], Tb_ml);
            mSub_AB_T<T, 3, 1>(Tb_ml, w_terms, Residual);

            T Kinv_T[9];
            T ustar_T[3];
            for(int i=0; i<9; ++i) Kinv_T[i] = T(Kinv[fsegip1][i]);
            for(int i=0; i<3; ++i) ustar_T[i] = T(ustar[fsegip1][i]);

            mMult_AB_T<T, 3, 3, 1>(Kinv_T, tau[actno], K2invResidual);
            mAdd_AB_T<T, 3, 1>(ustar_T, K2invResidual, u_0);

            T R_update[9];
            for(int i=0; i<9; ++i) R_update[i] = update_coil_state[i+9];

            for (int i = 0; i < 3; ++i) {
                p0[i] = update_coil_state[i+6] + T(0.5 * RigidSegmentLength) * R_update[i * 3 + 2];
            }
            for (int i = 0; i < 9; ++i) {
                R0[i] = update_coil_state[i+9];
            }
            for (int i = 0; i < NUM_COIL_STATES; ++i) {
                out_coil_state[actno][i] = update_coil_state[i];
            }
        }
    }

    for (int i = 0; i < 3; ++i) {
        out_u_new[i] = u_new[i];
        out_p_new[i] = p_new[i];
    }
    for (int i = 0; i < 9; ++i) {
        out_R_new[i] = R_new[i];
    }
}

// ============================================================================
// Templated DYNSolverIVP_T for forward-mode AD
// ============================================================================
template <typename T>
void DYNSolverIVP_T(DYNNLEqnParams_T<T>& in_Params, const T in_u0[3],
                    const T in_mL[NUM_ACT_SET][3], const T in_nL[NUM_ACT_SET][3], 
                    const T in_tau[NUM_ACT_SET][3], const T in_ftip[3],
                    T out_x_N[NUM_STATES], T out_coil_state[NUM_ACT_SET][NUM_COIL_STATES]) {

    T x_0[NUM_STATES];
    for (int i = 0; i < 3; i++) x_0[i] = T(in_Params.xi[i]); // p0
    for (int i = 3; i < 12; i++) x_0[i] = T(in_Params.xi[i]); // R0
    for (int i = 12; i < 15; i++) x_0[i] = in_u0[i-12]; // u0

    T p0[3], R0[9];
    for(int i=0; i<3; ++i) p0[i] = x_0[i];
    for(int i=0; i<9; ++i) R0[i] = x_0[i+3];

    T u_new[3], p_new[3], R_new[9];
    T dummy_p_loc[MAX_LOC_MARKERS][3];

    CRMIVP_DYN_T(in_Params, in_u0, p0, R0, in_mL, in_nL, in_tau, in_ftip, out_coil_state, u_new, p_new, R_new, dummy_p_loc);

    for (int i = 0; i < NUM_STATES; ++i) {
        if(i<3){
            out_x_N[i] = p_new[i];
        }else if(i<12){
            out_x_N[i] = R_new[i-3];
        }else{
            out_x_N[i] = u_new[i-12];
        }
    }
}

} // namespace CRMCatheterModel