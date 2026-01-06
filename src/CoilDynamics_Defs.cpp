#pragma once

namespace {
constexpr double kCoilTStep = 0.001;
}  // namespace

#include "CRMDYN.hpp"
#include "minpack_DYN.hpp"
#include "CRM_BVPIVP_APIDeclarations.hpp"
#include "CRM.hpp"
#include "CRMDYN_Numerical_Integration.hpp"
#include "CRM_BVPJacobian.hpp"
#include "CoilDynamics_Defs_Templates2.hpp"

 /**

  * Coil integration function in coil frame
  * @tparam
  * @param in_twist
  * @param n_L
  * @param g
  * @param R
  * @param actMass
  * @param actInertia
  * @param input_tau
  * @param vdot
  * @param wdot
  */
void CoilIntegrad(const double in_twist[6], const double in_n[3], double g[3], double R[9], double actMass, double actInertia[9], const double damping[6],
                  const double in_B0[3], const double in_muhat[9], const double in_mL[3], double twistdot[6]){

     double v[3], w[3], n_L[3], m_L[3], B0[3], muhat[9], Tb[3], tau[3];
     double RscTB0[3], RTg[3], w_v[3], w_hat[9], inertiaw[3], w_inertia_w[3], diff_tau_w[3];
     double vdot[3], wdot[3];

     mMult_ATB<3,3,1>(R, g, RTg);

     for (int i = 0; i < 3; ++i) {
        v[i] = in_twist[i];
        w[i] = in_twist[i+3];
    }
     for (int i = 0; i < 3; ++i) {
         n_L[i] = in_n[i];
         B0[i] = in_B0[i];
         m_L[i] = in_mL[i];
     }

     for (int i = 0; i < 9; ++i) {
         muhat[i] = in_muhat[i];
     }
     wHat(w,w_hat);
     mMult_AB<3,3,1>(w_hat, v, w_v);

     double damping_vec[3];
     for (int i = 0; i < 3; ++i) {
         damping_vec[i] = damping[i] * v[i] ;
     }

     for (int i = 0; i < 3; ++i) {
         vdot[i] = RTg[i]  - n_L[i] / actMass - w_v[i] - damping_vec[i];// / actMass; //RTg[i] - n_L[i] / actMass
     }

//     std::cout << "RTg[i] - n_L[i] / actMass: " << RTg[0] - n_L[0] / actMass << " " << RTg[1] - n_L[1] / actMass << " " << RTg[2] - n_L[2] / actMass <<  std::endl;

     double damping_wec[3], residual_w[3];
     for (int i = 0; i < 3; ++i) {
         damping_wec[i] = damping[i+3] * w[i] ;
     }

     mMult_AB<3,3,1>(actInertia, w, inertiaw);
     mMult_AB<3,3,1>(w_hat, inertiaw, w_inertia_w);

     mMult_ATB<3,3,1>(R,B0,RscTB0);
     mMult_AB<3,3,1>(muhat,RscTB0,Tb);
     mSub_AB<3, 1>(Tb, m_L, tau);


     mSub_AB<3,1>(tau, w_inertia_w, diff_tau_w);
     mSub_AB<3,1>(diff_tau_w, damping_wec, residual_w);

     ///  actInertia is diagonal
     wdot[0] = residual_w[0] / actInertia[0];
     wdot[1] = residual_w[1] / actInertia[4];
     wdot[2] = residual_w[2] / actInertia[8];

     for (int i = 0; i < 3; ++i) {
         twistdot[i] = vdot[i];
         twistdot[i+3] = wdot[i];
     }

 }


/**
 * Main Dynamics function of the coil
 * @tpar
 * @param T
 * @param v_L_pre
 * @param w_L_pre
 * @param in_p
 * @param in_R
 * @param in_n
 * @param g
 * @param actMass
 * @param actInertia
 * @param in_tau
 * @param out_coil_state
 */
void CoilDynamics( double in_coil_state[NUM_COIL_STATES], double in_n[3], double g[3],
                   double actMass, double actInertia[9], double damping[6], double DELTA_T, double in_B0[3], double in_muhat[9],
                   double in_mL[3], double out_coil_state[NUM_COIL_STATES], double out_xdot_n[6]){

//    for (int i = 0; i < 3; ++i) {
//        std::cout << "in_n: " << in_n[i] << std::endl;
//    }
//    for (int i = 0; i < 3; ++i) {
//        std::cout << "in_tau: " << in_tau[i] << std::endl;
//    }

//    std::cout << "R_pre: " << std::endl;
//    std::cout <<  in_coil_state[9] << " " << in_coil_state[10]<< " " << in_coil_state[11] <<  std::endl;
//    std::cout <<  in_coil_state[12]<< " " << in_coil_state[13]<< " " << in_coil_state[14] <<  std::endl;
//    std::cout << in_coil_state[15]<< " " << in_coil_state[16] << " " << in_coil_state[17] <<  std::endl;
//


    double x_nm3[NUM_COIL_STATES];
    double x_nm2[NUM_COIL_STATES];
    double x_nm1[NUM_COIL_STATES];
    double x_n[NUM_COIL_STATES];
    double x_np1[NUM_COIL_STATES];
    double xdot_nm3[6];
    double xdot_nm2[6];
    double xdot_nm1[6];
    double xdot_n[6];
    double nL[3], B0[3], muhat[9], mL[3];

    for (int i = 0; i < 3; ++i) {
        nL[i] = in_n[i];
        B0[i] = in_B0[i];
        mL[i] = in_mL[i];
    }

    for (int i = 0; i < 9; ++i) {
        muhat[i] = in_muhat[i];
    }

    // initialize the iteration items
    for (int i = 0; i < NUM_COIL_STATES; i++) {
            x_n[i] = in_coil_state[i];
    }


    int N = ceil(DELTA_T / kCoilTStep);

    for (int idx=0; idx<N; idx++) {

        if (idx<3) {  // RK2 initialization steps
            RK2_coildyn(x_n, nL, g,  actMass, actInertia, damping, B0, muhat, mL, x_np1, xdot_n );
        }
        else { 		 // ABM4 steps
            ABM4_coildyn(	x_n, xdot_nm1, xdot_nm2, xdot_nm3, x_nm1, x_nm2, x_nm3,
                             nL, g,  actMass, actInertia, damping,  B0, muhat, mL,x_np1, xdot_n);
            if ( isnan(x_n[0]) ) {
                std::cout << "Coil integration Unbounded!! " << std::endl;
//                exit( 3 );
            }
        }

//        std::cout << "input v_n: " << xdot_n[0] << " " << xdot_n[1] << " " << xdot_n[2] <<  std::endl;
//        std::cout << "input w_n: " << xdot_n[3] << " " << xdot_n[4] << " " << xdot_n[5] <<  std::endl;

        // update the iteration items
        for (int i = 0; i < NUM_COIL_STATES; i++) {
            x_nm3[i] = x_nm2[i];
            x_nm2[i] = x_nm1[i];
            x_nm1[i] = x_n[i];
            x_n[i] = x_np1[i];
        }
        for (int i = 0; i < 6; i++) {
            xdot_nm3[i]=xdot_nm2[i];
            xdot_nm2[i]=xdot_nm1[i];
            xdot_nm1[i]=xdot_n[i];
        }

    }

    // Copy final value
    for (int i=0; i<NUM_COIL_STATES; i++) {
        out_coil_state[i]=x_n[i];
    }
    for (int i = 0; i < 6; ++i) {
        out_xdot_n[i] = xdot_n[i];
    }

//    std::cout << "out_coil_state v_n: " << out_coil_state[0] << " " << out_coil_state[1] << " " << out_coil_state[2] <<  std::endl;
//    std::cout << "out_coil_state w_n: " << out_coil_state[3] << " " << out_coil_state[4] << " " << out_coil_state[5] <<  std::endl;
//    std::cout << "out_coil_state  v dot: " << out_xdot_n[0] << " " << out_xdot_n[1] << " " << out_xdot_n[2] <<  std::endl;
//    std::cout << "out_coil_state w dot: " << out_xdot_n[3] << " " << out_xdot_n[4] << " " << out_xdot_n[5] <<  std::endl;
//    std::cout << " ---------  " <<  std::endl;

}



//[x_np1, xdot_n] = RK2_step(x_n, t_n, h, Integrand)
void RK2_coildyn(double in_x_n[NUM_COIL_STATES], double in_n[3], double g[3],  double actMass, double actInertia[9], double damping[6],
                 double in_B0[3], double in_muhat[9], double in_mL[3],
                 double out_x_np1[NUM_COIL_STATES], double out_xdot_n[6] ) {


    double nL[3], B0[3], muhat[9], mL[3],  twist_n[6];       // from input
    double k1[6];
    double k2oh[6];
    double x_n_p_k1o2[6];
    double xdot_n[6];

    double R_n[9], p_n[3];				// variables used in analytical calculation
    for (int i = 0; i < 3; ++i) p_n[i] = in_x_n[i+6];
    for (int i = 0; i < 9; ++i) R_n[i] = in_x_n[i+9];
    for (int i = 0; i < 6; ++i) twist_n[i] = in_x_n[i];

    for (int i = 0; i < 3; ++i) {
        nL[i] = in_n[i];
        B0[i] = in_B0[i];
        mL[i] = in_mL[i];
    }
    for (int i = 0; i < 9; ++i) {
        muhat[i] = in_muhat[i];
    }

//    std::cout << "input v_n: " << twist_n[0] << " " << twist_n[1] << " " << twist_n[2] <<  std::endl;
//    std::cout << "input w_n: " << twist_n[3] << " " << twist_n[4] << " " << twist_n[5] <<  std::endl;

    //RK2_STEP_STEP1:
    CoilIntegrad(twist_n, nL, g, R_n, actMass, actInertia, damping,B0, muhat, mL, xdot_n);
    for (int i=0; i< 6 ; i++) {
        k1[i] 			= kCoilTStep * xdot_n[i];
        x_n_p_k1o2[i] 	= twist_n[i] + k1[i] * 0.5;
    }
#ifdef ANALYTICAL_SE3_STEP
    // we will calculate R_np1half and p_np1half analytically, without numerical integration
    double R_np1half[9], p_np1half[3];

    DYNSE3_TimeSpace(R_n, p_n, kCoilTStep * 0.5, twist_n, R_np1half, p_np1half) ;

#endif

    //RK2_STEP_STEP2:
    CoilIntegrad(x_n_p_k1o2, nL, g, R_np1half, actMass, actInertia, damping,B0, muhat, mL,  k2oh);

    for (int i = 0; i < 6; i++) {
        out_x_np1[i] = twist_n[i] + kCoilTStep * k2oh[i];
    }

#ifdef ANALYTICAL_SE3_STEP
    // we will calculate R_np1 and p_np1 analytically, without numerical integration
    double R_np1[9], p_np1[3];

    DYNSE3_TimeSpace(R_n, p_n, kCoilTStep,  x_n_p_k1o2, R_np1, p_np1) ;

    // copy these to the output state
    for (int i = 0; i < 3; i++) out_x_np1[i+6] = p_np1[i];
    for (int i = 0; i < 9; i++) out_x_np1[i+9] = R_np1[i];

#endif
    for (int i = 0; i < 6; i++) {
        out_xdot_n[i] = xdot_n[i];
    }

}


// [x_np1, xdot_n, xdot_nm1, xdot_nm2] = ABM4_step(x_n, t_n, xdot_nm1, xdot_nm2, xdot_nm3, h, Integrand)
void ABM4_coildyn(	double in_x_n[NUM_COIL_STATES],double in_xdot_nm1[6], double in_xdot_nm2[6], double in_xdot_nm3[6],
                   double in_x_nm1[NUM_COIL_STATES], double in_x_nm2[NUM_COIL_STATES], double in_x_nm3[NUM_COIL_STATES],
                   double in_n[3], double g[3],  double actMass, double actInertia[9], double damping[6], double in_B0[3], double in_muhat[9], double in_mL[3],
                   double out_x_np1[NUM_COIL_STATES], double out_xdot_n[6]) {

    const double P_COEFF_N=55.0/24.0, P_COEFF_Nm1=-59.0/24.0, P_COEFF_Nm2=37.0/24.0, P_COEFF_Nm3=-9.0/24.0;  // AB4 Predictor Coefficients
    const double C_COEFF_Np1=9.0/24.0, C_COEFF_N=19.0/24.0, C_COEFF_Nm1=-5.0/24.0, C_COEFF_Nm2=1.0/24.0;     // AM4 Corrector Coefficients
//    double x_n[NUM_COIL_STATES];       				// from input
    double twist_nm1[6];       				// from input
    double twist_nm2[6];       				// from input
    double twist_nm3[6];       				// from input
    double twist_n[6], R_n[9], p_n[3], nL[3];				// variables used in analytical calculation
    double x_np1_hat[6];    			// intermediate twist
    double xdot_np1_hat[6];	// intermediate
    double xdot_n[6];  		// for output
    double xdot_nm1[6];  	// from input
    double xdot_nm2[6];  	// from input
    double xdot_nm3[6];  	// from input

    for (int i = 0; i < 6; i++) {
        twist_nm1[i] = in_x_nm1[i];
        twist_nm2[i] = in_x_nm2[i];
        twist_nm3[i] = in_x_nm3[i];
    }
    for (int i = 0; i < 6; i++) {
        xdot_nm1[i]=in_xdot_nm1[i];
        xdot_nm2[i]=in_xdot_nm2[i];
        xdot_nm3[i]=in_xdot_nm3[i];
    }


    for (int i = 0; i < 3; ++i) p_n[i] = in_x_n[i+6];
    for (int i = 0; i < 9; ++i) R_n[i] = in_x_n[i+9];
    for (int i = 0; i < 6; ++i) twist_n[i] = in_x_n[i];

    double B0[3], muhat[9], mL[3];
    for (int i = 0; i < 3; ++i) {
        nL[i] = in_n[i];
        B0[i] = in_B0[i];
        mL[i] = in_mL[i];
    }
    for (int i = 0; i < 9; ++i) {
        muhat[i] = in_muhat[i];
    }


    double R_np1_hat[9], p_np1_hat[3];

    //ABM4_STEP_STEP1:
    CoilIntegrad(twist_n, nL, g, R_n, actMass, actInertia, damping,B0, muhat, mL, xdot_n);

    for (int i=0; i<6; i++) {
        x_np1_hat[i]    = twist_n[i] + kCoilTStep * ( P_COEFF_N * xdot_n[i] + P_COEFF_Nm1 * xdot_nm1[i] + P_COEFF_Nm2 * xdot_nm2[i] + P_COEFF_Nm3 * xdot_nm3[i] );
    }
#ifdef ANALYTICAL_SE3_STEP
    //      calculate R_np1_hat and p_np1_hat analytically, without numerical integration
    double twist_n_pred[6];
    for (int i = 0; i < 6; i++) twist_n_pred[i] = (P_COEFF_N * twist_n[i] + P_COEFF_Nm1 * twist_nm1[i] + P_COEFF_Nm2 * twist_nm2[i] + P_COEFF_Nm3 * twist_nm3[i]);
    DYNSE3_TimeSpace(R_n, p_n, kCoilTStep, twist_n, R_np1_hat, p_np1_hat) ;

//    for (int i = 0; i < 3; ++i) x_np1_hat[i+6] = p_np1_hat[i];
//    for (int i = 0; i < 9; ++i) x_np1_hat[i+9] = R_np1_hat[i];
#endif
    //ABM4_STEP_STEP2:
    CoilIntegrad(x_np1_hat, nL, g, R_np1_hat, actMass, actInertia, damping,B0, muhat, mL, xdot_np1_hat);

    for (int i=0; i<6; i++) {
        out_x_np1[i]    = twist_n[i] + kCoilTStep * ( C_COEFF_Np1 * xdot_np1_hat[i] + C_COEFF_N * xdot_n[i] + C_COEFF_Nm1 * xdot_nm1[i] + C_COEFF_Nm2 * xdot_nm2[i] );
    }
#ifdef ANALYTICAL_SE3_STEP
    //      calculate R_np1 and p_np1 analytically, without numerical integration
    double twist_n_corr[6], R_x_np1[9], p_x_np1[3];
    for (int i = 0; i < 6; i++) twist_n_corr[i] = (C_COEFF_Np1 * x_np1_hat[i] + C_COEFF_N * twist_n[i] + C_COEFF_Nm1 * twist_nm1[i] + C_COEFF_Nm2 * twist_nm2[i]);
    DYNSE3_TimeSpace(R_n, p_n, kCoilTStep, twist_n_corr, R_x_np1, p_x_np1) ;
    for (int i = 0; i < 3; ++i) out_x_np1[i+6] = p_x_np1[i];
    for (int i = 0; i < 9; ++i) out_x_np1[i+9] = R_x_np1[i];
#endif

    // copy to output
    for (int i=0; i<6; i++) {
        out_xdot_n[i] 	= xdot_n[i];
    }

}


// definitions needed for twist exponential calculation
#define EPS 1.0e-12   // the threshold for assuming ||u|| to be approximately 0, so that we should use pure translation equation


// calculate R_np1 and p_np1 analytically using twist exponential, without numerical integration
//   g_np1 = g_n * expm ( \hat{\xi}^b *h ),  where \xi^b= [ 0 0 1 u_n^T ]^T
//   g = [R p; 0 0 0 1];
void DYNSE3_TimeSpace(double in_R_n[9], double in_p_n[3], double h, double in_twist_n[6], double out_R_np1[9], double out_p_np1[3]) {
    double R_n[9], p_n[3], v_n[3], w_n[3];

    mCopy_AB<3*3>(in_R_n, R_n);
    mCopy_AB<3>(in_p_n, p_n);

    for (int i = 0; i < 3; ++i) {
        v_n[i] = in_twist_n[i];
        w_n[i] = in_twist_n[i+3];
    }

    // calculate R_np1 and p_np1 analytically, without numerical integration
    double Rdelta[9], pdelta[3];
    double umagsq, umagsqresp, umag, umagresp, unorm[3], delsumag, ImRuxvpuuTvds[3], Rnpd[3];
    umagsq = vNormSq<3>(w_n);

    double p_dot[3];
    mMult_AB<3,3,1>(R_n, v_n, p_dot);

    if (umagsq < EPS) {
        mCopy_AB<3*3>(R_n, out_R_np1);
        out_p_np1[0] = p_n[0] + p_dot[0] * h;
        out_p_np1[1] = p_n[1] + p_dot[1] * h;
        out_p_np1[2] = p_n[2] + p_dot[2] * h;
    }
    else {
        umagsqresp = 1.0 / umagsq;
        umag = sqrt(umagsq);
        umagresp = 1.0 / umag;
        mMult_sA<3, 1>(umagresp, w_n, unorm);
        delsumag = h * umag;
        RodriguesExpanded(unorm, delsumag, Rdelta);
        mMult_AB<3, 3, 3>(R_n, Rdelta, out_R_np1);

        double what[9], wxv[3], ImRdelta[9], ImRdeltawxv[3] , wwTv[3], wwTvh[3];
        wHat(w_n, what);
        mMult_AB<3,3,1>(what, v_n, wxv);

        ImRdelta[0] = 1 - Rdelta[0]; ImRdelta[1] = -Rdelta[1]; ImRdelta[2] = -Rdelta[2];
        ImRdelta[3] =  -Rdelta[3]; ImRdelta[4] = 1 -Rdelta[4]; ImRdelta[5] = -Rdelta[5];
        ImRdelta[6] = -Rdelta[6]; ImRdelta[7] = -Rdelta[7]; ImRdelta[8] = 1 - Rdelta[8];

        mMult_AB<3,3,1>(ImRdelta, wxv, ImRdeltawxv);

        wwTv[0] = w_n[0]* w_n[0] * v_n[0] + w_n[0]*w_n[1] * v_n[1] + w_n[0] * w_n[2] * v_n[2];
        wwTv[1] = w_n[1]* w_n[0] * v_n[1] + w_n[1]*w_n[1] * v_n[1] + w_n[1] * w_n[2] * v_n[2];
        wwTv[2] = w_n[2]* w_n[0] * v_n[0] + w_n[2]*w_n[1] * v_n[1] + w_n[2] * w_n[2] * v_n[2];

        for (int i = 0; i < 3; ++i) {
            wwTvh[i] = wwTv[i] * h;
        }

        mAdd_AB<3, 1>(ImRdeltawxv, wwTvh, ImRuxvpuuTvds);
        mMult_sA<3, 1>(umagsqresp, ImRuxvpuuTvds, pdelta);

        mMult_AB<3, 3, 1>(R_n, pdelta, Rnpd);
        mAdd_AB<3, 1>(p_n, Rnpd, out_p_np1);
    }
}

#undef EPS

void DYNNLEquation(double in_x[], double out_y[], DYNNLEqnParams& Params, double out_u0[3], double out_tau[NUM_ACT_SET*3]) {

    // output for time advance, not used in BVP, just placeholders
    double m_L[NUM_ACT_SET][3], n_L[NUM_ACT_SET][3], n_0[3];
    // don't forget to scale parameters before passing to the CRMSolverIVP
    for (int j= 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; i++) {
            m_L[j][i] = IVALUE_SCALE_M * in_x[i + j*6];
            n_L[j][i] = IVALUE_SCALE_N * in_x[i+ j*6 + 3];
        }
    }

    for (int i = 0; i < 3; i++) {
        n_0[i] = Params.TipForce[i];  // for free-tip, this parameter is not given by the nonlinear equation solver, and hence, does not need to be scaled
    }

//     for (int i = 0; i < NUM_ACT_SET; ++i) {
//         std::cout << "m_L dyn: " << m_L[i][0] << " " << m_L[i][1] << " " <<m_L[i][2] << std::endl;
//         std::cout << "n_L dyn: " << n_L[i][0] << " " << n_L[i][1] << " " <<n_L[i][2] << std::endl;
//     }

    double x_coil[NUM_ACT_SET][NUM_COIL_STATES], out_x_coil[NUM_ACT_SET][NUM_COIL_STATES];
    double MagMoment[NUM_ACT_SET][3], muhat[NUM_ACT_SET][9], actMass[NUM_ACT_SET], actInertia[NUM_ACT_SET][9];

    double mu[3], muhattemp[9];

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

        for (int i = 0; i <3 ; ++i) {
            MagMoment[j][i] = Params.MagMoment[j][i];
        }
        for (int i = 0; i < 3; ++i) {
            mu[i] =  MagMoment[j][i];
        }
        wHat(mu,muhattemp);
        for (int i = 0; i < 9; ++i) {
            muhat[j][i] = muhattemp[i];
        }
    }


    auto & K = Params.K;
    auto & Kinv = Params.Kinv;
    auto & ustar = Params.ustar;
    auto & SegBounds = Params.SegBounds;

    double net_mL[3], tau[3], K2invResidual[3], u_t[3], du[3];
    for (int i = 0; i < 3; ++i) {
        net_mL[i] = 0.0;
    }
//    double n_0[3] = {0,0,0}; //

    double tau_0[3] = {0.0,0.0,0.0};
    double p_t[3], R_t[9], u_tau[3], p_[3], R_[9];
    int fsegi, actno, actseg, actno_mn;
    double u_L[3], u_f[3], p_f[3], R_f[9];
    double out_xdot[6], residual[NUM_ACT_SET][6];

    double p_L[3], R_L[9];
    double v1[3], v2[3], v3[3], v_val[3];

    for (int i = 0; i < 3; ++i) {
        tau[i] = 0.0;
        K2invResidual[i] = 0.0;
        u_t[i] = 0.0;
        du[i] = 0.0;
        u_tau[i] = 0.0;
        p_[i] = 0.0;
        u_L[i] = 0.0;
        u_f[i] = 0.0;
        p_f[i] = 0.0;
        out_xdot[i] = 0.0;
        out_xdot[i + 3] = 0.0;
        p_L[i] = 0.0;
        v1[i] = 0.0;
        v2[i] = 0.0;
        v3[i] = 0.0;
        v_val[i] = 0.0;
        p_t[i] = 0.0;
    }
    for (int i = 0; i < 9; ++i) {
        R_t[i] = 0.0;
        R_[i] = 0.0;
        R_f[i] = 0.0;
        R_L[i] = 0.0;
    }

    // root configurations for residual
    double p_d[3], R_d[9];
    for (int i = 0; i < 3; ++i) {
        p_d[i] = Params.xi[i];
    }
    for (int i = 0; i < 9; ++i) {
        R_d[i] = Params.xi[3+i];
    }

    double RigidSegmentLength;
    double net_nL[3];  // placeholder for force applied on the flexible segment

    for (int i = 0; i < 3; ++i) {
        net_nL[i] = 0.0;
    }


    auto &  NUM_SEGMENTS= Params.no_segments;

    for (int segi = NUM_SEGMENTS-1; segi >=0; --segi) { // starting from the last segment
        if ( segi%2 == 0 ) {
            fsegi = segi>>1;

            if(segi == NUM_SEGMENTS-1){ //last segment is flexible
                // assume free tip no torque tau_0 = [0,0,0]
                mMult_AB<3,3,1>( Kinv[fsegi], tau_0, K2invResidual );
                mAdd_AB<3,1>( ustar[fsegi], K2invResidual, u_t );

                for (int i = 0; i < 3; ++i) {
                    p_t[i] = Params.xf[i];
                }
                for (int i = 0; i < 9; ++i) {
                    R_t[i] = Params.xf[3+i];
                }
                CRMFlexible_IVP_Back ( segi, p_t, R_t, Params, u_t , n_0,
                                       u_tau, p_, R_);

                mSub_AB<3,1>( u_tau, ustar[fsegi], du);
                mMult_AB<3,3,1>( K[fsegi], du, tau ); // moment at upper side of the coil

                actno = fsegi - 1;

                mSub_AB<3,1>( m_L[actno], tau, net_mL);  // the net torque applied to the downwards coil

                //copy to the output for forward calculation
                for (int i = 0; i < 3; ++i) {
                    out_tau[i+actno*3] = tau[i];
                }

            }else{ // the non-free tip flexible segments with coils on top

                actno = fsegi;
                mMult_AB<3,3,1>( Kinv[fsegi], m_L[actno], K2invResidual );
                mAdd_AB<3,1>( ustar[fsegi], K2invResidual, u_L );

                actseg = segi + 1; //should be the upper actuator

                //calculate starting positios and roations given last coil states
                RigidSegmentLength	=	SegBounds[actseg+1]-SegBounds[actseg];	// how far we need to move along the length of the rigid segment to reach the next flexible segment

                for (int i = 0; i < 9; ++i) {
                    R_L[i] = out_x_coil[actno][i+9];
                }
                for (int i = 0; i < 3; ++i) {
                    p_L[i] = out_x_coil[actno][i+6] - R_L[ i*3 +2 ]*RigidSegmentLength * 0.5;
                }

                CRMFlexible_IVP_Back ( segi, p_L, R_L, Params, u_L , n_L[actno],
                                       u_f, p_f, R_f);
                actno = fsegi - 1;
//                std::cout << "u_f: " << u_f[0] << " " << u_f[1] << " " << u_f[2] <<  std::endl;
//                std::cout << "p_f: " << p_f[0] << " " << p_f[1] << " " << p_f[2] <<  std::endl;

                if(actno > -1){ //if there is a coil linked below, we need to calculate the net torque again
                    mSub_AB<3,1>( u_f, ustar[fsegi], du);
                    mMult_AB<3,3,1>( K[fsegi], du, tau ); // calculate moment at upper side of the coil

                    mSub_AB<3,1>( m_L[actno], tau, net_mL); // net torque applied at the downwards coil

                    for (int i = 0; i < 3; ++i) {
                        out_tau[i+actno*3] = tau[i];
                    }
                }

            }
        }
        else{

            actno = (segi -1 )>>1;

//            std::cout << "n_L dyn: " << n_L[actno][0] << " " << n_L[actno][1] << " " << n_L[actno][2] << std::endl;
//            std::cout << "net_mL dyn: " << net_mL[0] << " " << net_mL[1] << " " << net_mL[2] << std::endl;

            if(actno< NUM_ACT_SET-1){
                mSub_AB<3,1>( n_L[actno], n_L[actno+1], net_nL);
            }else{
                mSub_AB<3,1>( n_L[actno], n_0, net_nL);
            }

//            for (int i = 0; i < NUM_COIL_STATES; ++i) {
//                std::cout << "x_coil : " << x_coil[actno][i] << std::endl;
//            }
//            for (int i = 0; i < 3; ++i) {
//                std::cout << "net_nL : " << net_nL[i] << std::endl;
//            }
//
//            std::cout << "actMass[actno] : " << actMass[actno] << std::endl;
//
//            for (int i = 0; i < 9; ++i) {
//                std::cout << "actInertia[actno] : " << actInertia[actno][i] << std::endl;
//            }
//
//
//            for (int i = 0; i < 6; ++i) {
//                std::cout << "Params.damping[actno] : " << Params.damping[actno][i] << std::endl;
//            }
//            for (int i = 0; i < 3; ++i) {
//                std::cout << "Params.B0, : " << Params.B0[i] << std::endl;
//            }
//            for (int i = 0; i < 9; ++i) {
//                std::cout << "muhat[actno]: " << muhat[actno][i] << std::endl;
//            }
//
//            for (int i = 0; i < 3; ++i) {
//                std::cout << "net_mL: " << net_mL[i] << std::endl;
//            }


            CoilDynamics(x_coil[actno], net_nL, Params.g, actMass[actno], actInertia[actno],
                         Params.damping[actno], Params.DELTA_T, Params.B0,
                         muhat[actno], net_mL, out_x_coil[actno], out_xdot);

//            for (int i = 0; i < NUM_COIL_STATES; ++i) {
//                std::cout << "out_xdot" << out_xdot[i] << std::endl;
//            }
//            std::cout << "------------------------------" << std::endl;
            // compute the residual against the last flexible segment above
            if(actno<NUM_ACT_SET-1){
                RigidSegmentLength	=	SegBounds[segi+1]-SegBounds[segi];	// how far we need to move along the length of the rigid segment to reach the next flexible segment

                //indices for the residual
                actno_mn = actno + 1;

                for (int i = 0; i < 9; ++i) {
                    R_L[i] = out_x_coil[actno][i+9];
                }
                for (int i = 0; i < 3; ++i) {
                    p_L[i] = out_x_coil[actno][i+6] + R_L[ i*3 +2 ]*RigidSegmentLength * 0.5;
                }
                for (int i = 0; i < 3; ++i) {
                    residual[actno_mn][i] = (p_f[i] - p_L[i]);
                }
                for (int i = 0; i < 3; ++i) {
                    v1[i] = R_f[i*3] - R_L[i*3];
                    v2[i] = R_f[1+ i*3] - R_L[1+ i*3];
                    v3[i] = R_f[2 + i*3]  - R_L[2 + i*3];
                }

                v_val[0] = vNormSq<3>(v1);
                v_val[1] = vNormSq<3>(v2);
                v_val[2] = vNormSq<3>(v3);
                for (int i = 0; i < 3; ++i) residual[actno_mn][i+3] = sqrt(v_val[i]);
            }

        }

    }

    mCopy_AB<3>(u_f, out_u0);

    // last segment
    actno = 0;
    for (int i = 0; i < 3; ++i) {
        residual[actno][i] = (p_f[i] - p_d[i]);
    }
    // calculate vector norm of the rotation matrices
    for (int i = 0; i < 3; ++i) {
        v1[i] = R_f[i*3] - R_d[i*3];
        v2[i] = R_f[1+ i*3] - R_d[1+ i*3];
        v3[i] = R_f[2 + i*3]  - R_d[2 + i*3];
    }

    v_val[0] = vNormSq<3>(v1);
    v_val[1] = vNormSq<3>(v2);
    v_val[2] = vNormSq<3>(v3);
    for (int i = 0; i < 3; ++i) residual[actno][i+3] = sqrt(v_val[i]);


//     for (int i = 0; i < NUM_ACT_SET; ++i) {
//         std::cout << "residual p: " << residual[i][0] << " " << residual[i][1] << " " <<residual[i][2] << std::endl;
//         std::cout << "residual R: " << residual[i][3] << " " << residual[i][4] << " " <<residual[i][5] << std::endl;
//     }
//
//     std::cout << " --------------------------------- " << std::endl;


    for (int i = 0; i < NUM_ACT_SET; ++i) {
        for (int j = 0; j < 3; ++j) {
            out_y[j + i*6] = RESIDUAL_SCALE_P * residual[i][j];
        }
        for (int j = 3; j < 6; ++j) {
            out_y[j + i*6] = RESIDUAL_SCALE_R * residual[i][j]; // 10000000000
        }
    }
//
//    std::cout << "out_y: " << out_y[0] << " " << out_y[1] << " " << out_y[2] <<  std::endl;
//    std::cout << "out_y: " << out_y[3] << " " << out_y[4] << " " << out_y[5] <<  std::endl;
//    std::cout << "out_y: " << out_y[6] << " " << out_y[7] << " " << out_y[8] <<  std::endl;
//    std::cout << "out_y: " << out_y[9] << " " << out_y[10] << " " << out_y[11] <<  std::endl;
}


//
//
//	NUMERICAL INTEGRATION FUNCTIONS
//
//
void CRMFlexible_IVP_Back ( int SegmentIndex, const double in_p[3], const double in_R[9],  DYNNLEqnParams& in_params,
                            const double in_u[3], const double in_n_L[3],
                            double out_u[3], double out_p[3], double out_R[9]){

//    std::cout << "in_u: " << in_u[0][0] << " " <<  in_u[0][1] << " " <<  in_u[0][2] << std::endl;
//    std::cout << "in_u: " << in_u[1][0] << " " <<  in_u[1][1] << " " <<  in_u[1][2] << std::endl;
    double n_L[3];
    for (int i = 0; i < 3; ++i) {
        n_L[i] = in_n_L[i];
    }

//    double xi[NUM_STATES];  			//  Initial value of the state for the next segment to be integrated
//    double xf[NUM_STATES];			//  Final value of the state for the last segment integrated

    double h;

    double l_zero[3]={0.0,0.0,0.0};
    double RigidSegmentLength;		// Length of the rigid segment - intermediate variable
    int   fsegno; 					// flexible segment no

    // we need to copy in_ftip to local variable
    double ftip[3] = {0.0,0.0,0.0}; //placeholder
    // we will copy anything we will access more than once (or write to) to local variables
    int	  NextLocMarker=in_params.NextLocMarker;
    int	  InitialLocMarker=NextLocMarker;
    // for others, we will create aliases
    auto & SegBounds = in_params.SegBounds;
    auto & SegSteps = in_params.SegSteps;

    auto & InsertedLength = in_params.InsertedLength;
    auto & dlambdainv = in_params.dlambdainv;
    auto & K = in_params.K;
    auto & Kinv = in_params.Kinv;
    auto & ustar = in_params.ustar;
    auto & fcumlambda = in_params.fcumlambda;
    auto & FinalValueOnly = in_params.FinalValueOnly;
    auto & LocMarkers = in_params.LocMarkers;
    auto & p_atLocMarkers = in_params.p_atLocMarkers;
    auto & no_locmarkers = in_params.no_locmarkers;

    // Flexible Segment
    // Prepare the CRMIntegrand Parameters
    fsegno=SegmentIndex>>1; // i/2, flexible segment no
    h= -1* (SegBounds[SegmentIndex+1] - SegBounds[SegmentIndex] )/(SegSteps[fsegno]*1.0);


    StateVector xi_statevec;  		//  Initial value of the state for the next segment to be integrated
    StateVector xf_statevec;  		//  Initial value of the state for the next segment to be integrated


    mCopy_AB<9>(in_R, xi_statevec._R);
    mCopy_AB<3>(in_p, xi_statevec._p);
    mCopy_AB<3>(in_u, xi_statevec._u);


    double DeltaPE = 0.0;
    auto& CalculateEnergy = in_params.CalculateEnergy;
    auto& no_fcum_steps = in_params.no_fcum_steps;
    auto& g = in_params.g;
    auto& rho = in_params.rho;

    // define the structure that will used to pass parameters to the CRMIntegrand
    CRMIntegrandParams IntegrandParams;
    // these parameters are same for all segments
    IntegrandParams.dlambdainv = dlambdainv;
    IntegrandParams.Li = InsertedLength;
    IntegrandParams.l = l_zero;  // we are assuming the distributed moment on the catheter body is zero
    IntegrandParams.no_fcum_steps = no_fcum_steps;
    IntegrandParams.fcumlambda = fcumlambda;

    IntegrandParams.ftip = ftip;
    IntegrandParams.g = g;

    // assign the parameters that vary from segment to segment
    IntegrandParams.K = K[fsegno];
    IntegrandParams.Kinv = Kinv[fsegno];
    IntegrandParams.ustar = ustar[fsegno];
    IntegrandParams.rho = rho[SegmentIndex];

    // Integrate
    ABM4_dyn(xi_statevec, SegBounds[SegmentIndex+1], SegSteps[fsegno], h,
         IntegrandParams, n_L, no_locmarkers, CalculateEnergy,
         FinalValueOnly, LocMarkers, NextLocMarker,
         xf_statevec, DeltaPE, p_atLocMarkers);

    for (int j=0; j < 3; j++) {
        out_p[j]= xf_statevec._p[j];
        out_u[j] = xf_statevec._u[j];
    }
    for (int j = 0; j < 9; ++j) {
        out_R[j] = xf_statevec._R[j];
    }



}


void CRMFlexForward_pass (  int SegmentIndex, const double in_p[3], const double in_R[9],  CRMIVPCoreParams in_params,
                            const double in_u[3], const double in_n_L[3],
                            double out_u[3], double out_p[3], double out_R[9], double out_p_atLocMarkers[][3]){

    double n_L[3];
    for (int i = 0; i < 3; ++i) {
        n_L[i] = in_n_L[i];
    }
//    double xi[NUM_STATES];  		//  Initial value of the state for the next segment to be integrated
//    double xf[NUM_STATES];			//  Final value of the state for the last segment integrated
    double h;
    int   fsegno; 					// flexible segment no

    // we need to copy in_ftip to local variable
    double ftip[3] = {0.0,0.0,0.0}; //placeholder
    // we will copy anything we will access more than once (or write to) to local variables
    int	  NextLocMarker=in_params.NextLocMarker;
    int	  InitialLocMarker=NextLocMarker;
    // for others, we will create aliases
    auto & SegBounds = in_params.SegBounds;
    auto & SegSteps = in_params.SegSteps;

    auto & InsertedLength = in_params.InsertedLength;
    auto & dlambdainv = in_params.dlambdainv;
    auto & K = in_params.K;
    auto & Kinv = in_params.Kinv;
    auto & ustar = in_params.ustar;
    auto & fcumlambda = in_params.fcumlambda;
    auto & FinalValueOnly = in_params.FinalValueOnly;
    auto & LocMarkers = in_params.LocMarkers;
    auto & p_atLocMarkers = in_params.p_atLocMarkers;
    auto & no_locmarkers = in_params.no_locmarkers;


    // Prepare the CRMIntegrand Parameters
    fsegno=SegmentIndex>>1; // i/2, flexible segment no
    h=(SegBounds[SegmentIndex+1]-SegBounds[SegmentIndex])/(SegSteps[fsegno]*1.0);

    StateVector xi_statevec;  		//  Initial value of the state for the next segment to be integrated
    StateVector xf_statevec;  		//  Initial value of the state for the next segment to be integrated

    mCopy_AB<9>(in_R, xi_statevec._R);
    mCopy_AB<3>(in_p, xi_statevec._p);
    mCopy_AB<3>(in_u, xi_statevec._u);

    double DeltaPE = 0.0;
    auto& CalculateEnergy = in_params.CalculateEnergy;
    auto& no_fcum_steps = in_params.no_fcum_steps;
    auto& g = in_params.g;
    auto& rho = in_params.rho;

    double l_zero[3] = { 0.0,0.0,0.0 };


    // define the structure that will used to pass parameters to the CRMIntegrand
    CRMIntegrandParams IntegrandParams;
    // these parameters are same for all segments
    IntegrandParams.dlambdainv = dlambdainv;
    IntegrandParams.Li = InsertedLength;
    IntegrandParams.l = l_zero;  // we are assuming the distributed moment on the catheter body is zero
    IntegrandParams.no_fcum_steps = no_fcum_steps;
    IntegrandParams.fcumlambda = fcumlambda;
    IntegrandParams.ftip = ftip;
    IntegrandParams.g = g;

    // assign the parameters that vary from segment to segment
    IntegrandParams.K = K[fsegno];
    IntegrandParams.Kinv = Kinv[fsegno];
    IntegrandParams.ustar = ustar[fsegno];
    IntegrandParams.rho = rho[SegmentIndex];


    // Integrate
    ABM4_dyn(xi_statevec, SegBounds[SegmentIndex], SegSteps[fsegno], h,
         IntegrandParams, n_L, no_locmarkers, CalculateEnergy,
         FinalValueOnly, LocMarkers, NextLocMarker,
         xf_statevec, DeltaPE, p_atLocMarkers);


    //then copy the marker locations to the output
    for (int i = 0; i < no_locmarkers; i++) {
        for (int j = 0; j < 3; j++) {
            out_p_atLocMarkers[i][j] = p_atLocMarkers[(no_locmarkers - 1) - i][j];
        }
    }

    // pass the outputs
    for (int j=0; j < 3; j++) {
        out_p[j]=xf_statevec._p[j];
        out_u[j]=xf_statevec._u[j];
    }
    for (int j = 0; j < 9; ++j) {
        out_R[j] = xf_statevec._R[j];
    }

}


void CRMIVP_DYN(	 CRMIVPCoreParams& CoreParams, const double in_u0[3], const double in_p0[3], const double in_R0[9],
                     const double in_mL[NUM_ACT_SET][3], const double in_nL[NUM_ACT_SET][3], const double in_tau[NUM_ACT_SET][3], const double in_ftip[3],
                     double out_coil_state[NUM_ACT_SET][NUM_COIL_STATES],  double out_u_new[3], double out_p_new[3], double out_R_new[9],
                     double out_p_atLocMarkers[][3]) {

    double x_coil[NUM_ACT_SET][NUM_COIL_STATES];
    double MagMoment[NUM_ACT_SET][3], muhat[NUM_ACT_SET][9], actMass[NUM_ACT_SET], actInertia[NUM_ACT_SET][9];

    double mu[3], muhattemp[9];
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

        for (int i = 0; i <3 ; ++i) {
            MagMoment[j][i] = CoreParams.MagMoment[j][i];
        }
        for (int i = 0; i < 3; ++i) {
            mu[i] =  MagMoment[j][i];
        }
        wHat(mu,muhattemp);
        for (int i = 0; i < 9; ++i) {
            muhat[j][i] = muhattemp[i];
        }
    }

    double u_new[3], p_new[3], R_new[9];

    int   actno, fsegip1;	// actuator no, flexible segment before, flexible segment after
    double RscTB0[3], Tb[3], Residual[3], inertiaw[3];
    auto & K = CoreParams.K;
    auto & Kinv = CoreParams.Kinv;
    auto & ustar = CoreParams.ustar;
    auto & B0 = CoreParams.B0;
    auto & SegBounds = CoreParams.SegBounds;
    double w[3], w_dot[3], w_hat[9], w_inertia_w[3], inertia_wdot[3], out_xdot[6], w_terms[3], Tb_ml[3], K2invResidual[3];

    double u_0[3], p0[3], R0[9], n_L[NUM_ACT_SET][3], m_L[NUM_ACT_SET][3], tau[NUM_ACT_SET][3];

    mCopy_ABm<NUM_ACT_SET, 3>(in_nL, n_L);
    mCopy_ABm<NUM_ACT_SET, 3>(in_mL, m_L);
    mCopy_ABm<NUM_ACT_SET, 3>(in_tau, tau);

    for (int i = 0; i < 3; ++i) {
        u_0[i] = in_u0[i];
        p0[i] = in_p0[i];
    }
    for (int i = 0; i < 9; ++i) {
        R0[i] = in_R0[i];
    }

    double RigidSegmentLength;

    double n_f[3], net_nL[3], net_mL[3];

//    double p_atLocMarkers[NUM_LOCALIZATION_MARKERS][3];
    auto & p_atLocMarkers = CoreParams.p_atLocMarkers;
    auto & no_locmarkers = CoreParams.no_locmarkers;

    auto & NUM_SEGMENTS = CoreParams.no_segments;
    double update_coil_state[NUM_COIL_STATES];
    //forward pass
    for (int SegmentIndex = 0; SegmentIndex < NUM_SEGMENTS; ++SegmentIndex) {
        if(SegmentIndex %2 == 0){

            actno = SegmentIndex >> 1;

            if(SegmentIndex == NUM_SEGMENTS -1){ // if this is free tip flexible segment
                for (int i = 0; i < 3; ++i) {
                    n_f[i] = in_ftip[i];
                }
//                n_f[0] = n_f[1] = n_f[2] = 0.0;
            }else{
                for (int i = 0; i < 3; ++i) {
                    n_f[i] =  n_L[actno][i];
                }
            }
            CRMFlexForward_pass (  SegmentIndex, p0, R0,  CoreParams, u_0 , n_f, u_new, p_new, R_new, p_atLocMarkers);
        }else{
            RigidSegmentLength=(SegBounds[SegmentIndex+1]-SegBounds[SegmentIndex]);

            actno=(SegmentIndex-1)>>1;		// actuator no
            fsegip1=actno+1;

            if(actno< NUM_ACT_SET-1){
                mSub_AB<3,1>( n_L[actno], n_L[actno+1], net_nL);
            }else{
                mSub_AB<3,1>( n_L[actno], in_ftip, net_nL);
            }
            mSub_AB<3,1>( m_L[actno], tau[actno], net_mL); // net torque applied at the downwards coil

            CoilDynamics(x_coil[actno], net_nL, CoreParams.g, actMass[actno], actInertia[actno], CoreParams.damping[actno], CoreParams.DELTA_T,
                         CoreParams.B0, muhat[actno], net_mL, update_coil_state, out_xdot);

            for (int i = 0; i < 3; ++i) {
                w[i] = update_coil_state[i+3];
                w_dot[i] = out_xdot[i+3];
            }
            wHat(w,w_hat);

            mMult_ATB<3,3,1>(&(update_coil_state[9]),B0,RscTB0);
            mMult_AB<3,3,1>(muhat[actno],RscTB0,Tb);

            mMult_AB<3,3,1>(actInertia[actno], w, inertiaw);
            mMult_AB<3,3,1>(w_hat, inertiaw, w_inertia_w);
            mMult_AB<3,3,1>(actInertia[actno], w_dot, inertia_wdot);
            mAdd_AB<3,1>(inertia_wdot, w_inertia_w, w_terms);

            mSub_AB<3,1>( Tb, m_L[actno], Tb_ml);
            mSub_AB<3,1>( Tb_ml, w_terms, Residual);

            mMult_AB<3,3,1>( Kinv[fsegip1], tau[actno], K2invResidual );
            mAdd_AB<3,1>( ustar[fsegip1], K2invResidual, u_0 );

            for (int i = 0; i < 3; ++i) {
                p0[i] = update_coil_state[i+6] + 0.5*RigidSegmentLength * update_coil_state[9+ i*3+2 ];
            }

            for (int i = 0; i < 9; ++i) {
                R0[i] = update_coil_state[i+9];
            }

            for (int i = 0; i < NUM_COIL_STATES; ++i) {
                out_coil_state[actno][i] = update_coil_state[i];
            }

        }
    }

    // return tip state
    for (int i = 0; i < 3; ++i) {
        out_u_new[i] = u_new[i];
        out_p_new[i] = p_new[i];
    }
    for (int i = 0; i < 9; ++i) {
        out_R_new[i] = R_new[i];
    }

    // Copy marker locations to the output
    if (!CoreParams.FinalValueOnly) {
        for (int i=0; i<no_locmarkers; i++) {
            for (int j=0; j<3; j++) {
                out_p_atLocMarkers[i][j]=p_atLocMarkers[(no_locmarkers-1)-i][j];
            }
        }
    }

}

void DynamicsBVP(	CRMShootingMethodParams& in_Params, const double xf[NUM_STATES],
                     double in_mL_initialguess[NUM_ACT_SET][3], double in_nL_initialguess[NUM_ACT_SET][3], double in_ftip_initialguess[3],
                     double out_u0[3], double out_mL[NUM_ACT_SET][3], double out_nL[NUM_ACT_SET][3], double out_tau[NUM_ACT_SET][3], double out_ftip[3], int& out_localmin) {

    ContactModeType ContactMode = in_Params.ContactMode;
    int NLEq_Dim;  // Dimension of the Nonlinear Equation to Solve
    NLEq_Dim = NUM_DYN_RESIDUAL;

    // Call CRMSolverIVP_Prep, to pre-process parameters
    double x_0[NUM_STATES];
    for (int i = 0; i < NUM_STATES; i++) {
        if (i < 3) {
            x_0[i] = in_Params.p0[i];
        }
        else if (i < 12) {
            x_0[i] = in_Params.R0[i - 3];
        }
        else if (i < 15) {
            x_0[i] = 0.0; //placeholder
        }
    }



    bool FinalValueOnly = true;
    DYNNLEqnParams DYNNLEParams(in_Params.no_flex_seg, in_Params.no_rigid_seg, in_Params.no_act_set, in_Params.no_locmarkers, in_Params.no_fcum_steps);


    CRMDYNSolverIVP_Prep(in_Params.no_flex_seg,in_Params.no_rigid_seg, in_Params.no_act_set, in_Params.no_locmarkers, in_Params.no_fcum_steps,
                         x_0, in_Params.IntegrationStepSize,
                         in_Params.Li, in_Params.dlambdainv, in_Params.rho, in_Params.SegmentTypes,
                         in_Params.SegEndLambdas, in_Params.LocMarkerLambdas,
                         in_Params.K, in_Params.Kinv, in_Params.ustar,
                         in_Params.MagMoment, in_Params.fcumlambda, in_Params.CoilAlignmentTurnAreaMatrix,
                         in_Params.B0, in_Params.g, in_Params.ActMass, in_Params.actInertia, in_Params.damping, in_Params.DELTA_T,
                         in_Params.v_L_pre, in_Params.w_L_pre, in_Params.p_pre, in_Params.R_pre,
                         in_mL_initialguess, in_nL_initialguess, FinalValueOnly, DYNNLEParams);

    DYNNLEParams.ContactMode = ContactMode;
    mCopy_AB<3>(in_Params.TipConstraintPoint, DYNNLEParams.TipConstraintPoint);
    mCopy_AB<3>(in_ftip_initialguess, DYNNLEParams.ftip_initialguess);

    for (int i = 0; i < NUM_STATES; ++i) {
        DYNNLEParams.xf[i] = xf[i];
    }

    // Scale parameters and call the nonlinear equation solver
//    const double uscaleinv = 1.0 / IVALUE_SCALE_U;
    const double nscaleinv = 1.0 / IVALUE_SCALE_N;
    const double mscaleinv = 1.0 / IVALUE_SCALE_M;

//    const double fscaleinv = 1.0 / IVALUE_SCALE_F;
    auto* initialguessscaled = new double [NLEq_Dim];
    auto* returnedparamscaled = new double [NLEq_Dim];

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; i++) {
            initialguessscaled[i + j*6] = mscaleinv * in_mL_initialguess[j][i];
            initialguessscaled[i + j*6 +3] = nscaleinv * in_nL_initialguess[j][i];
        }
    }

    for (int i = 0; i < 3; ++i) {
        DYNNLEParams.TipForce[i] = in_Params.TipForce[i];  // if the catheter is not in contact, the tip force specified within in_Params needs to be used; this would not be scaled as it is not changed by the solver
    }

    int localmin = 0;


    double* x = new double[NLEq_Dim]; // we will create a new variable here and not use initial guess scaled since truss-region-dogleg algorithm uses the same variable for both input and output

    double* residual = new double[NLEq_Dim];
    int info;
    int lwa = (NLEq_Dim * (3 * NLEq_Dim + 13)) / 2;
    double tol = 0.00001;
    double* wa = new double [lwa];
    for (int i = 0; i < NLEq_Dim; i++) x[i] = initialguessscaled[i];

//    int REPS = 100;
//    // Get starting timepoint
//    auto start = high_resolution_clock::now();
//    for (int cnt = 0; cnt < REPS; cnt++)
//        DYNNLEquation(x, returnedparamscaled, DYNNLEParams, out_u0);
//    auto stop = high_resolution_clock::now();
//    auto duration = duration_cast<microseconds>(stop - start);
//    std::cout << std::endl << "Average time taken by DYNNLEquation Solution in " << REPS << " repetitions: " << duration.count() / REPS << " microseconds" << std::endl;

    double tau[NUM_ACT_SET*3];


#if defined( TRUSTREGION_DYN )
    TrustRegionDogleg_dyn(NLEq_Dim, x, residual, tol, info, wa, lwa, DYNNLEParams, out_u0, tau);
#else // undefined
    exit(1);
#endif

    localmin = (info == 1) ? 0 : (info - 1);
    for (int i = 0; i < NLEq_Dim; i++) returnedparamscaled[i] = x[i];
    delete[] residual;
    delete[] x;
    delete[] wa;

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; i++) {
            out_mL[j][i] = IVALUE_SCALE_M * returnedparamscaled[i + j * 6];
            out_nL[j][i] = IVALUE_SCALE_N * returnedparamscaled[i + j * 6 + 3];
        }

        for (int i = 0; i < 3; ++i) {
            out_tau[j][i] = tau[i+j*3];
        }
    }

//    for (int i = 0; i < NUM_ACT_SET; ++i) {
//        std::cout << "out_mL: " << out_mL[i][0] << " " << out_mL[i][1] << " " <<out_mL[i][2] << std::endl;
//        std::cout << "out_nL: " << out_nL[i][0] << " " << out_nL[i][1] << " " <<out_nL[i][2] << std::endl;
//    }

    for (int i = 0; i < 3; ++i) {
        out_ftip[i] = in_Params.TipForce[i];  // if it is free-tip, return the tip force specified within in_Params
    }

    out_localmin = localmin;
    delete[] initialguessscaled;
    delete[] returnedparamscaled;

}


void DYNSolverIVP(	CRMShootingMethodParams& in_Params, const double in_u0[3],
                      const double in_mL[NUM_ACT_SET][3], const double in_nL[NUM_ACT_SET][3], const double in_tau[NUM_ACT_SET][3], const double in_ftip[3],
                      bool in_FinalValueOnly,
                      double out_x_N[NUM_STATES], double out_coil_state[NUM_ACT_SET][NUM_COIL_STATES],
                      double out_p_atLocMarkers[][3]){

    double x_0[NUM_STATES];
    for (int i = 0; i < NUM_STATES; i++) {
        if (i < 3) x_0[i] = in_Params.p0[i];
        else if (i < 12) x_0[i] = in_Params.R0[i - 3];
        else if (i < 15) x_0[i] = in_u0[i];
    }

    CRMIVPCoreParams CoreParams(in_Params.no_flex_seg, in_Params.no_rigid_seg, in_Params.no_act_set, in_Params.no_locmarkers, in_Params.no_fcum_steps);
    double u_0[3], n_L[NUM_ACT_SET][3], m_L[NUM_ACT_SET][3],  tau[NUM_ACT_SET][3], ftip[3];
    // We need to pass u_0 as input argument as the values in CoreParams will be overriden with the values provided in the input arguments - functionality needed for solving Boundary Value Problems (BVP)
    // copy to local variable
    for (int i = 0; i < 3; i++) u_0[i] = in_u0[i];
    for (int i = 0; i < 3; i++) ftip[i] = in_ftip[i];

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; i++) n_L[j][i] = in_nL[j][i];
        for (int i = 0; i < 3; i++) m_L[j][i] = in_mL[j][i];
        for (int i = 0; i < 3; i++) tau[j][i] = in_tau[j][i];
    }


//    std::cout << "w_L_pre: " << DYNNLEParams.w_L_pre[0][0] << " " << DYNNLEParams.w_L_pre[0][1] << " " << DYNNLEParams.w_L_pre[0][2] <<  std::endl;
//
//    std::cout << "pL_pre: " << DYNNLEParams.p_pre[0][0] << " " << DYNNLEParams.p_pre[0][1] << " " << DYNNLEParams.p_pre[0][2] <<  std::endl;
//    std::cout << "xf_pre: " << DYNNLEParams.xf[0] << " " << DYNNLEParams.xf[1] << " " << DYNNLEParams.xf[2] <<  std::endl;
//    std::cout << "InsertedLength: " << DYNNLEParams.InsertedLength <<  std::endl;
//    std::cout << "damping_: " << DYNNLEParams.damping[0][0] << " " << DYNNLEParams.damping[0][1] << " " << DYNNLEParams.damping[0][2] <<  std::endl;
//    std::cout << "ActMass: " <<  DYNNLEParams.ActMass[0] <<  std::endl;

    CRMDYNSolverIVP_Prep(in_Params.no_flex_seg,in_Params.no_rigid_seg, in_Params.no_act_set, in_Params.no_locmarkers, in_Params.no_fcum_steps,
                         x_0, in_Params.IntegrationStepSize,
                         in_Params.Li, in_Params.dlambdainv, in_Params.rho, in_Params.SegmentTypes,
                         in_Params.SegEndLambdas, in_Params.LocMarkerLambdas,
                         in_Params.K, in_Params.Kinv, in_Params.ustar,
                         in_Params.MagMoment, in_Params.fcumlambda, in_Params.CoilAlignmentTurnAreaMatrix,
                         in_Params.B0, in_Params.g, in_Params.ActMass, in_Params.actInertia, in_Params.damping, in_Params.DELTA_T,
                         in_Params.v_L_pre, in_Params.w_L_pre, in_Params.p_pre, in_Params.R_pre,
                         m_L, n_L, in_FinalValueOnly, CoreParams);

    double u_new[3], p_new[3], R_new[9];
    CRMIVP_DYN(	 CoreParams, u_0, in_Params.p0, in_Params.R0, m_L, n_L, tau, ftip, out_coil_state, u_new, p_new, R_new, out_p_atLocMarkers);

//    for (int i = 0; i < NUM_ACT_SET; ++i) {
//        std::cout << "w: " << out_coil_state[i][3] << " " << out_coil_state[i][4] << " " <<out_coil_state[i][5] << std::endl;
//        std::cout << "v: " << out_coil_state[i][0] << " " << out_coil_state[i][1] << " " <<out_coil_state[i][2] << std::endl;
//    }
    for (int i = 0; i < NUM_STATES; ++i) {
        if(i<3){
            out_x_N[i] = p_new[i];
        }else if(i<3+9){
            out_x_N[i] = R_new[i-3];
        }else{
            out_x_N[i] = u_new[i-3-9];
        }
    }

}

void DYNSolverIVP_JacobiansFullstate(
        CRMShootingMethodParams& in_Params,
        const double in_u0[3],
        const double in_mL[NUM_ACT_SET][3],
        const double in_nL[NUM_ACT_SET][3],
        const double in_tau[NUM_ACT_SET][3],
        const double in_ftip[3],
        const double in_x_coil[NUM_ACT_SET][NUM_COIL_STATES],
        const double in_xf[NUM_STATES],
        Eigen::MatrixXd& J_xf_y,
        Eigen::MatrixXd& J_xf_x,
        Eigen::MatrixXd& J_xcoil_y,
        Eigen::MatrixXd& J_xcoil_x) {

    const int dim_y = NUM_ACT_SET * 6;
    const int dim_x = NUM_ACT_SET * NUM_COIL_STATES + NUM_STATES;
    const int dim_xcoil = NUM_ACT_SET * NUM_COIL_STATES;

    J_xf_y.resize(NUM_STATES, dim_y);
    J_xf_x.resize(NUM_STATES, dim_x);
    J_xcoil_y.resize(dim_xcoil, dim_y);
    J_xcoil_x.resize(dim_xcoil, dim_x);
    J_xf_y.setZero();
    J_xf_x.setZero();
    J_xcoil_y.setZero();
    J_xcoil_x.setZero();

    Dual u0_dual[3];
    for (int i = 0; i < 3; ++i) {
        u0_dual[i] = Dual(in_u0[i], 0.0);
    }

    Dual tau_dual[NUM_ACT_SET][3];
    Dual ftip_dual[3];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            tau_dual[j][i] = Dual(in_tau[j][i], 0.0);
        }
    }
    for (int i = 0; i < 3; ++i) {
        ftip_dual[i] = Dual(in_ftip[i], 0.0);
    }

    Dual x_coil_base[NUM_ACT_SET][NUM_COIL_STATES];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < NUM_COIL_STATES; ++i) {
            x_coil_base[j][i] = Dual(in_x_coil[j][i], 0.0);
        }
    }

    Dual xf_base[NUM_STATES];
    for (int i = 0; i < NUM_STATES; ++i) {
        xf_base[i] = Dual(in_xf[i], 0.0);
    }

    for (int col = 0; col < dim_y; ++col) {
        Dual mL_dual[NUM_ACT_SET][3];
        Dual nL_dual[NUM_ACT_SET][3];
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < 3; ++i) {
                mL_dual[j][i] = Dual(in_mL[j][i], 0.0);
                nL_dual[j][i] = Dual(in_nL[j][i], 0.0);
            }
        }

        int act = col / 6;
        int offset = col % 6;
        if (offset < 3) {
            mL_dual[act][offset] = Dual(in_mL[act][offset], 1.0);
        } else {
            nL_dual[act][offset - 3] = Dual(in_nL[act][offset - 3], 1.0);
        }

        Dual out_xf_next[NUM_STATES];
        Dual out_x_coil_next[NUM_ACT_SET][NUM_COIL_STATES];

        DYNSolverIVP_T<Dual>(in_Params, u0_dual, mL_dual, nL_dual, tau_dual, ftip_dual,
                             x_coil_base, out_xf_next, out_x_coil_next);

        for (int i = 0; i < NUM_STATES; ++i) {
            J_xf_y(i, col) = out_xf_next[i].deriv;
        }
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < NUM_COIL_STATES; ++i) {
                int row = j * NUM_COIL_STATES + i;
                J_xcoil_y(row, col) = out_x_coil_next[j][i].deriv;
            }
        }
    }

    for (int col = 0; col < dim_x; ++col) {
        Dual mL_dual[NUM_ACT_SET][3];
        Dual nL_dual[NUM_ACT_SET][3];
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < 3; ++i) {
                mL_dual[j][i] = Dual(in_mL[j][i], 0.0);
                nL_dual[j][i] = Dual(in_nL[j][i], 0.0);
            }
        }

        Dual x_coil_dual[NUM_ACT_SET][NUM_COIL_STATES];
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < NUM_COIL_STATES; ++i) {
                x_coil_dual[j][i] = Dual(in_x_coil[j][i], 0.0);
            }
        }

        Dual xf_dual[NUM_STATES];
        for (int i = 0; i < NUM_STATES; ++i) {
            xf_dual[i] = Dual(in_xf[i], 0.0);
        }

        if (col < dim_xcoil) {
            int act = col / NUM_COIL_STATES;
            int offset = col % NUM_COIL_STATES;
            x_coil_dual[act][offset] = Dual(in_x_coil[act][offset], 1.0);
        } else {
            int xf_idx = col - dim_xcoil;
            xf_dual[xf_idx] = Dual(in_xf[xf_idx], 1.0);
        }

        Dual out_xf_next[NUM_STATES];
        Dual out_x_coil_next[NUM_ACT_SET][NUM_COIL_STATES];

        DYNSolverIVP_T<Dual>(in_Params, u0_dual, mL_dual, nL_dual, tau_dual, ftip_dual,
                             x_coil_dual, out_xf_next, out_x_coil_next);

        for (int i = 0; i < NUM_STATES; ++i) {
            J_xf_x(i, col) = out_xf_next[i].deriv;
        }
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < NUM_COIL_STATES; ++i) {
                int row = j * NUM_COIL_STATES + i;
                J_xcoil_x(row, col) = out_x_coil_next[j][i].deriv;
            }
        }
    }
}

void CRMDYNSolverIVP_Prep (
        int32_t in_no_flex_seg,
        int32_t in_no_rigid_seg,
        int32_t in_no_act_set,
        int32_t in_no_locmarkers,
        int32_t in_no_fcum_steps,
        double in_x_0[NUM_STATES], double in_IntegrationStepSize,
        double in_Li, double in_dlambdainv,
        double in_rho[/*in_no_flex_seg+in_no_rigid_seg*/],
        const CatheterSegmentType in_SegmentTypes[/*in_no_flex_seg+in_no_rigid_seg*/],
        double in_SegEndLambdas[/*no_segments=no_flex_seg+no_rigid_seg*/], double in_LocMarkerLambdas[/*no_locmarkers*/],
        double in_K[][9], double in_Kinv[][9], double in_ustar[][3],
        double in_MagMoment[][3], double in_fcumlambda[][3], 		double in_CoilAlignmentTurnAreaMatrix[/*in_no_act_set*/][9],
        double in_B0[3], double in_g[3], const double in_actMass[], double in_actInertia[][9],
        double in_damping[NUM_ACT_SET][6], double in_delta_t,
        double in_v_L_pre[NUM_ACT_SET][3], double in_w_L_pre[NUM_ACT_SET][3], double in_p_pre[NUM_ACT_SET][3], double in_R_pre[NUM_ACT_SET][9],
        double in_mL[NUM_ACT_SET][3], double in_nL[NUM_ACT_SET][3],
        bool in_FinalValueOnly, CRMIVPCoreParams &out_CoreParams) {

    // Process the incoming parameters (including changing from distal-proximal order to proximal-distal order)
    //   and package them to be passed to CRMSolverIVP_Core
    // Everything is copied to local variables (inside out_CoreParams), therefore, this function
    //   can be used to transfer data from global memory to device memory for subsequent computations
    // The packaged data can be used to call CRMSolverIVP_Core multiple times by changing
    //   only u[0..2] components of xi --- other parameters should not change

    int32_t in_no_segments = in_no_flex_seg + in_no_rigid_seg;

    double *SegEndLambdas = new double[in_no_segments];
    for (int i = 0; i < in_no_segments; i++) SegEndLambdas[i] = in_SegEndLambdas[i];

    // create aliases for variables in CoreParams
    auto & xi = out_CoreParams.xi;

    auto & SegBounds = out_CoreParams.SegBounds;
    auto & SegSteps = out_CoreParams.SegSteps;
    auto& rho = out_CoreParams.rho;
    auto& CoilAlignmentTurnAreaMatrix = out_CoreParams.CoilAlignmentTurnAreaMatrix;
    auto& FlexActIndex = out_CoreParams.FlexActIndex;

    auto & InsertedLength = out_CoreParams.InsertedLength;
    auto & dlambdainv = out_CoreParams.dlambdainv;
    auto & K = out_CoreParams.K;
    auto & Kinv = out_CoreParams.Kinv;
    auto & ustar = out_CoreParams.ustar;
    auto & fcumlambda = out_CoreParams.fcumlambda;
//	auto & ftip = out_CoreParams.ftip;
    auto & FinalValueOnly = out_CoreParams.FinalValueOnly;
    auto & LocMarkers = out_CoreParams.LocMarkers;

    auto & B0 = out_CoreParams.B0;
    auto & MagMoment = out_CoreParams.MagMoment;

    auto & StartSegmentIndex = out_CoreParams.StartSegmentIndex;
    auto & NextLocMarker = out_CoreParams.NextLocMarker;
    auto & p_atLocMarkers = out_CoreParams.p_atLocMarkers;
    auto& ActMass = out_CoreParams.ActMass;

    // local variables
    double IntegrationStepSize=in_IntegrationStepSize;
    double DeltaSInv=1.0/IntegrationStepSize;

    // process parameters as needed and copy into CoreParams
    mCopy_AB<NUM_STATES>(in_x_0,xi);		// Initial value of the state for the next segment to be integrated
    //  States are packed u[0..2],R[0..8],p[0..2]  (R: 3x3 matrix stored in row major order R11 R12 R13 R21 R22 R23 R31 R32 R33)
    mCopy_AB<3>(in_B0,B0);				//  B0 field vector of the MRI scanner (in spatial coordinates)

    dlambdainv=in_dlambdainv;				// reciprocal of dlambda (lambda stepsize used in discretizing fcumlambda)
    FinalValueOnly=in_FinalValueOnly;		// Flag used to indicate if only final value (xf) is returned (true) or if Marker Locations are returned as well (false)
    InsertedLength=in_Li;					// Inserted Length (length of the catheter from the entry point to the tip)

    // If the catheter is inserted more than the length of the catheter, clamp it to catheter length
    if (InsertedLength > SegEndLambdas[in_no_segments - 1]) InsertedLength = SegEndLambdas[in_no_segments - 1];

    // WE ARE GOING TO REORDER SEGMENT AND ACTUATOR UNITS SO THAT THEY ARE ORDERED FROM THE INSERTION POINT TO THE TIP
    // *  I.E., SWITCH TO PROXIMAL TO DISTAL ORDERING
    double tempdouble;
    StartSegmentIndex = in_no_segments - 1;		// Index of the segment where the integration to solve IVP will start -- the segment located at the entry point; note that segment indices start at 0
    SegBounds[in_no_segments] = InsertedLength;	// End s value of last segment is s=InsertedLength
    // Entry point has a value of s=0 -- we may not simulate full length of the most proximal segment in the chamber
    for (int i = 0; i < in_no_segments; i++) {
        tempdouble = InsertedLength - SegEndLambdas[i];
        if (tempdouble > 0.0) {
            SegBounds[(in_no_segments - 1) - i] = tempdouble;
            StartSegmentIndex--;				// Integration will start at the previous segment
        }
        else {
            SegBounds[(in_no_segments - 1) - i] = 0.0;	// This segment boundary is still inside the sheath
        }
    }
    for (int i = 0; i < in_no_flex_seg; i++) { 		// flexible catheter segment
        // Calculate the number of integration steps based on the given IntegrationStepSize
        SegSteps[i] = int(ceil((SegBounds[2 * i + 1] - SegBounds[2 * i]) * DeltaSInv));
    }

    NextLocMarker = in_no_locmarkers;
    if (!FinalValueOnly) {
        // While changing the order and converting from lambda to s,
        //   also find the index of the first localization marker after the entry point
        //   and assign (extrapolated) locations to markers which are still inside the sheath
        for (int i = 0; i < in_no_locmarkers; i++) {
            tempdouble = InsertedLength - in_LocMarkerLambdas[i];
            LocMarkers[(in_no_locmarkers - 1) - i] = tempdouble;
            if (tempdouble > 0.0) NextLocMarker--;
            else {   // and assign (extrapolated) locations to markers which are still inside the sheath
                LocMarkerUpdate(p_atLocMarkers[(in_no_locmarkers - 1) - i], xi, tempdouble);
            }
        }
    }

    // be careful - order is reversed in SegmentTypes, FlexActIndex, K, Kinv, ustar, rho, ActMass, CoilAlignmentTurnAreaMatrix, and MagMoment
    int32_t flexcnt = 0, actcnt = 0;
    for (int i = in_no_segments - 1; i >= 0; i--) {  // we will loop backwards to identify the counts
        // we are also going to store the indices of the flexible segments and the actuators
        if (in_SegmentTypes[i] == CatheterSegmentType::FLEXIBLE) FlexActIndex[(in_no_segments - 1) - i] = flexcnt++;
        else if (in_SegmentTypes[i] == CatheterSegmentType::RIGID_WITH_ACTUATOR)  FlexActIndex[(in_no_segments - 1) - i] = actcnt++;
    }
    for (int i = 0; i < in_no_segments; i++) {
        rho[(in_no_segments - 1) - i] = in_rho[i];
    }
    for (int i = 0; i < in_no_flex_seg; i++) {
        for (int j = 0; j < 9; j++) {
            K[(in_no_flex_seg - 1) - i][j] = in_K[i][j];
            Kinv[(in_no_flex_seg - 1) - i][j] = in_Kinv[i][j];
        }
        for (int j = 0; j < 3; j++) {
            ustar[(in_no_flex_seg - 1) - i][j] = in_ustar[i][j];
        }
    }
    for (int i = 0; i < in_no_act_set; i++) {
        for (int j = 0; j < 3; j++) {
            MagMoment[(in_no_act_set - 1) - i][j] = in_MagMoment[i][j];
        }
        ActMass[(in_no_act_set - 1) - i] = in_actMass[i];
        mCopy_AB<9>(in_CoilAlignmentTurnAreaMatrix[i], CoilAlignmentTurnAreaMatrix[(in_no_act_set - 1) - i]);
    }

    for (int i = 0; i < in_no_fcum_steps + 1; i++) for (int j = 0; j < 3; j++)
            fcumlambda[i][j] = in_fcumlambda[i][j];

    /*
     * The params for dynamics
     * */
    auto & g = out_CoreParams.g;
    mCopy_AB<3>(in_g, g);               // gravitational vector
    auto & v_L_pre = out_CoreParams.v_L_pre;
    auto & w_L_pre = out_CoreParams.w_L_pre;
    auto & p_pre = out_CoreParams.p_pre;
    auto & R_pre = out_CoreParams.R_pre;
    auto & actInertia = out_CoreParams.actInertia;
    auto & m_L = out_CoreParams.m_L;
    auto & n_L = out_CoreParams.n_L;
    auto & damping = out_CoreParams.damping;
    auto & delta_t = out_CoreParams.DELTA_T;
    delta_t = in_delta_t;

    for (int i = 0; i < in_no_act_set; ++i) {
        for (int j = 0; j < 3; ++j) {
            v_L_pre[i][j] = in_v_L_pre[i][j];
            w_L_pre[i][j] = in_w_L_pre[i][j];
            p_pre[i][j] = in_p_pre[i][j];
            m_L[i][j] = in_mL[i][j];
            n_L[i][j] = in_nL[i][j];
        }
        for (int j = 0; j < 9; ++j) {
            actInertia[i][j] = in_actInertia[i][j];
            R_pre[i][j] = in_R_pre[i][j];
        }
        for (int j = 0; j < 6; ++j) {
            damping[i][j] = in_damping[i][j];
        }
    }


}

CRMShootingMethodParams CRMDYNConstructShootingMethodParamSet(	CRMCatheterModelParams CathParams, CatheterConfiguration CathConfig,
                                                                  double InsertionLength, double ActuationCurrents[NUM_ACT_SET][3],
                                                                  ContactModeType ContactMode,
                                                                  double TipConstraintPoint[3], double TipForce[3],
                                                                  double IntegrationStepSize, double ActInertia[NUM_ACT_SET][9],
                                                                  double in_v_L_pre[NUM_ACT_SET][3], double in_w_L_pre[NUM_ACT_SET][3], double in_p_pre[NUM_ACT_SET][3],
                                                                  double in_R_pre[NUM_ACT_SET][9], double in_damping[NUM_ACT_SET][6], double in_DELTA_T) {

    // calculate no_fcum_steps
    double length = 0.0;
    double* SegEndLambdas = new double[CathParams.no_segments];  // allocate a temporary storage to store SegEndLambdas
    for (int i = 0; i < CathParams.no_segments; i++) {
        length += CathParams.SegLengths[i];
        SegEndLambdas[i] = length;
    }
    int32_t no_fcum_steps = int32_t(ceil(length / FCUM_DLAMBDA));

    // construct ShootingParams
    CRMShootingMethodParams ShootingParams(CathParams.no_flex_seg, CathParams.no_rigid_seg, CathParams.no_act_set, CathParams.no_locmarkers, no_fcum_steps);

    ShootingParams.IntegrationStepSize = IntegrationStepSize;
    double dlambda = length / double(no_fcum_steps);
    ShootingParams.dlambdainv = 1.0 / dlambda;

    mCopy_AB<3>(CathConfig.B0, ShootingParams.B0);
    mCopy_AB<3>(CathConfig.g, ShootingParams.g);
    mCopy_AB<9>(CathConfig.R0, ShootingParams.R0);
    mCopy_AB<3>(CathConfig.p0, ShootingParams.p0);

    ShootingParams.ContactMode = ContactMode;
    mCopy_AB<3>(TipConstraintPoint, ShootingParams.TipConstraintPoint);
    mCopy_AB<3>(TipForce, ShootingParams.TipForce);

    ShootingParams.Li = MIN(MAX(InsertionLength, 0), length);

    int32_t actcnt = 0;
    int32_t* ActNos = new int32_t[CathParams.no_segments];
    for (int i = 0; i < CathParams.no_segments; i++) {
        ShootingParams.SegmentTypes[i] = CathParams.SegmentTypes[i];
        if (CathParams.SegmentTypes[i] == CatheterSegmentType::RIGID_WITH_ACTUATOR) ActNos[i] = actcnt++;
        else ActNos[i] = -1;
    }

    // copy SegEndLambdas from temporary storage to ShootingParams.SegEndLambdas
    for (int i = 0; i < CathParams.no_segments; i++) ShootingParams.SegEndLambdas[i] = SegEndLambdas[i];
    delete[] SegEndLambdas; // delete temporary storage

    // copy rho values
    for (int i = 0; i < CathParams.no_segments; i++) {
        ShootingParams.rho[i] = CathParams.rho[i];
    }

    double oR, iR, mI, pmI, E, G, K[9], Kinv[9];
    for (int i = 0; i < CathParams.no_flex_seg; i++) {
        oR = CathParams.OuterRadius[i];
        iR = CathParams.InnerRadius[i];
        E = CathParams.YoungsModulus[i];
        G = CathParams.ShearModulus[i];
        mI = 0.25 * M_PI * (POW4(oR) - POW4(iR));	// Area moment of inertia along x - axis
        pmI = 0.5 * M_PI * (POW4(oR) - POW4(iR));	// Polar moment of inertia of area
        K[0] = E * mI;		K[1] = 0.0;			K[2] = 0.0;
        K[3] = 0.0;			K[4] = E * mI;		K[5] = 0.0;
        K[6] = 0.0;			K[7] = 0.0;			K[8] = G * pmI;
        Kinv[0] = 1.0 / (E * mI);		Kinv[1] = 0.0;				Kinv[2] = 0.0;
        Kinv[3] = 0.0;				Kinv[4] = 1.0 / (E * mI);		Kinv[5] = 0.0;
        Kinv[6] = 0.0;				Kinv[7] = 0.0;				Kinv[8] = 1.0 / (G * pmI);
        mCopy_AB<9>(K, ShootingParams.K[i]);
        mCopy_AB<9>(Kinv, ShootingParams.Kinv[i]);
    }
    for (int i = 0; i < CathParams.no_flex_seg; i++) for (int j = 0; j < 3; j++) ShootingParams.ustar[i][j] = CathParams.ustar[i][j];
    double CoilAlignMat[9], c0, s0, c1, s1;
    double tempf[3];
    for (int i = 0; i < CathParams.no_act_set; i++) {
        ShootingParams.ActMass[i] = CathParams.ActMass[i];
        c0 = cos(CathParams.CoilAlignmentAngles[i][0]);
        s0 = sin(CathParams.CoilAlignmentAngles[i][0]);
        c1 = cos(CathParams.CoilAlignmentAngles[i][1]);
        s1 = sin(CathParams.CoilAlignmentAngles[i][1]);
        mMult_AB<3, 3, 1>(CathParams.CoilTurnAreaMat[i], ActuationCurrents[i], tempf);
        CoilAlignMat[0] = c0;	CoilAlignMat[1] = -s1;	CoilAlignMat[2] = 0.0;
        CoilAlignMat[3] = s0;	CoilAlignMat[4] = c1;	CoilAlignMat[5] = 0.0;
        CoilAlignMat[6] = 0.0;	CoilAlignMat[7] = 0.0;	CoilAlignMat[8] = 1.0;
        mMult_AB<3, 3, 1>(CoilAlignMat, tempf, ShootingParams.MagMoment[i]);
        mMult_AB<3, 3, 3>(CoilAlignMat, CathParams.CoilTurnAreaMat[i], ShootingParams.CoilAlignmentTurnAreaMatrix[i]);


        for (int j = 0; j < 9; ++j) {
            ShootingParams.actInertia[i][j] = ActInertia[i][j];
        }

        for (int j = 0; j < 3; ++j) {
            ShootingParams.v_L_pre[i][j] = in_v_L_pre[i][j];
            ShootingParams.w_L_pre[i][j] = in_w_L_pre[i][j];
            ShootingParams.p_pre[i][j] = in_p_pre[i][j];
        }
        for (int j = 0; j < 9; ++j) {
            ShootingParams.R_pre[i][j] = in_R_pre[i][j];
        }
        for (int j = 0; j < 6; ++j) {
            ShootingParams.damping[i][j] = in_damping[i][j];
        }

    }

    ShootingParams.DELTA_T = in_DELTA_T;


    ShootingParams.fcumlambda[0][0] = 0.0;
    ShootingParams.fcumlambda[0][1] = 0.0;
    ShootingParams.fcumlambda[0][2] = 0.0;
    double cumpos = 0.0, lastpos = 0.0, mass = 0.0, segstart, segend, currpos;
    int32_t curr_segment = 0, last_segment = 0;
    // calculation loop
    mass = 0.0;
    for (int32_t i = 1; i < no_fcum_steps + 1; i++) {
        cumpos = i * dlambda;
        while ((cumpos > ShootingParams.SegEndLambdas[curr_segment]) && (curr_segment < CathParams.no_segments)) curr_segment++;		// let's find the current segment number,
        for (int32_t j = last_segment; j <= curr_segment; j++) {
            if (j == 0) segstart = 0.0; else segstart = ShootingParams.SegEndLambdas[j - 1];
            segend = ShootingParams.SegEndLambdas[j];
            currpos = MIN(ShootingParams.SegEndLambdas[j], cumpos);
            mass += CathParams.rho[j] * (currpos - lastpos); // we will add the partial mass of the flexible substrate
            if (CathParams.SegmentTypes[j] == CatheterSegmentType::RIGID_WITH_ACTUATOR) {
                if (segend == segstart) mass += CathParams.ActMass[ActNos[j]];  // point mass, since segment_length == 0
                else mass += CathParams.ActMass[ActNos[j]] * (currpos - lastpos) / (segend - segstart); // we will add the partial mass of the actuator
            }
            lastpos = currpos;
        }
        last_segment = curr_segment;
        mMult_sA<3, 1>(mass, CathConfig.g, ShootingParams.fcumlambda[i]);
    }

    delete[] ActNos;

    for (int i = 0; i < CathParams.no_locmarkers; i++) ShootingParams.LocMarkerLambdas[i] = CathParams.LocMarkers[i];

    return ShootingParams;

}



void rotationMatrixToEulerAngles(double in_R[9], double out_v[3]) {
    double sy = sqrt(in_R[0] * in_R[0] + in_R[3] * in_R[3]);
    bool singular = sy < 1e-6; // If

    double x, y, z;
    if (!singular) {
        x = atan2(in_R[7], in_R[8]);
        y = atan2(-in_R[6], sy);
        z = atan2(in_R[3], in_R[0]);
    } else {
        std::cout << "Singular Rotation matrix............" << std::endl;
        x = atan2(-in_R[5], in_R[4]);
        y = atan2(-in_R[6], sy);
        z = 0;
    }

    out_v[0] = x;
    out_v[1] = y;
    out_v[2] = z;
}
