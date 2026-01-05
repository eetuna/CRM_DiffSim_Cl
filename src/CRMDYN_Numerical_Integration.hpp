#pragma once

#include "CRM_MatrixOperations_Templates.hpp"

//
//
//	NUMERICAL INTEGRATION FUNCTIONS for IVP SOLVER
//
//

namespace CRMCatheterModel {

    inline void CRMIntegrand_dyn(double s, const StateVector& in_x, const CRMIntegrandParams in_Params, const double in_nL[3],
                      StateDerivativeVector& out_xdot) {

        double Length;
        double deltalambdainv;
        double fcum[3], ustardot[3];  //  We are assuming Kdot=0.0 (K=const)

        // copy inputs and parameters to local variables
        Length = in_Params.Li;
        deltalambdainv = in_Params.dlambdainv;
        auto& K = in_Params.K;
        auto& Kinv = in_Params.Kinv;
        auto& ustar = in_Params.ustar;
        auto& l = in_Params.l;
        for (int i = 0; i < 3; i++) {
            ustardot[i] = 0.0; //in_ustardot[i]; // we assume ustardot=0.0 since our rest shape model is piecewise constant curvature
        }

        // for simplicity, create aliases
        auto& u = in_x._u;
        auto& R = in_x._R;
        auto& udot = out_xdot._u;
#ifndef ANALYTICAL_SE3_STEP
        auto& p = in_x._p;				// we will not need this for analytical calculation
        auto& pdot = out_xdot._p;		// we will not need this for analytical calculation
        auto& Rdot = out_xdot._R;		// we will not need this for analytical calculation
#endif


        // calculate interpolated value of fcum
        double lambda = Length - s;
        double ix = lambda * deltalambdainv;
        double ird_f = floor(ix); // index for round down  -- doubleing point
        if (ird_f < 0) ird_f = 0;
        int ird = (int)ird_f;	//    integer index
        double iru_f = ceil(ix); 	// index for round up  -- doubleing point
        if (iru_f > in_Params.no_fcum_steps) iru_f = in_Params.no_fcum_steps;
        int iru = (int)iru_f;	//    integer index
        double ixmird = ix - ird;    // weight for interpolation
        double irumix = iru - ix;	// weight for interpolation
        for (int i = 0; i < 3; i++) {
            fcum[i] = in_Params.fcumlambda[iru][i] * ixmird + in_Params.fcumlambda[ird][i] * irumix;
        }

        // add the tip force to fcum
        for (int i = 0; i < 3; i++) {
            fcum[i] += in_Params.ftip[i];
        }

        double nL_spatial[3];
        mMult_AB<3,3,1>(R, in_nL, nL_spatial);
        // add the tip force to fcum
        for (int i = 0; i < 3; i++) {
            fcum[i]		+=	nL_spatial[i];
        }


        // calculate u_hat
        double u_hat[9];
        wHat(u, u_hat);

        // udot = ustardot - Kinv*((um*K+Kdot)*(u-ustar_s) + e3m*R'*intf + R'*l); % udot
        //
        //   e3hat*R' = [ -r12 -r22 -r32; r11 r21 r31; 0 0 0];
        double e3hatRT[9];
        e3hatRT[0] = -R[1];   e3hatRT[1] = -R[4];   e3hatRT[2] = -R[7];
        e3hatRT[3] = R[0];    e3hatRT[4] = R[3];    e3hatRT[5] = R[6];
        e3hatRT[6] = 0.0;     e3hatRT[7] = 0.0;     e3hatRT[8] = 0.0;
        double e3hatRTfcum[3];
        mMult_AB<3, 3, 1>(e3hatRT, fcum, e3hatRTfcum);			//  e3m*R'*intf
        double RTl[3];
        mMult_ATB<3, 3, 1>(R, l, RTl);								// R'*l
        double umustar[3];
        mSub_AB<3, 1>(u, ustar, umustar);							// (u-ustar_s)
        double Kumustar[3], uhatKumustar[3];
        mMult_AB<3, 3, 1>(K, umustar, Kumustar);
        mMult_AB<3, 3, 1>(u_hat, Kumustar, uhatKumustar); 		 	//(um*K+Kdot)*(u-ustar_s)  assuming Kdot=0
        double sumterm[3];
        mAdd_ABC<3, 1>(uhatKumustar, e3hatRTfcum, RTl, sumterm);	// ((um*K+Kdot)*(u-ustar_s) + e3m*R'*intf + R'*l)
        double KinvSum[3];
        mMult_AB<3, 3, 1>(Kinv, sumterm, KinvSum);					// Kinv*((um*K+Kdot)*(u-ustar_s) + e3m*R'*intf + R'*l)
        mSub_AB<3, 1>(ustardot, KinvSum, udot);					// udot = ustardot - Kinv*((um*K+Kdot)*(u-ustar_s) + e3m*R'*intf + R'*l);

#ifndef ANALYTICAL_SE3_STEP
        // we will not need these for analytical calculation
        // Rdot = R*u_hat
        mMult_AB<3, 3, 3>(R, u_hat, Rdot);
        // pdot = R*e3,
        for (int i = 0; i < 3; i++) {
            pdot[i] = R[i * 3 + 2];
        }
#endif

        //xdot(1:3) = R*e3;             % pdot
        //xdot(4:12) = reshape(R*um,9,1); % Rdot   --- Note that matlab code reshapes in column major order while we are saving in row major order
        //xdot(13:15) = ustardot - Kinv*((um*K+Kdot)*(u-ustar_s) + e3m*R'*intf + R'*l); % udot
        //  note: the sample code has matlab indexing starting from 1 to 15

    }

    // Templated version for forward-mode AD with Dual numbers
    template<typename T>
    void CRMIntegrand_dyn_T(double s, const StateVector_T<T>& in_x, const CRMIntegrandParams in_Params, const T in_nL[3],
                      StateDerivativeVector_T<T>& out_xdot) {

        double Length;
        double deltalambdainv;
        T fcum[3];
        double ustardot[3];  //  We are assuming Kdot=0.0 (K=const)

        // copy inputs and parameters to local variables
        Length = in_Params.Li;
        deltalambdainv = in_Params.dlambdainv;
        auto& K = in_Params.K;
        auto& Kinv = in_Params.Kinv;
        auto& ustar = in_Params.ustar;
        auto& l = in_Params.l;
        for (int i = 0; i < 3; i++) {
            ustardot[i] = 0.0; // we assume ustardot=0.0 since our rest shape model is piecewise constant curvature
        }

        // for simplicity, create aliases
        auto& u = in_x._u;
        auto& R = in_x._R;  // R is now type T (carries derivatives)
        auto& udot = out_xdot._u;

        // calculate interpolated value of fcum
        double lambda = Length - s;
        double ix = lambda * deltalambdainv;
        double ird_f = floor(ix);
        if (ird_f < 0) ird_f = 0;
        int ird = (int)ird_f;
        double iru_f = ceil(ix);
        if (iru_f > in_Params.no_fcum_steps) iru_f = in_Params.no_fcum_steps;
        int iru = (int)iru_f;
        double ixmird = ix - ird;
        double irumix = iru - ix;
        for (int i = 0; i < 3; i++) {
            fcum[i] = T(in_Params.fcumlambda[iru][i] * ixmird + in_Params.fcumlambda[ird][i] * irumix);
        }

        // add the tip force to fcum
        for (int i = 0; i < 3; i++) {
            fcum[i] += T(in_Params.ftip[i]);
        }

        T nL_spatial[3];
        mMult_AB_T<T, 3, 3, 1>(R, in_nL, nL_spatial);
        // add nL_spatial to fcum
        for (int i = 0; i < 3; i++) {
            fcum[i] += nL_spatial[i];
        }

        // calculate u_hat
        T u_hat[9];
        wHat_T(u, u_hat);

        // udot = ustardot - Kinv*((um*K+Kdot)*(u-ustar_s) + e3m*R'*intf + R'*l); % udot
        //
        //   e3hat*R' = [ -r12 -r22 -r32; r11 r21 r31; 0 0 0];
        T e3hatRT[9];
        e3hatRT[0] = -R[1];   e3hatRT[1] = -R[4];   e3hatRT[2] = -R[7];
        e3hatRT[3] = R[0];    e3hatRT[4] = R[3];    e3hatRT[5] = R[6];
        e3hatRT[6] = T(0.0);  e3hatRT[7] = T(0.0);  e3hatRT[8] = T(0.0);
        T e3hatRTfcum[3];
        mMult_AB_T<T, 3, 3, 1>(e3hatRT, fcum, e3hatRTfcum);  // e3m*R'*intf
        T RTl[3];
        T l_T[3];
        for (int i = 0; i < 3; ++i) l_T[i] = T(l[i]);
        mMult_ATB_T<T, 3, 3, 1>(R, l_T, RTl);  // R'*l
        T umustar[3];
        mSub_AB_T<T, 3, 1>(u, ustar, umustar);  // (u-ustar_s)
        T Kumustar[3], uhatKumustar[3];
        mMult_AB_T<T, 3, 3, 1>(K, umustar, Kumustar);
        mMult_AB_T<T, 3, 3, 1>(u_hat, Kumustar, uhatKumustar);  //(um*K+Kdot)*(u-ustar_s) assuming Kdot=0
        T sumterm[3];
        mAdd_ABC_T<T, 3, 1>(uhatKumustar, e3hatRTfcum, RTl, sumterm);  // ((um*K+Kdot)*(u-ustar_s) + e3m*R'*intf + R'*l)
        T KinvSum[3];
        mMult_AB_T<T, 3, 3, 1>(Kinv, sumterm, KinvSum);  // Kinv*((um*K+Kdot)*(u-ustar_s) + e3m*R'*intf + R'*l)
        mSub_AB_T<T, 3, 1>(ustardot, KinvSum, udot);  // udot = ustardot - Kinv*((um*K+Kdot)*(u-ustar_s) + e3m*R'*intf + R'*l);
    }



    //function [x_1toN] = ABM4(x_0, t_0, N, h, Integrand, initmethod)
    template <typename StVecType, typename ParamType>
    void ABM4_dyn(const StVecType& in_x_0, const double t_0, const int N, const double h,
              const ParamType in_Params, const double in_nL[3], int32_t in_no_locmarkers,
              const bool CalculateEnergy,
              const bool FinalValueOnly, const double in_LocMarkers[], int& inout_NextLocMarkerIdx,
              StVecType& out_x_N, double& out_PotentialEnergy, double out_p_atLocMarkers[][3]) {

        using _SVT = StVecType;
        using _DVT = StDerivativeVectType<StVecType>;

        _SVT x_nm3;
        _SVT x_nm2;
        _SVT x_nm1;
        _SVT x_n(in_x_0);
        _SVT x_np1;
        double t_n;
        _DVT xdot_nm3;
        _DVT xdot_nm2;
        _DVT xdot_nm1;
        _DVT xdot_n;

        out_PotentialEnergy = 0.0;
        double Kun[3], Kunp1[3], unTKun, unp1TKun1p, unTKunp1, unp1TKun;
        int NextLocMarkerIdx = inout_NextLocMarkerIdx;
        auto& LocMarkers = in_LocMarkers;  // create an alias
        double delta_n_overh, delta_np1_overh;
        double gTpsum, mgTpmid;

        // initialize the iteration items
        t_n = t_0;

        for (int idx = 0; idx < N; idx++) {

            if (idx < 3) {  // RK2 initialization steps
                RK2_step_dyn(x_n, t_n, h, in_Params, in_nL, x_np1, xdot_n);
            }
            else { 		 // ABM4 steps
                ABM4_step_dyn(x_n, t_n, h, xdot_nm1, xdot_nm2, xdot_nm3, x_nm1, x_nm2, x_nm3, in_Params, in_nL, x_np1, xdot_n);
            }

            //Project_State_to_Manifold(x_np1);
            // increment "time"
            t_n = t_n + h;

            if (CalculateEnergy) {
                //
                // Elastic Potential Energy: \int u^T K u \approx ( u_n^T K u_n + ( u_n^T K u_np1 + u_np1^T K u_n ) /2 + u_np1^T K u_np1 ) /3
                //
                mMult_AB<3, 3, 1>(in_Params.K, x_n._u, Kun);
                mMult_AB<3, 3, 1>(in_Params.K, x_np1._u, Kunp1);
                mMult_ATB<3, 1, 1>(x_n._u, Kun, &unTKun);
                mMult_ATB<3, 1, 1>(x_n._u, Kunp1, &unTKunp1);
                mMult_ATB<3, 1, 1>(x_np1._u, Kun, &unp1TKun);
                mMult_ATB<3, 1, 1>(x_np1._u, Kunp1, &unp1TKun1p);
                out_PotentialEnergy += h * (unTKun + 0.5 * (unTKunp1 + unp1TKun) + unp1TKun1p) / 3.0;
                //
                // Gravitational Potential Energy: mass * gravity * height = (\rho * h) * (- g^T p_mid)
                //
                gTpsum = 0.0;
                for (int ix = 0; ix < 3; ix++) gTpsum += in_Params.g[ix] * (x_n._p[ix] + x_np1._p[ix]);
                mgTpmid = -0.5 * gTpsum;
                out_PotentialEnergy += in_Params.rho * h * mgTpmid;
            }

            if (!FinalValueOnly) {
                // are there any localization markers?  If so, calculate their positions
                // remember, t_n has already been incremented
                while ((NextLocMarkerIdx < in_no_locmarkers) && (LocMarkers[NextLocMarkerIdx] <= t_n)) {
                    delta_np1_overh = (t_n - LocMarkers[NextLocMarkerIdx]) / h;
                    delta_n_overh = 1.0 - delta_np1_overh;
                    out_p_atLocMarkers[NextLocMarkerIdx][0] = delta_np1_overh * (x_n._p[0]) + delta_n_overh * (x_np1._p[0]);
                    out_p_atLocMarkers[NextLocMarkerIdx][1] = delta_np1_overh * (x_n._p[1]) + delta_n_overh * (x_np1._p[1]);
                    out_p_atLocMarkers[NextLocMarkerIdx][2] = delta_np1_overh * (x_n._p[2]) + delta_n_overh * (x_np1._p[2]);
                    NextLocMarkerIdx++;
                }
            }

            // update the iteration items
            x_nm3 = x_nm2;
            x_nm2 = x_nm1;
            x_nm1 = x_n;
            x_n = x_np1;
            xdot_nm3 = xdot_nm2;
            xdot_nm2 = xdot_nm1;
            xdot_nm1 = xdot_n;

        }

        // Copy final value
        out_x_N = x_n;
        inout_NextLocMarkerIdx = NextLocMarkerIdx;

    }


    // [x_np1, xdot_n, xdot_nm1, xdot_nm2] = ABM4_step(x_n, t_n, xdot_nm1, xdot_nm2, xdot_nm3, h, Integrand)
    template <typename StVecType, typename ParamType>
    void ABM4_step_dyn(const StVecType& in_x_n, double t_n, double h,
                   const StDerivativeVectType<StVecType>& in_xdot_nm1, const StDerivativeVectType<StVecType>& in_xdot_nm2, const StDerivativeVectType<StVecType>& in_xdot_nm3,
                   const StVecType& in_x_nm1, const StVecType& in_x_nm2, const StVecType& in_x_nm3,
                   const ParamType in_Params, const double in_nL[3],
                   StVecType& out_x_np1, StDerivativeVectType<StVecType>& out_xdot_n) {

        using _SVT = StVecType;
        using _DVT = StDerivativeVectType<StVecType>;

        const double P_COEFF_N = 55.0 / 24.0, P_COEFF_Nm1 = -59.0 / 24.0, P_COEFF_Nm2 = 37.0 / 24.0, P_COEFF_Nm3 = -9.0 / 24.0;  // AB4 Predictor Coefficients
        const double C_COEFF_Np1 = 9.0 / 24.0, C_COEFF_N = 19.0 / 24.0, C_COEFF_Nm1 = -5.0 / 24.0, C_COEFF_Nm2 = 1.0 / 24.0;     // AM4 Corrector Coefficients
        _SVT x_n(in_x_n);       		// from input
        _SVT x_nm1(in_x_nm1);       	// from input
        _SVT x_nm2(in_x_nm2);       	// from input
        _SVT x_nm3(in_x_nm3);       	// from input
        _SVT x_np1_hat;    				// intermediate
        _DVT xdot_np1_hat;				// intermediate
        auto& xdot_n = out_xdot_n;
        _DVT xdot_nm1(in_xdot_nm1);  	// from input
        _DVT xdot_nm2(in_xdot_nm2);  	// from input
        _DVT xdot_nm3(in_xdot_nm3);  	// from input


        //ABM4_STEP_STEP1:
        CRMIntegrand_dyn(t_n, x_n, in_Params, in_nL, xdot_n);
        x_np1_hat = x_n + h * (P_COEFF_N * xdot_n + P_COEFF_Nm1 * xdot_nm1 + P_COEFF_Nm2 * xdot_nm2 + P_COEFF_Nm3 * xdot_nm3);
#ifdef ANALYTICAL_SE3_STEP
        //      calculate R_np1_hat and p_np1_hat analytically, without numerical integration
		double u_n_pred[3];
		for (int i = 0; i < 3; i++) u_n_pred[i] = (P_COEFF_N * x_n._u[i] + P_COEFF_Nm1 * x_nm1._u[i] + P_COEFF_Nm2 * x_nm2._u[i] + P_COEFF_Nm3 * x_nm3._u[i]);
		SE3_Analytical_Step(x_n._R, x_n._p, u_n_pred, h, x_np1_hat._R /*R_np1_hat*/, x_np1_hat._p /*p_np1_hat*/);
#endif
        //ABM4_STEP_STEP2:
        CRMIntegrand_dyn(t_n + h, x_np1_hat, in_Params, in_nL, xdot_np1_hat);
        out_x_np1 = x_n + h * (C_COEFF_Np1 * xdot_np1_hat + C_COEFF_N * xdot_n + C_COEFF_Nm1 * xdot_nm1 + C_COEFF_Nm2 * xdot_nm2);
#ifdef ANALYTICAL_SE3_STEP
        //      calculate R_np1 and p_np1 analytically, without numerical integration
		double u_n_corr[3];
		for (int i = 0; i < 3; i++) u_n_corr[i] = (C_COEFF_Np1 * x_np1_hat._u[i] + C_COEFF_N * x_n._u[i] + C_COEFF_Nm1 * x_nm1._u[i] + C_COEFF_Nm2 * x_nm2._u[i]);
		SE3_Analytical_Step(x_n._R, x_n._p, u_n_corr, h, out_x_np1._R /*R_np1*/, out_x_np1._p /*p_np1*/);
#endif

    }


    //[x_np1, xdot_n] = RK2_step(x_n, t_n, h, Integrand)
    template <typename StVecType, typename ParamType>
    void RK2_step_dyn(const StVecType& in_x_n, const double t_n, const double h,
                  const ParamType in_Params, const double in_nL[3],
                  StVecType& out_x_np1, StDerivativeVectType<StVecType>& out_xdot_n) {

        using _SVT = StVecType;
        using _DVT = StDerivativeVectType<StVecType>;

        _SVT x_n(in_x_n);       // from input
        _DVT k1;				// intermediate
        _DVT k2oh;				// intermediate
        _SVT x_n_p_k1o2;		// intermediate
        auto& xdot_n = out_xdot_n;// for output

        //RK2_STEP_STEP1:
        CRMIntegrand_dyn(t_n, x_n, in_Params, in_nL, xdot_n);
        k1 = h * xdot_n;
        x_n_p_k1o2 = x_n + k1 * 0.5;
#ifdef ANALYTICAL_SE3_STEP
        // we will calculate R_n_p_k1o2 and p_n_p_k1o2 analytically, without numerical integration
		SE3_Analytical_Step(x_n._R, x_n._p, x_n._u, h * 0.5, x_n_p_k1o2._R, x_n_p_k1o2._p);
#endif

        //RK2_STEP_STEP2:
        CRMIntegrand_dyn(t_n + h * 0.5, x_n_p_k1o2, in_Params, in_nL, k2oh);
        out_x_np1 = x_n + h * k2oh;
#ifdef ANALYTICAL_SE3_STEP
        // we will calculate R_np1 and p_np1 analytically, without numerical integration
		SE3_Analytical_Step(x_n._R, x_n._p, x_n_p_k1o2._u/*u_np1half*/, h, out_x_np1._R, out_x_np1._p);
#endif

    }

}
