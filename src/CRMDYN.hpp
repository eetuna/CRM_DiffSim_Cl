#pragma once
#include <cmath>
#include "CRM_MatrixOperations.hpp"
#include <math.h>
#include <stdlib.h>
#include <Eigen/Dense>

#include "CRM.hpp"
#include "CRM_BVPIVP_APIDeclarations.hpp"

#include <iostream>
#include <chrono>

using namespace std::chrono;
using namespace CRMCatheterModel;


//#define NUM_STATES 15   // u[0..2],R[0..9],p[0..2]
#define ANALYTICAL_SE3_STEP

//#define NUM_FLEX_SEG 2								// Number of flexible segments
//#define NUM_SEGMENTS (NUM_ACT_SET+NUM_FLEX_SEG)		// Total number of segments
//#define NUM_LOCALIZATION_MARKERS 5					// Total number of localization markers
	// IMPORTANT NOTE: for now most proximal segment is assumed to be always flexible
	//    and the flexible and rigid segments are assumed to be alternating
	//    most distal segment can be flexible or rigid

#define NUM_COIL_STATES (6+3+9) // v[3], w[3], p[3] ,R[9]
#define NUM_RESIDUAL 3
#define NUM_DYN_RESIDUAL (NUM_ACT_SET*6) // (m+n) * NUM_ACT_SET
//#define DELTA_T 0.01
//#define NUM_CONTROL (NUM_ACT_SET * 3 + 1) // Control dimension for the dynamic model
//#define RESIDUAL_SCALE_F	1.0	// the residual for coil force coming out of the IVP will be multiplied with this scale to return to the Nonlinear Solver
#define RESIDUAL_SCALE_M	1.0 //(10.0)			// the residual for tip moment coming out of the IVP will be multiplied with this scale to return to the Nonlinear Solver

// Regularization scales used for Nonlinear Solver

#define IVALUE_SCALE_M  1.0 //
#define IVALUE_SCALE_N	1.0		// the variable used in Nonlinear Solver is multiplied with this scale to calculate ftip (tip force) that will be used in IVP
#define IVALUE_SCALE_U	1.0 //(0.01)			// the variable used in Nonlinear Solver is multiplied with this scale to calculate u (curvature) that will be used in IVP
#define IVALUE_SCALE_F	0.01//(0.01)			// the variable used in Nonlinear Solver is multiplied with this scale to calculate ftip (tip force) that will be used in IVP
#define RESIDUAL_SCALE_P	1 //(10.0)			// the residual for tip position error coming out of the IVP will be multiplied with this scale to return to the Nonlinear Solver
#define RESIDUAL_SCALE_R	1 //(10.0)			// the residual for tip position error coming out of the IVP will be multiplied with this scale to return to the Nonlinear Solver

// NL Solver method selection
#define TRUSTREGION_DYN							// Trust Region Method with the numerical jacobian (Default)

// Uncomment the following if CRMShootingMethodBVP will be executed on FPGA PL as a kernel
//#define BVP_AS_A_KERNEL


// FREE_TIP : no contact, FIXED_TIP : catheter tip is constrained at a given point
// Contact Mode: FREE_TIP ---  EQNDIMENSION will be 3	// NLEquation - domain: u[0..2]; range: m_tip[0..2]  (moment at tip)
// Contact Mode: FIXED_TIP --- EQNDIMENSION will be 6	// NLEquation domain: u[0..2], ftip[0..2]; range: m_tip[0..2], delta_p[0..2] (tip position error)


//used to add the extra variables for returning debug data from the kernel
//#define ADD_DEBUG
#ifdef ADD_DEBUG
#define DEBUG_BUFFER_LIMIT 2000 // size of the debug message buffer
#endif

//
// Implementation Note - 10/23/2021 MCC:
//    This version of the code has been implemented to be compatible with autodifferentiation (using autodiff library) using dual numbers.
//    Specifically, all of the CRM code has been templated such that all of the variables that would be differentiated (outputs) and variables that would be 
//    differentiated with respect to (inputs) are defined as the templated "adType".  Similarly, all of the local variables which has adType as an lvalue 
//    is also defined to be the adType.  All of the other i/o arguments of the API and the local variables remaing to be defined as regular double type.
//


////
//// ---------------------------------------------------------
////
//// Cosserat Rod Model Kinematics
////
//// ---------------------------------------------------------
////

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
        bool in_FinalValueOnly, CRMIVPCoreParams &out_CoreParams) ;

struct DYNNLEqnParams : CRMIVPCoreParams {
    DYNNLEqnParams(int32_t no_flex, int32_t no_rigid, int32_t no_act, int32_t no_loc, int32_t no_fcums) :
            CRMIVPCoreParams(no_flex, no_rigid, no_act, no_loc, no_fcums) {};		// input: number of localization markers
	ContactModeType ContactMode;			// Enumerated type defining catheter contact mode.  ContactMode == FREE_TIP if the catheter is not in contact with a surface, FIXED_TIP if catheter tip is constrained to be at TipContraintPoint
	double			TipConstraintPoint[3];	// The spatial coordinates of the point where the catheter tip is constrained to be (used if ContactMode == FIXED_TIP)
	double			TipForce[3];			// External point force (in spatial coordinates) applied at the tip of the catheter (\lambda = 0) (used if ContactMode == FREE_TIP)

    // terms used in the dynamics
//    double	u0_initialguess[3];
    double	ftip_initialguess[3];
    double xf[NUM_STATES]; //Tip state, if applicable
};

////
//// ---------------------------------------------------------
////
//// Dynamics Functions
////
//// ---------------------------------------------------------
////

void CoilIntegrad(const double in_twist[6], const double in_n[3], double g[3], double R[9], double actMass, double actInertia[9], const double damping[6],
                  const double in_B0[3], const double in_muhat[9], const double in_mL[3], double twistdot[6]);

void CoilDynamics( double in_coil_state[NUM_COIL_STATES], double in_n[3], double g[3],
                   double actMass, double actInertia[9], double damping[6], double DELTA_T, double in_B0[3], double in_muhat[9],
                   double in_mL[3], double out_coil_state[NUM_COIL_STATES], double out_xdot_n[6]);

void RK2_coildyn(double in_x_n[NUM_COIL_STATES], double in_n[3], double g[3],  double actMass, double actInertia[9], double damping[6],
                 double in_B0[3], double in_muhat[9], double in_mL[3],
                 double out_x_np1[NUM_COIL_STATES], double out_xdot_n[6] );

void ABM4_coildyn(	double in_x_n[NUM_COIL_STATES],double in_xdot_nm1[6], double in_xdot_nm2[6], double in_xdot_nm3[6],
                      double in_x_nm1[NUM_COIL_STATES], double in_x_nm2[NUM_COIL_STATES], double in_x_nm3[NUM_COIL_STATES],
                      double in_n[3], double g[3],  double actMass, double actInertia[9], double damping[6], double in_B0[3], double in_muhat[9], double in_mL[3],
                      double out_x_np1[NUM_COIL_STATES], double out_xdot_n[6]);

void DYNSE3_TimeSpace(double in_R_n[9], double in_p_n[3], double h, double in_twist_n[6], double out_R_np1[9], double out_p_np1[3]);


void DYNNLEquation(double in_x[], double out_y[], DYNNLEqnParams& Params, double out_u0[3], double out_tau[NUM_ACT_SET*3]);


void CRMFlexible_IVP_Back ( int SegmentIndex, const double in_p[3], const double in_R[9],  DYNNLEqnParams& in_params,
                            const double in_u[3], const double in_n_L[3],
                            double out_u[3], double out_p[3], double out_R[9]);

void CRMFlexForward_pass (  int SegmentIndex, const double in_p[3], const double in_R[9],  CRMIVPCoreParams in_params,
                            const double in_u[3], const double in_n_L[3],
                            double out_u[3], double out_p[3], double out_R[9], double out_p_atLocMarkers[][3]);

void CRMIVP_DYN(	 CRMIVPCoreParams& CoreParams, const double in_u0[3], const double in_p0[3], const double in_R0[9],
                     const double in_mL[NUM_ACT_SET][3], const double in_nL[NUM_ACT_SET][3],  const double in_tau[NUM_ACT_SET][3], const double in_ftip[3],
                     double out_coil_state[NUM_ACT_SET][NUM_COIL_STATES],  double out_u_new[3], double out_p_new[3], double out_R_new[9],
                     double out_p_atLocMarkers[][3]);

void DynamicsBVP(	CRMShootingMethodParams& in_Params, const double xf[NUM_STATES],
                     double in_mL_initialguess[NUM_ACT_SET][3], double in_nL_initialguess[NUM_ACT_SET][3], double in_ftip_initialguess[3],
                     double out_u0[3], double out_mL[NUM_ACT_SET][3], double out_nL[NUM_ACT_SET][3], double out_tau[NUM_ACT_SET][3], double out_ftip[3], int& out_localmin);


void DYNSolverIVP(	CRMShootingMethodParams& in_Params, const double in_u0[3],
                      const double in_mL[NUM_ACT_SET][3], const double in_nL[NUM_ACT_SET][3], const double in_tau[NUM_ACT_SET][3], const double in_ftip[3],
                      bool in_FinalValueOnly,
                      double out_x_N[NUM_STATES], double out_coil_state[NUM_ACT_SET][NUM_COIL_STATES],
                      double out_p_atLocMarkers[][3]);

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
        Eigen::MatrixXd& J_xcoil_x);

CRMShootingMethodParams CRMDYNConstructShootingMethodParamSet(	CRMCatheterModelParams CathParams, CatheterConfiguration CathConfig,
                                                                  double InsertionLength, double ActuationCurrents[NUM_ACT_SET][3],
                                                                  ContactModeType ContactMode,
                                                                  double TipConstraintPoint[3], double TipForce[3],
                                                                  double IntegrationStepSize, double ActInertia[NUM_ACT_SET][9],
                                                                  double in_v_L_pre[NUM_ACT_SET][3], double in_w_L_pre[NUM_ACT_SET][3], double in_p_pre[NUM_ACT_SET][3],
                                                                  double in_R_pre[NUM_ACT_SET][9], double in_damping[NUM_ACT_SET][6], double in_DELTA_T);


////
//// Numerical Integration Functions
////
void rotationMatrixToEulerAngles(double in_R[9], double out_v[3]);
