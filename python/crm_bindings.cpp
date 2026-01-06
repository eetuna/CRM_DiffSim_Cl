#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "CRM.hpp"
#include "CRM_DiffEquilibrium.hpp"
#include "CRM_TrueLegacyDynamics.hpp"
#include "CRM_ReferenceHarness.hpp"
#include "CRMDYN.hpp"

namespace py = pybind11;
using namespace CRMCatheterModel;

// Forward declarations
py::dict py_true_legacy_step_forward(
    py::array_t<double> x_coil_arr,
    py::array_t<double> xf_arr,
    py::array_t<double> u_arr,
    double dt,
    py::dict params_dict,
    py::object mL_guess_obj,
    py::object nL_guess_obj
);

py::dict py_true_legacy_step_vjp(
    py::array_t<double> x_coil_arr,
    py::array_t<double> xf_arr,
    py::array_t<double> u_arr,
    double dt,
    double L_inserted,
    py::dict params_dict,
    py::array_t<double> grad_tip_p_arr
);

py::dict py_true_legacy_step_vjp_batched(
    py::array_t<double> x_coil_arr,
    py::array_t<double> xf_arr,
    py::array_t<double> u_arr,
    double dt,
    double L_inserted,
    py::dict params_dict,
    py::array_t<double> grad_tip_p_batch_arr
);

py::dict py_true_legacy_linearize(
    py::array_t<double> x_coil_arr,
    py::array_t<double> xf_arr,
    py::array_t<double> u_arr,
    double dt,
    py::dict params_dict,
    py::object mL_guess_obj,
    py::object nL_guess_obj
);

py::dict py_crmdyn_reference_rollout(
    py::array_t<double> x0_coil_arr,
    py::array_t<double> x0_tip_arr,
    py::array_t<double> u_seq_arr,
    double dt,
    py::dict params_dict,
    bool use_warmstart
);

// P1-5: API Versioning
static const char* CRM_PACKAGE_VERSION = "1.0.0";
static const char* CRM_API_VERSION = "1.1.0";
static const char* CRM_API_CONTRACT_EQUILIBRIUM = "equilibrium_v1_1";
static const char* CRM_API_CONTRACT_DYNAMICS = "dynamics_v1_1";

// Helper: convert Python dict to CRMForwardKinematicsData
CRMForwardKinematicsData parse_fk_params(py::dict params_dict) {
    CRMForwardKinematicsData fk_params;

    // These must be persistent Python objects passed in the dict
    fk_params.CathParams = params_dict["CathParams"].cast<CRMCatheterModelParams*>();
    fk_params.CathConfig = params_dict["CathConfig"].cast<CatheterConfiguration*>();

    fk_params.ContactMode = static_cast<ContactModeType>(params_dict["ContactMode"].cast<int>());

    // Initialize TipConstraintPoint (required even for FREE_TIP mode)
    for (int i = 0; i < 3; i++) fk_params.TipConstraintPoint[i] = 0.0;
    if (params_dict.contains("TipConstraintPoint")) {
        auto tip_constraint = params_dict["TipConstraintPoint"].cast<std::array<double, 3>>();
        for (int i = 0; i < 3; i++) fk_params.TipConstraintPoint[i] = tip_constraint[i];
    }

    auto tip_force = params_dict["TipForce"].cast<std::array<double, 3>>();
    for (int i = 0; i < 3; i++) fk_params.TipForce[i] = tip_force[i];

    auto deltau0_init = params_dict["deltau0_initialguess"].cast<std::array<double, 3>>();
    for (int i = 0; i < 3; i++) fk_params.deltau0_initialguess[i] = deltau0_init[i];

    // Initialize ftip_initialguess (required for FIXED_TIP mode)
    for (int i = 0; i < 3; i++) fk_params.ftip_initialguess[i] = 0.0;
    if (params_dict.contains("ftip_initialguess")) {
        auto ftip_init = params_dict["ftip_initialguess"].cast<std::array<double, 3>>();
        for (int i = 0; i < 3; i++) fk_params.ftip_initialguess[i] = ftip_init[i];
    }

    fk_params.IntegrationStepSize = params_dict["IntegrationStepSize"].cast<double>();
    fk_params.FinalValueOnly = params_dict.contains("FinalValueOnly") ?
        params_dict["FinalValueOnly"].cast<bool>() : true;

    // Initialize marker/coil pointers to nullptr
    fk_params.ReportedMarkerPos = nullptr;
    fk_params.ReportedCoilOrient = nullptr;
    fk_params.ReportedCoilPos = nullptr;

    return fk_params;
}

// Wrapper for equilibrium_forward
py::dict py_equilibrium_forward(
    py::array_t<double> u_arr,
    double L_inserted,
    py::dict params_dict
) {
    auto u_buf = u_arr.request();
    if (u_buf.ndim != 1 || u_buf.shape[0] != NUM_ACT_SET * 3) {
        throw std::runtime_error("u must be 1D array of length " + std::to_string(NUM_ACT_SET * 3));
    }

    double* u = static_cast<double*>(u_buf.ptr);
    CRMForwardKinematicsData fk_params = parse_fk_params(params_dict);

    // Zero-initialize result structure
    EquilibriumResult result;
    std::memset(&result, 0, sizeof(EquilibriumResult));

    int status = equilibrium_forward(u, L_inserted, fk_params, result);

    py::dict out;
    out["status"] = status;

    // Return arrays with proper shape (copy data to ensure ownership)
    // Use std::vector to create properly strided arrays
    std::vector<ssize_t> shape = {3};
    std::vector<ssize_t> strides = {sizeof(double)};

    auto p_tip_arr = py::array_t<double>(shape, strides);
    auto deltau0_arr = py::array_t<double>(shape, strides);

    std::memcpy(p_tip_arr.mutable_data(), result.p_tip, 3 * sizeof(double));
    std::memcpy(deltau0_arr.mutable_data(), result.deltau0, 3 * sizeof(double));

    out["p_tip"] = p_tip_arr;
    out["deltau0"] = deltau0_arr;
    out["converged"] = result.converged;

    // Cache Jacobians for backward (return as 2D numpy arrays, row-major)
    // Shape (3, 3) for J_p_u0, J_u_u0, K_tip
    // Shape (3, 3*NUM_ACT_SET) for J_p_zc, J_u_zc
    auto J_p_u0_arr = py::array_t<double>({3, 3});
    auto J_u_u0_arr = py::array_t<double>({3, 3});
    auto J_p_zc_arr = py::array_t<double>({3, 3 * NUM_ACT_SET});
    auto J_u_zc_arr = py::array_t<double>({3, 3 * NUM_ACT_SET});
    auto K_tip_arr = py::array_t<double>({3, 3});

    // Copy data (row-major layout matches C++ storage)
    std::memcpy(J_p_u0_arr.mutable_data(), result.J_p_u0, 9 * sizeof(double));
    std::memcpy(J_u_u0_arr.mutable_data(), result.J_u_u0, 9 * sizeof(double));
    std::memcpy(J_p_zc_arr.mutable_data(), result.J_p_zc, 3 * NUM_ACT_SET * 3 * sizeof(double));
    std::memcpy(J_u_zc_arr.mutable_data(), result.J_u_zc, 3 * NUM_ACT_SET * 3 * sizeof(double));
    std::memcpy(K_tip_arr.mutable_data(), result.K_tip, 9 * sizeof(double));

    out["J_p_u0"] = J_p_u0_arr;
    out["J_u_u0"] = J_u_u0_arr;
    out["J_p_zc"] = J_p_zc_arr;
    out["J_u_zc"] = J_u_zc_arr;
    out["K_tip"] = K_tip_arr;

    out["nl_iterations"] = result.nl_iterations;
    out["final_residual"] = result.final_residual;
    out["lu_rank"] = result.lu_rank;
    out["rel_solve_residual"] = result.rel_solve_residual;
    out["exit_code"] = result.exit_code;

    // P1-5: API Versioning
    out["api_version"] = CRM_API_VERSION;
    out["api_contract"] = CRM_API_CONTRACT_EQUILIBRIUM;

    return out;
}

// Wrapper for equilibrium_backward
py::dict py_equilibrium_backward(
    py::dict fwd_result,
    py::array_t<double> grad_p_tip_arr
) {
    // Ensure input is float64 and C-contiguous
    auto grad_buf = grad_p_tip_arr.request();
    if (grad_buf.ndim != 1 || grad_buf.shape[0] != 3) {
        throw std::runtime_error("grad_p_tip must be 1D array of length 3");
    }
    if (!py::isinstance<py::array_t<double, py::array::c_style | py::array::forcecast>>(grad_p_tip_arr)) {
        throw std::runtime_error("grad_p_tip must be float64 C-contiguous array");
    }

    // Reconstruct EquilibriumResult from cached data
    EquilibriumResult cached;
    std::memset(&cached, 0, sizeof(EquilibriumResult));

    auto J_p_u0_arr = fwd_result["J_p_u0"].cast<py::array_t<double>>();
    auto J_u_u0_arr = fwd_result["J_u_u0"].cast<py::array_t<double>>();
    auto J_p_zc_arr = fwd_result["J_p_zc"].cast<py::array_t<double>>();
    auto J_u_zc_arr = fwd_result["J_u_zc"].cast<py::array_t<double>>();
    auto K_tip_arr = fwd_result["K_tip"].cast<py::array_t<double>>();

    // Verify shapes (should be 2D now)
    if (J_p_u0_arr.ndim() != 2 || J_p_u0_arr.shape(0) != 3 || J_p_u0_arr.shape(1) != 3) {
        throw std::runtime_error("J_p_u0 must be (3, 3) array");
    }
    if (J_u_u0_arr.ndim() != 2 || J_u_u0_arr.shape(0) != 3 || J_u_u0_arr.shape(1) != 3) {
        throw std::runtime_error("J_u_u0 must be (3, 3) array");
    }
    if (J_p_zc_arr.ndim() != 2 || J_p_zc_arr.shape(0) != 3 || J_p_zc_arr.shape(1) != 3 * NUM_ACT_SET) {
        throw std::runtime_error("J_p_zc must be (3, " + std::to_string(3 * NUM_ACT_SET) + ") array");
    }
    if (J_u_zc_arr.ndim() != 2 || J_u_zc_arr.shape(0) != 3 || J_u_zc_arr.shape(1) != 3 * NUM_ACT_SET) {
        throw std::runtime_error("J_u_zc must be (3, " + std::to_string(3 * NUM_ACT_SET) + ") array");
    }
    if (K_tip_arr.ndim() != 2 || K_tip_arr.shape(0) != 3 || K_tip_arr.shape(1) != 3) {
        throw std::runtime_error("K_tip must be (3, 3) array");
    }

    std::memcpy(cached.J_p_u0, J_p_u0_arr.data(), 9 * sizeof(double));
    std::memcpy(cached.J_u_u0, J_u_u0_arr.data(), 9 * sizeof(double));
    std::memcpy(cached.J_p_zc, J_p_zc_arr.data(), 3 * NUM_ACT_SET * 3 * sizeof(double));
    std::memcpy(cached.J_u_zc, J_u_zc_arr.data(), 3 * NUM_ACT_SET * 3 * sizeof(double));
    std::memcpy(cached.K_tip, K_tip_arr.data(), 9 * sizeof(double));

    double* grad_p_tip = static_cast<double*>(grad_buf.ptr);
    double grad_u[NUM_ACT_SET * 3];
    int lu_rank;
    double rel_residual;

    int status = equilibrium_backward(cached, grad_p_tip, grad_u, &lu_rank, &rel_residual);

    // Copy grad_u to properly allocated array with correct strides
    std::vector<ssize_t> shape = {NUM_ACT_SET * 3};
    std::vector<ssize_t> strides = {sizeof(double)};

    auto grad_u_arr = py::array_t<double>(shape, strides);
    std::memcpy(grad_u_arr.mutable_data(), grad_u, NUM_ACT_SET * 3 * sizeof(double));

    py::dict out;
    out["status"] = status;
    out["grad_u"] = grad_u_arr;
    out["lu_rank"] = lu_rank;
    out["rel_residual"] = rel_residual;

    // P1-5: API Versioning
    out["api_version"] = CRM_API_VERSION;
    out["api_contract"] = CRM_API_CONTRACT_EQUILIBRIUM;

    return out;
}

// Wrapper for loading catheter parameters from file
CRMCatheterModelParams* py_load_cath_params(const std::string& filepath) {
    try {
        // Load_CRMCatheterModelParams returns by value, so we need to copy-construct on heap
        CRMCatheterModelParams loaded = Load_CRMCatheterModelParams(filepath.c_str());
        return new CRMCatheterModelParams(loaded);  // Use copy constructor
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to load catheter parameters from ") +
                                 filepath + ": " + e.what());
    }
}

// Wrapper for loading catheter configuration from file
CatheterConfiguration* py_load_cath_config(const std::string& filepath) {
    try {
        // Load_CatheterConfiguration returns by value, allocate on heap
        CatheterConfiguration loaded = Load_CatheterConfiguration(filepath.c_str());
        CatheterConfiguration* config = new CatheterConfiguration();
        *config = loaded;  // Copy assign
        return config;
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Failed to load catheter configuration from ") +
                                 filepath + ": " + e.what());
    }
}


PYBIND11_MODULE(crm_diff_py, m) {
    m.doc() = "CRM Differentiable Simulator Python Bindings (CP2.3)";

    // P1-5: API Versioning - expose version constants
    // m.attr("__version__") = CRM_PACKAGE_VERSION;
    // m.attr("__api_version__") = CRM_API_VERSION;

    // P1-5: API Versioning - expose compatibility check
    // m.def("check_api_compat", &check_api_compat,
    //       py::arg("required_version"),
    //       "Check if the current API version is compatible with the required version.\n"
    //       "Raises RuntimeError if incompatible.\n"
    //       "Compatible if major version matches and current >= required.");

    // Expose ContactModeType enum
    py::enum_<ContactModeType>(m, "ContactModeType")
        .value("FREE_TIP", ContactModeType::FREE_TIP)
        .value("FIXED_TIP", ContactModeType::FIXED_TIP);

    // Expose opaque pointer types for catheter params and config
    py::class_<CRMCatheterModelParams>(m, "CRMCatheterModelParams");
    py::class_<CatheterConfiguration>(m, "CatheterConfiguration");

    // Expose parameter loading functions
    m.def("load_cath_params", &py_load_cath_params,
          py::return_value_policy::take_ownership,
          py::arg("filepath"),
          "Load catheter parameters from file");

    m.def("load_cath_config", &py_load_cath_config,
          py::return_value_policy::take_ownership,
          py::arg("filepath"),
          "Load catheter configuration from file");

    // Expose equilibrium primitives
    m.def("equilibrium_forward", &py_equilibrium_forward,
          py::arg("u"), py::arg("L_inserted"), py::arg("params_dict"),
          "Forward pass: compute tip position and cache Jacobians");

    m.def("equilibrium_backward", &py_equilibrium_backward,
          py::arg("fwd_result"), py::arg("grad_p_tip"),
          "Backward pass: compute gradient w.r.t. actuation currents");

    // A0: TRUE legacy stepping (DynamicsBVP → DYNSolverIVP)
    m.def("true_legacy_step_forward", &py_true_legacy_step_forward,
          py::arg("x_coil"), py::arg("xf"), py::arg("u"), py::arg("dt"),
          py::arg("params_dict"),
          py::arg("mL_guess") = py::none(), py::arg("nL_guess") = py::none(),
          "A0: TRUE legacy step (DynamicsBVP → DYNSolverIVP). Returns next state and observables.");

    // A0: TRUE legacy VJP (backward pass)
    m.def("true_legacy_step_vjp", &py_true_legacy_step_vjp,
          py::arg("x_coil"), py::arg("xf"), py::arg("u"), py::arg("dt"), py::arg("L_inserted"),
          py::arg("params_dict"), py::arg("grad_tip_p"),
          "A0: TRUE legacy VJP. Computes gradients w.r.t. x_coil, xf, and u given grad_tip_p.");

    // A3.5: TRUE legacy batched VJP (backward pass with multi-RHS)
    m.def("true_legacy_step_vjp_batched", &py_true_legacy_step_vjp_batched,
          py::arg("x_coil"), py::arg("xf"), py::arg("u"), py::arg("dt"), py::arg("L_inserted"),
          py::arg("params_dict"), py::arg("grad_tip_p_batch"),
          "A3.5: TRUE legacy batched VJP. Computes gradients for multiple RHS using one factorization.");

    // TRUE legacy linearization (compute A, B matrices via implicit function theorem)
    m.def("true_legacy_linearize", &py_true_legacy_linearize,
          py::arg("x_coil"), py::arg("xf"), py::arg("u"), py::arg("dt"),
          py::arg("params_dict"),
          py::arg("mL_guess") = py::none(), py::arg("nL_guess") = py::none(),
          "Compute A, B linearization matrices using implicit function theorem.");

    // Reference harness for CRMDYN_test.cpp regression testing
    m.def("crmdyn_reference_rollout", &py_crmdyn_reference_rollout,
          py::arg("x0_coil"), py::arg("x0_tip"), py::arg("u_seq"), py::arg("dt"),
          py::arg("params_dict"), py::arg("use_warmstart") = true,
          "Deterministic reference rollout matching CRMDYN_test.cpp logic.");
}

// A0: TRUE legacy step wrapper (DynamicsBVP → DYNSolverIVP)
py::dict py_true_legacy_step_forward(
    py::array_t<double> x_coil_arr,
    py::array_t<double> xf_arr,
    py::array_t<double> u_arr,
    double dt,
    py::dict params_dict,
    py::object mL_guess_obj = py::none(),
    py::object nL_guess_obj = py::none()
) {
    // Validate x_coil: should be [N_ACT, 18]
    auto x_coil_buf = x_coil_arr.request();
    if (x_coil_buf.ndim != 2 || x_coil_buf.shape[0] != NUM_ACT_SET || x_coil_buf.shape[1] != 18) {
        throw std::runtime_error(
            "x_coil must be shape [" + std::to_string(NUM_ACT_SET) + ", 18], got [" +
            std::to_string(x_coil_buf.shape[0]) + ", " + std::to_string(x_coil_buf.shape[1]) + "]"
        );
    }

    // Validate xf: should be [15]
    auto xf_buf = xf_arr.request();
    if (xf_buf.ndim != 1 || xf_buf.shape[0] != NUM_STATES) {
        throw std::runtime_error(
            "xf must be shape [15], got [" + std::to_string(xf_buf.shape[0]) + "]"
        );
    }

    // Validate u: should be [N_ACT, 3]
    auto u_buf = u_arr.request();
    if (u_buf.ndim != 2 || u_buf.shape[0] != NUM_ACT_SET || u_buf.shape[1] != 3) {
        throw std::runtime_error(
            "u must be shape [" + std::to_string(NUM_ACT_SET) + ", 3], got [" +
            std::to_string(u_buf.shape[0]) + ", " + std::to_string(u_buf.shape[1]) + "]"
        );
    }

    double* x_coil_data = static_cast<double*>(x_coil_buf.ptr);
    double* xf_data = static_cast<double*>(xf_buf.ptr);
    double* u_data = static_cast<double*>(u_buf.ptr);

    // Extract coil state components from x_coil
    // x_coil layout: [v[3], w[3], p[3], R[9]] per coil
    double v_L_pre[NUM_ACT_SET][3];
    double w_L_pre[NUM_ACT_SET][3];
    double p_pre[NUM_ACT_SET][3];
    double R_pre[NUM_ACT_SET][9];

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            v_L_pre[j][i] = x_coil_data[j * 18 + i];
            w_L_pre[j][i] = x_coil_data[j * 18 + 3 + i];
            p_pre[j][i] = x_coil_data[j * 18 + 6 + i];
        }
        for (int i = 0; i < 9; ++i) {
            R_pre[j][i] = x_coil_data[j * 18 + 9 + i];
        }
    }

    // Reshape u to ActuationCurrents format
    double ActuationCurrents[NUM_ACT_SET][3];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            ActuationCurrents[j][i] = u_data[j * 3 + i];
        }
    }

    // Parse FK params from dict
    CRMForwardKinematicsData fk_params = parse_fk_params(params_dict);

    // Extract catheter params and config
    CRMCatheterModelParams* CathParams = fk_params.CathParams;
    CatheterConfiguration* CathConfig = fk_params.CathConfig;

    // Get insertion length from params_dict (required parameter)
    if (!params_dict.contains("L_inserted")) {
        throw std::runtime_error("L_inserted must be provided in params_dict");
    }
    double L_inserted = params_dict["L_inserted"].cast<double>();

    // Construct actuator inertia (use diagonal approximation if not provided)
    double ActInertia[NUM_ACT_SET][9];
    if (params_dict.contains("ActInertia")) {
        auto act_inertia_arr = params_dict["ActInertia"].cast<py::array_t<double>>();
        auto act_inertia_buf = act_inertia_arr.request();
        double* act_inertia_data = static_cast<double*>(act_inertia_buf.ptr);
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < 9; ++i) {
                ActInertia[j][i] = act_inertia_data[j * 9 + i];
            }
        }
    } else {
        // Use hollow cylinder inertia formula (matching legacy CRMDYN_c.cpp)
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            double mass = CathParams->ActMass[j];
            double r_outer = CathParams->OuterRadius[0];
            double r_inner = CathParams->InnerRadius[0];
            double seg_length = CathParams->SegLengths[2*j + 1];

            double r_sum_sq = r_outer * r_outer + r_inner * r_inner;
            double I_zz = 0.5 * mass * r_sum_sq;
            double I_xx = 0.25 * mass * r_sum_sq + (1.0/12.0) * mass * seg_length * seg_length;

            ActInertia[j][0] = I_xx;  ActInertia[j][1] = 0.0;   ActInertia[j][2] = 0.0;
            ActInertia[j][3] = 0.0;   ActInertia[j][4] = I_xx;  ActInertia[j][5] = 0.0;
            ActInertia[j][6] = 0.0;   ActInertia[j][7] = 0.0;   ActInertia[j][8] = I_zz;
        }
    }

    // Construct damping (load from CathParams if not in dict)
    double damping[NUM_ACT_SET][6];
    if (params_dict.contains("damping")) {
        auto damping_arr = params_dict["damping"].cast<py::array_t<double>>();
        auto damping_buf = damping_arr.request();
        double* damping_data = static_cast<double*>(damping_buf.ptr);
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < 6; ++i) {
                damping[j][i] = damping_data[j * 6 + i];
            }
        }
    } else {
        // Load from catheter parameters (default: 0.1 from CRM_SupportFunctions.cpp)
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < 6; ++i) {
                damping[j][i] = CathParams->ActDamping[j][i];
            }
        }
    }

    // Construct shooting method params
    CRMShootingMethodParams params = CRMDYNConstructShootingMethodParamSet(
        *CathParams, *CathConfig,
        L_inserted, ActuationCurrents,
        fk_params.ContactMode,
        fk_params.TipConstraintPoint, fk_params.TipForce,
        fk_params.IntegrationStepSize, ActInertia,
        v_L_pre, w_L_pre, p_pre, R_pre,
        damping, dt
    );

    // Prepare warm-start guesses
    double mL_guess[NUM_ACT_SET][3];
    double nL_guess[NUM_ACT_SET][3];
    double ftip_guess[3] = {0.0, 0.0, 0.0};

    if (!mL_guess_obj.is_none() && !nL_guess_obj.is_none()) {
        auto mL_arr = mL_guess_obj.cast<py::array_t<double>>();
        auto nL_arr = nL_guess_obj.cast<py::array_t<double>>();

        auto mL_buf = mL_arr.request();
        auto nL_buf = nL_arr.request();

        double* mL_data = static_cast<double*>(mL_buf.ptr);
        double* nL_data = static_cast<double*>(nL_buf.ptr);

        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int i = 0; i < 3; ++i) {
                mL_guess[j][i] = mL_data[j * 3 + i];
                nL_guess[j][i] = nL_data[j * 3 + i];
            }
        }
    } else {
        // Use zero initial guess
        std::memset(mL_guess, 0, sizeof(mL_guess));
        std::memset(nL_guess, 0, sizeof(nL_guess));
    }

    // Allocate outputs for DynamicsBVP
    double out_u0[3];
    double out_mL[NUM_ACT_SET][3];
    double out_nL[NUM_ACT_SET][3];
    double out_tau[NUM_ACT_SET][3];
    double out_ftip[3];
    int out_localmin = -1;

    // Call DynamicsBVP
    DynamicsBVP(params, xf_data, mL_guess, nL_guess, ftip_guess,
                out_u0, out_mL, out_nL, out_tau, out_ftip, out_localmin);

    // Allocate outputs for DYNSolverIVP
    double out_xf[NUM_STATES];
    double out_coil_state[NUM_ACT_SET][NUM_COIL_STATES];
    double out_markers[10][3];  // Placeholder, actual size depends on config

    if (out_localmin != 0) {
        // BVP Convergence Failed: Return previous state (reject step)
        for (int i = 0; i < NUM_STATES; ++i) {
            out_xf[i] = xf_data[i];
        }
        for (int j = 0; j < NUM_ACT_SET; ++j) {
            for (int k = 0; k < NUM_COIL_STATES; ++k) {
                out_coil_state[j][k] = x_coil_data[j * 18 + k];
            }
        }
    } else {
        // Call DYNSolverIVP
        DYNSolverIVP(params, out_u0, out_mL, out_nL, out_tau, out_ftip,
                     true,  // FinalValueOnly
                     out_xf, out_coil_state, out_markers);
    }

    // Package results
    py::dict result;

    // Pack x_coil_next [N_ACT, 18]
    auto x_coil_next_arr = py::array_t<double>({NUM_ACT_SET, 18});
    double* x_coil_next_data = x_coil_next_arr.mutable_data();
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < NUM_COIL_STATES; ++i) {
            x_coil_next_data[j * 18 + i] = out_coil_state[j][i];
        }
    }
    result["x_coil_next"] = x_coil_next_arr;

    // Pack xf_next [15]
    auto xf_next_arr = py::array_t<double>(
        std::vector<size_t>{NUM_STATES},
        std::vector<size_t>{sizeof(double)}
    );
    double* xf_next_data = xf_next_arr.mutable_data();
    for (int i = 0; i < NUM_STATES; ++i) {
        xf_next_data[i] = out_xf[i];
    }
    result["xf_next"] = xf_next_arr;

    // Pack observables
    // Use explicit strides to ensure proper memory layout
    auto tip_p_arr = py::array_t<double>(
        std::vector<size_t>{3},          // shape
        std::vector<size_t>{sizeof(double)}  // strides
    );
    double* tip_p_data = tip_p_arr.mutable_data();
    for (int i = 0; i < 3; ++i) {
        tip_p_data[i] = out_xf[i];
    }
    result["tip_p"] = tip_p_arr;

    auto tip_R_arr = py::array_t<double>(
        std::vector<size_t>{9},
        std::vector<size_t>{sizeof(double)}
    );
    double* tip_R_data = tip_R_arr.mutable_data();
    for (int i = 0; i < 9; ++i) {
        tip_R_data[i] = out_xf[3 + i];
    }
    result["tip_R"] = tip_R_arr;

    auto tip_u_arr = py::array_t<double>(
        std::vector<size_t>{3},
        std::vector<size_t>{sizeof(double)}
    );
    double* tip_u_data = tip_u_arr.mutable_data();
    for (int i = 0; i < 3; ++i) {
        tip_u_data[i] = out_xf[12 + i];
    }
    result["tip_u"] = tip_u_arr;

    // Pack warm-start for next iteration
    auto mL_next_arr = py::array_t<double>({NUM_ACT_SET, 3});
    double* mL_next_data = mL_next_arr.mutable_data();
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            mL_next_data[j * 3 + i] = out_mL[j][i];
        }
    }
    result["mL_next"] = mL_next_arr;

    auto nL_next_arr = py::array_t<double>({NUM_ACT_SET, 3});
    double* nL_next_data = nL_next_arr.mutable_data();
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 3; ++i) {
            nL_next_data[j * 3 + i] = out_nL[j][i];
        }
    }
    result["nL_next"] = nL_next_arr;

    // Convergence status
    result["converged"] = (out_localmin == 0);
    result["localmin"] = out_localmin;

    return result;
}

// A0: TRUE legacy step VJP (backward pass for tip_p gradients)
py::dict py_true_legacy_step_vjp(
    py::array_t<double> x_coil_arr,
    py::array_t<double> xf_arr,
    py::array_t<double> u_arr,
    double dt,
    double L_inserted,
    py::dict params_dict,
    py::array_t<double> grad_tip_p_arr
) {
    // Validate inputs
    auto x_coil_buf = x_coil_arr.request();
    auto xf_buf = xf_arr.request();
    auto u_buf = u_arr.request();
    auto grad_buf = grad_tip_p_arr.request();

    if (x_coil_buf.ndim != 2 || x_coil_buf.shape[0] != NUM_ACT_SET || x_coil_buf.shape[1] != 18) {
        throw std::runtime_error("x_coil must be [" + std::to_string(NUM_ACT_SET) + ", 18]");
    }
    if (xf_buf.ndim != 1 || xf_buf.shape[0] != NUM_STATES) {
        throw std::runtime_error("xf must be [15]");
    }
    if (u_buf.ndim != 2 || u_buf.shape[0] != NUM_ACT_SET || u_buf.shape[1] != 3) {
        throw std::runtime_error("u must be [" + std::to_string(NUM_ACT_SET) + ", 3]");
    }
    if (grad_buf.ndim != 1 || grad_buf.shape[0] != 3) {
        throw std::runtime_error("grad_tip_p must be [3]");
    }

    double* x_coil_data = static_cast<double*>(x_coil_buf.ptr);
    double* xf_data = static_cast<double*>(xf_buf.ptr);
    double* u_data = static_cast<double*>(u_buf.ptr);
    double* grad_tip_p = static_cast<double*>(grad_buf.ptr);

    // Reshape inputs to C++ format
    double x_coil[NUM_ACT_SET][18];
    double u[NUM_ACT_SET][3];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 18; ++i) {
            x_coil[j][i] = x_coil_data[j * 18 + i];
        }
        for (int i = 0; i < 3; ++i) {
            u[j][i] = u_data[j * 3 + i];
        }
    }

    // Parse FK params
    CRMForwardKinematicsData fk_params = parse_fk_params(params_dict);

    // Forward pass to cache results
    TrueLegacyStepResult fwd_result;
    std::memset(&fwd_result, 0, sizeof(TrueLegacyStepResult));

    int status = true_legacy_step_forward(
        x_coil, xf_data, u, dt, L_inserted, fk_params,
        nullptr, nullptr, fwd_result
    );

    if (status != 0) {
        throw std::runtime_error("Forward pass failed in VJP");
    }

    // Backward pass - use batched code path (call batched with num_rhs=1)
    // This avoids any potential 2D array issues in the single-sample backward
    std::vector<double> grad_x_coil_vec(NUM_ACT_SET * 18);
    std::vector<double> grad_xf_vec(NUM_STATES);
    std::vector<double> grad_u_vec(NUM_ACT_SET * 3);

    int lu_rank;
    double rel_residual;

    // Call batched backward with num_rhs=1
    status = true_legacy_step_backward_batched(
        fwd_result, 1, grad_tip_p, fk_params,
        grad_x_coil_vec.data(), grad_xf_vec.data(), grad_u_vec.data(),
        &lu_rank, &rel_residual
    );

    if (status != 0) {
        throw std::runtime_error("Backward pass failed in VJP");
    }

    // Package results - exactly like batched version
    py::dict result;

    // Pack grad_x_coil [N, 18] - memcpy from vector like batched
    auto grad_x_coil_arr = py::array_t<double>(
        std::vector<ssize_t>{NUM_ACT_SET, 18}
    );
    std::memcpy(grad_x_coil_arr.mutable_data(), grad_x_coil_vec.data(),
                NUM_ACT_SET * 18 * sizeof(double));
    result["grad_x_coil"] = grad_x_coil_arr;

    // Pack grad_xf [15] - memcpy from vector like batched, with explicit shape
    auto grad_xf_arr = py::array_t<double>(
        std::vector<ssize_t>{NUM_STATES}
    );
    std::memcpy(grad_xf_arr.mutable_data(), grad_xf_vec.data(),
                NUM_STATES * sizeof(double));
    result["grad_xf"] = grad_xf_arr;

    // Pack grad_u [N, 3] - memcpy from vector like batched
    auto grad_u_arr = py::array_t<double>(
        std::vector<ssize_t>{NUM_ACT_SET, 3}
    );
    std::memcpy(grad_u_arr.mutable_data(), grad_u_vec.data(),
                NUM_ACT_SET * 3 * sizeof(double));
    result["grad_u"] = grad_u_arr;

    // Diagnostics
    result["status"] = status;
    result["lu_rank"] = lu_rank;
    result["rel_residual"] = rel_residual;

    return result;
}

// A3.5: TRUE legacy step batched VJP (backward pass with multi-RHS)
py::dict py_true_legacy_step_vjp_batched(
    py::array_t<double> x_coil_arr,
    py::array_t<double> xf_arr,
    py::array_t<double> u_arr,
    double dt,
    double L_inserted,
    py::dict params_dict,
    py::array_t<double> grad_tip_p_batch_arr
) {
    // Validate inputs
    auto x_coil_buf = x_coil_arr.request();
    auto xf_buf = xf_arr.request();
    auto u_buf = u_arr.request();
    auto grad_batch_buf = grad_tip_p_batch_arr.request();

    if (x_coil_buf.ndim != 2 || x_coil_buf.shape[0] != NUM_ACT_SET || x_coil_buf.shape[1] != 18) {
        throw std::runtime_error("x_coil must be [" + std::to_string(NUM_ACT_SET) + ", 18]");
    }
    if (xf_buf.ndim != 1 || xf_buf.shape[0] != NUM_STATES) {
        throw std::runtime_error("xf must be [15]");
    }
    if (u_buf.ndim != 2 || u_buf.shape[0] != NUM_ACT_SET || u_buf.shape[1] != 3) {
        throw std::runtime_error("u must be [" + std::to_string(NUM_ACT_SET) + ", 3]");
    }
    if (grad_batch_buf.ndim != 2 || grad_batch_buf.shape[1] != 3) {
        throw std::runtime_error("grad_tip_p_batch must be [num_rhs, 3]");
    }

    int num_rhs = grad_batch_buf.shape[0];

    double* x_coil_data = static_cast<double*>(x_coil_buf.ptr);
    double* xf_data = static_cast<double*>(xf_buf.ptr);
    double* u_data = static_cast<double*>(u_buf.ptr);
    double* grad_tip_p_batch = static_cast<double*>(grad_batch_buf.ptr);

    // Reshape inputs to C++ format
    double x_coil[NUM_ACT_SET][18];
    double u[NUM_ACT_SET][3];
    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 18; ++i) {
            x_coil[j][i] = x_coil_data[j * 18 + i];
        }
        for (int i = 0; i < 3; ++i) {
            u[j][i] = u_data[j * 3 + i];
        }
    }

    // Parse FK params
    CRMForwardKinematicsData fk_params = parse_fk_params(params_dict);

    // Forward pass to cache results
    TrueLegacyStepResult fwd_result;
    std::memset(&fwd_result, 0, sizeof(TrueLegacyStepResult));

    int status = true_legacy_step_forward(
        x_coil, xf_data, u, dt, L_inserted, fk_params,
        nullptr, nullptr, fwd_result
    );

    if (status != 0) {
        throw std::runtime_error("Forward pass failed in batched VJP");
    }

    // Allocate output buffers
    std::vector<double> grad_x_coil_batch(num_rhs * NUM_ACT_SET * 18);
    std::vector<double> grad_xf_batch(num_rhs * NUM_STATES);
    std::vector<double> grad_u_batch(num_rhs * NUM_ACT_SET * 3);

    // Batched backward pass
    int lu_rank;
    double rel_residual;

    status = true_legacy_step_backward_batched(
        fwd_result, num_rhs, grad_tip_p_batch, fk_params,
        grad_x_coil_batch.data(), grad_xf_batch.data(), grad_u_batch.data(),
        &lu_rank, &rel_residual
    );

    if (status != 0) {
        throw std::runtime_error("Batched backward pass failed in VJP");
    }

    // Package results
    py::dict result;

    // Pack grad_x_coil_batch [num_rhs, N, 18]
    auto grad_x_coil_arr = py::array_t<double>({num_rhs, NUM_ACT_SET, 18});
    std::memcpy(grad_x_coil_arr.mutable_data(), grad_x_coil_batch.data(),
                num_rhs * NUM_ACT_SET * 18 * sizeof(double));
    result["grad_x_coil"] = grad_x_coil_arr;

    // Pack grad_xf_batch [num_rhs, 15]
    auto grad_xf_arr = py::array_t<double>({num_rhs, NUM_STATES});
    std::memcpy(grad_xf_arr.mutable_data(), grad_xf_batch.data(),
                num_rhs * NUM_STATES * sizeof(double));
    result["grad_xf"] = grad_xf_arr;

    // Pack grad_u_batch [num_rhs, N, 3]
    auto grad_u_arr = py::array_t<double>({num_rhs, NUM_ACT_SET, 3});
    std::memcpy(grad_u_arr.mutable_data(), grad_u_batch.data(),
                num_rhs * NUM_ACT_SET * 3 * sizeof(double));
    result["grad_u"] = grad_u_arr;

    // Diagnostics
    result["status"] = status;
    result["lu_rank"] = lu_rank;
    result["rel_residual"] = rel_residual;
    result["num_rhs"] = num_rhs;

    return result;
}

// Linearization: Compute A, B matrices using implicit function theorem
py::dict py_true_legacy_linearize(
    py::array_t<double> x_coil_arr,
    py::array_t<double> xf_arr,
    py::array_t<double> u_arr,
    double dt,
    py::dict params_dict,
    py::object mL_guess_obj,
    py::object nL_guess_obj
) {
    // Validate x_coil: should be [N_ACT, 18]
    auto x_coil_buf = x_coil_arr.request();
    if (x_coil_buf.ndim != 2 || x_coil_buf.shape[0] != NUM_ACT_SET || x_coil_buf.shape[1] != 18) {
        throw std::runtime_error(
            "x_coil must be shape [" + std::to_string(NUM_ACT_SET) + ", 18]"
        );
    }

    // Validate xf: should be [15]
    auto xf_buf = xf_arr.request();
    if (xf_buf.ndim != 1 || xf_buf.shape[0] != NUM_STATES) {
        throw std::runtime_error(
            "xf must be shape [15]"
        );
    }

    // Validate u: should be [N_ACT, 3]
    auto u_buf = u_arr.request();
    if (u_buf.ndim != 2 || u_buf.shape[0] != NUM_ACT_SET || u_buf.shape[1] != 3) {
        throw std::runtime_error(
            "u must be shape [" + std::to_string(NUM_ACT_SET) + ", 3]"
        );
    }

    double* x_coil_data = static_cast<double*>(x_coil_buf.ptr);
    double* xf_data = static_cast<double*>(xf_buf.ptr);
    double* u_data = static_cast<double*>(u_buf.ptr);

    // Convert to C-style arrays
    double x_coil[NUM_ACT_SET][18];
    double xf[NUM_STATES];
    double u[NUM_ACT_SET][3];

    for (int j = 0; j < NUM_ACT_SET; ++j) {
        for (int i = 0; i < 18; ++i) {
            x_coil[j][i] = x_coil_data[j * 18 + i];
        }
        for (int i = 0; i < 3; ++i) {
            u[j][i] = u_data[j * 3 + i];
        }
    }
    for (int i = 0; i < NUM_STATES; ++i) {
        xf[i] = xf_data[i];
    }

    // Parse FK params
    CRMForwardKinematicsData fk_params = parse_fk_params(params_dict);

    // Get insertion length
    if (!params_dict.contains("L_inserted")) {
        throw std::runtime_error("L_inserted must be provided in params_dict");
    }
    double L_inserted = params_dict["L_inserted"].cast<double>();

    // Parse warm-start if provided
    const double* mL_guess = nullptr;
    const double* nL_guess = nullptr;

    if (!mL_guess_obj.is_none() && !nL_guess_obj.is_none()) {
        auto mL_arr = mL_guess_obj.cast<py::array_t<double>>();
        auto nL_arr = nL_guess_obj.cast<py::array_t<double>>();
        mL_guess = static_cast<const double*>(mL_arr.request().ptr);
        nL_guess = static_cast<const double*>(nL_arr.request().ptr);
    }

    // Step 1: Run forward pass to cache data
    TrueLegacyStepResult fwd_result;
    int status = true_legacy_step_forward(
        x_coil, xf, u, dt, L_inserted, fk_params, mL_guess, nL_guess, fwd_result
    );

    if (status != 0) {
        throw std::runtime_error("Forward step failed with status " + std::to_string(status));
    }

    // Step 2: Compute A, B matrices
    Eigen::MatrixXd A, B;
    int qr_rank;
    double rel_residual;

    status = true_legacy_linearize_implicit(
        fwd_result, fk_params, L_inserted, A, B, &qr_rank, &rel_residual
    );

    if (status != 0) {
        throw std::runtime_error("Linearization failed with status " + std::to_string(status));
    }

    // Step 3: Convert to numpy arrays
    const int state_dim = NUM_ACT_SET * 18 + NUM_STATES;
    const int control_dim = NUM_ACT_SET * 3;

    py::array_t<double> A_arr({state_dim, state_dim});
    py::array_t<double> B_arr({state_dim, control_dim});

    auto A_ptr = static_cast<double*>(A_arr.request().ptr);
    auto B_ptr = static_cast<double*>(B_arr.request().ptr);

    // Copy data (row-major)
    for (int i = 0; i < state_dim; ++i) {
        for (int j = 0; j < state_dim; ++j) {
            A_ptr[i * state_dim + j] = A(i, j);
        }
    }
    for (int i = 0; i < state_dim; ++i) {
        for (int j = 0; j < control_dim; ++j) {
            B_ptr[i * control_dim + j] = B(i, j);
        }
    }

    // Package results
    py::dict result;
    result["A"] = A_arr;
    result["B"] = B_arr;
    result["qr_rank"] = qr_rank;
    result["rel_residual"] = rel_residual;
    result["converged"] = fwd_result.converged;

    return result;
}

// Reference harness for CRMDYN_test.cpp regression testing
py::dict py_crmdyn_reference_rollout(
    py::array_t<double> x0_coil_arr,
    py::array_t<double> x0_tip_arr,
    py::array_t<double> u_seq_arr,
    double dt,
    py::dict params_dict,
    bool use_warmstart
) {
    // Validate x0_coil: should be [N_ACT, 18]
    auto x0_coil_buf = x0_coil_arr.request();
    if (x0_coil_buf.ndim != 2 || x0_coil_buf.shape[0] != NUM_ACT_SET || x0_coil_buf.shape[1] != 18) {
        throw std::runtime_error(
            "x0_coil must be shape [" + std::to_string(NUM_ACT_SET) + ", 18]"
        );
    }

    // Validate x0_tip: should be [NUM_STATES]
    auto x0_tip_buf = x0_tip_arr.request();
    if (x0_tip_buf.ndim != 1 || x0_tip_buf.shape[0] != NUM_STATES) {
        throw std::runtime_error(
            "x0_tip must be shape [" + std::to_string(NUM_STATES) + "]"
        );
    }

    // Validate u_seq: should be [num_steps, N_ACT, 3]
    auto u_seq_buf = u_seq_arr.request();
    if (u_seq_buf.ndim != 3 || u_seq_buf.shape[1] != NUM_ACT_SET || u_seq_buf.shape[2] != 3) {
        throw std::runtime_error(
            "u_seq must be shape [num_steps, " + std::to_string(NUM_ACT_SET) + ", 3]"
        );
    }
    int num_steps = static_cast<int>(u_seq_buf.shape[0]);

    // Get pointers to input data
    const double* x0_coil_ptr = static_cast<double*>(x0_coil_buf.ptr);
    const double* x0_tip_ptr = static_cast<double*>(x0_tip_buf.ptr);
    const double* u_seq_ptr = static_cast<double*>(u_seq_buf.ptr);

    // Parse catheter parameters
    CRMCatheterModelParams* cath_params = params_dict["CathParams"].cast<CRMCatheterModelParams*>();
    CatheterConfiguration* cath_config = params_dict["CathConfig"].cast<CatheterConfiguration*>();

    // Get other required parameters
    double L_inserted = params_dict["L_inserted"].cast<double>();
    double integration_step_size = params_dict["IntegrationStepSize"].cast<double>();

    // Contact mode and forces
    int contact_mode_int = params_dict["ContactMode"].cast<int>();
    ContactModeType contact_mode = static_cast<ContactModeType>(contact_mode_int);

    double tip_constraint_point[3] = {0.0, 0.0, 0.0};
    if (params_dict.contains("TipConstraintPoint")) {
        auto tcp = params_dict["TipConstraintPoint"].cast<std::array<double, 3>>();
        for (int i = 0; i < 3; i++) tip_constraint_point[i] = tcp[i];
    }

    double tip_force[3] = {0.0, 0.0, 0.0};
    if (params_dict.contains("TipForce")) {
        auto tf = params_dict["TipForce"].cast<std::array<double, 3>>();
        for (int i = 0; i < 3; i++) tip_force[i] = tf[i];
    }

    // Get damping coefficients
    auto damping_list = params_dict["damping"].cast<std::vector<std::vector<double>>>();
    double damping[NUM_ACT_SET][6];
    for (int i = 0; i < NUM_ACT_SET; i++) {
        for (int j = 0; j < 6; j++) {
            damping[i][j] = damping_list[i][j];
        }
    }

    // Get actuator inertia
    auto act_inertia_list = params_dict["ActInertia"].cast<std::vector<std::vector<double>>>();
    double act_inertia[NUM_ACT_SET][9];
    for (int i = 0; i < NUM_ACT_SET; i++) {
        for (int j = 0; j < 9; j++) {
            act_inertia[i][j] = act_inertia_list[i][j];
        }
    }

    // Copy x0_coil and x0_tip into C arrays
    double x0_coil[NUM_ACT_SET][18];
    double x0_tip[NUM_STATES];

    for (int i = 0; i < NUM_ACT_SET; i++) {
        for (int j = 0; j < 18; j++) {
            x0_coil[i][j] = x0_coil_ptr[i * 18 + j];
        }
    }
    for (int i = 0; i < NUM_STATES; i++) {
        x0_tip[i] = x0_tip_ptr[i];
    }

    // Allocate result structure
    CRMDYNReferenceRolloutResult* result = allocate_rollout_result(num_steps + 1);

    // Run rollout
    int status = crmdyn_reference_rollout(
        x0_coil, x0_tip, u_seq_ptr, num_steps + 1, dt,
        *cath_params, *cath_config, L_inserted, contact_mode,
        tip_constraint_point, tip_force, integration_step_size,
        act_inertia, damping, use_warmstart, result
    );

    if (status != 0) {
        free_rollout_result(result);
        throw std::runtime_error("Reference rollout failed");
    }

    // Package results into Python dict
    py::dict py_result;

    // X_traj: [num_steps+1, NUM_STATES]
    auto X_traj_arr = py::array_t<double>({num_steps + 1, NUM_STATES});
    double* X_traj_ptr = X_traj_arr.mutable_data();
    for (int i = 0; i < num_steps + 1; i++) {
        std::memcpy(X_traj_ptr + i * NUM_STATES, result->X_traj[i], NUM_STATES * sizeof(double));
    }
    py_result["X_traj"] = X_traj_arr;

    // X_coil_traj: [num_steps+1, NUM_ACT_SET, 18]
    auto X_coil_traj_arr = py::array_t<double>({num_steps + 1, NUM_ACT_SET, 18});
    double* X_coil_traj_ptr = X_coil_traj_arr.mutable_data();
    for (int i = 0; i < num_steps + 1; i++) {
        for (int j = 0; j < NUM_ACT_SET; j++) {
            std::memcpy(X_coil_traj_ptr + (i * NUM_ACT_SET + j) * 18,
                       result->X_coil_traj[i][j], 18 * sizeof(double));
        }
    }
    py_result["X_coil_traj"] = X_coil_traj_arr;

    // P_tip_traj: [num_steps+1, 3]
    auto P_tip_traj_arr = py::array_t<double>({num_steps + 1, 3});
    double* P_tip_traj_ptr = P_tip_traj_arr.mutable_data();
    for (int i = 0; i < num_steps + 1; i++) {
        std::memcpy(P_tip_traj_ptr + i * 3, result->P_tip_traj[i], 3 * sizeof(double));
    }
    py_result["P_tip_traj"] = P_tip_traj_arr;

    // Diagnostics: u0_traj, nL_traj, mL_traj, ftip_traj
    auto u0_traj_arr = py::array_t<double>({num_steps + 1, 3});
    double* u0_traj_ptr = u0_traj_arr.mutable_data();
    for (int i = 0; i < num_steps + 1; i++) {
        std::memcpy(u0_traj_ptr + i * 3, result->u0_traj[i], 3 * sizeof(double));
    }
    py_result["u0_traj"] = u0_traj_arr;

    auto nL_traj_arr = py::array_t<double>({num_steps + 1, NUM_ACT_SET, 3});
    double* nL_traj_ptr = nL_traj_arr.mutable_data();
    for (int i = 0; i < num_steps + 1; i++) {
        for (int j = 0; j < NUM_ACT_SET; j++) {
            std::memcpy(nL_traj_ptr + (i * NUM_ACT_SET + j) * 3,
                       result->nL_traj[i][j], 3 * sizeof(double));
        }
    }
    py_result["nL_traj"] = nL_traj_arr;

    auto mL_traj_arr = py::array_t<double>({num_steps + 1, NUM_ACT_SET, 3});
    double* mL_traj_ptr = mL_traj_arr.mutable_data();
    for (int i = 0; i < num_steps + 1; i++) {
        for (int j = 0; j < NUM_ACT_SET; j++) {
            std::memcpy(mL_traj_ptr + (i * NUM_ACT_SET + j) * 3,
                       result->mL_traj[i][j], 3 * sizeof(double));
        }
    }
    py_result["mL_traj"] = mL_traj_arr;

    auto ftip_traj_arr = py::array_t<double>({num_steps + 1, 3});
    double* ftip_traj_ptr = ftip_traj_arr.mutable_data();
    for (int i = 0; i < num_steps + 1; i++) {
        std::memcpy(ftip_traj_ptr + i * 3, result->ftip_traj[i], 3 * sizeof(double));
    }
    py_result["ftip_traj"] = ftip_traj_arr;

    // Convergence diagnostics
    auto converged_arr = py::array_t<int>({num_steps + 1});
    int* converged_ptr = converged_arr.mutable_data();
    std::memcpy(converged_ptr, result->converged, (num_steps + 1) * sizeof(int));
    py_result["converged"] = converged_arr;

    auto localmin_arr = py::array_t<int>({num_steps + 1});
    int* localmin_ptr = localmin_arr.mutable_data();
    std::memcpy(localmin_ptr, result->localmin, (num_steps + 1) * sizeof(int));
    py_result["localmin"] = localmin_arr;

    // Free C++ result structure
    free_rollout_result(result);

    return py_result;
}
