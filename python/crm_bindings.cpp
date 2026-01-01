#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "CRM.hpp"
#include "CRM_DiffEquilibrium.hpp"
#include "CRM_DiffDynamics.hpp"

namespace py = pybind11;
using namespace CRMCatheterModel;

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

// Wrapper for dynamics_forward
py::dict py_dynamics_forward(
    py::array_t<double> x_t_arr,
    py::array_t<double> u_t_arr,
    double dt,
    double L_inserted,
    py::dict params_dict
) {
    // Validate x_t
    auto x_t_buf = x_t_arr.request();
    if (x_t_buf.ndim != 1 || x_t_buf.shape[0] != 6) {
        throw std::runtime_error("x_t must be 1D array of length 6");
    }

    // Validate u_t
    auto u_t_buf = u_t_arr.request();
    if (u_t_buf.ndim != 1 || u_t_buf.shape[0] != NUM_ACT_SET * 3) {
        throw std::runtime_error("u_t must be 1D array of length " + std::to_string(NUM_ACT_SET * 3));
    }

    double* x_t = static_cast<double*>(x_t_buf.ptr);
    double* u_t = static_cast<double*>(u_t_buf.ptr);

    CRMForwardKinematicsData fk_params = parse_fk_params(params_dict);

    // Zero-initialize result structure
    DynamicsStepResult result;
    std::memset(&result, 0, sizeof(DynamicsStepResult));

    int status = dynamics_forward(x_t, u_t, dt, L_inserted, fk_params, result);

    py::dict out;
    out["status"] = status;

    // Return x_next (6,) with explicit strides
    std::vector<ssize_t> shape_xnext = {6};
    std::vector<ssize_t> strides_xnext = {sizeof(double)};
    auto x_next_arr = py::array_t<double>(shape_xnext, strides_xnext);
    std::memcpy(x_next_arr.mutable_data(), result.x_next, 6 * sizeof(double));
    out["x_next"] = x_next_arr;

    // Return observables with explicit strides
    std::vector<ssize_t> shape_3 = {3};
    std::vector<ssize_t> strides_3 = {sizeof(double)};
    auto p_tip_arr = py::array_t<double>(shape_3, strides_3);
    auto u_tip_arr = py::array_t<double>(shape_3, strides_3);
    std::memcpy(p_tip_arr.mutable_data(), result.p_tip, 3 * sizeof(double));
    std::memcpy(u_tip_arr.mutable_data(), result.u_tip, 3 * sizeof(double));
    out["p_tip"] = p_tip_arr;
    out["u_tip"] = u_tip_arr;

    // Return Jacobians (row-major 2D arrays)
    auto J_G_xnext_arr = py::array_t<double>({6, 6});
    auto J_G_xt_arr = py::array_t<double>({6, 6});
    auto J_G_ut_arr = py::array_t<double>({6, NUM_ACT_SET * 3});

    std::memcpy(J_G_xnext_arr.mutable_data(), result.J_G_xnext, 36 * sizeof(double));
    std::memcpy(J_G_xt_arr.mutable_data(), result.J_G_xt, 36 * sizeof(double));
    std::memcpy(J_G_ut_arr.mutable_data(), result.J_G_ut, 6 * NUM_ACT_SET * 3 * sizeof(double));

    out["J_G_xnext"] = J_G_xnext_arr;
    out["J_G_xt"] = J_G_xt_arr;
    out["J_G_ut"] = J_G_ut_arr;

    // Return equilibrium Jacobians at t+1
    auto J_p_u0_arr = py::array_t<double>({3, 3});
    auto J_p_ut_arr = py::array_t<double>({3, NUM_ACT_SET * 3});

    std::memcpy(J_p_u0_arr.mutable_data(), result.J_p_u0, 9 * sizeof(double));
    std::memcpy(J_p_ut_arr.mutable_data(), result.J_p_ut, 3 * NUM_ACT_SET * 3 * sizeof(double));

    out["J_p_u0"] = J_p_u0_arr;
    out["J_p_ut"] = J_p_ut_arr;

    // Return physics matrices (3x3)
    auto M_arr = py::array_t<double>({3, 3});
    auto D_arr = py::array_t<double>({3, 3});
    auto K_arr = py::array_t<double>({3, 3});

    std::memcpy(M_arr.mutable_data(), result.M, 9 * sizeof(double));
    std::memcpy(D_arr.mutable_data(), result.D, 9 * sizeof(double));
    std::memcpy(K_arr.mutable_data(), result.K, 9 * sizeof(double));

    out["M"] = M_arr;
    out["D"] = D_arr;
    out["K"] = K_arr;

    // Return cached inputs for backward pass matrix-dependence
    std::vector<ssize_t> shape_u_cached = {NUM_ACT_SET * 3};
    std::vector<ssize_t> strides_u_cached = {sizeof(double)};
    auto u_t_cached_arr = py::array_t<double>(shape_u_cached, strides_u_cached);
    std::memcpy(u_t_cached_arr.mutable_data(), result.u_t_cached, NUM_ACT_SET * 3 * sizeof(double));
    out["u_t_cached"] = u_t_cached_arr;

    out["dt_cached"] = result.dt_cached;
    out["L_inserted_cached"] = result.L_inserted_cached;

    auto K_tip_cached_arr = py::array_t<double>({3, 3});
    auto J_u_zc_cached_arr = py::array_t<double>({3, NUM_ACT_SET * 3});

    std::memcpy(K_tip_cached_arr.mutable_data(), result.K_tip_cached, 9 * sizeof(double));
    std::memcpy(J_u_zc_cached_arr.mutable_data(), result.J_u_zc_cached, 3 * NUM_ACT_SET * 3 * sizeof(double));

    out["K_tip_cached"] = K_tip_cached_arr;
    out["J_u_zc_cached"] = J_u_zc_cached_arr;

    // Return diagnostics
    out["lu_rank"] = result.lu_rank;
    out["rel_solve_residual"] = result.rel_solve_residual;
    out["solve_residual"] = result.solve_residual;
    out["converged"] = result.converged;
    out["exit_code"] = result.exit_code;

    // P1-5: API Versioning
    out["api_version"] = CRM_API_VERSION;
    out["api_contract"] = CRM_API_CONTRACT_DYNAMICS;

    return out;
}

// Wrapper for dynamics_backward
py::dict py_dynamics_backward(
    py::dict fwd_result,
    py::array_t<double> grad_x_next_arr,
    py::dict params_dict
) {
    // Validate grad_x_next
    auto grad_buf = grad_x_next_arr.request();
    if (grad_buf.ndim != 1 || grad_buf.shape[0] != 6) {
        throw std::runtime_error("grad_x_next must be 1D array of length 6");
    }
    if (!py::isinstance<py::array_t<double, py::array::c_style | py::array::forcecast>>(grad_x_next_arr)) {
        throw std::runtime_error("grad_x_next must be float64 C-contiguous array");
    }

    // Reconstruct DynamicsStepResult from cached data
    DynamicsStepResult cached;
    std::memset(&cached, 0, sizeof(DynamicsStepResult));

    // Extract x_next
    auto x_next_arr = fwd_result["x_next"].cast<py::array_t<double>>();
    if (x_next_arr.ndim() != 1 || x_next_arr.shape(0) != 6) {
        throw std::runtime_error("x_next must be (6,) array");
    }
    std::memcpy(cached.x_next, x_next_arr.data(), 6 * sizeof(double));

    // Extract Jacobians
    auto J_G_xnext_arr = fwd_result["J_G_xnext"].cast<py::array_t<double>>();
    auto J_G_xt_arr = fwd_result["J_G_xt"].cast<py::array_t<double>>();
    auto J_G_ut_arr = fwd_result["J_G_ut"].cast<py::array_t<double>>();

    if (J_G_xnext_arr.ndim() != 2 || J_G_xnext_arr.shape(0) != 6 || J_G_xnext_arr.shape(1) != 6) {
        throw std::runtime_error("J_G_xnext must be (6, 6) array");
    }
    if (J_G_xt_arr.ndim() != 2 || J_G_xt_arr.shape(0) != 6 || J_G_xt_arr.shape(1) != 6) {
        throw std::runtime_error("J_G_xt must be (6, 6) array");
    }
    if (J_G_ut_arr.ndim() != 2 || J_G_ut_arr.shape(0) != 6 || J_G_ut_arr.shape(1) != NUM_ACT_SET * 3) {
        throw std::runtime_error("J_G_ut must be (6, " + std::to_string(NUM_ACT_SET * 3) + ") array");
    }

    std::memcpy(cached.J_G_xnext, J_G_xnext_arr.data(), 36 * sizeof(double));
    std::memcpy(cached.J_G_xt, J_G_xt_arr.data(), 36 * sizeof(double));
    std::memcpy(cached.J_G_ut, J_G_ut_arr.data(), 6 * NUM_ACT_SET * 3 * sizeof(double));

    // Extract physics matrices
    auto M_arr = fwd_result["M"].cast<py::array_t<double>>();
    auto D_arr = fwd_result["D"].cast<py::array_t<double>>();
    auto K_arr = fwd_result["K"].cast<py::array_t<double>>();

    if (M_arr.ndim() != 2 || M_arr.shape(0) != 3 || M_arr.shape(1) != 3) {
        throw std::runtime_error("M must be (3, 3) array");
    }
    if (D_arr.ndim() != 2 || D_arr.shape(0) != 3 || D_arr.shape(1) != 3) {
        throw std::runtime_error("D must be (3, 3) array");
    }
    if (K_arr.ndim() != 2 || K_arr.shape(0) != 3 || K_arr.shape(1) != 3) {
        throw std::runtime_error("K must be (3, 3) array");
    }

    std::memcpy(cached.M, M_arr.data(), 9 * sizeof(double));
    std::memcpy(cached.D, D_arr.data(), 9 * sizeof(double));
    std::memcpy(cached.K, K_arr.data(), 9 * sizeof(double));

    // Extract cached inputs for matrix-dependence
    auto u_t_cached_arr = fwd_result["u_t_cached"].cast<py::array_t<double>>();
    if (u_t_cached_arr.ndim() != 1 || u_t_cached_arr.shape(0) != NUM_ACT_SET * 3) {
        throw std::runtime_error("u_t_cached must be (" + std::to_string(NUM_ACT_SET * 3) + ",) array");
    }
    std::memcpy(cached.u_t_cached, u_t_cached_arr.data(), NUM_ACT_SET * 3 * sizeof(double));

    cached.dt_cached = fwd_result["dt_cached"].cast<double>();
    cached.L_inserted_cached = fwd_result["L_inserted_cached"].cast<double>();

    auto K_tip_cached_arr = fwd_result["K_tip_cached"].cast<py::array_t<double>>();
    auto J_u_zc_cached_arr = fwd_result["J_u_zc_cached"].cast<py::array_t<double>>();

    if (K_tip_cached_arr.ndim() != 2 || K_tip_cached_arr.shape(0) != 3 || K_tip_cached_arr.shape(1) != 3) {
        throw std::runtime_error("K_tip_cached must be (3, 3) array");
    }
    if (J_u_zc_cached_arr.ndim() != 2 || J_u_zc_cached_arr.shape(0) != 3 || J_u_zc_cached_arr.shape(1) != NUM_ACT_SET * 3) {
        throw std::runtime_error("J_u_zc_cached must be (3, " + std::to_string(NUM_ACT_SET * 3) + ") array");
    }

    std::memcpy(cached.K_tip_cached, K_tip_cached_arr.data(), 9 * sizeof(double));
    std::memcpy(cached.J_u_zc_cached, J_u_zc_cached_arr.data(), 3 * NUM_ACT_SET * 3 * sizeof(double));

    // Parse FK params for backward pass
    CRMForwardKinematicsData fk_params = parse_fk_params(params_dict);

    // Call backward
    double* grad_x_next = static_cast<double*>(grad_buf.ptr);
    double grad_x_t[6];
    double grad_u_t[NUM_ACT_SET * 3];
    int lu_rank;
    double rel_residual;

    int status = dynamics_backward(cached, grad_x_next, fk_params, grad_x_t, grad_u_t, &lu_rank, &rel_residual);

    // DEBUG: Print what C++ returned (disabled)
    // std::cout << "DEBUG dynamics_backward: grad_x_t = [";
    // for (int i = 0; i < 6; i++) {
    //     std::cout << grad_x_t[i];
    //     if (i < 5) std::cout << ", ";
    // }
    // std::cout << "]" << std::endl;

    // Copy gradients to properly allocated arrays with explicit strides
    std::vector<ssize_t> shape_x = {6};
    std::vector<ssize_t> strides_x = {sizeof(double)};
    auto grad_x_t_arr = py::array_t<double>(shape_x, strides_x);
    std::memcpy(grad_x_t_arr.mutable_data(), grad_x_t, 6 * sizeof(double));

    std::vector<ssize_t> shape_u = {NUM_ACT_SET * 3};
    std::vector<ssize_t> strides_u = {sizeof(double)};
    auto grad_u_t_arr = py::array_t<double>(shape_u, strides_u);
    std::memcpy(grad_u_t_arr.mutable_data(), grad_u_t, NUM_ACT_SET * 3 * sizeof(double));

    py::dict out;
    out["status"] = status;
    out["grad_x_t"] = grad_x_t_arr;
    out["grad_u_t"] = grad_u_t_arr;
    out["lu_rank"] = lu_rank;
    out["rel_residual"] = rel_residual;

    // P1-5: API Versioning
    out["api_version"] = CRM_API_VERSION;
    out["api_contract"] = CRM_API_CONTRACT_DYNAMICS;

    return out;
}

// Load catheter parameters from file
CRMCatheterModelParams* py_load_cath_params(const std::string& filepath) {
    CRMCatheterModelParams params = Load_CRMCatheterModelParams(filepath.c_str());
    return new CRMCatheterModelParams(params);  // Use copy constructor
}

// Load catheter configuration from file
CatheterConfiguration* py_load_cath_config(const std::string& filepath) {
    CatheterConfiguration config = Load_CatheterConfiguration(filepath.c_str());
    return new CatheterConfiguration(config);  // Use copy constructor
}

// P1-5: API Versioning - compatibility check helper
bool check_api_compat(const std::string& required_version) {
    std::string current(CRM_API_VERSION);

    // Simple semantic versioning check: major.minor.patch
    // Compatible if major version matches and current >= required
    auto parse_version = [](const std::string& v) -> std::tuple<int, int, int> {
        int major = 0, minor = 0, patch = 0;
        std::sscanf(v.c_str(), "%d.%d.%d", &major, &minor, &patch);
        return {major, minor, patch};
    };

    auto [cur_major, cur_minor, cur_patch] = parse_version(current);
    auto [req_major, req_minor, req_patch] = parse_version(required_version);

    // Major version must match (breaking changes)
    if (cur_major != req_major) {
        throw std::runtime_error(
            "API major version mismatch: current=" + current +
            ", required=" + required_version +
            ". This indicates a breaking API change."
        );
    }

    // Current minor/patch must be >= required (backward compatible)
    if (cur_minor < req_minor || (cur_minor == req_minor && cur_patch < req_patch)) {
        throw std::runtime_error(
            "API version too old: current=" + current +
            ", required=" + required_version +
            ". Please update crm_diff_py."
        );
    }

    return true;
}

PYBIND11_MODULE(crm_diff_py, m) {
    m.doc() = "CRM Differentiable Simulator Python Bindings (CP2.3)";

    // P1-5: API Versioning - expose version constants
    m.attr("__version__") = CRM_PACKAGE_VERSION;
    m.attr("__api_version__") = CRM_API_VERSION;

    // P1-5: API Versioning - expose compatibility check
    m.def("check_api_compat", &check_api_compat,
          py::arg("required_version"),
          "Check if the current API version is compatible with the required version.\n"
          "Raises RuntimeError if incompatible.\n"
          "Compatible if major version matches and current >= required.");

    // Expose ContactModeType enum
    py::enum_<ContactModeType>(m, "ContactModeType")
        .value("FREE_TIP", ContactModeType::FREE_TIP)
        .value("FIXED_TIP", ContactModeType::FIXED_TIP);

    // Expose parameter loading functions
    m.def("load_cath_params", &py_load_cath_params,
          py::return_value_policy::take_ownership,
          "Load catheter parameters from file");

    m.def("load_cath_config", &py_load_cath_config,
          py::return_value_policy::take_ownership,
          "Load catheter configuration from file");

    // Expose equilibrium primitives
    m.def("equilibrium_forward", &py_equilibrium_forward,
          py::arg("u"), py::arg("L_inserted"), py::arg("params_dict"),
          "Forward pass: compute tip position and cache Jacobians");

    m.def("equilibrium_backward", &py_equilibrium_backward,
          py::arg("fwd_result"), py::arg("grad_p_tip"),
          "Backward pass: compute gradient w.r.t. actuation currents");

    // Expose dynamics primitives
    m.def("dynamics_forward", &py_dynamics_forward,
          py::arg("x_t"), py::arg("u_t"), py::arg("dt"), py::arg("L_inserted"), py::arg("params_dict"),
          "Dynamics forward pass: compute x_next and cache Jacobians");

    m.def("dynamics_backward", &py_dynamics_backward,
          py::arg("fwd_result"), py::arg("grad_x_next"), py::arg("params_dict"),
          "Dynamics backward pass: compute gradients w.r.t. x_t and u_t");

    // Expose opaque pointer types (needed for parameter dict)
    py::class_<CRMCatheterModelParams>(m, "CRMCatheterModelParams");
    py::class_<CatheterConfiguration>(m, "CatheterConfiguration");
}
