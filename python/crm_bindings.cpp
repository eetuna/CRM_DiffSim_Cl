#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "CRM.hpp"
#include "CRM_DiffEquilibrium.hpp"

namespace py = pybind11;
using namespace CRMCatheterModel;

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

PYBIND11_MODULE(crm_diff_py, m) {
    m.doc() = "CRM Differentiable Simulator Python Bindings (CP1.5)";

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

    // Expose opaque pointer types (needed for parameter dict)
    py::class_<CRMCatheterModelParams>(m, "CRMCatheterModelParams");
    py::class_<CatheterConfiguration>(m, "CatheterConfiguration");
}
