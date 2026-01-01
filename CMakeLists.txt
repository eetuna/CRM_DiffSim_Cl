cmake_minimum_required(VERSION 3.17)
project(CRM_DiffSim_Cl)

find_package(Eigen3 REQUIRED NO_MODULE)
find_package(pybind11 CONFIG)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_INCLUDE_CURRENT_DIR_IN_INTERFACE ON)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_MODULE_PATH
    ${CMAKE_MODULE_PATH}
    ${CMAKE_CURRENT_SOURCE_DIR}/../
)

set(CRMSOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_MatrixOperations.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_StateVector_Definitions.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_BVPIVP_APIDeclarations.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_IVP_NumericalIntegrationTemplates.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_IVPSolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_BVPSolver.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_IVPJacobian.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_IVPJacobian_InternalAPI.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_IVPJacobian.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_FK_InternalAPI.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_ForwardKinematics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_CatheterClass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_SupportFunctions.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_DiffEquilibrium.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_DiffEquilibrium.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_DiffDynamics.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRM_DiffDynamics.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/numerical/minpack.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/numerical/minpack_TemplateDefinitions.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/numerical/minpack_wGivenJac_TemplateDefinitions.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/numerical/minpack.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/numerical/minpack_DYN.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/numerical/minpack_DYN_Defs.cpp

    ${CMAKE_CURRENT_SOURCE_DIR}/src/CoilDynamics_Defs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRMDYN.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/CRMDYN_Numerical_Integration.hpp
)

add_library(CRMCPPLib STATIC
    ${CRMSOURCES}
)

target_include_directories(CRMCPPLib PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/numerical"
)

target_link_libraries(CRMCPPLib PUBLIC
    Eigen3::Eigen
)

target_compile_options(CRMCPPLib PRIVATE -Wno-pragma-once-outside-header)

# Enable -fPIC for shared library linking (needed for pybind11)
set_target_properties(CRMCPPLib PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_executable(CRMDYNTest
    main/CRMDYN_test.cpp
    main/CRMTest.h
    main/CRM_KinematicsTestFunctions.hpp
    main/CRM_KinematicsTestFunctions.cpp
    ${CRMSOURCES}
)

add_executable(CRMTest
    main/CRMTest.cpp
    main/CRMTest.h
    main/CRM_KinematicsTestFunctions.hpp
    main/CRM_KinematicsTestFunctions.cpp
    ${CRMSOURCES}
)

target_link_libraries(CRMTest PUBLIC
    Eigen3::Eigen
)

target_link_libraries(CRMDYNTest PUBLIC
    Eigen3::Eigen
)

target_include_directories(CRMTest PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/numerical"
)

target_include_directories(CRMDYNTest PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/numerical"
)

add_executable(test_cp15_harness
    test_cp15_harness.cpp
)

target_link_libraries(test_cp15_harness PUBLIC
    CRMCPPLib
)

target_include_directories(test_cp15_harness PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/numerical"
)

add_executable(test_physics_audit
    test_physics_audit.cpp
)

target_link_libraries(test_physics_audit PUBLIC
    CRMCPPLib
)

target_include_directories(test_physics_audit PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/numerical"
)

add_executable(test_li_sweep
    test_li_sweep.cpp
)

target_link_libraries(test_li_sweep PUBLIC
    CRMCPPLib
)

target_include_directories(test_li_sweep PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/numerical"
)

add_executable(test_backward_bug
    test_backward_bug.cpp
)

target_link_libraries(test_backward_bug PUBLIC
    CRMCPPLib
)

target_include_directories(test_backward_bug PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/numerical"
)

add_executable(test_golden_backward
    test_golden_backward.cpp
)

target_link_libraries(test_golden_backward PUBLIC
    CRMCPPLib
)

target_include_directories(test_golden_backward PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/numerical"
)

add_executable(test_straight_rod
    test_straight_rod.cpp
)

target_link_libraries(test_straight_rod PUBLIC
    CRMCPPLib
)

target_include_directories(test_straight_rod PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/numerical"
)

add_executable(test_cp21_dynamics_smoke
    test_cp21_dynamics_smoke.cpp
)

target_link_libraries(test_cp21_dynamics_smoke PUBLIC
    CRMCPPLib
)

target_include_directories(test_cp21_dynamics_smoke PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/numerical"
)

add_executable(test_cp22_dynamics_fd
    test_cp22_dynamics_fd.cpp
)

target_link_libraries(test_cp22_dynamics_fd PUBLIC
    CRMCPPLib
)

target_include_directories(test_cp22_dynamics_fd PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/numerical"
)

add_executable(test_cp31_linearization
    test_cp31_linearization.cpp
)

target_link_libraries(test_cp31_linearization PUBLIC
    CRMCPPLib
)

target_include_directories(test_cp31_linearization PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/numerical"
)

# Python bindings (optional, requires pybind11)
if(pybind11_FOUND)
    pybind11_add_module(crm_diff_py
        python/crm_bindings.cpp
    )
    target_link_libraries(crm_diff_py PRIVATE CRMCPPLib)
    target_compile_definitions(crm_diff_py PRIVATE VERSION_INFO="1.0.0")
endif()

# Enable CTest
enable_testing()

# PHASE A: Golden C++ backward test (CP1.4)
add_test(NAME golden_backward
         COMMAND test_golden_backward
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# PHASE B: C++ <-> Python smoke gate (CP1.5)
# Requires pybind11 module to be built
if(pybind11_FOUND)
    add_test(NAME cpp_python_smoke
             COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}
                     python3 ${CMAKE_SOURCE_DIR}/python/test_cp15_smoke.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# C++ reference harness (used by Python smoke test)
add_test(NAME cpp_reference_harness
         COMMAND test_cp15_harness
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# PHASE C: Straight-rod physics sanity test
add_test(NAME straight_rod_physics
         COMMAND test_straight_rod
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# CP2.1: Dynamics smoke test
add_test(NAME dynamics_smoke_cp21
         COMMAND test_cp21_dynamics_smoke
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# CP2.2: Dynamics finite difference validation
add_test(NAME dynamics_fd_cp22
         COMMAND test_cp22_dynamics_fd
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# CP2.3: Dynamics Python bindings smoke test
if(pybind11_FOUND)
    add_test(NAME dynamics_smoke_cp23_python
             COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}
                     python3 ${CMAKE_SOURCE_DIR}/python/test_cp23_dynamics_smoke.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# CP2.4: Dynamics PyTorch autograd gradcheck validation
if(pybind11_FOUND)
    add_test(NAME dynamics_gradcheck_cp24_python
             COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}
                     python3 ${CMAKE_SOURCE_DIR}/python/test_cp24_dynamics_gradcheck.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# CP2.5: Multi-step rollout + backprop smoke test
if(pybind11_FOUND)
    add_test(NAME dynamics_rollout_cp25_python
             COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}
                     python3 ${CMAKE_SOURCE_DIR}/python/test_cp25_dynamics_rollout_smoke.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# P1-1: Robustness sweep stress test (randomized operating points)
if(pybind11_FOUND)
    add_test(NAME dynamics_robustness_sweep_p1_1
             COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}
                     python3 ${CMAKE_SOURCE_DIR}/python/test_p1_1_robustness_sweep.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(dynamics_robustness_sweep_p1_1 PROPERTIES TIMEOUT 120)
endif()

# P1-5: API Versioning validation
if(pybind11_FOUND)
    add_test(NAME api_versioning_p1_5
             COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:$ENV{PYTHONPATH}
                     python3 ${CMAKE_SOURCE_DIR}/python/test_p1_5_api_versioning.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(api_versioning_p1_5 PROPERTIES TIMEOUT 5)
endif()

# CP3.1: Linearization validation (C++)
add_test(NAME test_cp31_linearization_cpp
         COMMAND test_cp31_linearization
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# CP3.1: Linearization validation (Python + PyTorch)
if(pybind11_FOUND)
    add_test(NAME dynamics_linearization_cp31_python
             COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:python:$ENV{PYTHONPATH}
                     python3 ${CMAKE_SOURCE_DIR}/python/test_cp31_linearization.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
endif()

# CP3.2: iLQR trajectory optimization to fixed target
if(pybind11_FOUND)
    add_test(NAME ilqr_fixed_target_cp32_python
             COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:python:$ENV{PYTHONPATH}
                     python3 ${CMAKE_SOURCE_DIR}/python/test_cp32_ilqr_fixed_target.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(ilqr_fixed_target_cp32_python PROPERTIES TIMEOUT 120)
endif()

# CP3.2.1: iLQR descent regression test
if(pybind11_FOUND)
    add_test(NAME ilqr_descent_regression_cp32_python
             COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:python:$ENV{PYTHONPATH}
                     python3 ${CMAKE_SOURCE_DIR}/python/test_cp32_ilqr_descent_regression.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(ilqr_descent_regression_cp32_python PROPERTIES TIMEOUT 60)
endif()

# CP3.3: MPC receding-horizon tracking
if(pybind11_FOUND)
    add_test(NAME mpc_tracking_cp33_python
             COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:python:$ENV{PYTHONPATH}
                     python3 ${CMAKE_SOURCE_DIR}/python/test_cp33_mpc_tracking.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(mpc_tracking_cp33_python PROPERTIES TIMEOUT 300)
endif()

# P1-3: Terminal Hessian improvement regression test (nightly-only, not in fast gate)
if(pybind11_FOUND)
    add_test(NAME terminal_hessian_improves_descent_p1_3
             COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:python:$ENV{PYTHONPATH}
                     python3 ${CMAKE_SOURCE_DIR}/python/test_p1_3_terminal_hessian_improves_descent.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(terminal_hessian_improves_descent_p1_3 PROPERTIES TIMEOUT 180)
endif()

# CP3.5: Dataset-driven NPZ trajectory replay regression test (nightly-only, not in fast gate)
if(pybind11_FOUND)
    add_test(NAME replay_npz_tracking_cp35
             COMMAND ${CMAKE_COMMAND} -E env PYTHONPATH=${CMAKE_BINARY_DIR}:${CMAKE_SOURCE_DIR}/python:$ENV{PYTHONPATH}
                     python3 ${CMAKE_SOURCE_DIR}/python/test_cp35_replay_npz_tracking.py
             WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    set_tests_properties(replay_npz_tracking_cp35 PROPERTIES TIMEOUT 180)
endif()
