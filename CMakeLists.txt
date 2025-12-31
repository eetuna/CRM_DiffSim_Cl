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

# Python bindings (optional, requires pybind11)
if(pybind11_FOUND)
    pybind11_add_module(crm_diff_py
        python/crm_bindings.cpp
    )
    target_link_libraries(crm_diff_py PRIVATE CRMCPPLib)
    target_compile_definitions(crm_diff_py PRIVATE VERSION_INFO="1.0.0")
endif()
