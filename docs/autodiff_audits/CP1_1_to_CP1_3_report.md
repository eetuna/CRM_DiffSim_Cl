# CRM Differentiable Simulator: Checkpoints CP1.1–CP1.3 Completion Report

**Generated:** 2024-12-30
**Repository:** CRM_DiffSim_Cl
**Implementation Phase:** v1.0 Equilibrium Forward/Backward (Partial)

---

## CP1.1: pybind11 Build Integration

### Modified Files
- `CMakeLists.txt`
- `python/crm_bindings.cpp` (new)

### What Changed

**CMakeLists.txt:**
- Line 5: Added `find_package(pybind11 CONFIG)` to detect pybind11
- Line 67: Set `POSITION_INDEPENDENT_CODE ON` for `CRMCPPLib` target (required for shared library linking)
- Lines 36-37: Added `CRM_DiffEquilibrium.hpp` and `CRM_DiffEquilibrium.cpp` to `CRMSOURCES`
- Lines 100-107: Added conditional pybind11 module target `crm_diff_py` that links to `CRMCPPLib`

**python/crm_bindings.cpp:**
- Created minimal PYBIND11_MODULE stub for initial build validation

### How to Reproduce

**Clean configure/build:**
```bash
cd /workspaces/CRM_DiffSim_Cl/build
rm -rf *
cmake ..
make crm_diff_py
```

**Verify module built:**
```bash
ls -lh build/crm_diff_py.cpython-310-x86_64-linux-gnu.so
python3 -c "import sys; sys.path.insert(0, 'build'); import crm_diff_py; print(crm_diff_py.__doc__)"
```

### Verification Evidence

**Test harness:** None (build-only checkpoint)

**Build artifacts:**
- `build/crm_diff_py.cpython-310-x86_64-linux-gnu.so` (459K)
- `build/libCRMCPPLib.a` (static library with -fPIC)

**Command to verify:**
```bash
python3 -c "import sys; sys.path.insert(0, 'build'); import crm_diff_py"
# Should import without error
```

### Evidence Snippets

**CMakeLists.txt (lines 100-107):**
```cmake
# Python bindings (optional, requires pybind11)
if(pybind11_FOUND)
    pybind11_add_module(crm_diff_py
        python/crm_bindings.cpp
    )
    target_link_libraries(crm_diff_py PRIVATE CRMCPPLib)
    target_compile_definitions(crm_diff_py PRIVATE VERSION_INFO="1.0.0")
endif()
```

**CMakeLists.txt (line 67):**
```cmake
set_target_properties(CRMCPPLib PROPERTIES POSITION_INDEPENDENT_CODE ON)
```

**python/crm_bindings.cpp (initial stub):**
```cpp
PYBIND11_MODULE(crm_diff_py, m) {
    m.doc() = "CRM Differentiable Simulator Python Bindings";
}
```

---

## CP1.2: equilibrium_forward() C++ Wrapper

### Modified Files
- `src/CRM_DiffEquilibrium.hpp` (new)
- `src/CRM_DiffEquilibrium.cpp` (new)
- `CMakeLists.txt` (added sources to build)

### What Changed

**src/CRM_DiffEquilibrium.hpp:**
- Defined `struct EquilibriumResult` with tip position, deltau0, converged status, and Jacobian placeholders (J_p_u0, J_u_u0, J_p_zc, J_u_zc, K_tip)
- Declared `int equilibrium_forward(...)` function signature

**src/CRM_DiffEquilibrium.cpp:**
- Implemented `equilibrium_forward()` that wraps `CRM_ForwardKinematics()`
- Constructs `CRMShootingMethodParams`, calls `CRMShootingMethodBVP()` to solve for deltau0
- Extracts p_tip and deltau0 from legacy solver output
- Initializes Jacobian storage to zero (populated in CP1.3)

### How to Reproduce

**Build test harness:**
```bash
cd /workspaces/CRM_DiffSim_Cl
g++ -std=c++17 -I./src -I./numerical -I/usr/include/eigen3 \
    test_cp12.cpp -L./build -lCRMCPPLib -o build/test_cp12
```

**Run verification:**
```bash
./build/test_cp12
```

**Expected output:**
```
CP1.2 Test - equilibrium_forward() wrapper

Input: u=[0,0,0], L=50mm
Status: 0 (0=success)
Converged: 0
p_tip: [0.947975, -2.87329, 49.8114]
deltau0: [-1.45029e-05, 3.31437e-06, -2.08075e-11]

CP1.2: PASS - equilibrium_forward() wrapper works
```

### Verification Evidence

**Test harness:** `test_cp12.cpp`
- Loads catheter parameters from `catheterdata/CatheterParameterSet_1_new.txt`
- Calls `equilibrium_forward()` with u=[0,0,0], L=50mm
- Verifies status==0, converged==0, p_tip is non-zero

**Command:**
```bash
./build/test_cp12
```

### Evidence Snippets

**src/CRM_DiffEquilibrium.hpp (struct EquilibriumResult):**
```cpp
struct EquilibriumResult {
    // Outputs
    double p_tip[3];                      // Tip position (mm)
    double deltau0[3];                    // Solved base curvature
    int converged;                        // 0=success, >0=local min

    // Cached Jacobians for backward (row-major storage)
    double J_p_u0[9];                     // ∂p_tip/∂Δu₀ (3×3)
    double J_u_u0[9];                     // ∂u_tip/∂Δu₀ (3×3)
```

**src/CRM_DiffEquilibrium.hpp (function declaration):**
```cpp
int equilibrium_forward(
    const double u[NUM_ACT_SET*3],       // Actuation currents (Amperes)
    double L_inserted,                    // Insertion length (mm)
    const CRMForwardKinematicsData& params,
    EquilibriumResult& out
);
```

**src/CRM_DiffEquilibrium.cpp (BVP call):**
```cpp
CRMShootingMethodBVP(BVPParams, deltau0_init, ftip_init,
                     deltau0_calc, ftip_calc, localmin);

out.converged = localmin;
for (int i = 0; i < 3; i++) {
    out.deltau0[i] = deltau0_calc[i];
}
```

---

## CP1.3: Jacobian Caching in Forward Pass

### Modified Files
- `src/CRM_DiffEquilibrium.cpp` (modified `equilibrium_forward()`)
- `src/CRM_IVPJacobian.cpp` (patched pseudoinverse at lines 91-137)

### What Changed

**src/CRM_DiffEquilibrium.cpp (`equilibrium_forward()`):**
- Lines 64-88: Call `CRMSolverIVP_Prep()` and `CRMSolverIVP_CoreWithJacobian<IVPJacobiansFull>()` instead of non-Jacobian path
- Lines 91-106: Extract Jacobian blocks from `AugmentedStateVector<IVPJacobiansFull> x_N`:
  - J_p_u0 (3×3) from `x_N._p_u0`
  - J_u_u0 (3×3) from `x_N._u_u0`
  - J_p_zc (3×3N) from `x_N._p_zc` (actuator order reversed)
  - J_u_zc (3×3N) from `x_N._u_zc` (actuator order reversed)
- Lines 109-117: Extract K_tip from last flexible segment (or Identity if rigid)

**src/CRM_IVPJacobian.cpp (`CRMSolverIVPJacobian()`):**
- Line 6: Added `#include <iostream>`
- Lines 91-137: Replaced `pseudoInverse()` with FullPivLU solve:
  - Line 93: `Eigen::FullPivLU<Matrix3d> lu_u_u0(JIVP_u_u0);`
  - Lines 94-103: Rank check, fail-fast if rank < 3
  - Lines 105-111: Solve `JIVP_u_u0 * X_z = JIVP_u_z` with residual check
  - Lines 113-119: Solve `JIVP_u_u0 * X_ft = JIVP_u_ft` with residual check
  - Lines 127-135: LU solve for FIXED_TIP Jacobian

### How to Reproduce

**Rebuild library with patched code:**
```bash
cd /workspaces/CRM_DiffSim_Cl/build
make CRMCPPLib -j4
```

**Build and run CP1.3 test:**
```bash
g++ -std=c++17 -I./src -I./numerical -I/usr/include/eigen3 \
    test_cp13.cpp -L./build -lCRMCPPLib -o build/test_cp13
./build/test_cp13
```

**Expected output:**
```
CP1.3 Test - Jacobian caching

Status: 0
p_tip: [0.947975, -2.87329, 49.8114]

J_p_u0 (3×3, row-major):
-0.0166632 740.729 43.9784
-740.767 -0.192095 32.4981
-47.328 -32.8331 0.178938

J_u_u0 (3×3, row-major):
0.986202 0.0128594 -0.161
-0.000714015 1.00079 0.0589993
0.206551 -0.073919 0.981065

K_tip (3×3, row-major):
22.8304 0 0
0 22.8304 0
0 0 20.2125

Frobenius norms:
||J_p_u0|| = 1050.58
||J_u_u0|| = 1.73619
||K_tip|| = 38.092

CP1.3: PASS - Jacobians cached
```

### Verification Evidence

**Test harness:** `test_cp13.cpp`
- Calls `equilibrium_forward()` with u=[0,0,0], L=50mm
- Verifies J_p_u0, J_u_u0, K_tip are non-zero (Frobenius norms > 1e-6)
- Prints Jacobian matrices for visual inspection

**Commands:**
```bash
./build/test_cp13
# Verify no "ERROR: JIVP_u_u0 rank-deficient" messages
# Verify Frobenius norms are non-zero
```

### Evidence Snippets

**src/CRM_DiffEquilibrium.cpp (IVP Jacobian call, lines 72-88):**
```cpp
CRMSolverIVP_Prep(
    BVPParams.no_flex_seg, BVPParams.no_rigid_seg, BVPParams.no_act_set,
    BVPParams.no_locmarkers, BVPParams.no_fcum_steps,
    x_0, BVPParams.IntegrationStepSize,
    BVPParams.Li, BVPParams.dlambdainv,
    BVPParams.SegmentTypes,
    BVPParams.SegEndLambdas, BVPParams.LocMarkerLambdas,
    BVPParams.rho,
    BVPParams.K, BVPParams.Kinv, BVPParams.ustar,
    BVPParams.ActMass,
    BVPParams.CoilAlignmentTurnAreaMatrix,
    BVPParams.MagMoment, BVPParams.fcumlambda,
    BVPParams.B0, BVPParams.g,
    false, true, CoreParams);

AugmentedStateVector<IVPJacobiansFull> x_N;
CRMSolverIVP_CoreWithJacobian(CoreParams, deltau0_calc, ftip_calc, x_N, MomentResidual);
```

**src/CRM_DiffEquilibrium.cpp (Jacobian extraction, lines 91-94):**
```cpp
for (int i = 0; i < 9; i++) {
    out.J_p_u0[i] = x_N._p_u0[i];
    out.J_u_u0[i] = x_N._u_u0[i];
}
```

**src/CRM_IVPJacobian.cpp (pseudoinverse patch, lines 91-103):**
```cpp
// PATCH (CP1.3): Replace pseudoinverse with LU solve
// Solve JIVP_u_u0 * X = RHS using FullPivLU
Eigen::FullPivLU<Matrix3d> lu_u_u0(JIVP_u_u0);
if (lu_u_u0.rank() < 3) {
    std::cerr << "ERROR: JIVP_u_u0 rank-deficient, rank=" << lu_u_u0.rank() << std::endl;
    // Fail-fast: return zero Jacobians
    MatrixXd zero_pz = MatrixXd::Zero(3, Cs + 1);
    MatrixXd zero_wsz = MatrixXd::Zero(3, Cs + 1);
    MatrixXd zero_pft = MatrixXd::Zero(3, 3);
    MatrixXd zero_wsft = MatrixXd::Zero(3, 3);
    MatrixXd zero_ftz = MatrixXd::Zero(3, Cs + 1);
    return { zero_pz, zero_wsz, zero_pft, zero_wsft, zero_ftz };
}
```

**src/CRM_IVPJacobian.cpp (LU solve + residual check, lines 105-111):**
```cpp
// Solve JIVP_u_u0 * X_z = JIVP_u_z
Matrix<double, 3, Cs + 1, RowMajor> X_z = lu_u_u0.solve(JIVP_u_z);
double resid_z = (JIVP_u_u0 * X_z - JIVP_u_z).norm();
double rel_resid_z = resid_z / std::max(JIVP_u_z.norm(), 1.0);
if (rel_resid_z > 1e-10) {
    std::cerr << "WARNING: JIVP_u_u0 solve residual=" << rel_resid_z << std::endl;
}
```

---

## Potentially Missed Changes

### Untracked Files
- `.devcontainer/` (IDE configuration, not source code)
- `.gitignore` (git configuration)
- `python/` (entire directory - appears to contain CP1.4/CP1.5 work):
  - `python/crm_bindings.cpp` (more extensive than CP1.1 stub)
  - `python/crm_diff_py.cpython-310-x86_64-linux-gnu.so` (compiled module)
  - `python/crm_equilibrium.py` (Python wrapper, likely CP1.5)
  - `python/test_cp15.py` (test harness, likely CP1.5/CP1.6)
  - `python/test_debug.py` (debugging script)
  - `python/test_config.py` (configuration)
- `requirements.txt` (Python dependencies)
- `test_cp12.cpp` (test harness for CP1.2)
- `test_cp13.cpp` (test harness for CP1.3)

### Build Artifacts in Repository
- `build/` (many modified files - these are generated, should be in .gitignore)
- All `.o`, `.a`, `.so` files under `build/`
- `build/CMakeCache.txt`, `build/Makefile`, etc.

### Changes Confidently Assigned to Later Checkpoints (Not CP1.1-CP1.3)
Based on system reminders, the following appear to be CP1.4+ work:
- `src/CRM_DiffEquilibrium.cpp`: `equilibrium_backward()` function (lines 136-196)
- `src/CRM_DiffEquilibrium.hpp`: `equilibrium_backward()` declaration (lines 38-46)
- `python/crm_bindings.cpp`: Full pybind11 implementation with `py_equilibrium_forward()`, `py_equilibrium_backward()` (CP1.5)
- `python/crm_equilibrium.py`: PyTorch autograd.Function wrapper (CP1.7)
- `python/test_cp15.py`: Python test suite (CP1.6)

### Uncertain Attribution
- **UNKNOWN**: When exactly was `POSITION_INDEPENDENT_CODE ON` added to CMakeLists.txt? (Attributed to CP1.1 based on timing and necessity for pybind11)
- **UNKNOWN**: Whether `python/crm_bindings.cpp` was created as a stub in CP1.1 or fully implemented later (system reminders suggest it was expanded in CP1.5)

---

## Reproduction Commands Summary

### Clean Build from Scratch
```bash
cd /workspaces/CRM_DiffSim_Cl/build
rm -rf *
cmake ..
make CRMCPPLib -j4
make crm_diff_py
```

### Verify CP1.1
```bash
ls -lh build/crm_diff_py.cpython-310-x86_64-linux-gnu.so
python3 -c "import sys; sys.path.insert(0, 'build'); import crm_diff_py"
```

### Verify CP1.2
```bash
g++ -std=c++17 -I./src -I./numerical -I/usr/include/eigen3 \
    test_cp12.cpp -L./build -lCRMCPPLib -o build/test_cp12
./build/test_cp12
# Expected: "CP1.2: PASS"
```

### Verify CP1.3
```bash
g++ -std=c++17 -I./src -I./numerical -I/usr/include/eigen3 \
    test_cp13.cpp -L./build -lCRMCPPLib -o build/test_cp13
./build/test_cp13
# Expected: "CP1.3: PASS"
```

---

**Report End**
