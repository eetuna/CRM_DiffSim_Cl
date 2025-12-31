# CI Safety Net - Regression Test Suite

## Overview

This document describes the permanent regression test suite and CI infrastructure that locks in the current working state of the CRM differentiable equilibrium solver.

**Purpose**: Ensure we never re-debug "physics vs bindings vs gradients" issues again.

**Last Updated**: 2025-12-31

---

## Test Suite Components

### PHASE A: Golden C++ Backward Test (CP1.4)

**File**: `test_golden_backward.cpp`

**What it tests**:
- Equilibrium backward pass using implicit differentiation
- Tests 3 canonical gradient directions: [1,0,0], [0,1,0], [0,0,1]
- Verifies backward solver correctness

**Assertions**:
- `status == 0` (solver converged)
- `lu_rank == 3` (full-rank system)
- `rel_residual < 1e-10` (accurate solve)
- `grad_u` is finite and non-zero

**Test configuration**:
- Catheter params: `./catheterdata/CatheterParameterSet_1_dyn.txt`
- Spatial config: `./catheterdata/CatheterSpatialConfiguration_1.txt`
- Actuation: `u = [0, 0, 0]`
- Insertion: `Li = 50.0 mm`

**Exit code**: 0 on success, 1 on failure

---

### PHASE B: C++ ↔ Python Smoke Gate (CP1.5)

**Files**:
- C++ harness: `test_cp15_harness.cpp`
- Python test: `python/test_cp15_smoke.py`

**What it tests**:
- Exact numerical agreement between C++ and Python bindings
- Forward pass outputs for smoke test case

**Compared values** (tolerance = 1e-12):
- `p_tip` (tip position)
- `deltau0` (base curvature)
- `J_p_u0`, `J_u_u0`, `J_p_zc`, `J_u_zc` (Jacobian matrices)
- `K_tip` (stiffness matrix)
- Frobenius norms of all matrices
- Configuration fingerprint (exact string match)

**Test configuration**:
- Same as PHASE A (u=[0,0,0], Li=50mm)

**Exit code**: 0 on exact match, 1 on mismatch (prints side-by-side diff)

---

### PHASE C: Straight-Rod Physics Sanity

**File**: `test_straight_rod.cpp`

**What it tests**:
- Straight-rod limit case: zero curvature, zero gravity
- Verifies basic physics: straight catheter should align with Z-axis

**Parameter files** (straight-rod mode):
- `./catheterdata/CatheterParameterSet_1_straight.txt` (ustar=0)
- `./catheterdata/CatheterSpatialConfiguration_1_straight.txt` (gravity=0)

**Test cases**:
- Li = 0 mm → p_tip ≈ [0, 0, 0]
- Li = 50 mm → p_tip ≈ [0, 0, 50]
- Li = 100 mm → p_tip ≈ [0, 0, 100]

**Assertions**:
- `|x| < 1e-9`
- `|y| < 1e-9`
- `|z - Li| < 1e-9`

**Exit code**: 0 on success, 1 on failure

---

## Local Test Execution

### Build all tests

```bash
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4 CRMCPPLib test_golden_backward test_cp15_harness test_straight_rod crm_diff_py
```

### Run all CTests

```bash
ctest --output-on-failure
```

**Expected output**:
```
Test project /workspaces/CRM_DiffSim_Cl/build
    Start 1: golden_backward
1/4 Test #1: golden_backward ..................   Passed
    Start 2: cpp_python_smoke
2/4 Test #2: cpp_python_smoke .................   Passed
    Start 3: cpp_reference_harness
3/4 Test #3: cpp_reference_harness ............   Passed
    Start 4: straight_rod_physics
4/4 Test #4: straight_rod_physics .............   Passed

100% tests passed, 0 tests failed out of 4
```

### Run individual tests

```bash
# Golden backward test
./build/test_golden_backward

# C++ reference harness
./build/test_cp15_harness

# C++ ↔ Python smoke gate
python3 python/test_cp15_smoke.py

# Straight-rod physics
./build/test_straight_rod
```

---

## GitHub Actions CI

### PR Checks (`.github/workflows/pr_checks.yml`)

**Triggers**:
- Pull requests to `main`
- Pushes to `main`

**Tests executed**:
1. Build in Release mode
2. `golden_backward` - C++ backward pass regression
3. `cpp_reference_harness` - C++ forward reference
4. `cpp_python_smoke` - C++ ↔ Python exact match gate

**Fail conditions**:
- Build failure
- Any test exit code != 0
- Python binding mismatch > 1e-12

---

### Nightly Tests (`.github/workflows/nightly.yml`)

**Triggers**:
- Scheduled: 2 AM UTC daily
- Manual dispatch

**Tests executed**:
1. `straight_rod_physics` - Straight-rod sanity checks
2. `test_li_sweep` - Li sweep baseline (C++)
3. `test_li_sweep_smoke.py` - Li sweep smoke test (Python)

**Artifacts**:
- `li_sweep_*.csv` results (retained 30 days)

**Fail conditions**:
- Straight-rod test failure (hard gate)
- Li sweep failures are logged but don't fail workflow (continue-on-error)

---

## Files Added/Modified

### New C++ Test Executables
- `test_golden_backward.cpp` (PHASE A)
- `test_straight_rod.cpp` (PHASE C)

### New Catheter Parameter Files
- `catheterdata/CatheterParameterSet_1_straight.txt` (ustar=0)
- `catheterdata/CatheterSpatialConfiguration_1_straight.txt` (gravity=0)

### Modified Files
- `CMakeLists.txt`:
  - Added `enable_testing()`
  - Added `test_golden_backward` and `test_straight_rod` executables
  - Added 4 CTest entries: `golden_backward`, `cpp_python_smoke`, `cpp_reference_harness`, `straight_rod_physics`

### CI Workflows
- `.github/workflows/pr_checks.yml` - PR and push checks
- `.github/workflows/nightly.yml` - Nightly regression suite

### Documentation
- `CI_SAFETY_NET.md` - This file

---

## Test Results Summary

### ✓ PHASE A: Golden Backward Test

```
Test 1/3: grad_p_tip = e_X = [1, 0, 0]
  status = 0, lu_rank = 3, rel_residual = 1.58e-16
  grad_u = [1.161, 0.117, -0.003]
  PASS

Test 2/3: grad_p_tip = e_Y = [0, 1, 0]
  status = 0, lu_rank = 3, rel_residual = 4.26e-21
  grad_u = [-0.138, -1.109, 83.022]
  PASS

Test 3/3: grad_p_tip = e_Z = [0, 0, 1]
  status = 0, lu_rank = 3, rel_residual = 3.50e-17
  grad_u = [0.001, -0.009, 0.701]
  PASS
```

### ✓ PHASE B: C++ ↔ Python Smoke Gate

All values match within 1e-12:
- ✓ p_tip: MATCH
- ✓ deltau0: MATCH
- ✓ J_p_u0: MATCH (max_abs=1.73e-14, max_rel=1.08e-15)
- ✓ J_u_u0: MATCH (max_abs=2.11e-15, max_rel=2.11e-15)
- ✓ J_p_zc: MATCH
- ✓ J_u_zc: MATCH
- ✓ K_tip: MATCH
- ✓ Frobenius norms: MATCH
- ✓ fingerprint: EXACT MATCH

### ✓ PHASE C: Straight-Rod Physics

```
Test 1/3: Li = 0 mm
  p_tip = [0, 0, 0]
  Errors: |x| = 0, |y| = 0, |z - Li| = 0
  PASS

Test 2/3: Li = 50 mm
  p_tip = [0, 0, 50.000000000000121]
  Errors: |x| = 0, |y| = 0, |z - Li| = 1.21e-13
  PASS

Test 3/3: Li = 100 mm
  p_tip = [0, 0, 100.00000000000023]
  Errors: |x| = 0, |y| = 0, |z - Li| = 2.27e-13
  PASS
```

---

## Maintenance Notes

### What NOT to modify

- **DO NOT** relax tolerances in tests
- **DO NOT** modify solver math without updating regression tests
- **DO NOT** skip/disable tests in CI without review

### Adding new tests

1. Create test executable in root directory: `test_<name>.cpp`
2. Add to `CMakeLists.txt`:
   ```cmake
   add_executable(test_<name> test_<name>.cpp)
   target_link_libraries(test_<name> PUBLIC CRMCPPLib)
   target_include_directories(test_<name> PUBLIC "${PROJECT_SOURCE_DIR}/src" "${PROJECT_SOURCE_DIR}/numerical")
   ```
3. Add CTest entry:
   ```cmake
   add_test(NAME <name> COMMAND test_<name> WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
   ```
4. Update CI workflows if needed

### Updating golden values

If solver improvements require updating golden values:
1. Run tests locally and verify new behavior is correct
2. Update test assertions/tolerances
3. Document changes in commit message
4. Update this document with new expected results

---

## Dependencies

### System packages (Ubuntu/Debian)
- `cmake >= 3.17`
- `g++` (C++17 support)
- `libeigen3-dev`
- `python3`
- `python3-pip`

### Python packages
- `numpy`
- `pybind11`

---

## Contact

For questions about this test suite, see:
- Git history: Commits on branch `lock_ci_safety_net`
- CI logs: GitHub Actions runs
- Test source code: `test_*.cpp` files

---

**Generated**: 2025-12-31
**Branch**: `lock_ci_safety_net`
**Commits**: 3 (PHASE A+B, PHASE C, PHASE D)
