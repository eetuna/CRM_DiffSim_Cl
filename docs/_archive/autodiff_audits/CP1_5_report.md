# CP1.5 STATUS REPORT: FAIL

**Status:** FAIL
**Date:** 2025-12-30
**Test:** PyTorch autograd integration with implicit VJP
**Failure Class:** Parameter marshalling / memory layout / lifetime issue in C++↔Python boundary

---

## Executive Summary

CP1.5 test suite **FAILS** on all 4 operating points (zero, small, moderate, large currents).

**Root Cause:** Jacobian matrices extracted from C++ equilibrium solver are **degenerate** when accessed from Python:
- All rows of each Jacobian matrix are **identical** (rank-1 matrices instead of full-rank)
- K_tip matrix has **non-diagonal structure** (all elements identical) instead of expected diagonal form
- Matrix `(K_tip * J_u_u0)^T` has rank 1 instead of rank 3, causing FullPivLU solver to fail

**Critical Finding:** CP1.4 backward logic (`equilibrium_backward()`) is mathematically correct. The failure occurs **upstream** in the forward pass Jacobian extraction/storage/marshalling.

---

## Modified/Created Files

### Source Files Created:
```
src/CRM_DiffEquilibrium.hpp          # C++ header for differentiable equilibrium primitive
src/CRM_DiffEquilibrium.cpp          # Forward/backward implementations
python/crm_bindings.cpp              # pybind11 bindings for Python integration
python/crm_equilibrium.py            # PyTorch autograd.Function wrapper
python/test_cp15.py                  # CP1.5 test suite (gradcheck + finite difference)
python/test_debug.py                 # Debug script to inspect Jacobian matrices
test_cp12.cpp                        # CP1.2: equilibrium_forward() wrapper test
test_cp13.cpp                        # CP1.3: Jacobian caching test
```

### Source Files Modified:
```
CMakeLists.txt                       # Added CRM_DiffEquilibrium to build, test targets
src/CRM_IVPJacobian.cpp              # (modified working tree, uncommitted changes)
```

### Build Artifacts:
```
build/test_cp12                      # CP1.2 executable (exists)
build/test_cp13                      # CP1.3 executable (exists)
build/libCRMCPPLib.a                 # Updated static library
build/crm_diff_py.*.so               # Python extension module (pybind11)
```

---

## Build Commands

### C++ Tests:
```bash
cd /workspaces/CRM_DiffSim_Cl
cmake -S . -B build
cmake --build build --target test_cp12
cmake --build build --target test_cp13

# Run tests (requires catheterdata/ in parent directory)
cd /workspaces/CRM_DiffSim_Cl
./build/test_cp12
./build/test_cp13
```

### Python Extension:
```bash
cd /workspaces/CRM_DiffSim_Cl/python
pip3 install -e .  # Builds crm_diff_py extension via setup.py
```

### CP1.5 Test:
```bash
cd /workspaces/CRM_DiffSim_Cl/python
python3 test_cp15.py
python3 test_debug.py  # Debug Jacobian matrices
```

---

## Exact Observed Failures

### Test Run Output (python/test_cp15.py):

```
CP1.5 Test: PyTorch gradcheck + finite difference validation
============================================================

============================================================
Testing: Zero currents
============================================================
u = [0. 0. 0.]
L_inserted = 50.0
p_tip = [0.94797527 0.94797527 0.94797527]

[1/2] Running torch.autograd.gradcheck...
  ✗ gradcheck FAIL (exception: equilibrium_backward failed with status 1 (rank=1, residual=3.95e-322) at batch 0)

[2/2] Running finite difference check...

✗ Zero currents: EXCEPTION - equilibrium_backward failed with status 1 (rank=1, residual=3.95e-322) at batch 0
Traceback (most recent call last):
  File "/workspaces/CRM_DiffSim_Cl/python/test_cp15.py", line 40, in finite_difference_check
    grad_u = torch.autograd.grad(p_tip, u_test, grad_out, retain_graph=True)[0]
  File "/home/vscode/.local/lib/python3.10/site-packages/torch/autograd/function.py", line 315, in apply
    return user_fn(self, *args)
  File "/workspaces/CRM_DiffSim_Cl/python/crm_equilibrium.py", line 110, in backward
    raise RuntimeError(
RuntimeError: equilibrium_backward failed with status 1 (rank=1, residual=3.95e-322) at batch 0

============================================================
CP1.5 TEST SUMMARY
============================================================
  ✗ Zero currents: FAIL
  ✗ Small currents: FAIL
  ✗ Moderate currents: FAIL
  ✗ Large currents: FAIL
============================================================
CP1.5: FAIL - Some tests failed
============================================================
```

**All 4 test cases fail identically** with:
- `status=1` (rank-deficient matrix in equilibrium_backward)
- `rank=1` (matrix `(K_tip * J_u_u0)^T` has rank 1, not 3)

---

## Side-by-Side C++ vs Python Behavior

### C++ Test (test_cp13):
```bash
$ cd /workspaces/CRM_DiffSim_Cl
$ ./build/test_cp13
terminate called after throwing an instance of 'std::runtime_error'
  what():  Unable to open Catheter Model Parameters file!
Aborted (core dumped)
```
**Issue:** Test harness not configured correctly (catheter data path issue). Cannot verify C++ behavior directly.

### Python Test (test_debug.py):
```bash
$ cd /workspaces/CRM_DiffSim_Cl/python
$ python3 test_debug.py

Forward result:
  status: 0
  p_tip: [0.94797527 0.94797527 0.94797527]
  deltau0: [-1.45028905e-05 -1.45028905e-05 -1.45028905e-05]

Cached Jacobians:
  J_p_u0 shape: (9,)
  J_p_u0:
[[-0.01666324 -0.01666324 -0.01666324]
 [-0.01666324 -0.01666324 -0.01666324]  ← ALL ROWS IDENTICAL
 [-0.01666324 -0.01666324 -0.01666324]]

  J_u_u0 shape: (9,)
  J_u_u0:
[[0.98620248 0.98620248 0.98620248]
 [0.98620248 0.98620248 0.98620248]      ← ALL ROWS IDENTICAL
 [0.98620248 0.98620248 0.98620248]]

  K_tip shape: (9,)
  K_tip:
[[22.83041619 22.83041619 22.83041619]
 [22.83041619 22.83041619 22.83041619]   ← NOT DIAGONAL! All rows identical
 [22.83041619 22.83041619 22.83041619]]

  J_p_zc shape: (9,)
  J_p_zc:
[[-2.05035842 -2.05035842 -2.05035842]
 [-2.05035842 -2.05035842 -2.05035842]   ← ALL ROWS IDENTICAL
 [-2.05035842 -2.05035842 -2.05035842]]

  J_u_zc shape: (9,)
  J_u_zc:
[[0.03699282 0.03699282 0.03699282]
 [0.03699282 0.03699282 0.03699282]      ← ALL ROWS IDENTICAL
 [0.03699282 0.03699282 0.03699282]]

  K_tip * J_u_u0:
[[67.5462395 67.5462395 67.5462395]
 [67.5462395 67.5462395 67.5462395]      ← Degenerate product
 [67.5462395 67.5462395 67.5462395]]

  rank(K_tip * J_u_u0): 1                ← SHOULD BE 3
  rank((K_tip * J_u_u0)^T): 1

  Singular values of (K_tip * J_u_u0)^T: [2.02638719e+02 1.64e-14 0.00000000e+00]
                                          ^^^^^^^^^^ Only one non-zero singular value
```

**Evidence of Degeneracy:**
1. **J_p_u0**: All rows identical → rank-1 matrix (should be full-rank 3×3)
2. **J_u_u0**: All rows identical → rank-1 matrix (should be full-rank 3×3)
3. **K_tip**: All rows identical, **NOT diagonal** (expected: diagonal matrix with different stiffness values)
4. **J_p_zc**: All rows identical → rank-1 matrix (should be full-rank 3×3)
5. **J_u_zc**: All rows identical → rank-1 matrix (should be full-rank 3×3)

---

## Evidence of Degenerate Jacobians

### Expected Behavior:
- **K_tip**: 3×3 diagonal matrix (bending/torsional stiffnesses)
  ```
  [k_bend_x      0           0      ]
  [0             k_bend_y    0      ]
  [0             0           k_tors ]
  ```
- **J_p_u0, J_u_u0**: Full-rank 3×3 matrices (independent rows)

### Observed Behavior in Python:
- **K_tip**: All 9 elements equal `22.83041619` (rank-1 matrix)
- **J_p_u0**: All rows `[-0.01666324, -0.01666324, -0.01666324]` (rank-1 matrix)
- **J_u_u0**: All rows `[0.98620248, 0.98620248, 0.98620248]` (rank-1 matrix)

### Impact on Backward Pass:
```cpp
// equilibrium_backward() in CRM_DiffEquilibrium.cpp:163
FullPivLU<Matrix3d> lu(K_J_u.transpose());
int rank = lu.rank();
if (rank < 3) {
    return 1;  // rank-deficient  ← TRIGGERS HERE
}
```

The matrix `(K_tip * J_u_u0)^T` has only **1 non-zero singular value**, making it rank-deficient. The FullPivLU solver correctly detects this and returns status=1, causing the entire backward pass to fail.

---

## Most Likely Failure Class

### Hypothesis: Memory Layout / Stride Mismatch in Row-Major ↔ Column-Major Conversion

**Symptom Pattern:**
- Each 3×3 matrix becomes rank-1 with all rows identical
- This suggests **data is being read with incorrect strides**

**Suspected Code Path:**

1. **C++ Storage** (src/CRM_DiffEquilibrium.cpp:97-113):
   ```cpp
   // Extract Jacobian blocks (row-major layout)
   for (int i = 0; i < 9; i++) {
       out.J_p_u0[i] = x_N._p_u0[i];  // Copy from AugmentedStateVector
       out.J_u_u0[i] = x_N._u_u0[i];
   }
   ```
   - Arrays `x_N._p_u0`, `x_N._u_u0` are **Eigen::Matrix row-major** (defined in state vector)
   - Copying via linear index `[i]` assumes **contiguous row-major layout**

2. **Python Marshalling** (python/crm_bindings.cpp:76-80):
   ```cpp
   out["J_p_u0"] = py::array_t<double>({9}, result.J_p_u0);
   out["J_u_u0"] = py::array_t<double>({9}, result.J_u_u0);
   out["K_tip"] = py::array_t<double>({9}, result.K_tip);
   ```
   - Creates **1D NumPy array** of shape `(9,)` from C++ array
   - No explicit layout/stride specification

3. **Python Reconstruction** (python/test_debug.py:32-33):
   ```python
   J_p_u0 = result['J_p_u0'].reshape(3, 3)  # Reshape 1D → 2D
   ```
   - NumPy `.reshape()` uses **C-contiguous (row-major)** by default
   - **IF** source data has wrong layout, reshape will produce garbage

**Potential Root Causes:**

### A. Eigen::Map Stride Mismatch in AugmentedStateVector
   - `x_N._p_u0` may be a **non-contiguous view** with stride != 1
   - Linear copy `out.J_p_u0[i] = x_N._p_u0[i]` samples wrong elements
   - **Example:** If `_p_u0` has column-major strides, copying as row-major reads:
     ```
     Element [0,0] → copied 9 times to out.J_p_u0[0..8]
     Elements [0,1], [0,2], [1,0], ... → never copied
     ```

### B. Incorrect K_tip Extraction
   ```cpp
   // src/CRM_DiffEquilibrium.cpp:116-124
   if (BVPParams.SegmentTypes[BVPParams.no_segments - 1] == FLEXIBLE) {
       for (int i = 0; i < 9; i++) {
           out.K_tip[i] = BVPParams.K[BVPParams.no_flex_seg - 1][i];
       }
   }
   ```
   - `BVPParams.K` is a **multi-dimensional array** (per-segment stiffness matrices)
   - Linear index `[i]` may not correspond to matrix elements
   - K_tip should be **diagonal**, but all elements are identical → suggests sampling from wrong memory location

### C. Pointer Aliasing / Lifetime Issue
   - `x_N._p_u0`, `x_N._u_u0` may be **temporary Eigen::Map objects**
   - Copying pointers instead of data → Python sees deallocated/reused memory

---

## Diagnostic Evidence Supporting Memory Layout Hypothesis

1. **Consistent Pattern Across All Matrices:**
   Every Jacobian matrix shows the **same degeneracy** (all rows identical). This is **not** a mathematical error in the solver, but a systematic data extraction bug.

2. **K_tip Non-Diagonal Structure:**
   K_tip should be diagonal by construction (stiffness matrix). The fact that all elements are identical (`22.83041619`) suggests the code is reading **one element 9 times** instead of reading 9 distinct elements.

3. **Forward Pass Succeeds:**
   `equilibrium_forward()` returns `status=0`, and `p_tip` values appear reasonable. This confirms the **physics solver works**, but Jacobian extraction is broken.

4. **Singular Value Spectrum:**
   ```
   Singular values: [202.6, 1.64e-14, 0.0]
   ```
   Only **one** non-zero singular value → matrix is **exactly** rank-1, not approximately rank-deficient. This rules out numerical conditioning issues and points to structural corruption.

---

## Why CP1.4 Logic is NOT the Cause

### CP1.4 Implementation (`equilibrium_backward()`)
The backward pass logic in `src/CRM_DiffEquilibrium.cpp:136-195` is mathematically correct:

```cpp
// Step 1: Compute K_tip * J_u_u0
Matrix3d K_J_u = K_tip_map * J_u_u0_map;

// Step 2: Compute RHS = J_p_u0^T * grad_p_tip
Vector3d rhs = J_p_u0_map.transpose() * grad_p_tip_map;

// Step 3: Solve (K_tip * J_u_u0)^T * λ = rhs
FullPivLU<Matrix3d> lu(K_J_u.transpose());
int rank = lu.rank();
if (rank < 3) return 1;  // rank-deficient

// Step 4: Relative residual check
Vector3d lambda = lu.solve(rhs);
Vector3d residual_vec = K_J_u.transpose() * lambda - rhs;
double rel_res = residual_norm / denom;
if (rel_res > 1e-10) return 2;

// Step 5: Compute gradient w.r.t. currents
grad_u_map = J_p_zc_map.transpose() * grad_p_tip_map;
grad_u_map -= J_u_zc_map.transpose() * K_lambda;
```

**This is a textbook implicit differentiation implementation:**
1. Solve adjoint equation: `(∂h/∂x)^T λ = (∂f/∂x)^T v` where `h(x,u)=0` is the constraint
2. Compute gradient: `∂f/∂u - (∂h/∂u)^T λ`

**The code correctly:**
- Uses FullPivLU for robust rank detection
- Checks both rank and residual before trusting the solution
- Returns diagnostic status codes (1=rank-deficient, 2=residual too large)

**The failure occurs because the INPUT data is corrupt:**
- If `K_tip`, `J_u_u0` are rank-1 matrices, then `K_J_u` is also rank-1
- The CP1.4 code correctly detects this and returns `status=1`
- **The bug is in CP1.5's forward pass**, not CP1.4's backward logic

---

## Recommended Next Steps (Not Implemented)

1. **Verify AugmentedStateVector Layout:**
   - Inspect `x_N._p_u0.innerStride()` and `x_N._p_u0.outerStride()`
   - Check if Eigen::Map is contiguous (stride=1) or strided

2. **Add Debug Logging to equilibrium_forward():**
   ```cpp
   std::cout << "x_N._p_u0:\n" << x_N._p_u0 << std::endl;
   std::cout << "Copied J_p_u0:\n";
   for (int i = 0; i < 3; i++) {
       for (int j = 0; j < 3; j++) {
           std::cout << out.J_p_u0[i*3 + j] << " ";
       }
       std::cout << std::endl;
   }
   ```

3. **Test Direct Eigen::Map in pybind11:**
   ```cpp
   // Instead of:
   out["J_p_u0"] = py::array_t<double>({9}, result.J_p_u0);

   // Try:
   py::array_t<double> J_p_u0_arr({3, 3});
   Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>>(
       J_p_u0_arr.mutable_data()
   ) = x_N._p_u0;  // Direct Eigen assignment
   out["J_p_u0"] = J_p_u0_arr;
   ```

4. **Validate K_tip Extraction:**
   - Print `BVPParams.K[no_flex_seg-1]` before copying
   - Verify it's diagonal
   - Check if array indexing `K[seg][i]` is row-major or column-major

---

## Conclusion

**CP1.5: FAIL**

All 4 test cases fail due to **degenerate Jacobian matrices** extracted from the C++ equilibrium solver. The matrices have rank 1 (all rows identical) instead of full rank, causing the implicit differentiation backward pass to fail with rank-deficient system detection.

**Root Cause Classification:** Parameter marshalling / memory layout / lifetime issue at the C++↔Python boundary, most likely in:
- Eigen::Map stride handling in `AugmentedStateVector` extraction
- K_tip array indexing (reading one element 9 times)
- Potential pointer aliasing in pybind11 memory lifetime

**CP1.4 Logic:** The `equilibrium_backward()` implementation is mathematically correct and properly detects the rank-deficiency in its input data. The bug is **upstream** in the forward pass Jacobian extraction.

**No fixes have been implemented per user directive.**

---

## Appendix: File Locations

### Test Scripts:
- `python/test_cp15.py` - Main CP1.5 test suite
- `python/test_debug.py` - Jacobian inspection script
- `test_cp12.cpp` - CP1.2 C++ test (equilibrium_forward wrapper)
- `test_cp13.cpp` - CP1.3 C++ test (Jacobian caching)

### Implementation:
- `src/CRM_DiffEquilibrium.hpp` - C++ header
- `src/CRM_DiffEquilibrium.cpp` - C++ forward/backward implementation
- `python/crm_bindings.cpp` - pybind11 bindings
- `python/crm_equilibrium.py` - PyTorch autograd wrapper

### Build System:
- `CMakeLists.txt` - Updated with new targets
- `build/test_cp12`, `build/test_cp13` - Compiled executables
- `build/crm_diff_py.*.so` - Python extension module

---

**Report End**
