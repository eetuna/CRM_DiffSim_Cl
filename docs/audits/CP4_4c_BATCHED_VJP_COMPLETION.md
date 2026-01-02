# CP4.4c: Batched VJP for Faster Jacobians — COMPLETION AUDIT

**Status**: ✅ **COMPLETED**
**Date**: 2026-01-02
**Speedup**: **2.62×** vs CP4.4b per-vector approach (target: ≥1.3×)

---

## Executive Summary

CP4.4c implements **batched vector-Jacobian products (VJPs)** to accelerate Jacobian computation for iLQR/MPC. By processing K adjoint vectors simultaneously and reusing expensive equilibrium solves, we achieve a **2.62× speedup** over the CP4.4b per-vector approach, significantly exceeding the 1.3× target.

**Key Innovation**: Instead of calling `equilibrium_forward()` K×N times (once per adjoint × once per control input), the batched implementation calls it only N times and computes gradients for all K adjoints simultaneously.

---

## Implementation Overview

### What Was Built

1. **C++ Batched VJP Function** (`dynamics_backward_batched`)
   - Takes K adjoint vectors (K×6 matrix) instead of a single vector
   - Returns K gradient pairs: (K×6 for ∂L/∂x_t, K×3 for ∂L/∂u_t)
   - Reuses equilibrium solves across all K adjoints

2. **Python API Exposure**
   - New: `crm_diff_py.dynamics_linearize_batched(x_t, u_t, dt, L_inserted, params_dict, V)`
   - Existing `dynamics_linearize()` refactored to use batched API internally (transparent speedup)

3. **Correctness & Performance Tests**
   - `test_cp44c_batched_vjp_correctness.py`: Exact match validation (atol=1e-8)
   - `test_cp44c_batched_vjp_benchmark.py`: Performance comparison vs per-vector

4. **CTest Integration**
   - Correctness test: CI gate (30s timeout)
   - Benchmark test: Nightly only (180s timeout)

---

## Performance Results

### Benchmark Configuration
- **Hardware**: Linux 5.15.167.4-microsoft-standard-WSL2
- **Iterations**: 100 linearizations
- **Operating Point**: Typical moving state (x≠0, u≠0)
- **dt**: 0.01s, **L_inserted**: 50mm

### Results
```
Method                    Time/Linearization    Total (100 iters)
--------------------------------------------------------
Per-vector VJP (CP4.4b)   63.06 ms             6.306 s
Batched VJP (CP4.4c)      24.03 ms             2.403 s
--------------------------------------------------------
Speedup:                  2.62×
```

**Analysis**:
- Target: ≥1.3× speedup → **EXCEEDED** (2.62×)
- Speedup primarily from:
  1. Single batched solve of A^T Λ = V instead of 6 separate solves
  2. Reusing 3 equilibrium_forward calls across all 6 adjoints (was 6×3=18 calls before)
  3. More efficient matrix operations (BLAS-optimized Eigen batched multiplies)

---

## Correctness Validation

All tests **PASS** with exact numerical agreement (max diff: 0.00e+00):
```
✓ Rest (x=0, u=0)              W_x/W_u max diff: 0.00e+00
✓ Actuated (x=0, u≠0)          W_x/W_u max diff: 0.00e+00
✓ Moving (x≠0, u≠0)            W_x/W_u max diff: 0.00e+00
✓ Random 1-5 (seeded)          W_x/W_u max diff: 0.00e+00
```

**Tolerance**: atol=1e-8, rtol=1e-6 (exact match expected and achieved)

---

## Code Changes

### 1. C++ Implementation

**File**: `src/CRM_DiffDynamics.hpp`
```cpp
// CP4.4c: Batched VJP for faster Jacobian computation
int dynamics_backward_batched(
    const DynamicsStepResult& fwd_result,
    const double* V,                       // Adjoint matrix (K×6, row-major)
    int K,                                 // Number of adjoints
    const CRMForwardKinematicsData& params,
    double* W_x,                           // Output: (K×6, row-major)
    double* W_u,                           // Output: (K×3N, row-major)
    int* lu_rank = nullptr,
    double* rel_residual = nullptr
);
```

**File**: `src/CRM_DiffDynamics.cpp` (lines 333-443)
- Solves `A^T Λ = V` for all K adjoints at once (batched matrix solve)
- Computes `W_x = -Λ^T C` via batched matrix multiply
- Loops over control inputs once, computing gradient contributions for all K adjoints per iteration
- **Key optimization**: Each `equilibrium_forward()` call is reused across all K adjoints

### 2. Python Bindings

**File**: `python/crm_bindings.cpp`

**New API** (lines 614-700):
```python
result = crm_diff_py.dynamics_linearize_batched(
    x_t,          # (6,) state
    u_t,          # (3,) control
    dt,           # time step
    L_inserted,   # insertion length
    params_dict,  # CRM params
    V             # (K, 6) adjoint matrix
)
# Returns: {'W_x': (K,6), 'W_u': (K,3), 'status': int, ...}
```

**Refactored API** (lines 506-593):
```python
# dynamics_linearize() now uses batched VJP internally
result = crm_diff_py.dynamics_linearize(x_t, u_t, dt, L_inserted, params_dict)
# Returns: {'A': (6,6), 'B': (6,3), ...} (same interface, faster implementation)
```

### 3. Test Files

**Correctness**: `python/test_cp44c_batched_vjp_correctness.py`
- Compares batched VJP vs per-vector VJP (reference)
- 8 test cases (3 deterministic + 5 random)
- Runtime: ~1.4s

**Benchmark**: `python/test_cp44c_batched_vjp_benchmark.py`
- 100 iterations of per-vector vs batched VJP
- Reports speedup and absolute/relative timings
- Runtime: ~9s

### 4. Build Configuration

**File**: `CMakeLists.txt` (lines 441-458)
```cmake
# CP4.4c: Batched VJP correctness test
add_test(NAME batched_vjp_correctness_cp44c ...)
set_tests_properties(batched_vjp_correctness_cp44c PROPERTIES TIMEOUT 30)

# CP4.4c: Batched VJP performance benchmark (nightly only)
add_test(NAME batched_vjp_benchmark_cp44c ...)
set_tests_properties(batched_vjp_benchmark_cp44c PROPERTIES TIMEOUT 180)
set_tests_properties(batched_vjp_benchmark_cp44c PROPERTIES LABELS "nightly")
```

---

## How to Reproduce

### Build
```bash
cmake -S . -B build
cmake --build build -j$(nproc)
```

### Run Tests (CTest)
```bash
# Correctness only (fast, CI gate)
ctest --test-dir build -R batched_vjp_correctness_cp44c --output-on-failure

# Benchmark (nightly)
ctest --test-dir build -R batched_vjp_benchmark_cp44c --output-on-failure

# All CP4.4c tests
ctest --test-dir build -R cp44c --output-on-failure
```

### Run Tests (Python)
```bash
# Correctness
python3 python/test_cp44c_batched_vjp_correctness.py

# Benchmark
python3 python/test_cp44c_batched_vjp_benchmark.py
```

---

## API Documentation

### `dynamics_linearize_batched(x_t, u_t, dt, L_inserted, params_dict, V)`

**Purpose**: Compute batched VJPs for arbitrary adjoint vectors.

**Parameters**:
- `x_t`: np.array (6,) — Current state [u_0, v_0]
- `u_t`: np.array (3,) — Control input (Amperes)
- `dt`: float — Time step (seconds)
- `L_inserted`: float — Insertion length (mm)
- `params_dict`: dict — CRM parameters (CathParams, CathConfig, etc.)
- `V`: np.array (K, 6) — Adjoint vectors (row-major)

**Returns**: dict
```python
{
    'W_x': np.array (K, 6),       # ∂L/∂x_t for each adjoint
    'W_u': np.array (K, 3),       # ∂L/∂u_t for each adjoint
    'status': int,                # 0=success, >0=error
    'lu_rank': int,               # Rank of linear solve
    'rel_residual': float,        # Solve accuracy
    'api_version': str,
    'api_contract': str
}
```

**Example**:
```python
import numpy as np
import crm_diff_py

# Setup (load params, etc.)
x_t = np.zeros(6)
u_t = np.array([0.1, 0.05, -0.05])
dt = 0.01
L_inserted = 50.0

# Define K=3 adjoint vectors
V = np.array([
    [1, 0, 0, 0, 0, 0],  # e_1
    [0, 1, 0, 0, 0, 0],  # e_2
    [0, 0, 1, 0, 0, 0],  # e_3
])

result = crm_diff_py.dynamics_linearize_batched(
    x_t, u_t, dt, L_inserted, params_dict, V
)

# result['W_x'][0, :] is the gradient w.r.t. x_t for adjoint V[0, :]
# result['W_u'][0, :] is the gradient w.r.t. u_t for adjoint V[0, :]
```

### `dynamics_linearize(x_t, u_t, dt, L_inserted, params_dict)`

**Changes**: Now uses batched VJP internally (2.62× faster, same API).

**Returns**: dict (unchanged)
```python
{
    'A': np.array (6, 6),         # ∂x_next/∂x_t
    'B': np.array (6, 3),         # ∂x_next/∂u_t
    'status': int,
    'lu_rank': int,
    'rel_solve_residual': float,
    'solve_residual': float,
    'converged': int,
    'api_version': str,
    'api_contract': str
}
```

---

## Technical Details

### Batched VJP Algorithm

**Problem**: Compute K VJPs efficiently.
- Given: Forward cached result (A, B, C, x_next, u_t, etc.)
- Given: K adjoint vectors V (K×6 matrix)
- Compute: W_x (K×6), W_u (K×3)

**Per-Vector Approach (CP4.4b baseline)**:
```
for k = 1 to K:
    Solve: A^T λ_k = V[k, :]           # Linear solve (6×6)
    Compute: W_x[k, :] = -C^T λ_k
    for i = 1 to 3*NUM_ACT_SET:
        Perturb u_t[i], call equilibrium_forward()
        Compute dA/du_i, dB/du_i via FD
        W_u[k, i] = -(dr/du_i)^T λ_k
```
**Cost**: K solves + K×N equilibrium calls

**Batched Approach (CP4.4c)**:
```
Solve: A^T Λ = V^T                     # Batched solve (6×K via reused factorization)
Compute: W_x = -Λ^T C                  # Batched matrix multiply (K×6)
for i = 1 to 3*NUM_ACT_SET:
    Perturb u_t[i], call equilibrium_forward()  # ONCE
    Compute dA/du_i, dB/du_i via FD
    for k = 1 to K:
        W_u[k, i] = -(dr/du_i)^T Λ[:, k]        # Dot product
```
**Cost**: 1 batched solve + N equilibrium calls

**Speedup**: Reduces equilibrium calls from K×N to N (6× reduction for K=6).

### Memory Layout

All matrices use **row-major** storage for Python compatibility:
- `V`: (K, 6) row-major → V[k, j] = V[k*6 + j]
- `W_x`: (K, 6) row-major
- `W_u`: (K, 3*NUM_ACT_SET) row-major

Eigen maps handle layout conversions internally.

---

## Integration Impact

### Affected Components

1. **iLQR Controller** (`python/control/ilqr.py`)
   - Now 2.62× faster per Jacobian computation
   - No code changes required (transparent speedup via `dynamics_linearize()`)

2. **MPC Solvers** (future work)
   - Will benefit from same speedup
   - Can optionally use `dynamics_linearize_batched()` for custom adjoint vectors

3. **DAgger Training** (CP4.4a)
   - Faster dataset generation via faster MPC expert
   - No code changes required

### Backward Compatibility

✅ **FULL COMPATIBILITY**:
- Existing `dynamics_linearize()` API unchanged
- Performance improvement is transparent
- New `dynamics_linearize_batched()` is optional

---

## Future Work

### Potential Optimizations

1. **GPU Acceleration**: Batch matrix operations could be offloaded to GPU
2. **Analytical Jacobians**: Replace FD derivatives ∂K/∂u with analytical expressions
3. **Sparse Exploits**: Leverage block structure of A, B, C matrices
4. **Parallel Equilibrium**: Run equilibrium_forward calls in parallel threads

### Expected Impact

- GPU batching could yield another 2-5× speedup
- Analytical Jacobians could eliminate equilibrium_forward calls entirely (~10× total)

---

## Verification Summary

### Tests Passing
```
✓ batched_vjp_correctness_cp44c (1.42s, CI gate)
✓ batched_vjp_benchmark_cp44c   (9.00s, nightly)
✓ All existing CP4.4b tests     (unchanged)
```

### Numerical Accuracy
- Batched vs per-vector: **exact match** (0.00e+00 diff)
- No regression in physics/dynamics correctness

### Performance Validation
- Speedup: **2.62× measured** (target: ≥1.3×)
- Variance: <5% across runs (consistent)

---

## Sign-Off

**Implementation**: ✅ Complete
**Testing**: ✅ Complete
**Documentation**: ✅ Complete
**Performance Target**: ✅ Exceeded (2.62× vs 1.3× target)

**Ready for**: Production use, CP4.4d (if needed), or merge to main.

---

## Appendix: File Manifest

### Modified Files
- `src/CRM_DiffDynamics.hpp` (lines 69-84: new API declaration)
- `src/CRM_DiffDynamics.cpp` (lines 333-443: batched implementation)
- `python/crm_bindings.cpp` (lines 506-700: refactored + new binding)
- `CMakeLists.txt` (lines 441-458: test definitions)

### New Files
- `python/test_cp44c_batched_vjp_correctness.py`
- `python/test_cp44c_batched_vjp_benchmark.py`
- `docs/audits/CP4_4c_BATCHED_VJP_COMPLETION.md` (this file)

### Build Artifacts
- `build/crm_diff_py.cpython-310-x86_64-linux-gnu.so` (recompiled with batched VJP)

### Test Outputs (Reproducible)
```bash
# Correctness
$ python3 python/test_cp44c_batched_vjp_correctness.py
✓ CP4.4c BATCHED VJP CORRECTNESS TEST PASS

# Benchmark
$ python3 python/test_cp44c_batched_vjp_benchmark.py
✓ PASS: Batched VJP is 2.62x faster (target: ≥1.3x)
```

---

**END OF COMPLETION AUDIT**
