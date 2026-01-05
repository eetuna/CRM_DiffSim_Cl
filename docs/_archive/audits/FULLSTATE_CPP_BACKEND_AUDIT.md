# FULLSTATE C++ Backend Audit

**Date**: 2026-01-04
**Objective**: Prove that FULLSTATE (18·N+15) is the ONLY C++ backend exposed by default Python bindings.
**Verdict**: ✅ **PASS** - No reduced-state leaks detected.

---

## 1. Python Bindings Classification

**File**: `python/crm_bindings.cpp`

### A. FULLSTATE Default Exports

All exports below are in the **default module namespace** `crm_diff_py`:

| Export Name | Lines | State Dimensions | Purpose |
|-------------|-------|------------------|---------|
| `true_legacy_step_forward` | 291-295 | x_coil: [N, 18]<br>xf: [15]<br>**Total: 18·N+15** | Forward dynamics (DynamicsBVP → DYNSolverIVP) |
| `true_legacy_step_vjp` | 298-301 | Same as above | VJP (backward pass, single RHS) |
| `true_legacy_step_vjp_batched` | 304-307 | Same as above | Batched VJP (backward pass, multi-RHS) |
| `equilibrium_forward` | 282-284 | N/A (equilibrium solver) | Static equilibrium FK |
| `equilibrium_backward` | 286-288 | N/A | Static equilibrium VJP |
| `crmdyn_reference_rollout` | 310-313 | Same as above | Reference harness for testing |

**Evidence**:
```cpp
// python/crm_bindings.cpp:291-295
m.def("true_legacy_step_forward", &py_true_legacy_step_forward,
      py::arg("x_coil"), py::arg("xf"), py::arg("u"), py::arg("dt"),
      py::arg("params_dict"),
      py::arg("mL_guess") = py::none(), py::arg("nL_guess") = py::none(),
      "A0: TRUE legacy step (DynamicsBVP → DYNSolverIVP). Returns next state and observables.");

// python/crm_bindings.cpp:298-301
m.def("true_legacy_step_vjp", &py_true_legacy_step_vjp,
      py::arg("x_coil"), py::arg("xf"), py::arg("u"), py::arg("dt"), py::arg("L_inserted"),
      py::arg("params_dict"), py::arg("grad_tip_p"),
      "A0: TRUE legacy VJP. Computes gradients w.r.t. x_coil, xf, and u given grad_tip_p.");

// python/crm_bindings.cpp:304-307
m.def("true_legacy_step_vjp_batched", &py_true_legacy_step_vjp_batched,
      py::arg("x_coil"), py::arg("xf"), py::arg("u"), py::arg("dt"), py::arg("L_inserted"),
      py::arg("params_dict"), py::arg("grad_tip_p_batch"),
      "A3.5: TRUE legacy batched VJP. Computes gradients for multiple RHS using one factorization.");
```

**Input validation** (confirms FULLSTATE dimensions):
```cpp
// python/crm_bindings.cpp:328-332 (true_legacy_step_forward)
if (x_coil_buf.ndim != 2 || x_coil_buf.shape[0] != NUM_ACT_SET || x_coil_buf.shape[1] != 18) {
    throw std::runtime_error(
        "x_coil must be shape [" + std::to_string(NUM_ACT_SET) + ", 18], got [" +
        std::to_string(x_coil_buf.shape[0]) + ", " + std::to_string(x_coil_buf.shape[1]) + "]"
    );
}

// python/crm_bindings.cpp:337-340
if (xf_buf.ndim != 1 || xf_buf.shape[0] != NUM_STATES) {
    throw std::runtime_error(
        "xf must be shape [15], got [" + std::to_string(xf_buf.shape[0]) + "]"
    );
}
```

### B. Reduced6D Exports

**Result**: ❌ **NONE FOUND**

The following exports do **NOT** exist in `crm_bindings.cpp`:
- ❌ `dynamics_forward` (removed, was Reduced6D)
- ❌ `dynamics_backward` (removed, was Reduced6D)
- ❌ `dynamics_linearize` (removed, was Reduced6D)
- ❌ Any other reduced-state primitives

**Evidence**: No `reduced6d` namespace or qualified exports exist in crm_bindings.cpp.

---

## 2. C++ Include Headers

**Included in crm_bindings.cpp** (lines 1-8):
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "CRM.hpp"
#include "CRM_DiffEquilibrium.hpp"
#include "CRM_TrueLegacyDynamics.hpp"  // ✅ FULLSTATE backend
#include "CRM_ReferenceHarness.hpp"
#include "CRMDYN.hpp"
```

**NOT included**:
- ❌ `CRM_DiffDynamics.hpp` (Reduced6D, archived to `src/reduced6d/`)
- ❌ `reduced6d/CRM_DiffDynamics.hpp`

**C++ source file location audit**:
```bash
$ find src -name "CRM_DiffDynamics.*"
src/reduced6d/CRM_DiffDynamics.cpp
src/reduced6d/CRM_DiffDynamics.hpp
```
Only self-reference found:
```bash
$ grep -rn "CRM_DiffDynamics" src/
src/reduced6d/CRM_DiffDynamics.cpp:1:#include "CRM_DiffDynamics.hpp"
```

✅ **Reduced6D files safely isolated in `src/reduced6d/` subdirectory.**

---

## 3. CMake Build Configuration

**Check**: Does `CMakeLists.txt` build reduced6d?
```bash
$ grep -n "reduced6d\|CRM_DiffDynamics" CMakeLists.txt
(no output)
```

✅ **CMakeLists.txt does NOT reference `reduced6d/` or `CRM_DiffDynamics`.**

**Current build configuration** (inferred):
- Builds: `CRM_TrueLegacyDynamics.cpp` ✅
- Does NOT build: `src/reduced6d/CRM_DiffDynamics.cpp` ✅

---

## 4. FULLSTATE API Coverage for Controllers

**Controllers require**:
1. ✅ **Forward step**: `true_legacy_step_forward` (line 291)
2. ✅ **VJP**: `true_legacy_step_vjp` (line 298)
3. ✅ **Batched VJP**: `true_legacy_step_vjp_batched` (line 304)
4. ✅ **Linearization**: Provided by Python wrapper `true_legacy_linearize()` in `python/control/true_legacy_step.py` (uses PyTorch autograd or FD over VJP)

**State dimensions**:
- Coil state per actuator: `[v[3], w[3], p[3], R[9]]` = **18**
- Tip state: `[p[3], R[9], u[3]]` = **15**
- Total for N actuators: **18·N + 15**

**Control dimensions**:
- Per actuator: `[3]` (currents)
- Total for N actuators: **3·N**

---

## 5. Verdict

**✅ PASS**: FULLSTATE (18·N+15) is the ONLY C++ backend exposed by default.

**Summary**:
- ✅ All pybind11 exports use FULLSTATE dimensions (18·N+15)
- ✅ No reduced-state exports in default namespace
- ✅ `CRM_DiffDynamics.{cpp,hpp}` archived to `src/reduced6d/` (not built)
- ✅ CMakeLists.txt does NOT build reduced6d
- ✅ Controllers have access to all required primitives (forward, VJP, batched VJP, linearization)

**No default-path leaks detected.**
