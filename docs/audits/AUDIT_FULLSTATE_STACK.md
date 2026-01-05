# FULLSTATE Stack Audit - End-to-End Verification

**Audit Date**: 2026-01-04
**Branch**: `true-legacy-dynamics-migration`
**Commit**: `aeb2c7ebd85389fed5c4c70d3c56ac4930109a2b`
**Auditor**: Automated (Claude Code)
**Scope**: AUDIT ONLY (no code modifications)

---

## Executive Summary

✅ **FULLSTATE COMPLIANCE: VERIFIED**

The repository implements TRUE legacy FULLSTATE dynamics (18·N + 15 hybrid state) as the **default and only** controller stack. All required properties are satisfied:

| Property | Status | Evidence |
|----------|--------|----------|
| **FULLSTATE Contract (18·N+15)** | ✅ PASS | python/crm_bindings.cpp:343-358, src/CRM.hpp:13-14 |
| **Entrypoints Correct** | ✅ PASS | python/crm_bindings.cpp:301-324 |
| **Differentiation Compliance** | ✅ PASS | src/CRM_TrueLegacyDynamics.cpp:160-276 |
| **Reduced6D Isolation** | ✅ PASS | Zero default imports (grep verified) |
| **Test Coverage** | ⚠️ PARTIAL | 6 FULLSTATE tests exist, missing VJP state gradcheck |

---

## A) Repo Provenance & Entrypoints

### A.1 Repository Status

```bash
Branch: true-legacy-dynamics-migration
Commit: aeb2c7ebd85389fed5c4c70d3c56ac4930109a2b
Status: Modified files present (build artifacts, archive cleanup)
```

**Recent commits**:
- `aeb2c7e` - cleaned repo before migration
- `f613076` - part completion of audits and archive of old 6d state
- `6d86781` - build cleaned
- `6ec4939` - pre B0 audit completed, archiving 6D old/simplified state

### A.2 TRUE Legacy Runtime Entrypoints

**Python Module**: `crm_diff_py` (pybind11)
**Location**: python/crm_bindings.cpp:259

#### Python Public API

| Function | Binding Location | C++ Backend |
|----------|-----------------|-------------|
| `true_legacy_step_forward` | python/crm_bindings.cpp:301-305 | python/crm_bindings.cpp:334-588 |
| `true_legacy_step_vjp` | python/crm_bindings.cpp:308-311 | python/crm_bindings.cpp:591-705 |
| `true_legacy_step_vjp_batched` | python/crm_bindings.cpp:314-317 | python/crm_bindings.cpp:708-818 |
| `true_legacy_linearize` | python/crm_bindings.cpp:320-324 | python/crm_bindings.cpp:821-949 |

**Evidence - Function Signature (python/crm_bindings.cpp:334-342)**:
```cpp
py::dict py_true_legacy_step_forward(
    py::array_t<double> x_coil_arr,     // [NUM_ACT_SET, 18]
    py::array_t<double> xf_arr,         // [15]
    py::array_t<double> u_arr,          // [NUM_ACT_SET, 3]
    double dt,
    py::dict params_dict,
    py::object mL_guess_obj = py::none(),
    py::object nL_guess_obj = py::none()
);
```

#### C++ Backend Chain

**Forward Path**:
1. `py_true_legacy_step_forward` (Python binding wrapper)
2. → `DynamicsBVP()` (BVP solver for interface forces/moments)
   Location: src/CRM_TrueLegacyDynamics.cpp:105-106
3. → `DYNSolverIVP()` (IVP integrator for state propagation)
   Location: src/CRM_TrueLegacyDynamics.cpp:112-113

**Backward Path**:
1. `py_true_legacy_step_vjp` / `py_true_legacy_step_vjp_batched`
2. → `true_legacy_step_backward` / `true_legacy_step_backward_batched`
   Location: src/CRM_TrueLegacyDynamics.cpp:130-348
3. → `CRMSolverIVPJacobian()` (analytic IVP Jacobians)
   Location: src/CRM_TrueLegacyDynamics.cpp:222-224
4. → `compute_bvp_jacobians_full_analytic()` (analytic BVP Jacobians)
   Location: src/CRM_TrueLegacyDynamics.cpp:269-274

### A.3 Controller Integration

**Python Wrapper**: python/control/true_legacy_step.py:23-267
**State Adapter**: python/control/true_legacy_state_adapter.py:1-366

**Controllers Using FULLSTATE**:
- `iLQRSolver`: python/control/ilqr.py:19 (`import crm_diff_py`)
- `MPCController`: python/control/mpc.py:17 (`import crm_diff_py`)
- `finite_horizon_lqr`: python/control/lqr.py:19 (`import crm_diff_py`)
- `HybridController`: python/control/hybrid_controller.py:37 (`import crm_diff_py`)

**Evidence - Module Exports (python/control/__init__.py:62)**:
```python
# NO imports from reduced6d, NO backward aliases
```

---

## B) FULLSTATE Contract Enforcement

### B.1 Authoritative Contract Document

**Location**: docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md
**Date**: 2026-01-03
**Status**: FROZEN — Ground Truth

### B.2 State Layout

#### C++ Constants (src/CRM.hpp:13-14, src/CRMDYN.hpp:27)

```cpp
#define NUM_STATES 15          // Tip state: p[3], R[9], u[3]
#define NUM_ACT_SET 1          // Number of actuator sets (coils)
#define NUM_COIL_STATES 18     // Coil state: v[3], w[3], p[3], R[9]
```

#### FULLSTATE Dimension Formula

**State Dimension** = `18 * NUM_ACT_SET + 15`

For `NUM_ACT_SET = 1`: **33 dimensions**

**Packing Order** (python/control/true_legacy_state_adapter.py:50-151):
1. **Coil states** (per coil j=0..N-1): `[v[3], w[3], p[3], R[9]]` → 18 elements
2. **Tip state**: `[p_tip[3], R_tip[9], u_tip[3]]` → 15 elements

#### State Component Breakdown

| Component | Indices | Dimension | Frame | Description |
|-----------|---------|-----------|-------|-------------|
| Coil j velocity | `j*18 + 0:3` | 3 | Body | Linear velocity |
| Coil j ang. velocity | `j*18 + 3:6` | 3 | Body | Angular velocity |
| Coil j position | `j*18 + 6:9` | 3 | Spatial | Position |
| Coil j orientation | `j*18 + 9:18` | 9 | Spatial | Rotation matrix (row-major) |
| Tip position | `18*N + 0:3` | 3 | Spatial | Tip position |
| Tip orientation | `18*N + 3:12` | 9 | Spatial | Tip rotation matrix |
| Tip curvature | `18*N + 12:15` | 3 | Body | Tip curvature |

### B.3 State Dimension Enforcement

**Python Validation** (python/crm_bindings.cpp:343-358):
```cpp
// Validate x_coil: should be [N_ACT, 18]
auto x_coil_buf = x_coil_arr.request();
if (x_coil_buf.ndim != 2 || x_coil_buf.shape[0] != NUM_ACT_SET || x_coil_buf.shape[1] != 18) {
    throw std::runtime_error(...);
}

// Validate xf: should be [15]
auto xf_buf = xf_arr.request();
if (xf_buf.ndim != 1 || xf_buf.shape[0] != NUM_STATES) {
    throw std::runtime_error("xf must be shape [15]");
}
```

**Python Wrapper** (python/control/true_legacy_step.py:80-100):
```python
expected_state_dim = true_legacy_state_dim(n_act)  # 18*n_act + 15
if x.shape[1] != expected_state_dim:
    raise ValueError(...)
```

**C++ Implementation** (src/CRM_TrueLegacyDynamics.hpp:12-13):
```cpp
double xf_next[NUM_STATES];              // Tip state: [p[3], R[9], u[3]]
double x_coil_next[NUM_ACT_SET][18];     // Coil states: [v[3], w[3], p[3], R[9]]
```

### B.4 Control Layout

**Control Dimension** = `3 * NUM_ACT_SET`

For `NUM_ACT_SET = 1`: **3 dimensions** (magnetic field components)

**Evidence** (python/crm_bindings.cpp:360-367):
```cpp
// Validate u: should be [N_ACT, 3]
auto u_buf = u_arr.request();
if (u_buf.ndim != 2 || u_buf.shape[0] != NUM_ACT_SET || u_buf.shape[1] != 3) {
    throw std::runtime_error(...);
}
```

### B.5 BVP Unknowns Are NOT State

**Critical Distinction** (docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md:21):
> BVP unknowns (mL, nL) are **NOT state** — they are solved fresh each step.

**Evidence** (src/CRM_TrueLegacyDynamics.cpp:104-108):
```cpp
// Call DynamicsBVP
DynamicsBVP(shooting_params, xf, mL_guess_arr, nL_guess_arr, ftip_guess,
            out.u0, out.mL, out.nL, out.tau, out.ftip, out.localmin);

out.converged = (out.localmin == 0) ? 1 : 0;
```

The BVP unknowns `(mL, nL)` are:
- **Inputs**: Warm-start guesses (optional, 6*N dimensions)
- **Outputs**: Solved values (returned for next step warm-start)
- **NOT persisted in state vector**

---

## C) Differentiation Compliance

### C.1 No Finite Differences

**Verification Command**:
```bash
grep -i "finite.*diff\|central.*diff\|forward.*diff\|eps\s*=\|perturbation" \
  src/CRM_TrueLegacyDynamics.cpp
```

**Result**: **Zero matches** ✅

### C.2 Analytic Jacobians Only

**IVP Jacobian** (src/CRM_TrueLegacyDynamics.cpp:214-224):
```cpp
// Call IVP Jacobian (analytic)
auto [J_u, J_n, J_p, J_R, J_ftip] = CRMSolverIVPJacobian(
    shooting_params, deltau0, ftip_copy, true, x_N, MomentResidual
);
```

**BVP Jacobian** (src/CRM_TrueLegacyDynamics.cpp:266-274):
```cpp
// Step 5: Compute BVP Jacobian blocks J_yy, J_yu, and J_yx
//         using strictly analytic methods (NO FD)
Eigen::MatrixXd J_yy, J_yu, J_yx;

compute_bvp_jacobians_full_analytic(
    fwd_result.mL, fwd_result.nL, fwd_result.u,
    params, fwd_result.xf, L_inserted, fwd_result.dt,
    fwd_result.x_coil,
    J_yy, J_yu, J_yx
);
```

### C.3 Implicit Function Theorem / Adjoint Method

**Adjoint System** (src/CRM_TrueLegacyDynamics.cpp:276-286):
```cpp
// Step 6: Solve adjoint system: (J_yy)^T * lambda = v_y
Eigen::VectorXd lambda;
int rank_used = dim_y;
double residual_norm = 0.0;

// Use QR decomposition for stable solve
Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr_solver(J_yy.transpose());
lambda = qr_solver.solve(v_y);
```

**IFT Application** (src/CRM_TrueLegacyDynamics.cpp:288-337):
```cpp
// Step 7: Compute gradients using implicit function theorem

// 7c. Add implicit terms: grad_x -= (J_yx)^T * lambda
Eigen::VectorXd implicit_grad_x = J_yx.transpose() * lambda;

// 7d. Gradients for u (actuation currents) - BVP adjoint
// grad_u = -(J_yu)^T * lambda
Eigen::VectorXd grad_u_vec = -J_yu.transpose() * lambda;
```

### C.4 No Differentiation Through Solver Iterations

**Key Principle**: Jacobians are computed **at the converged solution** only, using cached forward pass results.

**Evidence** (src/CRM_TrueLegacyDynamics.cpp:130-151):
```cpp
int true_legacy_step_backward(
    const TrueLegacyStepResult& fwd_result,  // ← Cached forward result
    const double grad_tip_p[3],
    ...
) {
    // Step 1: Cotangent on tip_p -> xf_next
    // Step 2: Compute IVP Jacobians analytically using cached solution
    // Step 3-7: Apply IFT using converged BVP unknowns
```

The BVP solver iterations (trust-region dogleg) are **NOT differentiated**. Only the final converged solution is used to compute Jacobians via IFT.

---

## D) Reduced6D Isolation

### D.1 Removed APIs

The following APIs **do not exist** in default bindings:

| Removed API | Status |
|-------------|--------|
| `crm_diff_py.dynamics_forward` | ❌ NOT BOUND |
| `crm_diff_py.dynamics_backward` | ❌ NOT BOUND |
| `crm_diff_py.dynamics_linearize` | ❌ NOT BOUND |
| `crm_diff_py.dynamics_linearize_batched` | ❌ NOT BOUND |

**Evidence** (grep command):
```bash
grep -r "crm_diff_py.dynamics_" python/control src --include="*.py" --include="*.cpp" \
  | grep -v "reduced6d" | grep -v "#"
```

**Result**: **Zero matches** ✅

### D.2 Archive Locations

**C++ Archive**: `src/reduced6d/`
- `CRM_DiffDynamics.cpp`
- `CRM_DiffDynamics.hpp`

**Python Archive**: `python/control/reduced6d/`
- `ilqr_6d.py`
- `mpc_6d.py`
- `lqr_6d.py`
- `hybrid_controller_6d.py`
- `hybrid_state_contract_9d.py`
- `step_hybrid_legacy_contract_9d.py`

### D.3 Default Imports Verification

**Evidence** (python/control/__init__.py:1-63):
- Lines 13-16: Import FULLSTATE controllers (ilqr, mpc, lqr, hybrid_controller)
- Lines 19-33: Import TRUE legacy adapters and step functions
- Line 62: `# NO imports from reduced6d, NO backward aliases`

**No reduced6d imports in default paths** ✅

### D.4 Sanity Gate Test

**Location**: tests/test_sanity_gate_no_reduced6d.py:11-51

**Purpose**: Enforce zero `crm_diff_py.dynamics_*` references in default code paths

**Coverage**:
- Searches `python/control` and `src` directories
- Excludes `reduced6d/` subdirectories
- Filters out comments and docstrings
- Fails build if violations found

---

## E) Test Coverage Matrix

### E.1 FULLSTATE Tests (tests/)

| Test | Type | Purpose | Status |
|------|------|---------|--------|
| `test_fullstate_step_smoke.py` | Smoke | Forward step correctness | ✅ EXISTS |
| `test_fullstate_linearize_shapes.py` | Unit | A, B matrix shapes | ✅ EXISTS |
| `test_fullstate_vjp_gradcheck_u.py` | Gradcheck | VJP vs FD for ∂u | ✅ EXISTS |
| `test_controller_smoke_fullstate.py` | Integration | iLQR/MPC/LQR smoke | ✅ EXISTS |
| `test_sanity_gate_no_reduced6d.py` | Policy | No reduced6d imports | ✅ EXISTS |
| `test_fullstate_quick_smoke.py` | Smoke | Quick sanity check | ✅ EXISTS |

### E.2 How to Run Tests

```bash
# Single test
PYTHONPATH=build:python:$PYTHONPATH python3 tests/test_fullstate_step_smoke.py

# All FULLSTATE tests
PYTHONPATH=build:python:$PYTHONPATH python3 -m pytest tests/test_fullstate_*.py -v

# Sanity gate
PYTHONPATH=build:python:$PYTHONPATH python3 tests/test_sanity_gate_no_reduced6d.py
```

### E.3 Missing Tests (Identified Gaps)

| Gap | Priority | Description |
|-----|----------|-------------|
| VJP gradcheck for state (∂x_coil, ∂xf) | HIGH | Currently only tests ∂u |
| Linearization numerical accuracy | MEDIUM | Verify A, B match finite differences |
| Batched VJP correctness | MEDIUM | Verify batched == looped single VJP |
| Convergence failure handling | LOW | Test non-converged BVP behavior |

---

## F) Differentiation Stack Verification

### F.1 Forward Pass Chain

```
Python: true_legacy_step(x, u, dt, ...)
  ↓
Python Binding: py_true_legacy_step_forward (crm_bindings.cpp:334)
  ↓
C++ Wrapper: true_legacy_step_forward (CRM_TrueLegacyDynamics.cpp:13)
  ↓
BVP Solver: DynamicsBVP(...) → solves for (mL, nL, u0, ftip)
  ↓
IVP Integrator: DYNSolverIVP(...) → propagates state forward
  ↓
Returns: x_coil_next, xf_next, observables
```

**Evidence**: src/CRM_TrueLegacyDynamics.cpp:105-113

### F.2 Backward Pass Chain (VJP)

```
Python: grad_x, grad_u = vjp(grad_tip_p)
  ↓
Python Binding: py_true_legacy_step_vjp_batched (crm_bindings.cpp:708)
  ↓
C++ Wrapper: true_legacy_step_backward_batched (CRM_TrueLegacyDynamics.cpp:351)
  ↓
Analytic IVP Jacobian: CRMSolverIVPJacobian(...)
  ↓
Analytic BVP Jacobian: compute_bvp_jacobians_full_analytic(...)
  ↓
Adjoint Solve: QR decomposition of J_yy^T
  ↓
IFT Application: grad_x = -J_yx^T * lambda, grad_u = -J_yu^T * lambda
  ↓
Returns: grad_x_coil, grad_xf, grad_u
```

**Evidence**: src/CRM_TrueLegacyDynamics.cpp:222-337

### F.3 Linearization Chain (A, B Matrices)

```
Python: A, B = true_legacy_linearize(x, u, dt, ...)
  ↓
Python Binding: py_true_legacy_linearize (crm_bindings.cpp:821)
  ↓
C++ Wrapper: true_legacy_linearize_implicit (CRM_TrueLegacyDynamics.hpp:88)
  ↓
Forward Pass: true_legacy_step_forward (cache result)
  ↓
Jacobian Computation: compute_bvp_jacobians_full_analytic
  ↓
IFT Application: A = G_x - G_y * (R_y^{-1} * R_x), B = G_u - G_y * (R_y^{-1} * R_u)
  ↓
Returns: A [state_dim, state_dim], B [state_dim, control_dim]
```

**Evidence**: src/CRM_TrueLegacyDynamics.cpp:529-705

---

## Findings Summary

### ✅ PASS

1. **FULLSTATE contract enforced**: 18*N+15 dimensions verified at all layers
2. **Entrypoints correct**: TRUE legacy APIs bound and operational
3. **Differentiation compliance**: Analytic Jacobians, IFT/adjoint, no FD
4. **Reduced6D isolation**: Zero default imports, proper archiving
5. **Basic test coverage**: 6 smoke/unit tests present

### ⚠️ WARNINGS

1. **Missing VJP gradcheck for state variables** (∂x_coil, ∂xf)
2. **No numerical accuracy tests for linearization**
3. **No batched VJP equivalence test**

### ❌ NO VIOLATIONS FOUND

All required properties satisfied. No contract violations detected.

---

## Recommendations

See `NEXT_STEP_RECOMMENDATION.md` for prioritized next action.
