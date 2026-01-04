# PRE-B0 C++ BINDING LAYER AUDIT

**Audit Date:** 2026-01-04
**Auditor:** Claude Code (Sonnet 4.5)
**Scope:** C++ Python bindings for dynamics/equilibrium (Post-Legacy)
**Current Branch:** `true-legacy-dynamics` (commit 6d86781)
**Authority:** `python/crm_bindings.cpp` and new wrapper files in `src/`

---

## EXECUTIVE SUMMARY

The C++ binding layer exposes 11 primary functions to Python via pybind11. The bindings are **clean wrappers** that delegate all physics to legacy C++ entrypoints, with **one violation**: default inertia tensor computation in the bindings (line 1045) should be moved to the physics layer or Python configuration.

**Key Findings:**
- ✅ Pure wrappers—no physics logic duplication
- ✅ Copy-based caching (Python-managed lifetime via numpy arrays)
- ✅ Status codes returned, not thrown (physics failures expected)
- ⚠️  **ONE VIOLATION:** Default inertia computation at `crm_bindings.cpp:1045`
- ✅ All legacy entrypoints correctly called

---

## 1. EXPOSED PYTHON API (11 FUNCTIONS)

### 1.1 Equilibrium Primitives (Static Solving)

**`equilibrium_forward(u, L_inserted, params_dict)`**

Location: `python/crm_bindings.cpp` (exposes `CRM_DiffEquilibrium.cpp::equilibrium_forward()`)

Solves BVP for tip position given actuation currents.

**Caches Jacobians:**
- `J_p_u0` [3×3]: ∂p_tip/∂Δu₀
- `J_u_u0` [3×3]: ∂u_tip/∂Δu₀
- `J_p_zc` [3×3N]: ∂p_tip/∂actuation_curvatures
- `J_u_zc` [3×3N]: ∂u_tip/∂actuation_curvatures
- `K_tip` [3×3]: Stiffness matrix at tip

**Calls:**
- `CRMShootingMethodBVP()` (solves BVP for base curvature)
- `CRMSolverIVPJacobian()` (computes Jacobians via augmented state integration)

---

**`equilibrium_backward(fwd_result, grad_p_tip)`**

Location: `python/crm_bindings.cpp` (exposes `CRM_DiffEquilibrium.cpp::equilibrium_backward()`)

Computes VJP using implicit differentiation (no backprop through solver iterations).

**Returns:**
- `grad_u` [3N]: Gradient w.r.t. actuation currents

**Uses cached data from `equilibrium_forward`:**
- Jacobians: `J_p_u0`, `J_u_u0`, `J_p_zc`, `J_u_zc`, `K_tip`

---

### 1.2 Simplified Dynamics Primitives (6-DOF Reduced Model)

**`dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)`**

Location: `python/crm_bindings.cpp` (exposes `CRM_DiffDynamics.cpp::dynamics_forward()`)

Computes next state x_{t+1} using implicit Euler integration.

**State:** 6D reduced (u_0[3], v_0[3])

**Caches:**
- `J_G_xnext` [6×6]: ∂G/∂x_{t+1}
- `J_G_xt` [6×6]: ∂G/∂x_t
- `J_G_ut` [6×3N]: ∂G/∂u_t
- Physics matrices: `M` [3×3], `D` [3×3], `K` [3×3]
- Inputs: `u_t_cached`, `dt_cached`, `K_tip_cached`

**Calls:**
- `equilibrium_forward()` (to get K_tip)
- `compute_physics_matrices()` (inline, uses `CathParams.ActMass/SegLengths`)
- `compute_jacobians()` (inline, assembles A, C, B from M, D, K)
- `Eigen::FullPivLU::solve()` (for implicit Euler step)

---

**`dynamics_backward(fwd_result, grad_x_next, params_dict)`**

Location: `python/crm_bindings.cpp` (exposes `CRM_DiffDynamics.cpp::dynamics_backward()`)

Single-sample VJP using cached forward data.

**Returns:**
- `grad_x_t` [6]: Gradient w.r.t. state at time t
- `grad_u_t` [3N]: Gradient w.r.t. actuation at time t

**Calls:**
- `equilibrium_backward()` (for ∂u/∂p gradients)
- `Eigen::FullPivLU::solve()` (adjoint linear system)

---

**`dynamics_backward_batched(fwd_result, V, params_dict)`**

Location: `python/crm_bindings.cpp` (exposes `CRM_DiffDynamics.cpp::dynamics_backward_batched()`)

Multi-RHS VJP for efficient Jacobian computation.

**Input:** `V` is K×6 matrix (K cotangent vectors)

**Returns:**
- `grad_x_t_batch` [K×6]
- `grad_u_t_batch` [K×3N]

**Efficiency:** Amortizes cost of linear system factorization across K right-hand sides.

---

**`dynamics_linearize(x_t, u_t, dt, L_inserted, params_dict)`**

Wrapper calling `forward` + `batched_backward`.

**Returns:**
- `A` [6×6]: State Jacobian
- `B` [6×3N]: Control Jacobian

**Use case:** iLQR/MPC linearization.

---

**`dynamics_linearize_batched(...)`**

Optimized multi-sample linearization (processes multiple (x_t, u_t) pairs).

---

### 1.3 True Legacy Dynamics (Full CRMDYN State)

**`true_legacy_step_forward(x_coil, xf, u, dt, params_dict, mL_guess, nL_guess)`**

Location: `python/crm_bindings.cpp:952-1206` (exposes `CRM_TrueLegacyDynamics.cpp::true_legacy_step_forward()`)

Wrapper around **DynamicsBVP → DYNSolverIVP** pipeline.

**State:**
- `x_coil` [N×18]: Per-coil state `[v, w, p, R]`
  - `v[3]`: Linear velocity
  - `w[3]`: Angular velocity
  - `p[3]`: Position
  - `R[9]`: Rotation matrix (row-major)
- `xf` [15]: Tip state `[p_tip, R_tip, u_tip]`

**Total state dimension:** `18*N + 15`

**Workflow:**
1. Validate inputs (shape checks for `x_coil`, `xf`, `u`)
2. Unpack coil state (extract `v_L_pre`, `w_L_pre`, `p_pre`, `R_pre`)
3. Construct shooting params (`CRMDYNConstructShootingMethodParamSet()`)
4. Call BVP solver: `DynamicsBVP()` → solves for `mL`, `nL`, `u0`
5. Propagate dynamics: `DYNSolverIVP()` → integrates rigid body motion
6. Package results (state, observables, warm-start guesses)

**Observables extracted:**
- `tip_p` [3]: First 3 elements of `xf_next`
- `tip_R` [9]: Elements 3-11 of `xf_next`
- `tip_u` [3]: Elements 12-14 of `xf_next`

**Convergence check:**
```cpp
converged = (out_localmin == 0);
```

---

**`true_legacy_step_vjp(x_coil, xf, u, dt, L_inserted, params_dict, grad_tip_p)`**

Location: `python/crm_bindings.cpp:1209-1324`

Single-sample VJP for tip position gradients.

**Workflow:**
1. Re-run forward (caches BVP solution in `TrueLegacyStepResult`)
2. Call batched backward with `num_rhs=1`

**Design note:** Single VJP delegates to batched code path for consistency.

**Returns:**
- `grad_x_coil` [N×18]
- `grad_xf` [15]
- `grad_u` [N×3]

---

**`true_legacy_step_vjp_batched(..., grad_tip_p_batch)`**

Location: `python/crm_bindings.cpp:1326-1450`

Processes K cotangent vectors in one call.

**Input:** `grad_tip_p_batch` [K×3] (row-major)

**Output:**
- `grad_x_coil_batch` [K×N×18]
- `grad_xf_batch` [K×15]
- `grad_u_batch` [K×N×3]

**Efficiency:** Amortizes cost of linear system factorization.

**Use case:** Trajectory optimization (iLQR/MPC with TRUE legacy state).

**Implementation:** `CRM_TrueLegacyDynamics.hpp::true_legacy_step_backward_batched()` (lines 71-81)

**Method:** Implicit function theorem on BVP solution (no backprop through solver iterations).

**Cached Data Structure:** `TrueLegacyStepResult` (lines 9-36)
- Stores full forward computation: inputs, outputs, BVP solution
- Enables implicit differentiation without re-solving BVP

---

### 1.4 Reference/Testing

**`crmdyn_reference_rollout(...)`**

Location: `python/crm_bindings.cpp` (exposes `CRM_ReferenceHarness.cpp::crmdyn_reference_rollout()`)

Deterministic trajectory rollout for regression testing.

**Purpose:** Parity gate between MATLAB/C++ reference implementations.

---

## 2. LIFETIME MANAGEMENT OF CACHED STATES

### 2.1 Strategy: Copy-on-Return, Python-Managed Lifetime

**Forward Pass:**
1. C++ allocates result struct on stack (e.g., `DynamicsStepResult`)
2. Populates Jacobians via `memcpy` from solver outputs
3. Bindings copy data to numpy arrays with explicit strides
4. Returns Python dict containing numpy arrays
5. **C++ struct goes out of scope immediately**

**Backward Pass:**
1. Python passes forward result dict back to C++
2. Bindings reconstruct C++ struct via `memcpy` from numpy arrays
3. Validates shapes/dtypes with runtime checks
4. Calls backward function with reconstructed data
5. **No persistent C++ state across calls**

**Memory Safety:** No raw pointers retained. `CathParams`/`CathConfig` are opaque Python-owned pointers (returned via `take_ownership` policy).

**Total `memcpy` calls:** 86 across all binding functions

---

## 3. LEGACY ENTRYPOINTS CALLED (MAPPING TABLE)

| Python Binding | C++ Wrapper | Legacy Physics Function |
|----------------|-------------|-------------------------|
| `equilibrium_forward` | `CRM_DiffEquilibrium.cpp` | `CRMShootingMethodBVP` |
| ↳ | ↳ | `CRMSolverIVPJacobian` |
| ↳ | ↳ | `CRMSolverIVP_CoreWithJacobian` |
| `dynamics_forward` | `CRM_DiffDynamics.cpp` | `equilibrium_forward` (to get K_tip) |
| ↳ | ↳ | `Eigen::FullPivLU::solve` (implicit Euler) |
| `dynamics_backward` | `CRM_DiffDynamics.cpp` | `equilibrium_backward` (∂u/∂p gradients) |
| ↳ | ↳ | `Eigen::FullPivLU::solve` (adjoint system) |
| `true_legacy_step_forward` | `CRM_TrueLegacyDynamics.cpp` | `CRMDYNConstructShootingMethodParamSet` |
| ↳ | ↳ | `DynamicsBVP` |
| ↳ | ↳ | `CRMShootingMethodCore` (trust-region) |
| ↳ | ↳ | `DYNSolverIVP` |
| ↳ | ↳ | `CRMSolverIVP_Core` (RK4 integration) |
| `true_legacy_step_vjp_batched` | `CRM_TrueLegacyDynamics.cpp` | `CRM_BVP_Adjoint_MultiRHS` |
| ↳ | ↳ | `CRM_IVP_Adjoint_MultiRHS` |

---

## 4. EXCEPTION HANDLING PHILOSOPHY

### 4.1 Input Validation (Before Physics Calls)

**Shape/dtype checks throw exceptions:**

```cpp
if (u_buf.ndim != 1 || u_buf.shape[0] != NUM_ACT_SET * 3) {
    throw std::runtime_error("u must be 1D array of length ...");
}
if (!py::isinstance<py::array_t<double, py::array::c_style | py::array::forcecast>>(...)) {
    throw std::runtime_error("grad_p_tip must be float64 C-contiguous array");
}
```

### 4.2 Physics Solver Status Codes (Returned, Not Thrown)

```cpp
int status = dynamics_forward(x_t, u_t, dt, L_inserted, fk_params, result);
if (status != 0) {
    out["exit_code"] = result.exit_code;  // 0=OK, 1=rank-deficient, 2=residual too large, 3=equilibrium failed
}
// No exception thrown—status returned to Python
```

**Convergence Diagnostics (Returned via Dict):**
```cpp
out["converged"] = result.converged;  // 0=success, >0=failed
out["lu_rank"] = result.lu_rank;
out["rel_solve_residual"] = result.rel_solve_residual;
out["nl_iterations"] = result.nl_iterations;  // Trust-region iterations
```

**Philosophy:** Physics failures are **not exceptions**—they're expected and reported via status codes. Only data type/shape errors throw.

---

## 5. ONE VIOLATION: PHYSICS LOGIC IN BINDINGS

### 5.1 Default Inertia Tensor Computation

**Location:** `python/crm_bindings.cpp:1045`

```cpp
// Use default diagonal inertia
for (int j = 0; j < NUM_ACT_SET; ++j) {
    for (int i = 0; i < 9; ++i) {
        ActInertia[j][i] = (i % 4 == 0) ? CathParams->ActMass[j] * 1e-6 : 0.0;
    }
}
```

**Issue:** This computes a default inertia tensor (diagonal with `1e-6` scaling factor) when `ActInertia` is not provided. This is **physics logic** (determining rotational inertia from mass), not pure wrapping.

**Recommendation:**
- **Option A:** Move to Python-side helper in `crm_config.py`
- **Option B:** Dedicated C++ function in `CRM.hpp` (e.g., `CRMComputeDefaultActInertia()`)

**Impact:** MEDIUM—not a critical safety issue, but violates binding layer purity.

---

## 6. THREAD SAFETY

**Assumption:** Single-threaded execution

**Evidence:**
- No mutex locks or atomic operations in bindings
- Global state (if any) not protected
- GIL (Global Interpreter Lock) provides Python-side thread safety

**Recommendation:** Document that concurrent calls to the same binding from multiple Python threads are NOT safe.

---

## 7. NO PHYSICS DUPLICATION VERIFICATION

### 7.1 All Other Physics Correctly Delegated

**Confirmed:**
- `CRM_DiffEquilibrium.cpp` → calls legacy equilibrium solvers
- `CRM_DiffDynamics.cpp` → calls legacy dynamics + equilibrium
- `CRM_TrueLegacyDynamics.cpp` → calls legacy `DynamicsBVP` and `DYNSolverIVP`
- Legacy solvers in: `CoilDynamics_Defs.cpp`, `CRM_BVPSolver.cpp`, `CRM_IVPSolver.cpp`

**Binding Layer Responsibilities:**
- Type conversion (numpy ↔ C arrays)
- Parameter unpacking (Python dicts → C structs)
- Result packaging (C structs → Python dicts with numpy arrays)
- Input validation (shape/dtype checking)

**File Sizes:**
- `crm_bindings.cpp`: 1628 lines (mostly boilerplate wrapping)
- `CRM_DiffDynamics.cpp`: 445 lines (physics)
- `CRM_DiffEquilibrium.cpp`: 197 lines (physics)
- `CRM_TrueLegacyDynamics.cpp`: 529 lines (physics)

---

## 8. CRITICAL FILES AUDITED

### 8.1 Binding Code

- `python/crm_bindings.cpp` — All Python-exposed functions (1628 lines)

### 8.2 Wrapper Code (New, Post-Legacy)

- `src/CRM_DiffEquilibrium.cpp` / `.hpp` — Equilibrium gradients (197 lines)
- `src/CRM_DiffDynamics.cpp` / `.hpp` — 6D dynamics gradients (445 lines)
- `src/CRM_TrueLegacyDynamics.cpp` / `.hpp` — TRUE legacy wrappers (529 lines)
- `src/CRM_BVPJacobian.cpp` / `.hpp` — TRUE legacy VJP Jacobians (385 lines)
- `src/CRM_ReferenceHarness.cpp` / `.hpp` — Regression testing harness (311 lines)

### 8.3 Modified Physics (ONE FILE)

- `src/CRM_IVPJacobian.cpp` — Lines 6, 95-100 (rank check added)

**Modification:** Prevents gradient explosion from rank-deficient Jacobians.

---

## 9. KEY FINDINGS SUMMARY

| Finding | Severity | Evidence |
|---------|----------|----------|
| Pure wrappers (no physics duplication) | ✅ Good | All physics delegated to legacy |
| Copy-based caching (safe lifetime) | ✅ Good | 86 memcpy calls, Python-owned arrays |
| Status codes (not exceptions) | ✅ Good | Physics failures expected |
| Default inertia in bindings | ⚠️ MEDIUM | `crm_bindings.cpp:1045` |
| No thread safety | ⚠️ LOW | Single-threaded assumption not documented |

---

## 10. SAFETY RECOMMENDATIONS FOR B0

### 10.1 Before System Identification

1. **Move default inertia computation** out of bindings (`crm_bindings.cpp:1045`)
   - Create `crm_config.compute_default_inertia()` in Python
   - Update caller to explicitly pass `ActInertia` array

2. **Document thread safety assumptions**
   - Add comment: "NOT SAFE for concurrent calls from multiple threads"

### 10.2 Runtime Validation

1. **Check status codes** returned by all forward/backward calls
2. **Validate finite values** in all returned arrays (detect NaN/Inf early)
3. **Monitor `lu_rank`** in dynamics backward (should be 6 for 6D, 18*N+15 for TRUE legacy)

---

## AUDIT STATUS

**Scope:** C++ binding layer audit (post-legacy wrappers)
**Modifications Made:** NONE (audit-only)
**Code Modified:** NO
**Violations Found:** ONE (default inertia at line 1045)

**Completion:** ✅ ALL BINDING LAYER AUDITED

**Next Steps:**
- Proceed to Python Control Layer Audit
- Then Cross-Layer Risk Analysis

---

**END OF C++ BINDING LAYER AUDIT**
