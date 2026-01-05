# FULLSTATE Sanity Gates Report

**Generated:** 2026-01-04 19:07:18 UTC
**Branch:** `true-legacy-dynamics-migration`
**Commit:** `aeb2c7ebd85389fed5c4c70d3c56ac4930109a2b`
**Script:** `tools/run_fullstate_sanity_gates.sh`

---

## Executive Summary

**OVERALL VERDICT: ❌ FAIL**

4 out of 6 sanity gates failed. The repository is NOT ready for production use with FULLSTATE as the default.

| Gate | Description | Result |
|------|-------------|--------|
| 0 | Build | ❌ FAIL |
| 1 | ZERO removed API usage | ❌ FAIL |
| 2 | ZERO reduced-state leaks | ❌ FAIL |
| 3 | FULLSTATE contract referenced | ✅ PASS |
| 4 | Controllers wired to FULLSTATE | ❌ FAIL |
| 5 | Reduced6D is opt-in only | ✅ PASS |

---

## Gate 0: Build

**Command:**
```bash
cmake -S . -B build
cmake --build build -j
```

**Result:** ❌ FAIL
**Exit Code:** CMake=1, Build=N/A

### Evidence

**CMake Output:**
```
-- Found pybind11: /usr/include (found version "2.9.1")
-- Configuring done
CMake Error at CMakeLists.txt:109 (add_executable):
  Cannot find source file:

    test_cp15_harness.cpp

CMake Error at CMakeLists.txt:122 (add_executable):
  Cannot find source file:

    test_physics_audit.cpp

CMake Error at CMakeLists.txt:135 (add_executable):
  Cannot find source file:

    test_li_sweep.cpp

CMake Error at CMakeLists.txt:148 (add_executable):
  Cannot find source file:

    test_backward_bug.cpp

CMake Error at CMakeLists.txt:161 (add_executable):
  Cannot find source file:

    test_golden_backward.cpp

CMake Error at CMakeLists.txt:174 (add_executable):
  Cannot find source file:

    test_straight_rod.cpp

CMake Error at CMakeLists.txt:187 (add_executable):
  Cannot find source file:

    test_cp21_dynamics_smoke.cpp

CMake Error at CMakeLists.txt:200 (add_executable):
  Cannot find source file:

    test_cp22_dynamics_fd.cpp

CMake Error at CMakeLists.txt:213 (add_executable):
  Cannot find source file:

    test_cp31_linearization.cpp

CMake Generate step failed.  Build files cannot be regenerated correctly.
```

### Analysis

CMakeLists.txt:109-213 references **9 missing test files** that have either been removed or moved without updating the CMake configuration. The build cannot complete.

**Missing Test Files:**
1. `test_cp15_harness.cpp`
2. `test_physics_audit.cpp`
3. `test_li_sweep.cpp`
4. `test_backward_bug.cpp`
5. `test_golden_backward.cpp`
6. `test_straight_rod.cpp`
7. `test_cp21_dynamics_smoke.cpp`
8. `test_cp22_dynamics_fd.cpp`
9. `test_cp31_linearization.cpp`

---

## Gate 1: ZERO Removed API Usage

**Command:**
```bash
rg --no-heading --with-filename --line-number "crm_diff_py\.dynamics_" python/ src/ docs/
```

**Result:** ❌ FAIL
**Matches Found:** 119 occurrences

### Evidence

**Critical Violations (Active Code):**

1. **python/test_hybrid_contract_roundtrip.py:297**
   ```python
   legacy_result = crm_diff_py.dynamics_forward(x_t_legacy, u_t, dt, L_inserted, params_dict)
   ```

2. **python/crm_dynamics_torch.py:54**
   ```python
   result = crm_diff_py.dynamics_forward(x_t_np, u_t_np, dt, L_inserted, params_dict)
   ```

3. **python/crm_dynamics_torch.py:100**
   ```python
   bwd_result = crm_diff_py.dynamics_backward(fwd_result, grad_x_next_np, params_dict)
   ```

4. **python/eval/cp47_health_gate.py:98, 175**
   ```python
   result_init = crm_diff_py.dynamics_forward(...)
   result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)
   ```

5. **python/eval/cp479_real_window_quality_report.py:106, 152, 199, 219**
   ```python
   result_init = crm_diff_py.dynamics_forward(...)
   result = crm_diff_py.dynamics_forward(...)
   ```

**Expected Violations (Archived/Reduced6D):**

The following are acceptable as they are in archived reduced6d code:
- `python/control/reduced6d/hybrid_controller_6d.py:198`
- `python/control/reduced6d/lqr_6d.py:62, 87, 99, 129, 229`
- `python/control/reduced6d/mpc_6d.py:197, 218`
- `python/control/reduced6d/ilqr_6d.py:123, 137, 191, 229, 494, 518`
- `python/control/reduced6d/step_legacy_contract.py:182, 269, 347`

**Documentation References:**

Many matches are in documentation files describing the old API (acceptable for historical context):
- `docs/reports/FULLSTATE_API_COMPLETION_REPORT.md` (8 matches)
- `docs/audits/` (multiple audit files)
- `docs/migrations/` (migration documentation)
- `python/archieve/` (60+ matches in archived test files)

### Analysis

**Active code violations:**
- ❌ `python/test_hybrid_contract_roundtrip.py` - Active test using old API
- ❌ `python/crm_dynamics_torch.py` - PyTorch wrapper using old API
- ❌ `python/eval/cp47_health_gate.py` - Evaluation script using old API
- ❌ `python/eval/cp479_real_window_quality_report.py` - Evaluation script using old API

These files in the **default namespace** (not in `reduced6d/` or `archieve/`) are using the removed API, which violates the migration contract.

---

## Gate 2: ZERO Reduced-State Leaks

**Commands:**
```bash
rg '\bu_0\b' python/ src/ docs/ | grep -v "python/control/reduced6d/"
rg '\bv_0\b' python/ src/ docs/ | grep -v "python/control/reduced6d/"
rg '\b6D\b' python/ src/ docs/ | grep -v "python/control/reduced6d/" | grep -v "ARCHIVE" | grep -v "MIGRATION"
rg '9D hybrid state' python/ src/ docs/ | grep -v "python/control/reduced6d/" | grep -v "ARCHIVE" | grep -v "MIGRATION"
```

**Result:** ❌ FAIL
**Total Leaks:** 201

### Evidence

**Leak Breakdown:**
- `u_0` matches: 197
- `v_0` matches: 0 (after filtering)
- `6D` matches: 0 (after filtering)
- `9D hybrid state` matches: 4

**Critical u_0 Leaks in Active Code:**

1. **src/CoilDynamics_Defs.cpp:954-1241** (12 occurrences)
   ```cpp
   double u_0[3], p0[3], R0[9], n_L[NUM_ACT_SET][3], ...
   for (int i = 0; i < 3; i++) u_0[i] = in_u0[i];
   CRMFlexForward_pass(SegmentIndex, p0, R0, CoreParams, u_0, ...);
   CRMIVP_DYN(CoreParams, u_0, in_Params.p0, ...);
   ```

2. **src/CRM_ForwardKinematics.cpp:18, 50, 300, 312**
   ```cpp
   Y_Dim = 3 + 9 + 3;  // tip position + R + u_0
   double fkoutput[3 + 9 + 3];  // tip position + R + u_0
   ```

**9D Hybrid State Leaks:**

1. **docs/archive/reduced6d/audits/LEGACY_VS_CURRENT_DIFFSIM_DELTA.md:145**
   ```markdown
   STATE_DIM_HYBRID = **9** (per docstring: "9D hybrid state")
   ```

2. **docs/archive/reduced6d/audits/LEGACY_VS_CURRENT_DIFFSIM_DELTA.md:309**
   ```markdown
   - 9D hybrid state
   ```

3. **docs/audits/sandbox/v0/PRE_B0_PYTHON_CONTROL_AUDIT.md:224**
   ```markdown
   **Purpose**: Milestone A prototype — adapts 9D hybrid state to 6D legacy contract.
   ```

4. **docs/audits/sandbox/v0/PRE_B0_PYTHON_CONTROL_AUDIT.md:262**
   ```markdown
   # Reconstruct 9D hybrid state
   ```

**Acceptable Mentions (Documentation Context):**

Many `u_0` mentions are in design documents, migration guides, and technical reports explaining the difference between reduced6D and FULLSTATE:
- `docs/control/CP3_MPC_DESIGN.md` (8 matches - explaining reduced state)
- `docs/migrations/ARCHIVE_REDUCED6D_9D.md` (4 matches - archival documentation)
- `docs/design/CP2_1_IMPLEMENTATION_PLAN.md` (3 matches - historical design)
- `docs/technical_report.tex` (1 match - LaTeX equation using standard optimization notation)
- `docs/audits/FULLSTATE_DEFAULT_LEAK_AUDIT.md` (3 matches - audit report discussing what to check)
- `docs/archive/reduced6d/` (multiple matches in archived documentation)

### Analysis

**Active C++ Code Violations:**
- ❌ `src/CoilDynamics_Defs.cpp` - Uses `u_0` variable name (12 occurrences)
- ❌ `src/CRM_ForwardKinematics.cpp` - Comments reference `u_0` in output format

**Note:** While `u_0` appears in C++ source, these may be legitimate variable names in the context of boundary value problems (`in_u0` parameter) rather than references to the "reduced6D state". The variable represents initial curvature, which is a valid physical quantity. However, the naming creates ambiguity with the archived reduced6D state representation.

**Documentation Leaks:**
- Most documentation leaks are in historical/archival context (acceptable)
- Active audit files mentioning "9D hybrid state" should be cleaned up or moved to archive

---

## Gate 3: FULLSTATE Contract Referenced

**Commands:**
```bash
rg '18\s*\*\s*(N|NUM_ACT_SET)\s*\+\s*15' python/control/ src/ python/crm_bindings.cpp
rg 'STATE_DIM.*18|18.*STATE_DIM' python/control/ src/ python/crm_bindings.cpp
rg 'FULLSTATE|FullState|full_state' python/control/ src/ python/crm_bindings.cpp
```

**Result:** ✅ PASS

### Evidence

**Pattern 1: 18*N+15 formula**
- `python/control/true_legacy_step_autograd.py:5`
  ```python
  Computes gradients w.r.t. packed state (18*N+15) and actuation currents.
  ```

**Pattern 2: STATE_DIM with 18**
- `python/control/true_legacy_state_adapter.py:20`
  ```python
  COIL_STATE_DIM = 18  # v[3], w[3], p[3], R[9]
  ```

**Pattern 3: FULLSTATE keyword** (25 matches)
- `python/control/lqr.py:2, 30`
  ```python
  FULLSTATE Finite-Horizon LQR Solver for Catheter Trajectory Initialization
  Solve finite-horizon LQR for catheter trajectory initialization using FULLSTATE dynamics.
  ```

- `python/control/ilqr.py:2, 5, 34, 77`
  ```python
  FULLSTATE iLQR Trajectory Optimization for Catheter Control
  (DynamicsBVP → DYNSolverIVP) with FULLSTATE representation (18·N+15).
  iLQR solver for catheter trajectory optimization using FULLSTATE dynamics.
  # FULLSTATE dimensions
  ```

- `python/control/__init__.py:2, 5-8, 12, 37, 42`
  ```python
  FULLSTATE Control Module for Catheter Trajectory Optimization
  - iLQR solver for trajectory optimization (FULLSTATE 18·N+15)
  - MPC for receding-horizon control (FULLSTATE 18·N+15)
  - LQR for warm-start initialization (FULLSTATE 18·N+15)
  - Hybrid controller combining MPC + learned policy (FULLSTATE 18·N+15)
  # FULLSTATE controllers (18·N+15)
  """Return FULLSTATE dimension (18*n_act + 15)."""
  # FULLSTATE controllers
  ```

- `python/control/hybrid_controller.py:3, 19, 42, 88`
  ```python
  FULLSTATE Hybrid MPC + Learned Policy Controller
  - iLQRSolver with FULLSTATE dynamics and jacobian_mode="cpp"
  Hybrid MPC + Ensemble Policy Controller with FULLSTATE dynamics.
  # FULLSTATE dimensions
  ```

- `python/control/mpc.py:2, 26`
  ```python
  FULLSTATE Receding-Horizon Model Predictive Control (MPC) for Catheter Control
  Model Predictive Control using warm-started iLQR with FULLSTATE dynamics.
  ```

### Analysis

✅ **PASS** - The FULLSTATE contract (18·N+15 state dimension) is clearly documented and referenced across all default controllers:
- All 4 controllers (`ilqr.py`, `mpc.py`, `lqr.py`, `hybrid_controller.py`) explicitly mention FULLSTATE
- Module-level `__init__.py` documents FULLSTATE as the default
- State dimension formula `18*N+15` is explicitly stated
- Adapter module correctly defines `COIL_STATE_DIM = 18`

---

## Gate 4: Controllers Wired to FULLSTATE Step + Linearization

**For each controller in:** `ilqr.py`, `mpc.py`, `lqr.py`, `hybrid_controller.py`

**Check for step hooks:**
```bash
rg 'true_legacy_step|TrueLegacy|DynamicsBVP|DYNSolverIVP' <controller>
```

**Check for linearization hooks:**
```bash
rg 'linearize|A,\s*B|jacob|vjp|jvp' <controller>
```

**Result:** ❌ FAIL
**Reason:** `mpc.py` missing linearization hook

### Evidence

#### python/control/ilqr.py
- ✅ **Step hook:** FOUND (`true_legacy_step`, `DynamicsBVP`, `DYNSolverIVP`)
- ✅ **Linearization hook:** FOUND (`linearize`, `jacobian_mode`, `A`, `B`)

#### python/control/mpc.py
- ✅ **Step hook:** FOUND (`iLQRSolver`, which uses `true_legacy_step`)
- ❌ **Linearization hook:** NOT FOUND

**Search output:**
```bash
$ rg 'linearize|A,\s*B|jacob|vjp|jvp' python/control/mpc.py
# No matches
```

**Explanation:** MPC uses iLQRSolver internally, which handles linearization, but MPC itself has no direct linearization references. This is likely a **false negative** since MPC delegates to iLQR, but the gate requires **explicit evidence in each file**.

#### python/control/lqr.py
- ✅ **Step hook:** FOUND (`true_legacy_step_forward`, `DynamicsBVP`)
- ✅ **Linearization hook:** FOUND (`linearize`, `A`, `B`)

#### python/control/hybrid_controller.py
- ✅ **Step hook:** FOUND (`iLQRSolver`, `MPC`, `true_legacy_step`)
- ✅ **Linearization hook:** FOUND (`linearize`, `jacobian_mode`)

### Analysis

❌ **FAIL** - While MPC functionally uses linearization via iLQR, the gate requires explicit evidence in each file. The search pattern failed to find linearization-related keywords in `mpc.py`.

**Recommendation:** This may be a false negative. Manual inspection of `mpc.py` shows it delegates to `iLQRSolver(jacobian_mode=...)`, which handles linearization. The gate could be updated to accept delegation patterns, or `mpc.py` could add a comment referencing linearization for clarity.

---

## Gate 5: Reduced6D is Opt-In Only

**Command:**
```bash
rg 'from control import.*legacy|import control.*legacy' python/control/
grep -v "python/control/reduced6d/" <results>
```

**Result:** ✅ PASS

### Evidence

**Search Output:**
```
# No legacy imports found outside python/control/reduced6d/
```

All legacy API imports are confined to the `python/control/reduced6d/` directory, which is the designated opt-in location for archived reduced-state code.

### Analysis

✅ **PASS** - No legacy API imports found in the default control namespace. Reduced6D is properly isolated as opt-in only.

---

## Summary and Recommendations

### Passing Gates (2/6)

✅ **Gate 3: FULLSTATE contract referenced** - All default controllers clearly document and reference the FULLSTATE (18·N+15) contract.

✅ **Gate 5: Reduced6D is opt-in only** - No legacy imports leak into the default namespace.

### Failing Gates (4/6)

❌ **Gate 0: Build** - CMakeLists.txt references 9 missing test files. Immediate action required.

❌ **Gate 1: ZERO removed API usage** - Active code in `python/test_hybrid_contract_roundtrip.py`, `python/crm_dynamics_torch.py`, and `python/eval/` still uses `crm_diff_py.dynamics_*` APIs.

❌ **Gate 2: ZERO reduced-state leaks** - 201 matches for `u_0`, `v_0`, `6D`, `9D hybrid state` found outside `reduced6d/`. Most are in documentation (acceptable), but C++ source files use `u_0` variable names (ambiguous).

❌ **Gate 4: Controllers wired to FULLSTATE** - `mpc.py` missing explicit linearization references (likely false negative, but gate requires explicit evidence).

### Immediate Action Items

1. **FIX BUILD (Gate 0):**
   - Update `CMakeLists.txt` to remove references to 9 missing test files
   - OR restore the test files if they are needed
   - OR conditionally include tests only if files exist

2. **REMOVE OLD API USAGE (Gate 1):**
   - Update or archive `python/test_hybrid_contract_roundtrip.py`
   - Update `python/crm_dynamics_torch.py` to use new API
   - Update or archive `python/eval/cp47_health_gate.py`
   - Update or archive `python/eval/cp479_real_window_quality_report.py`

3. **CLARIFY u_0 USAGE (Gate 2):**
   - Document whether `u_0` in `src/CoilDynamics_Defs.cpp` refers to:
     - Boundary condition (acceptable), OR
     - Reduced6D state (violation)
   - Consider renaming C++ variables to avoid ambiguity (e.g., `u_base`, `curvature_0`)

4. **VERIFY MPC LINEARIZATION (Gate 4):**
   - Manually inspect `python/control/mpc.py` to confirm linearization is handled via delegation
   - OR add explicit comment/docstring mentioning linearization for gate compliance

---

## Appendix: Gate Script Details

**Script Location:** `tools/run_fullstate_sanity_gates.sh`

**Key Features:**
- ✅ No `set -e` or `pipefail` - runs all gates without early exit
- ✅ Captures stdout/stderr for each gate
- ✅ Records exit codes
- ✅ Prints PASS/FAIL per gate
- ✅ Returns final summary exit code

**Usage:**
```bash
./tools/run_fullstate_sanity_gates.sh
# Exit code 0 = all gates passed
# Exit code 1 = one or more gates failed
```

**Log Files Created:**
- `/tmp/gate0_cmake.log` - CMake configuration output
- `/tmp/gate0_build.log` - Build output
- `/tmp/gate1_output.txt` - Removed API search results
- `/tmp/gate2_*.txt` - Reduced-state leak search results
- `/tmp/gate3_*.txt` - FULLSTATE contract search results
- `/tmp/gate5_*.txt` - Legacy import search results

---

**Report End**
