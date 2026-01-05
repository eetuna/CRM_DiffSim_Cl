# P1-5: API Versioning Completion Audit

**Date**: 2026-01-01
**Status**: ✅ COMPLETE
**Runtime**: < 2 seconds (target: < 2s)

---

## Executive Summary

Successfully implemented explicit API versioning for the Python-facing crm_diff_py module. All forward/backward entrypoints now return dicts with version metadata, enabling downstream users to reliably detect API compatibility. Implementation is minimal, non-breaking, and fully tested.

**Key Metrics**:
- Zero breaking changes (all existing dict keys preserved)
- New keys added: `api_version`, `api_contract` (2 per function)
- Test runtime: 0.45s (well under 2s target)
- All fast gate tests pass with no regressions

---

## What Changed

### 1. Version Constants (python/crm_bindings.cpp)

Added four version constants at namespace level (lines 11-15):

```cpp
// P1-5: API Versioning
static const char* CRM_PACKAGE_VERSION = "1.0.0";
static const char* CRM_API_VERSION = "1.1.0";
static const char* CRM_API_CONTRACT_EQUILIBRIUM = "equilibrium_v1_1";
static const char* CRM_API_CONTRACT_DYNAMICS = "dynamics_v1_1";
```

**Rationale**:
- `CRM_PACKAGE_VERSION`: Package version (aligns with CMake VERSION_INFO)
- `CRM_API_VERSION`: API contract version (1.1.0 reflects P1-5 additions)
- Contract strings distinguish equilibrium vs dynamics API surfaces

### 2. Module Attributes (python/crm_bindings.cpp)

Exposed version constants to Python (PYBIND11_MODULE, lines 509-511):

```cpp
// P1-5: API Versioning - expose version constants
m.attr("__version__") = CRM_PACKAGE_VERSION;
m.attr("__api_version__") = CRM_API_VERSION;
```

**Usage**:
```python
import crm_diff_py
print(crm_diff_py.__version__)      # "1.0.0"
print(crm_diff_py.__api_version__)  # "1.1.0"
```

### 3. Compatibility Check Helper (python/crm_bindings.cpp)

Added `check_api_compat()` function (lines 470-504) with semantic versioning logic:

```cpp
bool check_api_compat(const std::string& required_version)
```

**Behavior**:
- ✅ Compatible if major version matches AND current >= required
- ❌ Raises RuntimeError if major version mismatch (breaking change)
- ❌ Raises RuntimeError if current < required (need to update)

**Exposed to Python** (lines 513-518):
```python
crm_diff_py.check_api_compat("1.1.0")  # Returns True
crm_diff_py.check_api_compat("1.0.0")  # Returns True (backward compatible)
crm_diff_py.check_api_compat("2.0.0")  # Raises RuntimeError (major mismatch)
```

### 4. Version Metadata in Return Dicts

Added two keys to ALL forward/backward function outputs (NON-BREAKING):

**equilibrium_forward** (lines 125-127):
```cpp
out["api_version"] = CRM_API_VERSION;
out["api_contract"] = CRM_API_CONTRACT_EQUILIBRIUM;
```

**equilibrium_backward** (lines 199-201):
```cpp
out["api_version"] = CRM_API_VERSION;
out["api_contract"] = CRM_API_CONTRACT_EQUILIBRIUM;
```

**dynamics_forward** (lines 319-321):
```cpp
out["api_version"] = CRM_API_VERSION;
out["api_contract"] = CRM_API_CONTRACT_DYNAMICS;
```

**dynamics_backward** (lines 451-453):
```cpp
out["api_version"] = CRM_API_VERSION;
out["api_contract"] = CRM_API_CONTRACT_DYNAMICS;
```

**All existing keys remain unchanged** (no regressions).

---

## Files Modified

### C++ Bindings
- **python/crm_bindings.cpp** (4 sections modified)
  - Added version constants (lines 11-15)
  - Added check_api_compat() helper (lines 470-504)
  - Exposed module attributes (lines 509-511, 513-518)
  - Added version keys to 4 wrapper functions (equilibrium_forward/backward, dynamics_forward/backward)

### Build System
- **CMakeLists.txt** (1 test added, lines 301-308)
  - Added `api_versioning_p1_5` test with 5s timeout

### Tests
- **python/test_p1_5_api_versioning.py** (NEW, 335 lines)
  - Test module version attributes (__version__, __api_version__)
  - Test check_api_compat() validation logic
  - Test equilibrium_forward/backward versioning + regression check
  - Test dynamics_forward/backward versioning + regression check

### Documentation
- **docs/audits/P1_5_API_VERSIONING_COMPLETION.md** (THIS FILE)

---

## Version String Choices

### Package Version: "1.0.0"
Reflects stable v1.0 release of the crm_diff_py module. Aligns with CMakeLists.txt VERSION_INFO.

### API Version: "1.1.0"
- **Major = 1**: No breaking changes since CP2 API stabilization
- **Minor = 1**: P1-5 added new keys (`api_version`, `api_contract`) — backward compatible
- **Patch = 0**: No bug-fix-only releases yet

### Contract Strings
- `"equilibrium_v1_1"`: Equilibrium forward/backward API contract
- `"dynamics_v1_1"`: Dynamics forward/backward API contract

**Rationale**: Allows finer-grained detection of which API surface changed (e.g., if only equilibrium API evolves in the future, contract string updates to `equilibrium_v1_2` while dynamics remains `dynamics_v1_1`).

---

## How to Bump Version in the Future

### When to Bump What

| Change Type | Package Version | API Version | Contract String |
|-------------|----------------|-------------|-----------------|
| **Breaking change** (remove/rename keys) | Bump major | Bump major | Update (e.g., v2_0) |
| **Add new keys** (like P1-5) | Bump minor | Bump minor | Update (e.g., v1_2) |
| **Fix bug, no API change** | Bump patch | No change | No change |
| **Internal refactor, no API change** | No change | No change | No change |

### Step-by-Step Procedure

1. **Edit `python/crm_bindings.cpp`** (lines 12-15):
   ```cpp
   static const char* CRM_PACKAGE_VERSION = "X.Y.Z";  // Update
   static const char* CRM_API_VERSION = "A.B.C";       // Update
   static const char* CRM_API_CONTRACT_EQUILIBRIUM = "equilibrium_vA_B";
   static const char* CRM_API_CONTRACT_DYNAMICS = "dynamics_vA_B";
   ```

2. **Rebuild the module**:
   ```bash
   cd build
   cmake --build . --target crm_diff_py -j4
   ```

3. **Update tests** (if needed):
   - If breaking change, update test expectations in `python/test_p1_5_api_versioning.py`

4. **Run validation**:
   ```bash
   ctest -R api_versioning_p1_5 --output-on-failure
   ```

5. **Document in audit** (create new P1-X completion doc or update existing)

---

## Test Coverage

### Test Name: `api_versioning_p1_5`

**File**: `python/test_p1_5_api_versioning.py`
**Runtime**: 0.45s (target: < 2s) ✅
**Status**: PASSING ✅

### Test Matrix

| Test Case | Coverage |
|-----------|----------|
| Module attributes exist | `__version__`, `__api_version__` are valid semver |
| check_api_compat logic | Same version ✅, older version ✅, major mismatch ❌, newer version ❌ |
| equilibrium_forward | `api_version`, `api_contract` present + all existing keys intact |
| equilibrium_backward | `api_version`, `api_contract` present + all existing keys intact |
| dynamics_forward | `api_version`, `api_contract` present + all existing keys intact |
| dynamics_backward | `api_version`, `api_contract` present + all existing keys intact |

### Regression Validation

**Existing Keys Verified** (no removals):

- **equilibrium_forward**: status, p_tip, deltau0, converged, J_p_u0, J_u_u0, J_p_zc, J_u_zc, K_tip, nl_iterations, final_residual, lu_rank, rel_solve_residual, exit_code
- **equilibrium_backward**: status, grad_u, lu_rank, rel_residual
- **dynamics_forward**: status, x_next, p_tip, u_tip, J_G_xnext, J_G_xt, J_G_ut, J_p_u0, J_p_ut, M, D, K, u_t_cached, dt_cached, L_inserted_cached, K_tip_cached, J_u_zc_cached, lu_rank, rel_solve_residual, solve_residual, converged, exit_code
- **dynamics_backward**: status, grad_x_t, grad_u_t, lu_rank, rel_residual

---

## Acceptance Gates: PASSED ✅

### 1. P1-5 Test Passes
```bash
$ ctest -R api_versioning_p1_5 --output-on-failure
Test project /workspaces/CRM_DiffSim_Cl/build
    Start 11: api_versioning_p1_5
1/1 Test #11: api_versioning_p1_5 ..............   Passed    0.45 sec

100% tests passed, 0 tests failed out of 1
```

### 2. Fast Gate Tests Pass (No Regressions)
```bash
$ ctest -R "dynamics_smoke_cp21|dynamics_fd_cp22|dynamics_smoke_cp23_python" --output-on-failure
Test project /workspaces/CRM_DiffSim_Cl/build
    Start 5: dynamics_smoke_cp21
1/3 Test #5: dynamics_smoke_cp21 ..............   Passed    0.01 sec
    Start 6: dynamics_fd_cp22
2/3 Test #6: dynamics_fd_cp22 .................   Passed    0.32 sec
    Start 7: dynamics_smoke_cp23_python
3/3 Test #7: dynamics_smoke_cp23_python .......   Passed    0.37 sec

100% tests passed, 0 tests failed out of 3
```

### 3. No Breaking Changes
✅ All existing dict keys preserved (verified by test assertions)
✅ Only additive changes (`api_version`, `api_contract` keys)

---

## Commands to Run

### Run P1-5 Test
```bash
cd /workspaces/CRM_DiffSim_Cl/build
ctest -R api_versioning_p1_5 --output-on-failure
```

### Run Fast Gate Regression Subset
```bash
cd /workspaces/CRM_DiffSim_Cl/build
ctest -R "dynamics_smoke_cp21|dynamics_fd_cp22|dynamics_smoke_cp23_python" --output-on-failure
```

### Run All Tests (Full CI)
```bash
cd /workspaces/CRM_DiffSim_Cl/build
ctest --output-on-failure
```

### Rebuild Module (if modifying crm_bindings.cpp)
```bash
cd /workspaces/CRM_DiffSim_Cl/build
cmake --build . --target crm_diff_py -j4
```

---

## Git Diffstat

```
 CMakeLists.txt                     |   8 ++
 docs/audits/P1_5_API_VERSIONING_COMPLETION.md | 353 ++++++++++++++++++++++
 python/crm_bindings.cpp            |  47 +++
 python/test_p1_5_api_versioning.py | 335 ++++++++++++++++++++
 4 files changed, 743 insertions(+)
```

**Summary**:
- 3 files modified (CMakeLists.txt, crm_bindings.cpp)
- 2 files created (test, documentation)
- ~750 lines added (mostly test + docs)

---

## Suggested Commit Message

```
P1-5: Add explicit API versioning to Python bindings

WHAT:
- Added __version__ and __api_version__ module attributes
- Added check_api_compat() helper for runtime version validation
- Attached api_version and api_contract keys to all forward/backward dicts
  (equilibrium_forward/backward, dynamics_forward/backward)

WHY:
- Enables downstream users to reliably detect API compatibility
- Supports graceful evolution of the Python API surface
- Prevents silent breakage when upgrading crm_diff_py

HOW:
- Version constants in crm_bindings.cpp (CRM_API_VERSION = "1.1.0")
- Semantic versioning with major/minor/patch semantics
- NON-BREAKING: only added new dict keys, all existing keys preserved

TESTS:
- Added test_p1_5_api_versioning.py (runtime: 0.45s)
- Wired into CTest as api_versioning_p1_5
- Verified no regressions in fast gate tests (cp21, cp22, cp23)

DELIVERABLES:
✅ Version constants exposed (__version__, __api_version__)
✅ Version metadata in all dict outputs (api_version, api_contract)
✅ Runtime compatibility check (check_api_compat)
✅ Test coverage with regression validation
✅ Documentation (P1_5_API_VERSIONING_COMPLETION.md)

FILES MODIFIED:
- python/crm_bindings.cpp (version constants, helper, dict keys)
- CMakeLists.txt (test registration)
- python/test_p1_5_api_versioning.py (NEW)
- docs/audits/P1_5_API_VERSIONING_COMPLETION.md (NEW)

ACCEPTANCE GATES: ALL PASSED
- ctest -R api_versioning_p1_5 (0.45s, PASS)
- Fast gate tests (cp21, cp22, cp23_python, all PASS)
- Zero breaking changes (all existing keys present)
```

---

## Risk Assessment

**Risk Level**: 🟢 LOW

**Justification**:
1. **Non-breaking**: Only adds new dict keys, never removes/renames existing keys
2. **Minimal surface area**: Changes confined to pybind11 wrapper layer
3. **No physics/solver changes**: CRM core logic untouched
4. **Well-tested**: Explicit regression checks for all existing keys
5. **Fast gates pass**: cp21, cp22, cp23 tests confirm no regressions

**Potential Concerns**:
- ❌ None identified (additive-only change with explicit regression protection)

---

## Future Work (Optional)

### Short-term (Optional, not blocking)
- Add version checks to PyTorch autograd wrappers (crm_equilibrium.py, crm_dynamics_torch.py)
- Consider logging warnings if user's downstream code relies on old API assumptions

### Long-term (Out of Scope for P1-5)
- If major version bump (2.0.0) is needed, document migration guide
- Consider exposing version info in PyTorch wrapper classes

---

## Conclusion

P1-5 is **COMPLETE** and **PRODUCTION READY**. All acceptance gates passed with zero regressions. The API versioning system is minimal, non-breaking, and provides clear migration paths for future API evolution.

**Next Steps**: Merge to main and proceed to next priority task (if any).

---

**Signed-off**: Claude Sonnet 4.5 (2026-01-01)
