# Reduced6D and 9D State Archive Documentation

**Date**: 2026-01-04
**Objective**: Document archived Reduced6D/9D code and clarify TRUE vs. NON-TRUE legacy

---

## Overview

This document explains what was archived, where it lives, and **why** it's not the default.

### Critical Distinction

| Implementation | State Dim | Physics Solver | Status | Location |
|----------------|-----------|----------------|--------|----------|
| **TRUE Legacy** | 18·N+15 | `DynamicsBVP → DYNSolverIVP` | **DEFAULT** | `src/CRM_TrueLegacyDynamics.{cpp,hpp}` |
| **Reduced6D** | 6 | `equilibrium_forward` + approx | ARCHIVED | `src/reduced6d/CRM_DiffDynamics.{cpp,hpp}` |
| **9D Hybrid** | 9 | `hybrid_state_contract` | ARCHIVED | `python/control/reduced6d/*_9d.py` |

**Key Point**: Reduced6D and 9D are **approximations**, not the original CRMDYN_test.cpp reference.

---

## What Was Archived

### C++ Files

**Archived to `src/reduced6d/`:**
- `CRM_DiffDynamics.cpp` - Reduced6D dynamics implementation
- `CRM_DiffDynamics.hpp` - Reduced6D header

**Removed from**:
- `CMakeLists.txt` (no longer in default `CRMSOURCES`)
- `python/crm_bindings.cpp` (no `dynamics_forward()` in default namespace)

### Python Controllers

**Moved to `python/control/reduced6d/`:**

| Old Path | New Path | Description |
|----------|----------|-------------|
| `control/ilqr.py` | `control/reduced6d/ilqr_6d.py` | Reduced6D iLQR |
| `control/mpc.py` | `control/reduced6d/mpc_6d.py` | Reduced6D MPC |
| `control/lqr.py` | `control/reduced6d/lqr_6d.py` | Reduced6D LQR |
| `control/hybrid_controller.py` | `control/reduced6d/hybrid_controller_6d.py` | Reduced6D Hybrid |
| `control/hybrid_state_contract.py` | `control/reduced6d/hybrid_state_contract_9d.py` | 9D state contract |
| `control/step_hybrid_legacy_contract.py` | `control/reduced6d/step_hybrid_legacy_contract_9d.py` | 9D step contract |

**NOT imported by default** from `python/control/__init__.py`.

### Tests

**Moved to `python/test/reduced6d/`:**
- `test_a1_legacy_state_adapter.py`
- `test_a1_step_legacy_contract_forward.py`
- `test_a2_implicit_vjp_gradcheck.py`
- `test_a3_batched_vjp.py`

**Excluded from CI** (not registered in `CMakeLists.txt` ctest suite).

---

## Why Reduced6D Was Archived

### Problem 1: Not the TRUE Legacy Implementation

Reduced6D (`u_0[3], v_0[3]`) is **not** what CRMDYN_test.cpp uses:

```cpp
// CRMDYN_test.cpp (TRUE legacy)
struct CoreState {
    double v_L[NUM_ACT_SET][3];    // Linear velocity per coil
    double w_L[NUM_ACT_SET][3];    // Angular velocity per coil
    double p[NUM_ACT_SET][3];      // Position per coil
    double R[NUM_ACT_SET][9];      // Rotation matrix per coil (row-major)
    double p_tip[3];               // Tip position
    double R_tip[9];               // Tip rotation
    double u_tip[3];               // Tip curvature
};
// Total: 18*N + 15 for N actuator sets
```

Reduced6D **approximates** this by only tracking `u_0` and `v_0` at the **base**, losing:
- Per-coil state (v, w, p, R)
- Intermediate geometry
- Correct BVP → IVP dynamics flow

### Problem 2: Inconsistent with Reference

`DynamicsBVP → DYNSolverIVP` (TRUE legacy) is the **frozen reference** from CRMDYN_test.cpp regression tests.

Reduced6D was an experimental simplification that does **not match** the reference golden outputs.

### Problem 3: 9D Hybrid State Was Built on Reduced6D

The 9D hybrid state `[u_0[3], v_0[3], p_tip[3]]` extends Reduced6D by appending tip position. Since the base (Reduced6D) is not TRUE legacy, the 9D extension is also archived.

---

## Accessing Archived Code (Discouraged)

### If You Must Use Reduced6D

**For debugging/comparison ONLY** (not production):

```python
# Explicit import (will emit warnings)
from python.control.reduced6d.ilqr_6d import iLQRSolver as iLQRSolver6D
from python.control.reduced6d.legacy_state import LegacyState
from python.control.reduced6d.step_legacy_contract import step_legacy_contract

# Initialize 6D state
x0 = np.zeros(6)  # [u_0[3], v_0[3]]

# Use old 6D solver
solver = iLQRSolver6D(dt, L, params, horizon, Q=np.eye(6), R=np.eye(3), ...)
```

**WARNING**:
- Reduced6D is **NOT** covered by CI
- Reduced6D does **NOT** match CRMDYN_test.cpp reference
- Use only for legacy data replay or comparison

### Rebuild C++ Reduced6D (Not Recommended)

If you absolutely need the C++ Reduced6D bindings:

1. **Manually edit** `CMakeLists.txt`:
   ```cmake
   set(CRMSOURCES
       ...
       ${CMAKE_CURRENT_SOURCE_DIR}/src/reduced6d/CRM_DiffDynamics.cpp
       ${CMAKE_CURRENT_SOURCE_DIR}/src/reduced6d/CRM_DiffDynamics.hpp
   )
   ```

2. **Manually edit** `python/crm_bindings.cpp`:
   ```cpp
   #include "reduced6d/CRM_DiffDynamics.hpp"

   m.def("dynamics_forward", &py_dynamics_forward, ...);
   ```

3. **Rebuild**:
   ```bash
   cd build && cmake .. && make
   ```

**This is NOT supported** and will break zero-hit gates (see below).

---

## Zero-Hit Gates

After migration, the following patterns **must not** appear outside archived namespaces:

### Gate 1: No Reduced6D Symbols

```bash
# Search active code (MUST return zero hits)
rg -n "(CRM_DiffDynamics|dynamics_forward|STATE.*6\b|u_0\[|v_0\[)" \
   --glob '!build/**' \
   --glob '!**/archive/**' \
   --glob '!**/reduced6d/**' \
   src python
```

**Expected**: No matches (PASS)
**If ANY matches**: Migration incomplete (FAIL)

### Gate 2: FULLSTATE Is Default

```bash
# Verify default bindings (MUST show FULLSTATE)
rg "true_legacy_step|STATE_DIM_FULL|iLQRSolver" python/control/__init__.py
rg "CRM_TrueLegacyDynamics|true_legacy_step_forward" CMakeLists.txt python/crm_bindings.cpp
```

**Expected**: All commands show FULLSTATE functions
**If missing**: Default not set correctly (FAIL)

### Gate 3: Archive Not Imported

```bash
# Search for archive imports (MUST return zero hits)
rg "from.*archive\.|import.*archive" python --glob '!python/archive/**'
```

**Expected**: No matches (PASS)
**If ANY matches**: Accidental archive dependency (FAIL)

---

## File Inventory

### C++ (Archived)
- `src/reduced6d/CRM_DiffDynamics.cpp` (243 lines)
- `src/reduced6d/CRM_DiffDynamics.hpp` (71 lines)

### Python Controllers (Archived)
- `python/control/reduced6d/ilqr_6d.py` (648 lines)
- `python/control/reduced6d/mpc_6d.py` (260 lines)
- `python/control/reduced6d/lqr_6d.py` (248 lines)
- `python/control/reduced6d/hybrid_controller_6d.py` (480 lines)
- `python/control/reduced6d/hybrid_state_contract_9d.py` (9D state)
- `python/control/reduced6d/step_hybrid_legacy_contract_9d.py` (9D stepping)
- `python/control/reduced6d/legacy_state.py` (6D state adapter)
- `python/control/reduced6d/legacy_state_adapter.py` (6D packing)
- `python/control/reduced6d/step_legacy_contract.py` (6D step wrapper)

### Tests (Archived)
- `python/test/reduced6d/test_a1_legacy_state_adapter.py`
- `python/test/reduced6d/test_a1_step_legacy_contract_forward.py`
- `python/test/reduced6d/test_a2_implicit_vjp_gradcheck.py`
- `python/test/reduced6d/test_a3_batched_vjp.py`

**Total archived**: ~2000 lines of Python, ~300 lines of C++

---

## Rationale for NO Backward Compatibility

**Decision**: No shims, no aliases, imports **must break**.

**Why**:
1. **Correctness**: Reduced6D produces **different physics** than TRUE legacy
2. **Silent Bugs**: Aliasing Reduced6D → FULLSTATE would give wrong results without errors
3. **Clear Migration**: Force users to **intentionally** update to FULLSTATE dimensions
4. **Clean Codebase**: No technical debt from compatibility layers

**Example of what we AVOIDED**:
```python
# BAD (what we did NOT do)
from .reduced6d.ilqr_6d import iLQRSolver as iLQRSolver_LEGACY_COMPAT
# This would run wrong physics silently!
```

---

## Summary

- **Reduced6D (6D state)**: Archived to `src/reduced6d/`, `python/control/reduced6d/`
- **9D Hybrid State**: Archived to `python/control/reduced6d/*_9d.py`
- **Not TRUE Legacy**: Does not match CRMDYN_test.cpp reference
- **Opt-In Only**: Must explicitly import from `reduced6d/` namespace
- **No CI Coverage**: Tests in `python/test/reduced6d/` not run by default
- **No Shims**: Old imports fail with `ImportError` (by design)

Use **FULLSTATE (18·N+15)** with `true_legacy_step()` for all new work.
