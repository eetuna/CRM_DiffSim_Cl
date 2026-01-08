# Sprint S15 Controllers E2E (Codex)

## Branch + SHAs
- Branch: s15-controllers-e2e-codex
- Start SHA: 10d0b91487ba9829541309d00d683ec0d1da5f9f
- End SHA: 10d0b91487ba9829541309d00d683ec0d1da5f9f

## Canonical entrypoints (FULLSTATE)
- FULLSTATE step (Python): python/control/true_legacy_step.py:23
- FULLSTATE linearize (Python): python/control/true_legacy_step.py:309
- VJP binding (C++): python/crm_bindings.cpp:833
- Linearize binding (C++): python/crm_bindings.cpp:1063
- State dimension helper: python/control/true_legacy_state_adapter.py:25

## Issue Identified
- Root cause: CRMIntegrand fcum interpolation used a negative upper index (iru) when lambda < 0; the lower bound was not clamped, causing out-of-bounds access and huge fcum values that propagated to NaNs in Dual linearization.
- First producer: CRMIntegrand_dyn_T fcum interpolation (negative iru) in src/CRMDYN_Numerical_Integration.hpp:49

## Minimal Fix
- Clamp `iru_f` lower bound to 0 in all CRM integrands to prevent negative indexing.
  - src/CRMDYN_Numerical_Integration.hpp:50
  - src/CRMDYN_Numerical_Integration.hpp:227
  - src/CRM_IVPSolver.cpp:599
  - src/CRM_IVPJacobian.cpp:538

## Tests Added (Controller Smoke)
- tests/test_controller_ilqr_fullstate_smoke.py
- tests/test_controller_mpc_fullstate_smoke.py
- tests/test_controller_lqr_fullstate_smoke.py
- tests/test_controller_hybrid_fullstate_smoke.py

## Commands + Outputs
```
rm -rf build_s15
cmake -S . -B build_s15 -DCMAKE_BUILD_TYPE=Release
cmake --build build_s15 -j2
```
- Build: success (warnings only: pragma once, non-void function return)

```
PYTHONPATH=build_s15:python:$PYTHONPATH python3 -m pytest -q tests/test_controller_*fullstate* -s
```
- Result: 5 passed, 1 skipped

```
PYTHONPATH=build_s15:python:$PYTHONPATH python3 -m pytest -q tests/test_fullstate_step_smoke.py -s
```
- Result: 5 passed

```
PYTHONPATH=build_s15:python:$PYTHONPATH python3 -m pytest -q tests/test_fullstate_vjp_gradcheck_u.py -s
```
- Result: 1 passed (pytest warning: test returns bool)

```
rg -n "torch\.autograd|autograd\b|finite.?diff|\bFD\b" src python tests
```
- Hits only in python/archive/, python/quarantine/, python/test/reduced6d/, and src/reduced6d/ (non-default paths)

## Notes
- FULLSTATE VJP gradcheck no longer produces NaNs; VJP and implicit linearization match to 1e-13 relative error.
