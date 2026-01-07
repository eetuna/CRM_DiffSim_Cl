# SPRINT S14 Linearize NaN + Clean Build Fix

## Branch + SHA
- Branch: s14-codex-linearize-nan-cleanbuild-fix
- Start SHA: 10d0b91487ba9829541309d00d683ec0d1da5f9f
- End SHA: 10d0b91487ba9829541309d00d683ec0d1da5f9f

## Part A: Clean Release Build + Linearize Repro
### A1) Clean Release build
Commands:
- rm -rf build_clean
- cmake -S . -B build_clean -DCMAKE_BUILD_TYPE=Release
- cmake --build build_clean -j

First attempt (timed out, no compile/link error observed):
```
command timed out after 120840 milliseconds
[  4%] Building CXX object CMakeFiles/CRMTest.dir/main/CRMTest.cpp.o
...
[ 89%] Building CXX object CMakeFiles/CRMDYNTest.dir/src/CoilDynamics_Defs.cpp.o
/workspaces/CRM_DiffSim_Cl/src/CoilDynamics_Defs.cpp:1:9: warning: #pragma once in main file
    1 | #pragma once
      |         ^~~~
```

Rerun with fewer jobs (succeeds):
- cmake --build build_clean -j2

Output:
```
[ 31%] Built target CRMCPPLib
[ 33%] Building CXX object CMakeFiles/CRMTest.dir/main/CRM_KinematicsTestFunctions.cpp.o
Linking CXX executable CRMTest
Built target CRMTest
Consolidate compiler generated dependencies of target crm_diff_py
Built target crm_diff_py
Linking CXX executable CRMDYNTest
Built target CRMDYNTest
```

Result: clean Release build succeeds (no undefined symbols or link failures).

### A2) Linearization sanity/grad test
Canonical test used: tests/test_fullstate_linearize_shapes.py

Command:
- PYTHONPATH=build_clean:python:$PYTHONPATH python3 tests/test_fullstate_linearize_shapes.py

Output:
```
============================================================
Test 1: Linearization shapes
============================================================
Testing with n_act=1, state_dim=33, control_dim=3
A shape: (33, 33), expected: (33, 33)
B shape: (33, 3), expected: (33, 3)
PASS: Shapes are correct and matrices are non-trivial

============================================================
ALL TESTS PASSED
============================================================
```

Result: A,B finite in this test (no NaN/Inf).

## Part B: First NaN Producer Identification
- No NaN/Inf produced by the canonical FULLSTATE linearize test.
- No first-NaN producer identified (N/A).
- No CHECK_FINITE instrumentation added because no NaN reproduction occurred.

## Part C: Minimal Fix
- No fix required; no code changes made.

## Part D: Verification
1) Clean Release build from scratch: PASS (see Part A1)
2) Tests:

- PYTHONPATH=build_clean:python:$PYTHONPATH python3 tests/test_fullstate_step_smoke.py
```
============================================================
FULLSTATE Step Forward Smoke Test
============================================================
✓ Dimension test passed: x_next.shape=torch.Size([33]), tip_p.shape=torch.Size([3])
✓ Finiteness test passed: max|x_next|=2812.494737
✓ Determinism test passed: exact match on repeated calls
✓ Multi-step rollout test passed:
  - 5 steps completed successfully
  - Tip displacement: 0.163204 mm
  - Final tip position: [-0.07753513 -0.20185805 49.99939263]
  - All states finite
✓ Zero control stability test passed: 10 steps with zero control remained finite

============================================================
All tests passed!
============================================================
```

- PYTHONPATH=build_clean:python:$PYTHONPATH python3 tests/test_fullstate_vjp_gradcheck_u.py
```
============================================================
Test: VJP gradient check for control inputs
============================================================
VJP gradient:        [1.5151207  0.51166331 1.96418849]
Linearize gradient:  [1.5151207  0.51166331 1.96418849]
Relative error: 5.278147e-16
PASS: VJP gradients match implicit linearization

============================================================
ALL TESTS PASSED
============================================================
```

- PYTHONPATH=build_clean:python:$PYTHONPATH python3 tests/test_fullstate_linearize_shapes.py
```
============================================================
Test 1: Linearization shapes
============================================================
Testing with n_act=1, state_dim=33, control_dim=3
A shape: (33, 33), expected: (33, 33)
B shape: (33, 3), expected: (33, 3)
PASS: Shapes are correct and matrices are non-trivial

============================================================
ALL TESTS PASSED
============================================================
```

Result: clean build and required FULLSTATE tests pass; A,B finite in canonical linearize test.
