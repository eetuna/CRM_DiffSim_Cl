# SPRINT_S14_CODEX_NAN_FIX_AND_COMPLETION

Branch: s14-codex-plan-impl-codex-fix-nan
Start SHA: 2ddcdff021ae7f93ab4d4395f5120d04a11ae533
End SHA: 2ddcdff021ae7f93ab4d4395f5120d04a11ae533

## Reproduction

```
cmake --build build -j
pytest -q tests/test_bvp_jacobian_sanity.py
```

Observed: J_yx contained NaNs in `tests/test_bvp_jacobian_sanity.py`.

## First-NaN localization

Added finite checks in the single-path forward-mode AD (J_yx) path and traced the first non-finite to coil dynamics output:

- First non-finite: `out_x_coil` in `DYNNLEquation_YY_T` (templated BVP residual path)
- Root data issue: `actInertia` and `damping` in `CRMIVPCoreParams` were uninitialized when copied

## Fix

Copy missing dynamics fields in the `CRMIVPCoreParams` copy constructor so the forward-mode AD path sees initialized dynamics parameters.

- File: `src/CRM_IVPSolver.cpp`
- Change: copy `v_L_pre`, `w_L_pre`, `p_pre`, `R_pre`, `actInertia`, `damping`, `m_L`, `n_L`, and `DELTA_T` from the source object in `CRMIVPCoreParams::CRMIVPCoreParams(const CRMIVPCoreParams&)`.

This removes NaNs in coil dynamics and stabilizes J_yx.

## Tests

```
pytest -q tests/test_bvp_jacobian_sanity.py
```
Result: PASS

```
pytest -q tests/test_fullstate_vjp_parity.py
```
Result: PASS

```
pytest -q tests/test_fullstate_vjp_gradcheck_u.py
```
Result: PASS (PytestReturnNotNoneWarning from test body returning bool)

```
pytest -q tests/test_fullstate_step_smoke.py
```
Result: PASS

```
pytest -q tests/test_fullstate_linearize_shapes.py
```
Result: PASS (PytestReturnNotNoneWarning from test body returning bool)

```
pytest -q tests/test_fullstate_linearize_consistency.py
```
Result: PASS

## Grep proofs

```
rg -n "finite.?diff|\bFD\b|epsilon" src/ python/ tests/
```
Matches found (existing references outside FULLSTATE path), including:
- `python/control/true_legacy_step.py`
- `python/control/true_legacy_step_autograd.py`
- `python/control/reduced6d/step_hybrid_legacy_contract_9d.py`
- `src/reduced6d/CRM_DiffDynamics.cpp`
- `tests/test_gradient_decomposition.py`
- `python/archive/*`

```
rg -n "autograd|torch\.autograd" src/ python/ tests/
```
Matches found (existing references outside FULLSTATE path), including:
- `python/control/true_legacy_step_autograd.py`
- `python/control/ilqr.py`
- `python/crm_dynamics_torch.py`
- `python/archive/*`

## Notes

- NaN bug fixed at the first source: uninitialized dynamics parameters during `CRMIVPCoreParams` copy.
- FULLSTATE tests required by S14 now pass.
