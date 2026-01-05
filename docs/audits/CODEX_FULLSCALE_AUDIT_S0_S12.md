# CODEX FULLSCALE AUDIT S0–S12 (FULLSTATE)

## 1. Executive summary
- Authoritative forward chain confirmed: `DynamicsBVP → DYNSolverIVP` with residual `DYNNLEquation` in `src/CoilDynamics_Defs.cpp:427` and solver call `src/CoilDynamics_Defs.cpp:1157`.
- FULLSTATE VJP correctness is blocked by incorrect adjoint assembly and missing derivative paths (mL cotangents hard-zeroed; rotation sensitivities dropped).
- FULLSTATE linearization correctness is blocked by heuristic `J_yx` and a simplified `G_x` that does not match actual dynamics integration.
- Control Jacobians in VJP/linearize use a diagonal-turn-area assumption for `dTau/du`, which conflicts with the non-diagonal turn-area matrix used elsewhere.
- Dual residual derivatives include a divide-by-zero at the rotation norm (NaN risk at residual == 0).
- Python forward binding bypasses the BVP convergence guard and always advances with `DYNSolverIVP`, which can propagate unconverged BVP solutions.

### Runtime truth mapping (authoritative)
Call-chain diagram with file:line evidence:
```
python/crm_bindings.cpp:535 -> DynamicsBVP (src/CoilDynamics_Defs.cpp:1066)
  -> TrustRegionDogleg_dyn (src/CoilDynamics_Defs.cpp:1157)
     -> DYNNLEquation (src/CoilDynamics_Defs.cpp:427)
python/crm_bindings.cpp:545 -> DYNSolverIVP (src/CoilDynamics_Defs.cpp:1195)
```
C++ wrapper path with convergence guard:
```
src/CRM_TrueLegacyDynamics.cpp:118 -> DynamicsBVP
src/CRM_TrueLegacyDynamics.cpp:155 -> DYNSolverIVP
```

## 2. Definitive bug list (ranked by severity)

### B1 (Critical): Linearization A uses a simplified, non-physical `G_x`
Impact: `A` does not represent the actual forward dynamics and invalidates FULLSTATE linearization.
Evidence: `src/CRM_TrueLegacyDynamics.cpp:883`, `src/CRM_TrueLegacyDynamics.cpp:887`, `src/CRM_TrueLegacyDynamics.cpp:893`, `src/CRM_TrueLegacyDynamics.cpp:897`.

### B2 (Critical): `J_yx` is an empirical heuristic, not analytic
Impact: implicit terms for VJP and linearization use `dt * 0.1` scaling and sparse placeholders, producing incorrect gradients and A/B.
Evidence: `src/CRM_BVPJacobian.cpp:488`, `src/CRM_BVPJacobian.cpp:496`, `src/CRM_BVPJacobian.cpp:503`, `src/CRM_BVPJacobian.cpp:517`.

### B3 (Critical): Adjoint assembly zeroes all mL cotangents
Impact: adjoint solution is biased toward the nL block and can drive `grad_u` to zero even with nonzero `J_yu`.
Evidence: `src/CRM_TrueLegacyDynamics.cpp:313`, `src/CRM_TrueLegacyDynamics.cpp:317`, `src/CRM_TrueLegacyDynamics.cpp:601`, `src/CRM_TrueLegacyDynamics.cpp:605`.

### B4 (Critical): `dTau/du` uses a diagonal-turn-area assumption in VJP/linearize
Impact: incorrect control Jacobians in batched VJP and linearization; conflicts with the non-diagonal turn-area matrix used in BVP J_yu seeding.
Evidence: `src/CRM_TrueLegacyDynamics.cpp:670`, `src/CRM_TrueLegacyDynamics.cpp:676`, `src/CRM_TrueLegacyDynamics.cpp:928`, `src/CRM_TrueLegacyDynamics.cpp:933`.

### B5 (High): Rotation sensitivities are dropped in Dual paths
Impact: missing u→R→residual sensitivity in J_yu/J_yy and any derivative path that depends on rotation.
Evidence: `src/CRMDYN_Numerical_Integration.hpp:135`, `src/CoilDynamics_Defs_Templates.hpp:217`, `src/CoilDynamics_Defs_Templates.hpp:239`, `src/CoilDynamics_Defs_Templates2.hpp:410`, `src/CoilDynamics_Defs_Templates2.hpp:467`.

### B6 (High): Rotation residual derivative singular at zero
Impact: `val_sqrt == 0` yields divide-by-zero in Dual derivatives, causing NaNs in `J_yu`/`J_yy` exactly at convergence.
Evidence: `src/CoilDynamics_Defs_Templates2.hpp:523`, `src/CoilDynamics_Defs_Templates2.hpp:555`.

### B7 (High): Python forward binding advances even when BVP fails
Impact: `DYNSolverIVP` runs with unconverged BVP solutions, which can destabilize FULLSTATE forward correctness.
Evidence: `python/crm_bindings.cpp:535`, `python/crm_bindings.cpp:545` vs guard in `src/CRM_TrueLegacyDynamics.cpp:125`.

### B8 (Medium): Single-sample backward omits direct u→w pathway
Impact: `true_legacy_step_backward` uses only implicit `-(J_yu)^T λ`, diverging from batched path and linearization.
Evidence: `src/CRM_TrueLegacyDynamics.cpp:438`, `src/CRM_TrueLegacyDynamics.cpp:456`.

---

### Parameter consistency audit (forward vs backward)
Parameters used in both forward and backward paths (no mismatches unless noted):
- ActInertia (hollow-cylinder) forward: `src/CRM_TrueLegacyDynamics.cpp:64`; backward: `src/CRM_BVPJacobian.cpp:130`.
- Damping coefficients forward: `src/CRM_TrueLegacyDynamics.cpp:82`; backward: `src/CRM_BVPJacobian.cpp:148`.
- dt propagation into shooting params forward: `src/CRM_TrueLegacyDynamics.cpp:90`; backward: `src/CRM_BVPJacobian.cpp:156`.
- Scaling constants (IVALUE/RESIDUAL scales): `src/CRMDYN.hpp:37` through `src/CRMDYN.hpp:42`.

Mismatch flagged:
- Control Jacobian uses full turn-area matrix in BVP J_yu seeding (`src/CRM_BVPJacobian.cpp:327`, `src/CRM_BVPJacobian.cpp:360`) but linearization/batched VJP use a diagonal assumption (`src/CRM_TrueLegacyDynamics.cpp:676`, `src/CRM_TrueLegacyDynamics.cpp:933`). This is a correctness mismatch for `dTau/du`.

### Derivative correctness audit (Dual/VJP)
Dropped derivatives:
- Rotation state treated as double in flexible integrand: `src/CRMDYN_Numerical_Integration.hpp:135`.
- Rotation evolution uses `.val` and double-only Rodrigues update: `src/CoilDynamics_Defs_Templates.hpp:217`, `src/CoilDynamics_Defs_Templates.hpp:239`.
- `DYNNLEquation_T` extracts `R_L` via `.val` for Dual: `src/CoilDynamics_Defs_Templates2.hpp:410`, `src/CoilDynamics_Defs_Templates2.hpp:467`.

Clipping:
- No explicit derivative clipping found in current FULLSTATE Dual path.

Finite differences:
- BVP solver Jacobian uses forward differences (minpack hybrd): `numerical/minpack_DYN_Defs.cpp:111` and invoked by `src/CoilDynamics_Defs.cpp:1157`.

### Adjoint assembly audit
- `v_y` uses hardcoded zeros for mL components in both single and batched paths, preventing proper mL→lambda coupling: `src/CRM_TrueLegacyDynamics.cpp:313` and `src/CRM_TrueLegacyDynamics.cpp:601`.
- The solve uses `(J_yy)^T * lambda = v_y` via QR, which is mathematically consistent; the issue is the assembled `v_y` and the correctness of `J_yy` and `J_yu` inputs.

### Linearization A,B audit
- No FD/autograd is used in the implicit linearization path; it uses analytic IVP Jacobians and implicit solves.
- `A` uses simplified `G_x` (identity and `p_next = p + v*dt`) rather than the actual IVP dynamics: `src/CRM_TrueLegacyDynamics.cpp:883`.
- `B` uses a diagonal-turn-area torque derivative in `dTau/du`, inconsistent with the full turn-area matrix: `src/CRM_TrueLegacyDynamics.cpp:928`.
- `J_yx` used in implicit solves is heuristic (`dt * 0.1`), not derived from the true dynamics: `src/CRM_BVPJacobian.cpp:496`.

### Test integrity audit (FULLSTATE)
Meaningful FULLSTATE tests:
- Forward stability and determinism: `tests/test_fullstate_step_smoke.py:53`.
- Linearization shape and non-triviality: `tests/test_fullstate_linearize_shapes.py:53`.
- VJP gradcheck (uses FD reference): `tests/test_fullstate_vjp_gradcheck_u.py:66`.
- No reduced6d leakage check: `tests/test_sanity_gate_no_reduced6d.py:11`.

Tests that depend on FD/autograd or reduced6d:
- FD reference in VJP gradcheck: `tests/test_fullstate_vjp_gradcheck_u.py:83`.
- Torch autograd comparison in linearization test: `tests/test_fullstate_linearize_shapes.py:87`.

## 3. Which bugs are blockers for VJP gradcheck
- B2 (heuristic `J_yx`) — implicit gradients are numerically incorrect.
- B3 (mL cotangents zeroed) — adjoint solution is structurally wrong.
- B4 (incorrect `dTau/du`) — direct/implicit control sensitivities wrong.
- B5 (rotation derivatives dropped) — missing u→R→residual sensitivities.
- B6 (rotation residual derivative singularity) — NaNs at convergence.

## 4. Recommended minimal fix sequence (no code)
1. Fix adjoint assembly to include mL cotangent contributions (B3) and re-validate lambda structure.
2. Replace heuristic `J_yx` with a correct analytic or AD-computed version (B2), then re-check implicit terms.
3. Align `dTau/du` with the full turn-area matrix in all paths (B4) to match BVP J_yu seeding.
4. Restore rotation sensitivity propagation in Dual paths (B5) and guard the rotation residual derivative at zero (B6).
5. Update linearization `G_x` to match actual IVP dynamics (B1).
6. Add a BVP convergence guard to the Python forward binding (B7).

## 5. Proof that no reduced6d leaks exist in default imports
- Default control imports include only FULLSTATE modules and true legacy wrappers: `python/control/__init__.py:12` through `python/control/__init__.py:59`.
- Sanity gate test searches for reduced6d references in default stack code and excludes reduced6d/ explicitly: `tests/test_sanity_gate_no_reduced6d.py:11` and `tests/test_sanity_gate_no_reduced6d.py:26`.
