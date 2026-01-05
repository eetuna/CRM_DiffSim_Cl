# CODEX Audit: FULLSTATE Dual J_yu / VJP

## Scope and authoritative path
- **Runtime path (authoritative):** `DynamicsBVP → DYNSolverIVP → DYNNLEquation`.
- **Forward-mode AD path (FULLSTATE/J_yu):** `compute_bvp_jacobians_fmad → DYNNLEquation_T`.

### Runtime DYNNLEquation implementation actually used
- `DYNNLEquation(...)` is defined in `src/CoilDynamics_Defs.cpp:427` and is the concrete, non-templated implementation used by the runtime BVP solver (via `TrustRegionDogleg_dyn`, invoked from `DynamicsBVP` at `src/CoilDynamics_Defs.cpp:1156`).
- The forward-mode AD Jacobian path uses the templated `DYNNLEquation_T` in `src/CoilDynamics_Defs_Templates2.hpp:281`, called from `compute_bvp_jacobians_fmad` at `src/CRM_BVPJacobian.cpp:312`.

## Call-chain (FULLSTATE forward-mode AD)
```
compute_bvp_jacobians_fmad                 src/CRM_BVPJacobian.cpp:275-317
  └─ DYNNLEquation_T<Dual>                 src/CoilDynamics_Defs_Templates2.hpp:281
      ├─ CRMFlexible_IVP_Back_T            src/CoilDynamics_Defs_Templates2.hpp:20
      │   └─ CRMIntegrand_dyn_T            src/CRMDYN_Numerical_Integration.hpp:115
      └─ CoilDynamics_T                    src/CoilDynamics_Defs_Templates.hpp:445
          ├─ RK2_coildyn_T                 src/CoilDynamics_Defs_Templates.hpp:281
          ├─ ABM4_coildyn_T                src/CoilDynamics_Defs_Templates.hpp:348
          └─ CoilIntegrad_T                src/CoilDynamics_Defs_Templates.hpp:46
```

## Findings: derivative loss (drop to zero)
1) **Rotation derivatives are explicitly discarded in the flexible-segment integrand path.**
   - `CRMIntegrand_dyn_T` aliases `R` as a **double** array (`auto& R = in_x._R;  // R stays double`), so any Dual sensitivity of rotation is never propagated into `udot` or `fcum` computations. This zeroes all derivative contributions that should flow through `R`.
   - Location: `src/CRMDYN_Numerical_Integration.hpp:135-137`.
   - First observable consequence: residual rotation components in `DYNNLEquation_T` use `R_f`/`R_L` as **double** and are wrapped in `T(...)` only at the end, yielding `deriv = 0` for rotation residuals (see `src/CoilDynamics_Defs_Templates2.hpp:500-516` and `src/CoilDynamics_Defs_Templates2.hpp:552-558`).

2) **Position derivatives are dropped at the flexible-segment boundary output.**
   - `CRMFlexible_IVP_Back_T` returns `out_p` as **double**, explicitly extracting `xf_statevec._p[j].val`. This forces all position derivatives to zero when returning to `DYNNLEquation_T`.
   - Location: `src/CoilDynamics_Defs_Templates2.hpp:255-262`.
   - This is the first hard truncation of Dual derivatives on the flexible path (everything in `xf_statevec._p[j].deriv` is discarded).

3) **Coil dynamics discard stored position/rotation derivatives at each integration step.**
   - In both `RK2_coildyn_T` and `ABM4_coildyn_T`, the incoming coil state’s position and rotation are read with `.val` into doubles (`p_n`, `R_n`), zeroing any accumulated derivatives in those states.
   - Locations: `src/CoilDynamics_Defs_Templates.hpp:291-292` (RK2), `src/CoilDynamics_Defs_Templates.hpp:388-389` (ABM4).
   - This makes the coil state’s `p` and `R` derivative effectively reset every step, which causes downstream residual derivatives to collapse toward zero.

4) **Flexible-to-coil coupling explicitly strips position derivative before integration.**
   - `DYNNLEquation_T` builds `p_L_val` by extracting `.val` from `p_L` and passes it into `CRMFlexible_IVP_Back_T`, cutting the derivative path from coil state into flexible integration.
   - Location: `src/CoilDynamics_Defs_Templates2.hpp:431-439`.

## Findings: derivative explosion (~1e90..1e98)
1) **Stiff explicit integrator in `CoilDynamics_T` can produce huge derivatives before the manual clamp.**
   - `CoilIntegrad_T` computes angular acceleration with division by diagonal inertia (`wdot = residual_w / actInertia[...]`), which can amplify Dual derivatives sharply when inertia is small or residual terms are large.
   - Location: `src/CoilDynamics_Defs_Templates.hpp:163-166`.
   - These derivatives are then propagated through explicit ABM4 predictor/corrector steps (`x_np1_hat` / `out_x_np1`), which is known to be unstable for stiff dynamics and can blow sensitivities up by orders of magnitude.
   - Locations: `src/CoilDynamics_Defs_Templates.hpp:403-428`.

2) **The code itself acknowledges the explosion and clamps derivatives post-hoc.**
   - Immediately after `CoilDynamics_T` returns, derivatives in `out_xdot` are clipped to `±1e3`, with an in-code comment stating the explicit integrator causes exponential growth.
   - Location: `src/CoilDynamics_Defs_Templates2.hpp:480-489`.
   - This is strong evidence that derivatives are **already exploding** inside `CoilDynamics_T` before clamping; without the clamp, magnitudes can reach the reported 1e90..1e98 range in stiff regimes.

## Root-cause classification
- **D) Multiple issues**
  - **(A) Templating break / value extraction:** `.val` extraction and double-only APIs drop derivatives in multiple places (flexible segment outputs, coil state inputs, rotation paths).
  - **(C) Mathematical stiffness/sensitivity:** Explicit multistep integrator with stiff coil dynamics amplifies sensitivities; derivatives explode before being clipped.

## First derivative loss and first derivative explosion (code evidence)
- **First derivative loss (hard truncation):**
  - `out_p[j] = xf_statevec._p[j].val;` drops all position derivatives exiting the flexible integrator.
  - Location: `src/CoilDynamics_Defs_Templates2.hpp:255-262`.

- **First derivative explosion (amplification point):**
  - `wdot[...] = residual_w[...] / actInertia[...]` amplifies derivatives in `CoilIntegrad_T`, which are then propagated by the explicit ABM4 steps.
  - Location: `src/CoilDynamics_Defs_Templates.hpp:163-166` (amplification), `src/CoilDynamics_Defs_Templates.hpp:403-428` (explicit multi-step propagation).
  - The immediate evidence of explosion is the subsequent clamp at `src/CoilDynamics_Defs_Templates2.hpp:480-489`.

## Next fix candidates (audit-only, no patches)
1) **Make flexible-segment outputs Dual-safe.** Change `CRMFlexible_IVP_Back_T` to return `out_p` as `T` (or a parallel `out_p_T`) so derivatives in `_p` are not discarded (`src/CoilDynamics_Defs_Templates2.hpp:255-262`).
2) **Remove `.val` extraction for coil state in integrators.** Keep `p_n`/`R_n` in Dual form (or propagate their derivatives explicitly) in `RK2_coildyn_T` and `ABM4_coildyn_T` (`src/CoilDynamics_Defs_Templates.hpp:291-292`, `src/CoilDynamics_Defs_Templates.hpp:388-389`).
3) **Propagate rotation derivatives or explicitly zero them by design.** Currently `R` is treated as double in `CRMIntegrand_dyn_T` (`src/CRMDYN_Numerical_Integration.hpp:135-137`) and `DYNSE3_TimeSpace_T` (`src/CoilDynamics_Defs_Templates.hpp:232-236`), causing rotation residual derivatives to be identically zero; decide whether to carry or formally drop these derivatives.
4) **Stabilize the coil sensitivity integration.** The explicit ABM4 path is flagged as unstable for derivatives and requires clipping; consider a sensitivity-stable integrator or implicit sensitivity propagation instead of post-hoc clamp (`src/CoilDynamics_Defs_Templates.hpp:403-428`, `src/CoilDynamics_Defs_Templates2.hpp:480-489`).
5) **Replace heuristic derivative injection for `p_f`.** The line `p_f[i] = T(p_f_val[i], Params.DELTA_T * out_xdot[i].deriv)` overwrites any physically correct derivative from flexible integration and uses a quasi-static approximation; this should be replaced with a true Dual propagation path (`src/CoilDynamics_Defs_Templates2.hpp:441-450`).

