# CODEX Audit: Sprint S10 Dual Instability (Gradcheck Failure)

## Scope and Required Input
Grounded in `docs/audits/SPRINT_S10_DUAL_END_TO_END_COMPLETION.md` and current source.

## First Breakpoint(s): Dual.deriv Divergence/Collapse

### 1) First growth (explosion origin)
- **`src/CoilDynamics_Defs_Templates.hpp:173-175`**
  - `wdot[...] = residual_w[...] / T(actInertia[...]);`
  - This is the first point where Dual derivatives are amplified by **1 / actInertia** inside the coil dynamics integrand. Any small inertia or large residual magnifies sensitivities *before* explicit RK2/ABM4 propagation.

### 2) First collapse (NaN/underflow path for residual rotations)
- **`src/CoilDynamics_Defs_Templates2.hpp:523-525`** and **`src/CoilDynamics_Defs_Templates2.hpp:555-557`**
  - `deriv_sqrt = v_val[i].deriv / (2.0 * val_sqrt);`
  - When `val_sqrt == 0` (exactly matched rotations), this yields **0/0 → NaN**, collapsing derivatives in the rotation residual block. This is a separate NaN/underflow trigger downstream of the integrators.

## Minimal Call Chain (Dual path)
- `src/CRM_BVPJacobian.cpp:233-318` → seeds Dual and calls `DYNNLEquation_T<Dual>`
- `src/CoilDynamics_Defs_Templates2.hpp:279-458` → `DYNNLEquation_T` drives coil dynamics for each segment
- `src/CoilDynamics_Defs_Templates.hpp:451-521` → `CoilDynamics_T` orchestrates RK2/ABM4 substeps
- `src/CoilDynamics_Defs_Templates.hpp:285-354` → `RK2_coildyn_T` (first 3 steps)
- `src/CoilDynamics_Defs_Templates.hpp:359-446` → `ABM4_coildyn_T` (remaining steps)
- `src/CoilDynamics_Defs_Templates.hpp:60-181` → `CoilIntegrad_T` (stiff dynamics; first derivative amplification at `wdot`)

## Instability Classification (A–E)
- **Primary:**
  - **(A) True stiffness / ill-conditioning of tangent dynamics**
    - The angular dynamics divide by inertia (`1/actInertia`) and include stiff magnetic and damping terms; tangent equations inherit this stiffness.
  - **(B) Numerical instability from explicit integrator on tangent equations**
    - Explicit RK2/ABM4 propagates the stiff tangent system; sensitivity growth is amplified across explicit steps despite substepping.
- **Secondary (NaN collapse at residual stage):**
  - **(E) Sensitivity discontinuity / non-differentiability at zero norm**
    - Rotation residual uses `sqrt(v_val)` with `val_sqrt` in denominator; at `val_sqrt=0` the derivative is undefined and becomes NaN.
- **Not supported by evidence:**
  - **(C)** No remaining `.val` derivative stripping in the coil dynamics path that would explain explosion/underflow post-S10 fixes.
  - **(D)** Dual operator implementation in `src/CRM_BVPJacobian.hpp` is standard; no arithmetic bug indicated.

## Fix Direction (Production-Compliant)
Constraints honored: no FD, no torch.autograd, no solver-iteration differentiation; forward double physics unchanged.

### Ranked Options
1) **Implicit (A-stable) integration of the tangent system only**
   - **What:** Keep forward state integration explicit (unchanged); integrate sensitivities with an implicit method (BDF1/2, Radau IIA, or linearly implicit Rosenbrock) using the same forward trajectory.
   - **Pros:** Directly addresses stiffness; stabilizes Dual propagation without altering forward physics.
   - **Cons:** Requires Jacobian of dynamics wrt state (analytic or precomputed); more compute and implementation complexity.

2) **Linearly-implicit split: stiff angular block implicit, rest explicit**
   - **What:** Treat angular dynamics (`w`, `wdot`) implicitly in the tangent system; keep translational parts explicit (IMEX).
   - **Pros:** Smaller change surface; targets the stiffness source (`1/actInertia`) while preserving overall structure.
   - **Cons:** Needs careful block Jacobian derivation and coupling; still more complex than explicit.

3) **Exact per-step tangent update for rigid-body rotation/torque block**
   - **What:** Derive closed-form sensitivity update for the linearized rotational subsystem and compose with existing explicit step for remaining state.
   - **Pros:** Eliminates the unstable explicit propagation for the stiffest block; no change to forward values.
   - **Cons:** Requires analytic derivation and validation; may be brittle if model terms change.

4) **Guard non-differentiable rotation residuals at zero norm**
   - **What:** For Dual only, replace `v_val.deriv / (2*val_sqrt)` with a well-defined limit at `val_sqrt==0` (derivative set to 0) or a smooth epsilon-based analytic continuation.
   - **Pros:** Removes NaN collapse in rotation residuals; localized change; does not touch forward values.
   - **Cons:** Does not solve core stiffness/explosion in coil dynamics; must be paired with 1–3 for full stability.

## Is Implicit Sensitivity Integration Necessary?
- **Yes, for robust production stability in the current stiff regime.**
  - The first divergence occurs in the stiff angular acceleration (`wdot = residual_w / actInertia`) and is then *explicitly* amplified by RK2/ABM4. Substepping reduces step size but does not change the explicit stability region. This is consistent with S10 results (explosion at factor 2, collapse/NaN at 5–10).
  - A smaller targeted fix (Options 2 or 3) could work **only if** it neutralizes the stiff rotational block in the tangent equations. Without an A-stable or linearly-implicit treatment of that block, explicit tangent propagation remains unstable.

## Summary Verdict
The definitive root cause of the Sprint S10 VJP gradcheck failure is **stiff tangent dynamics amplified by explicit RK2/ABM4 sensitivity propagation**, with the earliest amplification at **`wdot = residual_w / actInertia`** in `CoilIntegrad_T`. A secondary NaN/underflow trigger exists in the rotation residual derivative at **`sqrt(v_val)`** when `val_sqrt==0`. Implicit (or linearly-implicit) tangent integration is the most reliable production-compliant fix; smaller targeted fixes are only viable if they stabilize the stiff rotational sensitivity block.
