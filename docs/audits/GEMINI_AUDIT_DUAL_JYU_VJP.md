# Audit Report: Dual Derivative Propagation in J_yu

**Date:** 2026-01-05
**Author:** Gemini Agent
**Subject:** Failure of Forward-Mode AD (Dual) for `J_yu = ∂r/∂u`

## 1. Executive Summary

The forward-mode automatic differentiation for `J_yu` fails because **derivatives are explicitly discarded** inside the templated numerical integrators (`RK2_coildyn_T` and `ABM4_coildyn_T`) and the SE(3) update function (`DYNSE3_TimeSpace_T`). This was implemented intentionally to suppress "numerical explosion" of derivatives, but it mathematically severs the dependence of the final state on the control input `u` (via `muhat`), resulting in incorrect or zero gradients.

## 2. Breakpoint Identification

The derivative path `u → muhat → CoilDynamics_T` is intact until it enters the integration loop. The chain is broken at the following exact locations:

### Breakpoint A: State Derivative Stripping in Integrators
Inside `RK2_coildyn_T` and `ABM4_coildyn_T`, the position ($p$) and rotation ($R$) states are cast to `double` via `.val`, discarding their accumulated derivatives before each integration step.

*   **File:** `src/CoilDynamics_Defs_Templates.hpp`
*   **Line 258** (RK2): `for (int i = 0; i < 3; ++i) p_n[i] = in_x_n[i+6].val;`
*   **Line 259** (RK2): `for (int i = 0; i < 9; ++i) R_n[i] = in_x_n[i+9].val;`
*   **Line 319** (ABM4): `for (int i = 0; i < 3; ++i) p_n[i] = in_x_n[i+6].val;`
*   **Line 320** (ABM4): `for (int i = 0; i < 9; ++i) R_n[i] = in_x_n[i+9].val;`

### Breakpoint B: Manual Derivative Reset in SE(3) Update
Inside `DYNSE3_TimeSpace_T`, the new position `out_p_np1` is constructed with a derivative that includes *only* the current step's velocity contribution, ignoring the history of position sensitivity.

*   **File:** `src/CoilDynamics_Defs_Templates.hpp`
*   **Line 223**: `out_p_np1[i] = T(p_n[i] + p_dot[i].val * h, p_dot[i].deriv * h);`
    *   **Analysis:** The constructor `T(val, deriv)` sets the derivative to `p_dot[i].deriv * h`. It fails to add `p_n[i].deriv`. Effectively: $\frac{dp_{n+1}}{du} \approx h \frac{dv_n}{du}$, instead of $\frac{dp_{n+1}}{du} = \frac{dp_n}{du} + h \frac{dv_n}{du}$.

### Breakpoint C: Derivative Clipping
A "safety net" in `DYNNLEquation_T` clips derivatives that exceed a threshold, further corrupting any partial gradients that might survive.

*   **File:** `src/CoilDynamics_Defs_Templates2.hpp`
*   **Line 479-484**: `if (std::abs(out_xdot[i].deriv) > MAX_DERIV) ...`

## 3. Causal Explanation

The implementation attempts to solve `∂r/∂u` using forward-mode AD on the existing explicit integration scheme (RK2/ABM4).
1.  **The Constraint:** The dynamics are stiff.
2.  **The Artifact:** Forward-mode AD on an explicit integrator for stiff systems leads to exponential growth of derivative values (explosion), rendering them numerically useless.
3.  **The "Fix":** To prevent this explosion, previous developers explicitly stripped derivatives from $p$ and $R$ at each time step (Breakpoints A & B) and clipped velocity derivatives (Breakpoint C).
4.  **The Result:** The computed Jacobian `J_yu` effectively measures "local sensitivity of the last time step" rather than the true sensitivity of the BVP residual to the controls, causing the VJP check to fail (either zero or incorrect values).

## 4. Candidate Fixes (Audit Only)

1.  **Remove Derivative Stripping (Correction):**
    *   Remove `.val` usage in `RK2_coildyn_T` and `ABM4_coildyn_T` for state variables.
    *   Update `DYNSE3_TimeSpace_T` to correctly accumulate derivatives: `out_p_np1 = p_n + p_dot * h` (using Dual arithmetic).
    *   *Risk:* Derivatives will likely explode as noted in the comments.

2.  **Switch to Implicit/Symplectic Integrator for Tangent Model:**
    *   The explosion is likely due to the instability of the explicit ABM4 scheme on the linearized equations. Using an implicit scheme (like BDF) or a geometric integrator that preserves the symplectic structure for the tangent dynamics would stabilize the derivatives without artificial clipping.

3.  **Check Physical Parameters/Conditioning:**
    *   If the explosion is physical (chaotic sensitivity), the problem formulation itself (time step, damping, stiffness) may need adjustment to be well-posed for optimization.
