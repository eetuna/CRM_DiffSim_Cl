# CP3: Model-Based Control Design — iLQR / MPC Integration

**Status**: DESIGN PHASE
**Date**: 2025-12-31
**Author**: Claude Code (Anthropic)
**Prerequisites**: CP2.1-CP2.6 (Differentiable Dynamics) COMPLETE

---

## Executive Summary

CP3 implements **model-based optimal control** for continuum catheter robots using the validated differentiable dynamics primitive from CP2. This checkpoint enables **trajectory optimization** and **receding-horizon control** for target-reaching and path-following tasks.

**What CP3 provides:**
1. **iLQR solver** for local trajectory optimization around nominal paths
2. **Receding-horizon MPC** for real-time control with replanning
3. **Task-space objectives** using tip position p_tip as primary feedback
4. **Warm-start infrastructure** for temporal coherence in sequential replanning
5. **CI smoke tests** ensuring controller stability and convergence

**What CP3 does NOT do:**
- ❌ Learning (no θ gradients, no neural networks)
- ❌ Global planning (local optimization around initial guess only)
- ❌ Contact handling (free-space navigation only)
- ❌ Multi-robot coordination
- ❌ Parameter estimation

**Design philosophy**: Build on CP2's provably correct gradients to implement classical model-based control with minimal complexity. Validate each component independently before integration.

---

## Why CP2 Makes CP3 Safe and Tractable

CP2's validated infrastructure provides **four critical guarantees** that make CP3 implementation low-risk:

### 1. Gradient Correctness (CP2.2 Validation)
**CP2 guarantee**: Analytical gradients match finite differences with rel_err < 1e-4 at all tested operating points.

**CP3 benefit**:
- iLQR backward pass can **trust** Jacobians ∂x_next/∂x_t and ∂x_next/∂u_t without re-validation
- No risk of incorrect gradients leading to divergent trajectories
- If optimization fails, we know it's due to problem structure (local minimum, poor initialization), not gradient bugs

**Safety implication**: We can implement iLQR's Riccati recursion **exactly as written in textbooks** without defensive programming around gradient accuracy.

### 2. Multi-Step Stability (CP2.5 Validation)
**CP2 guarantee**: 20-step rollouts remain numerically stable (||x|| < 1e3, all states finite) with differentiable backprop.

**CP3 benefit**:
- MPC horizon of T=10-20 steps is **within validated range**
- No risk of forward simulation exploding during trajectory optimization
- Backward Euler's unconditional stability prevents oscillatory instabilities

**Safety implication**: If MPC rollout diverges, it's due to **infeasible control constraints** or **physics limitations**, not numerical artifacts. This makes debugging tractable.

### 3. PyTorch Integration (CP2.4 Validation)
**CP2 guarantee**: `torch.autograd.gradcheck()` passes, enabling seamless integration with PyTorch-based optimizers.

**CP3 benefit**:
- Can use **off-the-shelf optimizers** (L-BFGS, Adam) for shooting methods without implementing custom solvers
- Hybrid approaches (iLQR for local, gradient descent for global search) work out-of-the-box
- Easy prototyping of cost functions with automatic differentiation

**Safety implication**: Initial CP3 implementation can use **shooting + PyTorch autograd** as a sanity check before implementing full iLQR. If they disagree, iLQR has a bug.

### 4. API Simplicity (No Hidden State)
**CP2 design**: State x=[u_0, v_0] is **complete** — no hidden internal state, no warmstart artifacts.

**CP3 benefit**:
- MPC replanning is **stateless** — each replan starts fresh with current measured state
- No risk of stale cached equilibrium causing drift
- Parallel trajectory evaluations (e.g., ensemble planning) are trivial

**Safety implication**: Controller is **debuggable** — if MPC fails, we can reproduce exact conditions by logging x_t, u_t, and replaying.

### Why This Matters for Risk Management

Traditional iLQR/MPC implementations often fail due to:
- **Gradient errors** → CP2.2 FD validation eliminates this
- **Forward instability** → CP2.5 rollout tests eliminate this
- **Implementation complexity** → CP2.4 PyTorch wrapper eliminates custom autodiff
- **Hidden state bugs** → CP2 API design eliminates this

**Result**: CP3 implementation risk is **low**. If it fails, the failure is **understood** (local minimum, constraint violation) rather than mysterious (numerical bug, gradient sign error).

---

## State and Control Definitions

### State Vector x ∈ ℝ⁶
CP3 **directly reuses** CP2's state representation without modification:
```
x = [u_0, v_0]
```

Where:
- **u_0 ∈ ℝ³**: Base curvature (1/mm)
  - Controls bending at the catheter base
  - Physically interpretable: u_0 ≈ [bend_x, bend_y, twist]

- **v_0 ∈ ℝ³**: Base curvature velocity (1/mm/s)
  - Rate of change of curvature
  - Required for second-order dynamics (inertia, damping)

**Why this choice**:
- ✅ Proven stable in CP2.5 multi-step rollouts
- ✅ Direct observable from equilibrium solver
- ✅ No coordinate transformations needed
- ✅ Gradient flow validated end-to-end

**What we do NOT use**:
- ❌ Joint angles (catheter is continuum, not articulated)
- ❌ Tip position directly (p_tip is **observable**, not state)
- ❌ Augmented state with cost-to-go (value function stays separate)

### Control Input u ∈ ℝ³
CP3 uses the same actuation model as CP2:
```
u_t = [I_1, I_2, I_3]  (Amperes)
```

Where:
- **I_i**: Current in actuation channel i (e.g., tendon tension or electromagnetic coil)
- **Physical constraints**: |I_i| ≤ I_max (typically 0.5-1.0 A)
- **Coupling**: u affects state through equilibrium-derived matrices K_tip(u), J_u_zc(u)

**Control constraints** (enforced in MPC):
```
u_min ≤ u_t ≤ u_max
||Δu_t|| ≤ Δu_max     (rate limit for actuator dynamics)
```

Default values:
- u_min = [-0.5, -0.5, -0.5] A
- u_max = [0.5, 0.5, 0.5] A
- Δu_max = 0.2 A per timestep (prevents bang-bang)

### Observables (Task Space)
While x is the **state**, control objectives are specified in **task space**:

**Primary observable**: Tip position p_tip ∈ ℝ³ (mm)
- Computed from x via equilibrium solver: `p_tip = equilibrium_forward(u_0, u_t, L_inserted).p_tip`
- **Gradients available**: ∂p_tip/∂u_0 and ∂p_tip/∂u_t cached in DynamicsStepResult
- This is where we specify goals: "reach p_target" or "follow trajectory p_ref(t)"

**Secondary observables** (available but not primary):
- Tip curvature u_tip ∈ ℝ³ (1/mm) — for orientation control
- Tip velocity ṗ_tip ∈ ℝ³ (mm/s) — for smooth motion objectives

**Key design decision**: Cost functions are **task-space** (p_tip) but optimization is **state-space** (x). This requires:
- Forward mapping: x → p_tip (via equilibrium)
- Gradient mapping: ∂L/∂p_tip → ∂L/∂x (via chain rule with cached Jacobians)

---

## Cost Function Structure

CP3 uses a **quadratic running cost** + **quadratic terminal cost** formulation compatible with iLQR.

### Running Cost (Stage Cost)
At each timestep t ∈ {0, ..., T-1}:
```
L(x_t, u_t, t) = L_state(x_t, t) + L_control(u_t) + L_task(x_t, t)
```

**1. State regularization** (optional, typically small):
```
L_state(x_t, t) = (1/2) ||x_t - x_eq||²_Q
```
- x_eq: Equilibrium state (typically x_eq = [0, 0, 0, 0, 0, 0] for neutral catheter)
- Q: State cost matrix (diagonal, small weights ~1e-3 to prevent state drift)

**2. Control effort** (always present):
```
L_control(u_t) = (1/2) ||u_t||²_R + (1/2) ||u_t - u_{t-1}||²_R_Δ
```
- R: Control effort penalty (diagonal, e.g., R = diag([1, 1, 1]))
- R_Δ: Control rate penalty (penalizes jerky actuation, e.g., R_Δ = 10·R)
- First term: Minimize energy
- Second term: Minimize acceleration (smoothness)

**3. Task-space tracking** (primary objective):
```
L_task(x_t, t) = (1/2) ||p_tip(x_t, u_t) - p_ref(t)||²_W
```
- p_tip(x_t, u_t): Tip position from equilibrium (observable)
- p_ref(t): Reference trajectory in task space (3D position at time t)
- W: Task-space weight matrix (diagonal, large e.g., W = 1000·I to prioritize tracking)

**Gradient computation**:
Since p_tip depends on both x_t and u_t via equilibrium:
```
∂L_task/∂x_t = (∂p_tip/∂x_t)ᵀ W (p_tip - p_ref)
∂L_task/∂u_t = (∂p_tip/∂u_t)ᵀ W (p_tip - p_ref)
```
where ∂p_tip/∂x_t is computed via chain rule from ∂p_tip/∂u_0 (cached in DynamicsStepResult).

### Terminal Cost
At final timestep T:
```
L_final(x_T) = (1/2) ||p_tip(x_T, u_T) - p_target||²_W_f
```
- p_target: Desired final tip position (goal state in task space)
- W_f: Terminal weight (typically W_f = 10·W to ensure convergence)

**Why terminal cost**:
- Ensures trajectory **ends** at goal, not just passes through
- Provides "pull" for iLQR backward pass to propagate gradients
- Required for finite-horizon optimality

### Cost Function Jacobians and Hessians (for iLQR)
iLQR requires second-order expansion around nominal trajectory. For quadratic costs:

**Running cost**:
```
∂L/∂x = Q(x - x_eq) + (∂p_tip/∂x)ᵀ W (p_tip - p_ref)
∂L/∂u = Ru + R_Δ(u - u_prev) + (∂p_tip/∂u)ᵀ W (p_tip - p_ref)

∂²L/∂x² = Q + (∂p_tip/∂x)ᵀ W (∂p_tip/∂x)     (Gauss-Newton approx)
∂²L/∂u² = R + R_Δ + (∂p_tip/∂u)ᵀ W (∂p_tip/∂u)
∂²L/∂u∂x = (∂p_tip/∂u)ᵀ W (∂p_tip/∂x)
```

**Gauss-Newton approximation**: We drop the second-derivative term `W (∂²p_tip/∂x²) (p_tip - p_ref)` to avoid computing Hessians of equilibrium solver. This is standard practice in iLQR and works well when tracking error is small.

### Constraint Handling (Box Constraints on Control)
For box constraints u_min ≤ u_t ≤ u_max:
- **iLQR**: Use **clamping** after each control update in forward pass
  ```python
  u_new = np.clip(u_nominal + k + K @ dx, u_min, u_max)
  ```
- **MPC shooting**: Use **projected gradient descent** or L-BFGS-B (box-constrained optimizer)

For general constraints (state constraints, obstacle avoidance):
- **Not in CP3.1-3.3** (deferred to future work)
- **CP3.4** may add simple barrier functions if needed for safety

---

## Linearization Strategy Using dynamics_backward

iLQR requires linearization of dynamics around a nominal trajectory:
```
x_{t+1} ≈ f(x̄_t, ū_t) + A_t(x_t - x̄_t) + B_t(u_t - ū_t)
```

Where:
- A_t = ∂f/∂x_t = ∂x_{t+1}/∂x_t
- B_t = ∂f/∂u_t = ∂x_{t+1}/∂u_t

### Jacobian Extraction Strategy

**Option 1: VJP-based extraction (CP3.1-3.2)**
Use `dynamics_backward` with canonical basis vectors:
```python
def extract_jacobians(x_t, u_t, dt, L_inserted, params_dict):
    # Forward pass
    result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)

    # Extract A = ∂x_next/∂x_t via 6 VJP calls
    A = np.zeros((6, 6))
    for i in range(6):
        e_i = np.zeros(6)
        e_i[i] = 1.0
        bwd_result = crm_diff_py.dynamics_backward(result, e_i, params_dict)
        A[:, i] = bwd_result['grad_x_t']  # i-th column of A^T → i-th row of A

    # Extract B = ∂x_next/∂u_t via 3 VJP calls
    B = np.zeros((6, 3))
    for j in range(3):
        e_j = np.zeros(6)
        # Need to map u_t direction to x_next direction — use adjoint
        # Actually, we need JVP here, not VJP!
        # See Option 2 below for correct approach.

    return A.T, B.T  # Transpose since VJP gives rows of J^T
```

**Issue with Option 1**: VJP gives us (J^T @ v), but we need columns of J. We'd need 6+3=9 backward calls, and the indexing is confusing.

**Option 2: PyTorch JVP (recommended for CP3.1-3.2)**
Use `torch.autograd.functional.jacobian` which handles batching:
```python
def extract_jacobians_torch(x_t, u_t, dt, L_inserted, params_dict):
    from crm_dynamics_torch import dynamics_step
    import torch

    x_t_torch = torch.tensor(x_t, dtype=torch.float64, requires_grad=True)
    u_t_torch = torch.tensor(u_t, dtype=torch.float64, requires_grad=True)

    def dynamics_fn(x, u):
        return dynamics_step(x, u, dt, L_inserted, params_dict)

    # Extract A = ∂x_next/∂x_t
    A = torch.autograd.functional.jacobian(
        lambda x: dynamics_fn(x, u_t_torch), x_t_torch
    ).numpy()

    # Extract B = ∂x_next/∂u_t
    B = torch.autograd.functional.jacobian(
        lambda u: dynamics_fn(x_t_torch, u), u_t_torch
    ).numpy()

    return A, B  # Shape: (6,6) and (6,3)
```

**Cost**: 6+3=9 calls to `dynamics_backward` internally (PyTorch uses VJP with basis vectors).

**Option 3: Direct C++ Jacobian assembly (CP3.3 optimization)**
For production MPC, avoid Python/PyTorch overhead by exposing Jacobians directly:
```cpp
// New C++ function (to be added in CP3.3)
void dynamics_linearize(
    const double x_t[6],
    const double u_t[3],
    double dt,
    double L_inserted,
    const CRMForwardKinematicsData& params,
    double A[36],  // Output: ∂x_next/∂x_t (6×6, row-major)
    double B[18]   // Output: ∂x_next/∂u_t (6×3, row-major)
);
```

This would:
1. Call `dynamics_forward` once
2. Internally call `dynamics_backward` 6+3 times with basis vectors
3. Assemble Jacobians directly in C++ (no numpy/torch overhead)

**Performance comparison** (estimated):
- Option 2 (PyTorch): ~10ms per linearization (Python overhead)
- Option 3 (C++): ~3ms per linearization (direct assembly)

**CP3 Plan**:
- **CP3.1-3.2**: Use Option 2 (simple, leverages existing PyTorch wrapper)
- **CP3.3**: Add Option 3 if MPC runtime is bottleneck (likely not needed for T=10-20)

### Dealing with Linearization Error

**Challenge**: Dynamics are nonlinear, linearization is only valid locally.

**iLQR mitigation**:
1. **Line search in forward pass**: Scale control update by α ∈ (0, 1] to ensure cost reduction
   ```python
   for alpha in [1.0, 0.5, 0.25, 0.1]:
       cost_new = evaluate_trajectory(x_nom, u_nom + alpha * du)
       if cost_new < cost_old:
           break
   ```

2. **Trust region**: Limit ||du|| via regularization λ in backward pass
   ```python
   Q_uu_reg = Q_uu + lambda * I
   ```

3. **Re-linearization**: Update nominal trajectory every iteration (standard iLQR)

**MPC mitigation**:
- Short horizon (T=10-20) limits accumulation of linearization error
- Replanning at every timestep corrects for model mismatch
- Feedback loop inherently robust to small errors

**Failure mode**: If linearization is too poor (e.g., large Δx, Δu), iLQR may diverge. Solution: Better initialization (see warm-start section).

---

## iLQR vs MPC: Choice and Justification

Both iLQR and MPC rely on the same dynamics linearization from CP2. The difference is **how they're used**:

### iLQR (Iterative Linear Quadratic Regulator)
**What it is**: Offline trajectory optimization via second-order expansion and Riccati recursion.

**Algorithm**:
1. Initialize nominal trajectory (x̄, ū) from initial guess
2. **Backward pass**: Compute optimal feedback gains K_t and feedforward terms k_t via Riccati recursion
3. **Forward pass**: Roll out new trajectory using linearized control law u_t = ū_t + k_t + K_t(x_t - x̄_t)
4. **Line search**: Scale updates by α to ensure cost reduction
5. Repeat until convergence (||du|| < ε or max_iters reached)

**Pros**:
- ✅ Globally optimal for LQR (if problem were truly linear-quadratic)
- ✅ Second-order convergence near solution (faster than gradient descent)
- ✅ Produces feedforward + feedback policy (robust to small disturbances)
- ✅ Well-studied, mature algorithm (textbook implementations available)

**Cons**:
- ❌ Offline only (too slow for real-time replanning in current implementation)
- ❌ Requires good initialization (local optimizer, can get stuck)
- ❌ Hessian inversion can be numerically unstable if Q_uu is near-singular

**Use cases**:
- Initial trajectory generation for MPC warm-start
- High-quality path planning for known start/goal pairs
- Benchmarking optimal control performance

### MPC (Model Predictive Control)
**What it is**: Receding-horizon online control — solve optimization at every timestep, execute first control, replan.

**Algorithm**:
1. Measure current state x_0
2. Solve trajectory optimization over horizon T (using iLQR or shooting)
3. Execute u_0^* (first control from optimized sequence)
4. Advance to next timestep, measure x_1
5. Shift horizon forward, replan from x_1

**Pros**:
- ✅ Real-time control (can replan every timestep)
- ✅ Handles disturbances and model mismatch via replanning
- ✅ Constraint satisfaction at every timestep (if enforced in optimization)
- ✅ Industry-standard for control of complex systems (robotics, autonomous vehicles)

**Cons**:
- ❌ Requires fast optimization (typically ~10-100ms for real-time at 10-100Hz)
- ❌ Warm-start critical for performance (cold-start infeasible in real-time)
- ❌ Theoretical guarantees (stability, convergence) require careful tuning

**Use cases**:
- Real-time catheter control during procedures
- Tracking time-varying references (moving targets)
- Handling external disturbances (patient motion, contact)

### CP3 Recommendation: Implement Both, Use Sequentially

**Phase 1 (CP3.1-3.2): iLQR**
- Easier to implement (no real-time constraints, can use PyTorch optimizers as fallback)
- Validates linearization correctness (if iLQR converges, dynamics gradients are usable)
- Generates high-quality nominal trajectories for MPC warm-start

**Phase 2 (CP3.3): MPC using iLQR as subroutine**
- Use iLQR as the optimization solver inside MPC's replanning loop
- Warm-start iLQR with shifted solution from previous timestep
- Execute first control, measure next state, repeat

**Hybrid approach**:
```
Offline (before procedure):
  - Use iLQR to generate nominal trajectory for target reaching

Online (during procedure):
  - At each timestep:
    1. Measure x_current
    2. Warm-start MPC with shifted iLQR solution
    3. Run 1-3 iLQR iterations (not to convergence, just refine)
    4. Execute u_0^*
    5. Advance
```

**Why this works**:
- iLQR provides high-quality initial guess (near-optimal)
- MPC corrects for disturbances without full re-optimization
- 1-3 iLQR iterations (~30-100ms) is tractable for 10Hz control

---

## Implementation Parameters

### Timestep and Discretization
- **dt = 0.01 s** (10 ms)
  - Matches CP2 validation tests
  - Fast enough for dynamics (catheter has inertia ~100Hz characteristic frequency)
  - Compatible with Backward Euler's stability region

- **Control frequency**: 10-100 Hz
  - If dt=0.01s → 100 Hz (aspirational for real-time MPC)
  - If dt=0.1s → 10 Hz (more realistic for initial implementation)

### Horizon Length
- **iLQR horizon**: T = 20 steps (0.2s physical time if dt=0.01s)
  - Matches CP2.5 validation
  - Long enough for meaningful planning
  - Short enough to avoid linearization error accumulation

- **MPC horizon**: T = 10-20 steps
  - Shorter horizon → faster optimization
  - Receding-horizon compensates for shorter lookahead
  - Tuning parameter (start with T=10, increase if needed)

### iLQR Convergence Criteria
- **Max iterations**: 50
- **Cost reduction tolerance**: Δcost < 1e-6
- **Control update tolerance**: ||du|| < 1e-4
- **Line search parameters**: α ∈ {1.0, 0.5, 0.25, 0.1, 0.05}
- **Regularization schedule**: λ ∈ [1e-6, 1e3], adapt based on Q_uu conditioning

### MPC Warm-Start Strategy
**Initial MPC call** (cold start):
- Use straight-line trajectory in state space: x_0 → x_goal
- Or use kinematic planner output (if available)
- Run full iLQR to convergence (~50 iterations)

**Subsequent MPC calls** (warm start):
- Shift previous solution: [u_1^*, ..., u_{T-1}^*, u_{T-1}^*] (repeat last control)
- Use as initial guess for next iLQR
- Run only 1-5 iLQR iterations (refinement, not full convergence)

**Warm-start validation** (CP3.3 test):
- Verify that warm-started MPC converges faster than cold-start (1-5 iters vs 20-50 iters)
- Check temporal coherence: ||u_t^* - u_{t-1}^*|| should be small for slowly-varying references

### Control and State Constraints
**Box constraints** (enforced via clipping):
```python
u_min = np.array([-0.5, -0.5, -0.5])  # Amperes
u_max = np.array([0.5, 0.5, 0.5])
```

**Rate limits** (enforced via cost penalty):
```python
R_delta = 10.0 * R  # Penalize ||u_t - u_{t-1}||^2
```

**State constraints** (NOT enforced in CP3.1-3.3):
- Obstacle avoidance: Future work
- Joint limits: Not applicable (continuum robot)
- Workspace bounds: Can add barrier function in cost if needed

### Cost Weights (Tuning Starting Points)
```python
# State cost (small, just for regularization)
Q = 1e-3 * np.eye(6)

# Control effort
R = 1.0 * np.eye(3)

# Control rate (smoothness)
R_delta = 10.0 * np.eye(3)

# Task-space tracking
W = 1000.0 * np.eye(3)  # Large weight prioritizes reaching target

# Terminal cost
W_f = 10.0 * W  # Even larger to ensure final convergence
```

**Tuning procedure** (in CP3.2):
1. Start with above defaults
2. If tracking is poor, increase W
3. If controls are jerky, increase R_delta
4. If terminal error is large, increase W_f
5. Document final weights in test results

---

## Failure Modes and Safeguards

CP3 is a **local optimization** method with known failure modes. This section documents **expected failures** and **mitigation strategies**.

### 1. Local Minima (Fundamental Limitation)
**Failure**: iLQR converges to suboptimal trajectory (e.g., wrong-side approach to target).

**Detection**:
- Final cost is high despite convergence (||∇cost|| < ε but cost > threshold)
- Visual inspection of trajectory shows obviously suboptimal path

**Mitigation**:
- **Multiple random initializations**: Run iLQR from 5-10 different initial guesses, pick best
- **Kinematic warm-start**: Use inverse kinematics or geometric planner to generate good initial guess
- **Hybrid global-local**: Use sampling-based planner (RRT) to get rough path, refine with iLQR

**Safeguard** (CP3.2):
- Test case with deliberately poor initialization (e.g., straight line through obstacle-like configuration)
- Verify that good initialization produces better cost than poor initialization
- Document when local minima are problematic (e.g., cluttered workspace)

### 2. Linearization Error (Nonlinearity)
**Failure**: iLQR diverges because linearization is invalid far from nominal trajectory.

**Detection**:
- Line search fails to reduce cost even with small α
- State trajectory explodes (||x_t|| → ∞)
- Controls saturate immediately (all u_t = u_max or u_min)

**Mitigation**:
- **Trust region regularization**: Increase λ in Q_uu_reg = Q_uu + λI to limit step size
- **Adaptive line search**: Try smaller α values (down to α=0.01)
- **Better initialization**: Start closer to feasible trajectory

**Safeguard** (CP3.1):
- Linearization sanity test: Compare linearized prediction vs actual rollout
  ```python
  x_pred = x_nom + A @ dx + B @ du  # Linearized
  x_actual = dynamics_step(x_nom + dx, u_nom + du, ...)
  error = ||x_pred - x_actual||
  assert error < 0.1 * ||dx||  # Linearization error should be small for small dx
  ```

### 3. Numerical Instability (Ill-Conditioned Hessians)
**Failure**: Q_uu matrix is near-singular, causing Riccati recursion to fail.

**Detection**:
- Regularization λ must be increased to very large values (λ > 1e3)
- Feedback gains K_t are very large (||K_t|| > 1e6)
- Control updates oscillate wildly between iterations

**Root causes**:
- Redundant actuators (multiple controls produce same tip motion)
- Poor cost function conditioning (W is too large relative to R)
- Timestep too large (dt > 0.1s can cause poor conditioning)

**Mitigation**:
- **Gauss-Newton Hessian**: Use Q_uu ≈ B^T W B (drop second-derivative terms) — already in design
- **Tikhonov regularization**: Always add λI to Q_uu, adaptively tune λ
- **Rebalance cost weights**: Decrease W if Q_uu is ill-conditioned

**Safeguard** (CP3.2):
- Log condition number of Q_uu at each timestep
- If cond(Q_uu) > 1e12, warn and increase regularization
- Test with degenerate actuator configurations (e.g., two channels identical)

### 4. Constraint Violation (Infeasible Trajectory)
**Failure**: Optimized trajectory requires u_t outside of [u_min, u_max].

**Detection**:
- After clipping, cost increases significantly
- Controls are saturated for many consecutive timesteps

**Mitigation**:
- **Penalty method**: Add barrier function to cost near constraints
  ```python
  L_barrier = 1e6 * sum(max(0, u_i - u_max)^2 + max(0, u_min - u_i)^2)
  ```
- **Projected iLQR**: Clip controls during forward pass, recompute gradients
- **Feasibility repair**: If infeasible, relax goal temporarily

**Safeguard** (CP3.3):
- Test case with deliberately unreachable target (requires u > u_max)
- Verify that optimizer saturates gracefully without diverging
- Document infeasible regions of workspace

### 5. Warm-Start Staleness (MPC Replanning)
**Failure**: Previous solution is poor initial guess for next MPC iteration (e.g., reference trajectory changes abruptly).

**Detection**:
- Warm-started MPC takes as many iterations as cold-start
- Temporal coherence check fails: ||u_t^* - u_{t-1}^*|| is large

**Mitigation**:
- **Hybrid warm-start**: If reference changes, do 1 cold-start iLQR, then resume warm-starting
- **Adaptive iteration budget**: Increase max_iters temporarily if warm-start quality is poor
- **Fallback to previous solution**: If optimization fails, use previous u_seq as safe fallback

**Safeguard** (CP3.3):
- Test with abrupt reference change (e.g., target jumps 10mm suddenly)
- Verify that MPC recovers within 3-5 replanning cycles
- Log warning if warm-start degrades (for debugging)

### 6. Real-Time Deadline Miss (MPC Timing)
**Failure**: Optimization does not finish within control period (e.g., iLQR takes >100ms but controller runs at 10Hz).

**Detection**:
- Wall-clock time exceeds dt before optimization finishes
- Missed deadlines accumulate (control lags behind real time)

**Mitigation**:
- **Early termination**: Stop iLQR after fixed time budget, use best solution so far
- **Asynchronous MPC**: Run optimization in background, use slightly stale solution
- **Reduce horizon**: Decrease T from 20 to 10 to cut computation time

**Safeguard** (CP3.4):
- Benchmark MPC timing on target hardware (log all iteration times)
- Set hard timeout: if optimization takes >80% of dt, terminate early
- Smoke test with real-time simulation loop (advance at fixed rate, measure lag)

### Summary of Risk Mitigation

| Failure Mode | Likelihood | Severity | Mitigation Checkpoint |
|--------------|------------|----------|----------------------|
| Local minima | High | Medium | CP3.2 (multi-init test) |
| Linearization error | Medium | High | CP3.1 (linearization validation) |
| Ill-conditioned Hessians | Low | Medium | CP3.2 (regularization tuning) |
| Constraint violation | Medium | Low | CP3.3 (barrier functions) |
| Warm-start staleness | Medium | Low | CP3.3 (temporal coherence test) |
| Timing deadline miss | High | High | CP3.4 (real-time smoke test) |

**Key insight**: All failure modes are **understood** and **detectable**. None are silent bugs that corrupt results. This makes CP3 safe to deploy incrementally.

---

## Checkpoint Plan

CP3 is divided into **4 incremental checkpoints**, each with clear deliverables and acceptance tests.

### CP3.1: Single-Step Linearization Sanity Test

**Goal**: Validate that CP2's `dynamics_backward` produces usable Jacobians for iLQR.

**Deliverables**:
1. **C++ test**: `test_cp31_linearization.cpp`
   - Extract A, B matrices at 3 operating points (rest, actuated, moving)
   - Compare linearized prediction vs actual rollout for small perturbations
   - Acceptance: ||x_actual - x_linearized|| / ||Δx|| < 0.1 for ||Δx|| < 0.01

2. **Python test**: `python/test_cp31_linearization.py`
   - Same test using PyTorch `jacobian()` to extract A, B
   - Verify that PyTorch and C++ Jacobians match
   - Acceptance: ||A_torch - A_cpp||_F / ||A_cpp||_F < 1e-6

**What we learn**:
- Are Jacobians numerically stable for iLQR use?
- Is linearization error acceptable for typical step sizes?
- Does PyTorch Jacobian extraction add significant overhead?

**Success criteria**:
- ✅ Both tests pass at all operating points
- ✅ Linearization error is <10% for perturbations ||Δx|| < 0.01, ||Δu|| < 0.05
- ✅ PyTorch Jacobian matches C++ within floating-point precision

**Failure modes**:
- If linearization error is large (>50%), dynamics may be too nonlinear for iLQR → need smaller dt
- If PyTorch/C++ mismatch, something is wrong with gradient flow → debug CP2.4

**Estimated effort**: 2-4 hours (mostly writing test harness)

---

### CP3.2: iLQR on Fixed Target

**Goal**: Implement full iLQR algorithm and demonstrate convergence to a fixed target.

**Deliverables**:
1. **Core iLQR implementation**: `python/ilqr_solver.py`
   - Backward pass: Riccati recursion to compute gains K_t, feedforward k_t
   - Forward pass: Rollout with control law u_t = ū_t + k_t + K_t(x_t - x̄_t)
   - Line search: Backtracking to ensure cost reduction
   - Convergence check: Stop when ||du|| < 1e-4 or max_iters reached

2. **Test**: `python/test_cp32_ilqr_fixed_target.py`
   - Initial state: x_0 = [0, 0, 0, 0, 0, 0] (rest)
   - Target: p_target = [10, 5, 0] mm (moderate reach)
   - Horizon: T = 20 steps, dt = 0.01s
   - Initial guess: Zero control u_t = [0, 0, 0] for all t
   - Optimize using iLQR
   - Acceptance:
     - Converges in <50 iterations
     - Final tip error ||p_tip(x_T) - p_target|| < 1.0 mm
     - All controls within bounds |u_t| < 0.5 A

3. **Visualization script**: `python/plot_ilqr_trajectory.py`
   - Plot tip trajectory in 3D
   - Plot control inputs over time
   - Plot cost reduction over iLQR iterations
   - Save to `docs/control/CP3_2_results/`

**What we learn**:
- Does iLQR converge for realistic targets?
- What is typical iteration count (10? 50? 100+)?
- Are cost weights Q, R, W well-tuned?
- Do controls saturate or remain interior?

**Success criteria**:
- ✅ iLQR converges to within 1mm of target
- ✅ Convergence in 10-50 iterations (not 200+)
- ✅ Trajectory is smooth (no jerky controls)
- ✅ Cost reduces monotonically (line search never fails catastrophically)

**Failure modes**:
- If iLQR diverges, linearization error too large → add trust region regularization
- If it converges to wrong solution, trapped in local minimum → test with better initialization
- If controls saturate, target may be unreachable → test with easier target first

**Tuning parameters** (document in test results):
- Cost weights Q, R, R_delta, W, W_f
- Regularization schedule for λ
- Line search parameters

**Estimated effort**: 1-2 days (core algorithm + debugging)

---

### CP3.3: Receding-Horizon MPC Rollout

**Goal**: Implement MPC loop with replanning and warm-starting.

**Deliverables**:
1. **MPC controller**: `python/mpc_controller.py`
   - `plan(x_current, p_ref, u_prev)`: Solve trajectory optimization from x_current
   - Warm-start: Use shifted u_prev as initial guess
   - Return: u_0^* (first control to execute)
   - Options:
     - `max_iters_cold_start=50` (first call)
     - `max_iters_warm_start=5` (subsequent calls)

2. **Test**: `python/test_cp33_mpc_tracking.py`
   - Reference trajectory: Straight line p_ref(t) = [t, 0, 0] for t ∈ [0, 2.0] seconds
   - MPC horizon: T = 10 steps
   - Replanning frequency: Every timestep (0.01s)
   - Simulate 200 steps (2.0s total)
   - Acceptance:
     - RMS tracking error < 2.0 mm
     - Controls smooth: ||u_t - u_{t-1}|| < 0.1 A
     - Warm-start converges in <10 iterations on average

3. **Benchmark**: Log timing for all MPC iterations
   - Report: min, median, max, p95 optimization time
   - Check: p95 < 100ms (required for 10Hz control)

4. **Visualization**: `python/plot_mpc_tracking.py`
   - Plot p_tip(t) vs p_ref(t) in 3D
   - Plot tracking error over time
   - Plot control inputs over time
   - Save to `docs/control/CP3_3_results/`

**What we learn**:
- Does warm-starting work as expected? (5 iters vs 50 iters)
- Is tracking error acceptable for realistic references?
- Can MPC run in real-time on typical hardware?

**Success criteria**:
- ✅ Tracking error < 2mm RMS for straight-line reference
- ✅ Warm-start reduces iterations by >5x vs cold-start
- ✅ Median optimization time < 50ms (viable for 20Hz control)
- ✅ No catastrophic failures (divergence, constraint violation) during 200-step rollout

**Failure modes**:
- If tracking error is large, horizon may be too short → increase T to 20
- If timing is too slow, too many iterations → reduce max_iters_warm_start to 3
- If warm-start doesn't help, shifting strategy may be wrong → investigate discontinuities in u_seq

**Extensions** (optional, if time permits):
- Test with curved reference (sinusoidal path)
- Test with moving target (step change in p_ref during rollout)
- Compare MPC vs open-loop iLQR (no replanning) to show benefit of feedback

**Estimated effort**: 1-2 days (MPC loop + warm-start debugging + benchmarking)

---

### CP3.4: CI Smoke Test for Controller Stability

**Goal**: Wire CP3 tests into CI to protect correctness in future development.

**Deliverables**:
1. **CTest entries**:
   ```cmake
   add_test(NAME ilqr_convergence_cp32 COMMAND python3 test_cp32_ilqr_fixed_target.py)
   add_test(NAME mpc_tracking_cp33 COMMAND python3 test_cp33_mpc_tracking.py)
   ```

2. **Smoke test parameters** (relaxed for CI speed):
   - iLQR: Single target, T=10 (not 20), max_iters=20 (not 50)
   - MPC: 50 steps (not 200), T=5 (not 10)
   - Acceptance: Same criteria but shorter horizon for speed

3. **GitHub Actions integration**:
   - Add CP3.2, CP3.3 tests to **nightly workflow** (not PR fast gate, too slow)
   - Estimated CI time: ~20-30 seconds for both tests

4. **Failure handling**:
   - If iLQR fails to converge, log cost trajectory and final state
   - If MPC tracking error exceeds threshold, upload trajectory plots as artifacts
   - Document failure modes in test output

**What we learn**:
- Are CP3 tests stable enough for CI? (no flakiness)
- What is acceptable CI runtime budget for control tests?
- Can we catch regressions in dynamics that break iLQR?

**Success criteria**:
- ✅ Tests pass 100% of the time on clean build
- ✅ Total CI time for CP3.2+CP3.3 < 30 seconds
- ✅ Test failures produce actionable diagnostics (logs, plots)

**Failure modes**:
- If tests are flaky (occasional failures), tolerances may be too tight → relax thresholds
- If CI time is too long, reduce horizon or number of timesteps
- If tests pass in CI but fail locally, environment mismatch (Python version, NumPy BLAS) → document dependencies

**Estimated effort**: 2-4 hours (mostly CMake and GitHub Actions config)

---

### Checkpoint Timeline Summary

| Checkpoint | Primary Goal | Estimated Effort | Dependencies |
|------------|-------------|------------------|--------------|
| **CP3.1** | Linearization validation | 2-4 hours | CP2.1-CP2.6 complete |
| **CP3.2** | iLQR implementation + convergence test | 1-2 days | CP3.1 pass |
| **CP3.3** | MPC with warm-start + tracking test | 1-2 days | CP3.2 pass |
| **CP3.4** | CI integration + smoke tests | 2-4 hours | CP3.2, CP3.3 pass |

**Total estimated effort**: 3-5 days for full CP3 completion.

**Critical path**:
1. CP3.1 must pass before implementing iLQR (validates linearization)
2. CP3.2 must pass before implementing MPC (validates optimizer)
3. CP3.3 must pass before CI integration (validates warm-start)

**Parallelization opportunities**: None (strictly sequential dependencies).

---

## File Structure

```
CRM_DiffSim_Cl/
├── src/
│   └── CRM_DiffDynamics.{hpp,cpp}        (CP2, no changes in CP3)
│
├── python/
│   ├── crm_dynamics_torch.py             (CP2, no changes in CP3)
│   ├── ilqr_solver.py                    (CP3.2 - NEW)
│   ├── mpc_controller.py                 (CP3.3 - NEW)
│   ├── cost_functions.py                 (CP3.2 - NEW, quadratic costs)
│   ├── test_cp31_linearization.py        (CP3.1 - NEW)
│   ├── test_cp32_ilqr_fixed_target.py    (CP3.2 - NEW)
│   ├── test_cp33_mpc_tracking.py         (CP3.3 - NEW)
│   ├── plot_ilqr_trajectory.py           (CP3.2 - NEW, visualization)
│   └── plot_mpc_tracking.py              (CP3.3 - NEW, visualization)
│
├── test_cp31_linearization.cpp           (CP3.1 - NEW, C++ Jacobian test)
│
├── docs/
│   └── control/
│       ├── CP3_MPC_DESIGN.md             (This document)
│       ├── CP3_1_COMPLETION.md           (CP3.1 report - FUTURE)
│       ├── CP3_2_COMPLETION.md           (CP3.2 report - FUTURE)
│       ├── CP3_3_COMPLETION.md           (CP3.3 report - FUTURE)
│       ├── CP3_4_COMPLETION.md           (CP3.4 report - FUTURE)
│       ├── CP3_2_results/                (Plots from iLQR tests)
│       └── CP3_3_results/                (Plots from MPC tests)
│
├── CMakeLists.txt                        (Add test_cp31_linearization target)
└── .github/workflows/nightly.yml         (Add CP3 tests to nightly gate)
```

**Key principles**:
- **No changes to CP2 code** — CP3 is pure application layer
- **Python-first for rapid prototyping** — C++ optimization later if needed
- **Separate cost functions module** — easy to swap in different objectives
- **Plotting scripts separate from tests** — tests run in CI, plots run manually for reports

---

## Open Questions and Future Extensions

### Questions to Resolve During Implementation

1. **Jacobian extraction performance**: Is PyTorch `jacobian()` fast enough, or do we need C++ `dynamics_linearize()`?
   - **Decision point**: CP3.1 benchmarks
   - **Threshold**: If >10ms per linearization, implement C++ version

2. **iLQR convergence rate**: How many iterations are typical for realistic targets?
   - **Decision point**: CP3.2 experiments
   - **Threshold**: If >50 iterations common, investigate second-order trust region methods

3. **MPC warm-start quality**: Does shifting work, or do we need more sophisticated prediction?
   - **Decision point**: CP3.3 tests
   - **Alternatives**: Predict next u_seq using constant velocity model of reference

4. **Real-time feasibility**: Can we hit 10Hz MPC on typical hardware (laptop CPU)?
   - **Decision point**: CP3.4 benchmarks
   - **Fallback**: Reduce horizon, use asynchronous MPC, or accept 1Hz replanning

### Extensions Beyond CP3 Scope

**Not implemented in CP3.1-3.4**, but natural follow-ons:

1. **Global trajectory optimization**
   - Use RRT or sampling-based planner to generate initial guess
   - Refine with iLQR
   - Useful for cluttered workspaces with local minima

2. **Obstacle avoidance**
   - Add barrier functions to cost: `L_obs = 1/(d - d_min)^2` where d is distance to obstacle
   - Requires collision detection infrastructure (not in CP2/CP3)

3. **Contact-aware MPC**
   - Use CP2's TipForce mechanism for known contacts
   - Requires contact model in dynamics (future CP2 extension)

4. **Stochastic MPC**
   - Model uncertainty in dynamics via process noise
   - Use chance constraints for robustness
   - Requires probabilistic forward simulation (not in CP2)

5. **Learning-based components**
   - Learn residual dynamics model to correct CP2's model error
   - Learn cost function weights from demonstrations
   - Use CP2 gradients to train policy networks

6. **Distributed MPC**
   - Multi-robot coordination
   - Requires extending state vector to include all robots

---

## Summary

CP3 implements **classical model-based optimal control** using CP2's validated differentiable dynamics. The design prioritizes:

1. **Correctness**: Leverage CP2's gradient validation to ensure iLQR is bug-free
2. **Simplicity**: Use textbook algorithms (iLQR, receding-horizon MPC) without exotic extensions
3. **Incrementality**: Four clear checkpoints, each independently testable
4. **Debuggability**: All failure modes are understood and detectable

**What makes this safe**:
- CP2 gradients are provably correct → iLQR backward pass is correct
- CP2 rollouts are stable → MPC forward simulation won't explode
- PyTorch integration works → rapid prototyping without custom autodiff
- No hidden state → controller is reproducible and debuggable

**What makes this useful**:
- iLQR generates high-quality trajectories for target reaching
- MPC handles disturbances and replanning in real-time
- Task-space objectives (p_tip tracking) are natural for clinical applications
- Warm-starting makes MPC tractable for online use

**Next steps after this design**:
1. Get user approval on design choices (cost function, horizon, parameters)
2. Implement CP3.1 (linearization test)
3. Proceed sequentially through CP3.2, CP3.3, CP3.4
4. Document results in completion reports

---

**End of CP3 Design Document**

*No implementation performed — design phase complete.*
*Ready for user approval and CP3.1 implementation.*
