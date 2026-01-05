# CP4.7: Hybrid MPC + Learned Policy Controller — DESIGN DOCUMENT

**Status**: DESIGN (awaiting approval)
**Date**: 2026-01-02
**Final CP4 Milestone**

---

## 1. Executive Summary

CP4.7 implements a **hybrid controller** that combines a fast learned ensemble policy with a correct-but-expensive MPC backup. The system is **provably no worse than MPC** while being significantly faster in the common case.

**Key Principle**: Use the fast policy by default; invoke MPC only when the ensemble disagrees (high uncertainty).

---

## 2. HybridController Logic (Decision Flow)

### 2.1 Core Algorithm

```
At each timestep t:
    1. Query ensemble → (u_mean, u_var, hiddens)
    2. Compute uncertainty: σ = max(u_var)
    3. Decision:
       IF σ < τ_low:
           u_t = u_mean                           # POLICY-ONLY (fast path)
       ELIF σ < τ_high:
           u_t = MPC(warm_start=u_mean)          # MPC-WARM (medium path)
       ELSE:
           u_t = MPC(warm_start=None)            # MPC-COLD (slow path, rare)
    4. Apply u_t, observe x_{t+1}, p_tip_{t+1}
    5. Safety check:
       IF dynamics_failed OR tracking_error > ε_safety:
           u_t = MPC(warm_start=None)            # SAFETY OVERRIDE
           Re-apply u_t
```

### 2.2 Class Structure

```python
class HybridController:
    """
    Hybrid MPC + Ensemble Policy Controller.

    Uses ensemble uncertainty to decide when to trust the policy vs invoke MPC.
    Provides performance of learned policy with safety guarantees of MPC.
    """

    def __init__(
        self,
        ensemble: EnsemblePolicy,         # From CP4.5
        mpc_solver: iLQRSolver,            # From CP3.2
        τ_low: float = 0.001,              # Policy-only threshold
        τ_high: float = 0.01,              # MPC-cold threshold
        tracking_safety_limit: float = 5.0, # mm, triggers MPC override
        dynamics_safety_checks: bool = True
    ):
        ...

    def step(
        self,
        x_t: np.ndarray,           # Current state (6,)
        p_tip_t: np.ndarray,       # Current tip position (3,)
        p_ref_horizon: np.ndarray, # Reference trajectory (H, 3)
        hiddens: Optional          # Recurrent hidden states
    ) -> Tuple[np.ndarray, dict]:
        """
        Returns: (u_t, info_dict)

        info_dict contains:
          - 'mode': 'policy' | 'mpc_warm' | 'mpc_cold' | 'safety_override'
          - 'uncertainty': float
          - 'mpc_called': bool
          - 'mpc_time_ms': float (if MPC called)
          - 'converged': bool (if MPC called)
        """
        ...

    def reset(self):
        """Reset hidden states and MPC warm-start cache."""
        ...
```

### 2.3 Decision Flow Diagram

```
                          ┌─────────────────┐
                          │ Ensemble Predict│
                          │ (u_mean, u_var) │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │ σ = max(u_var)  │
                          └────────┬────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              │                    │                    │
         σ < τ_low          τ_low ≤ σ < τ_high     σ ≥ τ_high
              │                    │                    │
              ▼                    ▼                    ▼
       ┌─────────────┐     ┌──────────────┐     ┌──────────────┐
       │ POLICY-ONLY │     │  MPC-WARM    │     │  MPC-COLD    │
       │ u = u_mean  │     │ init=u_mean  │     │ init=None    │
       │ (~0.1 ms)   │     │ (~10-50 ms)  │     │ (~50-200 ms) │
       └──────┬──────┘     └──────┬───────┘     └──────┬───────┘
              │                   │                    │
              └───────────────────┴────────────────────┘
                                  │
                          ┌───────▼───────┐
                          │ Apply u_t     │
                          │ Check Safety  │
                          └───────┬───────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
               Safe (normal)          Unsafe (dynamics fail OR
                    │                  tracking_error > ε_safety)
                    │                           │
                    ▼                           ▼
            ┌──────────────┐          ┌─────────────────┐
            │ Return u_t   │          │ SAFETY OVERRIDE │
            │              │          │ MPC-COLD, retry │
            └──────────────┘          └─────────────────┘
```

---

## 3. Uncertainty Thresholds and Fallback Rules

### 3.1 Threshold Selection

Based on CP4.5 data (r=0.641 correlation between uncertainty and error):

| Threshold | Value | Meaning | Expected Frequency |
|-----------|-------|---------|-------------------|
| τ_low     | 0.001 | High confidence → policy-only | 70-80% of steps |
| τ_high    | 0.01  | Medium confidence → MPC-warm | 15-25% of steps |
| (above)   | >0.01 | Low confidence → MPC-cold | 5-10% of steps |

### 3.2 Adaptive Threshold Calibration (Optional)

```python
def calibrate_thresholds(ensemble, validation_trajectories, target_mpc_rate=0.2):
    """
    Find τ_low such that MPC is called ~20% of the time on validation data.

    Algorithm:
    1. Roll out ensemble on validation set, record (uncertainty, tracking_error) pairs
    2. Sort by uncertainty
    3. Find τ_low where P(uncertainty > τ_low) ≈ target_mpc_rate
    4. Set τ_high = 10 * τ_low (heuristic)
    """
```

### 3.3 Fallback Rules

| Condition | Action | Rationale |
|-----------|--------|-----------|
| `max(u_var) < τ_low` | Use policy | Ensemble agrees → confident |
| `τ_low ≤ max(u_var) < τ_high` | MPC with warm-start | Moderate uncertainty, policy gives good init |
| `max(u_var) ≥ τ_high` | MPC cold-start | High uncertainty, don't trust policy |
| `dynamics_forward.status != 0` | MPC cold-start + retry | Solver failure, need expert |
| `lu_rank != 6` | MPC cold-start + retry | Jacobian singularity |
| `rel_residual > 1e-10` | MPC cold-start + retry | Poor convergence |
| `tracking_error > 5.0 mm` | MPC cold-start + retry | Safety limit exceeded |

---

## 4. MPC Warm-Start from Policy

### 4.1 Warm-Start Strategy

When MPC is invoked with warm-start from the policy:

```python
def generate_warm_start(ensemble, x_t, p_ref_horizon, hiddens, params_dict, dt, L_inserted):
    """
    Generate MPC warm-start by rolling out the policy.

    Args:
        ensemble: Trained ensemble policy
        x_t: Current state
        p_ref_horizon: Reference trajectory (H, 3)
        hiddens: Recurrent hidden states
        params_dict: Physics parameters
        dt, L_inserted: Dynamics parameters

    Returns:
        U_init: (H, 3) control sequence for MPC warm-start
    """
    H = len(p_ref_horizon)
    U_init = np.zeros((H, 3))
    x = x_t.copy()
    h = hiddens

    for t in range(H):
        # Get policy action (ignore uncertainty for warm-start)
        u_mean, _, h = ensemble.predict(x[:3], p_ref_horizon[t], h)  # Assuming p_tip ≈ x[:3]
        U_init[t] = np.clip(u_mean, -0.5, 0.5)

        # Simulate forward (optional, for better warm-start)
        # result = dynamics_forward(x, U_init[t], dt, L_inserted, params_dict)
        # if result['status'] == 0:
        #     x = result['x_next']

    return U_init
```

### 4.2 MPC Integration

```python
def invoke_mpc(self, x_t, p_ref_horizon, warm_start_mode):
    """
    Invoke MPC with appropriate warm-start.

    Args:
        warm_start_mode: 'policy' | 'shift' | 'cold'
    """
    if warm_start_mode == 'policy':
        U_init = self.generate_warm_start(...)
    elif warm_start_mode == 'shift':
        # Shift previous solution (standard MPC warm-start)
        U_init = np.vstack([self.U_prev[1:], self.U_prev[-1:]])
    else:  # cold
        U_init = None

    # Invoke iLQR solver (from CP3.2)
    solver = iLQRSolver(
        dt=self.dt,
        L_inserted=self.L_inserted,
        params_dict=self.params_dict,
        horizon=len(p_ref_horizon),
        p_target=p_ref_horizon[0],
        max_iters=self.mpc_max_iters,
        jacobian_mode="cpp"  # Use CP4.4c fast Jacobians
    )

    X, U, converged = solver.solve(x_t, U_init=U_init, verbose=False)
    self.U_prev = U  # Cache for next shift

    return U[0], converged
```

### 4.3 Expected Speedup from Warm-Start

| Mode | Typical iLQR Iterations | Time (est.) |
|------|------------------------|-------------|
| Cold-start | 15-30 | 50-200 ms |
| Shift warm-start | 5-15 | 20-80 ms |
| Policy warm-start | 3-10 | 10-50 ms |
| Policy-only (no MPC) | 0 | <1 ms |

---

## 5. Evaluation Metrics

### 5.1 Primary Metrics

| Metric | Definition | Target |
|--------|------------|--------|
| **MPC Call Rate** | `n_mpc_calls / n_total_steps` | ≤ 30% |
| **Tracking RMSE** | `sqrt(mean(||p_tip - p_ref||²))` | ≤ MPC-only RMSE + 0.5 mm |
| **Max Tracking Error** | `max(||p_tip - p_ref||)` | ≤ MPC-only max + 1.0 mm |
| **Runtime Speedup** | `time_mpc_only / time_hybrid` | ≥ 2× |
| **Safety Violations** | Steps where dynamics failed | 0 (same as MPC-only) |

### 5.2 Secondary Metrics

| Metric | Definition | Purpose |
|--------|------------|---------|
| **Mode Distribution** | % policy-only, mpc-warm, mpc-cold | Understand decision patterns |
| **Safety Override Rate** | % of steps triggering safety override | Should be < 5% |
| **Uncertainty Calibration** | Correlation(uncertainty, actual_error) | Validate uncertainty quality |
| **MPC Convergence Rate** | % of MPC calls that converge | Should be > 95% |
| **Per-Mode Latency** | Mean time for each decision mode | Performance profiling |

### 5.3 Metrics Collection

```python
class HybridControllerMetrics:
    """Collect and report hybrid controller metrics."""

    def __init__(self):
        self.steps = []
        self.modes = []  # 'policy', 'mpc_warm', 'mpc_cold', 'safety'
        self.uncertainties = []
        self.tracking_errors = []
        self.mpc_times = []
        self.mpc_converged = []

    def add_step(self, info_dict, p_tip, p_ref):
        ...

    def get_summary(self) -> dict:
        return {
            'n_steps': len(self.steps),
            'mpc_call_rate': sum(m != 'policy' for m in self.modes) / len(self.modes),
            'mode_distribution': {
                'policy': self.modes.count('policy') / len(self.modes),
                'mpc_warm': self.modes.count('mpc_warm') / len(self.modes),
                'mpc_cold': self.modes.count('mpc_cold') / len(self.modes),
                'safety': self.modes.count('safety') / len(self.modes),
            },
            'tracking_rmse': float(np.sqrt(np.mean(np.array(self.tracking_errors)**2))),
            'tracking_max': float(np.max(self.tracking_errors)),
            'mean_mpc_time_ms': float(np.mean(self.mpc_times)) if self.mpc_times else 0.0,
            'mpc_convergence_rate': sum(self.mpc_converged) / len(self.mpc_converged) if self.mpc_converged else 1.0,
            'uncertainty_error_correlation': float(np.corrcoef(self.uncertainties, self.tracking_errors)[0, 1])
        }
```

---

## 6. Acceptance Tests + CI Strategy

### 6.1 Test Suite

| Test | Purpose | Runtime | CI Type |
|------|---------|---------|---------|
| `test_cp47_hybrid_smoke.py` | Basic functionality | <60s | Every commit |
| `test_cp47_hybrid_benchmark.py` | Full evaluation | ~5 min | Nightly |
| `test_cp47_no_regression.py` | Compare to MPC-only | ~10 min | Pre-merge |

### 6.2 Smoke Test (`test_cp47_hybrid_smoke.py`)

```python
def test_hybrid_controller_creation():
    """Test 1: Controller can be created with valid ensemble."""
    ensemble = load_test_ensemble()
    mpc = create_test_mpc()
    controller = HybridController(ensemble, mpc)
    assert controller is not None

def test_hybrid_controller_step():
    """Test 2: Controller produces valid control output."""
    controller = create_test_controller()
    x_t = np.zeros(6)
    p_tip = np.array([0.0, 0.0, 50.0])
    p_ref_horizon = np.tile([1.0, 1.0, 52.0], (10, 1))

    u, info = controller.step(x_t, p_tip, p_ref_horizon, hiddens=None)

    assert u.shape == (3,)
    assert np.all(np.abs(u) <= 0.5)  # Within control limits
    assert info['mode'] in ['policy', 'mpc_warm', 'mpc_cold', 'safety']
    assert 'uncertainty' in info

def test_hybrid_controller_rollout():
    """Test 3: Full rollout without crashes."""
    controller = create_test_controller()
    metrics = HybridControllerMetrics()

    # 100-step rollout on test trajectory
    for t in range(100):
        u, info = controller.step(...)
        metrics.add_step(info, p_tip, p_ref[t])

    summary = metrics.get_summary()
    assert summary['mpc_call_rate'] < 0.5  # Policy should handle most steps
    assert summary['tracking_rmse'] < 10.0  # Reasonable tracking

def test_uncertainty_triggers_mpc():
    """Test 4: High uncertainty triggers MPC."""
    controller = create_test_controller()

    # Artificially create high-uncertainty scenario
    # (novel state far from training distribution)
    x_unusual = np.array([0.5, 0.5, 0.5, 0.0, 0.0, 0.0])  # Extreme curvature
    p_tip = np.array([50.0, 50.0, 100.0])  # Far from origin
    p_ref = np.array([[60.0, 60.0, 110.0]] * 10)

    u, info = controller.step(x_unusual, p_tip, p_ref, hiddens=None)

    # Should trigger MPC due to high uncertainty (or safety override)
    assert info['mode'] != 'policy' or info['mpc_called']
```

### 6.3 Benchmark Test (`test_cp47_hybrid_benchmark.py`)

```python
def test_hybrid_vs_mpc_only():
    """
    Compare hybrid controller to MPC-only baseline.

    PASS criteria:
    - Tracking RMSE ≤ MPC-only + 0.5 mm
    - Runtime ≤ MPC-only * 0.5 (i.e., 2× speedup)
    - MPC call rate ≤ 30%
    """
    # Load full test dataset
    datasets = load_test_datasets()

    # Run MPC-only baseline
    mpc_metrics = run_mpc_only(datasets)

    # Run hybrid controller
    hybrid_metrics = run_hybrid(datasets)

    # Assertions
    assert hybrid_metrics['tracking_rmse'] <= mpc_metrics['tracking_rmse'] + 0.5
    assert hybrid_metrics['runtime'] <= mpc_metrics['runtime'] * 0.5
    assert hybrid_metrics['mpc_call_rate'] <= 0.30

    # Report
    print(f"MPC-only RMSE: {mpc_metrics['tracking_rmse']:.3f} mm")
    print(f"Hybrid RMSE:   {hybrid_metrics['tracking_rmse']:.3f} mm")
    print(f"Speedup:       {mpc_metrics['runtime'] / hybrid_metrics['runtime']:.2f}x")
    print(f"MPC call rate: {hybrid_metrics['mpc_call_rate']:.1%}")
```

### 6.4 CTest Integration

```cmake
# CMakeLists.txt additions

# CP4.7: Hybrid controller smoke test (CI gate)
add_test(
    NAME hybrid_controller_smoke_cp47
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/test_cp47_hybrid_smoke.py
)
set_tests_properties(hybrid_controller_smoke_cp47 PROPERTIES TIMEOUT 60)

# CP4.7: Hybrid controller benchmark (nightly)
add_test(
    NAME hybrid_controller_benchmark_cp47
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/python/test_cp47_hybrid_benchmark.py
)
set_tests_properties(hybrid_controller_benchmark_cp47 PROPERTIES TIMEOUT 600)
set_tests_properties(hybrid_controller_benchmark_cp47 PROPERTIES LABELS "nightly")
```

---

## 7. Definition of "Done"

### 7.1 Required Deliverables

| Deliverable | Status |
|-------------|--------|
| `python/control/hybrid_controller.py` | Implementation of HybridController class |
| `python/test_cp47_hybrid_smoke.py` | Smoke test (<60s) |
| `python/test_cp47_hybrid_benchmark.py` | Full benchmark |
| `docs/audits/CP4_7_HYBRID_CONTROLLER_COMPLETION.md` | Completion audit |
| CTest entries in `CMakeLists.txt` | CI integration |

### 7.2 Acceptance Criteria

| Criterion | Target | Validation |
|-----------|--------|------------|
| Smoke test passes | <60s | `ctest -R hybrid_controller_smoke_cp47` |
| MPC call rate | ≤ 30% | Benchmark test |
| Tracking RMSE | ≤ MPC-only + 0.5 mm | Benchmark test |
| Runtime speedup | ≥ 2× vs MPC-only | Benchmark test |
| Safety violations | 0 (match MPC-only) | No dynamics failures |
| No physics changes | True | Code review |
| Reuses CP4.4-4.6 | True | Code review |

### 7.3 "Done" Checklist

```
[ ] HybridController class implemented
[ ] Uses EnsemblePolicy from CP4.5 for uncertainty
[ ] Uses iLQRSolver from CP3.2 with jacobian_mode="cpp" (CP4.4b/c)
[ ] MPC warm-start from policy rollout implemented
[ ] Safety override logic implemented
[ ] Metrics collection and reporting
[ ] Smoke test passing (<60s)
[ ] Benchmark test passing (meets all targets)
[ ] No regression vs MPC-only tracking accuracy
[ ] Completion audit written
[ ] CTest integration complete
```

---

## 8. Implementation Plan

### Phase 1: Core Implementation (~2 hours)

1. Create `python/control/hybrid_controller.py`:
   - `HybridController` class with decision logic
   - `HybridControllerMetrics` class for tracking
   - Integration with `EnsemblePolicy` (CP4.5) and `iLQRSolver` (CP3.2)

2. Implement warm-start generation:
   - Policy rollout for U_init
   - Shift-based warm-start for subsequent MPC calls

### Phase 2: Testing (~1 hour)

3. Create `python/test_cp47_hybrid_smoke.py`:
   - 4-5 test cases covering core functionality
   - Runtime < 60s

4. Create `python/test_cp47_hybrid_benchmark.py`:
   - Full comparison to MPC-only baseline
   - Validate all acceptance criteria

### Phase 3: Integration (~30 min)

5. Add CTest entries to `CMakeLists.txt`

6. Write completion audit `docs/audits/CP4_7_HYBRID_CONTROLLER_COMPLETION.md`

### Phase 4: Validation (~30 min)

7. Run full test suite
8. Verify metrics meet targets
9. Final code review

---

## 9. Risk Analysis

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Uncertainty calibration poor | Medium | Medium | Use adaptive threshold calibration |
| MPC call rate too high | Low | Medium | Adjust τ_low, retrain ensemble |
| Tracking regression | Low | High | Safety override ensures MPC fallback |
| Runtime not improved | Low | Medium | Profile and optimize bottlenecks |
| Integration issues | Low | Low | Reuse existing tested components |

---

## 10. Dependencies

### Required Components (all complete)

| Component | Source | Purpose |
|-----------|--------|---------|
| `EnsemblePolicy` | CP4.5 | Uncertainty estimation |
| `GRUPolicy` | CP4.4a | Base policy architecture |
| `iLQRSolver` | CP3.2 | MPC implementation |
| `dynamics_linearize` | CP4.4b/c | Fast Jacobians |
| `dynamics_forward` | CP2 | Dynamics simulation |
| `crm_diff_py` | Build | Python bindings |

### No New Dependencies

- No new ML frameworks
- No physics changes
- No external libraries

---

## 11. Summary

CP4.7 is the **final CP4 milestone** that unifies all prior work:

- **CP4.0-4.3**: DAgger training infrastructure
- **CP4.4a-c**: Fast recurrent policies + fast Jacobians
- **CP4.5**: Ensemble uncertainty estimation
- **CP4.6**: Multi-task capability

The hybrid controller achieves:
- **Performance**: ~2× speedup over MPC-only
- **Safety**: Provably no worse than MPC (safety override)
- **Simplicity**: Reuses existing components, no new learning

**Success metric**: MPC call rate ≤ 30% with tracking RMSE ≤ MPC-only + 0.5 mm.

---

**END OF DESIGN DOCUMENT**
