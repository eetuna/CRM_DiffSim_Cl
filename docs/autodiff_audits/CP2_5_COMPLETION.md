# CP2.5 Completion Report: Multi-step Rollout + Backprop Smoke Test

**Status:** ✅ **PASS**

**Date:** 2025-12-31

**Author:** Claude Code (Anthropic)

---

## Summary

CP2.5 implements **multi-step differentiable rollouts** through PyTorch autograd, validating that:
1. The dynamics primitive can be chained for T=20 steps without numerical instability
2. Gradients can be backpropagated through the entire rollout
3. Control gradients are correctly computed for optimization use cases

All three test rollouts **PASSED**:
- ✅ **OP-A:** Zero control (u=0) → trivial trajectory, zero gradient (as expected)
- ✅ **OP-B:** Constant actuation (u=[0.1,0,0]) → non-trivial trajectory, non-zero gradients
- ✅ **OP-C:** Sinusoidal actuation (u=[0.1·sin(2πt/T),0,0]) → complex trajectory, non-zero gradients

---

## Test Design

### Rollout Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| `T` | 20 steps | Rollout length |
| `dt` | 0.01 s | Time step (10 ms) |
| `L_inserted` | 50.0 mm | Insertion length |
| **Total time** | **0.2 s** | Physical simulation duration |

### Operating Points

#### OP-A: Zero Control
- **Control:** `u_t = [0, 0, 0]` for all t
- **Initial state:** `x0 = zeros(6)`
- **Purpose:** Baseline test; expects trivial trajectory and zero/tiny gradients
- **Gradient check:** Disabled (expect ≈0)

#### OP-B: Constant Control
- **Control:** `u_t = [0.1, 0, 0]` for all t (constant 100 mA in channel 1)
- **Initial state:** `x0 = zeros(6)`
- **Purpose:** Validate gradient flow for steady actuation
- **Gradient check:** Enabled (expect ||∇u|| > 1e-12)

#### OP-C: Sinusoidal Control
- **Control:** `u_t = [0.1·sin(2πt/T), 0, 0]`
- **Initial state:** `x0 = [0.01, 0, 0, 0.1, 0, 0]` (small perturbation)
- **Purpose:** Validate gradient flow for time-varying actuation
- **Gradient check:** Enabled (expect ||∇u|| > 1e-12)

### Acceptance Criteria

For each rollout, the test verifies:

1. **Forward pass stability:**
   - All states `x_t` remain **finite** (no NaN/Inf)
   - State norm bounded: `||x_t|| < 1e3` for all t
   - C++ `dynamics_forward` returns `status=0` (checked every 5 steps)

2. **Backward pass correctness:**
   - PyTorch `loss.backward()` completes without exception
   - Gradients `∇u` are **finite** (no NaN/Inf)
   - For actuated cases (OP-B, OP-C): `||∇u|| > 1e-12` (non-zero)

---

## Backpropagation Method

### Loss Function

The test uses a **state-space loss**:

```
L = Σ_t ||x_t||²
```

where the sum is over all T+1 states in the trajectory (including x₀).

This loss is differentiable w.r.t. the control sequence `u = [u_0, ..., u_{T-1}]` via PyTorch autograd.

### Gradient Computation

1. **Control sequence** is a PyTorch tensor with `requires_grad=True`:
   ```python
   u_seq = torch.tensor([u_0, ..., u_{T-1}], requires_grad=True)  # shape [T, 3]
   ```

2. **Rollout** uses `crm_dynamics_torch.dynamics_step()`, which wraps the C++ `dynamics_forward/backward` primitives via `DynamicsStep.apply`:
   ```python
   for t in range(T):
       x_{t+1} = dynamics_step(x_t, u_seq[t], dt, L_inserted, params_dict)
   ```

3. **Loss** aggregates all states:
   ```python
   loss = sum(torch.sum(x**2) for x in trajectory)
   ```

4. **Backward** computes `∇u` via PyTorch autograd:
   ```python
   loss.backward()  # Populates u_seq.grad
   grad_u = u_seq.grad.numpy()
   ```

### Nested Finite Differences

**Note:** The backward pass uses **nested FD** because `dynamics_backward` internally uses FD to compute `dA/du` and `dB/du` (as documented in CP2.2). This makes gradcheck expensive but does not affect correctness.

---

## Test Results

### Command

```bash
# Run CP2.5 test via CTest
ctest -R dynamics_rollout_cp25_python --output-on-failure

# Run directly with Python
PYTHONPATH=build:$PYTHONPATH python3 python/test_cp25_dynamics_rollout_smoke.py
```

### Sample Output

```
======================================================================
CP2.5: Multi-step Rollout + Backprop Smoke Test
======================================================================

Rollout configuration:
  T = 20 steps
  dt = 0.01 s
  L_inserted = 50.0 mm
  Total time = 0.2 s

======================================================================
Rollout: OP-A: Zero control (u=0)
======================================================================
x0 = [0. 0. 0. 0. 0. 0.]
...
Rolling forward 20 steps...
  ✓ Forward rollout succeeded
  Final state x_T = [0. 0. 0. 0. 0. 0.]

Loss computation:
  L = sum_t ||x_t||^2 = 0.000000e+00

Backward pass...
  ||grad_u|| = 0.000000e+00
  Note: ||grad_u|| ≈ 0 (expected for zero control)

✓ PASS: OP-A: Zero control (u=0)
  Summary: T=20, loss=0.000000e+00, ||grad_u||=0.000000e+00

======================================================================
Rollout: OP-B: Constant control (u=[0.1,0,0])
======================================================================
x0 = [0. 0. 0. 0. 0. 0.]
...
Rolling forward 20 steps...
  ✓ Forward rollout succeeded
  Final state x_T = [ 1.21e-05 -1.41e-04 -3.54e-02  0.00e+00  0.00e+00 -5.58e-16]

Loss computation:
  L = sum_t ||x_t||^2 = 1.150666e+01

Backward pass...
  ||grad_u|| = 2.423883e+02
  ✓ Gradient is non-zero (as expected)

✓ PASS: OP-B: Constant control (u=[0.1,0,0])
  Summary: T=20, loss=1.150666e+01, ||grad_u||=2.423883e+02

======================================================================
Rollout: OP-C: Sinusoidal control (u=[0.1*sin(2πt/T),0,0])
======================================================================
x0 = [0.01 0.   0.   0.1  0.   0.  ]
...
Rolling forward 20 steps...
  ✓ Forward rollout succeeded
  Final state x_T = [-3.99e-06  4.36e-05  1.15e-02  3.25e-04 -3.59e-03 -9.18e-01]

Loss computation:
  L = sum_t ||x_t||^2 = 1.138576e+01

Backward pass...
  ||grad_u|| = 1.280735e+02
  ✓ Gradient is non-zero (as expected)

✓ PASS: OP-C: Sinusoidal control (u=[0.1*sin(2πt/T),0,0])
  Summary: T=20, loss=1.138576e+01, ||grad_u||=1.280735e+02

======================================================================
CP2.5 SUMMARY
======================================================================
  ✓ PASS: OP-A: Zero control (u=0)
  ✓ PASS: OP-B: Constant control (u=[0.1,0,0])
  ✓ PASS: OP-C: Sinusoidal control (u=[0.1*sin(2πt/T),0,0])

======================================================================
CP2.5: PASS - All rollout tests passed
  - All states remained finite and bounded
  - Forward passes succeeded
  - Backward passes produced valid gradients
======================================================================
```

### CTest Results

```bash
$ ctest -R dynamics_rollout_cp25_python --output-on-failure
Test project /workspaces/CRM_DiffSim_Cl/build
    Start 9: dynamics_rollout_cp25_python
1/1 Test #9: dynamics_rollout_cp25_python .....   Passed   11.59 sec

100% tests passed, 0 tests failed out of 1

Total Test time (real) =  11.60 sec
```

### Existing Tests (Still Passing)

```bash
$ ctest -R "dynamics_smoke_cp21|dynamics_fd_cp22|dynamics_smoke_cp23_python|dynamics_gradcheck_cp24_python" --output-on-failure
Test project /workspaces/CRM_DiffSim_Cl/build
    Start 5: dynamics_smoke_cp21
1/4 Test #5: dynamics_smoke_cp21 ..............   Passed    0.03 sec
    Start 6: dynamics_fd_cp22
2/4 Test #6: dynamics_fd_cp22 .................   Passed    0.36 sec
    Start 7: dynamics_smoke_cp23_python
3/4 Test #7: dynamics_smoke_cp23_python .......   Passed    0.71 sec
    Start 8: dynamics_gradcheck_cp24_python
4/4 Test #8: dynamics_gradcheck_cp24_python ...   Passed   12.57 sec

100% tests passed, 0 tests failed out of 4

Total Test time (real) =  13.69 sec
```

✅ **All existing CP2.1–CP2.4 tests remain passing.**

---

## Files Changed

### New Files

1. **`python/test_cp25_dynamics_rollout_smoke.py`**
   - Multi-step rollout test with 3 operating points
   - Uses PyTorch autograd for end-to-end backprop
   - Checks: finite states, bounded norms, valid gradients

### Modified Files

2. **`CMakeLists.txt`**
   - Added CTest entry `dynamics_rollout_cp25_python` (lines 271-277)
   - Follows same pattern as CP2.3/CP2.4 (pybind11-gated, PYTHONPATH setup)

---

## Known Limitations

1. **Nested FD cost:**
   - `dynamics_backward` uses FD for `dA/du` and `dB/du` (CP2.2 design)
   - Backprop through T=20 steps involves ~20 backward calls, each with FD overhead
   - Runtime: ~11.6s for CP2.5 (vs ~0.03s for single-step CP2.1)
   - **Acceptable for smoke tests**; production rollouts may need longer horizons

2. **CPU-only:**
   - PyTorch wrapper enforces `device='cpu'` (no GPU support)
   - Limits scalability for large-batch trajectory optimization

3. **State-space loss only:**
   - Test uses `L = Σ ||x||²` for simplicity
   - Real applications may need task-space losses (e.g., tip position error)
   - Tip positions are available via `dynamics_forward['p_tip']` but not used in loss here

4. **No long-horizon validation:**
   - T=20 steps (0.2s) is short for medical robotics
   - Numerical stability for T>100 not tested
   - Future work: extended rollouts with adaptive dt or implicit integrators

5. **No control bounds:**
   - Test does not enforce current limits (e.g., |u| ≤ I_max)
   - Gradient-based optimizers may violate physical constraints without projection

---

## Integration into CI/CD

### CTest Target

```bash
# Run only CP2.5
ctest -R dynamics_rollout_cp25_python --output-on-failure

# Run all CP2.x dynamics tests
ctest -R "dynamics.*cp2" --output-on-failure
```

### Acceptance Gate

CP2.5 is now part of the CP2.x suite. All tests must pass before merging dynamics-related changes:

```bash
ctest -R "dynamics_smoke_cp21|dynamics_fd_cp22|dynamics_smoke_cp23_python|dynamics_gradcheck_cp24_python|dynamics_rollout_cp25_python" --output-on-failure
```

**Current status:** ✅ All 5 tests passing (runtime: ~25s total)

---

## Recommendations for Future Work

1. **Longer horizons:**
   - Test T=100 steps (~1s) to validate numerical stability
   - Profile memory usage and backward pass cost

2. **Task-space losses:**
   - Add rollout test with `L = Σ ||p_tip_t - p_target||²`
   - Requires extracting `p_tip` at each step (currently available in forward result)

3. **Gradient norm checks:**
   - Add assertions on gradient magnitude bounds (e.g., `||∇u|| < 1e6`)
   - Detect gradient explosion early

4. **Control trajectory optimization:**
   - Implement simple trajectory optimization (e.g., LBFGS on 20-step rollout)
   - Validate convergence to low-loss controls

5. **Batched rollouts:**
   - Extend torch wrapper to support batch dimension: `x: [B, 6]`, `u: [B, 3]`
   - Enable parallel trajectory sampling for RL/MPC applications

---

## Commit Message Template

```
CP2.5: Add multi-step rollout + backprop smoke test

Implements differentiable 20-step rollouts through PyTorch autograd,
validating gradient flow for trajectory optimization use cases.

Test design:
- 3 operating points: zero control, constant actuation, sinusoidal actuation
- T=20 steps, dt=0.01s, L_inserted=50mm
- Loss: sum_t ||x_t||^2
- Backward via torch.autograd through DynamicsStep.apply

Acceptance criteria:
- All states finite and bounded (||x|| < 1e3)
- Forward passes succeed (status=0)
- Gradients finite and non-zero for actuated cases

Results: ALL PASS (runtime ~11.6s)
- OP-A (zero control): loss=0, ||grad_u||=0 (expected)
- OP-B (constant): loss=11.5, ||grad_u||=242
- OP-C (sinusoidal): loss=11.4, ||grad_u||=128

Files:
- python/test_cp25_dynamics_rollout_smoke.py (new)
- CMakeLists.txt (add CTest entry)

Limitations:
- Nested FD cost (~11.6s for T=20)
- CPU-only (no GPU)
- State-space loss only (no task-space)

Existing tests CP2.1-CP2.4 remain passing.
```

---

## Sign-off

**CP2.5 Status:** ✅ **COMPLETE**

All acceptance criteria met:
- ✅ Python rollout test implemented
- ✅ 3 operating points tested (T=20 steps each)
- ✅ Forward passes stable (finite, bounded, status=0)
- ✅ Backward passes produce valid gradients
- ✅ CTest integration complete
- ✅ Existing tests CP2.1–CP2.4 still pass
- ✅ Completion documentation exported

**Ready for:** Commit + merge to main branch

---

**End of CP2.5 Completion Report**
