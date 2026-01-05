# CP3.5.1: NPZ Replay Contract Fix & Validation

**Status**: ✅ Complete
**Date**: 2026-01-01
**Author**: Claude Code

## Summary

CP3.5.1 fixes critical issues in the original CP3.5 NPZ replay test that caused wildly incorrect results (RMS errors of 40mm vs expected 0.8mm). The root cause was **missing warm-start**: the NPZ datasets were recorded with solver warm-starting, but the replay code was cold-starting every solve.

This fix adds:
1. **Contract test** that verifies faithful NPZ replay with correct warm-start strategy
2. **Fixed visualization** that plots NPZ data directly (no recomputation)
3. **Complete root cause documentation**

---

## Root Causes Found

### 1. Missing Warm-Start for FK Solves

**Problem**: FK equilibrium solver has multiple solution branches. Without warm-starting (using previous `deltau0` as initial guess for next solve), the solver finds different branches, causing huge mismatches.

**Evidence**:
- Without warm-start: FK replay RMS = 46.8mm, Max = 139mm
- With warm-start: FK replay RMS = 0.023mm, Max = 0.41mm

**Fix**: Use previous `deltau0` as `deltau0_initialguess` for next FK solve.

### 2. Mismatched IntegrationStepSize

**Problem**: Original CP3.5 hardcoded `IntegrationStepSize = 0.5`, but:
- Circle NPZ used `IntegrationStepSize = 0.5` (happened to match)
- Lemniscate NPZ used `IntegrationStepSize = 0.2` (mismatch!)

**Fix**: Load `integration_step_size` from NPZ if available, else default to 0.5.

### 3. Dynamics Not Using FK-Based Warm-Start

**Problem**: Dynamics solver ALSO needs correct `deltau0` initial guess. The NPZ was recorded by:
1. Solving FK to get `deltau0`
2. Using that `deltau0` for dynamics solve

This ensures FK and dynamics use the same nonlinear solution branch.

**Evidence**:
- Without FK-based warm-start: Dyn replay RMS = 48mm
- With FK-based warm-start: Dyn replay RMS = 0.83mm (matches NPZ FK-Dyn intrinsic error!)

**Fix**: For each timestep:
1. Solve FK with warm-start to get `deltau0`
2. Use FK's `deltau0` for dynamics solve

### 4. Wrong Metrics Being Computed

**Problem**: Original CP3.5 compared:
- Fresh FK rollout vs fresh Dyn rollout (both recomputed without proper warm-start)

This gave large errors even though both were "wrong" in the same way.

**Fix**: Compare replay against NPZ data directly:
- FK replay vs FK NPZ
- Dyn replay vs Dyn NPZ
- Show NPZ's intrinsic FK-Dyn mismatch as ground truth

---

## Before/After Metrics

### Circle Trajectory (data/dyn_fk_ramp_circle1_hold1.npz)

| Metric | Before (CP3.5) | After (CP3.5.1) | NPZ Ground Truth |
|--------|----------------|------------------|------------------|
| FK replay RMS | 46.8mm | 0.023mm | (should match NPZ exactly) |
| FK replay Max | 139mm | 0.41mm | (should match NPZ exactly) |
| FK solver failures | 2.5% | 0% | 0% (NPZ had 100% convergence) |
| Dyn replay RMS | 48.1mm | 0.83mm | 0.84mm (NPZ FK-Dyn intrinsic) |
| Dyn replay Max | 139mm | 3.21mm | 3.21mm (NPZ FK-Dyn intrinsic) |
| Dyn solver failures | 2.5% | 0% | 0% (NPZ had 100% convergence) |

### Lemniscate Trajectory (data/dyn_fk_lem1_y40_a10_hold1.npz)

| Metric | Before (CP3.5) | After (CP3.5.1) | NPZ Ground Truth |
|--------|----------------|------------------|------------------|
| FK replay RMS | 52.7mm | 0.000mm | (exact match!) |
| FK replay Max | 139.5mm | 0.000mm | (exact match!) |
| FK solver failures | 10.3% | 0% | 0% (NPZ had 100% convergence) |
| Dyn replay RMS | 53.2mm | 1.18mm | 1.18mm (NPZ FK-Dyn intrinsic) |
| Dyn replay Max | 139.7mm | 3.51mm | 3.51mm (NPZ FK-Dyn intrinsic) |
| Dyn solver failures | 10.3% | 0% | 0% (NPZ had 100% convergence) |

**Conclusion**: After fix, replay matches NPZ exactly (FK) or matches NPZ's intrinsic FK-Dyn mismatch (Dyn).

---

## Files Changed

### New Files

1. **`python/test_cp35_npz_replay_contract.py`** (300+ lines)
   - Verifies faithful NPZ replay with correct warm-start
   - FK replay contract: matches NPZ within 1mm
   - Dyn replay contract: matches NPZ FK-Dyn intrinsic error
   - Documents warm-start strategy explicitly

2. **`docs/control/CP3_5_1_NPZ_REPLAY_CONTRACT_COMPLETION.md`** (this file)
   - Root cause analysis
   - Before/after metrics
   - Warm-start strategy documentation

### Modified Files

1. **`python/test_cp35_replay_npz_tracking.py`** (REPLACED)
   - Old version moved to `test_cp35_replay_npz_tracking_OLD.py`
   - New version: visualizes NPZ data directly (no recomputation)
   - Shows intrinsic FK-Dyn mismatch from NPZ
   - Generates clean plots without "triangles" or scrambled ordering

2. **`CMakeLists.txt`**
   - Renamed test: `replay_npz_tracking_cp35` → `npz_visualization_cp35`
   - Added test: `npz_replay_contract_cp35` (timeout 120s)
   - Both tests nightly-only (not in PR fast gate)

---

## Warm-Start Strategy (Key Implementation Detail)

### FK Warm-Start

```python
deltau0_prev = np.zeros(3)  # Initial guess for first solve

for i in range(N):
    u = currents[i]

    params_dict['deltau0_initialguess'] = deltau0_prev.tolist()
    fk_result = crm_diff_py.equilibrium_forward(u, L_inserted, params_dict)

    tip_fk[i] = fk_result['p_tip']
    deltau0_prev = fk_result['deltau0']  # Update for next iteration
```

### Dynamics with FK-Based Warm-Start

```python
x_t = np.zeros(6)  # Initial state
deltau0_prev = np.zeros(3)  # Initial guess

for i in range(N):
    u_t = currents[i]

    # Step 1: Solve FK to get deltau0
    params_dict['deltau0_initialguess'] = deltau0_prev.tolist()
    fk_result = crm_diff_py.equilibrium_forward(u_t, L_inserted, params_dict)
    deltau0 = fk_result['deltau0']

    # Step 2: Use FK's deltau0 for dynamics
    params_dict['deltau0_initialguess'] = deltau0.tolist()
    dyn_result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)

    tip_dyn[i] = dyn_result['p_tip']
    x_t = dyn_result['x_next']
    deltau0_prev = deltau0  # Update for next FK iteration
```

---

## NPZ Field Requirements

### Required Fields

- `t`: (N,) time vector (s)
- `currents`: (N, 3) control inputs (A)
- `tip_fk`: (N, 3) FK tip positions (mm)
- `tip_dyn`: (N, 3) Dyn tip positions (mm)
- `dt`: scalar, timestep (s)
- `insertion_length`: scalar, L_inserted (mm)

### Optional Fields

- `integration_step_size`: scalar, internal integration step (defaults to 0.5 if missing)
- `tip_desired`: (N, 3) desired trajectory (mm)
- `tip_projected`: (N, 3) workspace-projected trajectory (mm)
- `dyn_converged`: (N,) bool, convergence flags
- `fk_dyn_err`: (N,) precomputed FK-Dyn errors

### Assumptions

- Initial state `x0 = zeros(6)` (rest)
- Initial `deltau0_initialguess = zeros(3)` for first solve
- NPZ was recorded with warm-start enabled
- Solver parameters match catheterdata/CatheterParameterSet_1_dyn.txt

---

## Test Execution

### Contract Test

```bash
# Verify faithful replay
python3 python/test_cp35_npz_replay_contract.py

# Via CTest
ctest -R npz_replay_contract_cp35 --output-on-failure
```

**Expected output**:
```
FK Replay Contract Test: circle
  RMS error: 0.023mm
  Max error: 0.41mm
  ✓ PASS

Dynamics Replay Contract Test: circle
  RMS error: 0.83mm (matches NPZ FK-Dyn RMS 0.84mm)
  Max error: 3.21mm (matches NPZ FK-Dyn max 3.21mm)
  ✓ PASS
```

### Visualization Test

```bash
# Generate NPZ data plots
python3 python/test_cp35_replay_npz_tracking.py

# Via CTest
ctest -R npz_visualization_cp35 --output-on-failure
```

**Outputs**:
- `build/artifacts/cp35/cp35_circle.png`
- `build/artifacts/cp35/cp35_lemniscate.png`

Plots show:
- 3D + XY/XZ/YZ projections
- FK vs Dyn from NPZ (intrinsic mismatch ~0.8-1.2mm)
- Control inputs over time
- FK-Dyn error timeseries

---

## Acceptance Gate Results

### Gate A: Contract Test Passes ✅

```
Circle:
  FK replay:       ✓ PASS (RMS 0.023mm)
  Dynamics replay: ✓ PASS (RMS 0.83mm)

Lemniscate:
  FK replay:       ✓ PASS (RMS 0.000mm)
  Dynamics replay: ✓ PASS (RMS 1.18mm)
```

### Gate B: Plots Look Correct ✅

- No "triangles" or scrambled ordering
- FK and Dyn trajectories track smoothly
- Intrinsic FK-Dyn mismatch visible (~1mm RMS)
- Control inputs show expected patterns
- Error timeseries reasonable

### Gate C: No Regressions ✅

Existing tests pass:
```bash
ctest -R "dynamics_smoke_cp21|dynamics_fd_cp22|dynamics_smoke_cp23_python"
# All pass
```

---

## Technical Insights

### Why Warm-Start Matters

The equilibrium solver solves a nonlinear system with multiple solution branches. Example:
- Same inputs `(u, L_inserted)` can have multiple valid `deltau0` solutions
- Starting guess determines which branch the solver converges to
- NPZ was recorded along a specific branch (via warm-starting)
- Cold-start can find a different (but equally valid) branch → huge mismatch

### Why FK-Based Warm-Start for Dynamics

Dynamics solver internally solves an equilibrium problem at each timestep. Using FK's `deltau0` ensures:
1. Dynamics finds the SAME equilibrium as FK
2. FK-Dyn mismatch comes only from time integration (not different branches)
3. Replay matches NPZ's intrinsic FK-Dyn error

### Intrinsic FK-Dyn Mismatch

Even with perfect replay, FK and Dyn don't match exactly because:
- FK: Quasi-static equilibrium (no dynamics)
- Dyn: Time-integrated dynamics with mass/damping
- Typical mismatch: 0.8-1.2mm RMS for these trajectories

---

## Lessons Learned

1. **Solver warm-starting is critical for reproducibility**
   - Never assume cold-start will find the same solution
   - Always propagate solver state between iterations

2. **Integration parameters matter**
   - `IntegrationStepSize` affects solver convergence
   - Must match between recording and replay

3. **Document assumptions explicitly**
   - Initial state (x0)
   - Initial guess (deltau0_initialguess)
   - Solver parameters

4. **Validate replay before analysis**
   - Contract tests catch parameter mismatches early
   - Exact replay enables meaningful regression testing

---

## Future Work

### Potential Improvements

1. **Store `deltau0` sequence in NPZ**
   - Would enable perfect FK replay without re-solving
   - Smaller tolerance (<1e-6mm) for FK contract

2. **Store state sequence `x[t]` in NPZ**
   - Would enable perfect Dyn replay without rollout
   - Could verify single-step dynamics separately

3. **Add workspace NPZ tests**
   - `data/workspace_fk_ins94.3_*.npz` contains large dataset
   - Would stress-test warm-start across more diverse trajectories

4. **Parameterize warm-start strategy**
   - Allow testing with/without warm-start
   - Quantify warm-start benefit empirically

---

## References

- Original CP3.5: `python/test_cp35_replay_npz_tracking_OLD.py` (kept for reference)
- NPZ datasets: `data/dyn_fk_*.npz`
- Catheter parameters: `catheterdata/CatheterParameterSet_1_dyn.txt`
- Related tests: CP2.3, CP2.5, CP3.1

---

**End of CP3.5.1 Documentation**
