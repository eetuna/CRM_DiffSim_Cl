# CP5.1: Real-NPZ Windowed Ensemble Training - Completion Audit

**Status**: ✓ Implemented
**Date**: 2026-01-02
**Assignee**: Claude Sonnet 4.5

---

## Overview

CP5.1 implements **windowed ensemble training on real NPZ data** to address the domain shift problem identified in CP5.0. Instead of training on full-trajectory rollouts (which timeout on 640-step real datasets) or synthetic golden data (which doesn't generalize), CP5.1 trains directly on **valid windows** extracted from real NPZ files using hold-aware filtering (CP4.7.8).

This checkpoint delivers:
1. **Windowed behavioral cloning training** - Collects MPC expert actions on real windows, trains ensemble via supervised learning
2. **Smoke test** (<60s CI gate) - Validates training pipeline on minimal dataset
3. **Nightly closure report** - Full pipeline: train → calibrate → validate → report
4. **Infrastructure reuse** - Leverages CP4.7.8 (hold-aware windowing), CP4.7.9 (quality benchmarking), existing GRUPolicy/EnsemblePolicy

---

## Motivation: Why Windowed Training?

### CP5.0 Findings

CP5.0 validated hybrid controller infrastructure on real NPZ but **quality gates failed**:
- **MPC call rate**: 100% (target ≤30%)
- **Speedup**: 0.46× (target ≥2.0×)
- **Root cause**: Ensemble trained on synthetic golden data (smooth trajectories, ~2mm RMSE) doesn't generalize to real NPZ data (complex trajectories, ~44mm RMSE)

### CP5.1 Solution

**Train directly on real NPZ windows** instead of synthetic data:
- Extract valid windows from real datasets using health gate + hold-aware filtering
- Run MPC expert on each window to collect reference actions
- Train ensemble via behavioral cloning (supervised learning)
- **Result**: Ensemble learns real-world dynamics → low uncertainty on real data → reduced MPC call rate

### Why Windows Instead of Full Trajectories?

1. **Computational feasibility**: Real NPZ trajectories are 640 steps (vs. 20-50 for golden data) → DAgger rollouts timeout (>180s per trajectory)
2. **Data efficiency**: Windows provide dense training signal without full rollout overhead
3. **Leverages existing infrastructure**: CP4.7.8 hold-aware windowing already validated
4. **Scalable**: Can train on many windows from few datasets

---

## Implementation

### 1. Windowed Behavioral Cloning (`train_cp51_real_window_ensemble.py`)

**Training Pipeline**:

```
1. Load real NPZ datasets
   ↓
2. Run health gate with hold-aware filtering (CP4.7.8)
   → Extract valid windows
   ↓
3. For each valid window:
   - Initialize state
   - Run MPC expert at each timestep
   - Collect (p_tip, p_ref, u_expert) samples
   ↓
4. Create train/val split (80/20)
   ↓
5. For each ensemble member:
   - Create GRUPolicy with unique seed
   - Train via supervised learning (MSE loss)
   - Early stopping on validation loss
   - Save checkpoint
   ↓
6. Save ensemble metadata + training metrics
```

**Key Features**:
- **Hold-aware**: Automatically skips NaN hold periods
- **Health gate**: Only uses windows that pass physics validation
- **Behavioral cloning**: Simple supervised learning (no rollouts needed)
- **Early stopping**: Prevents overfitting (patience=10)
- **Metrics tracking**: Records val loss, training time per member

**Configuration**:
```python
# Default
--max_windows 20      # Windows to collect
--window_steps 20     # Window size (timesteps)
--stride_steps 20     # Stride between windows
--n_members 3         # Ensemble members
--epochs 50           # Training epochs per member

# Fast mode
--fast                # 5 windows, 20 epochs
```

**Runtime** (default config):
- Window collection: ~15s per window (MPC expert)
- Training: ~5s per member (50 epochs, ~20 samples)
- **Total**: ~6 minutes for 20 windows, 3 members

**Artifacts**:
- `build/artifacts/cp51_real_ensemble_ensemble_metadata.json` - Ensemble config
- `build/artifacts/cp51_real_ensemble_member{i}_seed{s}_policy.pth` - Checkpoints
- `build/artifacts/cp51_real_ensemble_metrics.json` - Training metrics

---

### 2. Smoke Test (`test_cp51_real_window_training_smoke.py`)

**Purpose**: Fast CI test that validates windowed training works.

**Configuration**:
- Dataset: 1 real NPZ
- Windows: 2 (10 steps each)
- Members: 1
- Epochs: 10
- MPC: horizon=3, max_iters=3 (fast)

**Acceptance Criteria**:
- ✓ Runtime < 60s
- ✓ At least 1 valid window collected
- ✓ Training completes without errors
- ✓ Validation loss is finite
- ✓ Artifacts created

**Actual Results** (verified 2026-01-02):
```
Time elapsed: 18.6s
Windows collected: 2
Training samples: 20
Best val loss: 0.008537

Acceptance: time < 60s: ✓ PASS
Acceptance: windows >= 1: ✓ PASS
Acceptance: finite loss: ✓ PASS

✓ SMOKE TEST PASSED
```

**Command**:
```bash
python3 python/test_cp51_real_window_training_smoke.py
```

---

### 3. Nightly Closure Report (`run_cp51_real_closure_report.py`)

**Purpose**: Full CP5.1 pipeline to validate whether real-window training closes the domain shift gap.

**Pipeline**:
```
[1/3] Train CP5.1 ensemble on real windows (or skip if exists)
      ↓
[2/3] Run quality benchmark with CP5.1 ensemble
      - Same as CP5.0 but using new ensemble
      ↓
[3/3] Evaluate closure criteria
      - MPC RMSE ≤ 50mm (relaxed for real data)
      - Hybrid RMSE ≤ MPC + 0.5mm
      - MPC call rate ≤ 30% (target)
      - Speedup ≥ 2.0×
      - Safety violations == 0
```

**Configuration**:
```bash
# Full run (default)
python3 python/run_cp51_real_closure_report.py

# Skip training if ensemble exists
python3 python/run_cp51_real_closure_report.py --skip_training

# Custom budgets
python3 python/run_cp51_real_closure_report.py \
  --max_windows 30 \
  --window_steps 20
```

**Outputs**:
- `build/artifacts/cp51_closure_report.json` - Closure evaluation
- Quality metrics (RMSE, MPC call rate, speedup)
- Gate pass/fail status
- Comparison to CP5.0 baseline

**Expected Runtime**: ~30-60 minutes (20 windows for training + 20 for validation)

---

## Testing and Validation

### Smoke Test (Verified 2026-01-02)

**Command**:
```bash
python3 python/test_cp51_real_window_training_smoke.py
```

**Output**:
```
================================================================================
CP5.1: Real-NPZ Windowed Training - Smoke Test
================================================================================
Target: <60s, 1 member, 1 dataset, 2 windows, 10 epochs

[1/4] Loading real NPZ datasets...
✓ Using: dyn_fk_lem1_y40_a10_L94_hold1.npz

[2/4] Loading physics parameters...
✓ Loaded physics parameters

[3/4] Collecting valid windows...
✓ Found 2 valid window(s)

[4/4] Training tiny ensemble...
================================================================================
CP5.1: Windowed Ensemble Training
================================================================================
Windows: 2
Ensemble members: 1
Epochs: 10

[1/2] Collecting MPC expert data on windows...
  Collection time: 8.6s
  Train: 16 samples
  Val:   4 samples

[2/2] Training 1 ensemble members...
  Member 0 (seed=42)
    ✓ Trained in 2.3s, val_loss=0.008537

✓ Ensemble metadata saved
✓ Training metrics saved

================================================================================
SMOKE TEST SUMMARY
================================================================================
Time elapsed: 18.6s
Windows collected: 2
Training samples: 20
Best val loss: 0.008537

Acceptance: time < 60s: ✓ PASS
Acceptance: windows >= 1: ✓ PASS
Acceptance: finite loss: ✓ PASS
================================================================================
✓ SMOKE TEST PASSED
================================================================================
```

**Interpretation**:
- ✓ Runs in 18.6s (well under 60s budget)
- ✓ Successfully collects valid windows from real NPZ with hold-aware filtering
- ✓ MPC expert data collection works (20 samples from 2 windows)
- ✓ Training converges (val_loss=0.0085)
- ✓ Artifacts created correctly

---

### CTest Integration

**Added Tests**:

1. **Smoke Test** (CI gate, Test #46):
```cmake
add_test(NAME real_window_training_smoke_cp51
         COMMAND python3 python/test_cp51_real_window_training_smoke.py)
set_tests_properties(real_window_training_smoke_cp51 PROPERTIES TIMEOUT 60)
```

2. **Nightly Closure Report** (Test #47):
```cmake
add_test(NAME real_window_closure_report_cp51
         COMMAND python3 python/run_cp51_real_closure_report.py)
set_tests_properties(real_window_closure_report_cp51 PROPERTIES TIMEOUT 1800)
set_tests_properties(real_window_closure_report_cp51 PROPERTIES LABELS "nightly")
```

**Verification**:
```bash
# Check registration
ctest -N | grep cp51

# Run smoke test
ctest -R real_window_training_smoke_cp51 --output-on-failure

# Run nightly closure
ctest -R real_window_closure_report_cp51 -V

# Run all nightly tests
ctest -L nightly --output-on-failure
```

---

## Architecture Details

### Windowed Data Collection

**Function**: `collect_window_expert_data()`

**Process**:
1. For each windowed dataset:
   - Initialize state: `x_t = zeros(6)`
   - For each timestep in window:
     - Get MPC horizon reference
     - Solve iLQR with `jacobian_mode="cpp"` (CP4.4c fast Jacobians)
     - Record `(p_tip_t, p_ref[t], u_expert)`
     - Execute action, update state
   - Skip window if MPC or dynamics fails
2. Return arrays: `(p_tips, p_refs, u_experts)`

**Robustness**:
- Skips windows with MPC solver failures
- Skips windows with dynamics failures (status != 0)
- Only uses **complete** windows (all timesteps succeeded)

---

### Behavioral Cloning Training

**Function**: `train_policy_bc()`

**Loss**: MSE between predicted and expert actions
```python
loss = MSELoss(u_pred, u_expert)
```

**Training Loop**:
```python
for epoch in range(epochs):
    for p_tips, p_refs, u_experts in train_loader:
        hidden = policy.init_hidden(batch_size)
        u_pred, hidden = policy(p_tips, p_refs, hidden)

        loss = MSE(u_pred, u_experts)

        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm(policy.parameters(), 1.0)
        optimizer.step()

    # Validation
    val_loss = evaluate(val_loader)

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break
```

**Hyperparameters**:
- Optimizer: Adam, lr=1e-3
- Batch size: 64
- Gradient clipping: max_norm=1.0
- Early stopping patience: 10 epochs

---

## Constraints Respected

✓ **No CP2/CP3 physics modifications** - Only data/pipeline changes
✓ **Reuses existing infrastructure**:
  - CP4.7.8 hold-aware windowing
  - CP4.7.4 sliding-window health gate
  - CP4.7.9 quality benchmarking
  - GRUPolicy / EnsemblePolicy (CP4.5)
  - Fast Jacobians (CP4.4c)
✓ **Bounded budgets**:
  - Smoke test: <60s
  - Nightly: 30min timeout
✓ **No plan mode** - Direct implementation
✓ **No new CP numbers under CP4** - Named CP5.1

---

## Expected Impact on CP5.0 Metrics

### Baseline (CP5.0 with synthetic-trained ensemble)

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| MPC RMSE | 44.03mm | ≤50mm | ✓ |
| Hybrid RMSE | 44.03mm | ≤MPC+0.5mm | ✓ |
| **MPC call rate** | **100.0%** | **≤30%** | **✗** |
| **Speedup** | **0.46×** | **≥2.0×** | **✗** |
| Safety violations | 0 | 0 | ✓ |

### Expected (CP5.1 with real-window-trained ensemble)

After training on real NPZ windows, the ensemble should:
- **Recognize real-world states** → lower uncertainty
- **Execute policy more often** → MPC call rate 10-30% (target)
- **Achieve speedup** → 2-4× faster than MPC-only

**Hypothesis**: MPC call rate will drop from 100% to ≤30% because the ensemble has actually seen real-world dynamics during training.

**Verification**: Run `run_cp51_real_closure_report.py` to validate.

---

## Deliverables

✓ **Training Script**: `python/train_cp51_real_window_ensemble.py`
  - Windowed data collection from real NPZ
  - Behavioral cloning training
  - Ensemble metadata + checkpoints

✓ **Smoke Test**: `python/test_cp51_real_window_training_smoke.py`
  - <60s runtime (actual: 18.6s)
  - Validates pipeline works
  - CI gate

✓ **Nightly Report**: `python/run_cp51_real_closure_report.py`
  - Full closure pipeline
  - Quality gate evaluation
  - Comparison to CP5.0

✓ **CTest Integration**:
  - Test #46: `real_window_training_smoke_cp51` (CI)
  - Test #47: `real_window_closure_report_cp51` (nightly)

✓ **Completion Audit**: `docs/audits/CP5_1_REAL_WINDOW_TRAINING_COMPLETION.md` (this document)

---

## Usage Examples

### Train Ensemble on Real Windows

```bash
# Default: 20 windows, 3 members, 50 epochs
PYTHONPATH=build:python python3 python/train_cp51_real_window_ensemble.py

# Fast mode: 5 windows, 3 members, 20 epochs
PYTHONPATH=build:python python3 python/train_cp51_real_window_ensemble.py --fast

# Custom config
PYTHONPATH=build:python python3 python/train_cp51_real_window_ensemble.py \
  --max_datasets 2 \
  --max_windows 30 \
  --window_steps 20 \
  --n_members 5 \
  --epochs 100
```

### Run Closure Report

```bash
# Full pipeline (train + benchmark + evaluate)
PYTHONPATH=build:python python3 python/run_cp51_real_closure_report.py

# Use existing ensemble
PYTHONPATH=build:python python3 python/run_cp51_real_closure_report.py --skip_training
```

### CI Testing

```bash
# Smoke test only
ctest -R real_window_training_smoke_cp51 --output-on-failure

# Nightly full closure
ctest -R real_window_closure_report_cp51 -V
```

---

## Next Steps

After CP5.1 implementation:

1. **Run nightly closure report** to validate quality gate closure:
   ```bash
   ctest -R real_window_closure_report_cp51 -V
   ```

2. **If gates pass**: Document CP5.0 closure achieved via CP5.1 training

3. **If gates partially pass** (e.g., MPC call rate 40% instead of 30%):
   - Run threshold calibration on CP5.1 ensemble
   - Increase training data (more windows, more datasets)

4. **If gates fail**: Investigate root causes:
   - Check ensemble uncertainty on real windows
   - Verify policy executes correctly
   - Review window selection (are they representative?)

---

## Known Limitations

1. **Small training datasets**: Real NPZ datasets are limited (6 total, ~200 valid timesteps each after hold filtering)
   - Mitigation: Collect more real NPZ datasets
   - Mitigation: Data augmentation (if applicable)

2. **Window size trade-off**:
   - Small windows (10-20 steps): Fast training, but limited temporal context
   - Large windows (50+ steps): More context, but MPC expert slower
   - Current choice: 20 steps (balance)

3. **No online learning**: Once trained, ensemble is fixed
   - Future: Implement online DAgger for continuous improvement

4. **Generalization to new datasets**: Trained on specific real datasets, may not generalize to significantly different trajectories
   - Mitigation: Train on diverse set of real datasets

---

## References

- **CP5.0**: Hybrid Controller Performance Closure (domain shift problem)
- **CP4.7.8**: Real-NPZ Triage + Hold-Aware Windowing
- **CP4.7.4**: Sliding-Window Health Gate
- **CP4.7.9**: Quality-Gated Benchmarks
- **CP4.5**: Ensemble DAgger Training
- **CP4.4c**: Fast CPP Jacobians

---

## Sign-Off

**Implementation**: ✓ Complete
**Testing**: ✓ Verified (smoke test passed)
**Documentation**: ✓ Complete

**Acceptance Criteria Met**:
- [✓] Windowed training pipeline implemented
- [✓] Behavioral cloning on real NPZ windows works
- [✓] Smoke test passes (<60s, verified)
- [✓] CTest integration complete
- [✓] Nightly closure report ready
- [✓] No CP2/CP3 physics modifications
- [✓] Reuses existing infrastructure

**Impact**:
- **Technical**: Addresses CP5.0 domain shift by training on real data
- **Expected**: MPC call rate reduction from 100% to ≤30%
- **Validation**: Pending nightly closure report results

**Reproduction Commands**:
```bash
# Smoke test (18.6s)
python3 python/test_cp51_real_window_training_smoke.py

# Full training (6-10 minutes)
python3 python/train_cp51_real_window_ensemble.py --max_windows 20

# Nightly closure report (30-60 minutes)
python3 python/run_cp51_real_closure_report.py

# Via CTest
ctest -R cp51 --output-on-failure
```
