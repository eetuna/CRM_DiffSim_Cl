# CP5.0: Hybrid Controller Performance Closure - Findings

**Date**: 2026-01-02
**Status**: Partial Completion - Infrastructure Validated, Quality Gates Not Yet Met

---

## Executive Summary

CP5.0 successfully validated the hybrid controller infrastructure on real NPZ data using CP4.7.8 (hold-aware windowing) and CP4.7.9 (quality benchmarking). However, **quality gates were not met** due to the ensemble being trained on synthetic data that doesn't generalize well to real NPZ trajectories.

**Key Finding**: The hybrid controller calls MPC at 100% of timesteps on real data (vs. target ≤30%), indicating the ensemble's uncertainty is consistently above threshold when encountering real-world trajectories.

---

## What Was Accomplished

### 1. Real NPZ Window Quality Benchmark ✓

**Configuration**:
- Dataset: `dyn_fk_lem1_y40_a10_L94_hold1.npz` (real NPZ with hold periods)
- Windows tested: 10 valid windows (20 steps each, stride 20)
- Hold-aware filtering: Enabled (CP4.7.8) - automatically skipped 120 NaN timesteps
- Ensemble: cp45_ensemble_dagger (3 members, trained on synthetic data)
- Thresholds: τ_low=0.001, τ_high=0.01 (defaults, not calibrated for real data)

**Results**:

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Valid windows | 10 | >0 | ✓ PASS |
| MPC RMSE (mean) | 44.03mm | ≤10mm | ✗ FAIL |
| Hybrid RMSE (mean) | 44.03mm | ≤MPC+0.5mm | ✓ PASS |
| MPC call rate | 100.0% | ≤30% | ✗ FAIL |
| Speedup | 0.46× | ≥2.0× | ✗ FAIL |
| Safety violations | 0 | 0 | ✓ PASS |

**Runtime**: 2865s (~48 minutes for 10 windows)

**Interpretation**:
- MPC tracking error is high (44mm) on real data vs. golden data (<2mm), indicating real trajectories are significantly more challenging
- Hybrid controller achieves identical RMSE to MPC (good) but calls MPC at every step (bad)
- 100% MPC call rate indicates ensemble uncertainty is always above τ_high=0.01
- Slower than MPC-only due to ensemble overhead without policy execution benefits

---

## Why Quality Gates Failed

### Root Cause: Domain Shift

The cp45 ensemble was trained on **synthetic golden datasets** (smooth circles, lines, lemniscates) but is being validated on **real NPZ data** with:
- More complex, non-smooth trajectories
- Higher tracking errors (44mm vs 2mm baseline)
- Different dynamics patterns

When the ensemble encounters out-of-distribution states, it correctly reports high uncertainty, causing the hybrid controller to fall back to MPC at every step.

### Evidence

1. **Uncertainty behavior**: 100% MPC call rate means `uncertainty > τ_high` at all timesteps
2. **Performance**: Hybrid RMSE matches MPC exactly (policy was never executed)
3. **Baseline comparison**: Real data MPC RMSE (44mm) >> Golden data MPC RMSE (~2mm)

---

## Infrastructure Validation ✓

Despite not meeting quality gates, CP5.0 successfully demonstrated:

### 1. CP4.7.8 Hold-Aware Windowing
- ✓ Automatically filtered 120/320 (37.5%) hold-period timesteps
- ✓ Generated 10 valid windows from post-hold region (steps 120-320)
- ✓ Zero NaN-related failures

### 2. CP4.7.9 Quality Benchmarking
- ✓ Ran MPC baseline on 10 real windows
- ✓ Ran hybrid controller on 10 real windows
- ✓ Generated comprehensive quality statistics (RMSE distributions, MPC call rate, speedup)
- ✓ Zero safety violations (no physics solver failures)

### 3. End-to-End Pipeline
- ✓ Load real NPZ → health gate → windowing → MPC baseline → hybrid benchmark → quality report
- ✓ Graceful handling of hold periods
- ✓ JSON artifact generation for reproducibility

---

## Path to Full CP5.0 Closure

To meet the CP5.0 goal ("at least one real dataset meets all quality gates"), choose **one** of these strategies:

### Option A: Train Ensemble on Real Data Windows (Recommended)

**Approach**: Create training dataset from real NPZ windows

**Steps**:
1. Generate windowed training data from real NPZ files (skip hold periods)
2. Limit window length (e.g., 20-50 steps) to avoid rollout timeouts
3. Train ensemble with DAgger on these windows
4. Validate on held-out real windows

**Pros**:
- Ensemble learns real-world dynamics
- Low uncertainty on in-distribution real data
- Sustainable solution

**Cons**:
- Requires infrastructure changes (windowed DAgger training)
- Longer development time

**Estimated effort**: 2-3 days (implement windowed training, retrain ensemble)

---

### Option B: Calibrate Thresholds for High-Uncertainty Regime (Pragmatic)

**Approach**: Accept that ensemble has high uncertainty on real data; adjust thresholds to allow policy execution anyway

**Steps**:
1. Run threshold calibration on real NPZ windows (use `run_cp50_threshold_calibration.py`)
2. Search for τ_low, τ_high that allow ≥70% policy execution while maintaining safety
3. Re-run quality benchmark with new thresholds

**Pros**:
- Uses existing ensemble (no retraining)
- Fast iteration

**Cons**:
- May not find thresholds that satisfy both MPC call rate (≤30%) and safety
- Uncertainty estimates are legitimately high (ensemble hasn't seen this data)
- Brittle: ensemble will still fail on new real trajectories

**Estimated effort**: 4-8 hours (calibration grid search is compute-intensive)

**Expected outcome**: Likely to improve MPC call rate to 50-70% (not target 30%)

---

### Option C: Relax Quality Gates for Real Data (Compromise)

**Approach**: Acknowledge that real data is fundamentally harder than golden data

**Proposed gates for real NPZ**:
- MPC RMSE ≤ 50mm (relaxed from 10mm) - accounts for trajectory difficulty
- MPC call rate ≤ 70% (relaxed from 30%) - allows ensemble uncertainty
- Speedup ≥ 1.5× (relaxed from 2.0×) - less aggressive target

**With these gates, current results**:
- ✓ MPC RMSE: 44mm ≤ 50mm
- ✗ MPC call rate: 100% > 70%
- ✗ Speedup: 0.46× < 1.5×

**Pros**:
- Realistic expectations for real-world data
- May pass with Option B threshold tuning

**Cons**:
- Compromises on original performance targets
- Requires stakeholder buy-in

---

## Computational Notes

### Benchmark Runtime Breakdown

For 10 windows (20 steps each):
- Health gate (MPC horizon=10): 702s (~70s per window, ~3.5s per step)
- MPC baseline: ~350s (~35s per window)
- Hybrid controller: ~750s (~75s per window, slower due to ensemble overhead with 100% MPC calls)

**Total**: ~2865s (~48 minutes)

**Scaling estimate**:
- 50 windows (CP5.0 target): ~240 minutes (~4 hours)
- Full trajectory (320 steps after hold): ~1120s (~19 minutes) per trajectory

### Timeout Issues Encountered

1. **Full-trajectory training**: Rollout timeouts at 60/640 steps (DAgger rollout_timeout=180s insufficient for 640-step real trajectories)
2. **Large window budgets**: 50 windows × 50 steps exceeded 600s timeout for quality report

**Mitigation**: Used reduced scope (10 windows × 20 steps) for proof-of-concept

---

## Deliverables

### Artifacts Generated

1. **Quality Report**: `build/artifacts/cp50_quality_report.json`
   - 10 real NPZ windows validated
   - Full quality statistics (RMSE, MPC call rate, speedup)

2. **Consolidated Report**: `build/artifacts/cp50_consolidated_report.json`
   - Status: PARTIAL
   - Success criteria evaluation

3. **Summary**: `build/artifacts/cp50_summary.md`
   - Quick reference for component status

### Scripts Created

1. `python/train_cp50_real_ensemble.py` - Train ensemble on real NPZ only (not completed due to timeouts)
2. `python/run_cp50_threshold_calibration.py` - Calibrate thresholds for real data
3. `python/run_cp50_quality_report.py` - Wrapper for CP4.7.9 with CP5.0 config
4. `python/generate_cp50_report.py` - Consolidated report generator

---

## Recommendations

**For Immediate CP5.0 Closure (Next 1-2 Days)**:

1. **Run threshold calibration** with wider ranges:
   ```bash
   python3 python/run_cp50_threshold_calibration.py --max_configs 20
   ```
   - Search τ_low ∈ [1e-5, 1e-4, 5e-4, 1e-3, 5e-3]
   - Search τ_high ∈ [0.01, 0.05, 0.1, 0.2, 0.5]
   - Goal: Find configuration that achieves <70% MPC call rate while maintaining safety

2. **Re-run quality benchmark** with calibrated thresholds:
   ```bash
   python3 python/run_cp50_quality_report.py --max_windows 20
   ```

3. **Generate final report**:
   ```bash
   python3 python/generate_cp50_report.py
   ```

4. **If threshold calibration fails to meet gates**: Recommend relaxed gates (Option C) with justification that real-world data is fundamentally harder than synthetic golden datasets.

**For Long-Term Solution**:

- Implement windowed DAgger training (Option A)
- Collect more diverse real NPZ datasets for training
- Consider hybrid approach: train on mix of golden + real windows

---

## Conclusion

CP5.0 demonstrated that the **infrastructure works correctly** on real NPZ data:
- ✓ Hold-aware windowing successfully filters NaN periods
- ✓ Quality benchmarking pipeline runs end-to-end
- ✓ Zero safety violations (robust physics)

However, **performance gates are not met** because:
- ✗ Ensemble trained on synthetic data doesn't generalize to real trajectories
- ✗ Default thresholds not calibrated for real data uncertainty profiles

The system is **production-ready from a safety perspective** (zero failures) but **not yet achieving hybrid controller benefits** (100% MPC usage).

Next steps depend on priorities:
- **Fast closure**: Threshold calibration + relaxed gates
- **Sustainable solution**: Train on real data windows

---

## References

- CP4.7.8: Real NPZ Triage & Hold-Aware Windowing
- CP4.7.9: Quality-Gated Benchmarks
- CP4.5: Ensemble DAgger Training
- CP4.7.1: Threshold Calibration

---

**Artifacts Location**: `./build/artifacts/cp50_*`
**Time Invested**: ~2 hours (infrastructure exploration, quality benchmarking)
**Compute Time**: ~48 minutes (quality report on 10 windows)
