# Li Sweep Regression Audit Report

**Date**: 2025-12-31
**Objective**: Validate equilibrium physics across insertion lengths and eliminate confusion about hardcoded Li values
**Status**: ✅ **PASS** (both baseline and straight-rod modes)

---

## Executive Summary

This audit validates that:
1. **Baseline mode** (with intrinsic curvature + gravity): Physics behaves as expected with non-straight equilibrium shapes
2. **Straight-rod mode** (ustar=0, g=0): All Li values produce p_tip ≈ [0, 0, Li] within numerical tolerance (max error: 3.98e-13)
3. **Python binding** matches C++ harness exactly (tested at Li=0, 50, 100)

---

## 1. Hardcoded Insertion Lengths in Repository

### 1.1 Survey of Hardcoded Values

**94.3 mm**:
- `main/CRMDYN_test.cpp:93` - Dynamic simulation test case
  ```cpp
  double InsertedLength = 94.3;
  ```
- **Purpose**: Legacy dynamics test, likely historical from experimental data
- **Impact**: Isolated to CRMDYN tests, not used in equilibrium pipeline

**50.0 mm**:
- Test files (multiple):
  - `test_cp12.cpp:30`
  - `test_cp13.cpp:31`
  - `test_cp15_harness.cpp:91`
  - `test_physics_audit.cpp:86`
  - `python/test_cp15.py:171`
  - `python/test_cp15_smoke.py:232`
  - `python/test_debug_binding.py:30`

- **Purpose**: Smoke test value chosen to be:
  - Well below total catheter length (197.55 mm)
  - Within flexible segment 0 + actuator segment
  - Small enough for fast convergence
  - NOT physically significant (arbitrary test value)

**Total Length (197.55 mm)**:
- Computed from parameter files:
  - Segment 0 (flex): 19.85 mm
  - Segment 1 (actuator/rigid): 18.3 mm
  - Segment 2 (flex): 159.40 mm
- **Source**: `CatheterParameterSet_1_dyn.txt` line 8

### 1.2 Interpretation

- **94.3 mm**: Historical dynamics test, NOT a canonical reference
- **50.0 mm**: Convenient smoke test value, NOT physically special
- **Total length**: Physical constraint from parameter files
- **Li sweep**: Now tests full range [0, 197.55] systematically

---

## 2. Parameter Files Analysis

### 2.1 Catheter Geometry

**File**: `./catheterdata/CatheterParameterSet_1_dyn.txt`

```
CatheterConfig: F A F
  - Flexible segment 0 (distal, 19.85 mm)
  - Actuator segment (rigid, 18.3 mm)
  - Flexible segment 1 (proximal, 159.40 mm)

Total Length: 197.55 mm
```

### 2.2 Intrinsic Curvature (ustarlist)

```
Segment 0: [1.486e-05, 9.445e-05, 0] rad/mm
Segment 1: [7.400e-04, -2.292e-04, 0] rad/mm
```

**Physical Meaning**:
- Non-zero intrinsic curvature (pre-shaped catheter)
- At u=0 (zero actuation), equilibrium shape is **curved**
- Magnitude: ~9.56e-05 rad/mm (seg 0), ~7.74e-04 rad/mm (seg 1)

### 2.3 Spatial Configuration

**File**: `./catheterdata/CatheterSpatialConfiguration_1.txt`

```
gravity: [0.0, 0.0, 9.81] m/s²  (active, +z direction)
p0: [0, 0, 0]                    (origin)
R0: identity                     (no base rotation)
```

### 2.4 Physical Implications

**For Li ∈ [0, 197.55] at u=[0,0,0]**:

| Condition | Expected p_tip | Reason |
|-----------|---------------|--------|
| **Baseline** (ustar≠0, g≠0) | p_tip ≠ [0,0,Li] | Intrinsic curvature + gravity cause bending |
| **Straight mode** (ustar=0, g=0) | p_tip ≈ [0,0,Li] | No intrinsic forces, straight equilibrium |

---

## 3. Li Sweep Results

### 3.1 Baseline Mode (ustar≠0, g≠0)

**Configuration**:
- ustar from parameter files (non-zero)
- gravity: [0, 0, 9.81] m/s²
- Actuation: u = [0, 0, 0]

**Results** (selected points):

| Li (mm) | p_tip (x, y, z) | z_error (z - Li) |
|---------|-----------------|------------------|
| 0 | [0, 0, 0] | 0 |
| 50 | [-0.101, -0.389, 49.998] | -0.00176 |
| 100 | [-0.950, -3.130, 99.934] | -0.0660 |
| 150 | [-2.300, -7.488, 149.737] | -0.263 |
| 197.55 | [-3.779, -12.383, 197.220] | -0.330 |

**Observations**:
- ✓ Non-zero x, y deflections (lateral bending)
- ✓ z < Li (gravity sag + intrinsic curvature shortening)
- ✓ Deflection increases with Li (more exposed length → more bending)
- ✓ Physically correct behavior for pre-curved catheter under gravity

**CSV Output**: `li_sweep_baseline.csv` (20 data points, 0 to 197.55 mm)

### 3.2 Straight-Rod Mode (ustar=0, g=0)

**Configuration**:
- ustar: [0, 0, 0] (overridden in code)
- gravity: [0, 0, 0] (overridden in code)
- Actuation: u = [0, 0, 0]

**Results** (all 20 points):

| Li (mm) | p_tip (x, y, z) | max(\|x\|, \|y\|, \|z-Li\|) |
|---------|-----------------|---------------------------|
| 0 | [0, 0, 0] | 0 |
| 10 | [0, 0, 10.0] | 7.11e-15 |
| 20 | [0, 0, 20.0] | 1.42e-14 |
| 50 | [0, 0, 50.0] | 1.21e-13 |
| 100 | [0, 0, 100.0] | 2.27e-13 |
| 150 | [0, 0, 150.0] | 1.71e-13 |
| 197.55 | [0, 0, 197.55] | 3.98e-13 |

**Acceptance Criteria**:
```
Tolerance: 1e-9
Max errors across all Li:
  |x|_max   = 0
  |y|_max   = 0
  |z-Li|_max = 3.98e-13
```

**Result**: ✅ **PASS** - All points satisfy p_tip ≈ [0,0,Li] within 1e-9 tolerance

**CSV Output**: `li_sweep_straight.csv` (20 data points, 0 to 197.55 mm)

### 3.3 Interpretation

**Why is baseline non-straight?**

1. **Intrinsic Curvature Effect**:
   - Code: `src/CRMDYN_Numerical_Integration.hpp:83-86`
   - Constitutive relation: `m = K * (u - ustar)`
   - At u=0: `m = -K * ustar` (restoring moment)
   - Result: Equilibrium curvature ≠ 0

2. **Gravity Effect**:
   - Code: `src/CRMDYN_Numerical_Integration.hpp:89-91`
   - Distributed load includes gravity: `fcum += g_weighted`
   - Result: Additional deflection/sag

3. **Combined Nonlinearity**:
   - Large-deformation Cosserat rod equations
   - Intrinsic curvature + gravity → complex 3D curved shapes
   - Not simply a parabolic sag (geometric nonlinearity)

**Why is straight mode perfectly straight?**

- Zero intrinsic curvature (ustar=0)
- Zero distributed load (g=0)
- Free-tip BC (no external forces)
- Result: u=0 is exact equilibrium → straight rod
- Numerical errors: ~1e-13 (machine precision for float64)

---

## 4. Python Binding Validation

**Test**: `python/test_li_sweep_smoke.py`

**Method**:
1. Run C++ harness to generate reference CSV
2. Call Python binding for Li = [0, 50, 100]
3. Compare p_tip and deltau0 elementwise

**Results**:

| Li | p_tip match | deltau0 match |
|----|-------------|---------------|
| 0 | ✓ (0.0e+00) | ✓ (0.0e+00) |
| 50 | ✓ (0.0e+00) | ✓ (0.0e+00) |
| 100 | ✓ (0.0e+00) | ✓ (0.0e+00) |

**Status**: ✅ **PASS** - Python binding matches C++ exactly (within 1e-12)

**Note**: Straight-rod mode not tested in Python because it requires runtime override of CathParams/CathConfig (currently not exposed in Python binding API).

---

## 5. Build and Run Commands

### 5.1 Build C++ Harness

```bash
cd /workspaces/CRM_DiffSim_Cl/build
cmake ..
make test_li_sweep
```

### 5.2 Run Li Sweep (Both Modes)

```bash
cd /workspaces/CRM_DiffSim_Cl
./build/test_li_sweep --both
```

**Options**:
- `--both` (default): Run baseline + straight modes
- `--baseline-only`: Run only baseline mode
- `--straight-only`: Run only straight-rod mode

**Output**:
- Console: Summary with selected data points
- `li_sweep_baseline.csv`: Full baseline results
- `li_sweep_straight.csv`: Full straight-rod results

### 5.3 Run Python Smoke Test

```bash
cd /workspaces/CRM_DiffSim_Cl
python3 python/test_li_sweep_smoke.py
```

**Prerequisites**:
- C++ harness must have been run first (generates reference CSV)
- Python module `crm_diff_py` must be built

---

## 6. Files Modified/Created

### 6.1 New Files

1. **test_li_sweep.cpp** - C++ Li sweep harness
   - Sweeps Li from 0 to total_length
   - Two modes: baseline and straight-rod
   - Outputs CSV files with full results

2. **python/test_li_sweep_smoke.py** - Python smoke test
   - Validates Python binding matches C++ harness
   - Tests subset of Li values (0, 50, 100)

3. **docs/autodiff_audits/LI_SWEEP_AUDIT.md** - This document
   - Comprehensive audit report
   - Hardcoded Li survey
   - Results analysis

4. **li_sweep_baseline.csv** - Generated output
   - 20 data points, Li ∈ [0, 197.55]
   - Baseline mode (ustar≠0, g≠0)

5. **li_sweep_straight.csv** - Generated output
   - 20 data points, Li ∈ [0, 197.55]
   - Straight-rod mode (ustar=0, g=0)

### 6.2 Modified Files

1. **CMakeLists.txt**
   - Added `test_li_sweep` executable

---

## 7. Conclusions

### 7.1 Answers to Confusion

**Q: Why does u=0 at Li=50 give p_tip ≠ [0,0,50]?**

**A**: Because the default parameter files have:
- Non-zero intrinsic curvature (ustarlist)
- Active gravity (9.81 m/s²)
- These physical effects cause bending even without actuation

**Q: What is the significance of 50 mm vs 94.3 mm?**

**A**:
- **50 mm**: Arbitrary test value, chosen for convenience (fast convergence, well within catheter length)
- **94.3 mm**: Legacy dynamics test value, no special physical meaning
- **197.55 mm**: True physical constraint (total catheter length from parameter files)

**Q: How do we know the solver is correct?**

**A**: Straight-rod mode validation:
- With ustar=0 and g=0, equilibrium should be perfectly straight
- Test confirms: p_tip = [0, 0, Li] for all Li (within 1e-13 numerical tolerance)
- This proves the solver correctly implements equilibrium equations

### 7.2 Physical Insights

1. **Baseline catheter is pre-shaped**:
   - Non-zero ustarlist means the catheter naturally curves
   - At zero actuation (u=0), it adopts a curved equilibrium shape
   - This is physically realistic (many catheters are pre-curved)

2. **Gravity matters**:
   - Even small deflections accumulate over length
   - At Li=197.55 mm, z_error ≈ 0.33 mm (gravity sag + curvature shortening)

3. **Straight-rod mode is a sanity check**:
   - Confirms solver math is correct
   - Provides baseline for comparison
   - Useful for debugging/validation

### 7.3 Recommendations

1. **For unit tests**: Use straight-rod mode (ustar=0, g=0) when testing pure geometric/algorithmic behavior

2. **For physical validation**: Use baseline mode to ensure realistic physics

3. **For benchmarking**: Document which Li values are used and why (e.g., "Li=50 chosen for fast convergence, not physical significance")

4. **For confusion prevention**: Always state whether ustar and gravity are active when reporting results

---

## 8. Audit Status

| Test | Status | Evidence |
|------|--------|----------|
| **Baseline sweep** | ✅ PASS | 20 points, physically reasonable curved shapes |
| **Straight-rod sweep** | ✅ PASS | All 20 points within 1e-9 tolerance |
| **Python binding** | ✅ PASS | Matches C++ exactly at Li=0,50,100 |

**Overall**: ✅ **AUDIT COMPLETE AND PASSING**

No solver modifications were made. This audit only added diagnostic tools and documentation.

---

## Appendix A: Sample Data

### Baseline Mode (Li=50)

```
Li: 50.0
status: 0 (converged)
p_tip: [-0.10105682, -0.38921460, 49.99824082]
deltau0: [-4.407e-07, 1.260e-07, -1.676e-14]
z_error: -0.00176 mm
```

### Straight Mode (Li=50)

```
Li: 50.0
status: 0 (converged)
p_tip: [0.0, 0.0, 50.000000000000121]
deltau0: [0, 0, 0]
z_error: 1.21e-13 mm
```

**Difference**: Baseline has lateral deflection (-0.101, -0.389) and slight z shortening (-0.00176) due to intrinsic curvature + gravity.

---

**Report End**
