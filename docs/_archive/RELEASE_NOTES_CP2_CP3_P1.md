# RELEASE NOTES: CP2/CP3 Foundation + P1 Hardening

**Release Branch**: `stable/cp2-cp3-p1`
**Date**: 2026-01-01
**Base**: Commit 26201c2 ("last edits before moving next stage")
**Status**: ✅ Production-Ready Snapshot (No Main Merge)

---

## 🎯 Purpose

This release branch provides a **stable, tested, and documented snapshot** of the CRM Differentiable Simulator with:

1. **CP2.x** (Differentiable Dynamics): Full backward-mode automatic differentiation through catheter dynamics
2. **CP3.x** (Control Stack): Linearization, iLQR trajectory optimization, and MPC tracking
3. **P1.x** (Hardening): Robustness testing, performance optimization, and API versioning

**NOT merged to `main`** — this branch exists as a shareable reference implementation for:
- External collaborators requiring stable API
- Integration testing with downstream systems
- Reproducible research baselines
- Future development branching points

---

## 📦 What's Included

### CP2.x: Differentiable Dynamics (CP2.1 → CP2.6)

**Core Functionality**:
- ✅ **Forward Dynamics**: Single-step integration `(x_t, u_t, dt) → x_{t+1}` with cached Jacobians
- ✅ **Backward Pass**: Efficient VJP through dynamics using cached data
- ✅ **Equilibrium Primitives**: Forward/backward for static tip positioning
- ✅ **Matrix Dependence**: Correct handling of ∂M/∂u, ∂D/∂u, ∂K/∂u via nested finite differences
- ✅ **PyTorch Integration**: `torch.autograd.Function` wrappers with `gradcheck` validation
- ✅ **Multi-Step Rollouts**: Validated up to T=20 steps with stable backpropagation

**API Exposure** (Python via pybind11):
```python
import crm_diff_py

# Equilibrium (static solving)
result = crm_diff_py.equilibrium_forward(u, L_inserted, params_dict)
grad = crm_diff_py.equilibrium_backward(result, grad_p_tip)

# Dynamics (time-stepping)
result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)
grad = crm_diff_py.dynamics_backward(result, grad_x_next, params_dict)
```

**PyTorch Wrappers**:
```python
from crm_dynamics_torch import dynamics_step
from crm_equilibrium import equilibrium_tip_position

# Single-step differentiable dynamics (autograd-compatible)
x_next = dynamics_step(x_t, u_t, dt, L_inserted, params_dict)

# Static equilibrium with gradients
p_tip = equilibrium_tip_position(u, L_inserted, params_dict)
```

**Key Features**:
- State: `x ∈ ℝ⁶` (base curvature + velocity)
- Control: `u ∈ ℝ³` (3 actuator currents in Amperes)
- Observables: `p_tip ∈ ℝ³` (tip position in mm)
- Gradient Accuracy: Relative error < 1e-4 (FD-validated)
- Performance: ~0.1ms forward, ~0.3ms backward (single step)

**Checkpoints**:
- **CP2.1**: C++ core implementation
- **CP2.2**: Finite difference validation + matrix-dependence fix
- **CP2.3**: Python bindings (pybind11)
- **CP2.4**: PyTorch autograd wrapper + gradcheck
- **CP2.5**: Multi-step rollout smoke tests (T=20)
- **CP2.6**: CI integration (PR + nightly gates)

---

### CP3.x: Control Stack (CP3.0 → CP3.4)

**Core Functionality**:
- ✅ **Linearization**: Extract A_t, B_t matrices at any operating point
- ✅ **iLQR Solver**: Iterative Linear Quadratic Regulator with line search and optional LQR warm-start
- ✅ **MPC Framework**: Receding-horizon model predictive control
- ✅ **Terminal Cost Options**: Quadratic or exact FD-based Hessian for final state

**API Exposure** (Python control module):
```python
from control.ilqr import iLQRSolver

solver = iLQRSolver(
    dynamics_fn=dynamics_step,
    horizon=T,
    Q=Q, R=R, Qf=Qf,
    max_iters=100,
    use_lqr_warmstart=True,
    terminal_hessian_mode='fd_exact'  # or 'quadratic'
)

result = solver.solve(x0, u_init, target_state, params_dict)
# Returns: u_opt (control sequence), x_traj (state trajectory), metrics
```

**Key Features**:
- Horizon: T = 5–50 steps (tested)
- Convergence: Armijo line search with backtracking
- Warm-start: LQR initial guess option (P1-2)
- Terminal Hessian: FD-based exact option for robustness (P1-3)
- Performance: ~2–10s for T=10–20 (depends on convergence)

**Known Limitations** (see Section 7 below):
- ⚠️ iLQR line search may fail with high terminal weights (passive dynamics issue)
- ⚠️ Gauss-Newton Hessian approximation can be too optimistic
- ✅ Mitigation: Use MPC with short horizons (T=5–10) or exact terminal Hessian

**Checkpoints**:
- **CP3.0**: MPC design document
- **CP3.1**: Linearization validation (C++ + Python)
- **CP3.2**: iLQR trajectory optimization
- **CP3.2.1**: iLQR descent regression test
- **CP3.3**: MPC receding-horizon tracking
- **CP3.4**: Control CI integration (nightly-only for long tests)

---

### P1.x: Hardening & Optimization (P1-1 → P1-5)

**P1-1: Robustness Sweep** ✅
- Randomized stress test across 1000+ operating points
- Validates gradient correctness at diverse (L, u, x) configurations
- Runtime: ~120s (nightly-only)
- Caught zero regressions (dynamics are robust)

**P1-2: iLQR LQR Warm-Start** ✅
- Adds optional LQR initialization for iLQR
- Improves convergence reliability by 30–50% (fewer line search failures)
- API: `use_lqr_warmstart=True` parameter
- Minimal overhead (<5% added cost)

**P1-3: Terminal Hessian Improvement** ✅
- Adds FD-based exact terminal Hessian option for iLQR
- Addresses Gauss-Newton approximation inaccuracies
- API: `terminal_hessian_mode='fd_exact'`
- Trade-off: Slower but more robust (~2× terminal step cost)

**P1-4: Backward-Pass Performance** ✅
- Profiled dynamics backward pass (bottleneck: nested FD)
- Documented optimization path: analytic ∂A/∂u, ∂B/∂u (complex, deferred)
- Current performance: ~0.3ms per backward step (acceptable for T<50)

**P1-5: API Versioning** ✅
- Added explicit version metadata to Python API
- Module attributes: `__version__`, `__api_version__`
- All dict outputs include: `api_version`, `api_contract`
- Runtime compatibility check: `crm_diff_py.check_api_compat(required_version)`
- Semantic versioning (major.minor.patch)
- Current API version: **1.1.0**

**P1 Impact Summary**:
- ✅ Zero breaking changes to CP2/CP3 API
- ✅ All enhancements are opt-in (backward compatible)
- ✅ Improved reliability and debuggability
- ✅ Future-proofed for API evolution

---

## 🧪 Test Coverage

### Fast Gate Tests (< 5s, run on every PR)

| Test Name | Runtime | Coverage |
|-----------|---------|----------|
| `golden_backward` | 0.01s | C++ backward-pass golden reference |
| `cpp_python_smoke` | 0.30s | C++ ↔ Python binding consistency |
| `straight_rod_physics` | 0.02s | Physics sanity check (no bending) |
| `dynamics_smoke_cp21` | 0.02s | C++ dynamics smoke test |
| `dynamics_fd_cp22` | 0.34s | Finite difference validation (gradients) |
| `dynamics_smoke_cp23_python` | 0.48s | Python bindings smoke test |
| `api_versioning_p1_5` | 0.41s | API version metadata validation |

**Total Fast Gate**: ~1.6s
**Trigger**: Every PR to main/protected branches

### Extended Tests (run nightly or on-demand)

| Test Name | Runtime | Coverage |
|-----------|---------|----------|
| `dynamics_gradcheck_cp24_python` | ~5s | PyTorch gradcheck at 3 operating points |
| `dynamics_rollout_cp25_python` | ~3s | 20-step rollout backpropagation |
| `dynamics_robustness_sweep_p1_1` | ~120s | 1000+ randomized operating points |
| `dynamics_linearization_cp31_python` | ~2s | A_t, B_t matrix extraction validation |
| `ilqr_fixed_target_cp32_python` | ~15s | iLQR convergence to fixed target |
| `ilqr_descent_regression_cp32_python` | ~10s | iLQR cost reduction regression check |
| `mpc_tracking_cp33_python` | ~8s | MPC receding-horizon tracking |
| `terminal_hessian_improves_descent_p1_3` | ~20s | FD Hessian vs quadratic comparison |

**Total Nightly Suite**: ~180s
**Trigger**: Daily cron schedule + manual workflow dispatch

### CI Behavior Summary

**PR Checks** (`.github/workflows/pr_checks.yml`):
- Runs on: Pull requests to `main`, `cp2_6_ci_integration`, `stable/*` branches
- Tests: Fast gate subset (< 5s timeout)
- Policy: **Blocking** (PR cannot merge if tests fail)
- Purpose: Prevent regressions in core dynamics/bindings

**Nightly Checks** (`.github/workflows/nightly.yml`):
- Runs on: Daily cron (00:00 UTC) + manual trigger
- Tests: Full suite including long-running control tests
- Policy: **Non-blocking** (informational)
- Purpose: Catch edge cases, validate control stack robustness

---

## 📂 Repository Structure

```
CRM_DiffSim_Cl/
├── .github/workflows/
│   ├── pr_checks.yml          # Fast gate CI (< 5s tests)
│   └── nightly.yml            # Full nightly suite (all tests)
│
├── src/
│   ├── CRM_DiffEquilibrium.cpp  # Equilibrium forward/backward
│   ├── CRM_DiffDynamics.cpp     # Dynamics forward/backward
│   └── *.hpp                    # Headers
│
├── python/
│   ├── crm_bindings.cpp         # Pybind11 wrapper (equilibrium + dynamics)
│   ├── crm_equilibrium.py       # PyTorch autograd wrapper (equilibrium)
│   ├── crm_dynamics_torch.py    # PyTorch autograd wrapper (dynamics)
│   ├── control/
│   │   ├── ilqr.py             # iLQR solver
│   │   ├── lqr.py              # LQR utilities
│   │   └── mpc.py              # MPC framework
│   └── test_*.py                # Python test suite (CP2.x, CP3.x, P1.x)
│
├── docs/
│   ├── audits/
│   │   ├── PROJECT_AUDIT_CP2_CP3.md          # Comprehensive CP2/CP3 audit
│   │   ├── P1_1_ROBUSTNESS_SWEEP_COMPLETION.md
│   │   ├── P1_2_ILQR_LQR_WARMSTART_COMPLETION.md
│   │   ├── P1_3_TERMINAL_HESSIAN_COMPLETION.md
│   │   ├── P1_4_PROFILE_OPTIMIZE_COMPLETION.md
│   │   └── P1_5_API_VERSIONING_COMPLETION.md
│   ├── control/
│   │   └── CP3_MPC_DESIGN.md    # MPC framework design doc
│   └── RELEASE_NOTES_CP2_CP3_P1.md  # THIS FILE
│
├── CMakeLists.txt               # Build configuration + CTest registration
└── catheterdata/                # Parameter files for tests
```

---

## 🚀 Quick Start

### Build & Test

```bash
# Clone and build
git clone <repo-url>
cd CRM_DiffSim_Cl
git checkout stable/cp2-cp3-p1

mkdir -p build && cd build
cmake ..
make -j4

# Run fast gate tests (< 5s)
ctest -R "dynamics_smoke_cp21|dynamics_fd_cp22|dynamics_smoke_cp23_python|api_versioning_p1_5" --output-on-failure

# Run full test suite
ctest --output-on-failure
```

### Python Usage Example

```python
import sys
sys.path.insert(0, 'build')  # Add build/ to path
import crm_diff_py
import numpy as np

# Load catheter parameters
cath_params = crm_diff_py.load_cath_params("./catheterdata/CatheterParameterSet_1_dyn.txt")
cath_config = crm_diff_py.load_cath_config("./catheterdata/CatheterSpatialConfiguration_1.txt")

params_dict = {
    'CathParams': cath_params,
    'CathConfig': cath_config,
    'ContactMode': int(crm_diff_py.ContactModeType.FREE_TIP),
    'TipForce': [0.0, 0.0, 0.0],
    'deltau0_initialguess': [0.0, 0.0, 0.0],
    'IntegrationStepSize': 0.5,
    'FinalValueOnly': True,
}

# Run dynamics step
x_t = np.zeros(6, dtype=np.float64)
u_t = np.array([0.1, -0.05, 0.0], dtype=np.float64)
dt = 0.01
L_inserted = 100.0

result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)
print(f"Next state: {result['x_next']}")
print(f"Tip position: {result['p_tip']}")
print(f"API version: {result['api_version']}")  # "1.1.0"
```

### PyTorch Integration Example

```python
import torch
from crm_dynamics_torch import dynamics_step, load_default_catheter_params

# Load parameters
params_dict = load_default_catheter_params(
    "./catheterdata/CatheterParameterSet_1_dyn.txt",
    "./catheterdata/CatheterSpatialConfiguration_1.txt"
)

# Create differentiable trajectory
x_t = torch.zeros(6, dtype=torch.float64, requires_grad=True)
u_t = torch.tensor([0.1, -0.05, 0.0], dtype=torch.float64, requires_grad=True)
dt = 0.01
L_inserted = 100.0

# Forward pass (computes x_next with autograd support)
x_next = dynamics_step(x_t, u_t, dt, L_inserted, params_dict)

# Backward pass (compute gradients)
loss = (x_next**2).sum()
loss.backward()

print(f"∂loss/∂u = {u_t.grad}")  # Gradient w.r.t. control
```

### iLQR Trajectory Optimization Example

```python
from control.ilqr import iLQRSolver
from crm_dynamics_torch import dynamics_step, load_default_catheter_params
import torch
import numpy as np

# Setup
params_dict = load_default_catheter_params(
    "./catheterdata/CatheterParameterSet_1_dyn.txt",
    "./catheterdata/CatheterSpatialConfiguration_1.txt"
)

# Define cost matrices
T = 15  # Horizon
Q = torch.diag(torch.tensor([10, 10, 10, 1, 1, 1], dtype=torch.float64))
R = 0.1 * torch.eye(3, dtype=torch.float64)
Qf = 100 * torch.eye(6, dtype=torch.float64)

# Target state
target_state = torch.tensor([0.1, 0.05, 0.0, 0.0, 0.0, 0.0], dtype=torch.float64)

# Initial state
x0 = torch.zeros(6, dtype=torch.float64)
u_init = torch.zeros((T, 3), dtype=torch.float64)

# Create solver with P1 enhancements
solver = iLQRSolver(
    dynamics_fn=dynamics_step,
    horizon=T,
    Q=Q, R=R, Qf=Qf,
    dt=0.01,
    L_inserted=100.0,
    params_dict=params_dict,
    max_iters=50,
    use_lqr_warmstart=True,         # P1-2: LQR initialization
    terminal_hessian_mode='fd_exact' # P1-3: Exact terminal Hessian
)

# Solve
result = solver.solve(x0, u_init, target_state)

print(f"Converged: {result['converged']}")
print(f"Final cost: {result['final_cost']:.6f}")
print(f"Iterations: {result['num_iters']}")
print(f"Optimal control sequence shape: {result['u_opt'].shape}")
```

---

## ⚙️ API Version & Compatibility

**Current API Version**: `1.1.0` (as of P1-5)

### Semantic Versioning Rules

- **Major version** (X.0.0): Breaking changes (removed/renamed keys, incompatible behavior)
- **Minor version** (1.X.0): Backward-compatible additions (new keys, new optional parameters)
- **Patch version** (1.1.X): Bug fixes with no API changes

### Version Checking

```python
import crm_diff_py

# Check module version
print(crm_diff_py.__version__)      # "1.0.0" (package version)
print(crm_diff_py.__api_version__)  # "1.1.0" (API contract version)

# Runtime compatibility check
try:
    crm_diff_py.check_api_compat("1.1.0")  # Returns True if compatible
    print("API compatible!")
except RuntimeError as e:
    print(f"API incompatible: {e}")
```

### Version Metadata in Outputs

All `dynamics_forward`, `dynamics_backward`, `equilibrium_forward`, `equilibrium_backward` return dicts now include:

```python
result = crm_diff_py.dynamics_forward(...)
print(result['api_version'])   # "1.1.0"
print(result['api_contract'])  # "dynamics_v1_1"
```

This enables downstream code to verify compatibility:

```python
def safe_dynamics_call(x_t, u_t, dt, L_inserted, params_dict):
    result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)

    # Verify API version
    if result['api_version'] != '1.1.0':
        raise RuntimeError(f"Unexpected API version: {result['api_version']}")

    return result['x_next']
```

---

## 🎓 Known Limitations & Recommended Tracks

### Performance Bottleneck (Documented, Acceptable)

**Issue**: Nested finite differences in matrix-dependence (∂A/∂u, ∂B/∂u) add ~3× overhead to backward pass.

**Root Cause**:
- Dynamics Jacobians A_t, B_t depend on M(u), D(u), K(u) matrices
- Computing ∂A/∂u analytically requires differentiating through equilibrium Newton solver (complex)
- Current approach: FD perturbations around u_t (correct but slow)

**Impact**:
- Single backward step: ~0.3ms (vs ~0.1ms forward)
- 20-step rollout backprop: ~10s (PyTorch overhead + 3× equilibrium calls)
- Acceptable for T < 50, problematic for T > 100

**Mitigation Paths**:
1. **Short-term** (DONE): Document trade-off, profile to confirm bottleneck (P1-4)
2. **Medium-term** (RECOMMENDED): Implement analytic ∂M/∂u, ∂D/∂u, ∂K/∂u via implicit differentiation
3. **Long-term** (RESEARCH): GPU acceleration with cuBLAS/cuSOLVER (requires architecture redesign)

**Status**: P1-4 profiling confirms bottleneck. Analytic gradients deferred (engineering effort vs. impact trade-off).

---

### iLQR Line Search Fragility (Fundamental Limitation)

**Issue**: iLQR may fail to converge with high terminal weights (Qf >> Q) due to passive dynamics attraction.

**Root Cause**:
- Catheter has passive equilibrium attractor: small controls → state decays to rest
- Gauss-Newton Hessian approximation (Q_uu ≈ R + B^T V_xx B) omits passive restoring forces
- Line search can overshoot, causing cost increase instead of descent
- Most pronounced when target is far from equilibrium and terminal weight is high

**Manifestation**:
- Symptom: "Line search failed" warnings, non-convergence
- Reproducible: Qf = 1000× Q, target far from origin, T < 10
- Does NOT occur: Qf ≈ Q, longer horizons (T > 15), MPC with T=5–10

**Mitigation** (P1-2, P1-3):
1. ✅ **LQR warm-start** (P1-2): Initialize with LQR solution → 30–50% fewer failures
2. ✅ **Exact terminal Hessian** (P1-3): Use FD-based Hessian at final step → more robust but 2× slower
3. ⚠️ **Tune weights**: Start with Qf ≈ 10× Q, gradually increase
4. ⚠️ **Use MPC**: Receding horizon with T=5–10 avoids long-range prediction errors

**Long-term Fix** (Out of Scope):
- Trust-region DDP (Differential Dynamic Programming) with constrained updates
- Second-order line search (considers curvature, not just gradient)
- Requires significant algorithmic redesign

**Status**: Documented workarounds sufficient for typical use cases. Not a bug, but intrinsic to local quadratic approximations.

---

### Recommended Next Tracks

Based on the current state, here are three recommended development tracks:

#### Track 1: Control Robustness & Reliability (Medium Priority)
**Goal**: Make iLQR/MPC production-ready for real-world deployment

**Tasks**:
1. Implement trust-region DDP variant (replaces line search)
2. Add automatic weight tuning heuristics (adaptive Qf scaling)
3. Expand MPC tracking tests to diverse trajectories (circles, figure-8, etc.)
4. Add disturbance rejection tests (unexpected forces, parameter mismatch)

**Estimated Effort**: 3–4 weeks
**Impact**: High (enables robust closed-loop control)

---

#### Track 2: Learning & System Identification (High Value)
**Goal**: Enable data-driven parameter estimation and model learning

**Tasks**:
1. Add parameter gradients ∂/∂θ (mass, stiffness, damping, geometry)
2. Implement gradient-based system ID pipeline (fit model to trajectory data)
3. Create learned dynamics wrapper (e.g., neural network residual correction)
4. Validate on real catheter data (if available)

**Estimated Effort**: 4–6 weeks
**Impact**: Very High (unlocks adaptive control, digital twin applications)

---

#### Track 3: Performance & Scalability (Lower Priority, High ROI if needed)
**Goal**: 10× speedup to enable long-horizon optimization (T > 100)

**Tasks**:
1. Implement analytic ∂A/∂u, ∂B/∂u (implicit differentiation through equilibrium)
2. Profile and optimize equilibrium Newton solver (current bottleneck)
3. GPU acceleration (cuBLAS/cuSOLVER port, batched operations)
4. Investigate sparse Jacobian exploitation (catheter has local structure)

**Estimated Effort**: 6–8 weeks
**Impact**: Moderate (unless long-horizon planning is critical)

**Recommendation**: Defer until use case demands T > 50

---

#### Track 4: Packaging & Distribution (Deployment Focus)
**Goal**: Make library easy to install and use for external collaborators

**Tasks**:
1. Create proper Python package structure (`setup.py`, `pyproject.toml`)
2. Add conda/pip installation instructions
3. Pre-build binary wheels for common platforms (Linux, macOS, Windows)
4. Create Docker container with pre-built environment
5. Add user-facing tutorials and examples (Jupyter notebooks)
6. Document parameter file format and calibration workflow

**Estimated Effort**: 2–3 weeks
**Impact**: High (lowers barrier to adoption, improves reproducibility)

**Recommendation**: Prioritize if library will be shared externally

---

## 📋 Verification Commands

### Reproduce Test Results

```bash
# Fast gate subset (< 2s, recommended for quick validation)
cd /workspaces/CRM_DiffSim_Cl/build
ctest -R "dynamics_smoke_cp21|dynamics_fd_cp22|dynamics_smoke_cp23_python|api_versioning_p1_5" --output-on-failure

# Expected output:
#   Test #5:  dynamics_smoke_cp21 .............. Passed  0.02 sec
#   Test #6:  dynamics_fd_cp22 ................. Passed  0.34 sec
#   Test #7:  dynamics_smoke_cp23_python ....... Passed  0.48 sec
#   Test #11: api_versioning_p1_5 .............. Passed  0.41 sec
#   100% tests passed, 0 tests failed out of 4

# Full test suite (nightly)
ctest --output-on-failure

# Expected: 14/14 tests passing (some may be skipped if pybind11 not found)
```

### Verify API Version

```bash
cd /workspaces/CRM_DiffSim_Cl/build
python3 -c "import sys; sys.path.insert(0, '.'); import crm_diff_py; print(f'Version: {crm_diff_py.__version__}, API: {crm_diff_py.__api_version__}')"

# Expected output:
#   Version: 1.0.0, API: 1.1.0
```

### Check CI Status (if GitHub Actions enabled)

```bash
# View recent workflow runs
gh run list --limit 5

# View specific run details
gh run view <run-id>
```

---

## 📄 Documentation Index

### Core Technical Docs
- **[PROJECT_AUDIT_CP2_CP3.md](docs/audits/PROJECT_AUDIT_CP2_CP3.md)**: Comprehensive audit of CP2/CP3 implementation
- **[CP3_MPC_DESIGN.md](docs/control/CP3_MPC_DESIGN.md)**: MPC framework design document

### P1 Completion Audits
- **[P1_1_ROBUSTNESS_SWEEP_COMPLETION.md](docs/audits/P1_1_ROBUSTNESS_SWEEP_COMPLETION.md)**: Randomized stress testing
- **[P1_2_ILQR_LQR_WARMSTART_COMPLETION.md](docs/audits/P1_2_ILQR_LQR_WARMSTART_COMPLETION.md)**: LQR initialization for iLQR
- **[P1_3_TERMINAL_HESSIAN_COMPLETION.md](docs/audits/P1_3_TERMINAL_HESSIAN_COMPLETION.md)**: FD-based exact Hessian option
- **[P1_4_PROFILE_OPTIMIZE_COMPLETION.md](docs/audits/P1_4_PROFILE_OPTIMIZE_COMPLETION.md)**: Performance profiling
- **[P1_5_API_VERSIONING_COMPLETION.md](docs/audits/P1_5_API_VERSIONING_COMPLETION.md)**: API versioning implementation

### Source Code
- **C++ Core**: `src/CRM_DiffEquilibrium.cpp`, `src/CRM_DiffDynamics.cpp`
- **Python Bindings**: `python/crm_bindings.cpp`
- **PyTorch Wrappers**: `python/crm_dynamics_torch.py`, `python/crm_equilibrium.py`
- **Control**: `python/control/ilqr.py`, `python/control/mpc.py`, `python/control/lqr.py`

---

## 🔒 Stability Guarantee

This release branch (`stable/cp2-cp3-p1`) is **locked** and will NOT be rebased or force-pushed. All commits are immutable and reproducible.

**Git Hash**: (Will be filled after final commit)
**Base Commit**: 26201c2 ("last edits before moving next stage")
**Final Commit**: (Will be filled after adding this release notes file)

**Reproducibility**:
```bash
git checkout stable/cp2-cp3-p1
git log --oneline --graph --all  # Verify commit history
git describe --always --dirty     # Check for uncommitted changes
```

---

## 📞 Contact & Support

**Maintainer**: Eser Erdem Tuna <eet12@case.edu>
**Repository**: (Add GitHub URL here)
**Branch**: `stable/cp2-cp3-p1`
**Issues**: (Add issue tracker URL)

For questions about:
- **API usage**: See PyTorch examples above or `python/test_*.py` files
- **Control tuning**: Read `docs/audits/PROJECT_AUDIT_CP2_CP3.md` Section 6
- **Performance**: See P1-4 completion doc
- **CI setup**: Examine `.github/workflows/*.yml`

---

## 🎉 Conclusion

This release provides a **mathematically validated, CI-protected, and well-documented** foundation for catheter control research. The CP2/CP3 APIs are stable, the P1 hardening work has improved robustness, and the codebase is ready for:

1. ✅ External collaboration (stable API with versioning)
2. ✅ Integration testing (reproducible build + tests)
3. ✅ Future development (branch from here for new features)
4. ✅ Research baselines (known performance characteristics)

**Key Strengths**:
- Gradient correctness validated (< 1e-4 error)
- PyTorch integration tested with `gradcheck`
- CI gates prevent regressions
- Known limitations documented with mitigation paths

**Not Included** (intentionally deferred):
- GPU acceleration
- Parameter gradients ∂/∂θ
- Contact/collision dynamics
- Analytic matrix derivatives (∂A/∂u, ∂B/∂u)

**Status**: ✅ Production-ready for research use. Recommended next track: Control Robustness (Track 1) or Learning (Track 2) depending on application needs.

---

**End of Release Notes**
