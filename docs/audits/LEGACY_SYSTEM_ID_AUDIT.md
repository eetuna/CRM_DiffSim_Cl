# Legacy System Identification Audit

**Date:** 2026-01-08
**Auditor:** Claude Code
**Scope:** `legacy_worktree/` (original main branch before FULLSTATE infrastructure)

---

## A) High-Level Summary

The legacy CRM catheter system employed **nonlinear grey-box system identification** using MATLAB's System Identification Toolbox (`nlgreyest`) combined with **ARX linear identification** for preliminary parameter exploration.

### Problem Class
- **Primary Method:** Nonlinear grey-box estimation with finite-difference gradients
- **Secondary Method:** ARX (AutoRegressive with eXogenous inputs) for linear dynamics analysis
- **Objective:** Fit physical catheter model parameters (stiffness, damping, coil alignment, mass) to experimental trajectory data

### Key Characteristics
1. **Physics-based model:** Full Cosserat rod dynamics integrated into MATLAB MEX function
2. **Black-box gradient computation:** Backward finite differences (`opt.GradientOptions.DifferencingScheme = 'Backward approximation'`)
3. **Experimental data-driven:** Benchtop camera tracking of catheter coil positions under magnetic actuation
4. **Iterative batch optimization:** Levenberg-Marquardt via `lsqnonlin` with max 20 iterations

---

## B) Detailed File-by-File Analysis

### System Identification Core Files

| File | Parameters Estimated | Method | Data Source | Notes |
|------|---------------------|--------|-------------|-------|
| **`matlab/greybox_modeling.m`** | damping (4 params), E (2 params), coil alignment (2 params), coil turn-area (3 params), mass (1 param), segment lengths (3 params) | `nlgreyest` with `lsqnonlin`, backward FD gradients, 20 max iterations | Experimental benchtop data: `3D_dynamic_response_data_0124/output_trajectories/*.txt` | Main system ID script. Outputs saved to `model_1.mat`, `model_2.mat`, `model_3.mat` |
| **`matlab/linear_identification.m`** | ARX model coefficients for torque vs. angles/angular velocities | `arx(z, [na, nb, nk])` | Derived from experimental trajectories | Exploratory linear dynamics - identifies coupling between magnetic torque and catheter angles |
| **`Mexfiles/CRMDYN_c.cpp`** | N/A (simulation forward model) | MEX wrapper for Cosserat dynamics BVP/IVP solver | Called by `nlgreyest` during optimization | Implements `compute_dx` (state derivatives) and `compute_y` (outputs = coil positions) |
| **`matlab/jacobian_forward_difference.m`** | N/A (utility) | Forward FD: `Df(:,i) = (f(x+h*e_i) - f(x)) / h` | N/A | Generic Jacobian approximation (not directly called in main pipeline) |
| **`matlab/load_parameters.m`** | Geometric/physical parameter setup | Analytical calculation of MuMatrix (magnetic moment matrix from coil geometry) | Catheter design specs | Computes coil turn-area matrix, weights, inertia from alignment angles |
| **`matlab/read_input_output.m`** | N/A (data preprocessing) | Interpolates camera data (16.7 ms) to match current inputs (1 ms) | Experimental data files | Synchronizes input currents with output trajectories |
| **`matlab/data_stitch.m`** | N/A (data merging) | Concatenates multiple experimental runs at different frequencies | Multiple trajectory files | Prepares training dataset for `nlgreyest` |

### Parameter Files

| File | Purpose |
|------|---------|
| **`catheterdata/CatheterParameterSet_1_new.txt`** | Initial guess parameters for system ID (E=5.3948, G=2.3881, mass=5.77e-5 kg) |
| **`catheterdata/ParameterEstResult_ran_dynamics.txt`** | Estimated parameters from previous run (E=99.9998, G=18.3759, mass=6.64e-5 kg, alignment angles, etc.) |
| **`parameters/model_1.mat`, `model_2.mat`, `model_3.mat`** | Saved `nlgreyest` output objects containing optimized parameters |

---

## C) Detailed System ID Methodology

### 1. Parameters Estimated

**From `greybox_modeling.m` (lines 213-224):**
```matlab
Parameters = {damping; Ts; radius_; E_; Coil_align; Coil_turnarea; mass_; SegmentLengths};
nlgr.Parameters(1).Fixed = false;  % damping [4 params]
nlgr.Parameters(4).Fixed = false;  % E_ (Young's modulus) [2 params]
nlgr.Parameters(5).Fixed = false;  % Coil_align [2 params]
nlgr.Parameters(6).Fixed = false;  % Coil_turnarea [3 params]
nlgr.Parameters(7).Fixed = false;  % mass_ [1 param]
nlgr.Parameters(8).Fixed = false;  % SegmentLengths [3 params]
```

**Parameter Breakdown:**
- **Damping** (4 values): Linear/angular velocity damping for coil sections
- **Young's Modulus E** (2 values): Bending stiffness for flexible segments
- **Shear Modulus G**: Fixed (coupled to E via Poisson ratio)
- **Coil Alignment** (2 angles): Magnetic moment orientation corrections
- **Coil Turn-Area** (3 values): Effective magnetic dipole strength per coil
- **Actuator Mass** (1 value): Coil assembly inertia
- **Segment Lengths** (3 values): Geometric discretization of catheter

**Total:** 15 free parameters

### 2. Method

**Optimization Framework:**
- **Algorithm:** `lsqnonlin` (trust-region-reflective or Levenberg-Marquardt)
- **Gradient Computation:** Backward finite differences (no analytic Jacobian)
- **Tolerance:** `FunctionTolerance = 0.00001`
- **Max Iterations:** 20

**Model Structure:**
- **State-space order:** `[Ny=3, Nu=3, Nx=39]`
  - 3 outputs (coil x,y,z position)
  - 3 inputs (currents to 3 coils)
  - 39 states (v, w, m, n, p, R, xf) representing velocities, moments, position, rotation, and full catheter state
- **Time step:** 1 ms (Ts = 0.001 s)

**Forward Model (`CRMDYN_c.cpp`):**
1. Solves **boundary value problem (BVP)** for internal forces/moments given boundary conditions
2. Integrates **initial value problem (IVP)** along catheter length to compute Cosserat deformation
3. Outputs coil position (p) as observable

**Workflow (from `greybox_modeling.m`):**
1. Load experimental data (currents + camera-tracked coil positions)
2. Simulate model with initial parameter guess → get predicted trajectory
3. Compute alignment offset: `dy_init = coil_position_mat(:,1) - p_mat(:,1)`
4. Subtract offset from experimental data
5. Call `nlgreyest(z, nlgr, opt)` → MATLAB iteratively:
   - Perturbs parameters
   - Calls `CRMDYN_c.cpp` via MEX to simulate
   - Computes residual (predicted - measured positions)
   - Estimates Jacobian via finite differences
   - Updates parameters
6. Save optimized model to `.mat` file

### 3. Data Used

**Source:** Benchtop experimental setup with camera-based tracking
- **Directory:** `3D_dynamic_response_data_0124/`
- **Inputs:** Current commands to 3 electromagnetic coils (`.mat` files)
- **Outputs:** 3D coil positions from camera tracking (`.txt` files with format: `< base xyz > < coil xyz > < tip xyz >`)

**Example Trajectories:**
- `circle01_01.txt`, `circle03_01.txt`, `circle05_01.txt`, `circle08_01.txt`, `circle10_01.txt`, `circle100_01.txt` (varying frequencies)
- `lemniscate01_01.txt` (figure-8 path)

**Data Preprocessing:**
- Camera framerate: 60 Hz (16.7 ms)
- Current timestep: 1 ms
- `read_input_output.m` linearly interpolates camera data to match current timestep
- `data_stitch.m` concatenates multiple frequencies into single training set

**Data Characteristics:**
- Slow quasi-static motions (circle frequencies 0.01 Hz to 100 Hz based on filenames)
- Coil position measured relative to base marker
- No external force sensors (free-tip mode assumed)

### 4. Model Assumed

**Full 3D Cosserat Rod Dynamics** (from `CRMDYN_c.cpp`):
- **Kinematics:** Spatial position `p(λ)`, rotation matrix `R(λ)`, strain `u(λ)`
- **Dynamics:** Momentum balance with inertia, damping, gravity, magnetic torques
- **Discretization:** 3 segments (base, coil section, tip)
- **Boundary Conditions:**
  - Base: clamped (fixed orientation)
  - Tip: free (ContactMode::FREE_TIP) or fixed point constraint
- **Magnetic Actuation:** `τ_mag = MuMatrix * I × B` where `I` = currents, `B` = external field
- **Numerical Integration:** Stepsize 0.2 mm along catheter length

**NOT a reduced model** - full Cosserat formulation with:
- Bending/shear/torsional compliance
- Distributed mass/inertia
- Velocity-dependent damping

### 5. Differentiation Used

**Finite Differences (Black-Box Gradients)**
- **Where:** MATLAB's `nlgreyest` calls MEX function repeatedly
- **Scheme:** Backward approximation: `∂f/∂p ≈ (f(p) - f(p - h)) / h`
- **No analytic Jacobians** - simulation code (`CRMDYN_c.cpp`) does NOT compute `∂output/∂parameters`

**Why finite differences?**
- Cosserat BVP solver involves iterative shooting method (nested nonlinear solves)
- Analytic differentiation would require adjoint method through nested solvers
- Legacy code predates modern AD tools

**`jacobian_forward_difference.m` presence:**
- Generic utility function (not actively called in main pipeline)
- May have been used for manual sensitivity studies

### 6. Optimization Loop Structure

**Outer Loop:** MATLAB `nlgreyest` (Levenberg-Marquardt)
- Perturbs parameter vector `p`
- Calls MEX function for each perturbation

**Inner Loop:** C++ `CRMDYN_c.cpp` MEX function
- Receives: state `x`, input `u`, parameters `p`, time `t`
- Solves BVP (shooting method) for boundary forces
- Integrates IVP forward to get full catheter shape
- Returns: state derivatives `dx` and outputs `y` (coil position)

**Simulation Time-Stepping (inside MATLAB):**
- MATLAB ODE solver (default: `ode45` unless overridden) steps through time
- Each timestep calls `CRMDYN_c.cpp` to compute `dx`

**Parameter Update:**
```
repeat until convergence:
  1. MATLAB picks new parameter guess
  2. Simulate full trajectory (calls MEX ~1000 times for 1-second dataset)
  3. Compute cost = ||y_predicted - y_measured||^2
  4. Estimate gradient via FD (repeat steps 1-3 with perturbed params)
  5. Update parameters via Levenberg-Marquardt
```

### 7. Outputs Produced

**Saved Parameter Files:**
- `parameters/model_1.mat`, `model_2.mat`, `model_3.mat` - MATLAB `idnlgrey` objects with optimized parameters
- `catheterdata/ParameterEstResult_ran_dynamics.txt` - parameter values exported to text for reuse

**Plots (generated but not saved programmatically):**
- Figure 1: Comparison of simulated vs. measured coil x/y/z positions over time
- Figure 2: 3D trajectory overlay (predicted vs. measured)
- Figure 3: Aligned trajectories (after offset correction)
- Figure 7: `compare(z, nlgr_model)` - MATLAB system ID toolbox validation plot

**Convergence Metrics:**
- Printed to console during `opt.Display = 'on'`
- No explicit logging of parameter evolution or cost function history

**Validation Tests:**
- `compare(z, nlgr_model, compareOptions('InitialCondition', 'e'))` - simulate model on training data
- No dedicated test set or cross-validation

---

## D) Key Takeaways for Modern Pipeline

### What Can Be Reused Conceptually

1. **Parameter selection is well-justified:**
   - Damping, E/G, coil alignment, mass are physically meaningful and observable
   - Segment lengths account for geometric uncertainty

2. **Experimental protocol is sound:**
   - Benchtop camera tracking provides ground truth
   - Multiple trajectory types (circle, lemniscate) at varying speeds ensure diverse excitation
   - Data interpolation ensures temporal alignment

3. **Grey-box approach leverages physics:**
   - Constraining search to physical model prevents overfitting
   - Interpretable parameters unlike pure black-box neural nets

4. **Validation via simulation-experiment comparison:**
   - Visual inspection of trajectories is critical for catching systematic errors

### What Is Obsolete

1. **Finite-difference gradients:**
   - Modern differentiable simulator can provide exact gradients via autodiff
   - FD requires ~15 forward passes per iteration (one per parameter)
   - Prone to numerical noise and step-size tuning issues

2. **MATLAB MEX dependency:**
   - Hard to maintain, debug, and extend
   - Python/JAX ecosystem offers better tooling

3. **20-iteration limit:**
   - Arbitrary cutoff likely due to computational cost of FD
   - Gradient-based optimization with exact derivatives can converge faster

4. **No uncertainty quantification:**
   - Point estimates only (no confidence intervals, Hessian approximation, or bootstrapping)
   - Modern tools (e.g., Laplace approximation, MCMC) could quantify parameter uncertainty

5. **Manual data preprocessing:**
   - `data_stitch.m` hardcodes file paths and indices
   - Modern pipelines use automated data loaders with configuration files

### What Assumptions No Longer Hold

1. **Quasi-static assumption:**
   - File naming (e.g., `circle100_01.txt`) suggests frequencies up to 100 Hz tested
   - Legacy system ID may have been limited to slow motions where inertia is negligible
   - **Modern in-vivo use cases require accurate dynamics at higher speeds**

2. **Free-tip boundary condition:**
   - `ContactMode::FREE_TIP` hardcoded in `CRMDYN_c.cpp`
   - Real surgical scenarios involve contact forces
   - **Modern pipeline must handle contact/constraints**

3. **Fixed discretization:**
   - 3 segments hardcoded
   - Adaptive mesh refinement may improve accuracy/speed tradeoffs

4. **Single-catheter focus:**
   - Parameters estimated for one prototype ("11.P35.1.a")
   - **Modern pipeline should enable parameter databases for multiple designs**

5. **Offline batch optimization:**
   - All data collected upfront, then optimized
   - **Real-time adaptive identification during procedures not supported**

### Recommendations for Differentiable Pipeline

1. **Replace FD with autodiff:**
   - Use JAX/PyTorch to backpropagate through Cosserat solver
   - Expect 10-100× speedup in gradient computation

2. **Add regularization:**
   - Legacy code has no explicit regularization (relies on physical model structure)
   - Consider L2 penalty on parameter deviations from nominal values

3. **Implement cross-validation:**
   - Split trajectory data into train/validation/test sets
   - Report generalization error, not just training fit

4. **Quantify uncertainty:**
   - Compute Hessian at optimum → parameter covariance
   - Use ensembles or Bayesian methods for robustness

5. **Automate data pipeline:**
   - Replace manual MATLAB scripts with configurable Python data loaders
   - Support streaming/online identification

6. **Extend to contact scenarios:**
   - Generalize beyond free-tip assumption
   - Estimate contact stiffness/friction if needed

7. **Version control parameters:**
   - Track parameter evolution across experiments
   - Store metadata (catheter serial number, date, operator) with `.mat` files

---

## E) Execution Flow Summary

### Top-Level Entry Point
**`matlab/greybox_modeling.m`** (primary system ID script)

### Pipeline Steps
```
1. greybox_modeling.m starts
   ├─> read_input_output.m
   │    └─> Loads experimental data from 3D_dynamic_response_data_0124/
   ├─> load_parameters.m (optional, for manual setup)
   ├─> CRMDYN_c_mex.cpp (initial forward simulation to get alignment offset)
   │    └─> Calls CRMDYN.hpp (C++ Cosserat solver)
   ├─> nlgreyest(z, nlgr, opt)
   │    └─> Iteratively calls CRMDYN_c.cpp (MEX wrapper)
   │         └─> compute_dx() and compute_y()
   │              └─> DynamicsBVP() + DYNSolverIVP()
   └─> compare(z, nlgr_model) for validation plots
```

### Alternative Entry Point (Exploratory)
**`matlab/linear_identification.m`**
- Runs `arx()` on experimental data to fit linear ARX models
- Separate from main grey-box pipeline
- Likely used for initial parameter exploration or model reduction studies

---

## F) Data Inventory

### Experimental Data Files
- **Inputs:** `3D_dynamic_response_data_0124/input_currents/*.mat`
  - `circleCurrents.mat`
  - `lemniscateCurrents.mat`

- **Outputs:** `3D_dynamic_response_data_0124/output_trajectories/*.txt`
  - Format: `timestamp < base_xyz > < coil_xyz > < tip_xyz > < normal_xyz >`
  - Multiple frequencies: circle01, circle03, circle05, circle08, circle10, circle100
  - Multiple prototypes/runs: `_01`, `_02` suffixes

### Parameter Files
- **Initial Guesses:**
  - `catheterdata/CatheterParameterSet_1_new.txt`
  - `catheterdata/CatheterParameterSet_1.txt`
  - `catheterdata/CatheterParameterSet_2.txt`

- **Optimized Results:**
  - `catheterdata/ParameterEstResult_ran_dynamics.txt`
  - `parameters/model_1.mat`, `model_2.mat`, `model_3.mat`

---

## G) Conclusions

The legacy system ID approach was **state-of-the-art for its time** (circa 2015-2020 based on MATLAB idioms), combining:
- Physics-based Cosserat rod model
- Experimental benchtop validation
- Nonlinear optimization with MATLAB's toolbox

However, it suffers from:
- **Computational inefficiency** (finite differences)
- **Lack of uncertainty quantification**
- **Limited scalability** (manual data handling, 20-iteration limit)
- **No contact modeling**

**The modern differentiable pipeline should:**
1. Preserve the physics-based model and parameter choices
2. Replace FD gradients with autodiff
3. Add uncertainty quantification and cross-validation
4. Generalize to contact scenarios
5. Automate data handling for multi-catheter parameter databases

This will enable **real-time parameter adaptation**, **robust control design**, and **transfer to in-vivo conditions**.

---

**END OF AUDIT**
