# Next Step Recommendation - Single Highest-Priority Action

**Audit Date**: 2026-01-04
**Branch**: `true-legacy-dynamics-migration`
**Commit**: `aeb2c7ebd85389fed5c4c70d3c56ac4930109a2b`
**Context**: Post-FULLSTATE migration completion audit

---

## Executive Summary

**Recommended Next Action**: **Add VJP gradient check test for state variables (∂x_coil, ∂xf)**

**Scope**: Single PR, 150-250 lines of code
**Risk**: LOW (pure test addition, no production code changes)
**Impact**: HIGH (validates critical backward pass correctness for state gradients)

---

## Current State Assessment

### What Works ✅

1. **FULLSTATE contract enforced** (18*N+15 state)
2. **Entrypoints operational** (true_legacy_step_forward, vjp, linearize)
3. **Differentiation implemented** (analytic Jacobians, IFT/adjoint)
4. **Reduced6D isolated** (zero default imports)
5. **Basic test coverage exists**:
   - Forward step smoke test ✓
   - Linearization shapes test ✓
   - VJP gradcheck for **control** (∂u) ✓
   - Controller smoke tests ✓
   - Sanity gate (no reduced6d) ✓

### Critical Gap ⚠️

**Missing VJP gradcheck for state variables**:
- ✅ `test_fullstate_vjp_gradcheck_u.py` validates ∂L/∂u
- ❌ **NO test validates ∂L/∂x_coil or ∂L/∂xf**

**Risk if not addressed**:
- State gradients could be incorrect (wrong signs, missing terms, dimension errors)
- Controllers using state gradients (LQR, iLQR value function) would be unreliable
- Silent failures in trajectory optimization

**Current test coverage**: `tests/test_fullstate_vjp_gradcheck_u.py:1-63`
- Only checks control gradients (line 42-54)
- Does NOT check state gradients

---

## Recommended Action: Add State VJP Gradient Check Test

### Justification

**Why this over other gaps?**

| Alternative | Priority | Rationale for Deferring |
|-------------|----------|-------------------------|
| Linearization numerical accuracy | MEDIUM | Linearization uses same Jacobians as VJP; if VJP state gradients pass, linearization likely correct |
| Batched VJP equivalence | MEDIUM | Batched code path uses same Jacobian computation; VJP correctness is prerequisite |
| Convergence failure handling | LOW | Edge case handling; correctness of nominal path is higher priority |

**State VJP is foundational**:
1. **Prerequisite for controllers**: LQR/iLQR need accurate ∂x_{t+1}/∂x_t
2. **Most complex gradient path**: State gradients involve IVP Jacobians (∂p, ∂R) + BVP implicit terms
3. **Highest risk of silent failure**: Wrong state gradients → bad trajectories without obvious errors

### Evidence of Current Gap

**File**: `tests/test_fullstate_vjp_gradcheck_u.py`
**Lines 42-54**: Only control gradient checked
```python
def test_vjp_gradcheck_u():
    # ... setup ...

    # Compute VJP gradients
    result_vjp = crm_diff_py.true_legacy_step_vjp(...)
    grad_u_vjp = result_vjp['grad_u']  # ← Only checking this

    # Finite difference reference
    grad_u_fd = compute_finite_diff_grad_u(...)

    # Compare
    np.testing.assert_allclose(grad_u_vjp, grad_u_fd, rtol=1e-4, atol=1e-6)
```

**Missing**:
```python
# Should also check:
grad_x_coil_vjp = result_vjp['grad_x_coil']  # ← NOT TESTED
grad_xf_vjp = result_vjp['grad_xf']          # ← NOT TESTED

# Compare against finite differences
assert_allclose(grad_x_coil_vjp, grad_x_coil_fd, ...)
assert_allclose(grad_xf_vjp, grad_xf_fd, ...)
```

---

## Implementation Plan (Single PR)

### File to Create

**Path**: `tests/test_fullstate_vjp_gradcheck_state.py`
**Size**: ~150-250 lines

### Test Structure

```python
"""
Test FULLSTATE VJP gradient correctness for state variables.

Verifies that ∂L/∂x_coil and ∂L/∂xf computed via VJP match finite difference.
"""

import sys
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/build')
sys.path.insert(0, '/workspaces/CRM_DiffSim_Cl/python')

import numpy as np
import crm_diff_py
from control.true_legacy_state_adapter import pack_true_legacy_state

def create_test_state(n_act=1):
    """Create a simple test state."""
    # ... similar to test_fullstate_vjp_gradcheck_u.py ...

def compute_finite_diff_grad_x_coil(x_coil, xf, u, dt, params, grad_tip_p, eps=1e-7):
    """Compute ∂L/∂x_coil via finite differences."""
    grad_x_coil = np.zeros_like(x_coil)

    for j in range(x_coil.shape[0]):
        for i in range(x_coil.shape[1]):
            # Perturb x_coil[j, i]
            x_coil_plus = x_coil.copy()
            x_coil_plus[j, i] += eps

            # Forward pass
            result_plus = crm_diff_py.true_legacy_step_forward(
                x_coil_plus, xf, u, dt, params
            )
            tip_p_plus = result_plus['tip_p']

            # Central difference
            x_coil_minus = x_coil.copy()
            x_coil_minus[j, i] -= eps
            result_minus = crm_diff_py.true_legacy_step_forward(
                x_coil_minus, xf, u, dt, params
            )
            tip_p_minus = result_minus['tip_p']

            # Gradient via chain rule: ∂L/∂x_coil[j,i] = grad_tip_p · ∂tip_p/∂x_coil[j,i]
            dtip_p_dx = (tip_p_plus - tip_p_minus) / (2 * eps)
            grad_x_coil[j, i] = np.dot(grad_tip_p, dtip_p_dx)

    return grad_x_coil

def compute_finite_diff_grad_xf(x_coil, xf, u, dt, params, grad_tip_p, eps=1e-7):
    """Compute ∂L/∂xf via finite differences."""
    grad_xf = np.zeros_like(xf)

    for i in range(len(xf)):
        # Perturb xf[i]
        xf_plus = xf.copy()
        xf_plus[i] += eps

        result_plus = crm_diff_py.true_legacy_step_forward(
            x_coil, xf_plus, u, dt, params
        )
        tip_p_plus = result_plus['tip_p']

        xf_minus = xf.copy()
        xf_minus[i] -= eps
        result_minus = crm_diff_py.true_legacy_step_forward(
            x_coil, xf_minus, u, dt, params
        )
        tip_p_minus = result_minus['tip_p']

        dtip_p_dxf = (tip_p_plus - tip_p_minus) / (2 * eps)
        grad_xf[i] = np.dot(grad_tip_p, dtip_p_dxf)

    return grad_xf

def test_vjp_gradcheck_x_coil():
    """Test VJP gradient correctness for x_coil (coil state)."""
    n_act = 1
    x_coil, xf, u, dt, params = create_test_state(n_act)

    # Upstream gradient (random)
    grad_tip_p = np.random.randn(3)

    # VJP gradients
    result_vjp = crm_diff_py.true_legacy_step_vjp(
        x_coil, xf, u, dt, params['L_inserted'], params, grad_tip_p
    )
    grad_x_coil_vjp = result_vjp['grad_x_coil']

    # Finite difference reference
    grad_x_coil_fd = compute_finite_diff_grad_x_coil(
        x_coil, xf, u, dt, params, grad_tip_p
    )

    # Compare
    np.testing.assert_allclose(
        grad_x_coil_vjp, grad_x_coil_fd,
        rtol=1e-4, atol=1e-6,
        err_msg="VJP grad_x_coil does not match finite differences"
    )

    print("PASS: grad_x_coil VJP matches finite differences")

def test_vjp_gradcheck_xf():
    """Test VJP gradient correctness for xf (tip state)."""
    n_act = 1
    x_coil, xf, u, dt, params = create_test_state(n_act)

    # Upstream gradient (random)
    grad_tip_p = np.random.randn(3)

    # VJP gradients
    result_vjp = crm_diff_py.true_legacy_step_vjp(
        x_coil, xf, u, dt, params['L_inserted'], params, grad_tip_p
    )
    grad_xf_vjp = result_vjp['grad_xf']

    # Finite difference reference
    grad_xf_fd = compute_finite_diff_grad_xf(
        x_coil, xf, u, dt, params, grad_tip_p
    )

    # Compare
    np.testing.assert_allclose(
        grad_xf_vjp, grad_xf_fd,
        rtol=1e-4, atol=1e-6,
        err_msg="VJP grad_xf does not match finite differences"
    )

    print("PASS: grad_xf VJP matches finite differences")

if __name__ == "__main__":
    test_vjp_gradcheck_x_coil()
    test_vjp_gradcheck_xf()
    print("\nAll state VJP gradient checks passed!")
```

### Expected Test Runtime

- **Per-element FD**: ~10ms (forward pass)
- **x_coil**: 18 elements → ~180ms
- **xf**: 15 elements → ~150ms
- **Total**: ~350ms per test
- **Both tests**: <1 second ✅

### Acceptance Criteria

1. ✅ Test runs without errors
2. ✅ VJP gradients match FD within rtol=1e-4, atol=1e-6
3. ✅ Test added to CI / sanity gate script

---

## Risk Analysis

### Risk: Test Failure (VJP Implementation Bug)

**Probability**: LOW
**Impact**: HIGH (would reveal critical bug)

**Why LOW probability**:
- Control gradients (∂u) already pass gradcheck
- Same Jacobian computation machinery used for state gradients
- IVP Jacobian code is well-tested (CRMSolverIVPJacobian)

**If test fails**:
- Bug likely in implicit term wiring (J_yx transpose multiplication)
- Fix in `src/CRM_TrueLegacyDynamics.cpp:313-327`
- Re-run test until pass

### Risk: Test Too Slow

**Probability**: LOW
**Impact**: LOW

**Mitigation**:
- Finite differences are embarrassingly parallel (can optimize later)
- Initial single-threaded implementation acceptable (<1 sec)
- If needed: use multiprocessing.Pool for FD loop

### Risk: Numerical Instability

**Probability**: MEDIUM
**Impact**: LOW

**Symptoms**: Test flakiness due to BVP non-convergence on perturbed states

**Mitigation**:
- Use simple test state (zero velocities, identity rotation)
- Use larger perturbation eps=1e-6 if needed
- Skip rotation matrix elements (indices 9-17) if unstable, focus on position/velocity

---

## Alternatives Considered

### Alternative 1: Linearization Numerical Accuracy Test

**Description**: Verify A, B matrices match finite differences

**Why deferred**:
- Linearization uses same Jacobians as VJP
- If VJP state gradients correct, linearization likely correct
- Can add as follow-up PR

### Alternative 2: Batched VJP Equivalence Test

**Description**: Verify batched VJP == looped single VJP

**Why deferred**:
- Batched code path uses same Jacobian computation
- Lower risk (mostly memory layout difference)
- VJP correctness is prerequisite

### Alternative 3: End-to-End Controller Trajectory Test

**Description**: Run iLQR and verify trajectory converges

**Why deferred**:
- Higher-level test (harder to debug failures)
- Requires correct state VJP as prerequisite
- Better as integration test after unit test passes

---

## Success Criteria

**Definition of Done**:
1. ✅ Test file `tests/test_fullstate_vjp_gradcheck_state.py` created
2. ✅ Test passes locally:
   ```bash
   PYTHONPATH=build:python:$PYTHONPATH python3 tests/test_fullstate_vjp_gradcheck_state.py
   ```
3. ✅ Test passes in CI (if integrated)
4. ✅ Documentation updated (this audit) with test status

**Verification Command**:
```bash
# Run new test
PYTHONPATH=build:python:$PYTHONPATH python3 tests/test_fullstate_vjp_gradcheck_state.py

# Expected output:
# PASS: grad_x_coil VJP matches finite differences
# PASS: grad_xf VJP matches finite differences
# All state VJP gradient checks passed!
```

---

## Implementation Estimate

**Effort**: 2-3 hours
- 1 hour: Write test structure and FD functions
- 0.5 hour: Debug shape/dimension issues
- 0.5 hour: Tune tolerances and edge cases
- 0.5 hour: Documentation and CI integration

**Scope**: Single PR, no production code changes

**Files Modified**:
1. **NEW**: `tests/test_fullstate_vjp_gradcheck_state.py` (~200 lines)
2. **UPDATED**: `docs/audits/AUDIT_FULLSTATE_STACK.md` (test status)

---

## Decision Rationale

**Why this action is highest priority**:

1. **Foundational correctness**: State gradients are critical for all trajectory optimization
2. **Minimal scope**: Pure test addition, no risk to production code
3. **High signal**: If test fails, reveals critical bug; if passes, high confidence in VJP stack
4. **Prerequisite for other work**: Linearization tests, batched VJP tests depend on this
5. **Single PR**: Can be completed and merged quickly

**Why NOT other actions**:
- Linearization test: Lower priority (same Jacobians)
- Batched VJP test: Lower priority (simpler code path)
- Convergence handling: Edge case (nominal path more important)
- Documentation improvements: Lower urgency (audits complete)

---

## Execution Plan

### Step 1: Create Test File (30 min)

```bash
cd /workspaces/CRM_DiffSim_Cl
cat > tests/test_fullstate_vjp_gradcheck_state.py <<'EOF'
# ... paste template above ...
EOF
```

### Step 2: Run Test (5 min)

```bash
PYTHONPATH=build:python:$PYTHONPATH python3 tests/test_fullstate_vjp_gradcheck_state.py
```

### Step 3: Debug if Needed (0-60 min)

If test fails:
1. Check shape errors (dimension mismatches)
2. Check tolerance settings (rtol, atol)
3. Inspect specific gradient elements (print intermediate values)
4. If systematic error: check VJP implementation in `src/CRM_TrueLegacyDynamics.cpp`

### Step 4: Update Audit (10 min)

Update `docs/audits/AUDIT_FULLSTATE_STACK.md`:
- Section E.1: Add `test_fullstate_vjp_gradcheck_state.py` to table
- Section E.3: Remove "VJP gradcheck for state" from Missing Tests

### Step 5: Commit (5 min)

```bash
git add tests/test_fullstate_vjp_gradcheck_state.py
git add docs/audits/AUDIT_FULLSTATE_STACK.md
git commit -m "Add VJP gradient check test for state variables (x_coil, xf)

Validates ∂L/∂x_coil and ∂L/∂xf via comparison with finite differences.
Completes VJP gradient test coverage (previously only tested ∂L/∂u).

Test runtime: <1 second
Tolerance: rtol=1e-4, atol=1e-6

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>"
```

---

## Conclusion

**Recommended Next Action**: Add state VJP gradient check test

**Single Deliverable**: `tests/test_fullstate_vjp_gradcheck_state.py`

**Impact**: Validates critical backward pass correctness for state gradients

**Risk**: LOW (test-only, no production changes)

**Effort**: 2-3 hours

**This action provides maximum validation value with minimal risk and scope.**
