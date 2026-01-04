"""
CP3.2: Control module for catheter trajectory optimization

Includes:
- iLQR solver for trajectory optimization
- Milestone A (prototype): Hybrid state contract with implicit VJP
- A1: Legacy 6D state adapter (contract-exact)
"""
from .ilqr import iLQRSolver

# Milestone A prototype (9D hybrid state)
from .hybrid_state_contract import (
    HybridState, HybridStepResult, HybridVJPResult,
    pack_hybrid_state, unpack_hybrid_state,
    hybrid_to_legacy, legacy_to_hybrid,
    make_default_theta,
    STATE_DIM_HYBRID, STATE_DIM_DYNAMICS, STATE_DIM_OBSERVABLE,
    CONTROL_DIM, THETA_DIM, DEFAULT_THETA,
)
from .step_hybrid_legacy_contract import (
    step_hybrid_legacy_contract, vjp_hybrid_legacy_contract,
    hybrid_dynamics_step, HybridDynamicsStep,
)

# REDUCED 6D (ARCHIVED - NON-LEGACY)
# The reduced 6D state dynamics (u_0[3], v_0[3]) has been ARCHIVED.
# It is NOT the TRUE legacy implementation.
#
# To access (emits DeprecationWarning):
#   from python.control.reduced6d import LegacyState, step_legacy_contract
#
# For TRUE legacy (18·N+15), use true_legacy_step below.

# A0: TRUE legacy state adapter (18*N+15, no reduction)
from .true_legacy_state_adapter import (
    COIL_STATE_DIM, TIP_STATE_DIM,
    true_legacy_state_dim, true_legacy_warmstart_dim,
    pack_true_legacy_state, unpack_true_legacy_state,
    pack_true_legacy_warmstart, unpack_true_legacy_warmstart,
)
from .true_legacy_step import (
    true_legacy_step,
    true_legacy_linearize,
    true_legacy_tip_jacobian,
)
from .true_legacy_step_autograd import (
    true_legacy_step_torch,
    TrueLegacyStepFn,
)

__all__ = [
    # iLQR
    'iLQRSolver',

    # Milestone A prototype (9D hybrid state)
    'HybridState', 'HybridStepResult', 'HybridVJPResult',
    'pack_hybrid_state', 'unpack_hybrid_state',
    'hybrid_to_legacy', 'legacy_to_hybrid',
    'make_default_theta',
    'STATE_DIM_HYBRID', 'STATE_DIM_DYNAMICS', 'STATE_DIM_OBSERVABLE',
    'CONTROL_DIM', 'THETA_DIM', 'DEFAULT_THETA',
    'step_hybrid_legacy_contract', 'vjp_hybrid_legacy_contract',
    'hybrid_dynamics_step', 'HybridDynamicsStep',

    # Reduced 6D (ARCHIVED - import from python.control.reduced6d if needed)

    # TRUE LEGACY (18·N+15) - DEFAULT DYNAMICS
    'COIL_STATE_DIM', 'TIP_STATE_DIM',
    'true_legacy_state_dim', 'true_legacy_warmstart_dim',
    'pack_true_legacy_state', 'unpack_true_legacy_state',
    'pack_true_legacy_warmstart', 'unpack_true_legacy_warmstart',
    'true_legacy_step',
    'true_legacy_step_torch',
    'true_legacy_linearize',
    'true_legacy_tip_jacobian',
    'TrueLegacyStepFn',
]
