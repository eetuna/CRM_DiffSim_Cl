"""
FULLSTATE Control Module for Catheter Trajectory Optimization

Includes:
- iLQR solver for trajectory optimization (FULLSTATE 18·N+15)
- MPC for receding-horizon control (FULLSTATE 18·N+15)
- LQR for warm-start initialization (FULLSTATE 18·N+15)
- Hybrid controller combining MPC + learned policy (FULLSTATE 18·N+15)
- TRUE legacy state adapter (18·N+15, no reduction)
"""

# FULLSTATE controllers (18·N+15)
from .ilqr import iLQRSolver
from .mpc import MPCController
from .lqr import finite_horizon_lqr
from .hybrid_controller import HybridController, HybridControllerMetrics

# TRUE LEGACY (18·N+15) - DEFAULT DYNAMICS
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


def STATE_DIM_FULL(n_act):
    """Return FULLSTATE dimension (18*n_act + 15)."""
    return 18 * n_act + 15


__all__ = [
    # FULLSTATE controllers
    'iLQRSolver',
    'MPCController',
    'finite_horizon_lqr',
    'HybridController',
    'HybridControllerMetrics',

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
    'STATE_DIM_FULL',
]

# NO imports from reduced6d, NO backward aliases
