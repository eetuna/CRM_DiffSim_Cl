"""
REDUCED 6D DYNAMICS (NON-LEGACY, ARCHIVED)

This module contains the reduced 6D state dynamics (u_0[3], v_0[3]).
This is NOT the TRUE legacy implementation.

TRUE legacy uses 18·N+15 state (full rigid body dynamics).
See: python/control/true_legacy_step.py

DEPRECATION WARNING: This module is archived and should not be used
for new code. Migrate to TRUE legacy dynamics.
"""
import warnings

warnings.warn(
    "Reduced6D dynamics is NON-legacy and archived. "
    "Use true_legacy_step for TRUE legacy (18·N+15) dynamics.",
    DeprecationWarning,
    stacklevel=2
)

from .legacy_state import LegacyState
from .legacy_state_adapter import (
    pack_legacy_state,
    unpack_legacy_state,
    legacy_state_to_numpy,
    numpy_to_legacy_state,
)
from .step_legacy_contract import (
    step_legacy_contract,
    vjp_legacy_contract,
    vjp_legacy_contract_batched,
)

__all__ = [
    'LegacyState',
    'pack_legacy_state',
    'unpack_legacy_state',
    'legacy_state_to_numpy',
    'numpy_to_legacy_state',
    'step_legacy_contract',
    'vjp_legacy_contract',
    'vjp_legacy_contract_batched',
]
