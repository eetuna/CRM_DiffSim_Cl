"""
Legacy 6D State Representation

Enforces the authoritative legacy contract from:
  docs/contracts/LEGACY_HYBRID_STATE_CONTRACT.md

State layout (6D):
  x[0:3] = u_0 (base curvature, 1/mm)
  x[3:6] = v_0 (base curvature velocity, 1/mm/s)

Observables (p_tip, u_tip) are NOT part of the state.
"""

import numpy as np
from typing import Union
from dataclasses import dataclass

# Constants (from frozen contract)
STATE_DIM_LEGACY = 6
U0_DIM = 3
V0_DIM = 3


@dataclass(frozen=True)
class LegacyState:
    """
    Immutable 6D legacy state representation.

    This enforces the exact legacy contract: state contains ONLY
    the dynamic variables (u_0, v_0). Observables (p_tip, u_tip)
    are returned separately and are NOT part of the state vector.

    Fields:
        u_0: Base curvature (3,) in 1/mm
        v_0: Base curvature velocity (3,) in 1/mm/s

    Evidence: src/CRM_DiffDynamics.hpp:11
        double x_next[6];  // [u_0_{t+1}, v_0_{t+1}]
    """
    u_0: np.ndarray  # (3,) float64
    v_0: np.ndarray  # (3,) float64

    def __post_init__(self):
        """Validate shapes and dtypes on construction."""
        # Use object.__setattr__ because dataclass is frozen
        if self.u_0.shape != (U0_DIM,):
            raise ValueError(f"u_0 must be ({U0_DIM},), got {self.u_0.shape}")
        if self.v_0.shape != (V0_DIM,):
            raise ValueError(f"v_0 must be ({V0_DIM},), got {self.v_0.shape}")
        if self.u_0.dtype != np.float64:
            raise ValueError(f"u_0 must be float64, got {self.u_0.dtype}")
        if self.v_0.dtype != np.float64:
            raise ValueError(f"v_0 must be float64, got {self.v_0.dtype}")

    def to_numpy(self) -> np.ndarray:
        """
        Convert to flat (6,) numpy array.

        Returns:
            numpy array (6,) float64: [u_0, v_0] concatenated
        """
        return np.concatenate([self.u_0, self.v_0])

    @classmethod
    def from_numpy(cls, x: np.ndarray) -> 'LegacyState':
        """
        Create LegacyState from flat (6,) or batch (B, 6) array.

        Args:
            x: numpy array (6,) or (B, 6) where B must be 1

        Returns:
            LegacyState object

        Raises:
            ValueError: If shape is incorrect

        Note:
            For batch (B, 6), only B=1 is supported (single state extraction).
            For full batch conversion, use batch_to_legacy_states().
        """
        x = np.asarray(x, dtype=np.float64)

        if x.ndim == 1:
            if x.shape[0] != STATE_DIM_LEGACY:
                raise ValueError(
                    f"State must be ({STATE_DIM_LEGACY},), got {x.shape}"
                )
            return cls(u_0=x[0:3].copy(), v_0=x[3:6].copy())

        elif x.ndim == 2:
            if x.shape[1] != STATE_DIM_LEGACY:
                raise ValueError(
                    f"State must be (B, {STATE_DIM_LEGACY}), got {x.shape}"
                )
            # Only allow B=1 for single-state extraction
            if x.shape[0] != 1:
                raise ValueError(
                    f"Batch conversion only supports B=1, got B={x.shape[0]}. "
                    f"Use batch_to_legacy_states() for full batch conversion."
                )
            return cls(u_0=x[0, 0:3].copy(), v_0=x[0, 3:6].copy())

        else:
            raise ValueError(f"State array must be 1D or 2D, got {x.ndim}D")

    @classmethod
    def zeros(cls, dtype=np.float64) -> 'LegacyState':
        """
        Create zero state.

        Args:
            dtype: Data type (default: float64)

        Returns:
            LegacyState with all zeros
        """
        return cls(
            u_0=np.zeros(U0_DIM, dtype=dtype),
            v_0=np.zeros(V0_DIM, dtype=dtype)
        )


__all__ = [
    'LegacyState',
    'STATE_DIM_LEGACY',
    'U0_DIM',
    'V0_DIM',
]
