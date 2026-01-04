"""
Legacy State Adapter Functions

Pure conversion functions between LegacyState and numpy arrays.

NO physics, NO solver calls, NO reordering beyond the contract.
This module contains ONLY I/O transformations.
"""

import numpy as np
from typing import Tuple, List
from .legacy_state import LegacyState, STATE_DIM_LEGACY, U0_DIM, V0_DIM


def pack_legacy_state(u_0: np.ndarray, v_0: np.ndarray) -> LegacyState:
    """
    Pack curvature components into LegacyState.

    Args:
        u_0: Base curvature (3,) in 1/mm
        v_0: Base curvature velocity (3,) in 1/mm/s

    Returns:
        LegacyState object

    Raises:
        ValueError: If shapes are incorrect

    Example:
        >>> u_0 = np.array([0.01, 0.0, 0.0])
        >>> v_0 = np.array([0.1, 0.0, 0.0])
        >>> state = pack_legacy_state(u_0, v_0)
    """
    u_0 = np.asarray(u_0, dtype=np.float64)
    v_0 = np.asarray(v_0, dtype=np.float64)

    if u_0.shape != (U0_DIM,):
        raise ValueError(f"u_0 must be ({U0_DIM},), got {u_0.shape}")
    if v_0.shape != (V0_DIM,):
        raise ValueError(f"v_0 must be ({V0_DIM},), got {v_0.shape}")

    return LegacyState(u_0=u_0.copy(), v_0=v_0.copy())


def unpack_legacy_state(state: LegacyState) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unpack LegacyState into components.

    Args:
        state: LegacyState object

    Returns:
        (u_0, v_0) tuple of numpy arrays

    Raises:
        TypeError: If state is not LegacyState

    Example:
        >>> state = LegacyState.zeros()
        >>> u_0, v_0 = unpack_legacy_state(state)
    """
    if not isinstance(state, LegacyState):
        raise TypeError(f"Expected LegacyState, got {type(state)}")

    return state.u_0.copy(), state.v_0.copy()


def legacy_state_to_numpy(state: LegacyState) -> np.ndarray:
    """
    Convert LegacyState to numpy array (6,).

    Args:
        state: LegacyState object

    Returns:
        numpy array (6,) float64: [u_0, v_0] concatenated

    Raises:
        TypeError: If state is not LegacyState

    Example:
        >>> state = LegacyState.zeros()
        >>> x = legacy_state_to_numpy(state)
        >>> assert x.shape == (6,)
    """
    if not isinstance(state, LegacyState):
        raise TypeError(f"Expected LegacyState, got {type(state)}")

    return state.to_numpy()


def numpy_to_legacy_state(x: np.ndarray) -> LegacyState:
    """
    Convert numpy array to LegacyState.

    Args:
        x: numpy array (6,) or (1, 6)

    Returns:
        LegacyState object

    Raises:
        ValueError: If shape is incorrect or contains non-finite values

    Example:
        >>> x = np.zeros(6)
        >>> state = numpy_to_legacy_state(x)
    """
    x = np.asarray(x, dtype=np.float64)

    if not np.all(np.isfinite(x)):
        raise ValueError("State array contains non-finite values (NaN or Inf)")

    return LegacyState.from_numpy(x)


def batch_to_legacy_states(x_batch: np.ndarray) -> List[LegacyState]:
    """
    Convert batch (B, 6) to list of LegacyState.

    Args:
        x_batch: numpy array (B, 6)

    Returns:
        List of B LegacyState objects

    Raises:
        ValueError: If shape is incorrect

    Example:
        >>> x_batch = np.zeros((3, 6))
        >>> states = batch_to_legacy_states(x_batch)
        >>> assert len(states) == 3
    """
    x_batch = np.asarray(x_batch, dtype=np.float64)

    if x_batch.ndim != 2 or x_batch.shape[1] != STATE_DIM_LEGACY:
        raise ValueError(
            f"Batch must be (B, {STATE_DIM_LEGACY}), got {x_batch.shape}"
        )

    # Convert each row to a LegacyState
    return [
        LegacyState.from_numpy(x_batch[i:i+1])
        for i in range(x_batch.shape[0])
    ]


__all__ = [
    'pack_legacy_state',
    'unpack_legacy_state',
    'legacy_state_to_numpy',
    'numpy_to_legacy_state',
    'batch_to_legacy_states',
]
