"""
Milestone A: Hybrid State Contract

Defines the legacy-hybrid state representation for the catheter dynamics
with support for implicit VJP (Vector-Jacobian Product).

State Layout:
    Hybrid state x_hybrid is a 9D vector:
        x_hybrid[0:3] = u_0      - base curvature (1/mm)
        x_hybrid[3:6] = v_0      - base curvature velocity (1/mm/s)
        x_hybrid[6:9] = p_tip    - tip position (mm)

    The first 6 elements match the legacy CP2 state representation.
    The p_tip observable is computed from equilibrium at u_0.

Theta (Parameter) Vector:
    A small set of tunable physics parameters:
        theta[0] = damping_scale    - multiplier for damping matrix D
        theta[1] = stiffness_scale  - multiplier for stiffness matrix K
        theta[2] = mass_scale       - multiplier for inertia matrix M

    Default: theta = [1.0, 1.0, 1.0] (unmodified physics)

API Contract:
    - Input:  x_t_hybrid (9,), u_t (3,), dt, L, theta (3,)
    - Output: x_{t+1}_hybrid (9,) + diagnostics dict
    - VJP: Gradients w.r.t. x_t_hybrid, u_t, and theta
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

# State dimensions
STATE_DIM_DYNAMICS = 6    # Legacy CP2 state: [u_0, v_0]
STATE_DIM_OBSERVABLE = 3  # Tip position p_tip
STATE_DIM_HYBRID = 9      # Full hybrid state

# Control dimension
CONTROL_DIM = 3           # Actuation currents

# Parameter dimension (theta)
THETA_DIM = 3             # [damping_scale, stiffness_scale, mass_scale]

# Default theta: no scaling (identity)
DEFAULT_THETA = np.array([1.0, 1.0, 1.0], dtype=np.float64)


@dataclass
class HybridState:
    """
    Hybrid state representation combining dynamics state with observables.

    Attributes:
        u_0: Base curvature (3,) in 1/mm
        v_0: Base curvature velocity (3,) in 1/mm/s
        p_tip: Tip position (3,) in mm
    """
    u_0: np.ndarray    # (3,) float64
    v_0: np.ndarray    # (3,) float64
    p_tip: np.ndarray  # (3,) float64

    @classmethod
    def from_packed(cls, x_hybrid: np.ndarray) -> 'HybridState':
        """Unpack a 9D hybrid state vector."""
        if x_hybrid.shape != (STATE_DIM_HYBRID,):
            raise ValueError(f"Expected shape ({STATE_DIM_HYBRID},), got {x_hybrid.shape}")
        return cls(
            u_0=x_hybrid[0:3].copy(),
            v_0=x_hybrid[3:6].copy(),
            p_tip=x_hybrid[6:9].copy()
        )

    def to_packed(self) -> np.ndarray:
        """Pack into a 9D hybrid state vector."""
        return np.concatenate([self.u_0, self.v_0, self.p_tip])

    def to_legacy_state(self) -> np.ndarray:
        """Extract the 6D legacy dynamics state [u_0, v_0]."""
        return np.concatenate([self.u_0, self.v_0])

    @classmethod
    def from_legacy_and_observable(
        cls,
        x_legacy: np.ndarray,
        p_tip: np.ndarray
    ) -> 'HybridState':
        """
        Create hybrid state from legacy 6D state and tip position observable.

        Args:
            x_legacy: Legacy state [u_0, v_0] (6,)
            p_tip: Tip position (3,)
        """
        if x_legacy.shape != (STATE_DIM_DYNAMICS,):
            raise ValueError(f"Expected x_legacy shape ({STATE_DIM_DYNAMICS},), got {x_legacy.shape}")
        if p_tip.shape != (STATE_DIM_OBSERVABLE,):
            raise ValueError(f"Expected p_tip shape ({STATE_DIM_OBSERVABLE},), got {p_tip.shape}")
        return cls(
            u_0=x_legacy[0:3].copy(),
            v_0=x_legacy[3:6].copy(),
            p_tip=p_tip.copy()
        )


def pack_hybrid_state(u_0: np.ndarray, v_0: np.ndarray, p_tip: np.ndarray) -> np.ndarray:
    """
    Pack components into a 9D hybrid state vector.

    Args:
        u_0: Base curvature (3,) in 1/mm
        v_0: Base curvature velocity (3,) in 1/mm/s
        p_tip: Tip position (3,) in mm

    Returns:
        x_hybrid: Packed state (9,)
    """
    return np.concatenate([
        np.asarray(u_0, dtype=np.float64),
        np.asarray(v_0, dtype=np.float64),
        np.asarray(p_tip, dtype=np.float64)
    ])


def unpack_hybrid_state(x_hybrid: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Unpack a 9D hybrid state vector into components.

    Args:
        x_hybrid: Packed state (9,)

    Returns:
        u_0: Base curvature (3,)
        v_0: Base curvature velocity (3,)
        p_tip: Tip position (3,)
    """
    if x_hybrid.shape != (STATE_DIM_HYBRID,):
        raise ValueError(f"Expected shape ({STATE_DIM_HYBRID},), got {x_hybrid.shape}")
    return x_hybrid[0:3], x_hybrid[3:6], x_hybrid[6:9]


def hybrid_to_legacy(x_hybrid: np.ndarray) -> np.ndarray:
    """
    Convert 9D hybrid state to 6D legacy state (drops p_tip).

    Args:
        x_hybrid: Hybrid state (9,)

    Returns:
        x_legacy: Legacy state [u_0, v_0] (6,)
    """
    if x_hybrid.shape != (STATE_DIM_HYBRID,):
        raise ValueError(f"Expected shape ({STATE_DIM_HYBRID},), got {x_hybrid.shape}")
    return x_hybrid[0:6].copy()


def legacy_to_hybrid(x_legacy: np.ndarray, p_tip: np.ndarray) -> np.ndarray:
    """
    Convert 6D legacy state to 9D hybrid state by adding p_tip.

    Args:
        x_legacy: Legacy state [u_0, v_0] (6,)
        p_tip: Tip position (3,)

    Returns:
        x_hybrid: Hybrid state (9,)
    """
    if x_legacy.shape != (STATE_DIM_DYNAMICS,):
        raise ValueError(f"Expected x_legacy shape ({STATE_DIM_DYNAMICS},), got {x_legacy.shape}")
    if p_tip.shape != (STATE_DIM_OBSERVABLE,):
        raise ValueError(f"Expected p_tip shape ({STATE_DIM_OBSERVABLE},), got {p_tip.shape}")
    return np.concatenate([x_legacy, p_tip])


def make_default_theta() -> np.ndarray:
    """
    Create default theta (parameter) vector with no scaling.

    Returns:
        theta: [1.0, 1.0, 1.0] (3,)
    """
    return DEFAULT_THETA.copy()


@dataclass
class HybridStepResult:
    """
    Result of a hybrid dynamics step.

    Attributes:
        x_next_hybrid: Next hybrid state (9,)
        x_next_legacy: Next legacy state (6,) - for compatibility
        p_tip_next: Next tip position (3,)
        u_tip_next: Next tip curvature (3,)
        status: 0 = success, non-zero = failure
        diagnostics: Additional info (Jacobians, solver stats, etc.)
    """
    x_next_hybrid: np.ndarray    # (9,)
    x_next_legacy: np.ndarray    # (6,)
    p_tip_next: np.ndarray       # (3,)
    u_tip_next: np.ndarray       # (3,)
    status: int
    diagnostics: Dict[str, Any]

    @property
    def success(self) -> bool:
        return self.status == 0


@dataclass
class HybridVJPResult:
    """
    Result of hybrid VJP (backward pass).

    Attributes:
        grad_x_t_hybrid: Gradient w.r.t. input hybrid state (9,)
        grad_u_t: Gradient w.r.t. control input (3,)
        grad_theta: Gradient w.r.t. theta parameters (3,)
        status: 0 = success, non-zero = failure
        diagnostics: Additional info (solver stats, etc.)
    """
    grad_x_t_hybrid: np.ndarray  # (9,)
    grad_u_t: np.ndarray         # (3,)
    grad_theta: np.ndarray       # (3,)
    status: int
    diagnostics: Dict[str, Any]

    @property
    def success(self) -> bool:
        return self.status == 0


# Validation utilities
def validate_hybrid_state(x_hybrid: np.ndarray, name: str = "x_hybrid") -> None:
    """Validate that x_hybrid has correct shape and dtype."""
    if not isinstance(x_hybrid, np.ndarray):
        raise TypeError(f"{name} must be numpy array, got {type(x_hybrid)}")
    if x_hybrid.shape != (STATE_DIM_HYBRID,):
        raise ValueError(f"{name} must have shape ({STATE_DIM_HYBRID},), got {x_hybrid.shape}")
    if x_hybrid.dtype != np.float64:
        raise ValueError(f"{name} must be float64, got {x_hybrid.dtype}")


def validate_control(u_t: np.ndarray, name: str = "u_t") -> None:
    """Validate that u_t has correct shape and dtype."""
    if not isinstance(u_t, np.ndarray):
        raise TypeError(f"{name} must be numpy array, got {type(u_t)}")
    if u_t.shape != (CONTROL_DIM,):
        raise ValueError(f"{name} must have shape ({CONTROL_DIM},), got {u_t.shape}")
    if u_t.dtype != np.float64:
        raise ValueError(f"{name} must be float64, got {u_t.dtype}")


def validate_theta(theta: np.ndarray, name: str = "theta") -> None:
    """Validate that theta has correct shape and dtype."""
    if not isinstance(theta, np.ndarray):
        raise TypeError(f"{name} must be numpy array, got {type(theta)}")
    if theta.shape != (THETA_DIM,):
        raise ValueError(f"{name} must have shape ({THETA_DIM},), got {theta.shape}")
    if theta.dtype != np.float64:
        raise ValueError(f"{name} must be float64, got {theta.dtype}")
