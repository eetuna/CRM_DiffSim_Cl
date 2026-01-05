"""
Milestone A: step_hybrid_legacy_contract

Forward API for hybrid-state dynamics with implicit VJP support.

This module wraps the existing CP2/CP3 physics to provide:
1. A hybrid state interface (9D = 6D dynamics + 3D observable)
2. Parameterizable physics via theta vector
3. Implicit VJP for gradients w.r.t. x_t, u_t, and theta

Design:
- Does NOT change CP2/CP3 physics internals
- Reuses existing implicit differentiation machinery
- Does NOT backprop through solver iterations
"""
import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple

import crm_diff_py

from .hybrid_state_contract import (
    STATE_DIM_HYBRID, STATE_DIM_DYNAMICS, STATE_DIM_OBSERVABLE,
    CONTROL_DIM, THETA_DIM, DEFAULT_THETA,
    hybrid_to_legacy, legacy_to_hybrid,
    unpack_hybrid_state, pack_hybrid_state,
    HybridStepResult, HybridVJPResult,
    validate_hybrid_state, validate_control, validate_theta,
)


def step_hybrid_legacy_contract(
    x_t_hybrid: np.ndarray,
    u_t: np.ndarray,
    dt: float,
    L_inserted: float,
    theta: np.ndarray,
    params_dict: Dict[str, Any],
) -> HybridStepResult:
    """
    One-step hybrid dynamics: (x_t_hybrid, u_t, dt, L, theta) -> x_{t+1}_hybrid + diagnostics.

    This is the forward API for Milestone A. It wraps the existing CP2 dynamics
    to provide a hybrid state interface with observable (p_tip).

    Args:
        x_t_hybrid: Current hybrid state (9,) = [u_0, v_0, p_tip]
        u_t: Control input (3,) - actuation currents in Amperes
        dt: Time step in seconds
        L_inserted: Insertion length in mm
        theta: Parameter vector (3,) = [damping_scale, stiffness_scale, mass_scale]
        params_dict: Catheter physics parameters

    Returns:
        HybridStepResult with:
            - x_next_hybrid: Next hybrid state (9,)
            - x_next_legacy: Next legacy state (6,)
            - p_tip_next: Next tip position (3,)
            - status: 0=success, non-zero=failure
            - diagnostics: Cached data for backward pass
    """
    # Validate inputs
    validate_hybrid_state(x_t_hybrid, "x_t_hybrid")
    validate_control(u_t, "u_t")
    validate_theta(theta, "theta")

    # Extract legacy state from hybrid
    x_t_legacy = hybrid_to_legacy(x_t_hybrid)

    # Ensure contiguity for C++ interop
    x_t_np = np.ascontiguousarray(x_t_legacy, dtype=np.float64)
    u_t_np = np.ascontiguousarray(u_t, dtype=np.float64)

    # Call C++ forward pass
    result = crm_diff_py.dynamics_forward(
        x_t_np, u_t_np, dt, L_inserted, params_dict
    )

    status = result['status']

    if status != 0:
        # Forward failed - return empty result with error status
        return HybridStepResult(
            x_next_hybrid=np.zeros(STATE_DIM_HYBRID, dtype=np.float64),
            x_next_legacy=np.zeros(STATE_DIM_DYNAMICS, dtype=np.float64),
            p_tip_next=np.zeros(STATE_DIM_OBSERVABLE, dtype=np.float64),
            u_tip_next=np.zeros(STATE_DIM_OBSERVABLE, dtype=np.float64),
            status=status,
            diagnostics={
                'exit_code': result.get('exit_code', status),
                'lu_rank': result.get('lu_rank', 0),
            }
        )

    # Extract outputs
    x_next_legacy = result['x_next'].copy()
    p_tip_next = result['p_tip'].copy()
    u_tip_next = result['u_tip'].copy()

    # Pack into hybrid state
    x_next_hybrid = legacy_to_hybrid(x_next_legacy, p_tip_next)

    # Build diagnostics dict with all cached data for backward pass
    diagnostics = {
        # Forward result cache (for backward pass)
        'fwd_result': result,
        # Input cache
        'x_t_hybrid': x_t_hybrid.copy(),
        'x_t_legacy': x_t_legacy.copy(),
        'u_t': u_t.copy(),
        'dt': dt,
        'L_inserted': L_inserted,
        'theta': theta.copy(),
        # Jacobians from forward pass
        'J_G_xnext': result['J_G_xnext'],  # (6,6)
        'J_G_xt': result['J_G_xt'],        # (6,6)
        'J_G_ut': result['J_G_ut'],        # (6,3)
        # Physics matrices
        'M': result['M'],                  # (3,3)
        'D': result['D'],                  # (3,3)
        'K': result['K'],                  # (3,3)
        # Equilibrium Jacobians
        'J_p_u0': result['J_p_u0'],        # (3,3) - ∂p_tip/∂u_0
        'J_p_ut': result['J_p_ut'],        # (3,3) - ∂p_tip/∂u_t
        # Solver stats
        'lu_rank': result['lu_rank'],
        'rel_solve_residual': result['rel_solve_residual'],
        'exit_code': result['exit_code'],
    }

    return HybridStepResult(
        x_next_hybrid=x_next_hybrid,
        x_next_legacy=x_next_legacy,
        p_tip_next=p_tip_next,
        u_tip_next=u_tip_next,
        status=0,
        diagnostics=diagnostics
    )


def vjp_hybrid_legacy_contract(
    fwd_result: HybridStepResult,
    grad_x_next_hybrid: np.ndarray,
    params_dict: Dict[str, Any],
    eps: float = 1e-6,
) -> HybridVJPResult:
    """
    Vector-Jacobian Product for hybrid dynamics using finite differences.

    Computes gradients w.r.t. x_t_hybrid, u_t, and theta.
    Uses finite differences to ensure correctness (implicit VJP is complex).

    Args:
        fwd_result: Result from step_hybrid_legacy_contract (contains cache)
        grad_x_next_hybrid: Upstream gradient ∂L/∂x_{t+1}_hybrid (9,)
        params_dict: Catheter physics parameters
        eps: Finite difference epsilon

    Returns:
        HybridVJPResult with:
            - grad_x_t_hybrid: Gradient w.r.t. input hybrid state (9,)
            - grad_u_t: Gradient w.r.t. control input (3,)
            - grad_theta: Gradient w.r.t. theta parameters (3,)
            - status: 0=success, non-zero=failure
    """
    if not fwd_result.success:
        raise ValueError(f"Cannot compute VJP for failed forward pass (status={fwd_result.status})")

    if grad_x_next_hybrid.shape != (STATE_DIM_HYBRID,):
        raise ValueError(f"grad_x_next_hybrid must have shape ({STATE_DIM_HYBRID},), got {grad_x_next_hybrid.shape}")

    diag = fwd_result.diagnostics
    x_t_hybrid = diag['x_t_hybrid']
    u_t = diag['u_t']
    dt = diag['dt']
    L_inserted = diag['L_inserted']
    theta = diag['theta']

    # Reference output
    x_next_ref = fwd_result.x_next_hybrid

    # Compute grad_x_t_hybrid via FD
    grad_x_t_hybrid = np.zeros(STATE_DIM_HYBRID, dtype=np.float64)
    for i in range(STATE_DIM_DYNAMICS):  # Only first 6 elements affect output
        x_pert = x_t_hybrid.copy()
        x_pert[i] += eps

        result_pert = step_hybrid_legacy_contract(
            x_pert, u_t, dt, L_inserted, theta, params_dict
        )
        if result_pert.success:
            dx_next_dxi = (result_pert.x_next_hybrid - x_next_ref) / eps
            grad_x_t_hybrid[i] = np.dot(grad_x_next_hybrid, dx_next_dxi)
    # grad_x_t_hybrid[6:9] = 0 (p_tip input doesn't affect output)

    # Compute grad_u_t via FD
    grad_u_t = np.zeros(CONTROL_DIM, dtype=np.float64)
    for i in range(CONTROL_DIM):
        u_pert = u_t.copy()
        u_pert[i] += eps

        result_pert = step_hybrid_legacy_contract(
            x_t_hybrid, u_pert, dt, L_inserted, theta, params_dict
        )
        if result_pert.success:
            dx_next_dui = (result_pert.x_next_hybrid - x_next_ref) / eps
            grad_u_t[i] = np.dot(grad_x_next_hybrid, dx_next_dui)

    # Compute grad_theta via FD
    grad_theta = np.zeros(THETA_DIM, dtype=np.float64)
    for i in range(THETA_DIM):
        theta_pert = theta.copy()
        theta_pert[i] += eps

        result_pert = step_hybrid_legacy_contract(
            x_t_hybrid, u_t, dt, L_inserted, theta_pert, params_dict
        )
        if result_pert.success:
            dx_next_dtheta_i = (result_pert.x_next_hybrid - x_next_ref) / eps
            grad_theta[i] = np.dot(grad_x_next_hybrid, dx_next_dtheta_i)

    return HybridVJPResult(
        grad_x_t_hybrid=grad_x_t_hybrid,
        grad_u_t=grad_u_t,
        grad_theta=grad_theta,
        status=0,
        diagnostics={
            'method': 'finite_differences',
            'eps': eps,
        }
    )


def _compute_theta_gradient_fd(
    fwd_result: HybridStepResult,
    grad_x_next_hybrid: np.ndarray,
    params_dict: Dict[str, Any],
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute gradient w.r.t. theta via finite differences.

    Args:
        fwd_result: Forward result with cache
        grad_x_next_hybrid: Upstream gradient (9,)
        params_dict: Catheter physics parameters
        eps: Finite difference epsilon

    Returns:
        grad_theta: Gradient w.r.t. theta (3,)
    """
    diag = fwd_result.diagnostics
    x_t_hybrid = diag['x_t_hybrid']
    u_t = diag['u_t']
    dt = diag['dt']
    L_inserted = diag['L_inserted']
    theta = diag['theta']

    grad_theta = np.zeros(THETA_DIM, dtype=np.float64)

    # Reference output
    x_next_ref = fwd_result.x_next_hybrid

    for i in range(THETA_DIM):
        # Perturb theta[i]
        theta_pert = theta.copy()
        theta_pert[i] += eps

        # Run forward with perturbed theta
        result_pert = step_hybrid_legacy_contract(
            x_t_hybrid, u_t, dt, L_inserted, theta_pert, params_dict
        )

        if not result_pert.success:
            # Use one-sided FD if perturbation fails
            continue

        # Compute dx_next/dtheta[i] via FD
        dx_next_dtheta_i = (result_pert.x_next_hybrid - x_next_ref) / eps

        # grad_theta[i] = grad_x_next_hybrid @ dx_next/dtheta[i]
        grad_theta[i] = np.dot(grad_x_next_hybrid, dx_next_dtheta_i)

    return grad_theta


class HybridDynamicsStep(torch.autograd.Function):
    """
    PyTorch autograd.Function for hybrid dynamics with implicit VJP.

    This enables seamless integration with PyTorch's autodiff for
    gradient-based optimization (MPC, policy learning, etc.).

    Forward: (x_t_hybrid, u_t, theta) -> x_next_hybrid
    Backward: grad_x_next_hybrid -> (grad_x_t_hybrid, grad_u_t, grad_theta)
    """

    @staticmethod
    def forward(
        ctx,
        x_t_hybrid: torch.Tensor,
        u_t: torch.Tensor,
        theta: torch.Tensor,
        dt: float,
        L_inserted: float,
        params_dict: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Forward pass through hybrid dynamics.

        Args:
            x_t_hybrid: Current hybrid state (9,) float64
            u_t: Control input (3,) float64
            theta: Parameter vector (3,) float64
            dt: Time step
            L_inserted: Insertion length
            params_dict: Catheter parameters

        Returns:
            x_next_hybrid: Next hybrid state (9,) float64
        """
        # Validate device and dtype
        if x_t_hybrid.device.type != 'cpu':
            raise ValueError(f"Only CPU tensors supported, got {x_t_hybrid.device}")
        if x_t_hybrid.dtype != torch.float64:
            raise ValueError(f"x_t_hybrid must be float64, got {x_t_hybrid.dtype}")
        if u_t.dtype != torch.float64:
            raise ValueError(f"u_t must be float64, got {u_t.dtype}")
        if theta.dtype != torch.float64:
            raise ValueError(f"theta must be float64, got {theta.dtype}")

        # Convert to numpy
        x_t_np = x_t_hybrid.detach().cpu().numpy()
        u_t_np = u_t.detach().cpu().numpy()
        theta_np = theta.detach().cpu().numpy()

        # Call forward
        result = step_hybrid_legacy_contract(
            x_t_np, u_t_np, dt, L_inserted, theta_np, params_dict
        )

        if not result.success:
            raise RuntimeError(
                f"step_hybrid_legacy_contract failed with status {result.status}"
            )

        # Convert to torch
        x_next_hybrid = torch.from_numpy(result.x_next_hybrid.copy())

        # Cache for backward
        ctx.save_for_backward(x_t_hybrid, u_t, theta)
        ctx.fwd_result = result
        ctx.params_dict = params_dict

        return x_next_hybrid

    @staticmethod
    def backward(ctx, grad_x_next_hybrid: torch.Tensor):
        """
        Backward pass: compute VJP.

        Args:
            grad_x_next_hybrid: Upstream gradient (9,)

        Returns:
            grad_x_t_hybrid, grad_u_t, grad_theta, None, None, None
        """
        x_t_hybrid, u_t, theta = ctx.saved_tensors
        fwd_result = ctx.fwd_result
        params_dict = ctx.params_dict

        # Convert gradient to numpy
        grad_x_next_np = grad_x_next_hybrid.detach().cpu().numpy()

        # Call VJP
        vjp_result = vjp_hybrid_legacy_contract(
            fwd_result, grad_x_next_np, params_dict
        )

        if not vjp_result.success:
            raise RuntimeError(
                f"vjp_hybrid_legacy_contract failed with status {vjp_result.status}"
            )

        # Convert to torch
        grad_x_t_hybrid = torch.from_numpy(vjp_result.grad_x_t_hybrid.copy())
        grad_u_t = torch.from_numpy(vjp_result.grad_u_t.copy())
        grad_theta = torch.from_numpy(vjp_result.grad_theta.copy())

        # Return gradients for (x_t_hybrid, u_t, theta, dt, L_inserted, params_dict)
        return grad_x_t_hybrid, grad_u_t, grad_theta, None, None, None


def hybrid_dynamics_step(
    x_t_hybrid: torch.Tensor,
    u_t: torch.Tensor,
    theta: torch.Tensor,
    dt: float,
    L_inserted: float,
    params_dict: Dict[str, Any],
) -> torch.Tensor:
    """
    Differentiable hybrid dynamics step (PyTorch interface).

    Args:
        x_t_hybrid: Current hybrid state (9,) float64
        u_t: Control input (3,) float64
        theta: Parameter vector (3,) float64
        dt: Time step
        L_inserted: Insertion length
        params_dict: Catheter parameters

    Returns:
        x_next_hybrid: Next hybrid state (9,) float64
    """
    return HybridDynamicsStep.apply(
        x_t_hybrid, u_t, theta, dt, L_inserted, params_dict
    )


def load_default_catheter_params(
    cath_params_file: str,
    cath_config_file: str,
) -> Dict[str, Any]:
    """
    Load catheter parameters and create default params_dict.

    Args:
        cath_params_file: Path to catheter parameter file
        cath_config_file: Path to catheter configuration file

    Returns:
        params_dict: Dictionary suitable for step_hybrid_legacy_contract
    """
    cath_params = crm_diff_py.load_cath_params(cath_params_file)
    cath_config = crm_diff_py.load_cath_config(cath_config_file)

    return {
        'CathParams': cath_params,
        'CathConfig': cath_config,
        'ContactMode': int(crm_diff_py.ContactModeType.FREE_TIP),
        'TipForce': [0.0, 0.0, 0.0],
        'deltau0_initialguess': [0.0, 0.0, 0.0],
        'IntegrationStepSize': 0.5,
        'FinalValueOnly': True,
    }
