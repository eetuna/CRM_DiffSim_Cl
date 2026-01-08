"""
TRUE Legacy Step with PyTorch Autograd (A0 Milestone)

Implements PyTorch autograd Function for TRUE legacy step with implicit VJP.
Computes gradients w.r.t. packed state (18*N+15) and actuation currents.

Ground truth: docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple

import crm_diff_py
from .true_legacy_state_adapter import (
    pack_true_legacy_state,
    unpack_true_legacy_state,
    true_legacy_state_dim,
)


class TrueLegacyStepFn(torch.autograd.Function):
    """
    PyTorch autograd Function for TRUE legacy one-step dynamics.

    Forward: DynamicsBVP → DYNSolverIVP
    Backward: Implicit VJP using finite-difference Jacobians
    """

    @staticmethod
    def forward(ctx, x, u, dt, n_act, catheter_params, L_inserted, return_x_next):
        """
        Forward pass: TRUE legacy step.

        Args:
            x: Packed TRUE legacy state [18*n_act + 15] (torch.Tensor, float64)
            u: Actuation currents [n_act, 3] (torch.Tensor, float64)
            dt: Time step (float)
            n_act: Number of actuators (int)
            catheter_params: Dict with CathParams and CathConfig
            L_inserted: Insertion length (float, mm)
            return_x_next: If True, return x_next as well as tip_p

        Returns:
            tip_p: Tip position [3] (if return_x_next=False)
            (tip_p, x_next): Tip position and next state (if return_x_next=True)
        """
        # Validate inputs
        if x.dtype != torch.float64:
            raise ValueError("x must be float64")
        if u.dtype != torch.float64:
            raise ValueError("u must be float64")
        if x.dim() != 1:
            raise ValueError("x must be 1D (unbatched)")
        if u.dim() != 2 or u.shape[0] != n_act or u.shape[1] != 3:
            raise ValueError(f"u must be [{n_act}, 3]")

        # Unpack state
        x_coil, xf = unpack_true_legacy_state(x, n_act)

        # Convert to numpy (C-contiguous, float64)
        x_coil_np = x_coil.detach().cpu().numpy().astype(np.float64, order='C')
        xf_np = xf.detach().cpu().numpy().astype(np.float64, order='C')
        u_np = u.detach().cpu().numpy().astype(np.float64, order='C')

        # Add L_inserted to params (the existing binding expects it there)
        params_with_L = dict(catheter_params)
        params_with_L['L_inserted'] = L_inserted

        # Call C++ forward
        result = crm_diff_py.true_legacy_step_forward(
            x_coil_np, xf_np, u_np, dt, params_with_L, None, None
        )

        # Extract outputs
        tip_p_np = result['tip_p']  # [3]
        x_coil_next_np = result['x_coil_next']  # [n_act, 18]
        xf_next_np = result['xf_next']  # [15]
        converged = result['converged']

        if not converged:
            raise RuntimeError("BVP solver failed to converge in forward pass")

        # Convert to torch
        tip_p = torch.from_numpy(tip_p_np).to(dtype=torch.float64, device=x.device)

        # Save for backward
        ctx.save_for_backward(x, u)
        ctx.n_act = n_act
        ctx.catheter_params = catheter_params
        ctx.dt = dt
        ctx.L_inserted = L_inserted
        ctx.return_x_next = return_x_next

        if return_x_next:
            x_coil_next = torch.from_numpy(x_coil_next_np).to(dtype=torch.float64, device=x.device)
            xf_next = torch.from_numpy(xf_next_np).to(dtype=torch.float64, device=x.device)
            x_next = pack_true_legacy_state(x_coil_next, xf_next)
            return tip_p, x_next
        else:
            return tip_p

    @staticmethod
    def backward(ctx, *grad_outputs):
        """
        Backward pass: Compute VJP using implicit differentiation.

        Args:
            grad_outputs: Upstream gradients
                - grad_tip_p [3] if return_x_next=False
                - (grad_tip_p [3], grad_x_next [state_dim]) if return_x_next=True

        Returns:
            (grad_x, grad_u, None, None, None, None, None)
        """
        x, u = ctx.saved_tensors
        n_act = ctx.n_act
        catheter_params = ctx.catheter_params
        dt = ctx.dt
        L_inserted = ctx.L_inserted
        return_x_next = ctx.return_x_next

        # Extract gradients
        if return_x_next:
            grad_tip_p_direct, grad_x_next = grad_outputs

            # Combine grad_tip_p from both direct and x_next paths
            if grad_tip_p_direct is None:
                grad_tip_p_direct = torch.zeros(3, dtype=torch.float64)

            # Extract grad_tip_p from grad_x_next if present
            # tip_p is the first 3 components of xf_next, which is at the end of x_next
            if grad_x_next is not None:
                _, grad_xf_next = unpack_true_legacy_state(grad_x_next, n_act)
                grad_tip_p_from_xnext = grad_xf_next[:3]
                grad_tip_p = grad_tip_p_direct + grad_tip_p_from_xnext
            else:
                grad_tip_p = grad_tip_p_direct
                grad_x_next = None
        else:
            grad_tip_p = grad_outputs[0]
            if grad_tip_p is None:
                grad_tip_p = torch.zeros(3, dtype=torch.float64)
            grad_x_next = None

        # Convert to numpy
        x_coil, xf = unpack_true_legacy_state(x, n_act)
        x_coil_np = x_coil.detach().cpu().numpy().astype(np.float64, order='C')
        xf_np = xf.detach().cpu().numpy().astype(np.float64, order='C')
        u_np = u.detach().cpu().numpy().astype(np.float64, order='C')
        grad_tip_p_np = grad_tip_p.detach().cpu().numpy().astype(np.float64, order='C')

        # Add L_inserted to params
        params_with_L = dict(catheter_params)
        params_with_L['L_inserted'] = L_inserted

        # Call C++ VJP to compute gradients from tip_p cotangent
        vjp_result = crm_diff_py.true_legacy_step_vjp(
            x_coil_np, xf_np, u_np, dt, L_inserted, params_with_L, grad_tip_p_np
        )

        # Extract gradients from tip_p path
        grad_x_coil_np = vjp_result['grad_x_coil']  # [n_act, 18]
        grad_xf_np = vjp_result['grad_xf']  # [15]
        grad_u_np = vjp_result['grad_u']  # [n_act, 3]

        # Convert to torch
        grad_x_coil = torch.from_numpy(grad_x_coil_np).to(dtype=torch.float64, device=x.device)
        grad_xf = torch.from_numpy(grad_xf_np).to(dtype=torch.float64, device=x.device)
        grad_u = torch.from_numpy(grad_u_np).to(dtype=torch.float64, device=x.device)

        # Pack grad_x
        grad_x = pack_true_legacy_state(grad_x_coil, grad_xf)

        # Return gradients (matching forward signature)
        # (x, u, dt, n_act, catheter_params, L_inserted, return_x_next)
        return grad_x, grad_u, None, None, None, None, None


def true_legacy_step_torch(
    x: torch.Tensor,
    u: torch.Tensor,
    dt: float,
    *,
    n_act: int,
    catheter_params: Dict,
    L_inserted: float = 100.0,
    return_x_next: bool = False,
) -> torch.Tensor:
    """
    TRUE legacy step with PyTorch autograd support.

    Computes one forward step using TRUE legacy dynamics (DynamicsBVP → DYNSolverIVP)
    with gradient support via implicit VJP.

    Args:
        x: Packed TRUE legacy state [18*n_act + 15] (requires_grad=True for gradients)
        u: Actuation currents [n_act, 3] (requires_grad=True for gradients)
        dt: Time step (seconds)
        n_act: Number of actuators
        catheter_params: Dict from load_default_catheter_params()
        L_inserted: Insertion length (mm), default 100.0
        return_x_next: If True, return (tip_p, x_next); otherwise just tip_p

    Returns:
        tip_p: Tip position [3] (if return_x_next=False)
        (tip_p, x_next): Tip position and next state (if return_x_next=True)

    Example:
        >>> x = torch.zeros(33, dtype=torch.float64, requires_grad=True)  # N=1
        >>> u = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
        >>> tip_p = true_legacy_step_torch(x, u, 0.01, n_act=1, catheter_params=params)
        >>> loss = tip_p.sum()
        >>> loss.backward()
        >>> print(x.grad)  # Gradients w.r.t. state
        >>> print(u.grad)  # Gradients w.r.t. currents
    """
    return TrueLegacyStepFn.apply(x, u, dt, n_act, catheter_params, L_inserted, return_x_next)
