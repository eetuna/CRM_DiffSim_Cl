"""
TRUE Legacy Step Wrapper (A0 Contract)

Implements one-step forward wrapper using TRUE legacy state (18·N + 15)
and the canonical stepping sequence: DynamicsBVP → DYNSolverIVP

Ground truth: docs/contracts/TRUE_LEGACY_HYBRID_STATE_CONTRACT.md
Evidence: docs/audits/TRUE_LEGACY_STEP_ENTRYPOINT_AUDIT.md
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple

import crm_diff_py
from .true_legacy_state_adapter import (
    pack_true_legacy_state,
    unpack_true_legacy_state,
    true_legacy_state_dim,
    COIL_STATE_DIM,
    TIP_STATE_DIM,
)
def true_legacy_step(
    x: torch.Tensor,
    u: torch.Tensor,
    dt: float,
    *,
    n_act: int,
    catheter_params: Optional[Dict] = None,
    warmstart: Optional[Dict] = None,
    return_orientation: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    """
    One-step forward using TRUE legacy dynamics (DynamicsBVP → DYNSolverIVP).

    State representation: 18·N + 15 (no reduction, full rotation matrices)
    - Coil states: [v, w, p, R] per coil (18 each)
    - Tip state: [p, R, u] (15 total)

    Args:
        x: Packed TRUE legacy state
            - Unbatched: [18*n_act + 15]
            - Batched: [B, 18*n_act + 15]
        u: Actuation currents (Amperes)
            - Unbatched: [n_act, 3]
            - Batched: [B, n_act, 3]
        dt: Timestep (seconds)
        n_act: Number of actuator sets (coils)
        catheter_params: Optional dict containing catheter parameters.
            If None, uses default parameters.
        warmstart: Optional dict containing warm-start guesses:
            - 'mL_guess': [n_act, 3] or [B, n_act, 3] (interface moments)
            - 'nL_guess': [n_act, 3] or [B, n_act, 3] (interface forces)
        return_orientation: If True, return tip orientation in observables

    Returns:
        (x_next, obs) where:
        - x_next: Packed TRUE legacy state (same shape as x)
        - obs: Dict containing:
            - 'tip_p': Tip position [3] or [B, 3]
            - 'tip_R': Tip orientation [9] or [B, 9] (if return_orientation=True)
            - 'tip_u': Tip curvature [3] or [B, 3]
            - 'converged': bool or [B] bool tensor
            - 'warmstart_next': Dict with updated mL_guess and nL_guess

    Raises:
        ValueError: If input shapes are invalid
    """
    # Detect if inputs are torch tensors or numpy arrays
    is_torch_input = isinstance(x, torch.Tensor)

    # Store original dtype and device for torch, or dtype for numpy
    if is_torch_input:
        original_dtype = x.dtype
        original_device = x.device
    else:
        original_dtype = x.dtype
        original_device = None

    # Validate inputs
    x_ndim = x.ndim
    if x_ndim not in [1, 2]:
        raise ValueError(
            f"x must be 1D [state_dim] or 2D [B, state_dim], got shape {x.shape}"
        )

    is_batched = (x_ndim == 2)
    expected_state_dim = true_legacy_state_dim(n_act)

    if is_batched:
        batch_size = x.shape[0]
        if x.shape[1] != expected_state_dim:
            raise ValueError(
                f"Batched x last dimension must be {expected_state_dim} for n_act={n_act}, "
                f"got {x.shape[1]}"
            )

        # Validate u shape
        if u.dim() != 3 or u.shape[0] != batch_size or u.shape[1] != n_act or u.shape[2] != 3:
            raise ValueError(
                f"Batched u must have shape [{batch_size}, {n_act}, 3], got {u.shape}"
            )
    else:
        if x.shape[0] != expected_state_dim:
            raise ValueError(
                f"x dimension must be {expected_state_dim} for n_act={n_act}, "
                f"got {x.shape[0]}"
            )

        # Validate u shape
        if u.dim() != 2 or u.shape[0] != n_act or u.shape[1] != 3:
            raise ValueError(
                f"Unbatched u must have shape [{n_act}, 3], got {u.shape}"
            )

    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    # Get catheter parameters
    if catheter_params is None:
        raise ValueError(
            "catheter_params must be provided. "
            "Use load_default_catheter_params(cath_params_file, cath_config_file) to load defaults."
        )

    # Add required L_inserted if not present
    if 'L_inserted' not in catheter_params:
        # Use a default value (will come from CathConfig if not provided)
        catheter_params = dict(catheter_params)  # Make a copy
        catheter_params['L_inserted'] = 100.0  # Default insertion length in mm

    # Process unbatched or batched
    if is_batched:
        # Process each batch element separately
        x_next_list = []
        obs_list = {
            'tip_p': [],
            'tip_R': [] if return_orientation else None,
            'tip_u': [],
            'converged': [],
        }
        warmstart_next_list = {'mL_guess': [], 'nL_guess': []}

        for b in range(batch_size):
            x_b = x[b]
            u_b = u[b]
            warmstart_b = None
            if warmstart is not None and 'mL_guess' in warmstart and 'nL_guess' in warmstart:
                warmstart_b = {
                    'mL_guess': warmstart['mL_guess'][b],
                    'nL_guess': warmstart['nL_guess'][b],
                }

            # Call single-instance wrapper
            x_next_b, obs_b = _true_legacy_step_single(
                x_b, u_b, dt, n_act, catheter_params, warmstart_b, return_orientation
            )

            x_next_list.append(x_next_b)
            obs_list['tip_p'].append(obs_b['tip_p'])
            if return_orientation:
                obs_list['tip_R'].append(obs_b['tip_R'])
            obs_list['tip_u'].append(obs_b['tip_u'])
            obs_list['converged'].append(obs_b['converged'])
            warmstart_next_list['mL_guess'].append(obs_b['warmstart_next']['mL_guess'])
            warmstart_next_list['nL_guess'].append(obs_b['warmstart_next']['nL_guess'])

        # Stack results
        x_next = torch.stack(x_next_list)
        obs = {
            'tip_p': torch.stack(obs_list['tip_p']),
            'tip_u': torch.stack(obs_list['tip_u']),
            'converged': torch.tensor(obs_list['converged'], dtype=torch.bool, device=original_device),
            'warmstart_next': {
                'mL_guess': torch.stack(warmstart_next_list['mL_guess']),
                'nL_guess': torch.stack(warmstart_next_list['nL_guess']),
            }
        }
        if return_orientation:
            obs['tip_R'] = torch.stack(obs_list['tip_R'])

    else:
        # Unbatched case
        x_next, obs = _true_legacy_step_single(
            x, u, dt, n_act, catheter_params, warmstart, return_orientation
        )

    return x_next, obs


def _true_legacy_step_single(
    x: torch.Tensor,
    u: torch.Tensor,
    dt: float,
    n_act: int,
    catheter_params: Dict,
    warmstart: Optional[Dict],
    return_orientation: bool,
) -> Tuple[torch.Tensor, Dict]:
    """
    Single-instance TRUE legacy step (unbatched).

    This calls the C++ binding crm_diff_py.true_legacy_step_forward.
    """
    is_torch_input = isinstance(x, torch.Tensor)
    if is_torch_input:
        original_dtype = x.dtype
        original_device = x.device
    else:
        original_dtype = x.dtype
        original_device = None

    # Unpack TRUE legacy state
    x_coil, xf = unpack_true_legacy_state(x, n_act)
    # x_coil: [n_act, 18]
    # xf: [15]

    # Convert to numpy for C++ binding (ensure float64 and contiguous)
    if is_torch_input:
        x_coil_np = x_coil.detach().cpu().numpy().astype(np.float64, order='C', copy=False)
        xf_np = xf.detach().cpu().numpy().astype(np.float64, order='C', copy=False)
        u_np = u.detach().cpu().numpy().astype(np.float64, order='C', copy=False)
    else:
        x_coil_np = np.asarray(x_coil, dtype=np.float64, order='C')
        xf_np = np.asarray(xf, dtype=np.float64, order='C')
        u_np = np.asarray(u, dtype=np.float64, order='C')

    # Extract warm-start if provided
    mL_guess_np = None
    nL_guess_np = None
    if warmstart is not None and 'mL_guess' in warmstart and 'nL_guess' in warmstart:
        mL_guess = warmstart['mL_guess']
        nL_guess = warmstart['nL_guess']

        # Validate shapes
        if mL_guess.shape != (n_act, 3):
            raise ValueError(
                f"mL_guess must have shape [{n_act}, 3], got {mL_guess.shape}"
            )
        if nL_guess.shape != (n_act, 3):
            raise ValueError(
                f"nL_guess must have shape [{n_act}, 3], got {nL_guess.shape}"
            )

        if is_torch_input:
            mL_guess_np = mL_guess.detach().cpu().numpy().astype(np.float64, order='C', copy=False)
            nL_guess_np = nL_guess.detach().cpu().numpy().astype(np.float64, order='C', copy=False)
        else:
            mL_guess_np = np.asarray(mL_guess, dtype=np.float64, order='C')
            nL_guess_np = np.asarray(nL_guess, dtype=np.float64, order='C')

    # Call C++ binding
    result = crm_diff_py.true_legacy_step_forward(
        x_coil_np, xf_np, u_np, dt, catheter_params, mL_guess_np, nL_guess_np
    )

    # Extract results
    x_coil_next_np = result['x_coil_next']  # [n_act, 18]
    xf_next_np = result['xf_next']          # [15]
    tip_p_np = result['tip_p']              # [3]
    tip_R_np = result['tip_R']              # [9]
    tip_u_np = result['tip_u']              # [3]
    mL_next_np = result['mL_next']          # [n_act, 3]
    nL_next_np = result['nL_next']          # [n_act, 3]
    converged = result['converged']         # bool

    # Convert back to torch tensors or keep as numpy
    if is_torch_input:
        x_coil_next = torch.from_numpy(x_coil_next_np).to(dtype=original_dtype, device=original_device)
        xf_next = torch.from_numpy(xf_next_np).to(dtype=original_dtype, device=original_device)
    else:
        x_coil_next = x_coil_next_np.astype(original_dtype, copy=False)
        xf_next = xf_next_np.astype(original_dtype, copy=False)

    # Pack next state
    x_next = pack_true_legacy_state(x_coil_next, xf_next)

    # Package observables
    if is_torch_input:
        obs = {
            'tip_p': torch.from_numpy(tip_p_np).to(dtype=original_dtype, device=original_device),
            'tip_u': torch.from_numpy(tip_u_np).to(dtype=original_dtype, device=original_device),
            'converged': converged,
            'warmstart_next': {
                'mL_guess': torch.from_numpy(mL_next_np).to(dtype=original_dtype, device=original_device),
                'nL_guess': torch.from_numpy(nL_next_np).to(dtype=original_dtype, device=original_device),
            }
        }
    else:
        obs = {
            'tip_p': tip_p_np.astype(original_dtype, copy=False),
            'tip_u': tip_u_np.astype(original_dtype, copy=False),
            'converged': converged,
            'warmstart_next': {
                'mL_guess': mL_next_np.astype(original_dtype, copy=False),
                'nL_guess': nL_next_np.astype(original_dtype, copy=False),
            }
        }

    if return_orientation:
        if is_torch_input:
            obs['tip_R'] = torch.from_numpy(tip_R_np).to(dtype=original_dtype, device=original_device)
        else:
            obs['tip_R'] = tip_R_np.astype(original_dtype, copy=False)

    return x_next, obs


def true_legacy_linearize(
    x: np.ndarray,
    u: np.ndarray,
    dt: float,
    *,
    n_act: int,
    catheter_params: Dict,
    L_inserted: float = 100.0,
    method: str = "implicit",  # "implicit", "torch_autograd", or "finite_diff"
    eps: float = 1e-7
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute linearization Jacobians A, B for TRUE legacy dynamics.

    A = ∂x_next/∂x  (state Jacobian)
    B = ∂x_next/∂u  (control Jacobian)

    Args:
        x: State [18*n_act + 15]
        u: Control [n_act, 3]
        dt: Timestep
        n_act: Number of actuators
        catheter_params: Physics parameters
        L_inserted: Insertion depth (mm)
        method: "implicit" (default, analytic via IFT), "torch_autograd", or "finite_diff"
        eps: Finite difference epsilon (if method="finite_diff")

    Returns:
        A: State Jacobian [state_dim, state_dim]
        B: Control Jacobian [state_dim, n_act*3]
    """
    state_dim = 18 * n_act + 15
    control_dim = n_act * 3

    if method == "implicit":
        return _linearize_implicit(x, u, dt, n_act, catheter_params, L_inserted)
    elif method == "torch_autograd":
        return _linearize_autograd(x, u, dt, n_act, catheter_params, L_inserted)
    elif method == "finite_diff":
        return _linearize_finite_diff(x, u, dt, n_act, catheter_params, L_inserted, eps)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'implicit', 'torch_autograd', or 'finite_diff'")


def _linearize_implicit(x, u, dt, n_act, catheter_params, L_inserted):
    """Compute Jacobians using C++ implicit function theorem implementation."""
    # Unpack state
    x_coil, xf = unpack_true_legacy_state(x, n_act)

    # Ensure numpy arrays with correct dtype and layout
    x_coil_np = np.ascontiguousarray(x_coil, dtype=np.float64)
    xf_np = np.ascontiguousarray(xf, dtype=np.float64)
    u_np = np.ascontiguousarray(u, dtype=np.float64)

    # Update catheter_params with L_inserted
    params_dict = catheter_params.copy()
    params_dict['L_inserted'] = L_inserted

    # Call C++ binding
    result = crm_diff_py.true_legacy_linearize(
        x_coil_np, xf_np, u_np, dt, params_dict
    )

    A = result['A']  # [state_dim, state_dim]
    B = result['B']  # [state_dim, control_dim]

    # Flatten B to match expected shape [state_dim, n_act*3]
    B_flat = B.reshape(B.shape[0], -1)

    return A, B_flat


def _linearize_autograd(x, u, dt, n_act, catheter_params, L_inserted):
    """Compute Jacobians using PyTorch autograd."""
    from .true_legacy_step_autograd import true_legacy_step_torch

    # Convert to torch
    x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    u_torch_flat = torch.tensor(u.flatten(), dtype=torch.float64, requires_grad=True)
    u_reshaped = u_torch_flat.reshape(n_act, 3)

    # A = ∂x_next/∂x (state Jacobian)
    def dynamics_x(x_in):
        _, x_next = true_legacy_step_torch(
            x_in, u_reshaped, dt, n_act=n_act,
            catheter_params=catheter_params,
            L_inserted=L_inserted,
            return_x_next=True
        )
        return x_next

    A = torch.autograd.functional.jacobian(dynamics_x, x_torch).detach().numpy()

    # B = ∂x_next/∂u (control Jacobian)
    def dynamics_u(u_flat):
        u_in = u_flat.reshape(n_act, 3)
        _, x_next = true_legacy_step_torch(
            x_torch, u_in, dt, n_act=n_act,
            catheter_params=catheter_params,
            L_inserted=L_inserted,
            return_x_next=True
        )
        return x_next

    B_flat = torch.autograd.functional.jacobian(dynamics_u, u_torch_flat).detach().numpy()

    return A, B_flat


def _linearize_finite_diff(x, u, dt, n_act, catheter_params, L_inserted, eps):
    """Compute Jacobians using finite differences."""
    from .true_legacy_step_autograd import true_legacy_step_torch

    state_dim = 18 * n_act + 15
    control_dim = n_act * 3

    # Nominal forward
    x_torch = torch.from_numpy(x.astype(np.float64))
    u_torch = torch.from_numpy(u.astype(np.float64))
    _, x_next_nom = true_legacy_step_torch(
        x_torch, u_torch, dt, n_act=n_act,
        catheter_params=catheter_params,
        L_inserted=L_inserted,
        return_x_next=True
    )
    x_next_nom = x_next_nom.detach().numpy()

    # A = ∂x_next/∂x via forward differences
    A = np.zeros((state_dim, state_dim))
    for i in range(state_dim):
        x_pert = x.copy()
        x_pert[i] += eps
        _, x_next_pert = true_legacy_step_torch(
            torch.from_numpy(x_pert.astype(np.float64)), u_torch, dt,
            n_act=n_act, catheter_params=catheter_params,
            L_inserted=L_inserted, return_x_next=True
        )
        A[:, i] = (x_next_pert.detach().numpy() - x_next_nom) / eps

    # B = ∂x_next/∂u (u is [n_act, 3])
    B = np.zeros((state_dim, control_dim))
    u_flat = u.flatten()
    for j in range(control_dim):
        u_pert_flat = u_flat.copy()
        u_pert_flat[j] += eps
        u_pert = u_pert_flat.reshape(n_act, 3)
        _, x_next_pert = true_legacy_step_torch(
            x_torch, torch.from_numpy(u_pert.astype(np.float64)), dt,
            n_act=n_act, catheter_params=catheter_params,
            L_inserted=L_inserted, return_x_next=True
        )
        B[:, j] = (x_next_pert.detach().numpy() - x_next_nom) / eps

    return A, B


def true_legacy_tip_jacobian(
    x: np.ndarray,
    n_act: int,
    method: str = "analytic"  # or "autograd"
) -> np.ndarray:
    """
    Compute ∂p_tip/∂x for TRUE legacy state.

    The tip position p_tip is the first 3 elements of the tip state xf,
    which is the last 15 elements of the packed state x.

    Args:
        x: State [18*n_act + 15]
        n_act: Number of actuators
        method: "analytic" (extract from state) or "autograd"

    Returns:
        J_p: [3, 18*n_act+15] Jacobian of tip position w.r.t. state
    """
    state_dim = 18 * n_act + 15

    if method == "analytic":
        # p_tip is first 3 elements of xf (last 15 elements of x)
        J_p = np.zeros((3, state_dim))
        tip_state_start = 18 * n_act  # Tip state starts here
        J_p[0, tip_state_start + 0] = 1.0  # p_x
        J_p[1, tip_state_start + 1] = 1.0  # p_y
        J_p[2, tip_state_start + 2] = 1.0  # p_z
        return J_p

    elif method == "autograd":
        import torch
        x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)

        # Extract tip position (first 3 of last 15)
        tip_state_start = 18 * n_act
        p_tip = x_torch[tip_state_start:tip_state_start+3]

        # Compute Jacobian
        J_p = torch.autograd.functional.jacobian(
            lambda x_in: x_in[tip_state_start:tip_state_start+3],
            x_torch
        ).detach().numpy()

        return J_p

    else:
        raise ValueError(f"Unknown method: {method}. Use 'analytic' or 'autograd'")
