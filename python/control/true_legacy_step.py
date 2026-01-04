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
    # Store original dtype and device
    original_dtype = x.dtype
    original_device = x.device

    # Validate inputs
    if x.dim() not in [1, 2]:
        raise ValueError(
            f"x must be 1D [state_dim] or 2D [B, state_dim], got shape {x.shape}"
        )

    is_batched = (x.dim() == 2)
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
    original_dtype = x.dtype
    original_device = x.device

    # Unpack TRUE legacy state
    x_coil, xf = unpack_true_legacy_state(x, n_act)
    # x_coil: [n_act, 18]
    # xf: [15]

    # Convert to numpy for C++ binding (ensure float64 and contiguous)
    x_coil_np = x_coil.detach().cpu().numpy().astype(np.float64, order='C', copy=False)
    xf_np = xf.detach().cpu().numpy().astype(np.float64, order='C', copy=False)
    u_np = u.detach().cpu().numpy().astype(np.float64, order='C', copy=False)

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

        mL_guess_np = mL_guess.detach().cpu().numpy().astype(np.float64, order='C', copy=False)
        nL_guess_np = nL_guess.detach().cpu().numpy().astype(np.float64, order='C', copy=False)

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

    # Convert back to torch tensors
    x_coil_next = torch.from_numpy(x_coil_next_np).to(dtype=original_dtype, device=original_device)
    xf_next = torch.from_numpy(xf_next_np).to(dtype=original_dtype, device=original_device)

    # Pack next state
    x_next = pack_true_legacy_state(x_coil_next, xf_next)

    # Package observables
    obs = {
        'tip_p': torch.from_numpy(tip_p_np).to(dtype=original_dtype, device=original_device),
        'tip_u': torch.from_numpy(tip_u_np).to(dtype=original_dtype, device=original_device),
        'converged': converged,
        'warmstart_next': {
            'mL_guess': torch.from_numpy(mL_next_np).to(dtype=original_dtype, device=original_device),
            'nL_guess': torch.from_numpy(nL_next_np).to(dtype=original_dtype, device=original_device),
        }
    }

    if return_orientation:
        obs['tip_R'] = torch.from_numpy(tip_R_np).to(dtype=original_dtype, device=original_device)

    return x_next, obs
