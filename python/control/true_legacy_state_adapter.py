"""
TRUE Legacy State Adapter (A0 Frozen Contract)

Implements pack/unpack for the persisted timestep state in TRUE legacy dynamics.

Core state dimension: 18·N + 15 where N = NUM_ACT_SET (number of coils/actuator sets)
- Coil states: v[3], w[3], p[3], R[9 row-major] for each coil (18 per coil)
- Tip state: p_tip[3], R_tip[9 row-major], u_tip[3] (15 total)

Warm-start guesses (optional, separate): 6·N
- Per coil: mL[3], nL[3]

This is the TRUE legacy contract (no reduction, no quaternions).
BVP unknowns are algebraic, NOT part of core state unless explicitly stored.
"""

import torch

# Constants
COIL_STATE_DIM = 18  # v[3], w[3], p[3], R[9]
TIP_STATE_DIM = 15   # p_tip[3], R_tip[9], u_tip[3]


def true_legacy_state_dim(n_act: int) -> int:
    """
    Compute the TRUE legacy state dimension.

    Args:
        n_act: Number of actuator sets (coils)

    Returns:
        State dimension: 18·N + 15
    """
    return COIL_STATE_DIM * n_act + TIP_STATE_DIM


def true_legacy_warmstart_dim(n_act: int) -> int:
    """
    Compute the TRUE legacy warm-start dimension (optional).

    Args:
        n_act: Number of actuator sets (coils)

    Returns:
        Warm-start dimension: 6·N (mL[3] + nL[3] per coil)
    """
    return 6 * n_act


def pack_true_legacy_state(
    x_coil: torch.Tensor,
    xf: torch.Tensor
) -> torch.Tensor:
    """
    Pack TRUE legacy state into a flat vector.

    Packing order:
    1. For each coil j=0..N-1: v[3], w[3], p[3], R[9 row-major] (18 elements)
    2. Tip state: p_tip[3], R_tip[9 row-major], u_tip[3] (15 elements)

    Args:
        x_coil: Coil states, shape [N, 18] (unbatched) or [B, N, 18] (batched)
        xf: Tip state, shape [15] (unbatched) or [B, 15] (batched)

    Returns:
        Packed state, shape [18·N + 15] (unbatched) or [B, 18·N + 15] (batched)

    Raises:
        ValueError: If input shapes are invalid or inconsistent
    """
    # Validate inputs
    if x_coil.dim() not in [2, 3]:
        raise ValueError(
            f"x_coil must be 2D [N, 18] or 3D [B, N, 18], got shape {x_coil.shape}"
        )
    if xf.dim() not in [1, 2]:
        raise ValueError(
            f"xf must be 1D [15] or 2D [B, 15], got shape {xf.shape}"
        )

    # Check if batched
    is_batched = (x_coil.dim() == 3)

    if is_batched:
        # Batched case: x_coil [B, N, 18], xf [B, 15]
        if xf.dim() != 2:
            raise ValueError(
                f"Batched x_coil requires batched xf, got x_coil shape {x_coil.shape}, xf shape {xf.shape}"
            )
        batch_size = x_coil.shape[0]
        n_act = x_coil.shape[1]

        if x_coil.shape[2] != COIL_STATE_DIM:
            raise ValueError(
                f"x_coil last dimension must be {COIL_STATE_DIM}, got {x_coil.shape[2]}"
            )
        if xf.shape[0] != batch_size:
            raise ValueError(
                f"Batch size mismatch: x_coil has {batch_size}, xf has {xf.shape[0]}"
            )
        if xf.shape[1] != TIP_STATE_DIM:
            raise ValueError(
                f"xf last dimension must be {TIP_STATE_DIM}, got {xf.shape[1]}"
            )

        # Flatten coil states: [B, N, 18] -> [B, N*18]
        x_coil_flat = x_coil.reshape(batch_size, -1)

        # Concatenate: [B, N*18 + 15]
        packed = torch.cat([x_coil_flat, xf], dim=1)

    else:
        # Unbatched case: x_coil [N, 18], xf [15]
        if xf.dim() != 1:
            raise ValueError(
                f"Unbatched x_coil requires unbatched xf, got x_coil shape {x_coil.shape}, xf shape {xf.shape}"
            )
        n_act = x_coil.shape[0]

        if x_coil.shape[1] != COIL_STATE_DIM:
            raise ValueError(
                f"x_coil last dimension must be {COIL_STATE_DIM}, got {x_coil.shape[1]}"
            )
        if xf.shape[0] != TIP_STATE_DIM:
            raise ValueError(
                f"xf dimension must be {TIP_STATE_DIM}, got {xf.shape[0]}"
            )

        # Flatten coil states: [N, 18] -> [N*18]
        x_coil_flat = x_coil.reshape(-1)

        # Concatenate: [N*18 + 15]
        packed = torch.cat([x_coil_flat, xf], dim=0)

    return packed.contiguous()


def unpack_true_legacy_state(
    x: torch.Tensor,
    n_act: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unpack TRUE legacy state from a flat vector.

    Args:
        x: Packed state, shape [18·N + 15] (unbatched) or [B, 18·N + 15] (batched)
        n_act: Number of actuator sets (coils)

    Returns:
        Tuple (x_coil, xf) where:
        - x_coil: Coil states, shape [N, 18] (unbatched) or [B, N, 18] (batched)
        - xf: Tip state, shape [15] (unbatched) or [B, 15] (batched)

    Raises:
        ValueError: If input shape is invalid for the given n_act
    """
    expected_dim = true_legacy_state_dim(n_act)

    if x.dim() not in [1, 2]:
        raise ValueError(
            f"x must be 1D [state_dim] or 2D [B, state_dim], got shape {x.shape}"
        )

    is_batched = (x.dim() == 2)

    if is_batched:
        # Batched case: x [B, 18·N + 15]
        batch_size = x.shape[0]
        if x.shape[1] != expected_dim:
            raise ValueError(
                f"Batched x last dimension must be {expected_dim} for n_act={n_act}, got {x.shape[1]}"
            )

        coil_dim = COIL_STATE_DIM * n_act

        # Split: [B, N*18] and [B, 15]
        x_coil_flat = x[:, :coil_dim]
        xf = x[:, coil_dim:]

        # Reshape coil states: [B, N*18] -> [B, N, 18]
        x_coil = x_coil_flat.reshape(batch_size, n_act, COIL_STATE_DIM)

    else:
        # Unbatched case: x [18·N + 15]
        if x.shape[0] != expected_dim:
            raise ValueError(
                f"x dimension must be {expected_dim} for n_act={n_act}, got {x.shape[0]}"
            )

        coil_dim = COIL_STATE_DIM * n_act

        # Split: [N*18] and [15]
        x_coil_flat = x[:coil_dim]
        xf = x[coil_dim:]

        # Reshape coil states: [N*18] -> [N, 18]
        x_coil = x_coil_flat.reshape(n_act, COIL_STATE_DIM)

    return x_coil, xf


def pack_true_legacy_warmstart(
    mL_guess: torch.Tensor,
    nL_guess: torch.Tensor
) -> torch.Tensor:
    """
    Pack TRUE legacy warm-start guesses into a flat vector.

    Packing order: For each coil j=0..N-1: mL[3], nL[3]

    Args:
        mL_guess: Moment guesses, shape [N, 3] (unbatched) or [B, N, 3] (batched)
        nL_guess: Force guesses, shape [N, 3] (unbatched) or [B, N, 3] (batched)

    Returns:
        Packed warm-start, shape [6·N] (unbatched) or [B, 6·N] (batched)

    Raises:
        ValueError: If input shapes are invalid or inconsistent
    """
    # Validate inputs
    if mL_guess.dim() not in [2, 3]:
        raise ValueError(
            f"mL_guess must be 2D [N, 3] or 3D [B, N, 3], got shape {mL_guess.shape}"
        )
    if nL_guess.dim() not in [2, 3]:
        raise ValueError(
            f"nL_guess must be 2D [N, 3] or 3D [B, N, 3], got shape {nL_guess.shape}"
        )

    # Check if batched
    is_batched = (mL_guess.dim() == 3)

    if is_batched:
        # Batched case: [B, N, 3]
        if nL_guess.dim() != 3:
            raise ValueError(
                f"Batched mL_guess requires batched nL_guess, got shapes {mL_guess.shape}, {nL_guess.shape}"
            )
        batch_size = mL_guess.shape[0]
        n_act = mL_guess.shape[1]

        if mL_guess.shape[2] != 3:
            raise ValueError(
                f"mL_guess last dimension must be 3, got {mL_guess.shape[2]}"
            )
        if nL_guess.shape[0] != batch_size or nL_guess.shape[1] != n_act:
            raise ValueError(
                f"Shape mismatch: mL_guess {mL_guess.shape}, nL_guess {nL_guess.shape}"
            )
        if nL_guess.shape[2] != 3:
            raise ValueError(
                f"nL_guess last dimension must be 3, got {nL_guess.shape[2]}"
            )

        # Stack per coil: [B, N, 6]
        warmstart_per_coil = torch.cat([mL_guess, nL_guess], dim=2)

        # Flatten: [B, N*6]
        packed = warmstart_per_coil.reshape(batch_size, -1)

    else:
        # Unbatched case: [N, 3]
        if nL_guess.dim() != 2:
            raise ValueError(
                f"Unbatched mL_guess requires unbatched nL_guess, got shapes {mL_guess.shape}, {nL_guess.shape}"
            )
        n_act = mL_guess.shape[0]

        if mL_guess.shape[1] != 3:
            raise ValueError(
                f"mL_guess last dimension must be 3, got {mL_guess.shape[1]}"
            )
        if nL_guess.shape[0] != n_act:
            raise ValueError(
                f"Shape mismatch: mL_guess {mL_guess.shape}, nL_guess {nL_guess.shape}"
            )
        if nL_guess.shape[1] != 3:
            raise ValueError(
                f"nL_guess last dimension must be 3, got {nL_guess.shape[1]}"
            )

        # Stack per coil: [N, 6]
        warmstart_per_coil = torch.cat([mL_guess, nL_guess], dim=1)

        # Flatten: [N*6]
        packed = warmstart_per_coil.reshape(-1)

    return packed.contiguous()


def unpack_true_legacy_warmstart(
    w: torch.Tensor,
    n_act: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Unpack TRUE legacy warm-start guesses from a flat vector.

    Args:
        w: Packed warm-start, shape [6·N] (unbatched) or [B, 6·N] (batched)
        n_act: Number of actuator sets (coils)

    Returns:
        Tuple (mL_guess, nL_guess) where:
        - mL_guess: Moment guesses, shape [N, 3] (unbatched) or [B, N, 3] (batched)
        - nL_guess: Force guesses, shape [N, 3] (unbatched) or [B, N, 3] (batched)

    Raises:
        ValueError: If input shape is invalid for the given n_act
    """
    expected_dim = true_legacy_warmstart_dim(n_act)

    if w.dim() not in [1, 2]:
        raise ValueError(
            f"w must be 1D [warmstart_dim] or 2D [B, warmstart_dim], got shape {w.shape}"
        )

    is_batched = (w.dim() == 2)

    if is_batched:
        # Batched case: w [B, 6·N]
        batch_size = w.shape[0]
        if w.shape[1] != expected_dim:
            raise ValueError(
                f"Batched w last dimension must be {expected_dim} for n_act={n_act}, got {w.shape[1]}"
            )

        # Reshape: [B, 6*N] -> [B, N, 6]
        w_reshaped = w.reshape(batch_size, n_act, 6)

        # Split: [B, N, 3] each
        mL_guess = w_reshaped[:, :, :3]
        nL_guess = w_reshaped[:, :, 3:]

    else:
        # Unbatched case: w [6·N]
        if w.shape[0] != expected_dim:
            raise ValueError(
                f"w dimension must be {expected_dim} for n_act={n_act}, got {w.shape[0]}"
            )

        # Reshape: [6*N] -> [N, 6]
        w_reshaped = w.reshape(n_act, 6)

        # Split: [N, 3] each
        mL_guess = w_reshaped[:, :3]
        nL_guess = w_reshaped[:, 3:]

    return mL_guess, nL_guess
