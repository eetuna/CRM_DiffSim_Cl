"""
Legacy Contract Step Function (Forward + Backward/VJP)

Wraps crm_diff_py.dynamics_forward and dynamics_backward with contract-exact I/O:
  - Forward: LegacyState (6D) → LegacyState (6D) + observables
  - Backward: Implicit VJP using cached Jacobians (NO finite differences)

Evidence:
  - Forward: src/CRM_DiffDynamics.hpp:48-55 (dynamics_forward)
  - Backward: src/CRM_DiffDynamics.hpp:59-67 (dynamics_backward)

Scope (A1 + A2):
  - Forward pass with cached result for backward
  - Implicit VJP via dynamics_backward
  - PyTorch autograd integration
  - NO theta parameter (not in legacy contract)
  - NO physics changes
"""

import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import crm_diff_py
from .legacy_state import LegacyState
from .legacy_state_adapter import legacy_state_to_numpy, numpy_to_legacy_state


@dataclass
class LegacyStepResult:
    """
    Result of legacy dynamics forward step.

    Attributes:
        x_next: Next state (LegacyState, 6D)
        observables: Dict containing:
            - p_tip: (3,) tip position in mm
            - u_tip: (3,) tip curvature in 1/mm
        diagnostics: Dict containing solver status:
            - status: int (0=success, non-zero=failure)
            - converged: bool (True if successful)
            - exit_code: int (0=OK, 1=rank-deficient, 2=residual large)
            - lu_rank: int (expected 6)
            - solve_residual: float (should be < 1e-10)
        _fwd_cache: Private cached forward result for VJP (do not access directly)

    Evidence: src/CRM_DiffDynamics.hpp:9-44 (DynamicsStepResult struct)
    """
    x_next: LegacyState
    observables: Dict[str, np.ndarray]
    diagnostics: Dict[str, Any]
    _fwd_cache: Optional[Dict[str, Any]] = field(default=None, repr=False)

    @property
    def success(self) -> bool:
        """True if step succeeded (status == 0)."""
        return self.diagnostics.get('status', -1) == 0


@dataclass
class LegacyVJPResult:
    """
    Result of legacy dynamics VJP (backward pass).

    Attributes:
        grad_x_t: Gradient w.r.t. input state (6,) float64
        grad_u_t: Gradient w.r.t. control (3,) float64
        diagnostics: Dict containing:
            - status: int (0=success, 1=rank-deficient, 2=residual large, 3=equilibrium failed)
            - lu_rank: int (expected 6)
            - rel_residual: float (should be < 1e-10)

    Evidence: src/CRM_DiffDynamics.hpp:59-67 (dynamics_backward)
    """
    grad_x_t: np.ndarray   # (6,) float64
    grad_u_t: np.ndarray   # (3,) float64
    diagnostics: Dict[str, Any]

    @property
    def success(self) -> bool:
        """True if backward pass succeeded (status == 0)."""
        return self.diagnostics.get('status', -1) == 0


@dataclass
class LegacyBatchedVJPResult:
    """
    Result of batched legacy dynamics VJP (backward pass).

    Processes K adjoint vectors in a single call, amortizing cost for
    trajectory optimization (iLQR / MPC).

    Attributes:
        grad_x_t: Gradients w.r.t. input state (K, 6) float64
        grad_u_t: Gradients w.r.t. control (K, 3) float64
        diagnostics: Dict containing:
            - status: int (0=success, 1=rank-deficient, 2=residual large, 3=eq failed)
            - lu_rank: int (expected 6)
            - rel_residual: float (should be < 1e-10)
            - K: int (number of adjoint vectors processed)

    Evidence: src/CRM_DiffDynamics.hpp:75-84 (dynamics_backward_batched)
    """
    grad_x_t: np.ndarray   # (K, 6) float64
    grad_u_t: np.ndarray   # (K, 3) float64
    diagnostics: Dict[str, Any]

    @property
    def success(self) -> bool:
        """True if backward pass succeeded (status == 0)."""
        return self.diagnostics.get('status', -1) == 0


def step_legacy_contract(
    x_t: LegacyState,
    u_t: np.ndarray,
    dt: float,
    L_inserted: float,
    params: Dict[str, Any],
) -> LegacyStepResult:
    """
    Execute one legacy dynamics timestep (forward pass).

    This is a thin wrapper around crm_diff_py.dynamics_forward that
    enforces the legacy contract: 6D state in/out, observables separate.
    The full forward result is cached for use in vjp_legacy_contract().

    Args:
        x_t: Current state (LegacyState, 6D)
        u_t: Control input (3,) - actuation currents in Amperes
        dt: Time step in seconds (must be positive)
        L_inserted: Insertion length in mm (must be non-negative)
        params: Catheter parameters dict (from load_default_catheter_params)

    Returns:
        LegacyStepResult containing:
            - x_next: Next state (LegacyState)
            - observables: {p_tip, u_tip}
            - diagnostics: {status, converged, exit_code, lu_rank, solve_residual}
            - _fwd_cache: Cached forward result for VJP

    Raises:
        TypeError: If x_t is not LegacyState
        ValueError: If inputs have wrong shape/dtype

    Example:
        >>> state = LegacyState.zeros()
        >>> u_t = np.zeros(3)
        >>> params = load_default_catheter_params(param_file, config_file)
        >>> result = step_legacy_contract(state, u_t, 0.01, 100.0, params)
        >>> if result.success:
        ...     print(f"p_tip = {result.observables['p_tip']}")

    Evidence: src/CRM_DiffDynamics.hpp:48-55
    """
    # Validate input state
    if not isinstance(x_t, LegacyState):
        raise TypeError(f"x_t must be LegacyState, got {type(x_t)}")

    # Validate control
    u_t = np.asarray(u_t, dtype=np.float64)
    if u_t.shape != (3,):
        raise ValueError(f"u_t must be (3,), got {u_t.shape}")
    if not np.all(np.isfinite(u_t)):
        raise ValueError("u_t contains non-finite values (NaN or Inf)")

    # Validate scalars
    if not (isinstance(dt, (int, float)) and dt > 0):
        raise ValueError(f"dt must be positive scalar, got {dt}")
    if not (isinstance(L_inserted, (int, float)) and L_inserted >= 0):
        raise ValueError(f"L_inserted must be non-negative, got {L_inserted}")

    # Convert state to numpy (6,) array
    x_t_np = legacy_state_to_numpy(x_t)

    # Ensure contiguous arrays for C++ binding
    x_t_np = np.ascontiguousarray(x_t_np, dtype=np.float64)
    u_t_np = np.ascontiguousarray(u_t, dtype=np.float64)

    # Call C++ binding
    # Evidence: python/crm_bindings.cpp:721-723
    result = crm_diff_py.dynamics_forward(
        x_t_np, u_t_np, float(dt), float(L_inserted), params
    )

    # Extract outputs
    status = result['status']
    x_next_np = result['x_next'].copy()
    p_tip = result['p_tip'].copy()
    u_tip = result['u_tip'].copy()

    # Convert state back to LegacyState
    x_next = numpy_to_legacy_state(x_next_np)

    # Build observables dict (these are NOT part of the state)
    observables = {
        'p_tip': p_tip,   # Tip position (3,) mm
        'u_tip': u_tip,   # Tip curvature (3,) 1/mm
    }

    # Build diagnostics dict
    diagnostics = {
        'status': status,
        'converged': result.get('converged', status == 0),
        'exit_code': result.get('exit_code', -1),
        'lu_rank': result.get('lu_rank', -1),
        'solve_residual': result.get('solve_residual', float('nan')),
    }

    return LegacyStepResult(
        x_next=x_next,
        observables=observables,
        diagnostics=diagnostics,
        _fwd_cache=result,  # Cache full forward result for VJP
    )


def vjp_legacy_contract(
    fwd_result: LegacyStepResult,
    grad_x_next: np.ndarray,
    params: Dict[str, Any],
) -> LegacyVJPResult:
    """
    Compute VJP (backward pass) using implicit differentiation.

    Uses cached Jacobians from forward pass. NO finite differences.

    Args:
        fwd_result: Cached forward result from step_legacy_contract()
        grad_x_next: Upstream gradient (6,) - ∂L/∂x_{t+1}
        params: Same params dict used in forward

    Returns:
        LegacyVJPResult with:
            - grad_x_t: (6,) ∂L/∂x_t
            - grad_u_t: (3,) ∂L/∂u_t
            - diagnostics: {status, lu_rank, rel_residual}

    Raises:
        ValueError: If fwd_result has no cached forward data
        ValueError: If grad_x_next has wrong shape

    Example:
        >>> fwd_result = step_legacy_contract(state, u_t, dt, L, params)
        >>> grad_x_next = np.array([1., 0., 0., 0., 0., 0.])
        >>> vjp_result = vjp_legacy_contract(fwd_result, grad_x_next, params)
        >>> print(f"grad_x_t = {vjp_result.grad_x_t}")

    Evidence: src/CRM_DiffDynamics.hpp:59-67
    """
    # Validate cached forward result
    if fwd_result._fwd_cache is None:
        raise ValueError(
            "Forward result has no cached data for backward pass. "
            "Ensure step_legacy_contract() was called with caching enabled."
        )

    # Validate upstream gradient
    grad_x_next = np.asarray(grad_x_next, dtype=np.float64)
    if grad_x_next.shape != (6,):
        raise ValueError(f"grad_x_next must be (6,), got {grad_x_next.shape}")
    if not np.all(np.isfinite(grad_x_next)):
        raise ValueError("grad_x_next contains non-finite values (NaN or Inf)")

    grad_x_next = np.ascontiguousarray(grad_x_next, dtype=np.float64)

    # Call C++ implicit backward
    # Evidence: python/crm_bindings.cpp:327-456
    bwd_result = crm_diff_py.dynamics_backward(
        fwd_result._fwd_cache,
        grad_x_next,
        params
    )

    # Build diagnostics
    diagnostics = {
        'status': bwd_result['status'],
        'lu_rank': bwd_result.get('lu_rank', -1),
        'rel_residual': bwd_result.get('rel_residual', float('nan')),
    }

    return LegacyVJPResult(
        grad_x_t=bwd_result['grad_x_t'].copy(),
        grad_u_t=bwd_result['grad_u_t'].copy(),
        diagnostics=diagnostics,
    )


def vjp_legacy_contract_batched(
    fwd_result: LegacyStepResult,
    V: np.ndarray,
    params: Dict[str, Any],
) -> LegacyBatchedVJPResult:
    """
    Compute batched VJP (K adjoint vectors) using implicit differentiation.

    Uses cached Jacobians from forward pass. NO finite differences.
    Processes K adjoint vectors in a single call, amortizing cost for
    trajectory optimization (iLQR / MPC).

    Args:
        fwd_result: Cached forward result from step_legacy_contract()
        V: Adjoint vectors (K, 6) - K upstream gradients ∂L/∂x_{t+1}
        params: Same params dict used in forward

    Returns:
        LegacyBatchedVJPResult with:
            - grad_x_t: (K, 6) ∂L/∂x_t for each adjoint
            - grad_u_t: (K, 3) ∂L/∂u_t for each adjoint
            - diagnostics: {status, lu_rank, rel_residual, K}

    Raises:
        ValueError: If fwd_result has no cached forward data
        ValueError: If V has wrong shape

    Example:
        >>> fwd_result = step_legacy_contract(state, u_t, dt, L, params)
        >>> V = np.random.randn(4, 6)  # 4 adjoint vectors
        >>> batched_vjp = vjp_legacy_contract_batched(fwd_result, V, params)
        >>> print(f"grad_x_t shape: {batched_vjp.grad_x_t.shape}")  # (4, 6)

    Evidence: src/CRM_DiffDynamics.hpp:75-84
    """
    # Validate cached forward result
    if fwd_result._fwd_cache is None:
        raise ValueError(
            "Forward result has no cached data for backward pass. "
            "Ensure step_legacy_contract() was called with caching enabled."
        )

    # Validate V array
    V = np.asarray(V, dtype=np.float64)
    if V.ndim != 2:
        raise ValueError(f"V must be 2D array (K, 6), got {V.ndim}D")
    if V.shape[1] != 6:
        raise ValueError(f"V must have shape (K, 6), got {V.shape}")
    if V.shape[0] < 1:
        raise ValueError("V must have at least 1 row (K >= 1)")
    if not np.all(np.isfinite(V)):
        raise ValueError("V contains non-finite values (NaN or Inf)")

    V = np.ascontiguousarray(V, dtype=np.float64)
    K = V.shape[0]

    # Call C++ batched backward
    # Evidence: python/crm_bindings.cpp:py_dynamics_backward_batched
    bwd_result = crm_diff_py.dynamics_backward_batched(
        fwd_result._fwd_cache,
        V,
        params
    )

    # Build diagnostics
    diagnostics = {
        'status': bwd_result['status'],
        'lu_rank': bwd_result.get('lu_rank', -1),
        'rel_residual': bwd_result.get('rel_residual', float('nan')),
        'K': K,
    }

    return LegacyBatchedVJPResult(
        grad_x_t=bwd_result['grad_x_t'].copy(),
        grad_u_t=bwd_result['grad_u_t'].copy(),
        diagnostics=diagnostics,
    )


def load_default_catheter_params(
    cath_params_file: str,
    cath_config_file: str,
) -> Dict[str, Any]:
    """
    Load catheter parameters from files.

    Args:
        cath_params_file: Path to catheter parameters file
                          (e.g., "./catheterdata/CatheterParameterSet_1_dyn.txt")
        cath_config_file: Path to catheter configuration file
                          (e.g., "./catheterdata/CatheterSpatialConfiguration_1.txt")

    Returns:
        params_dict ready for use with step_legacy_contract()

    Example:
        >>> params = load_default_catheter_params(
        ...     "./catheterdata/CatheterParameterSet_1_dyn.txt",
        ...     "./catheterdata/CatheterSpatialConfiguration_1.txt"
        ... )
    """
    # Load binary parameter files (returns opaque C++ pointers)
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


# ============================================================================
# PyTorch Integration (A2)
# ============================================================================

try:
    import torch

    class LegacyHybridStep(torch.autograd.Function):
        """
        PyTorch autograd wrapper for legacy 6D dynamics step.

        Forward: x_t (6,) + u_t (3,) → x_next (6,)
        Backward: Uses implicit differentiation via dynamics_backward (no FD)

        Usage:
            x_next = LegacyHybridStep.apply(x_t, u_t, dt, L_inserted, params)

        Or use the convenience wrapper:
            x_next = legacy_dynamics_step(x_t, u_t, dt, L_inserted, params)

        Note:
            Only x_t and u_t require gradients. dt, L_inserted, and params
            are treated as constants (gradients returned as None).
        """

        @staticmethod
        def forward(ctx, x_t, u_t, dt, L_inserted, params):
            """Forward pass - calls step_legacy_contract."""
            # Validate inputs are tensors
            if not isinstance(x_t, torch.Tensor):
                raise TypeError(f"x_t must be torch.Tensor, got {type(x_t)}")
            if not isinstance(u_t, torch.Tensor):
                raise TypeError(f"u_t must be torch.Tensor, got {type(u_t)}")

            # Convert to numpy
            x_t_np = np.ascontiguousarray(
                x_t.detach().cpu().numpy(), dtype=np.float64
            )
            u_t_np = np.ascontiguousarray(
                u_t.detach().cpu().numpy(), dtype=np.float64
            )

            # Create LegacyState
            state = LegacyState.from_numpy(x_t_np)

            # Forward pass
            result = step_legacy_contract(state, u_t_np, dt, L_inserted, params)

            if not result.success:
                raise RuntimeError(
                    f"step_legacy_contract failed: status={result.diagnostics['status']}, "
                    f"exit_code={result.diagnostics.get('exit_code', 'N/A')}"
                )

            # Save for backward
            ctx.save_for_backward(x_t, u_t)
            ctx.fwd_result = result
            ctx.params = params

            # Return x_next as tensor
            x_next_np = result.x_next.to_numpy()
            return torch.from_numpy(x_next_np.copy()).to(x_t.device, x_t.dtype)

        @staticmethod
        def backward(ctx, grad_x_next):
            """Backward pass - uses implicit VJP (no finite differences)."""
            x_t, u_t = ctx.saved_tensors

            # Convert upstream gradient
            grad_x_next_np = np.ascontiguousarray(
                grad_x_next.detach().cpu().numpy(), dtype=np.float64
            )

            # Call implicit VJP
            vjp_result = vjp_legacy_contract(
                ctx.fwd_result, grad_x_next_np, ctx.params
            )

            if not vjp_result.success:
                raise RuntimeError(
                    f"vjp_legacy_contract failed: status={vjp_result.diagnostics['status']}, "
                    f"lu_rank={vjp_result.diagnostics['lu_rank']}, "
                    f"rel_residual={vjp_result.diagnostics['rel_residual']:.2e}"
                )

            # Convert gradients back to tensors
            grad_x_t = torch.from_numpy(
                vjp_result.grad_x_t.copy()
            ).to(x_t.device, x_t.dtype)
            grad_u_t = torch.from_numpy(
                vjp_result.grad_u_t.copy()
            ).to(u_t.device, u_t.dtype)

            # Return gradients (None for dt, L_inserted, params)
            return grad_x_t, grad_u_t, None, None, None


    def legacy_dynamics_step(x_t, u_t, dt, L_inserted, params):
        """
        Convenience wrapper for LegacyHybridStep.apply().

        Args:
            x_t: State tensor (6,) - requires_grad=True for gradients
            u_t: Control tensor (3,) - requires_grad=True for gradients
            dt: Time step in seconds
            L_inserted: Insertion length in mm
            params: Catheter parameters dict

        Returns:
            x_next: Next state tensor (6,)

        Example:
            >>> x_t = torch.zeros(6, dtype=torch.float64, requires_grad=True)
            >>> u_t = torch.tensor([0.1, 0., 0.], dtype=torch.float64, requires_grad=True)
            >>> x_next = legacy_dynamics_step(x_t, u_t, 0.01, 100.0, params)
            >>> loss = x_next.sum()
            >>> loss.backward()
            >>> print(x_t.grad, u_t.grad)
        """
        return LegacyHybridStep.apply(x_t, u_t, dt, L_inserted, params)


    def legacy_dynamics_batched_vjp(
        fwd_result: LegacyStepResult,
        V: torch.Tensor,
        params: Dict[str, Any],
    ) -> tuple:
        """
        Batched VJP helper for trajectory optimization.

        Computes K VJPs in a single call, amortizing cost for iLQR / MPC.
        Does NOT use standard autograd - call this directly for batched gradients.

        Args:
            fwd_result: Cached forward result from step_legacy_contract()
            V: (K, 6) tensor of upstream gradients ∂L/∂x_{t+1}
            params: Catheter parameters dict

        Returns:
            Tuple of:
                - grad_x_t: (K, 6) tensor of gradients w.r.t. x_t
                - grad_u_t: (K, 3) tensor of gradients w.r.t. u_t

        Example:
            >>> # Run forward pass
            >>> state = LegacyState.zeros()
            >>> u_t = np.array([0.1, 0., 0.])
            >>> fwd_result = step_legacy_contract(state, u_t, 0.01, 100.0, params)
            >>>
            >>> # Compute 4 VJPs at once
            >>> V = torch.randn(4, 6, dtype=torch.float64)
            >>> grad_x_t, grad_u_t = legacy_dynamics_batched_vjp(fwd_result, V, params)
            >>> print(grad_x_t.shape, grad_u_t.shape)  # (4, 6), (4, 3)

        Note:
            This is a helper for trajectory optimizers, NOT for use with autograd.
            For single-sample autograd, use legacy_dynamics_step() instead.
        """
        if not isinstance(V, torch.Tensor):
            raise TypeError(f"V must be torch.Tensor, got {type(V)}")

        # Convert to numpy
        V_np = np.ascontiguousarray(
            V.detach().cpu().numpy(), dtype=np.float64
        )

        # Call batched VJP
        vjp_result = vjp_legacy_contract_batched(fwd_result, V_np, params)

        if not vjp_result.success:
            raise RuntimeError(
                f"vjp_legacy_contract_batched failed: status={vjp_result.diagnostics['status']}, "
                f"lu_rank={vjp_result.diagnostics['lu_rank']}, "
                f"rel_residual={vjp_result.diagnostics['rel_residual']:.2e}"
            )

        # Convert back to tensors
        grad_x_t = torch.from_numpy(
            vjp_result.grad_x_t.copy()
        ).to(V.device, V.dtype)
        grad_u_t = torch.from_numpy(
            vjp_result.grad_u_t.copy()
        ).to(V.device, V.dtype)

        return grad_x_t, grad_u_t

    _TORCH_AVAILABLE = True

except ImportError:
    # PyTorch not available - define stubs
    LegacyHybridStep = None
    legacy_dynamics_step = None
    legacy_dynamics_batched_vjp = None
    _TORCH_AVAILABLE = False


__all__ = [
    'LegacyStepResult',
    'LegacyVJPResult',
    'LegacyBatchedVJPResult',
    'step_legacy_contract',
    'vjp_legacy_contract',
    'vjp_legacy_contract_batched',
    'load_default_catheter_params',
    'LegacyHybridStep',
    'legacy_dynamics_step',
    'legacy_dynamics_batched_vjp',
]
