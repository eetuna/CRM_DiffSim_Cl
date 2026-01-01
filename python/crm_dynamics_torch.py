"""
CP2.4: PyTorch autograd wrapper for differentiable dynamics primitive
"""
import torch
import numpy as np
import crm_diff_py


class DynamicsStep(torch.autograd.Function):
    """
    PyTorch autograd.Function for one-step catheter dynamics with implicit VJP.

    Forward: (x_t, u_t) -> x_next
    Backward: grad_x_next -> (grad_x_t, grad_u_t)

    This wrapper integrates the CP2.2 matrix-dependence fix by passing all
    required cached data through ctx.
    """

    @staticmethod
    def forward(ctx, x_t, u_t, dt, L_inserted, params_dict):
        """
        Args:
            x_t: Tensor of shape [6] (float64) - Current state [u_0, v_0]
            u_t: Tensor of shape [3] (float64) - Control inputs (currents in Amperes)
            dt: float - Time step (seconds)
            L_inserted: float - Insertion length (mm)
            params_dict: dict - Physics parameters

        Returns:
            x_next: Tensor of shape [6] (float64) - Next state
        """
        # Device checks
        if x_t.device.type != 'cpu':
            raise ValueError(f"Only CPU tensors supported, got x_t on {x_t.device}")
        if u_t.device.type != 'cpu':
            raise ValueError(f"Only CPU tensors supported, got u_t on {u_t.device}")

        # Shape and dtype checks
        if x_t.shape != (6,):
            raise ValueError(f"x_t must have shape (6,), got {x_t.shape}")
        if u_t.shape != (3,):
            raise ValueError(f"u_t must have shape (3,), got {u_t.shape}")
        if x_t.dtype != torch.float64:
            raise ValueError(f"x_t must be float64, got {x_t.dtype}")
        if u_t.dtype != torch.float64:
            raise ValueError(f"u_t must be float64, got {u_t.dtype}")

        # Convert to numpy (ensure contiguity, copy only if needed)
        x_t_np = np.ascontiguousarray(x_t.detach().cpu().numpy())
        u_t_np = np.ascontiguousarray(u_t.detach().cpu().numpy())

        # Call C++ forward
        result = crm_diff_py.dynamics_forward(x_t_np, u_t_np, dt, L_inserted, params_dict)

        if result['status'] != 0:
            raise RuntimeError(
                f"dynamics_forward failed with status {result['status']} "
                f"(rank={result['lu_rank']}, exit_code={result['exit_code']})"
            )

        # Extract x_next and convert to torch (C++ returns owned array, no copy needed)
        # Use torch.from_numpy which creates a view, but result dict owns the memory
        x_next = torch.from_numpy(result['x_next']).clone()  # Clone to avoid aliasing issues

        # Cache full forward result and params_dict for backward
        ctx.save_for_backward(x_t, u_t)
        ctx.fwd_result = result  # Store entire dict
        ctx.params_dict = params_dict
        ctx.dt = dt
        ctx.L_inserted = L_inserted

        return x_next

    @staticmethod
    def backward(ctx, grad_x_next):
        """
        Args:
            grad_x_next: Tensor of shape [6] (float64) - Upstream gradient

        Returns:
            grad_x_t: Tensor of shape [6] (float64)
            grad_u_t: Tensor of shape [3] (float64)
            None (for dt)
            None (for L_inserted)
            None (for params_dict)
        """
        x_t, u_t = ctx.saved_tensors
        fwd_result = ctx.fwd_result
        params_dict = ctx.params_dict

        # Shape check
        if grad_x_next.shape != (6,):
            raise ValueError(f"grad_x_next must have shape (6,), got {grad_x_next.shape}")

        # Convert to numpy (ensure contiguity, copy only if needed)
        grad_x_next_np = np.ascontiguousarray(grad_x_next.detach().cpu().numpy())

        # Call C++ backward
        bwd_result = crm_diff_py.dynamics_backward(fwd_result, grad_x_next_np, params_dict)

        if bwd_result['status'] != 0:
            raise RuntimeError(
                f"dynamics_backward failed with status {bwd_result['status']} "
                f"(rank={bwd_result['lu_rank']}, residual={bwd_result['rel_residual']:.2e})"
            )

        # Extract gradients and convert to torch (C++ returns owned arrays)
        grad_x_t = torch.from_numpy(bwd_result['grad_x_t']).clone()
        grad_u_t = torch.from_numpy(bwd_result['grad_u_t']).clone()

        # Return gradients for (x_t, u_t, dt, L_inserted, params_dict)
        # Only x_t and u_t have gradients; others return None
        return grad_x_t, grad_u_t, None, None, None


def dynamics_step(x_t, u_t, dt, L_inserted, params_dict):
    """
    Differentiable one-step catheter dynamics: (state, control) -> next_state.

    Args:
        x_t: Tensor of shape [6] (float64)
            Current state [u_0, v_0] where:
            - u_0: base curvature (3,) in 1/mm
            - v_0: base curvature velocity (3,) in 1/mm/s
        u_t: Tensor of shape [3] (float64)
            Actuation currents (Amperes)
        dt: float
            Time step (seconds)
        L_inserted: float
            Insertion length (mm)
        params_dict: dict
            Catheter parameters (see crm_diff_py.dynamics_forward)

    Returns:
        x_next: Tensor of shape [6] (float64)
            Next state at t+dt
    """
    return DynamicsStep.apply(x_t, u_t, dt, L_inserted, params_dict)


def load_default_catheter_params(cath_params_file, cath_config_file):
    """
    Load catheter parameters and configuration from files.

    Args:
        cath_params_file: str, path to catheter parameter file
        cath_config_file: str, path to catheter configuration file

    Returns:
        params_dict: dict suitable for dynamics_step
    """
    cath_params = crm_diff_py.load_cath_params(cath_params_file)
    cath_config = crm_diff_py.load_cath_config(cath_config_file)

    params_dict = {
        'CathParams': cath_params,
        'CathConfig': cath_config,
        'ContactMode': int(crm_diff_py.ContactModeType.FREE_TIP),
        'TipForce': [0.0, 0.0, 0.0],
        'deltau0_initialguess': [0.0, 0.0, 0.0],
        'IntegrationStepSize': 0.5,
        'FinalValueOnly': True,
    }

    return params_dict
