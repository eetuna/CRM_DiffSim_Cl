"""
CP1.5: PyTorch autograd wrapper for differentiable equilibrium primitive
"""
import torch
import numpy as np
import crm_diff_py


class EquilibriumFunction(torch.autograd.Function):
    """
    PyTorch autograd.Function for catheter equilibrium with implicit VJP.

    Forward: u [B, 3*NUM_ACT_SET] -> p_tip [B, 3]
    Backward: grad_p_tip [B, 3] -> grad_u [B, 3*NUM_ACT_SET]

    Supports batching via per-element loop.
    """

    @staticmethod
    def forward(ctx, u, L_inserted, params_dict):
        """
        Args:
            u: Tensor of shape [B, 3*NUM_ACT_SET] or [3*NUM_ACT_SET] (float64)
            L_inserted: float, insertion length (mm)
            params_dict: dict with CathParams, CathConfig, ContactMode, etc.

        Returns:
            p_tip: Tensor of shape [B, 3] or [3] (float64)
        """
        # Handle unbatched input
        if u.ndim == 1:
            u = u.unsqueeze(0)
            unbatched = True
        else:
            unbatched = False

        B, num_u = u.shape
        assert num_u == 3, f"Expected 3 actuator currents, got {num_u}"

        # Convert to numpy for C++ call
        u_np = u.detach().cpu().numpy()

        # Storage for outputs and cached data
        p_tip_list = []
        cached_results = []

        # Loop over batch
        for b in range(B):
            u_b = u_np[b]

            # Call C++ forward
            result = crm_diff_py.equilibrium_forward(u_b, L_inserted, params_dict)

            if result['status'] != 0:
                raise RuntimeError(f"equilibrium_forward failed with status {result['status']} at batch {b}")

            p_tip_list.append(result['p_tip'])
            cached_results.append(result)

        # Stack outputs
        p_tip_np = np.stack(p_tip_list, axis=0)  # [B, 3]
        p_tip = torch.from_numpy(p_tip_np).to(u.device)

        # Cache for backward
        ctx.save_for_backward(u)
        ctx.cached_results = cached_results
        ctx.unbatched = unbatched

        if unbatched:
            p_tip = p_tip.squeeze(0)

        return p_tip

    @staticmethod
    def backward(ctx, grad_p_tip):
        """
        Args:
            grad_p_tip: Tensor of shape [B, 3] or [3] (float64)

        Returns:
            grad_u: Tensor of shape [B, 3*NUM_ACT_SET] or [3*NUM_ACT_SET] (float64)
            None (for L_inserted)
            None (for params_dict)
        """
        u, = ctx.saved_tensors
        cached_results = ctx.cached_results
        unbatched = ctx.unbatched

        # Handle unbatched gradient
        if unbatched:
            grad_p_tip = grad_p_tip.unsqueeze(0)

        B = grad_p_tip.shape[0]

        # Convert to numpy
        grad_p_tip_np = grad_p_tip.detach().cpu().numpy()

        # Storage for gradients
        grad_u_list = []

        # Loop over batch
        for b in range(B):
            grad_p_tip_b = grad_p_tip_np[b]
            fwd_result = cached_results[b]

            # Call C++ backward
            bwd_result = crm_diff_py.equilibrium_backward(fwd_result, grad_p_tip_b)

            if bwd_result['status'] != 0:
                raise RuntimeError(
                    f"equilibrium_backward failed with status {bwd_result['status']} "
                    f"(rank={bwd_result['lu_rank']}, residual={bwd_result['rel_residual']:.2e}) "
                    f"at batch {b}"
                )

            grad_u_list.append(bwd_result['grad_u'])

        # Stack gradients
        grad_u_np = np.stack(grad_u_list, axis=0)  # [B, 3*NUM_ACT_SET]
        grad_u = torch.from_numpy(grad_u_np).to(u.device)

        if unbatched:
            grad_u = grad_u.squeeze(0)

        return grad_u, None, None


def equilibrium_tip_position(u, L_inserted, params_dict):
    """
    Differentiable catheter equilibrium: actuation currents -> tip position.

    Args:
        u: Tensor of shape [B, 3] or [3] (float64)
            Actuation currents (Amperes)
        L_inserted: float
            Insertion length (mm)
        params_dict: dict
            Catheter parameters (see crm_diff_py.equilibrium_forward)

    Returns:
        p_tip: Tensor of shape [B, 3] or [3] (float64)
            Tip position in spatial coordinates (mm)
    """
    return EquilibriumFunction.apply(u, L_inserted, params_dict)


def load_default_catheter_params(cath_params_file, cath_config_file):
    """
    Load catheter parameters and configuration from files.

    Args:
        cath_params_file: str, path to catheter parameter file
        cath_config_file: str, path to catheter configuration file

    Returns:
        params_dict: dict suitable for equilibrium_tip_position
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
