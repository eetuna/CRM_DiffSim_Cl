"""
Finite-Horizon LQR Solver for Catheter Trajectory Initialization

Provides LQR warm-start capability for iLQR optimization by:
1. Linearizing dynamics along a nominal trajectory (typically zero control)
2. Solving backward Riccati recursion with quadratic cost
3. Computing time-varying feedback gains K_t and feedforward terms k_t
4. Applying feedback law to generate an improved control sequence

This initialization improves iLQR convergence by providing a dynamically-feasible
starting point that already incorporates feedback stabilization.
"""
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
from crm_dynamics_torch import dynamics_step


def finite_horizon_lqr(x0, p_target, dt, L_inserted, params_dict, horizon,
                       Q=None, R=None, terminal_weight=1.0,
                       u_nominal=None, verbose=False):
    """
    Solve finite-horizon LQR for catheter trajectory initialization.

    Linearizes dynamics along a nominal trajectory (zero or provided control),
    then solves LQR to produce a warm-start control sequence.

    Args:
        x0: np.array (6,), initial state
        p_target: np.array (3,), target tip position (mm)
        dt: float, timestep (seconds)
        L_inserted: float, insertion length (mm)
        params_dict: dict, catheter parameters
        horizon: int, planning horizon
        Q: np.array (6, 6), state cost matrix (default: zeros)
        R: np.array (3, 3), control cost matrix (default: I)
        terminal_weight: float, weight on terminal tip error
        u_nominal: np.array (horizon, 3), nominal control for linearization (default: zeros)
        verbose: bool, print debug info

    Returns:
        U_lqr: np.array (horizon, 3), LQR-optimized control sequence
        X_lqr: np.array (horizon+1, 6), resulting state trajectory
        P_tip_lqr: np.array (horizon+1, 3), resulting tip positions
    """
    import crm_diff_py

    # Default cost matrices
    if Q is None:
        Q = np.zeros((6, 6))
    if R is None:
        R = np.eye(3)

    # Nominal trajectory for linearization
    if u_nominal is None:
        # Use a small bias toward target to seed the linearization
        # Compute rough direction from initial tip to target
        import crm_diff_py
        result_init = crm_diff_py.dynamics_forward(x0, np.zeros(3), 0.0, L_inserted, params_dict)
        p_init = result_init['p_tip']
        direction = p_target - p_init

        # Simple heuristic: small constant control in direction of target
        # This provides a non-zero nominal trajectory to linearize around
        u_bias = np.zeros(3)
        if np.linalg.norm(direction[:2]) > 0.1:  # If target is off-axis
            # Use first two coils to steer
            u_bias[0] = 0.05 * np.sign(direction[0]) if abs(direction[0]) > 0.1 else 0.0
            u_bias[1] = 0.05 * np.sign(direction[1]) if abs(direction[1]) > 0.1 else 0.0

        U_nom = np.tile(u_bias, (horizon, 1))
    else:
        U_nom = u_nominal.copy()

    if verbose:
        print(f"[LQR] Computing linearization along nominal trajectory...")

    # ===== STEP 1: Nominal rollout + linearization =====
    X_nom = np.zeros((horizon + 1, 6))
    P_tip_nom = np.zeros((horizon + 1, 3))
    X_nom[0] = x0

    # Get initial tip position
    result_init = crm_diff_py.dynamics_forward(x0, np.zeros(3), 0.0, L_inserted, params_dict)
    P_tip_nom[0] = result_init['p_tip']

    A_list = []  # Jacobians ∂x_next/∂x_t
    B_list = []  # Jacobians ∂x_next/∂u_t
    J_p_list = []  # Tip Jacobians ∂p_tip/∂x

    for t in range(horizon):
        x_t = X_nom[t]
        u_t = U_nom[t]

        # Forward rollout
        result = crm_diff_py.dynamics_forward(x_t, u_t, dt, L_inserted, params_dict)
        if result['status'] != 0:
            raise RuntimeError(f"[LQR] Nominal rollout failed at t={t}: status={result['status']}")

        X_nom[t+1] = result['x_next']
        P_tip_nom[t+1] = result['p_tip']

        # Linearize via PyTorch autograd
        x_t_torch = torch.tensor(x_t, dtype=torch.float64, requires_grad=True)
        u_t_torch = torch.tensor(u_t, dtype=torch.float64, requires_grad=True)

        # A = ∂x_next/∂x_t
        A_t = torch.autograd.functional.jacobian(
            lambda x: dynamics_step(x, u_t_torch, dt, L_inserted, params_dict),
            x_t_torch
        ).numpy()

        # B = ∂x_next/∂u_t
        B_t = torch.autograd.functional.jacobian(
            lambda u: dynamics_step(x_t_torch, u, dt, L_inserted, params_dict),
            u_t_torch
        ).numpy()

        A_list.append(A_t)
        B_list.append(B_t)

        # Tip Jacobian ∂p_tip/∂x (for terminal cost)
        x_next_torch = torch.tensor(X_nom[t+1], dtype=torch.float64, requires_grad=True)

        def tip_position_fn(x_in):
            res = crm_diff_py.dynamics_forward(
                x_in.detach().cpu().numpy(),
                np.zeros(3),
                0.0,
                L_inserted,
                params_dict
            )
            return torch.from_numpy(res['p_tip'])

        J_p_t = torch.autograd.functional.jacobian(tip_position_fn, x_next_torch).numpy()
        J_p_list.append(J_p_t)

    # ===== STEP 2: Terminal cost derivatives =====
    # L_T = terminal_weight * ||p_tip(x_T) - p_target||^2
    tip_error = P_tip_nom[-1] - p_target
    J_p_final = J_p_list[-1]

    # V_x = ∂L_T/∂x = 2 * w * J_p^T * (p_tip - p_target)
    V_x = 2.0 * terminal_weight * J_p_final.T @ tip_error  # (6,)

    # V_xx = ∂²L_T/∂x² ≈ 2 * w * J_p^T * J_p (Gauss-Newton)
    V_xx = 2.0 * terminal_weight * (J_p_final.T @ J_p_final)  # (6, 6)
    V_xx = 0.5 * (V_xx + V_xx.T)  # Symmetrize

    if verbose:
        print(f"[LQR] Terminal cost: tip_error = {np.linalg.norm(tip_error):.4f} mm")

    # ===== STEP 3: Backward Riccati recursion =====
    K_gains = []  # Feedback gains
    k_feedforward = []  # Feedforward terms

    for t in range(horizon - 1, -1, -1):
        A_t = A_list[t]
        B_t = B_list[t]
        x_t = X_nom[t]
        u_t = U_nom[t]

        # Running cost derivatives
        l_x = 2.0 * Q @ x_t  # (6,)
        l_u = 2.0 * R @ u_t  # (3,)
        l_xx = 2.0 * Q  # (6, 6)
        l_uu = 2.0 * R  # (3, 3)
        l_ux = np.zeros((3, 6))

        # Q-function derivatives
        Q_x = l_x + A_t.T @ V_x
        Q_u = l_u + B_t.T @ V_x
        Q_xx = l_xx + A_t.T @ V_xx @ A_t
        Q_ux = l_ux + B_t.T @ V_xx @ A_t
        Q_uu = l_uu + B_t.T @ V_xx @ B_t

        # Symmetrize
        Q_xx = 0.5 * (Q_xx + Q_xx.T)
        Q_uu = 0.5 * (Q_uu + Q_uu.T)

        # LQR gains (regularization for numerical stability)
        reg = 1e-4
        Q_uu_reg = Q_uu + reg * np.eye(3)

        try:
            # k = -Q_uu^{-1} Q_u (feedforward)
            k_t = -np.linalg.solve(Q_uu_reg, Q_u)
            # K = -Q_uu^{-1} Q_ux (feedback)
            K_t = -np.linalg.solve(Q_uu_reg, Q_ux)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse
            k_t = -np.linalg.pinv(Q_uu_reg) @ Q_u
            K_t = -np.linalg.pinv(Q_uu_reg) @ Q_ux

        # Value function update
        V_x = Q_x + K_t.T @ Q_uu @ k_t + K_t.T @ Q_u + Q_ux.T @ k_t
        V_xx = Q_xx + K_t.T @ Q_uu @ K_t + K_t.T @ Q_ux + Q_ux.T @ K_t
        V_xx = 0.5 * (V_xx + V_xx.T)

        # Store gains (prepend since going backward)
        K_gains.insert(0, K_t)
        k_feedforward.insert(0, k_t)

    if verbose:
        print(f"[LQR] Backward pass complete, gains computed for {horizon} timesteps")

    # ===== STEP 4: Forward pass with LQR control =====
    X_lqr = np.zeros((horizon + 1, 6))
    U_lqr = np.zeros((horizon, 3))
    P_tip_lqr = np.zeros((horizon + 1, 3))

    X_lqr[0] = x0
    P_tip_lqr[0] = P_tip_nom[0]

    for t in range(horizon):
        # LQR control law: u = u_nom + k + K(x - x_nom)
        dx = X_lqr[t] - X_nom[t]
        du = k_feedforward[t] + K_gains[t] @ dx
        u_t = U_nom[t] + du

        # Clamp to safe range
        u_t = np.clip(u_t, -0.5, 0.5)
        U_lqr[t] = u_t

        # Forward dynamics
        result = crm_diff_py.dynamics_forward(X_lqr[t], u_t, dt, L_inserted, params_dict)
        if result['status'] != 0:
            # LQR rollout failed, fall back to nominal
            if verbose:
                print(f"[LQR] Warning: forward pass failed at t={t}, using nominal control")
            X_lqr[t+1] = X_nom[t+1]
            P_tip_lqr[t+1] = P_tip_nom[t+1]
            U_lqr[t] = U_nom[t]
        else:
            X_lqr[t+1] = result['x_next']
            P_tip_lqr[t+1] = result['p_tip']

    if verbose:
        final_tip_error = np.linalg.norm(P_tip_lqr[-1] - p_target)
        print(f"[LQR] Forward pass complete")
        print(f"[LQR] Final tip error: {final_tip_error:.4f} mm")
        print(f"[LQR] Max control: {np.max(np.abs(U_lqr)):.4f} A")

    return U_lqr, X_lqr, P_tip_lqr
