"""
FULLSTATE Finite-Horizon LQR Solver for Catheter Trajectory Initialization

Provides LQR warm-start capability for iLQR optimization using TRUE legacy dynamics (18·N+15):
1. Linearizing dynamics along a nominal trajectory (typically zero control)
2. Solving backward Riccati recursion with quadratic cost
3. Computing time-varying feedback gains K_t and feedforward terms k_t
4. Applying feedback law to generate an improved control sequence

This initialization improves iLQR convergence by providing a dynamically-feasible
starting point that already incorporates feedback stabilization.
"""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
import crm_diff_py
from control.true_legacy_step import true_legacy_linearize, true_legacy_tip_jacobian
from control.true_legacy_state_adapter import (
    pack_true_legacy_state, unpack_true_legacy_state, true_legacy_state_dim
)


def finite_horizon_lqr(x0, p_target, dt, L_inserted, params_dict, horizon,
                       n_act=1, Q=None, R=None, terminal_weight=1.0,
                       u_nominal=None, verbose=False):
    """
    Solve finite-horizon LQR for catheter trajectory initialization using FULLSTATE dynamics.

    Linearizes dynamics along a nominal trajectory (zero or provided control),
    then solves LQR to produce a warm-start control sequence.

    Args:
        x0: np.array (state_dim,), initial state
        p_target: np.array (3,), target tip position (mm)
        dt: float, timestep (seconds)
        L_inserted: float, insertion length (mm)
        params_dict: dict, catheter parameters
        horizon: int, planning horizon
        n_act: int, number of actuator sets (default: 1)
        Q: np.array (state_dim, state_dim), state cost matrix (default: zeros)
        R: np.array (3*n_act, 3*n_act), control cost matrix (default: I)
        terminal_weight: float, weight on terminal tip error
        u_nominal: np.array (horizon, 3*n_act), nominal control for linearization (default: zeros)
        verbose: bool, print debug info

    Returns:
        U_lqr: np.array (horizon, 3*n_act), LQR-optimized control sequence
        X_lqr: np.array (horizon+1, state_dim), resulting state trajectory
        P_tip_lqr: np.array (horizon+1, 3), resulting tip positions
    """
    state_dim = true_legacy_state_dim(n_act)  # 18*n_act + 15
    control_dim = 3 * n_act

    # Default cost matrices
    if Q is None:
        Q = np.zeros((state_dim, state_dim))
    if R is None:
        R = np.eye(control_dim)

    # Nominal trajectory for linearization
    if u_nominal is None:
        # Use a small bias toward target to seed the linearization
        x_coil_init, xf_init = unpack_true_legacy_state(x0, n_act)
        p_init = xf_init[:3]
        direction = p_target - p_init

        # Simple heuristic: small constant control in direction of target
        u_bias = np.zeros(control_dim)
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
    X_nom = np.zeros((horizon + 1, state_dim))
    P_tip_nom = np.zeros((horizon + 1, 3))
    X_nom[0] = x0

    # Get initial tip position
    x_coil_0, xf_0 = unpack_true_legacy_state(x0, n_act)
    P_tip_nom[0] = xf_0[:3]

    A_list = []  # Jacobians ∂x_next/∂x_t
    B_list = []  # Jacobians ∂x_next/∂u_t
    J_p_list = []  # Tip Jacobians ∂p_tip/∂x

    for t in range(horizon):
        x_t = X_nom[t]
        u_t = U_nom[t]

        # Forward rollout
        x_coil_t, xf_t = unpack_true_legacy_state(x_t, n_act)
        u_t_reshaped = u_t.reshape(n_act, 3)
        result = crm_diff_py.true_legacy_step_forward(
            x_coil_t, xf_t, u_t_reshaped, dt, params_dict
        )
        if not result['converged']:
            raise RuntimeError(f"[LQR] Nominal rollout failed at t={t}: converged={result['converged']}")

        X_nom[t+1] = pack_true_legacy_state(result['x_coil_next'], result['xf_next'])
        P_tip_nom[t+1] = result['xf_next'][:3]

        # Linearize via analytic implicit function theorem
        u_t_reshaped = u_t.reshape(n_act, 3)
        A_t, B_t = true_legacy_linearize(
            x_t, u_t_reshaped, dt,
            n_act=n_act,
            catheter_params=params_dict,
            L_inserted=L_inserted,
            method="implicit"
        )

        A_list.append(A_t)
        B_list.append(B_t)

        # Tip Jacobian ∂p_tip/∂x (for terminal cost)
        # Use simple extraction: tip position is first 3 elements of xf in packed state
        # For FULLSTATE (18*n_act + 15), xf starts at index 18*n_act
        J_p_t = np.zeros((3, state_dim))
        xf_start = 18 * n_act
        J_p_t[:, xf_start:xf_start+3] = np.eye(3)  # ∂p_tip/∂xf[:3] = I
        J_p_list.append(J_p_t)

    # ===== STEP 2: Terminal cost derivatives =====
    # L_T = terminal_weight * ||p_tip(x_T) - p_target||^2
    tip_error = P_tip_nom[-1] - p_target
    J_p_final = J_p_list[-1]

    # V_x = ∂L_T/∂x = 2 * w * J_p^T * (p_tip - p_target)
    V_x = 2.0 * terminal_weight * J_p_final.T @ tip_error  # (state_dim,)

    # V_xx = ∂²L_T/∂x² ≈ 2 * w * J_p^T * J_p (Gauss-Newton)
    V_xx = 2.0 * terminal_weight * (J_p_final.T @ J_p_final)  # (state_dim, state_dim)
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
        l_x = 2.0 * Q @ x_t  # (state_dim,)
        l_u = 2.0 * R @ u_t  # (control_dim,)
        l_xx = 2.0 * Q  # (state_dim, state_dim)
        l_uu = 2.0 * R  # (control_dim, control_dim)
        l_ux = np.zeros((control_dim, state_dim))

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
        Q_uu_reg = Q_uu + reg * np.eye(control_dim)

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
    X_lqr = np.zeros((horizon + 1, state_dim))
    U_lqr = np.zeros((horizon, control_dim))
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
        x_coil_t, xf_t = unpack_true_legacy_state(X_lqr[t], n_act)
        u_t_reshaped = u_t.reshape(n_act, 3)
        result = crm_diff_py.true_legacy_step_forward(
            x_coil_t, xf_t, u_t_reshaped, dt, params_dict
        )
        if not result['converged']:
            # LQR rollout failed, fall back to nominal
            if verbose:
                print(f"[LQR] Warning: forward pass failed at t={t}, using nominal control")
            X_lqr[t+1] = X_nom[t+1]
            P_tip_lqr[t+1] = P_tip_nom[t+1]
            U_lqr[t] = U_nom[t]
        else:
            X_lqr[t+1] = pack_true_legacy_state(result['x_coil_next'], result['xf_next'])
            P_tip_lqr[t+1] = result['xf_next'][:3]

    if verbose:
        final_tip_error = np.linalg.norm(P_tip_lqr[-1] - p_target)
        print(f"[LQR] Forward pass complete")
        print(f"[LQR] Final tip error: {final_tip_error:.4f} mm")
        print(f"[LQR] Max control: {np.max(np.abs(U_lqr)):.4f} A")

    return U_lqr, X_lqr, P_tip_lqr
