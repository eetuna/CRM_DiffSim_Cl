"""
CP3.2: iLQR Trajectory Optimization for Catheter Control

Implements iterative Linear Quadratic Regulator (iLQR) to drive the catheter
tip to a fixed target position using the CP2 differentiable dynamics primitive.

Algorithm:
1. Forward rollout with current control sequence
2. Linearization: extract A_t, B_t Jacobians at each timestep
3. Backward pass: Riccati recursion to compute gains K_t, k_t
4. Forward pass: line search to update trajectory
5. Iterate until convergence

State: x = [u_0, v_0] ∈ ℝ⁶ (curvature + rate)
Control: u ∈ ℝ³ (currents)
Observable: p_tip(x) ∈ ℝ³ (tip position)

P1-2 Enhancement: Added LQR warm-start capability via init_method="lqr"
P1-3 Enhancement: Added FD-based exact terminal Hessian via terminal_hessian_mode="fd_exact"
"""
import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
import crm_diff_py
from crm_dynamics_torch import dynamics_step
from control.lqr import finite_horizon_lqr


class iLQRSolver:
    """
    iLQR solver for catheter trajectory optimization to a fixed tip target.

    Cost function:
        Running: L_t = x_t^T Q x_t + u_t^T R u_t
        Terminal: L_T = ||p_tip(x_T) - p_target||^2
    """

    def __init__(self, dt, L_inserted, params_dict, horizon,
                 Q=None, R=None, p_target=None, terminal_weight=100.0,
                 max_iters=50, tol=1e-3, reg_init=1e-3, reg_scale=10.0,
                 line_search_alphas=None, terminal_hessian_mode="gn", eps_hessian=1e-5,
                 jacobian_mode="torch"):
        """
        Args:
            dt: float, timestep (seconds)
            L_inserted: float, insertion length (mm)
            params_dict: dict, catheter parameters
            horizon: int, planning horizon (number of steps)
            Q: np.array (6, 6), state cost matrix (default: zeros)
            R: np.array (3, 3), control cost matrix (default: 0.1 * I)
            p_target: np.array (3,), target tip position (mm)
            terminal_weight: float, weight on terminal tip error (default: 100.0)
            max_iters: int, maximum iLQR iterations
            tol: float, convergence tolerance on tip error (mm)
            reg_init: float, initial regularization
            reg_scale: float, regularization scaling factor
            line_search_alphas: list of floats, line search step sizes
            terminal_hessian_mode: str, terminal Hessian computation mode
                                  "gn": Gauss-Newton approximation (default, backward compatible)
                                  "fd_exact": FD-based exact Hessian (P1-3 enhancement)
            eps_hessian: float, finite difference step size for fd_exact mode (default: 1e-5)
            jacobian_mode: str, Jacobian computation mode (CP4.4b)
                          "torch": PyTorch autograd (default, backward compatible)
                          "cpp": C++ dynamics_linearize (faster)
        """
        self.dt = dt
        self.L_inserted = L_inserted
        self.params_dict = params_dict
        self.horizon = horizon

        # Cost matrices
        self.Q = Q if Q is not None else np.zeros((6, 6))
        self.R = R if R is not None else 0.1 * np.eye(3)
        self.p_target = p_target if p_target is not None else np.zeros(3)
        self.terminal_weight = terminal_weight

        # Solver parameters
        self.max_iters = max_iters
        self.tol = tol
        self.reg = reg_init
        self.reg_scale = reg_scale
        self.line_search_alphas = line_search_alphas if line_search_alphas is not None else \
                                  [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]

        # P1-3: Terminal Hessian options
        if terminal_hessian_mode not in ["gn", "fd_exact"]:
            raise ValueError(f"terminal_hessian_mode must be 'gn' or 'fd_exact', got '{terminal_hessian_mode}'")
        self.terminal_hessian_mode = terminal_hessian_mode
        self.eps_hessian = eps_hessian

        # CP4.4b: Jacobian mode
        if jacobian_mode not in ["torch", "cpp"]:
            raise ValueError(f"jacobian_mode must be 'torch' or 'cpp', got '{jacobian_mode}'")
        self.jacobian_mode = jacobian_mode

        # History for diagnostics
        self.cost_history = []
        self.tip_error_history = []
        self.reg_history = []

    def forward_rollout(self, x0, U):
        """
        Simulate trajectory with control sequence U.

        Args:
            x0: np.array (6,), initial state
            U: np.array (horizon, 3), control sequence

        Returns:
            X: np.array (horizon+1, 6), state trajectory
            P_tip: np.array (horizon+1, 3), tip positions
            cost: float, total trajectory cost
        """
        X = np.zeros((self.horizon + 1, 6))
        P_tip = np.zeros((self.horizon + 1, 3))
        cost = 0.0

        X[0] = x0

        # Get initial tip position
        result = crm_diff_py.dynamics_forward(x0, np.zeros(3), 0.0, self.L_inserted, self.params_dict)
        if result['status'] != 0:
            raise RuntimeError(f"Initial tip position computation failed: status={result['status']}")
        P_tip[0] = result['p_tip']

        # Rollout
        for t in range(self.horizon):
            x_t = X[t]
            u_t = U[t]

            # Running cost
            cost += x_t @ self.Q @ x_t + u_t @ self.R @ u_t

            # Dynamics step
            result = crm_diff_py.dynamics_forward(x_t, u_t, self.dt, self.L_inserted, self.params_dict)
            if result['status'] != 0:
                raise RuntimeError(f"Forward rollout failed at t={t}: status={result['status']}")

            X[t+1] = result['x_next']
            P_tip[t+1] = result['p_tip']

        # Terminal cost (weighted)
        tip_error = P_tip[-1] - self.p_target
        cost += self.terminal_weight * np.dot(tip_error, tip_error)

        return X, P_tip, cost

    def extract_jacobians_pytorch(self, x_t, u_t):
        """
        Extract linearization A_t, B_t using PyTorch autograd.

        Args:
            x_t: np.array (6,)
            u_t: np.array (3,)

        Returns:
            A: np.array (6, 6), state Jacobian
            B: np.array (6, 3), control Jacobian
        """
        x_t_torch = torch.tensor(x_t, dtype=torch.float64, requires_grad=True)
        u_t_torch = torch.tensor(u_t, dtype=torch.float64, requires_grad=True)

        # A = ∂x_next/∂x_t
        A = torch.autograd.functional.jacobian(
            lambda x: dynamics_step(x, u_t_torch, self.dt, self.L_inserted, self.params_dict),
            x_t_torch
        ).numpy()

        # B = ∂x_next/∂u_t
        B = torch.autograd.functional.jacobian(
            lambda u: dynamics_step(x_t_torch, u, self.dt, self.L_inserted, self.params_dict),
            u_t_torch
        ).numpy()

        return A, B

    def extract_jacobians_cpp(self, x_t, u_t):
        """
        CP4.4b: Extract linearization A_t, B_t using C++ dynamics_linearize.

        Args:
            x_t: np.array (6,)
            u_t: np.array (3,)

        Returns:
            A: np.array (6, 6), state Jacobian
            B: np.array (6, 3), control Jacobian
        """
        result = crm_diff_py.dynamics_linearize(
            x_t, u_t, self.dt, self.L_inserted, self.params_dict
        )
        return result['A'], result['B']

    def extract_jacobians(self, x_t, u_t):
        """
        Extract linearization A_t, B_t using configured jacobian_mode.

        Args:
            x_t: np.array (6,)
            u_t: np.array (3,)

        Returns:
            A: np.array (6, 6), state Jacobian
            B: np.array (6, 3), control Jacobian
        """
        if self.jacobian_mode == "torch":
            return self.extract_jacobians_pytorch(x_t, u_t)
        elif self.jacobian_mode == "cpp":
            return self.extract_jacobians_cpp(x_t, u_t)
        else:
            raise ValueError(f"Unknown jacobian_mode: {self.jacobian_mode}")

    def compute_tip_jacobian(self, x):
        """
        Compute ∂p_tip/∂x using PyTorch autograd.

        Args:
            x: np.array (6,), state

        Returns:
            J_p: np.array (3, 6), tip position Jacobian
        """
        x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)

        def tip_position_fn(x_in):
            # Use dynamics_forward with zero control and zero dt to get tip position
            result = crm_diff_py.dynamics_forward(
                x_in.detach().cpu().numpy(),
                np.zeros(3),
                0.0,
                self.L_inserted,
                self.params_dict
            )
            return torch.from_numpy(result['p_tip'])

        # J_p = ∂p_tip/∂x (3, 6)
        J_p = torch.autograd.functional.jacobian(tip_position_fn, x_torch).numpy()

        return J_p

    def compute_terminal_hessian_fd_exact(self, x, tip_error):
        """
        Compute exact-ish terminal Hessian via finite differences (P1-3).

        Terminal cost: l_T(x) = w * ||p_tip(x) - p_target||^2
        Exact Hessian: H = 2w * (J^T J + Σ_k e_k * H_k)
        where e = p_tip - p_target, J = ∂p_tip/∂x, H_k = ∂²p_k/∂x²

        We approximate the second-order term via FD of J w.r.t. x:
        For each state dim i:
            J_plus = ∂p_tip/∂x|_{x + eps*e_i}
            J_minus = ∂p_tip/∂x|_{x - eps*e_i}
            dJ/dx_i ≈ (J_plus - J_minus) / (2*eps)
        Then: (Σ_k e_k * H_k)_{ij} ≈ Σ_k e_k * (dJ_kj/dx_i)

        Args:
            x: np.array (6,), terminal state
            tip_error: np.array (3,), error vector e = p_tip - p_target

        Returns:
            V_xx: np.array (6, 6), exact terminal Hessian
        """
        # Compute Jacobian at nominal point
        J_p = self.compute_tip_jacobian(x)  # (3, 6)

        # Gauss-Newton term: J^T J
        H_gn = J_p.T @ J_p  # (6, 6)

        # Second-order term via FD: Σ_k e_k * H_k
        H_second_order = np.zeros((6, 6))

        eps = self.eps_hessian

        for i in range(6):
            # Perturb state in dimension i
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps

            # Compute Jacobians at perturbed states
            J_plus = self.compute_tip_jacobian(x_plus)   # (3, 6)
            J_minus = self.compute_tip_jacobian(x_minus)  # (3, 6)

            # Finite difference approximation of dJ/dx_i
            dJ_dxi = (J_plus - J_minus) / (2.0 * eps)  # (3, 6)

            # Contract with error vector: Σ_k e_k * (dJ_kj/dx_i) for all j
            # This gives row i of the Hessian
            H_second_order[i, :] = tip_error @ dJ_dxi  # (6,)

        # Combine terms and apply weight
        H_exact = 2.0 * self.terminal_weight * (H_gn + H_second_order)

        # Symmetrize (FD introduces numerical asymmetry)
        H_exact = 0.5 * (H_exact + H_exact.T)

        return H_exact

    def backward_pass(self, X, U, verbose_debug=False):
        """
        Backward Riccati recursion to compute control gains.

        Args:
            X: np.array (horizon+1, 6), nominal state trajectory
            U: np.array (horizon, 3), nominal control sequence
            verbose_debug: bool, print diagnostic info

        Returns:
            K: list of np.array (3, 6), feedback gains
            k: list of np.array (3,), feedforward terms
            dV: np.array (2,), expected cost reduction [linear, quadratic]
            success: bool, whether backward pass succeeded
        """
        # Linearize dynamics at each timestep
        A_list = []
        B_list = []
        for t in range(self.horizon):
            A_t, B_t = self.extract_jacobians(X[t], U[t])
            A_list.append(A_t)
            B_list.append(B_t)

        # Terminal cost derivatives
        # L_T = w * ||p_tip(x_T) - p_target||^2
        # ∂L_T/∂x = 2 * w * J_p^T * (p_tip - p_target)
        # ∂²L_T/∂x²: Depends on terminal_hessian_mode

        p_tip_final = self.compute_tip_position(X[-1])
        tip_error = p_tip_final - self.p_target
        J_p = self.compute_tip_jacobian(X[-1])  # (3, 6)

        # Terminal cost gradient (same for both modes)
        V_x = 2.0 * self.terminal_weight * J_p.T @ tip_error  # (6,)

        # Terminal cost Hessian (mode-dependent)
        if self.terminal_hessian_mode == "gn":
            # Gauss-Newton approximation: ∂²L/∂x² ≈ 2*w * J_p^T * J_p
            V_xx = 2.0 * self.terminal_weight * (J_p.T @ J_p)  # (6, 6)
            # Ensure symmetry
            V_xx = 0.5 * (V_xx + V_xx.T)
        elif self.terminal_hessian_mode == "fd_exact":
            # FD-based exact Hessian (P1-3)
            V_xx = self.compute_terminal_hessian_fd_exact(X[-1], tip_error)
        else:
            raise ValueError(f"Unknown terminal_hessian_mode: {self.terminal_hessian_mode}")

        # Check terminal Hessian symmetry (diagnostic)
        if verbose_debug:
            V_xx_asymmetry = np.linalg.norm(V_xx - V_xx.T, 'fro')
            print(f"  [DEBUG] Terminal V_xx asymmetry: {V_xx_asymmetry:.2e} (mode={self.terminal_hessian_mode})")

        K = []
        k = []
        dV = np.array([0.0, 0.0])  # Expected cost change: linear and quadratic terms

        # Backward pass
        for t in range(self.horizon - 1, -1, -1):
            A_t = A_list[t]
            B_t = B_list[t]
            x_t = X[t]
            u_t = U[t]

            # Running cost derivatives
            l_x = 2.0 * self.Q @ x_t
            l_u = 2.0 * self.R @ u_t
            l_xx = 2.0 * self.Q
            l_uu = 2.0 * self.R
            l_ux = np.zeros((3, 6))

            # Q-function derivatives
            Q_x = l_x + A_t.T @ V_x
            Q_u = l_u + B_t.T @ V_x
            Q_xx = l_xx + A_t.T @ V_xx @ A_t
            Q_ux = l_ux + B_t.T @ V_xx @ A_t
            Q_uu = l_uu + B_t.T @ V_xx @ B_t

            # Symmetrize Q-matrices
            Q_xx = 0.5 * (Q_xx + Q_xx.T)
            Q_uu = 0.5 * (Q_uu + Q_uu.T)

            # Check symmetry (diagnostic)
            if verbose_debug and t == self.horizon - 1:
                Q_xx_asymmetry = np.linalg.norm(Q_xx - Q_xx.T, 'fro')
                Q_uu_asymmetry = np.linalg.norm(Q_uu - Q_uu.T, 'fro')
                print(f"  [DEBUG] t={t}: Q_xx asymmetry={Q_xx_asymmetry:.2e}, Q_uu asymmetry={Q_uu_asymmetry:.2e}")

            # Levenberg-Marquardt regularization with Cholesky solve
            # Try to factorize Q_uu + μ*I, increasing μ until PD
            mu = self.reg
            max_mu_tries = 10
            cholesky_success = False

            for mu_attempt in range(max_mu_tries):
                Q_uu_reg = Q_uu + mu * np.eye(3)

                try:
                    # Cholesky decomposition (fails if not PD)
                    L = np.linalg.cholesky(Q_uu_reg)
                    cholesky_success = True

                    # Solve using Cholesky: Q_uu_reg^-1 = (L L^T)^-1
                    # For Q_uu_inv @ Q_u: solve L L^T x = Q_u
                    y = np.linalg.solve(L, Q_u)
                    k_t = -np.linalg.solve(L.T, y)

                    # For Q_uu_inv @ Q_ux: solve L L^T X = Q_ux
                    Y = np.linalg.solve(L, Q_ux)
                    K_t = -np.linalg.solve(L.T, Y)

                    # Update regularization for next timestep
                    if mu_attempt == 0:
                        # Successfully factorized with current mu, no change needed
                        pass
                    else:
                        # Had to increase mu, keep it for now
                        self.reg = mu

                    break

                except np.linalg.LinAlgError:
                    # Cholesky failed, Q_uu_reg not PD
                    if verbose_debug and mu_attempt == 0:
                        eigvals = np.linalg.eigvalsh(Q_uu_reg)
                        print(f"  [DEBUG] t={t}: Q_uu not PD, min_eigval={eigvals[0]:.2e}, increasing μ={mu:.2e}")

                    # Increase regularization
                    mu *= self.reg_scale

                    if mu > 1e6:
                        # Regularization too large, fail
                        if verbose_debug:
                            print(f"  [DEBUG] Backward pass failed: μ > 1e6")
                        return None, None, dV, False

            if not cholesky_success:
                return None, None, dV, False

            # Expected cost reduction (negative for descent)
            dV[0] += k_t.T @ Q_u  # Linear term
            dV[1] += 0.5 * k_t.T @ Q_uu @ k_t  # Quadratic term

            # Value function update for next iteration
            V_x = Q_x + K_t.T @ Q_uu @ k_t + K_t.T @ Q_u + Q_ux.T @ k_t
            V_xx = Q_xx + K_t.T @ Q_uu @ K_t + K_t.T @ Q_ux + Q_ux.T @ K_t
            V_xx = 0.5 * (V_xx + V_xx.T)  # Symmetrize

            # Prepend (we're going backwards)
            K.insert(0, K_t)
            k.insert(0, k_t)

        return K, k, dV, True

    def forward_pass(self, x0, X_nom, U_nom, K, k, alpha):
        """
        Forward pass with line search parameter alpha.

        Control law: u_t = u_nom_t + alpha * k_t + K_t @ (x_t - x_nom_t)

        Args:
            x0: np.array (6,), initial state
            X_nom: np.array (horizon+1, 6), nominal trajectory
            U_nom: np.array (horizon, 3), nominal controls
            K: list of np.array (3, 6), feedback gains
            k: list of np.array (3,), feedforward terms
            alpha: float, line search parameter

        Returns:
            X_new: np.array (horizon+1, 6), new trajectory
            U_new: np.array (horizon, 3), new controls
            cost_new: float, new cost
        """
        X_new = np.zeros((self.horizon + 1, 6))
        U_new = np.zeros((self.horizon, 3))
        cost_new = 0.0

        X_new[0] = x0

        for t in range(self.horizon):
            # Compute control
            dx = X_new[t] - X_nom[t]
            du = alpha * k[t] + K[t] @ dx
            u_new = U_nom[t] + du

            # Clamp controls to safe range
            u_new = np.clip(u_new, -0.5, 0.5)
            U_new[t] = u_new

            # Running cost
            cost_new += X_new[t] @ self.Q @ X_new[t] + u_new @ self.R @ u_new

            # Dynamics step
            result = crm_diff_py.dynamics_forward(X_new[t], u_new, self.dt, self.L_inserted, self.params_dict)
            if result['status'] != 0:
                # Return infinite cost if dynamics fail
                return None, None, np.inf

            X_new[t+1] = result['x_next']

        # Terminal cost (weighted)
        p_tip_final = self.compute_tip_position(X_new[-1])
        tip_error = p_tip_final - self.p_target
        cost_new += self.terminal_weight * np.dot(tip_error, tip_error)

        return X_new, U_new, cost_new

    def compute_tip_position(self, x):
        """
        Compute tip position for a given state.

        Args:
            x: np.array (6,), state

        Returns:
            p_tip: np.array (3,), tip position (mm)
        """
        result = crm_diff_py.dynamics_forward(x, np.zeros(3), 0.0, self.L_inserted, self.params_dict)
        if result['status'] != 0:
            raise RuntimeError(f"Tip position computation failed: status={result['status']}")
        return result['p_tip']

    def solve(self, x0, U_init=None, init_method="zero", verbose=True):
        """
        Solve iLQR trajectory optimization problem.

        Args:
            x0: np.array (6,), initial state
            U_init: np.array (horizon, 3), initial control sequence (optional, overrides init_method)
            init_method: str, initialization method if U_init is None
                        "zero": zero control (default, backward compatible)
                        "lqr": LQR warm-start (P1-2 enhancement)
            verbose: bool, print iteration info

        Returns:
            X: np.array (horizon+1, 6), optimal state trajectory
            U: np.array (horizon, 3), optimal control sequence
            converged: bool, whether solver converged
        """
        # Initialize controls
        if U_init is not None:
            # Explicit initialization provided, use it
            U = U_init.copy()
            if verbose:
                print(f"[iLQR] Using provided U_init")
        elif init_method == "lqr":
            # LQR warm-start (P1-2)
            if verbose:
                print(f"[iLQR] Computing LQR warm-start...")
            U, _, _ = finite_horizon_lqr(
                x0, self.p_target, self.dt, self.L_inserted, self.params_dict,
                self.horizon, Q=self.Q, R=self.R, terminal_weight=self.terminal_weight,
                u_nominal=None, verbose=verbose
            )
            if verbose:
                print(f"[iLQR] LQR warm-start complete (max |u|={np.max(np.abs(U)):.4f} A)")
        else:
            # Default: zero initialization
            U = np.zeros((self.horizon, 3))

        # Initial rollout
        X, P_tip, cost = self.forward_rollout(x0, U)
        tip_error = np.linalg.norm(P_tip[-1] - self.p_target)

        self.cost_history = [cost]
        self.tip_error_history = [tip_error]
        self.reg_history = [self.reg]

        if verbose:
            print(f"iLQR Iteration 0: cost={cost:.6f}, tip_error={tip_error:.6f} mm")

        converged = False

        for iter in range(self.max_iters):
            # Backward pass
            K, k, dV, bp_success = self.backward_pass(X, U, verbose_debug=False)

            if not bp_success:
                # Backward pass failed, increase regularization and retry
                self.reg *= self.reg_scale
                if verbose:
                    print(f"  Backward pass failed, increasing reg to {self.reg:.6e}")

                if self.reg > 1e6:
                    if verbose:
                        print(f"iLQR diverged: regularization too large")
                    break
                continue

            # Predicted cost reduction
            dV_pred = dV[0]  # Use linear term for conservative estimate

            # Line search
            cost_improved = False
            best_alpha = None
            best_cost = cost

            for alpha in self.line_search_alphas:
                X_new, U_new, cost_new = self.forward_pass(x0, X, U, K, k, alpha)

                if cost_new < cost:
                    # Accept step
                    X = X_new
                    U = U_new
                    cost = cost_new
                    cost_improved = True
                    best_alpha = alpha

                    # Re-compute full rollout to get tip positions
                    X, P_tip, cost = self.forward_rollout(x0, U)

                    # Decrease regularization on success
                    self.reg = max(self.reg / self.reg_scale, 1e-6)
                    break

            if not cost_improved:
                # Increase regularization and retry
                self.reg *= self.reg_scale
                if verbose:
                    print(f"  Line search failed, increasing reg to {self.reg:.6e}")

                if self.reg > 1e6:
                    if verbose:
                        print(f"iLQR diverged: regularization too large")
                    break
                continue

            # Compute tip error from updated trajectory
            tip_error = np.linalg.norm(P_tip[-1] - self.p_target)

            self.cost_history.append(cost)
            self.tip_error_history.append(tip_error)
            self.reg_history.append(self.reg)

            if verbose:
                print(f"iLQR Iteration {iter+1}: cost={cost:.6f}, tip_error={tip_error:.6f} mm, reg={self.reg:.6e}, alpha={best_alpha:.3f}")

            # Check convergence
            if tip_error < self.tol:
                converged = True
                if verbose:
                    print(f"\niLQR converged in {iter+1} iterations!")
                    print(f"  Final cost: {cost:.6f}")
                    print(f"  Final tip error: {tip_error:.6f} mm")
                break

        return X, U, converged
