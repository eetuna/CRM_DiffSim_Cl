"""
FULLSTATE iLQR Trajectory Optimization for Catheter Control

Implements iterative Linear Quadratic Regulator (iLQR) using TRUE legacy dynamics
(DynamicsBVP → DYNSolverIVP) with FULLSTATE representation (18·N+15).

Algorithm:
1. Forward rollout with current control sequence
2. Linearization: extract A_t, B_t Jacobians at each timestep
3. Backward pass: Riccati recursion to compute gains K_t, k_t
4. Forward pass: line search to update trajectory
5. Iterate until convergence

State: x = [x_coil[N,18], xf[15]] ∈ ℝ^(18·N+15) (TRUE legacy full state)
Control: u ∈ ℝ^(3·N) (currents)
Observable: p_tip(x) ∈ ℝ³ (tip position)
"""
import numpy as np
import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
import crm_diff_py
from control.true_legacy_step_autograd import true_legacy_step_torch
from control.true_legacy_state_adapter import (
    pack_true_legacy_state, unpack_true_legacy_state, true_legacy_state_dim
)
from control.true_legacy_step import true_legacy_linearize, true_legacy_tip_jacobian
from control.lqr import finite_horizon_lqr


class iLQRSolver:
    """
    iLQR solver for catheter trajectory optimization using FULLSTATE dynamics.

    Cost function:
        Running: L_t = x_t^T Q x_t + u_t^T R u_t
        Terminal: L_T = ||p_tip(x_T) - p_target||^2
    """

    def __init__(self, dt, L_inserted, params_dict, horizon,
                 n_act=1,
                 Q=None, R=None, p_target=None, terminal_weight=100.0,
                 max_iters=50, tol=1e-3, reg_init=1e-3, reg_scale=10.0,
                 line_search_alphas=None, terminal_hessian_mode="gn", eps_hessian=1e-5,
                 jacobian_mode="implicit"):
        """
        Args:
            dt: float, timestep (seconds)
            L_inserted: float, insertion length (mm)
            params_dict: dict, catheter parameters
            horizon: int, planning horizon (number of steps)
            n_act: int, number of actuator sets (default: 1)
            Q: np.array (state_dim, state_dim), state cost matrix (default: zeros)
            R: np.array (3*n_act, 3*n_act), control cost matrix (default: 0.1 * I)
            p_target: np.array (3,), target tip position (mm)
            terminal_weight: float, weight on terminal tip error (default: 100.0)
            max_iters: int, maximum iLQR iterations
            tol: float, convergence tolerance on tip error (mm)
            reg_init: float, initial regularization
            reg_scale: float, regularization scaling factor
            line_search_alphas: list of floats, line search step sizes
            terminal_hessian_mode: str, terminal Hessian computation mode
                                  "gn": Gauss-Newton approximation (default)
                                  "fd_exact": FD-based exact Hessian
            eps_hessian: float, finite difference step size for fd_exact mode
            jacobian_mode: str, Jacobian computation mode
                          "implicit": Analytic implicit function theorem (default, fastest)
                          "torch": PyTorch autograd
                          "cpp": Alias for "implicit"
        """
        self.dt = dt
        self.L_inserted = L_inserted
        self.params_dict = params_dict
        self.horizon = horizon
        self.n_act = n_act

        # FULLSTATE dimensions
        self.state_dim = true_legacy_state_dim(n_act)  # 18*n_act + 15
        self.control_dim = 3 * n_act

        # Cost matrices
        self.Q = Q if Q is not None else np.zeros((self.state_dim, self.state_dim))
        self.R = R if R is not None else 0.1 * np.eye(self.control_dim)
        self.p_target = p_target if p_target is not None else np.zeros(3)
        self.terminal_weight = terminal_weight

        # Solver parameters
        self.max_iters = max_iters
        self.tol = tol
        self.reg = reg_init
        self.reg_scale = reg_scale
        self.line_search_alphas = line_search_alphas if line_search_alphas is not None else \
                                  [1.0, 0.5, 0.25, 0.1, 0.05, 0.01]

        # Terminal Hessian options
        if terminal_hessian_mode not in ["gn", "fd_exact"]:
            raise ValueError(f"terminal_hessian_mode must be 'gn' or 'fd_exact', got '{terminal_hessian_mode}'")
        self.terminal_hessian_mode = terminal_hessian_mode
        self.eps_hessian = eps_hessian

        # Jacobian mode
        if jacobian_mode not in ["implicit", "torch", "cpp"]:
            raise ValueError(f"jacobian_mode must be 'implicit', 'torch', or 'cpp', got '{jacobian_mode}'")
        # Map "cpp" to "implicit" for backward compatibility
        self.jacobian_mode = "implicit" if jacobian_mode == "cpp" else jacobian_mode

        # History for diagnostics
        self.cost_history = []
        self.tip_error_history = []
        self.reg_history = []

    def forward_rollout(self, x0, U):
        """
        Simulate trajectory with control sequence U.

        Args:
            x0: np.array (state_dim,), initial state
            U: np.array (horizon, control_dim), control sequence

        Returns:
            X: np.array (horizon+1, state_dim), state trajectory
            P_tip: np.array (horizon+1, 3), tip positions
            cost: float, total trajectory cost
        """
        X = np.zeros((self.horizon + 1, self.state_dim))
        P_tip = np.zeros((self.horizon + 1, 3))
        cost = 0.0

        X[0] = x0

        # Get initial tip position
        x_coil, xf = unpack_true_legacy_state(x0, self.n_act)
        P_tip[0] = xf[:3]  # First 3 elements of xf are p_tip

        # Rollout
        for t in range(self.horizon):
            x_t = X[t]
            u_t = U[t]

            # Running cost
            cost += x_t @ self.Q @ x_t + u_t @ self.R @ u_t

            # Dynamics step using true_legacy_step
            x_coil_t, xf_t = unpack_true_legacy_state(x_t, self.n_act)
            u_t_reshaped = u_t.reshape(self.n_act, 3)
            result = crm_diff_py.true_legacy_step_forward(
                x_coil_t, xf_t, u_t_reshaped, self.dt, self.params_dict
            )
            if not result['converged']:
                raise RuntimeError(f"Forward rollout failed at t={t}: converged={result['converged']}")

            # Pack next state
            X[t+1] = pack_true_legacy_state(result['x_coil_next'], result['xf_next'])
            P_tip[t+1] = result['xf_next'][:3]

        # Terminal cost (weighted)
        tip_error = P_tip[-1] - self.p_target
        cost += self.terminal_weight * np.dot(tip_error, tip_error)

        return X, P_tip, cost

    def extract_jacobians_pytorch(self, x_t, u_t):
        """
        Extract linearization A_t, B_t using PyTorch autograd.

        Args:
            x_t: np.array (state_dim,)
            u_t: np.array (control_dim,)

        Returns:
            A: np.array (state_dim, state_dim), state Jacobian
            B: np.array (state_dim, control_dim), control Jacobian
        """
        x_t_torch = torch.tensor(x_t, dtype=torch.float64, requires_grad=True)
        u_t_torch = torch.tensor(u_t, dtype=torch.float64, requires_grad=True)

        # A = ∂x_next/∂x_t
        A = torch.autograd.functional.jacobian(
            lambda x: true_legacy_step_torch(x, u_t_torch, self.dt, self.L_inserted, self.params_dict, self.n_act),
            x_t_torch
        ).numpy()

        # B = ∂x_next/∂u_t
        B = torch.autograd.functional.jacobian(
            lambda u: true_legacy_step_torch(x_t_torch, u, self.dt, self.L_inserted, self.params_dict, self.n_act),
            u_t_torch
        ).numpy()

        return A, B

    def extract_jacobians_cpp(self, x_t, u_t):
        """
        Extract linearization A_t, B_t using C++ implicit linearization.

        Args:
            x_t: np.array (state_dim,)
            u_t: np.array (control_dim,)

        Returns:
            A: np.array (state_dim, state_dim), state Jacobian
            B: np.array (state_dim, control_dim), control Jacobian
        """
        # Reshape u to [n_act, 3] for true_legacy_linearize
        u_reshaped = u_t.reshape(self.n_act, 3)
        A, B = true_legacy_linearize(
            x_t, u_reshaped, self.dt,
            n_act=self.n_act,
            catheter_params=self.params_dict,
            L_inserted=self.L_inserted,
            method="implicit"
        )
        return A, B

    def extract_jacobians(self, x_t, u_t):
        """
        Extract linearization A_t, B_t using configured jacobian_mode.

        Args:
            x_t: np.array (state_dim,)
            u_t: np.array (control_dim,)

        Returns:
            A: np.array (state_dim, state_dim), state Jacobian
            B: np.array (state_dim, control_dim), control Jacobian
        """
        if self.jacobian_mode == "torch":
            return self.extract_jacobians_pytorch(x_t, u_t)
        elif self.jacobian_mode == "implicit":
            return self.extract_jacobians_cpp(x_t, u_t)
        else:
            raise ValueError(f"Unknown jacobian_mode: {self.jacobian_mode}")

    def compute_tip_jacobian(self, x):
        """
        Compute ∂p_tip/∂x using true_legacy_tip_jacobian or PyTorch autograd.

        Args:
            x: np.array (state_dim,), state

        Returns:
            J_p: np.array (3, state_dim), tip position Jacobian
        """
        if self.jacobian_mode == "cpp":
            # Use C++ implementation
            result = true_legacy_tip_jacobian(x, self.L_inserted, self.params_dict, self.n_act)
            return result['J_tip']
        else:
            # Use PyTorch autograd
            x_torch = torch.tensor(x, dtype=torch.float64, requires_grad=True)

            def tip_position_fn(x_in):
                x_coil, xf = unpack_true_legacy_state(x_in.detach().cpu().numpy(), self.n_act)
                # Tip position is first 3 elements of xf
                return torch.from_numpy(xf[:3])

            J_p = torch.autograd.functional.jacobian(tip_position_fn, x_torch).numpy()
            return J_p

    def compute_terminal_hessian_fd_exact(self, x, tip_error):
        """
        Compute exact terminal Hessian via finite differences.

        Terminal cost: l_T(x) = w * ||p_tip(x) - p_target||^2
        Exact Hessian: H = 2w * (J^T J + Σ_k e_k * H_k)

        Args:
            x: np.array (state_dim,), terminal state
            tip_error: np.array (3,), error vector e = p_tip - p_target

        Returns:
            V_xx: np.array (state_dim, state_dim), exact terminal Hessian
        """
        # Compute Jacobian at nominal point
        J_p = self.compute_tip_jacobian(x)  # (3, state_dim)

        # Gauss-Newton term: J^T J
        H_gn = J_p.T @ J_p  # (state_dim, state_dim)

        # Second-order term via FD: Σ_k e_k * H_k
        H_second_order = np.zeros((self.state_dim, self.state_dim))

        eps = self.eps_hessian

        for i in range(self.state_dim):
            # Perturb state in dimension i
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += eps
            x_minus[i] -= eps

            # Compute Jacobians at perturbed states
            J_plus = self.compute_tip_jacobian(x_plus)   # (3, state_dim)
            J_minus = self.compute_tip_jacobian(x_minus)  # (3, state_dim)

            # Finite difference approximation of dJ/dx_i
            dJ_dxi = (J_plus - J_minus) / (2.0 * eps)  # (3, state_dim)

            # Contract with error vector
            H_second_order[i, :] = tip_error @ dJ_dxi  # (state_dim,)

        # Combine terms and apply weight
        H_exact = 2.0 * self.terminal_weight * (H_gn + H_second_order)

        # Symmetrize
        H_exact = 0.5 * (H_exact + H_exact.T)

        return H_exact

    def backward_pass(self, X, U, verbose_debug=False):
        """
        Backward Riccati recursion to compute control gains.

        Args:
            X: np.array (horizon+1, state_dim), nominal state trajectory
            U: np.array (horizon, control_dim), nominal control sequence
            verbose_debug: bool, print diagnostic info

        Returns:
            K: list of np.array (control_dim, state_dim), feedback gains
            k: list of np.array (control_dim,), feedforward terms
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
        x_coil_final, xf_final = unpack_true_legacy_state(X[-1], self.n_act)
        p_tip_final = xf_final[:3]
        tip_error = p_tip_final - self.p_target
        J_p = self.compute_tip_jacobian(X[-1])  # (3, state_dim)

        # Terminal cost gradient
        V_x = 2.0 * self.terminal_weight * J_p.T @ tip_error  # (state_dim,)

        # Terminal cost Hessian
        if self.terminal_hessian_mode == "gn":
            V_xx = 2.0 * self.terminal_weight * (J_p.T @ J_p)  # (state_dim, state_dim)
            V_xx = 0.5 * (V_xx + V_xx.T)
        elif self.terminal_hessian_mode == "fd_exact":
            V_xx = self.compute_terminal_hessian_fd_exact(X[-1], tip_error)
        else:
            raise ValueError(f"Unknown terminal_hessian_mode: {self.terminal_hessian_mode}")

        if verbose_debug:
            V_xx_asymmetry = np.linalg.norm(V_xx - V_xx.T, 'fro')
            print(f"  [DEBUG] Terminal V_xx asymmetry: {V_xx_asymmetry:.2e} (mode={self.terminal_hessian_mode})")

        K = []
        k = []
        dV = np.array([0.0, 0.0])

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
            l_ux = np.zeros((self.control_dim, self.state_dim))

            # Q-function derivatives
            Q_x = l_x + A_t.T @ V_x
            Q_u = l_u + B_t.T @ V_x
            Q_xx = l_xx + A_t.T @ V_xx @ A_t
            Q_ux = l_ux + B_t.T @ V_xx @ A_t
            Q_uu = l_uu + B_t.T @ V_xx @ B_t

            # Symmetrize
            Q_xx = 0.5 * (Q_xx + Q_xx.T)
            Q_uu = 0.5 * (Q_uu + Q_uu.T)

            if verbose_debug and t == self.horizon - 1:
                Q_xx_asymmetry = np.linalg.norm(Q_xx - Q_xx.T, 'fro')
                Q_uu_asymmetry = np.linalg.norm(Q_uu - Q_uu.T, 'fro')
                print(f"  [DEBUG] t={t}: Q_xx asymmetry={Q_xx_asymmetry:.2e}, Q_uu asymmetry={Q_uu_asymmetry:.2e}")

            # Levenberg-Marquardt regularization
            mu = self.reg
            max_mu_tries = 10
            cholesky_success = False

            for mu_attempt in range(max_mu_tries):
                Q_uu_reg = Q_uu + mu * np.eye(self.control_dim)

                try:
                    L = np.linalg.cholesky(Q_uu_reg)
                    cholesky_success = True

                    y = np.linalg.solve(L, Q_u)
                    k_t = -np.linalg.solve(L.T, y)

                    Y = np.linalg.solve(L, Q_ux)
                    K_t = -np.linalg.solve(L.T, Y)

                    if mu_attempt > 0:
                        self.reg = mu

                    break

                except np.linalg.LinAlgError:
                    if verbose_debug and mu_attempt == 0:
                        eigvals = np.linalg.eigvalsh(Q_uu_reg)
                        print(f"  [DEBUG] t={t}: Q_uu not PD, min_eigval={eigvals[0]:.2e}, increasing μ={mu:.2e}")

                    mu *= self.reg_scale

                    if mu > 1e6:
                        if verbose_debug:
                            print(f"  [DEBUG] Backward pass failed: μ > 1e6")
                        return None, None, dV, False

            if not cholesky_success:
                return None, None, dV, False

            # Expected cost reduction
            dV[0] += k_t.T @ Q_u
            dV[1] += 0.5 * k_t.T @ Q_uu @ k_t

            # Value function update
            V_x = Q_x + K_t.T @ Q_uu @ k_t + K_t.T @ Q_u + Q_ux.T @ k_t
            V_xx = Q_xx + K_t.T @ Q_uu @ K_t + K_t.T @ Q_ux + Q_ux.T @ K_t
            V_xx = 0.5 * (V_xx + V_xx.T)

            K.insert(0, K_t)
            k.insert(0, k_t)

        return K, k, dV, True

    def forward_pass(self, x0, X_nom, U_nom, K, k, alpha):
        """
        Forward pass with line search parameter alpha.

        Args:
            x0: np.array (state_dim,), initial state
            X_nom: np.array (horizon+1, state_dim), nominal trajectory
            U_nom: np.array (horizon, control_dim), nominal controls
            K: list of np.array (control_dim, state_dim), feedback gains
            k: list of np.array (control_dim,), feedforward terms
            alpha: float, line search parameter

        Returns:
            X_new: np.array (horizon+1, state_dim), new trajectory
            U_new: np.array (horizon, control_dim), new controls
            cost_new: float, new cost
        """
        X_new = np.zeros((self.horizon + 1, self.state_dim))
        U_new = np.zeros((self.horizon, self.control_dim))
        cost_new = 0.0

        X_new[0] = x0

        for t in range(self.horizon):
            # Compute control
            dx = X_new[t] - X_nom[t]
            du = alpha * k[t] + K[t] @ dx
            u_new = U_nom[t] + du

            # Clamp controls
            u_new = np.clip(u_new, -0.5, 0.5)
            U_new[t] = u_new

            # Running cost
            cost_new += X_new[t] @ self.Q @ X_new[t] + u_new @ self.R @ u_new

            # Dynamics step
            x_coil_t, xf_t = unpack_true_legacy_state(X_new[t], self.n_act)
            u_new_reshaped = u_new.reshape(self.n_act, 3)
            result = crm_diff_py.true_legacy_step_forward(
                x_coil_t, xf_t, u_new_reshaped, self.dt, self.params_dict
            )
            if not result['converged']:
                return None, None, np.inf

            X_new[t+1] = pack_true_legacy_state(result['x_coil_next'], result['xf_next'])

        # Terminal cost
        x_coil_final, xf_final = unpack_true_legacy_state(X_new[-1], self.n_act)
        p_tip_final = xf_final[:3]
        tip_error = p_tip_final - self.p_target
        cost_new += self.terminal_weight * np.dot(tip_error, tip_error)

        return X_new, U_new, cost_new

    def solve(self, x0, U_init=None, init_method="zero", verbose=True):
        """
        Solve iLQR trajectory optimization problem.

        Args:
            x0: np.array (state_dim,), initial state
            U_init: np.array (horizon, control_dim), initial control sequence
            init_method: str, initialization method if U_init is None
                        "zero": zero control (default)
                        "lqr": LQR warm-start
            verbose: bool, print iteration info

        Returns:
            X: np.array (horizon+1, state_dim), optimal state trajectory
            U: np.array (horizon, control_dim), optimal control sequence
            converged: bool, whether solver converged
        """
        # Initialize controls
        if U_init is not None:
            U = U_init.copy()
            if verbose:
                print(f"[iLQR] Using provided U_init")
        elif init_method == "lqr":
            if verbose:
                print(f"[iLQR] Computing LQR warm-start...")
            U, _, _ = finite_horizon_lqr(
                x0, self.p_target, self.dt, self.L_inserted, self.params_dict,
                self.horizon, n_act=self.n_act, Q=self.Q, R=self.R,
                terminal_weight=self.terminal_weight, u_nominal=None, verbose=verbose
            )
            if verbose:
                print(f"[iLQR] LQR warm-start complete (max |u|={np.max(np.abs(U)):.4f} A)")
        else:
            U = np.zeros((self.horizon, self.control_dim))

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
                self.reg *= self.reg_scale
                if verbose:
                    print(f"  Backward pass failed, increasing reg to {self.reg:.6e}")

                if self.reg > 1e6:
                    if verbose:
                        print(f"iLQR diverged: regularization too large")
                    break
                continue

            # Line search
            cost_improved = False
            best_alpha = None

            for alpha in self.line_search_alphas:
                X_new, U_new, cost_new = self.forward_pass(x0, X, U, K, k, alpha)

                if cost_new < cost:
                    X = X_new
                    U = U_new
                    cost = cost_new
                    cost_improved = True
                    best_alpha = alpha

                    X, P_tip, cost = self.forward_rollout(x0, U)

                    self.reg = max(self.reg / self.reg_scale, 1e-6)
                    break

            if not cost_improved:
                self.reg *= self.reg_scale
                if verbose:
                    print(f"  Line search failed, increasing reg to {self.reg:.6e}")

                if self.reg > 1e6:
                    if verbose:
                        print(f"iLQR diverged: regularization too large")
                    break
                continue

            tip_error = np.linalg.norm(P_tip[-1] - self.p_target)

            self.cost_history.append(cost)
            self.tip_error_history.append(tip_error)
            self.reg_history.append(self.reg)

            if verbose:
                print(f"iLQR Iteration {iter+1}: cost={cost:.6f}, tip_error={tip_error:.6f} mm, reg={self.reg:.6e}, alpha={best_alpha:.3f}")

            if tip_error < self.tol:
                converged = True
                if verbose:
                    print(f"\niLQR converged in {iter+1} iterations!")
                    print(f"  Final cost: {cost:.6f}")
                    print(f"  Final tip error: {tip_error:.6f} mm")
                break

        return X, U, converged
