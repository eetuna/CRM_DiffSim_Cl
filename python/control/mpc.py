"""
FULLSTATE Receding-Horizon Model Predictive Control (MPC) for Catheter Control

Implements MPC using warm-started iLQR with TRUE legacy dynamics (18·N+15).
At each timestep:
1. Solve finite-horizon optimal control problem using iLQR
2. Apply first control input
3. Shift solution to warm-start next iteration
4. Repeat

This provides real-time trajectory tracking with replanning.
"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))
import crm_diff_py
from control.ilqr import iLQRSolver
from control.true_legacy_state_adapter import (
    pack_true_legacy_state, unpack_true_legacy_state, true_legacy_state_dim
)


class MPCController:
    """
    Model Predictive Control using warm-started iLQR with FULLSTATE dynamics.

    At each timestep:
    - Solve optimal control over horizon T
    - Apply first control u_0
    - Shift solution [u_1, ..., u_{T-1}] and append u_{T-1}
    - Replan from new state
    """

    def __init__(self, dt, L_inserted, params_dict, horizon=10,
                 n_act=1, Q_tip=1.0, R=None, max_ilqr_iters=3,
                 cost_decrease_tol=1e-2, verbose=False):
        """
        Args:
            dt: float, timestep (seconds)
            L_inserted: float, insertion length (mm)
            params_dict: dict, catheter parameters
            horizon: int, MPC planning horizon (default: 10)
            n_act: int, number of actuator sets (default: 1)
            Q_tip: float, weight on tip position tracking error (default: 1.0)
            R: np.array (3*n_act, 3*n_act), control cost matrix (default: 0.01 * I)
            max_ilqr_iters: int, max iLQR iterations per MPC step (default: 3)
            cost_decrease_tol: float, early exit if cost decrease < tol (default: 1e-2)
            verbose: bool, print debug info
        """
        self.dt = dt
        self.L_inserted = L_inserted
        self.params_dict = params_dict
        self.horizon = horizon
        self.n_act = n_act
        self.Q_tip = Q_tip

        self.state_dim = true_legacy_state_dim(n_act)  # 18*n_act + 15
        self.control_dim = 3 * n_act

        self.R = R if R is not None else 0.01 * np.eye(self.control_dim)
        self.max_ilqr_iters = max_ilqr_iters
        self.cost_decrease_tol = cost_decrease_tol
        self.verbose = verbose

        # Warm-start buffer
        self.U_prev = None

        # Trajectory reference (set via set_reference)
        self.p_target_fn = None

    def set_reference(self, p_target_fn):
        """
        Set reference trajectory function.

        Args:
            p_target_fn: callable, p_target_fn(t) -> np.array (3,)
                         Returns desired tip position at time t
        """
        self.p_target_fn = p_target_fn

    def reset(self):
        """Reset warm-start buffer."""
        self.U_prev = None

    def compute_control(self, x_current, t_current):
        """
        Compute MPC control at current timestep.

        Args:
            x_current: np.array (state_dim,), current state
            t_current: float, current time (seconds)

        Returns:
            u_mpc: np.array (control_dim,), control to apply
            info: dict, diagnostic information
        """
        if x_current.shape != (self.state_dim,):
            raise ValueError(
                f"x_current must have shape ({self.state_dim},), got {x_current.shape}"
            )
        # Get target position at end of horizon for look-ahead tracking
        t_horizon = t_current + self.horizon * self.dt
        p_target = self.p_target_fn(t_horizon)

        # Cost matrices
        Q = np.zeros((self.state_dim, self.state_dim))  # No direct state cost
        R = self.R

        # Create iLQR solver for this MPC step
        solver = iLQRSolver(
            dt=self.dt,
            L_inserted=self.L_inserted,
            params_dict=self.params_dict,
            horizon=self.horizon,
            n_act=self.n_act,
            Q=Q,
            R=R,
            p_target=p_target,
            terminal_weight=self.Q_tip,
            max_iters=self.max_ilqr_iters,
            tol=1e-2,
            reg_init=1e-3,
            reg_scale=10.0,
            line_search_alphas=[1.0, 0.5, 0.25, 0.1]
        )

        # Warm-start: shift previous solution
        if self.U_prev is None:
            # Cold start with zero controls for stability
            U_init = np.zeros((self.horizon, self.control_dim))
        else:
            # Shift previous solution and append last control
            U_init = np.zeros((self.horizon, self.control_dim))
            U_init[:-1] = self.U_prev[1:]  # Shift left
            U_init[-1] = self.U_prev[-1]    # Repeat last control

        # Solve iLQR
        X_opt, U_opt, converged = solver.solve(
            x_current,
            U_init=U_init,
            verbose=self.verbose
        )

        # Store solution for warm-start
        self.U_prev = U_opt.copy()

        # Extract first control (MPC receding horizon)
        u_mpc = U_opt[0]

        # Diagnostic info
        info = {
            'converged': converged,
            'num_iters': len(solver.cost_history) - 1,
            'final_cost': solver.cost_history[-1],
            'cost_history': solver.cost_history,
            'p_target': p_target,
            'U_horizon': U_opt,
            'X_horizon': X_opt
        }

        if self.verbose:
            print(f"[MPC t={t_current:.3f}s] iters={info['num_iters']}, "
                  f"cost={info['final_cost']:.6f}, target={p_target}")

        return u_mpc, info


def simulate_mpc_tracking(controller, x0, t_start, t_end, dt):
    """
    Simulate closed-loop MPC tracking.

    Args:
        controller: MPCController instance
        x0: np.array (state_dim,), initial state
        t_start: float, start time (seconds)
        t_end: float, end time (seconds)
        dt: float, simulation timestep

    Returns:
        t_history: np.array (N,), time history
        x_history: np.array (N, state_dim), state history
        u_history: np.array (N, control_dim), control history
        p_tip_history: np.array (N, 3), tip position history
        p_target_history: np.array (N, 3), target position history
        info_history: list of dicts, diagnostic info per timestep
    """
    # Reset controller
    controller.reset()

    # Simulation setup
    num_steps = int((t_end - t_start) / dt)

    t_history = np.zeros(num_steps + 1)
    x_history = np.zeros((num_steps + 1, controller.state_dim))
    u_history = np.zeros((num_steps, controller.control_dim))
    p_tip_history = np.zeros((num_steps + 1, 3))
    p_target_history = np.zeros((num_steps + 1, 3))
    info_history = []

    # Initial state
    x_current = x0.copy()
    t_current = t_start

    t_history[0] = t_current
    x_history[0] = x_current

    # Get initial tip position
    x_coil_0, xf_0 = unpack_true_legacy_state(x_current, controller.n_act)
    p_tip_history[0] = xf_0[:3]
    p_target_history[0] = controller.p_target_fn(t_current)

    # Closed-loop simulation
    warmstart = None
    for step in range(num_steps):
        # Progress indicator
        if step % 10 == 0:
            print(f"  MPC step {step}/{num_steps} (t={t_current:.3f}s)", flush=True)

        # Compute MPC control
        u_mpc, info = controller.compute_control(x_current, t_current)
        u_history[step] = u_mpc
        info_history.append(info)

        # Apply control and step dynamics
        x_coil_t, xf_t = unpack_true_legacy_state(x_current, controller.n_act)
        result = crm_diff_py.true_legacy_step_forward(
            x_coil_t,
            xf_t,
            u_mpc,
            dt,
            controller.params_dict,
            None if warmstart is None else warmstart['mL_guess'],
            None if warmstart is None else warmstart['nL_guess'],
        )
        if not result['converged']:
            raise RuntimeError(f"Dynamics step failed at t={t_current}: converged={result['converged']}")
        warmstart = {
            'mL_guess': np.ascontiguousarray(result['mL_next'], dtype=np.float64),
            'nL_guess': np.ascontiguousarray(result['nL_next'], dtype=np.float64),
        }

        # Update state
        x_current = pack_true_legacy_state(result['x_coil_next'], result['xf_next'])
        t_current = t_start + (step + 1) * dt

        # Record
        t_history[step + 1] = t_current
        x_history[step + 1] = x_current
        p_tip_history[step + 1] = result['xf_next'][:3]
        p_target_history[step + 1] = controller.p_target_fn(t_current)

    return (t_history, x_history, u_history, p_tip_history,
            p_target_history, info_history)


def compute_tracking_metrics(p_tip_history, p_target_history):
    """
    Compute tracking performance metrics.

    Args:
        p_tip_history: np.array (N, 3), tip positions
        p_target_history: np.array (N, 3), target positions

    Returns:
        metrics: dict with 'rms_error', 'max_error', 'mean_error'
    """
    errors = np.linalg.norm(p_tip_history - p_target_history, axis=1)

    metrics = {
        'rms_error': np.sqrt(np.mean(errors**2)),
        'max_error': np.max(errors),
        'mean_error': np.mean(errors),
        'final_error': errors[-1]
    }

    return metrics
