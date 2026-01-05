"""
CP3.3: Receding-Horizon Model Predictive Control (MPC) for Catheter Control

Implements MPC using warm-started iLQR as the underlying optimizer.
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


class MPCController:
    """
    Model Predictive Control using warm-started iLQR.

    At each timestep:
    - Solve optimal control over horizon T
    - Apply first control u_0
    - Shift solution [u_1, ..., u_{T-1}] and append u_{T-1}
    - Replan from new state
    """

    def __init__(self, dt, L_inserted, params_dict, horizon=10,
                 Q_tip=1.0, R=None, max_ilqr_iters=3,
                 cost_decrease_tol=1e-2, verbose=False):
        """
        Args:
            dt: float, timestep (seconds)
            L_inserted: float, insertion length (mm)
            params_dict: dict, catheter parameters
            horizon: int, MPC planning horizon (default: 10)
            Q_tip: float, weight on tip position tracking error (default: 1.0)
            R: np.array (3, 3), control cost matrix (default: 0.01 * I)
            max_ilqr_iters: int, max iLQR iterations per MPC step (default: 3)
            cost_decrease_tol: float, early exit if cost decrease < tol (default: 1e-2)
            verbose: bool, print debug info
        """
        self.dt = dt
        self.L_inserted = L_inserted
        self.params_dict = params_dict
        self.horizon = horizon
        self.Q_tip = Q_tip
        self.R = R if R is not None else 0.01 * np.eye(3)
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
            x_current: np.array (6,), current state
            t_current: float, current time (seconds)

        Returns:
            u_mpc: np.array (3,), control to apply
            info: dict, diagnostic information
        """
        # Get target position at end of horizon for look-ahead tracking
        t_horizon = t_current + self.horizon * self.dt
        p_target = self.p_target_fn(t_horizon)

        # Cost matrices
        Q = np.zeros((6, 6))  # No direct state cost
        R = self.R

        # Create iLQR solver for this MPC step
        # Use terminal weight to penalize tip error at horizon end
        solver = iLQRSolver(
            dt=self.dt,
            L_inserted=self.L_inserted,
            params_dict=self.params_dict,
            horizon=self.horizon,
            Q=Q,
            R=R,
            p_target=p_target,
            terminal_weight=self.Q_tip,
            max_iters=self.max_ilqr_iters,
            tol=1e-2,  # Not critical since we limit iterations
            reg_init=1e-3,
            reg_scale=10.0,
            line_search_alphas=[1.0, 0.5, 0.25, 0.1]
        )

        # Warm-start: shift previous solution
        if self.U_prev is None:
            # Cold start with small random perturbation to break symmetry
            np.random.seed(42)  # Reproducible
            U_init = np.random.randn(self.horizon, 3) * 0.02
        else:
            # Shift previous solution and append last control
            U_init = np.zeros((self.horizon, 3))
            U_init[:-1] = self.U_prev[1:]  # Shift left
            U_init[-1] = self.U_prev[-1]    # Repeat last control

        # Solve iLQR with early termination
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
        x0: np.array (6,), initial state
        t_start: float, start time (seconds)
        t_end: float, end time (seconds)
        dt: float, simulation timestep

    Returns:
        t_history: np.array (N,), time history
        x_history: np.array (N, 6), state history
        u_history: np.array (N, 3), control history
        p_tip_history: np.array (N, 3), tip position history
        p_target_history: np.array (N, 3), target position history
        info_history: list of dicts, diagnostic info per timestep
    """
    # Reset controller
    controller.reset()

    # Simulation setup
    num_steps = int((t_end - t_start) / dt)

    t_history = np.zeros(num_steps + 1)
    x_history = np.zeros((num_steps + 1, 6))
    u_history = np.zeros((num_steps, 3))
    p_tip_history = np.zeros((num_steps + 1, 3))
    p_target_history = np.zeros((num_steps + 1, 3))
    info_history = []

    # Initial state
    x_current = x0.copy()
    t_current = t_start

    t_history[0] = t_current
    x_history[0] = x_current

    # Get initial tip position
    result = crm_diff_py.dynamics_forward(
        x_current, np.zeros(3), 0.0,
        controller.L_inserted, controller.params_dict
    )
    if result['status'] != 0:
        raise RuntimeError(f"Initial tip computation failed: status={result['status']}")
    p_tip_history[0] = result['p_tip']
    p_target_history[0] = controller.p_target_fn(t_current)

    # Closed-loop simulation
    for step in range(num_steps):
        # Progress indicator
        if step % 10 == 0:
            print(f"  MPC step {step}/{num_steps} (t={t_current:.3f}s)", flush=True)

        # Compute MPC control
        u_mpc, info = controller.compute_control(x_current, t_current)
        u_history[step] = u_mpc
        info_history.append(info)

        # Apply control and step dynamics
        result = crm_diff_py.dynamics_forward(
            x_current, u_mpc, dt,
            controller.L_inserted, controller.params_dict
        )
        if result['status'] != 0:
            raise RuntimeError(f"Dynamics step failed at t={t_current}: status={result['status']}")

        # Update state
        x_current = result['x_next']
        t_current = t_start + (step + 1) * dt

        # Record
        t_history[step + 1] = t_current
        x_history[step + 1] = x_current
        p_tip_history[step + 1] = result['p_tip']
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
