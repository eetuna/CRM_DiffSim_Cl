#!/usr/bin/env python3
"""
CP4.7: Hybrid MPC + Learned Policy Controller

Combines fast ensemble policy with correct-but-expensive MPC backup.
Uses ensemble uncertainty to decide when to trust the policy vs invoke MPC.

Key principle: Fast by default, safe when needed.

Decision logic:
    σ = max(ensemble_variance)
    if σ < τ_low:       → POLICY-ONLY (fast, ~0.1 ms)
    elif σ < τ_high:    → MPC-WARM (medium, ~10-50 ms)
    else:               → MPC-COLD (slow, ~50-200 ms)
    + SAFETY OVERRIDE if dynamics fail or tracking error exceeds limit

Reuses:
- EnsemblePolicy (CP4.5) for uncertainty estimation
- iLQRSolver (CP3.2) for MPC with jacobian_mode="cpp" (CP4.4c)
- dynamics_forward (CP2) for safety checks
"""

import sys
import os
import time
import numpy as np
from typing import Optional, Tuple, Dict, Any

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))

from models.ensemble_policy import EnsemblePolicy
from control.ilqr import iLQRSolver
import crm_diff_py


class HybridController:
    """
    Hybrid MPC + Ensemble Policy Controller.

    Uses ensemble uncertainty to decide when to trust the policy vs invoke MPC.
    Provides performance of learned policy with safety guarantees of MPC.
    """

    def __init__(
        self,
        ensemble: EnsemblePolicy,
        dt: float,
        L_inserted: float,
        params_dict: dict,
        tau_low: float = 0.001,
        tau_high: float = 0.01,
        tracking_safety_limit: float = 5.0,
        mpc_horizon: int = 10,
        mpc_max_iters: int = 10,
        mpc_tol: float = 1e-3,
        dynamics_safety_checks: bool = True,
        control_limits: Tuple[float, float] = (-0.5, 0.5)
    ):
        """
        Initialize hybrid controller.

        Args:
            ensemble: Trained ensemble policy (from CP4.5)
            dt: Timestep (seconds)
            L_inserted: Insertion length (mm)
            params_dict: Physics parameters
            tau_low: Policy-only threshold (default: 0.001)
            tau_high: MPC-cold threshold (default: 0.01)
            tracking_safety_limit: Max tracking error before override (mm)
            mpc_horizon: MPC planning horizon
            mpc_max_iters: Max iLQR iterations
            mpc_tol: iLQR convergence tolerance (mm)
            dynamics_safety_checks: Enable safety checks
            control_limits: Min/max control (Amperes)
        """
        self.ensemble = ensemble
        self.dt = dt
        self.L_inserted = L_inserted
        self.params_dict = params_dict

        # Thresholds
        self.tau_low = tau_low
        self.tau_high = tau_high
        self.tracking_safety_limit = tracking_safety_limit

        # MPC parameters
        self.mpc_horizon = mpc_horizon
        self.mpc_max_iters = mpc_max_iters
        self.mpc_tol = mpc_tol

        # Safety
        self.dynamics_safety_checks = dynamics_safety_checks
        self.control_limits = control_limits

        # State
        self.hiddens = None
        self.U_prev = None  # Previous MPC solution for shift warm-start
        self.step_count = 0

    def reset(self):
        """Reset controller state (hidden states and MPC cache)."""
        self.hiddens = None
        self.U_prev = None
        self.step_count = 0

    def step(
        self,
        x_t: np.ndarray,
        p_tip_t: np.ndarray,
        p_ref_horizon: np.ndarray,
        hiddens: Optional = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Execute one control step.

        Args:
            x_t: Current state (6,)
            p_tip_t: Current tip position (3,)
            p_ref_horizon: Reference trajectory (H, 3) where H >= mpc_horizon
            hiddens: Recurrent hidden states (optional, uses internal state if None)

        Returns:
            u_t: Control action (3,)
            info: Dict with:
                - 'mode': 'policy' | 'mpc_warm' | 'mpc_cold' | 'safety_override'
                - 'uncertainty': float (max variance)
                - 'mpc_called': bool
                - 'mpc_time_ms': float (if MPC called)
                - 'converged': bool (if MPC called)
                - 'tracking_error': float (if available)
        """
        self.step_count += 1

        # Use provided hiddens or internal state
        if hiddens is None:
            if self.hiddens is None:
                self.hiddens = self.ensemble.init_hidden(batch_size=1)
            hiddens = self.hiddens

        # Step 1: Query ensemble
        u_mean, u_var, hiddens_new = self.ensemble.predict(
            p_tip_t.astype(np.float32),
            p_ref_horizon[0].astype(np.float32),
            hiddens
        )
        u_mean = u_mean.astype(np.float64)
        u_var = u_var.astype(np.float64)

        # Update hidden state
        self.hiddens = hiddens_new

        # Step 2: Compute uncertainty
        uncertainty = float(np.max(u_var))

        # Step 3: Decision logic
        info = {
            'uncertainty': uncertainty,
            'mpc_called': False,
            'step': self.step_count
        }

        # Initialize control
        u_t = None
        mode = None

        # Decision tree
        if uncertainty < self.tau_low:
            # POLICY-ONLY (fast path)
            mode = 'policy'
            u_t = np.clip(u_mean, *self.control_limits)

        elif uncertainty < self.tau_high:
            # MPC-WARM (medium path)
            mode = 'mpc_warm'
            warm_start_mode = 'policy'
            u_t, mpc_info = self._invoke_mpc(
                x_t, p_ref_horizon, warm_start_mode, u_policy_hint=u_mean
            )
            info.update(mpc_info)
            info['mpc_called'] = True

        else:
            # MPC-COLD (slow path, high uncertainty)
            mode = 'mpc_cold'
            warm_start_mode = 'shift' if self.U_prev is not None else 'cold'
            u_t, mpc_info = self._invoke_mpc(
                x_t, p_ref_horizon, warm_start_mode, u_policy_hint=None
            )
            info.update(mpc_info)
            info['mpc_called'] = True

        info['mode'] = mode

        # Step 4: Safety check
        if self.dynamics_safety_checks:
            # Try the control and check if it's safe
            result = crm_diff_py.dynamics_forward(
                x_t, u_t, self.dt, self.L_inserted, self.params_dict
            )

            status = result['status']
            lu_rank = result.get('lu_rank', 6)
            rel_residual = result.get('rel_solve_residual', 0.0)

            dynamics_safe = (
                status == 0 and
                lu_rank == 6 and
                rel_residual < 1e-10
            )

            # Check tracking error
            if dynamics_safe:
                p_tip_next = result['p_tip']
                tracking_error = float(np.linalg.norm(p_tip_next - p_ref_horizon[0]))
                info['tracking_error'] = tracking_error

                if tracking_error > self.tracking_safety_limit:
                    dynamics_safe = False

            # SAFETY OVERRIDE
            if not dynamics_safe:
                # Force MPC with cold start
                mode = 'safety_override'
                u_t, mpc_info = self._invoke_mpc(
                    x_t, p_ref_horizon, 'cold', u_policy_hint=None
                )
                info.update(mpc_info)
                info['mpc_called'] = True
                info['mode'] = mode
                info['safety_triggered'] = True

        return u_t, info

    def _invoke_mpc(
        self,
        x_t: np.ndarray,
        p_ref_horizon: np.ndarray,
        warm_start_mode: str,
        u_policy_hint: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Invoke MPC solver.

        Args:
            x_t: Current state (6,)
            p_ref_horizon: Reference trajectory (H, 3)
            warm_start_mode: 'policy' | 'shift' | 'cold'
            u_policy_hint: Policy suggestion for warm-start (if mode='policy')

        Returns:
            u_t: Control action (3,)
            info: Dict with MPC details
        """
        t_start = time.time()

        # Prepare horizon
        H = min(self.mpc_horizon, len(p_ref_horizon))
        p_ref_mpc = p_ref_horizon[:H]

        # Pad if needed
        if len(p_ref_mpc) < self.mpc_horizon:
            p_ref_mpc = np.vstack([
                p_ref_mpc,
                np.tile(p_ref_mpc[-1], (self.mpc_horizon - len(p_ref_mpc), 1))
            ])

        # Generate warm-start
        U_init = None
        if warm_start_mode == 'policy' and u_policy_hint is not None:
            U_init = self._generate_policy_warm_start(
                x_t, p_ref_mpc, u_policy_hint
            )
        elif warm_start_mode == 'shift' and self.U_prev is not None:
            U_init = np.vstack([self.U_prev[1:], self.U_prev[-1:]])

        # Create iLQR solver
        solver = iLQRSolver(
            dt=self.dt,
            L_inserted=self.L_inserted,
            params_dict=self.params_dict,
            horizon=self.mpc_horizon,
            Q=np.zeros((6, 6)),  # No state cost
            R=0.01 * np.eye(3),  # Control regularization
            p_target=p_ref_mpc[0],  # Terminal target
            terminal_weight=1.0,
            max_iters=self.mpc_max_iters,
            tol=self.mpc_tol,
            jacobian_mode="cpp"  # Use CP4.4c fast Jacobians
        )

        # Solve
        try:
            X, U, converged = solver.solve(x_t, U_init=U_init, verbose=False)
            u_t = np.clip(U[0], *self.control_limits)
            self.U_prev = U  # Cache for shift warm-start

            mpc_time = time.time() - t_start

            return u_t, {
                'converged': converged,
                'mpc_time_ms': mpc_time * 1000.0,
                'mpc_iters': len(solver.cost_history),
                'warm_start_mode': warm_start_mode
            }

        except Exception as e:
            # MPC failed, return zero control
            mpc_time = time.time() - t_start
            return np.zeros(3), {
                'converged': False,
                'mpc_time_ms': mpc_time * 1000.0,
                'error': str(e),
                'warm_start_mode': warm_start_mode
            }

    def _generate_policy_warm_start(
        self,
        x_t: np.ndarray,
        p_ref_horizon: np.ndarray,
        u_policy_hint: np.ndarray
    ) -> np.ndarray:
        """
        Generate MPC warm-start by rolling out the policy.

        Args:
            x_t: Current state (6,)
            p_ref_horizon: Reference trajectory (H, 3)
            u_policy_hint: First control from policy

        Returns:
            U_init: Control sequence (H, 3)
        """
        H = len(p_ref_horizon)
        U_init = np.zeros((H, 3))

        # Use policy hint for first control
        U_init[0] = np.clip(u_policy_hint, *self.control_limits)

        # Fill rest with repeated hint (simple heuristic)
        # More sophisticated: could rollout policy, but adds complexity
        for t in range(1, H):
            U_init[t] = U_init[0]

        return U_init


class HybridControllerMetrics:
    """Collect and report hybrid controller metrics."""

    def __init__(self):
        self.steps = []
        self.modes = []
        self.uncertainties = []
        self.tracking_errors = []
        self.mpc_times = []
        self.mpc_converged = []
        self.mpc_iters = []

    def add_step(
        self,
        info: Dict[str, Any],
        p_tip: np.ndarray,
        p_ref: np.ndarray
    ):
        """
        Add a timestep.

        Args:
            info: Info dict from HybridController.step()
            p_tip: Actual tip position (3,)
            p_ref: Reference tip position (3,)
        """
        self.steps.append(info.get('step', len(self.steps)))
        self.modes.append(info['mode'])
        self.uncertainties.append(info['uncertainty'])

        # Tracking error
        tracking_error = float(np.linalg.norm(p_tip - p_ref))
        self.tracking_errors.append(tracking_error)

        # MPC info
        if info.get('mpc_called', False):
            self.mpc_times.append(info.get('mpc_time_ms', 0.0))
            self.mpc_converged.append(info.get('converged', False))
            self.mpc_iters.append(info.get('mpc_iters', 0))

    def get_summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        n_steps = len(self.steps)
        if n_steps == 0:
            return {}

        # Mode distribution
        mode_counts = {}
        for mode in ['policy', 'mpc_warm', 'mpc_cold', 'safety_override']:
            mode_counts[mode] = self.modes.count(mode)

        mode_distribution = {
            mode: count / n_steps
            for mode, count in mode_counts.items()
        }

        # MPC metrics
        n_mpc_calls = sum(1 for m in self.modes if m != 'policy')
        mpc_call_rate = n_mpc_calls / n_steps

        # Tracking metrics
        tracking_errors_arr = np.array(self.tracking_errors)
        tracking_rmse = float(np.sqrt(np.mean(tracking_errors_arr**2)))
        tracking_max = float(np.max(tracking_errors_arr))
        tracking_mean = float(np.mean(tracking_errors_arr))

        # Uncertainty-error correlation
        if len(self.uncertainties) > 1 and len(self.tracking_errors) > 1:
            corr_matrix = np.corrcoef(self.uncertainties, self.tracking_errors)
            uncertainty_error_corr = float(corr_matrix[0, 1])
        else:
            uncertainty_error_corr = 0.0

        summary = {
            'n_steps': n_steps,
            'mpc_call_rate': mpc_call_rate,
            'mode_distribution': mode_distribution,
            'tracking_rmse': tracking_rmse,
            'tracking_max': tracking_max,
            'tracking_mean': tracking_mean,
            'mean_uncertainty': float(np.mean(self.uncertainties)),
            'uncertainty_error_correlation': uncertainty_error_corr
        }

        # MPC-specific metrics
        if self.mpc_times:
            summary['mean_mpc_time_ms'] = float(np.mean(self.mpc_times))
            summary['max_mpc_time_ms'] = float(np.max(self.mpc_times))

        if self.mpc_converged:
            summary['mpc_convergence_rate'] = sum(self.mpc_converged) / len(self.mpc_converged)

        if self.mpc_iters:
            summary['mean_mpc_iters'] = float(np.mean(self.mpc_iters))

        return summary

    def print_summary(self):
        """Print summary to console."""
        summary = self.get_summary()

        print("\nHybrid Controller Metrics Summary")
        print("=" * 60)
        print(f"Total steps: {summary['n_steps']}")
        print(f"MPC call rate: {summary['mpc_call_rate']:.1%}")
        print()

        print("Mode Distribution:")
        for mode, frac in summary['mode_distribution'].items():
            print(f"  {mode:20s}: {frac:6.1%}")
        print()

        print("Tracking Performance:")
        print(f"  RMSE:  {summary['tracking_rmse']:.4f} mm")
        print(f"  Max:   {summary['tracking_max']:.4f} mm")
        print(f"  Mean:  {summary['tracking_mean']:.4f} mm")
        print()

        print("Uncertainty:")
        print(f"  Mean: {summary['mean_uncertainty']:.6f}")
        print(f"  Uncertainty-Error Correlation: {summary['uncertainty_error_correlation']:.3f}")
        print()

        if 'mean_mpc_time_ms' in summary:
            print("MPC Performance:")
            print(f"  Mean time: {summary['mean_mpc_time_ms']:.2f} ms")
            print(f"  Max time:  {summary['max_mpc_time_ms']:.2f} ms")
            if 'mpc_convergence_rate' in summary:
                print(f"  Convergence rate: {summary['mpc_convergence_rate']:.1%}")
            if 'mean_mpc_iters' in summary:
                print(f"  Mean iterations: {summary['mean_mpc_iters']:.1f}")

        print("=" * 60)
