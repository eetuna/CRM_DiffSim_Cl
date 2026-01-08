#!/usr/bin/env python3
"""
FULLSTATE Hybrid MPC + Learned Policy Controller

Combines fast ensemble policy with correct-but-expensive MPC backup using TRUE legacy dynamics (18·N+15).
Uses ensemble uncertainty to decide when to trust the policy vs invoke MPC.

Key principle: Fast by default, safe when needed.

Decision logic:
    σ = max(ensemble_variance)
    if σ < τ_low:       → POLICY-ONLY (fast, ~0.1 ms)
    elif σ < τ_high:    → MPC-WARM (medium, ~10-50 ms)
    else:               → MPC-COLD (slow, ~50-200 ms)
    + SAFETY OVERRIDE if dynamics fail or tracking error exceeds limit

Reuses:
- EnsemblePolicy for uncertainty estimation
- iLQRSolver with FULLSTATE dynamics and jacobian_mode="implicit"
- true_legacy_step_forward for safety checks
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
from control.true_legacy_state_adapter import (
    pack_true_legacy_state, unpack_true_legacy_state, true_legacy_state_dim
)
import crm_diff_py


class HybridController:
    """
    Hybrid MPC + Ensemble Policy Controller with FULLSTATE dynamics.

    Uses ensemble uncertainty to decide when to trust the policy vs invoke MPC.
    Provides performance of learned policy with safety guarantees of MPC.
    """

    def __init__(
        self,
        ensemble: EnsemblePolicy,
        dt: float,
        L_inserted: float,
        params_dict: dict,
        n_act: int = 1,
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
            ensemble: Trained ensemble policy
            dt: Timestep (seconds)
            L_inserted: Insertion length (mm)
            params_dict: Physics parameters
            n_act: Number of actuator sets (default: 1)
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
        self.n_act = n_act

        # FULLSTATE dimensions
        self.state_dim = true_legacy_state_dim(n_act)  # 18*n_act + 15
        self.control_dim = 3 * n_act

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
            x_t: Current state (state_dim,)
            p_tip_t: Current tip position (3,)
            p_ref_horizon: Reference trajectory (H, 3) where H >= mpc_horizon
            hiddens: Recurrent hidden states (optional, uses internal state if None)

        Returns:
            u_t: Control action (control_dim,)
            info: Dict with:
                - 'mode': 'policy' | 'mpc_warm' | 'mpc_cold' | 'safety_override'
                - 'uncertainty': float (max variance)
                - 'mpc_called': bool
                - 'mpc_time_ms': float (if MPC called)
                - 'converged': bool (if MPC called)
                - 'tracking_error': float (if available)
        """
        self.step_count += 1
        if x_t.shape != (self.state_dim,):
            raise ValueError(
                f"x_t must have shape ({self.state_dim},), got {x_t.shape}"
            )
        if p_tip_t.shape != (3,):
            raise ValueError(f"p_tip_t must have shape (3,), got {p_tip_t.shape}")
        if p_ref_horizon.ndim != 2 or p_ref_horizon.shape[1] != 3:
            raise ValueError(
                f"p_ref_horizon must have shape (H, 3), got {p_ref_horizon.shape}"
            )

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
            x_coil_t, xf_t = unpack_true_legacy_state(x_t, self.n_act)
            result = crm_diff_py.true_legacy_step_forward(
                x_coil_t, xf_t, u_t.reshape(self.n_act, 3), self.dt, self.params_dict
            )

            if 'status' in result:
                dynamics_safe = (result['status'] == 0)
            else:
                dynamics_safe = bool(result.get('converged', False))

            # Check tracking error
            if dynamics_safe:
                p_tip_next = result['xf_next'][:3]
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
            x_t: Current state (state_dim,)
            p_ref_horizon: Reference trajectory (H, 3)
            warm_start_mode: 'policy' | 'shift' | 'cold'
            u_policy_hint: Policy suggestion for warm-start (if mode='policy')

        Returns:
            u_t: Control action (control_dim,)
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
            n_act=self.n_act,
            Q=np.zeros((self.state_dim, self.state_dim)),  # No state cost
            R=0.01 * np.eye(self.control_dim),  # Control regularization
            p_target=p_ref_mpc[0],  # Terminal target
            terminal_weight=1.0,
            max_iters=self.mpc_max_iters,
            tol=self.mpc_tol,
            jacobian_mode="implicit"  # Use analytic implicit Jacobians
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
            return np.zeros(self.control_dim), {
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
            x_t: Current state (state_dim,)
            p_ref_horizon: Reference trajectory (H, 3)
            u_policy_hint: First control from policy

        Returns:
            U_init: Control sequence (H, control_dim)
        """
        H = len(p_ref_horizon)
        U_init = np.zeros((H, self.control_dim))

        # Use policy hint for first control
        U_init[0] = np.clip(u_policy_hint, *self.control_limits)

        # Fill rest with repeated hint (simple heuristic)
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

    def record(self, info: Dict[str, Any]):
        """Record metrics from a control step."""
        self.steps.append(info['step'])
        self.modes.append(info['mode'])
        self.uncertainties.append(info['uncertainty'])
        if 'tracking_error' in info:
            self.tracking_errors.append(info['tracking_error'])
        if info['mpc_called']:
            self.mpc_times.append(info['mpc_time_ms'])
            self.mpc_converged.append(info['converged'])
            self.mpc_iters.append(info['mpc_iters'])

    def summary(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        mode_counts = {}
        for mode in set(self.modes):
            mode_counts[mode] = self.modes.count(mode)

        summary = {
            'total_steps': len(self.steps),
            'mode_counts': mode_counts,
            'mode_percentages': {
                mode: 100.0 * count / len(self.steps)
                for mode, count in mode_counts.items()
            },
            'mean_uncertainty': float(np.mean(self.uncertainties)),
            'max_uncertainty': float(np.max(self.uncertainties)),
        }

        if self.tracking_errors:
            summary['mean_tracking_error'] = float(np.mean(self.tracking_errors))
            summary['max_tracking_error'] = float(np.max(self.tracking_errors))

        if self.mpc_times:
            summary['mpc_calls'] = len(self.mpc_times)
            summary['mean_mpc_time_ms'] = float(np.mean(self.mpc_times))
            summary['max_mpc_time_ms'] = float(np.max(self.mpc_times))
            summary['mpc_convergence_rate'] = float(np.mean(self.mpc_converged))
            summary['mean_mpc_iters'] = float(np.mean(self.mpc_iters))

        return summary
