#!/usr/bin/env python3
"""
CP4.5: Ensemble Policy with Uncertainty Estimation

Combines N independently trained recurrent policies to provide:
1. Mean prediction (ensemble output)
2. Variance estimation (epistemic uncertainty)

Epistemic uncertainty indicates model disagreement, which can be used to
trigger expert queries in DAgger when the policy is uncertain.

Key features:
- Load N policy checkpoints (different random seeds)
- Compute mean and per-action variance
- Compatible with GRU/LSTM policies
- Same predict() interface as single policies
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union


class EnsemblePolicy:
    """
    Ensemble of N recurrent policies with uncertainty estimation.

    The ensemble computes:
    - Mean control: u_mean = (1/N) Σ u_i
    - Variance (epistemic uncertainty): var = (1/N) Σ (u_i - u_mean)²

    Usage:
        ensemble = EnsemblePolicy(policy_paths, PolicyClass)
        u_mean, u_var, hiddens = ensemble.predict(p_tip, p_ref, hiddens)

        if np.max(u_var) > threshold:
            # High uncertainty → query expert
            u = query_expert(...)
        else:
            # Low uncertainty → trust ensemble
            u = u_mean
    """

    def __init__(
        self,
        policy_paths: List[str],
        policy_class,
        policy_kwargs: dict = None,
        device: str = 'cpu'
    ):
        """
        Initialize ensemble from saved checkpoints.

        Args:
            policy_paths: List of paths to policy checkpoints (.pth files)
            policy_class: Policy class (e.g., GRUPolicy, LSTMPolicy)
            policy_kwargs: Arguments for policy constructor (e.g., hidden_dim=64)
            device: Device for inference ('cpu' or 'cuda')
        """
        if policy_kwargs is None:
            policy_kwargs = {}

        self.device = device
        self.n_policies = len(policy_paths)

        if self.n_policies == 0:
            raise ValueError("Must provide at least one policy path")

        # Load policies
        self.policies = []
        for path in policy_paths:
            policy = policy_class(**policy_kwargs)
            state_dict = torch.load(path, map_location=device)
            policy.load_state_dict(state_dict)
            policy.to(device)
            policy.eval()
            self.policies.append(policy)

        # Extract architecture info from first policy
        self.hidden_dim = self.policies[0].hidden_dim
        self.num_layers = self.policies[0].num_layers
        self.output_dim = self.policies[0].output_dim

        print(f"EnsemblePolicy: Loaded {self.n_policies} policies")
        print(f"  Architecture: {policy_class.__name__}, hidden_dim={self.hidden_dim}")

    def init_hidden(
        self,
        batch_size: int = 1,
        device: Optional[str] = None
    ) -> List[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Initialize hidden states for all policies in the ensemble.

        Args:
            batch_size: Batch size (default: 1)
            device: Device override (default: use ensemble device)

        Returns:
            List of hidden states (one per policy)
        """
        if device is None:
            device = self.device

        hiddens = []
        for policy in self.policies:
            hidden = policy.init_hidden(batch_size=batch_size, device=device)
            hiddens.append(hidden)

        return hiddens

    def predict(
        self,
        p_tip: np.ndarray,
        p_ref: np.ndarray,
        hiddens: Optional[List] = None
    ) -> Tuple[np.ndarray, np.ndarray, List]:
        """
        Predict control input with uncertainty estimation.

        Args:
            p_tip: np.array [3] - Current tip position
            p_ref: np.array [3] - Reference tip position
            hiddens: List of hidden states (one per policy, optional)

        Returns:
            u_mean: np.array [3] - Mean control input across ensemble
            u_var: np.array [3] - Variance (epistemic uncertainty) per action
            hiddens: List of updated hidden states
        """
        if hiddens is None:
            hiddens = self.init_hidden(batch_size=1, device=self.device)

        if len(hiddens) != self.n_policies:
            raise ValueError(
                f"Expected {self.n_policies} hidden states, got {len(hiddens)}"
            )

        # Collect predictions from all policies
        predictions = []
        new_hiddens = []

        for policy, hidden in zip(self.policies, hiddens):
            u_pred, hidden_new = policy.predict(p_tip, p_ref, hidden)
            predictions.append(u_pred)
            new_hiddens.append(hidden_new)

        # Stack predictions: [n_policies, output_dim]
        predictions = np.stack(predictions, axis=0)

        # Compute ensemble statistics
        u_mean = np.mean(predictions, axis=0)  # [output_dim]
        u_var = np.var(predictions, axis=0)    # [output_dim] (epistemic uncertainty)

        return u_mean, u_var, new_hiddens

    def predict_all(
        self,
        p_tip: np.ndarray,
        p_ref: np.ndarray,
        hiddens: Optional[List] = None
    ) -> Tuple[np.ndarray, List]:
        """
        Get individual predictions from all policies (for analysis).

        Args:
            p_tip: np.array [3] - Current tip position
            p_ref: np.array [3] - Reference tip position
            hiddens: List of hidden states (optional)

        Returns:
            predictions: np.array [n_policies, output_dim] - All predictions
            hiddens: List of updated hidden states
        """
        if hiddens is None:
            hiddens = self.init_hidden(batch_size=1, device=self.device)

        predictions = []
        new_hiddens = []

        for policy, hidden in zip(self.policies, hiddens):
            u_pred, hidden_new = policy.predict(p_tip, p_ref, hidden)
            predictions.append(u_pred)
            new_hiddens.append(hidden_new)

        predictions = np.stack(predictions, axis=0)
        return predictions, new_hiddens

    def get_uncertainty_stats(self, u_var: np.ndarray) -> dict:
        """
        Compute uncertainty statistics for logging/analysis.

        Args:
            u_var: np.array [output_dim] - Variance per action

        Returns:
            dict with uncertainty metrics
        """
        return {
            'max_variance': float(np.max(u_var)),
            'mean_variance': float(np.mean(u_var)),
            'total_variance': float(np.sum(u_var)),
            'std_dev': np.sqrt(u_var).tolist()  # Per-action std dev
        }

    def should_query_expert(
        self,
        u_var: np.ndarray,
        threshold: float = 0.001
    ) -> bool:
        """
        Decision rule: query expert if uncertainty exceeds threshold.

        Args:
            u_var: np.array [output_dim] - Variance per action
            threshold: Variance threshold (default: 0.001)

        Returns:
            bool: True if should query expert, False otherwise
        """
        return np.max(u_var) > threshold

    def __len__(self):
        """Return number of policies in ensemble."""
        return self.n_policies

    def __repr__(self):
        """String representation."""
        return (
            f"EnsemblePolicy(n_policies={self.n_policies}, "
            f"hidden_dim={self.hidden_dim}, device={self.device})"
        )


class EnsembleMetrics:
    """
    Track ensemble performance and uncertainty-error correlation.

    Helps answer: "Does high uncertainty correlate with high error?"
    """

    def __init__(self):
        self.uncertainties = []  # Max variance per timestep
        self.errors = []         # Tracking error per timestep
        self.u_means = []        # Mean predictions
        self.u_vars = []         # Variances

    def add(
        self,
        u_mean: np.ndarray,
        u_var: np.ndarray,
        p_tip: np.ndarray,
        p_ref: np.ndarray
    ):
        """
        Record timestep data.

        Args:
            u_mean: np.array [3] - Mean control
            u_var: np.array [3] - Variance per action
            p_tip: np.array [3] - Current tip position
            p_ref: np.array [3] - Reference position
        """
        uncertainty = np.max(u_var)
        error = np.linalg.norm(p_tip - p_ref)

        self.uncertainties.append(float(uncertainty))
        self.errors.append(float(error))
        self.u_means.append(u_mean.copy())
        self.u_vars.append(u_var.copy())

    def compute_correlation(self) -> float:
        """
        Compute Pearson correlation between uncertainty and error.

        Returns:
            float: Correlation coefficient in [-1, 1]
                   > 0.5: Good (uncertainty predicts error)
                   < 0.3: Poor (uncertainty uninformative)
        """
        if len(self.uncertainties) < 2:
            return 0.0

        uncertainties = np.array(self.uncertainties)
        errors = np.array(self.errors)

        # Pearson correlation
        corr = np.corrcoef(uncertainties, errors)[0, 1]
        return float(corr)

    def get_summary(self) -> dict:
        """
        Get summary statistics.

        Returns:
            dict with metrics
        """
        if len(self.uncertainties) == 0:
            return {
                'n_samples': 0,
                'mean_uncertainty': 0.0,
                'mean_error': 0.0,
                'correlation': 0.0
            }

        uncertainties = np.array(self.uncertainties)
        errors = np.array(self.errors)

        return {
            'n_samples': len(self.uncertainties),
            'mean_uncertainty': float(np.mean(uncertainties)),
            'std_uncertainty': float(np.std(uncertainties)),
            'max_uncertainty': float(np.max(uncertainties)),
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'max_error': float(np.max(errors)),
            'correlation': self.compute_correlation()
        }
