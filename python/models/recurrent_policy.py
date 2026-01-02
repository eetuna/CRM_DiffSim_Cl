#!/usr/bin/env python3
"""
CP4.4a: Recurrent Policies for Catheter Control

Implements GRU and LSTM-based policies that maintain temporal context
for improved tracking performance.

Key features:
- Same input/output interface as CP4.1 BehaviorCloningPolicy
- Hidden state management for sequential rollouts
- Compatible with DAgger training infrastructure
"""

import numpy as np
import torch
import torch.nn as nn


class GRUPolicy(nn.Module):
    """
    GRU-based recurrent policy for catheter control.

    Architecture:
        Input: [p_tip (3), p_ref (3)] → 6D
        GRU: hidden_dim (default: 64)
        Output: u (3)

    The policy maintains a hidden state across timesteps, allowing it to
    model temporal dependencies and velocity information.
    """

    def __init__(self, input_dim=6, hidden_dim=64, output_dim=3, num_layers=1):
        """
        Args:
            input_dim: Input dimension (default: 6 = p_tip + p_ref)
            hidden_dim: GRU hidden dimension (default: 64)
            output_dim: Output dimension (default: 3 = control currents)
            num_layers: Number of GRU layers (default: 1)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # GRU layer
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'gru' in name:
                    nn.init.xavier_uniform_(param)
                elif 'fc' in name:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, p_tip, p_ref, hidden=None):
        """
        Forward pass through the GRU policy.

        Args:
            p_tip: Tensor [batch, 3] or [batch, seq_len, 3] - Current tip position
            p_ref: Tensor [batch, 3] or [batch, seq_len, 3] - Reference tip position
            hidden: Tensor [num_layers, batch, hidden_dim] - Hidden state (optional)

        Returns:
            u: Tensor [batch, 3] or [batch, seq_len, 3] - Predicted control inputs
            hidden: Tensor [num_layers, batch, hidden_dim] - Updated hidden state
        """
        # Concatenate inputs
        x = torch.cat([p_tip, p_ref], dim=-1)  # [batch, (seq_len,) 6]

        # Ensure 3D input for GRU: [batch, seq_len, input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, 6]
            squeeze_output = True
        else:
            squeeze_output = False

        # GRU forward
        gru_out, hidden = self.gru(x, hidden)  # gru_out: [batch, seq_len, hidden_dim]

        # Output layer
        u = self.fc(gru_out)  # [batch, seq_len, 3]

        # Squeeze if input was 2D
        if squeeze_output:
            u = u.squeeze(1)  # [batch, 3]

        return u, hidden

    def init_hidden(self, batch_size=1, device='cpu'):
        """
        Initialize hidden state to zeros.

        Args:
            batch_size: Batch size (default: 1)
            device: Device for tensor (default: 'cpu')

        Returns:
            hidden: Tensor [num_layers, batch_size, hidden_dim]
        """
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    def predict(self, p_tip, p_ref, hidden=None):
        """
        Predict control input (no gradient) for single timestep.

        Args:
            p_tip: np.array [3] - Current tip position
            p_ref: np.array [3] - Reference tip position
            hidden: Tensor [num_layers, 1, hidden_dim] - Hidden state (optional)

        Returns:
            u: np.array [3] - Predicted control input
            hidden: Tensor [num_layers, 1, hidden_dim] - Updated hidden state
        """
        self.eval()
        with torch.no_grad():
            p_tip_t = torch.from_numpy(p_tip).float().unsqueeze(0)  # [1, 3]
            p_ref_t = torch.from_numpy(p_ref).float().unsqueeze(0)  # [1, 3]

            # Initialize hidden if not provided
            if hidden is None:
                hidden = self.init_hidden(batch_size=1, device=p_tip_t.device)

            u_t, hidden = self.forward(p_tip_t, p_ref_t, hidden)
            return u_t.squeeze(0).numpy(), hidden


class LSTMPolicy(nn.Module):
    """
    LSTM-based recurrent policy for catheter control.

    Architecture:
        Input: [p_tip (3), p_ref (3)] → 6D
        LSTM: hidden_dim (default: 64)
        Output: u (3)

    Similar to GRUPolicy but uses LSTM cells which maintain both
    hidden state (h) and cell state (c).
    """

    def __init__(self, input_dim=6, hidden_dim=64, output_dim=3, num_layers=1):
        """
        Args:
            input_dim: Input dimension (default: 6 = p_tip + p_ref)
            hidden_dim: LSTM hidden dimension (default: 64)
            output_dim: Output dimension (default: 3 = control currents)
            num_layers: Number of LSTM layers (default: 1)
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        # Output layer
        self.fc = nn.Linear(hidden_dim, output_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                elif 'fc' in name:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, p_tip, p_ref, hidden=None):
        """
        Forward pass through the LSTM policy.

        Args:
            p_tip: Tensor [batch, 3] or [batch, seq_len, 3] - Current tip position
            p_ref: Tensor [batch, 3] or [batch, seq_len, 3] - Reference tip position
            hidden: Tuple of (h, c) tensors [num_layers, batch, hidden_dim] (optional)

        Returns:
            u: Tensor [batch, 3] or [batch, seq_len, 3] - Predicted control inputs
            hidden: Tuple of (h, c) tensors [num_layers, batch, hidden_dim]
        """
        # Concatenate inputs
        x = torch.cat([p_tip, p_ref], dim=-1)  # [batch, (seq_len,) 6]

        # Ensure 3D input for LSTM: [batch, seq_len, input_dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, 6]
            squeeze_output = True
        else:
            squeeze_output = False

        # LSTM forward
        lstm_out, hidden = self.lstm(x, hidden)  # lstm_out: [batch, seq_len, hidden_dim]

        # Output layer
        u = self.fc(lstm_out)  # [batch, seq_len, 3]

        # Squeeze if input was 2D
        if squeeze_output:
            u = u.squeeze(1)  # [batch, 3]

        return u, hidden

    def init_hidden(self, batch_size=1, device='cpu'):
        """
        Initialize hidden state (h, c) to zeros.

        Args:
            batch_size: Batch size (default: 1)
            device: Device for tensor (default: 'cpu')

        Returns:
            hidden: Tuple of (h, c) tensors [num_layers, batch_size, hidden_dim]
        """
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)

    def predict(self, p_tip, p_ref, hidden=None):
        """
        Predict control input (no gradient) for single timestep.

        Args:
            p_tip: np.array [3] - Current tip position
            p_ref: np.array [3] - Reference tip position
            hidden: Tuple of (h, c) tensors [num_layers, 1, hidden_dim] (optional)

        Returns:
            u: np.array [3] - Predicted control input
            hidden: Tuple of (h, c) tensors [num_layers, 1, hidden_dim]
        """
        self.eval()
        with torch.no_grad():
            p_tip_t = torch.from_numpy(p_tip).float().unsqueeze(0)  # [1, 3]
            p_ref_t = torch.from_numpy(p_ref).float().unsqueeze(0)  # [1, 3]

            # Initialize hidden if not provided
            if hidden is None:
                hidden = self.init_hidden(batch_size=1, device=p_tip_t.device)

            u_t, hidden = self.forward(p_tip_t, p_ref_t, hidden)
            return u_t.squeeze(0).numpy(), hidden
