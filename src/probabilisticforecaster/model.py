"""Transformer model architecture with causal masking and probabilistic output heads.

Implements a causal Transformer encoder that produces Gaussian distribution
parameters (μ̂, σ̂) for forward return prediction on 5-minute forex bars.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from probabilisticforecaster.config import ForecasterConfig


class TransformerLayer(nn.Module):
    """Single Transformer encoder layer with custom QKV projections and causal masking.

    Uses Linear→Tanh→Linear projections for Q, K, V to bound attention logits
    and prevent gradient explosion under NLL loss training.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # QKV projections: Linear → Tanh → Linear for gradient stability
        self.q_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )
        self.k_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )
        self.v_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, d_model),
        )

        # Output projection after multi-head attention
        self.out_proj = nn.Linear(d_model, d_model)

        # Layer normalization after attention and feed-forward
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network: Linear → ReLU → Linear
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through one Transformer layer.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
            mask: Causal mask of shape (seq_len, seq_len), True where attention
                  should be blocked (upper triangle).

        Returns:
            Output tensor of shape (batch, seq_len, d_model).
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V with bounded projections
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)  # (batch, seq_len, d_model)
        v = self.v_proj(x)  # (batch, seq_len, d_model)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch, num_heads, seq_len, head_dim)

        # Scaled dot-product attention with causal mask
        scale = self.head_dim**0.5
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale
        # Shape: (batch, num_heads, seq_len, seq_len)

        # Apply causal mask: fill future positions with -inf
        attn_weights = attn_weights.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        attn_output = torch.matmul(attn_weights, v)
        # Shape: (batch, num_heads, seq_len, head_dim)

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        # Residual connection + LayerNorm after attention
        x = self.norm1(x + attn_output)

        # Feed-forward network with residual connection + LayerNorm
        ff_output = self.ffn(x)
        ff_output = self.dropout(ff_output)
        x = self.norm2(x + ff_output)

        return x


class ProbabilisticTransformer(nn.Module):
    """Transformer encoder with causal mask and probabilistic output heads.

    Input:  (batch, seq_len, num_features), shape (B, 36, 16)
    Output: (mu, sigma) each of shape (B, seq_len, 1)

    The model produces predictions for every position in the sequence
    (sequence-to-sequence), but only the last position is used at inference.
    """

    def __init__(self, config: ForecasterConfig):
        super().__init__()
        self.config = config
        d_model = config.num_features  # 16

        # Stack of Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )

        # μ head: ReLU → Linear(d, 1), unbounded mean prediction
        self.mu_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model, 1),
        )

        # σ head: ReLU → Linear(d, 1) → Softplus, strictly positive std dev
        self.sigma_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the Transformer model.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features).

        Returns:
            Tuple of (mu, sigma) tensors, each of shape (batch, seq_len, 1).

        Raises:
            ValueError: If input feature dimension does not equal 16.
        """
        if x.shape[-1] != self.config.num_features:
            raise ValueError(
                f"Input feature dimension {x.shape[-1]} != expected {self.config.num_features}"
            )

        seq_len = x.shape[1]

        # Create causal mask: upper-triangular, True where attention is blocked
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()

        # Pass through stacked Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Probabilistic output heads
        mu = self.mu_head(x)  # (batch, seq_len, 1)
        sigma = self.sigma_head(x)  # (batch, seq_len, 1)

        return mu, sigma
