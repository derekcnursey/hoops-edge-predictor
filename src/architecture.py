"""MLP model architectures for spread regression and win probability classification.

MLPRegressor: predicts (mu, log_sigma) for a Gaussian distribution over the home spread.
MLPClassifier: predicts home win probability via logit.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """MLP that outputs (mu, log_sigma) for Gaussian NLL loss.

    Architecture:
        input_dim → hidden1 → BN → ReLU → Dropout →
        hidden2 → BN → ReLU → Linear(2) [mu, log_sigma]
    """

    def __init__(
        self,
        input_dim: int = 37,
        hidden1: int = 256,
        hidden2: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden2, 2)  # [mu, log_sigma]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, raw_sigma) where raw_sigma needs softplus + clamp."""
        h = self.net(x)
        out = self.head(h)
        mu = out[:, 0]
        log_sigma = out[:, 1]
        return mu, log_sigma


class MLPClassifier(nn.Module):
    """MLP that outputs a logit for BCEWithLogitsLoss.

    Architecture:
        input_dim → hidden1 → BN → ReLU → Dropout →
        hidden1//2 → BN → ReLU → Linear(1) [logit]
    """

    def __init__(
        self,
        input_dim: int = 37,
        hidden1: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()
        hidden2 = hidden1 // 2
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logit (scalar per sample)."""
        h = self.net(x)
        return self.head(h).squeeze(-1)


def gaussian_nll_loss(
    mu: torch.Tensor,
    raw_sigma: torch.Tensor,
    target: torch.Tensor,
    sigma_min: float = 0.5,
    sigma_max: float = 30.0,
    eps: float = 1e-3,
) -> torch.Tensor:
    """Gaussian negative log-likelihood loss with clamped sigma.

    sigma = softplus(raw_sigma) + eps, then clamped to [sigma_min, sigma_max].
    """
    sigma = torch.nn.functional.softplus(raw_sigma) + eps
    sigma = sigma.clamp(min=sigma_min, max=sigma_max)
    nll = 0.5 * torch.log(2 * torch.pi * sigma**2) + (target - mu) ** 2 / (2 * sigma**2)
    return nll.mean()
