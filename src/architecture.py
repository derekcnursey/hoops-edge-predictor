"""MLP model architectures for spread regression and win probability classification.

MLPRegressor: shared backbone → (mu, log_sigma) heads.
MLPRegressorSplit: shared first layer → separate mu/sigma pathways.
MLPClassifier: predicts home win probability via logit.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLPRegressor(nn.Module):
    """Shared-backbone MLP that outputs (mu, log_sigma) via exp() parameterization.

    Architecture:
        input_dim → hidden1 → BN → ReLU → Dropout →
        hidden2 → BN → ReLU → mu_head(1) + sigma_head(1)

    Sigma = exp(log_sigma).clamp(0.5, 30.0), initialized near empirical residual std.
    """

    def __init__(
        self,
        input_dim: int = 53,
        hidden1: int = 384,
        hidden2: int = 256,
        dropout: float = 0.2,
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
        self.mu_head = nn.Linear(hidden2, 1)
        self.sigma_head = nn.Linear(hidden2, 1)

        # Initialize sigma bias to log(12) ≈ 2.48 so sigma starts near
        # the empirical residual std (~12 points)
        nn.init.constant_(self.sigma_head.bias, 2.48)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_sigma) — caller applies exp() + clamp for sigma."""
        h = self.net(x)
        mu = self.mu_head(h).squeeze(-1)
        log_sigma = self.sigma_head(h).squeeze(-1)
        return mu, log_sigma


class MLPRegressorSplit(nn.Module):
    """Split-head MLP: shared first layer, then separate mu/sigma pathways.

    This prevents mu gradient updates from killing sigma neurons.

    Architecture:
        input_dim → hidden1 → BN → ReLU → Dropout →
        ├── mu_path:    hidden2 → BN → ReLU → Linear(1) [mu]
        └── sigma_path: hidden2 → BN → ReLU → Linear(1) [log_sigma]
    """

    def __init__(
        self,
        input_dim: int = 53,
        hidden1: int = 384,
        hidden2: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mu_head = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 1),
        )

        # Initialize sigma output bias to log(12) ≈ 2.48
        nn.init.constant_(self.sigma_head[-1].bias, 2.48)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (mu, log_sigma) — caller applies exp() + clamp for sigma."""
        h = self.shared(x)
        mu = self.mu_head(h).squeeze(-1)
        log_sigma = self.sigma_head(h).squeeze(-1)
        return mu, log_sigma


class MLPClassifier(nn.Module):
    """MLP that outputs a logit for BCEWithLogitsLoss.

    Architecture:
        input_dim → hidden1 → BN → ReLU → Dropout →
        hidden1//2 → BN → ReLU → Linear(1) [logit]
    """

    def __init__(
        self,
        input_dim: int = 53,
        hidden1: int = 384,
        dropout: float = 0.2,
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
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    sigma_min: float = 0.5,
    sigma_max: float = 30.0,
) -> torch.Tensor:
    """Gaussian NLL with exp() sigma parameterization.

    sigma = exp(log_sigma).clamp(sigma_min, sigma_max)
    loss = 0.5 * log(2π σ²) + (y - μ)² / (2σ²)
    """
    sigma = torch.exp(log_sigma).clamp(min=sigma_min, max=sigma_max)
    nll = 0.5 * torch.log(2 * torch.pi * sigma**2) + (target - mu) ** 2 / (2 * sigma**2)
    return nll, sigma


def laplacian_nll_loss(
    mu: torch.Tensor,
    log_sigma: torch.Tensor,
    target: torch.Tensor,
    sigma_min: float = 0.5,
    sigma_max: float = 30.0,
) -> torch.Tensor:
    """Laplacian NLL with exp() sigma parameterization.

    sigma = exp(log_sigma).clamp(sigma_min, sigma_max)
    loss = log(2σ) + |y - μ| / σ
    """
    sigma = torch.exp(log_sigma).clamp(min=sigma_min, max=sigma_max)
    nll = torch.log(2 * sigma) + torch.abs(target - mu) / sigma
    return nll, sigma
