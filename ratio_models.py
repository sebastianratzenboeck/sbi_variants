from __future__ import annotations

import torch
import torch.nn as nn


def _mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    *,
    depth: int = 3,
    dropout: float = 0.0,
) -> nn.Sequential:
    if depth < 2:
        raise ValueError(f"depth must be >=2, got {depth}")
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    for _ in range(depth - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class ConditionalRatioEstimator(nn.Module):
    """Classifier-based ratio estimator for NRE/AMNRE.

    The network predicts logits for:
      - joint pairs (theta, x) ~ p(theta, x)
      - marginal pairs (theta', x) ~ p(theta)p(x)

    The logit approximates log r(x|theta) where r is the likelihood-to-evidence ratio.
    """

    def __init__(
        self,
        *,
        encoder: nn.Module,
        theta_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        use_mask_condition: bool = False,
    ):
        super().__init__()
        self.encoder = encoder
        self.theta_dim = int(theta_dim)
        self.use_mask_condition = bool(use_mask_condition)

        theta_in_dim = self.theta_dim * (2 if self.use_mask_condition else 1)
        self.classifier = _mlp(
            in_dim=int(encoder.output_dim) + theta_in_dim,
            hidden_dim=hidden_dim,
            out_dim=1,
            depth=3,
            dropout=dropout,
        )

    def _build_theta_features(
        self,
        theta: torch.Tensor,  # (B, D)
        mask: torch.Tensor | None = None,  # (B, D)
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(theta)
        theta_masked = theta * mask
        if self.use_mask_condition:
            return torch.cat([theta_masked, mask], dim=-1)
        return theta_masked

    def logits_from_context(
        self,
        theta: torch.Tensor,    # (B, D)
        ctx: torch.Tensor,      # (B, C)
        mask: torch.Tensor | None = None,  # (B, D)
    ) -> torch.Tensor:
        theta_feat = self._build_theta_features(theta, mask=mask)
        inp = torch.cat([ctx, theta_feat], dim=-1)
        return self.classifier(inp).squeeze(-1)

    def encode_context(
        self,
        values: torch.Tensor,
        errors: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.encoder(values, errors, observed_mask)

    def log_ratio(
        self,
        theta: torch.Tensor,
        values: torch.Tensor,
        errors: torch.Tensor,
        observed_mask: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        ctx = self.encode_context(values, errors, observed_mask)
        return self.logits_from_context(theta, ctx, mask=mask)

