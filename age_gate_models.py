from __future__ import annotations

import torch
import torch.nn as nn


class AgeGateClassifier(nn.Module):
    def __init__(
        self,
        *,
        encoder: nn.Module,
        num_regimes: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_regimes < 2:
            raise ValueError(f"num_regimes must be >= 2, got {num_regimes}")
        self.encoder = encoder
        self.num_regimes = int(num_regimes)
        layers: list[nn.Module] = [
            nn.Linear(int(encoder.output_dim), int(hidden_dim)),
            nn.SiLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(float(dropout)))
        layers.append(nn.Linear(int(hidden_dim), self.num_regimes))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        values: torch.Tensor,
        errors: torch.Tensor,
        observed_mask: torch.Tensor,
    ) -> torch.Tensor:
        ctx = self.encoder(values, errors, observed_mask)
        return self.classifier(ctx)


@torch.no_grad()
def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    t = max(float(temperature), 1e-6)
    return logits / t
