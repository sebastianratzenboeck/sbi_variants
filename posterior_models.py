from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import TimeEmbed


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, depth: int = 3, dropout: float = 0.0) -> nn.Sequential:
    if depth < 2:
        raise ValueError(f"depth must be >=2, got {depth}")
    layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    for _ in range(depth - 2):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class ConditionalFMPosterior(nn.Module):
    """Conditional flow-matching posterior model for p(theta | x_obs)."""

    def __init__(
        self,
        encoder: nn.Module,
        theta_dim: int,
        hidden_dim: int = 256,
        time_embed_dim: int = 64,
        sigma_min: float = 1e-3,
        time_prior_exponent: float = 0.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if time_prior_exponent <= -1.0:
            raise ValueError(f"time_prior_exponent must be > -1, got {time_prior_exponent}")
        self.encoder = encoder
        self.theta_dim = int(theta_dim)
        self.sigma_min = float(sigma_min)
        self.time_prior_exponent = float(time_prior_exponent)

        self.time_embed = TimeEmbed(time_embed_dim=time_embed_dim)
        self.theta_proj = nn.Linear(theta_dim, hidden_dim)
        self.time_proj = nn.Linear(time_embed_dim, hidden_dim)
        self.ctx_proj = nn.Linear(encoder.output_dim, hidden_dim)
        self.out = _mlp(hidden_dim, hidden_dim, theta_dim, depth=3, dropout=dropout)

    def sample_t(self, batch_size: int, device: torch.device) -> torch.Tensor:
        u = torch.rand(batch_size, device=device)
        return u.pow(1.0 / (1.0 + self.time_prior_exponent))

    def predict_velocity(self, theta_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # theta_t: (B, D), t: (B,), ctx: (B, C)
        t_emb = self.time_embed(t.view(-1, 1, 1))
        h = self.theta_proj(theta_t) + self.time_proj(t_emb) + self.ctx_proj(ctx)
        h = F.silu(h)
        return self.out(h)

    def loss(
        self,
        theta: torch.Tensor,        # (B, D)
        values: torch.Tensor,       # (B, N)
        errors: torch.Tensor,       # (B, N)
        observed_mask: torch.Tensor,  # (B, N)
        sample_weights: torch.Tensor | None = None,  # (B,)
    ) -> torch.Tensor:
        B = theta.shape[0]
        ctx = self.encoder(values, errors, observed_mask)

        x0 = torch.randn_like(theta)
        t = self.sample_t(B, theta.device)
        tt = t.unsqueeze(-1)
        k = (1.0 - self.sigma_min)
        theta_t = (1.0 - k * tt) * x0 + tt * theta

        v_tgt = theta - k * x0
        v_pred = self.predict_velocity(theta_t, t, ctx)
        per_sample = (v_pred - v_tgt).pow(2).mean(dim=1)
        if sample_weights is None:
            return per_sample.mean()
        w = sample_weights.to(per_sample.dtype).reshape(-1)
        denom = w.sum().clamp_min(1e-8)
        return (per_sample * w).sum() / denom

    @torch.no_grad()
    def sample(
        self,
        values: torch.Tensor,        # (B, N)
        errors: torch.Tensor,        # (B, N)
        observed_mask: torch.Tensor,  # (B, N)
        num_samples: int = 256,
        steps: int = 64,
        t0: float = 0.0,
        t1: float = 1.0,
    ) -> torch.Tensor:
        B = values.shape[0]
        if steps <= 0:
            raise ValueError(f"steps must be >0, got {steps}")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be >0, got {num_samples}")

        ctx = self.encoder(values, errors, observed_mask)  # (B, C)
        ctx = ctx.repeat_interleave(num_samples, dim=0)    # (B*S, C)

        total = B * num_samples
        theta = torch.randn(total, self.theta_dim, device=values.device)
        dt = (t1 - t0) / float(steps)

        was_training = self.training
        self.eval()
        try:
            for i in range(steps):
                t = t0 + i * dt
                t_batch = torch.full((total,), float(t), device=values.device, dtype=values.dtype)
                v = self.predict_velocity(theta, t_batch, ctx)
                theta = theta + dt * v
        finally:
            if was_training:
                self.train()

        return theta.view(B, num_samples, self.theta_dim)


class ConditionalFlowPosterior(nn.Module):
    """Conditional normalizing-flow posterior (package-backed: zuko or nflows)."""

    def __init__(
        self,
        encoder: nn.Module,
        theta_dim: int,
        *,
        backend: str = "zuko",
        flow_family: str = "nsf",
        num_transforms: int = 8,
        hidden_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_transforms < 1:
            raise ValueError(f"num_transforms must be >=1, got {num_transforms}")
        self.encoder = encoder
        self.theta_dim = int(theta_dim)
        self.context_dim = int(encoder.output_dim)
        self.backend = str(backend).lower()
        self.flow_family = str(flow_family).lower()
        self.num_transforms = int(num_transforms)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        self.flow = self._build_flow()

    def _build_flow(self):
        if self.backend == "zuko":
            try:
                import zuko.flows as zf
            except Exception as e:
                raise ImportError(
                    "zuko backend requested but import failed. Install with `pip install zuko`."
                ) from e

            flow_map = {
                "nsf": zf.NSF,
                "maf": zf.MAF,
                "nice": zf.NICE,
            }
            if self.flow_family not in flow_map:
                raise ValueError(
                    f"Unsupported zuko flow_family '{self.flow_family}'. "
                    "Use one of: nsf, maf, nice."
                )
            cls = flow_map[self.flow_family]
            return cls(
                features=self.theta_dim,
                context=self.context_dim,
                transforms=self.num_transforms,
                hidden_features=(self.hidden_dim, self.hidden_dim),
            )

        if self.backend == "nflows":
            try:
                from nflows.distributions.normal import StandardNormal
                from nflows.flows.base import Flow
                from nflows.transforms.autoregressive import (
                    MaskedAffineAutoregressiveTransform,
                    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
                )
                from nflows.transforms.base import CompositeTransform
                from nflows.transforms.permutations import ReversePermutation
            except Exception as e:
                raise ImportError(
                    "nflows backend requested but import failed. Install with `pip install nflows`."
                ) from e

            transforms = []
            for _ in range(self.num_transforms):
                transforms.append(ReversePermutation(features=self.theta_dim))
                if self.flow_family == "maf":
                    transforms.append(
                        MaskedAffineAutoregressiveTransform(
                            features=self.theta_dim,
                            hidden_features=self.hidden_dim,
                            context_features=self.context_dim,
                            num_blocks=2,
                            dropout_probability=self.dropout,
                            use_batch_norm=False,
                        )
                    )
                elif self.flow_family == "nsf":
                    transforms.append(
                        MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                            features=self.theta_dim,
                            hidden_features=self.hidden_dim,
                            context_features=self.context_dim,
                            num_bins=10,
                            tails="linear",
                            tail_bound=5.0,
                            num_blocks=2,
                            dropout_probability=self.dropout,
                            use_batch_norm=False,
                        )
                    )
                else:
                    raise ValueError(
                        f"Unsupported nflows flow_family '{self.flow_family}'. "
                        "Use one of: maf, nsf."
                    )

            transform = CompositeTransform(transforms)
            base = StandardNormal([self.theta_dim])
            return Flow(transform, base)

        raise ValueError(
            f"Unsupported backend '{self.backend}'. Use one of: zuko, nflows."
        )

    def nll(
        self,
        theta: torch.Tensor,        # (B, D)
        values: torch.Tensor,       # (B, N)
        errors: torch.Tensor,       # (B, N)
        observed_mask: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:
        ctx = self.encoder(values, errors, observed_mask)
        if self.backend == "zuko":
            dist = self.flow(ctx)
            return -dist.log_prob(theta)
        if self.backend == "nflows":
            return -self.flow.log_prob(theta, context=ctx)
        raise RuntimeError(f"Unknown backend '{self.backend}'.")

    def loss(
        self,
        theta: torch.Tensor,
        values: torch.Tensor,
        errors: torch.Tensor,
        observed_mask: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        nll = self.nll(theta, values, errors, observed_mask)
        if sample_weights is None:
            return nll.mean()
        w = sample_weights.to(nll.dtype).reshape(-1)
        denom = w.sum().clamp_min(1e-8)
        return (nll * w).sum() / denom

    @torch.no_grad()
    def sample(
        self,
        values: torch.Tensor,        # (B, N)
        errors: torch.Tensor,        # (B, N)
        observed_mask: torch.Tensor,  # (B, N)
        num_samples: int = 256,
    ) -> torch.Tensor:
        B = values.shape[0]
        if num_samples <= 0:
            raise ValueError(f"num_samples must be >0, got {num_samples}")

        ctx = self.encoder(values, errors, observed_mask)  # (B, C)
        if self.backend == "zuko":
            # zuko: sample shape (S, B, D) -> (B, S, D)
            dist = self.flow(ctx)
            s = dist.sample((num_samples,))
            return s.permute(1, 0, 2).contiguous()
        if self.backend == "nflows":
            # nflows: sample shape (B, S, D)
            return self.flow.sample(num_samples, context=ctx)
        raise RuntimeError(f"Unknown backend '{self.backend}'.")
