from __future__ import annotations

import torch
import torch.nn as nn

from columns import OBS_COLS
from sampling import build_inference_edge_mask
from transformer import (
    MissingnessContextEncoder,
    Simformer,
    Tokenizer,
    TransformerBlock,
)


def _build_default_survey_obs_groups(columns: list[str]) -> list[list[int]]:
    groups = {
        "gaia": [],
        "2mass": [],
        "wise": [],
        "ps1": [],
        "decam": [],
    }
    for idx, col in enumerate(columns):
        if col not in OBS_COLS:
            continue
        if col.startswith("GAIA_") or col == "parallax_obs":
            groups["gaia"].append(idx)
        elif col.startswith("2MASS_"):
            groups["2mass"].append(idx)
        elif col.startswith("WISE_"):
            groups["wise"].append(idx)
        elif col.startswith("PS1_"):
            groups["ps1"].append(idx)
        elif col.startswith("CTIO_DECam"):
            groups["decam"].append(idx)
    return [groups["gaia"], groups["2mass"], groups["wise"], groups["ps1"], groups["decam"]]


class ObservationEncoder(nn.Module):
    """Transformer encoder over observed-data tokens for direct p(theta|x_obs)."""

    def __init__(
        self,
        input_columns: list[str],
        dim_value: int = 24,
        dim_id: int = 24,
        value_calibration_type: str = "scalar_film",
        dim_error: int = 16,
        error_embed_type: str = "mlp_regime",
        dim_observed: int = 8,
        attn_embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 4,
        widening_factor: int = 4,
        dropout: float = 0.05,
        use_missingness_context: bool = True,
        missingness_context_hidden_dim: int = 64,
    ):
        super().__init__()
        self.input_columns = [str(c) for c in input_columns]
        self.num_nodes = len(self.input_columns)
        self.output_dim = int(attn_embed_dim)
        self.use_missingness_context = bool(use_missingness_context)

        # Conditioning is intentionally disabled for this SBI variant.
        self.tokenizer = Tokenizer(
            dim_value=dim_value,
            dim_id=dim_id,
            dim_condition=0,
            attn_embed_dim=attn_embed_dim,
            num_nodes=self.num_nodes,
            value_calibration_type=value_calibration_type,
            dim_error=dim_error,
            use_error_embedding=(dim_error is not None and dim_error > 0),
            error_embed_type=error_embed_type,
            dim_observed=dim_observed,
            use_observed_embedding=(dim_observed is not None and dim_observed > 0),
        )

        obs_local_idx = [i for i, c in enumerate(self.input_columns) if c in OBS_COLS]
        self.obs_start_idx = min(obs_local_idx) if obs_local_idx else self.num_nodes
        survey_groups = _build_default_survey_obs_groups(self.input_columns)
        if self.use_missingness_context:
            self.missingness_context_encoder = MissingnessContextEncoder(
                out_dim=attn_embed_dim,
                obs_start_idx=self.obs_start_idx,
                survey_obs_groups=survey_groups,
                hidden_dim=missingness_context_hidden_dim,
            )
        else:
            self.missingness_context_encoder = None

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=attn_embed_dim,
                    num_heads=num_heads,
                    widening_factor=widening_factor,
                    dropout_rate=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(attn_embed_dim)

    def forward(
        self,
        values: torch.Tensor,       # (B, N)
        errors: torch.Tensor,       # (B, N)
        observed_mask: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:
        B, N = values.shape
        if N != self.num_nodes:
            raise ValueError(
                f"Expected {self.num_nodes} input nodes, got {N}. "
                "Check input column layout."
            )

        x = values.unsqueeze(-1)
        e = errors.unsqueeze(-1)
        node_ids = torch.arange(N, device=values.device).unsqueeze(0).expand(B, -1)
        cond_mask = torch.zeros(B, N, 1, device=values.device, dtype=values.dtype)

        tokens = self.tokenizer(
            x,
            node_ids,
            cond_mask,
            errors=e,
            observed_mask=observed_mask,
        )

        edge_mask = build_inference_edge_mask(
            batch_size=B,
            num_nodes=N,
            observed_mask=observed_mask,
            device=values.device,
        )

        if self.use_missingness_context and (self.missingness_context_encoder is not None):
            context_token = self.missingness_context_encoder(
                x=x,
                observed_mask=observed_mask,
                errors=e,
            )
            tokens = torch.cat([tokens, context_token], dim=1)
            edge_mask = Simformer._augment_edge_mask_with_context(
                edge_mask=edge_mask,
                observed_mask=observed_mask,
                num_nodes=N,
            )

        for block in self.blocks:
            tokens = block(tokens, edge_mask, context=None)

        tokens_main = tokens[:, :N, :]
        obs = (observed_mask > 0.5).to(tokens_main.dtype).unsqueeze(-1)
        pooled = (tokens_main * obs).sum(dim=1) / obs.sum(dim=1).clamp_min(1.0)

        if self.use_missingness_context and tokens.shape[1] == N + 1:
            pooled = pooled + tokens[:, N, :]

        return self.out_norm(pooled)
