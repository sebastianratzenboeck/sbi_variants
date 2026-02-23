import torch
import torch.nn as nn


class ValueEmbed(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.linear = nn.Linear(1, embed_dim)

    def forward(self, x):
        return self.linear(x)


class ValueCalibratorScalar(nn.Module):
    """Node-specific scalar FiLM calibration for value embeddings.

    For node i:
      h_i' = gamma_i * h_i + beta_i
    where gamma_i and beta_i are learned per-node scalars.
    """

    def __init__(self, num_nodes):
        super().__init__()
        self.gamma = nn.Embedding(num_nodes, 1)
        self.beta = nn.Embedding(num_nodes, 1)
        nn.init.ones_(self.gamma.weight)   # start as identity
        nn.init.zeros_(self.beta.weight)   # start with no shift

    def forward(self, value_emb, node_ids):
        # value_emb: (B, N, D), node_ids: (B, N)
        g = self.gamma(node_ids).to(value_emb.dtype)  # (B, N, 1)
        b = self.beta(node_ids).to(value_emb.dtype)   # (B, N, 1)
        return g * value_emb + b


class ErrorEmbed(nn.Module):
    """
    Embed measurement errors using Fourier features for robust scale handling.
    """

    def __init__(self, embed_dim, fourier_dim=128, scale=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        # Random Fourier features (fixed, not learned)
        self.register_buffer('B', torch.randn(fourier_dim // 2) * scale * 2 * torch.pi)
        # Project Fourier features to desired embedding dimension
        self.projection = nn.Linear(fourier_dim, embed_dim)

    def forward(self, errors):
        # errors: (B, N, 1) — expected to be standardized log-errors with sentinels
        e = errors.squeeze(-1)  # (B, N)
        # Safety: map any stray NaN → unobserved sentinel before math
        e = torch.nan_to_num(e, nan=5.0)
        # Fourier features
        e_proj = e.unsqueeze(-1) * self.B  # (B, N, fourier_dim//2)
        e_fourier = torch.cat([torch.sin(e_proj), torch.cos(e_proj)], dim=-1)  # (B, N, fourier_dim)
        # Project to embedding dimension
        e_embed = self.projection(e_fourier)  # (B, N, embed_dim)
        return e_embed


class ErrorEmbedMLP(nn.Module):
    """Embed standardized log-errors with explicit regime handling.

    Uses three regimes:
      - real measurement (continuous z branch)
      - perfect sentinel (z <= -4.9)
      - unobserved sentinel (z >= +4.9)
    """

    def __init__(self, embed_dim, hidden_dim=64):
        super().__init__()
        self.embed_dim = embed_dim
        self.regime_embedding = nn.Embedding(3, embed_dim)  # 0=real, 1=perfect, 2=unobs
        self.real_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, errors):
        # errors: (B, N, 1) standardized log-errors with sentinels
        e = errors.squeeze(-1)  # (B, N)
        e = torch.nan_to_num(e, nan=5.0)

        # Sentinel thresholds follow normalize_errors() convention.
        is_perfect = (e <= -4.9)
        is_unobs = (e >= 4.9)
        is_real = ~(is_perfect | is_unobs)

        regime_id = is_perfect.long() + 2 * is_unobs.long()
        regime_emb = self.regime_embedding(regime_id)  # (B, N, D)

        z_real = torch.where(is_real, e, torch.zeros_like(e))
        feats = torch.stack(
            [z_real, z_real * z_real, torch.log1p(torch.abs(z_real))],
            dim=-1,
        )  # (B, N, 3)
        real_emb = self.real_mlp(feats)  # (B, N, D)

        return regime_emb + is_real.to(regime_emb.dtype).unsqueeze(-1) * real_emb


class ConditionEmbed(nn.Module):
    """
    Two-state embedding for condition status:
      0 = not conditioned
      1 = conditioned
    """
    def __init__(self, dim_condition):
        super().__init__()
        # Shape: (2, dim_condition), row 0=off, row 1=on.
        table = torch.randn(2, dim_condition) * 0.5
        table[0].zero_()  # start close to previous behavior (off ~= zero)
        self.condition_embedding = nn.Parameter(table)

    def forward(self, condition_mask):
        # condition_mask: (B, N) or (B, N, 1)
        if condition_mask.dim() == 3:
            condition_mask = condition_mask.squeeze(-1)
        # Robust against float masks from clipping/multiplication.
        state = (condition_mask > 0.5).long().clamp_(0, 1)  # (B, N)
        return self.condition_embedding[state]  # (B, N, C)


class ObservedEmbed(nn.Module):
    """
    Embed whether a variable was actually observed/measured.
    Two-state embedding:
      0 = unobserved
      1 = observed
    """
    def __init__(self, dim_observed):
        super().__init__()
        self.embed_dim = dim_observed
        table = torch.randn(2, dim_observed) * 0.5
        table[0].zero_()  # start close to previous behavior (off ~= zero)
        self.observed_embedding = nn.Parameter(table)

    def forward(self, observed_mask):
        # observed_mask: (B, N) or (B, N, 1)
        if observed_mask.dim() == 3:
            observed_mask = observed_mask.squeeze(-1)
        state = (observed_mask > 0.5).long().clamp_(0, 1)  # (B, N)
        return self.observed_embedding[state]  # (B, N, D)


class NodeIDEmbed(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, embed_dim)

    def forward(self, node_ids):
        return self.embedding(node_ids)


class TimeEmbed(nn.Module):
    def __init__(self, time_embed_dim, input_dim=1):
        super().__init__()
        assert time_embed_dim % 2 == 0, "time_embed_dim must be even"
        self.B = nn.Parameter(torch.randn(time_embed_dim // 2, input_dim) * 2 * torch.pi)

    def forward(self, t):
        # t shape: (B, 1, 1)
        t = t.squeeze(-1)  # shape: (B, 1)
        proj = 2 * torch.pi * t @ self.B.T  # (B, time_embed_dim//2 + 1)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (B, time_embed_dim)


class MissingnessContextEncoder(nn.Module):
    """Encode pattern-level missingness/error statistics into one global token."""

    def __init__(
        self,
        out_dim,
        obs_start_idx,
        survey_obs_groups=None,
        hidden_dim=64,
    ):
        super().__init__()
        self.obs_start_idx = int(obs_start_idx)
        self.survey_obs_groups = [list(g) for g in (survey_obs_groups or [])]
        self.log_err_perfect = -4.9
        self.log_err_unobs = 4.9

        # Features:
        #   frac_obs_total
        #   frac_obs_by_survey (one per group)
        #   mean_real_err, std_real_err, frac_perfect, frac_unobs
        feat_dim = 1 + len(self.survey_obs_groups) + 4
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x, observed_mask=None, errors=None):
        # x: (B, N, 1)
        B, N, _ = x.shape
        dev = x.device
        dt = x.dtype

        if observed_mask is None:
            obs = torch.ones(B, N, device=dev, dtype=dt)
        else:
            obs = observed_mask.squeeze(-1) if observed_mask.dim() == 3 else observed_mask
            obs = (obs > 0.5).to(device=dev, dtype=dt)

        obs_block = obs[:, self.obs_start_idx:] if self.obs_start_idx < N else obs[:, 0:0]
        if obs_block.shape[1] > 0:
            frac_obs_total = obs_block.mean(dim=1, keepdim=True)
        else:
            frac_obs_total = torch.zeros(B, 1, device=dev, dtype=dt)

        survey_fracs = []
        for group in self.survey_obs_groups:
            if not group:
                survey_fracs.append(torch.zeros(B, 1, device=dev, dtype=dt))
                continue
            idx = torch.tensor(group, dtype=torch.long, device=dev)
            frac = obs.index_select(1, idx).mean(dim=1, keepdim=True)
            survey_fracs.append(frac)

        if errors is None:
            mean_real = torch.zeros(B, 1, device=dev, dtype=dt)
            std_real = torch.zeros(B, 1, device=dev, dtype=dt)
            frac_perfect = torch.zeros(B, 1, device=dev, dtype=dt)
            frac_unobs = torch.zeros(B, 1, device=dev, dtype=dt)
        else:
            e = errors.squeeze(-1) if errors.dim() == 3 else errors
            e = e.to(device=dev, dtype=dt)
            e_block = e[:, self.obs_start_idx:] if self.obs_start_idx < N else e[:, 0:0]
            if e_block.shape[1] == 0:
                mean_real = torch.zeros(B, 1, device=dev, dtype=dt)
                std_real = torch.zeros(B, 1, device=dev, dtype=dt)
                frac_perfect = torch.zeros(B, 1, device=dev, dtype=dt)
                frac_unobs = torch.zeros(B, 1, device=dev, dtype=dt)
            else:
                is_real = (
                    (e_block > self.log_err_perfect)
                    & (e_block < self.log_err_unobs)
                    & (obs_block > 0.5)
                )
                is_perfect = (e_block <= self.log_err_perfect) & (obs_block > 0.5)
                is_unobs = (e_block >= self.log_err_unobs) | (obs_block <= 0.5)

                real_count = is_real.sum(dim=1, keepdim=True).clamp_min(1).to(dt)
                real_vals = torch.where(is_real, e_block, torch.zeros_like(e_block))
                mean_real = real_vals.sum(dim=1, keepdim=True) / real_count
                var_real = torch.where(
                    is_real,
                    (e_block - mean_real).pow(2),
                    torch.zeros_like(e_block),
                ).sum(dim=1, keepdim=True) / real_count
                std_real = torch.sqrt(var_real + 1e-8)

                denom = float(e_block.shape[1])
                frac_perfect = is_perfect.to(dt).sum(dim=1, keepdim=True) / denom
                frac_unobs = is_unobs.to(dt).sum(dim=1, keepdim=True) / denom

        feat_parts = [frac_obs_total] + survey_fracs + [mean_real, std_real, frac_perfect, frac_unobs]
        feats = torch.cat(feat_parts, dim=-1)  # (B, F)
        token = self.mlp(feats).unsqueeze(1)  # (B, 1, out_dim)
        return token

class Tokenizer(nn.Module):
    def __init__(
            self,
            dim_value,
            dim_id,
            dim_condition,
            attn_embed_dim,
            num_nodes,
            value_calibration_type="none",  # "none" | "scalar_film"
            dim_error=None,  # dimension for error embedding
            use_error_embedding=True,  # whether to use error embeddings
            error_embed_type="rff",  # "rff" or "mlp_regime"
            dim_observed=None,  # dimension for observed embedding
            use_observed_embedding=True,  # whether to use observed embeddings
    ):
        super().__init__()
        self.use_error_embedding = use_error_embedding
        self.error_embed_type = error_embed_type
        self.use_observed_embedding = use_observed_embedding
        self.value_calibration_type = value_calibration_type

        self.value_embed = ValueEmbed(dim_value)
        self.id_embed = NodeIDEmbed(num_nodes, dim_id)
        self.cond_embed = ConditionEmbed(dim_condition)
        if value_calibration_type == "none":
            self.value_calibrator = None
        elif value_calibration_type == "scalar_film":
            self.value_calibrator = ValueCalibratorScalar(num_nodes)
        else:
            raise ValueError(
                f"Unsupported value_calibration_type '{value_calibration_type}'. "
                "Use one of: 'none', 'scalar_film'."
            )

        total_dim = dim_value + dim_id + dim_condition

        # Error embedding
        if use_error_embedding and (dim_error is not None):
            if error_embed_type == "rff":
                self.error_embed = ErrorEmbed(dim_error, fourier_dim=128, scale=1.0)
            elif error_embed_type == "mlp_regime":
                self.error_embed = ErrorEmbedMLP(dim_error)
            else:
                raise ValueError(
                    f"Unsupported error_embed_type '{error_embed_type}'. "
                    "Use one of: 'rff', 'mlp_regime'."
                )
            total_dim += dim_error

        # Observed embedding
        if use_observed_embedding and (dim_observed is not None):
            self.observed_embed = ObservedEmbed(dim_observed)
            total_dim += dim_observed

        # Linear projection to the attention embedding dimension
        self.output_proj = nn.Linear(total_dim, attn_embed_dim)

    def forward(self, x, node_ids, condition_mask, errors=None, observed_mask=None):
        val_emb = self.value_embed(x)  # (B, N, dim_value)
        if self.value_calibrator is not None:
            val_emb = self.value_calibrator(val_emb, node_ids)
        id_emb = self.id_embed(node_ids)  # (B, N, dim_id)
        cond_emb = self.cond_embed(condition_mask)  # (B, N, dim_condition)
        # Concatenate embeddings
        embeddings = [val_emb, id_emb, cond_emb]

        # Error embedding (if module exists, always include — default to zeros)
        if self.use_error_embedding and hasattr(self, 'error_embed'):
            if errors is not None:
                if errors.dim() == 2:
                    errors = errors.unsqueeze(-1)
                err_emb = self.error_embed(errors)  # (B, N, dim_error)
            else:
                B, N = val_emb.shape[:2]
                err_emb = torch.zeros(B, N, self.error_embed.embed_dim,
                                      device=val_emb.device, dtype=val_emb.dtype)
            embeddings.append(err_emb)

        # Observed embedding (if module exists, always include — default to zeros)
        if self.use_observed_embedding and hasattr(self, 'observed_embed'):
            if observed_mask is not None:
                obs_emb = self.observed_embed(observed_mask)  # (B, N, dim_observed)
            else:
                B, N = val_emb.shape[:2]
                obs_emb = torch.zeros(B, N, self.observed_embed.observed_embedding.shape[-1],
                                      device=val_emb.device, dtype=val_emb.dtype)
            embeddings.append(obs_emb)

        # Concatenate all embeddings
        token = torch.cat(embeddings, dim=-1)  # (B, N, total_dim)
        # Project to attention embedding dimension
        return self.output_proj(token)  # (B, N, attn_embed_dim)


class MaskedMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        # No attention dropout — random edge masks already provide stronger
        # attention regularization; extra dropout here is redundant and creates
        # a train/val asymmetry (dropout disabled during eval).
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.0)

    def forward(self, x, attn_mask):  # attn_mask shape: (B, N, N)
        B, N, _ = x.shape
        if attn_mask is not None:
            # Ensure float mask with -inf where attention is blocked
            # attn_mask --> ~attn_mask, so False becomes True (1) = blocked, True -> False (0) = allowed
            float_mask = (~attn_mask).float() * -1e30  # Now: True = -1e30, False = 0
            # Expand to all attention heads, shape must be: (B * num_heads, N, N)
            float_mask = float_mask.repeat_interleave(self.num_heads, dim=0)
            x_out, _ = self.attn(x, x, x, attn_mask=float_mask)
        else:
            x_out, _ = self.attn(x, x, x)
        return x_out


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, widening_factor=4, dropout_rate=0.1):
        super().__init__()
        self.attn = MaskedMultiheadAttention(embed_dim, num_heads, dropout=dropout_rate)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout_attn = nn.Dropout(dropout_rate)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * widening_factor),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim * widening_factor, embed_dim),
            nn.Dropout(dropout_rate),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, attn_mask, context=None):
        # Broadcast context once; inject into both attention and FFN branches.
        ctx = None
        if context is not None:
            ctx = context
            while ctx.ndim < x.ndim:
                ctx = ctx.unsqueeze(1)

        # Skip connection + attention (explicit per-block time conditioning).
        x_attn = self.norm1(x)
        if ctx is not None:
            x_attn = x_attn + ctx
        x = x + self.dropout_attn(self.attn(x_attn, attn_mask))

        # Skip connection + feedforward (same context injection).
        x_ff = self.norm2(x)
        if ctx is not None:
            x_ff = x_ff + ctx
        x = x + self.ff(x_ff)
        return x


class Simformer(nn.Module):
    def __init__(self,
                 num_nodes,  # Number of nodes (i.e., variables/features) in x
                 # Tokenizer parameters
                 dim_value,      # Dimension of the value embedding
                 dim_id,         # Dimension of the node ID embedding
                 dim_condition,  # Dimension of the condition embedding
                 value_calibration_type="none",  # "none" | "scalar_film"
                 dim_error=None,  # Dimension of error embedding (optional)
                 use_error_embedding=True,  # Whether to use errors
                 error_embed_type="rff",  # "rff" or "mlp_regime"
                 dim_observed=None,  # Dimension of observed embedding (optional)
                 use_observed_embedding=True,  # Whether to use observed mask
                 use_missingness_context=False,  # add one global missingness token
                 obs_start_idx=0,  # start index of observation block
                 survey_obs_groups=None,  # list[list[int]] absolute node indices
                 missingness_context_hidden_dim=64,
                 # Attention embedding dimension
                 attn_embed_dim=64,  # Dimension of the attention embedding
                 num_heads=4,        # Number of attention heads
                 num_layers=3,       # Number of transformer layers
                 widening_factor=4,  # Widening factor for feedforward layers
                 time_embed_dim=32,
                 dropout=0.1):
        super().__init__()
        self.use_error_embedding = use_error_embedding
        self.use_observed_embedding = use_observed_embedding
        self.use_missingness_context = use_missingness_context

        self.tokenizer = Tokenizer(
            dim_value,
            dim_id,
            dim_condition,
            attn_embed_dim,
            num_nodes,
            value_calibration_type=value_calibration_type,
            dim_error=dim_error,
            use_error_embedding=use_error_embedding,
            error_embed_type=error_embed_type,
            dim_observed=dim_observed,
            use_observed_embedding=use_observed_embedding,
        )
        if use_missingness_context:
            self.missingness_context_encoder = MissingnessContextEncoder(
                out_dim=attn_embed_dim,
                obs_start_idx=obs_start_idx,
                survey_obs_groups=survey_obs_groups,
                hidden_dim=missingness_context_hidden_dim,
            )
        else:
            self.missingness_context_encoder = None
        self.time_embed = TimeEmbed(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, attn_embed_dim)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(attn_embed_dim, num_heads, widening_factor=widening_factor, dropout_rate=dropout)
            for _ in range(num_layers)
        ])
        self.output_layer = nn.Linear(attn_embed_dim, 1)

    @staticmethod
    def _augment_edge_mask_with_context(edge_mask, observed_mask, num_nodes):
        """Append one global token to the edge mask and connect it to observed nodes."""
        B = edge_mask.shape[0] if edge_mask is not None else observed_mask.shape[0]
        dev = edge_mask.device if edge_mask is not None else observed_mask.device
        if edge_mask is None:
            edge_mask = torch.ones(B, num_nodes, num_nodes, dtype=torch.bool, device=dev)

        out = torch.zeros(B, num_nodes + 1, num_nodes + 1, dtype=torch.bool, device=edge_mask.device)
        out[:, :num_nodes, :num_nodes] = edge_mask.bool()
        out[:, num_nodes, num_nodes] = True

        if observed_mask is None:
            conn = torch.ones(B, num_nodes, dtype=torch.bool, device=edge_mask.device)
        else:
            obs = observed_mask.squeeze(-1) if observed_mask.dim() == 3 else observed_mask
            conn = obs.to(device=edge_mask.device).bool()

        out[:, num_nodes, :num_nodes] = conn
        out[:, :num_nodes, num_nodes] = conn
        return out

    def forward(self, t, x, node_ids, condition_mask, edge_mask, errors=None, observed_mask=None):
        if condition_mask.dim() == 2:
            condition_mask = condition_mask.unsqueeze(-1)   # (B, N, 1)
        condition_mask = condition_mask.to(x.dtype)         # dtype match

        num_nodes = x.shape[1]
        tokens = self.tokenizer(x, node_ids, condition_mask, errors=errors, observed_mask=observed_mask)
        if self.use_missingness_context and (self.missingness_context_encoder is not None):
            context_token = self.missingness_context_encoder(
                x=x,
                observed_mask=observed_mask,
                errors=errors,
            )
            tokens = torch.cat([tokens, context_token], dim=1)  # (B, N+1, D)
            edge_mask = self._augment_edge_mask_with_context(
                edge_mask=edge_mask,
                observed_mask=observed_mask,
                num_nodes=num_nodes,
            )
        t_context = self.time_embed(t)  # shape: (B, time_embed_dim)
        t_context = self.time_proj(t_context)  # shape: (B, attn_embed_dim)

        for block in self.transformer_blocks:
            # reshape for MultiheadAttention: (M, B, embed_dim)
            tokens = block(tokens, edge_mask, context=t_context)
        # return self.output_layer(tokens)
        # Predict velocity
        v = self.output_layer(tokens[:, :num_nodes, :])  # (B, M, 1)
        # Zero velocity on conditioned coords
        if condition_mask is not None:
            v = v * (1.0 - condition_mask)  # broadcast-safe
        return v
