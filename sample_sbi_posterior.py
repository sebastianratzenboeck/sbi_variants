#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

try:
    from .columns import COLOR_DEFINITIONS, OBS_COLS, OBS_ERR_COLS
    from .inference_utils import NormStats
    from .prepare_data import galactic_to_unitvec
    from .data import column_indices, compute_colors_from_cache, load_cache_arrays
    from .encoder import ObservationEncoder
    from .posterior_models import (
        ConditionalFMPosterior,
        ConditionalFlowPosterior,
        CrossAttentionConditionalFMPosterior,
    )
except ImportError:
    from columns import COLOR_DEFINITIONS, OBS_COLS, OBS_ERR_COLS
    from inference_utils import NormStats
    from prepare_data import galactic_to_unitvec
    from sbi_variants.data import column_indices, compute_colors_from_cache, load_cache_arrays
    from sbi_variants.encoder import ObservationEncoder
    from sbi_variants.posterior_models import (
        ConditionalFMPosterior,
        ConditionalFlowPosterior,
        CrossAttentionConditionalFMPosterior,
    )


OBS_ERROR_FLOOR = 1e-6


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    p = argparse.ArgumentParser(
        description="Sample from direct SBI posterior models in sbi_variants/.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="Optional JSON config file. CLI flags override config values.")
    p.add_argument("--model-dir", type=str, default=None,
                   help="Directory containing posterior config/checkpoint files.")
    p.add_argument("--run-name", type=str, default="sbi_variant",
                   help="Run name used during train_sbi_posterior.py.")

    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--cache-path", type=str, default=None,
                     help="Path to build_arrays_cache(.npz) for cached sampling.")
    src.add_argument("--obs-file", type=str, default=None,
                     help="CSV/Parquet observation file for direct sampling.")

    p.add_argument("--index-file", type=str, default=None,
                   help="Optional .npy row indices used with --cache-path.")
    p.add_argument("--id-column", type=str, default=None,
                   help="Optional ID column from --obs-file to keep in summary.")
    p.add_argument("--max-stars", type=int, default=None,
                   help="Optional max number of stars to sample.")
    p.add_argument("--sample-mode", type=str, default="head", choices=["head", "random"])
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num-samples", type=int, default=512,
                   help="Posterior draws per star.")
    p.add_argument("--steps", type=int, default=128,
                   help="Euler steps for flow-matching sampler (ignored for normalizing-flow methods).")
    p.add_argument("--batch-size", type=int, default=256,
                   help="Number of stars processed per chunk.")
    p.add_argument("--device", type=str, default=None)

    p.add_argument("--output-prefix", type=str, default=None,
                   help="Prefix for output files (default: <model-dir>/posterior_<run-name>).")
    p.add_argument("--denormalize", action=argparse.BooleanOptionalAction, default=True,
                   help="If true, also write denormalized physical-space samples.")

    if pre_args.config:
        with open(pre_args.config) as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            raise ValueError(f"Config at {pre_args.config} must be a JSON object (dict).")
        valid = {a.dest for a in p._actions}
        unknown = sorted(k for k in cfg.keys() if k not in valid)
        if unknown:
            raise ValueError(
                f"Unknown config keys in {pre_args.config}: {unknown[:8]}"
                + ("..." if len(unknown) > 8 else "")
            )
        p.set_defaults(**cfg)

    args = p.parse_args()
    if not args.model_dir:
        p.error("Missing required argument after applying config/CLI: --model-dir")
    if (args.cache_path is None) == (args.obs_file is None):
        p.error("Specify exactly one of --cache-path or --obs-file (via config or CLI).")
    return args


def _load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_state_dict(path: str, device: str) -> dict:
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if state and all(str(k).startswith("_orig_mod.") for k in state.keys()):
        state = {k[len("_orig_mod."):]: v for k, v in state.items()}
    return state


def _build_model_from_config(
    config: dict,
    *,
    input_columns_override: Sequence[str] | None = None,
) -> torch.nn.Module:
    input_columns = (
        [str(c) for c in input_columns_override]
        if input_columns_override is not None
        else [str(c) for c in config["input_columns"]]
    )
    theta_columns = [str(c) for c in config["theta_columns"]]
    encoder = ObservationEncoder(
        input_columns=input_columns,
        dim_value=int(config.get("dim_value", 24)),
        dim_id=int(config.get("dim_id", 24)),
        value_calibration_type=str(config.get("value_calibration_type", "scalar_film")),
        dim_error=int(config.get("dim_error", 16)),
        error_embed_type=str(config.get("error_embed_type", "mlp_regime")),
        dim_observed=int(config.get("dim_observed", 8)),
        attn_embed_dim=int(config.get("attn_embed_dim", 128)),
        num_heads=int(config.get("num_heads", 8)),
        num_layers=int(config.get("num_layers", 4)),
        widening_factor=int(config.get("widening_factor", 4)),
        dropout=float(config.get("dropout", 0.05)),
        use_missingness_context=bool(config.get("use_missingness_context", False)),
        missingness_context_hidden_dim=int(config.get("missingness_context_hidden_dim", 64)),
        pooling_mode=str(config.get("pooling_mode", "mean")),
    )
    method = str(config.get("method", "flow_matching"))
    if method == "flow_matching":
        return ConditionalFMPosterior(
            encoder=encoder,
            theta_dim=len(theta_columns),
            hidden_dim=int(config.get("fm_hidden_dim", 256)),
            time_embed_dim=int(config.get("time_embed_dim", 64)),
            sigma_min=float(config.get("sigma_min", 1e-3)),
            time_prior_exponent=float(config.get("time_prior_exponent", 0.0)),
            dropout=float(config.get("dropout", 0.05)),
        )
    if method == "flow_matching_xattn":
        return CrossAttentionConditionalFMPosterior(
            encoder=encoder,
            theta_dim=len(theta_columns),
            hidden_dim=int(config.get("fm_hidden_dim", 256)),
            time_embed_dim=int(config.get("time_embed_dim", 64)),
            sigma_min=float(config.get("sigma_min", 1e-3)),
            time_prior_exponent=float(config.get("time_prior_exponent", 0.0)),
            dropout=float(config.get("dropout", 0.05)),
            num_heads=int(config.get("xattn_num_heads", 4)),
        )
    if method in ("realnvp", "normalizing_flow"):
        return ConditionalFlowPosterior(
            encoder=encoder,
            theta_dim=len(theta_columns),
            backend=str(config.get("nf_backend", "zuko")),
            flow_family=str(config.get("nf_family", "nsf")),
            num_transforms=int(config.get("nf_num_coupling_layers", 8)),
            hidden_dim=int(config.get("nf_hidden_dim", 256)),
            dropout=float(config.get("dropout", 0.05)),
        )
    raise ValueError(f"Unsupported method '{method}' in posterior config.")


def _resolve_color_definitions(
    color_names: Sequence[str],
) -> list[tuple[str, str, str]]:
    by_name = {name: (name, m1, m2) for (name, m1, m2) in COLOR_DEFINITIONS}
    missing = [str(n) for n in color_names if str(n) not in by_name]
    if missing:
        raise ValueError(
            "Unknown color definitions requested by checkpoint metadata: "
            f"{missing[:5]}" + ("..." if len(missing) > 5 else "")
        )
    return [by_name[str(n)] for n in color_names]


def _resolve_input_layout(
    config: dict,
    norm_stats: NormStats,
) -> tuple[list[str], list[str], bool, list[str], np.ndarray | None, np.ndarray | None]:
    base_source = config.get(
        "input_columns_base",
        norm_stats.input_columns_base if norm_stats.input_columns_base is not None else config["input_columns"],
    )
    base_cols = [str(c) for c in base_source]
    use_colors = bool(config.get("use_colors", False) or norm_stats.use_colors)
    model_source = config.get(
        "input_columns_with_colors",
        norm_stats.input_columns_meta if norm_stats.input_columns_meta is not None else config["input_columns"],
    )
    model_cols = [str(c) for c in model_source]
    if not use_colors:
        return base_cols, base_cols, False, [], None, None

    if len(model_cols) <= len(base_cols):
        # Older configs may not carry input_columns_with_colors; infer from metadata.
        if norm_stats.input_columns_meta is not None and len(norm_stats.input_columns_meta) > len(base_cols):
            model_cols = list(norm_stats.input_columns_meta)
        else:
            color_tail = [str(c) for c in config.get("color_names", [])]
            model_cols = base_cols + color_tail

    color_names = list(norm_stats.color_names) if norm_stats.color_names else [str(c) for c in config.get("color_names", [])]
    if not color_names:
        # Last fallback: derive from model column tail.
        color_names = model_cols[len(base_cols):]
    if not color_names:
        raise ValueError(
            "Checkpoint indicates use_colors=true but no color_names are available in config/metadata."
        )
    if norm_stats.color_means is None or norm_stats.color_stds is None:
        raise ValueError(
            "Checkpoint indicates use_colors=true but color normalization stats are missing "
            "from posterior_norm_meta_*.npz."
        )

    if norm_stats.color_names:
        idx = {n: i for i, n in enumerate(norm_stats.color_names)}
        missing = [n for n in color_names if n not in idx]
        if missing:
            raise ValueError(
                "Color names requested by config are missing from normalization metadata: "
                f"{missing[:5]}" + ("..." if len(missing) > 5 else "")
            )
        color_means = np.asarray([norm_stats.color_means[idx[n]] for n in color_names], dtype=np.float32)
        color_stds = np.asarray([norm_stats.color_stds[idx[n]] for n in color_names], dtype=np.float32)
    else:
        color_means = np.asarray(norm_stats.color_means, dtype=np.float32)
        color_stds = np.asarray(norm_stats.color_stds, dtype=np.float32)
        if color_means.shape[0] != len(color_names) or color_stds.shape[0] != len(color_names):
            raise ValueError(
                "Color normalization stats length does not match resolved color_names."
            )

    expected_model_cols = base_cols + list(color_names)
    if model_cols != expected_model_cols:
        model_cols = expected_model_cols

    return base_cols, model_cols, True, list(color_names), color_means, color_stds


def _select_rows(
    n_rows: int,
    *,
    index_file: str | None,
    max_stars: int | None,
    sample_mode: str,
    seed: int,
) -> np.ndarray:
    if index_file is None:
        idx = np.arange(n_rows, dtype=np.int64)
    else:
        idx = np.load(index_file).astype(np.int64)
    idx = np.unique(idx)
    idx = idx[(idx >= 0) & (idx < n_rows)]
    if idx.size == 0:
        raise ValueError("No valid rows selected.")

    if max_stars is not None and max_stars > 0 and idx.size > max_stars:
        if sample_mode == "head":
            idx = idx[:max_stars]
        else:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(idx, size=max_stars, replace=False).astype(np.int64))
    return idx


def _prepare_from_cache(
    cache_path: str,
    *,
    input_columns: Sequence[str],
    index_file: str | None,
    max_stars: int | None,
    sample_mode: str,
    seed: int,
    use_colors: bool = False,
    color_definitions: Sequence[tuple[str, str, str]] | None = None,
    color_means: np.ndarray | None = None,
    color_stds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = load_cache_arrays(cache_path)
    in_idx = column_indices(cache.columns, input_columns, role="input")
    rows = _select_rows(
        cache.values_norm.shape[0],
        index_file=index_file,
        max_stars=max_stars,
        sample_mode=sample_mode,
        seed=seed,
    )
    values = np.nan_to_num(
        cache.values_norm[rows][:, in_idx], nan=0.0,
    ).astype(np.float32)
    errors = cache.errors_norm[rows][:, in_idx].astype(np.float32)
    observed = cache.observed_mask[rows][:, in_idx].astype(np.float32)

    if use_colors:
        if color_definitions is None:
            raise ValueError("use_colors=True requires color_definitions.")
        if color_means is None or color_stds is None:
            raise ValueError("use_colors=True requires color_means/color_stds.")
        (
            colors_norm_train,
            color_err_norm,
            color_obs,
            _color_names,
            color_means_train,
            color_stds_train,
        ) = compute_colors_from_cache(
            cache=cache,
            row_indices=rows,
            color_definitions=list(color_definitions),
        )
        # Re-normalize with saved training-set stats to match model inputs.
        colors_raw = colors_norm_train * color_stds_train + color_means_train
        denom = np.where(color_stds > 1e-8, color_stds, 1.0)
        colors_norm = (colors_raw - color_means) / denom
        colors_norm[~np.isfinite(colors_norm)] = 0.0

        values = np.concatenate([values, colors_norm.astype(np.float32)], axis=1)
        errors = np.concatenate([errors, color_err_norm.astype(np.float32)], axis=1)
        observed = np.concatenate([observed, color_obs.astype(np.float32)], axis=1)

    star_ids = rows
    return values, errors, observed, star_ids


def _subset_obs_df(
    obs_df: pd.DataFrame,
    *,
    max_stars: int | None,
    sample_mode: str,
    seed: int,
) -> pd.DataFrame:
    if max_stars is None or max_stars <= 0 or len(obs_df) <= max_stars:
        return obs_df
    if sample_mode == "head":
        return obs_df.head(max_stars).copy()
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.arange(len(obs_df)), size=max_stars, replace=False))
    return obs_df.iloc[idx].copy()


def _prepare_from_obs_file(
    obs_file: str,
    *,
    input_columns: Sequence[str],
    norm_stats: NormStats,
    id_column: str | None,
    max_stars: int | None,
    sample_mode: str,
    seed: int,
    use_colors: bool = False,
    color_definitions: Sequence[tuple[str, str, str]] | None = None,
    color_means: np.ndarray | None = None,
    color_stds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if obs_file.endswith(".parquet"):
        obs_df = pd.read_parquet(obs_file)
    else:
        obs_df = pd.read_csv(obs_file)
    obs_df = _subset_obs_df(obs_df, max_stars=max_stars, sample_mode=sample_mode, seed=seed)

    n = len(obs_df)
    input_columns = [str(c) for c in input_columns]
    m = len(input_columns)
    col_to_local = {c: i for i, c in enumerate(input_columns)}

    values_raw = np.full((n, m), np.nan, dtype=np.float32)
    for c in input_columns:
        if c in obs_df.columns:
            values_raw[:, col_to_local[c]] = pd.to_numeric(obs_df[c], errors="coerce").values.astype(np.float32)

    # Populate sky unit vectors if needed
    need_sky = [c for c in ("sky_ux", "sky_uy", "sky_uz") if c in col_to_local]
    if need_sky:
        if all(c in obs_df.columns for c in ("sky_ux", "sky_uy", "sky_uz")):
            for c in ("sky_ux", "sky_uy", "sky_uz"):
                if c in col_to_local:
                    values_raw[:, col_to_local[c]] = pd.to_numeric(obs_df[c], errors="coerce").values.astype(np.float32)
        elif all(c in obs_df.columns for c in ("glon", "glat")):
            ux, uy, uz = galactic_to_unitvec(
                pd.to_numeric(obs_df["glon"], errors="coerce").values,
                pd.to_numeric(obs_df["glat"], errors="coerce").values,
            )
            if "sky_ux" in col_to_local:
                values_raw[:, col_to_local["sky_ux"]] = ux.astype(np.float32)
            if "sky_uy" in col_to_local:
                values_raw[:, col_to_local["sky_uy"]] = uy.astype(np.float32)
            if "sky_uz" in col_to_local:
                values_raw[:, col_to_local["sky_uz"]] = uz.astype(np.float32)
        else:
            raise ValueError(
                "Obs file must contain (sky_ux, sky_uy, sky_uz) or (glon, glat) for sky features."
            )

    obs_to_err = dict(zip(OBS_COLS, OBS_ERR_COLS))
    observed = np.ones((n, m), dtype=np.float32)
    errors_raw = np.zeros((n, m), dtype=np.float32)
    for c in input_columns:
        i = col_to_local[c]
        if c in OBS_COLS:
            is_obs = ~np.isnan(values_raw[:, i])
            observed[:, i] = is_obs.astype(np.float32)
            errors_raw[:, i] = np.nan
            err_col = obs_to_err[c]
            if err_col in obs_df.columns:
                err_vals = pd.to_numeric(obs_df[err_col], errors="coerce").values.astype(np.float32)
                err_vals = np.where(is_obs, err_vals, np.nan).astype(np.float32)
                bad = is_obs & (~np.isfinite(err_vals) | (err_vals <= 0.0))
                if bad.any():
                    err_vals[bad] = OBS_ERROR_FLOOR
                errors_raw[:, i] = err_vals
        else:
            is_obs = ~np.isnan(values_raw[:, i])
            observed[:, i] = is_obs.astype(np.float32)
            errors_raw[:, i] = np.where(is_obs, 0.0, np.nan).astype(np.float32)

    if need_sky:
        for c in need_sky:
            i = col_to_local[c]
            if np.isnan(values_raw[:, i]).any():
                raise ValueError(f"Sky feature '{c}' has NaN values after preprocessing.")
            observed[:, i] = 1.0
            errors_raw[:, i] = 0.0

    input_full_idx = [norm_stats.column_index(c) for c in input_columns]
    values_norm = norm_stats.normalize_numpy(values_raw, column_indices=input_full_idx).astype(np.float32)
    values_norm = np.nan_to_num(values_norm, nan=0.0)
    errors_norm = norm_stats.normalize_errors(errors_raw).astype(np.float32)

    if use_colors:
        if color_definitions is None:
            raise ValueError("use_colors=True requires color_definitions.")
        if color_means is None or color_stds is None:
            raise ValueError("use_colors=True requires color_means/color_stds.")
        if len(color_definitions) != int(color_means.shape[0]) or len(color_definitions) != int(color_stds.shape[0]):
            raise ValueError(
                "Color metadata mismatch: color_definitions and color stats lengths differ."
            )

        n_colors = len(color_definitions)
        color_vals_norm = np.zeros((n, n_colors), dtype=np.float32)
        color_err_norm = np.full((n, n_colors), 5.0, dtype=np.float32)  # unobserved sentinel
        color_obs = np.zeros((n, n_colors), dtype=np.float32)

        for i, (_name, c1, c2) in enumerate(color_definitions):
            if c1 not in col_to_local or c2 not in col_to_local:
                raise ValueError(
                    f"Color definition uses source column not in model input columns: ({c1}, {c2})."
                )
            j1 = col_to_local[c1]
            j2 = col_to_local[c2]

            obs_pair = (observed[:, j1] > 0.5) & (observed[:, j2] > 0.5)
            raw_color = values_raw[:, j1] - values_raw[:, j2]
            denom = float(color_stds[i]) if float(color_stds[i]) > 1e-8 else 1.0
            c_norm = (raw_color - float(color_means[i])) / denom
            c_norm[~np.isfinite(c_norm)] = 0.0
            c_norm[~obs_pair] = 0.0
            color_vals_norm[:, i] = c_norm.astype(np.float32)
            color_obs[:, i] = obs_pair.astype(np.float32)

            err1 = errors_raw[:, j1]
            err2 = errors_raw[:, j2]
            raw_err = np.sqrt(err1**2 + err2**2)
            valid_err = obs_pair & np.isfinite(raw_err) & (raw_err > 0.0)
            if np.any(valid_err):
                color_err_norm[valid_err, i] = (
                    np.log(np.clip(raw_err[valid_err], 1e-10, None)) - norm_stats.log_err_mean
                ) / max(float(norm_stats.log_err_std), 1e-8)

        values_norm = np.concatenate([values_norm, color_vals_norm], axis=1)
        errors_norm = np.concatenate([errors_norm, color_err_norm.astype(np.float32)], axis=1)
        observed = np.concatenate([observed, color_obs.astype(np.float32)], axis=1)

    if id_column is not None and id_column in obs_df.columns:
        star_ids = obs_df[id_column].to_numpy()
    else:
        star_ids = obs_df.index.to_numpy()
    return values_norm, errors_norm, observed.astype(np.float32), star_ids


def _sample_chunk(
    model: torch.nn.Module,
    method: str,
    values: torch.Tensor,
    errors: torch.Tensor,
    observed: torch.Tensor,
    *,
    num_samples: int,
    steps: int,
) -> torch.Tensor:
    if method == "flow_matching":
        return model.sample(
            values=values,
            errors=errors,
            observed_mask=observed,
            num_samples=num_samples,
            steps=steps,
        )
    if method in ("realnvp", "normalizing_flow"):
        return model.sample(
            values=values,
            errors=errors,
            observed_mask=observed,
            num_samples=num_samples,
        )
    raise ValueError(f"Unsupported method '{method}'.")


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dir = args.model_dir
    run_name = args.run_name
    cfg_path = os.path.join(model_dir, f"posterior_config_{run_name}.json")
    ckpt_path = os.path.join(model_dir, f"best_model_{run_name}.pt")
    meta_path = os.path.join(model_dir, f"posterior_norm_meta_{run_name}.npz")
    if not os.path.exists(meta_path):
        fallback_meta = os.path.join(model_dir, "norm_stats.npz")
        if os.path.exists(fallback_meta):
            meta_path = fallback_meta
        else:
            raise FileNotFoundError(
                f"No normalization metadata found at {meta_path} or {fallback_meta}"
            )

    norm_stats = NormStats(meta_path)
    config = _load_json(cfg_path)
    method = str(config.get("method", "flow_matching"))
    theta_columns = [str(c) for c in config["theta_columns"]]
    (
        input_columns_base,
        input_columns_model,
        use_colors,
        color_names,
        color_means,
        color_stds,
    ) = _resolve_input_layout(config, norm_stats)
    color_definitions = _resolve_color_definitions(color_names) if use_colors else []

    model = _build_model_from_config(config, input_columns_override=input_columns_model)
    state_dict = _load_state_dict(ckpt_path, device=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded model: method={method}, params={n_params:,}")
    if use_colors:
        print(
            f"Color inputs enabled: base={len(input_columns_base)}, "
            f"colors={len(color_names)}, total={len(input_columns_model)}"
        )
    theta_idx_full = [norm_stats.column_index(c) for c in theta_columns]

    if args.cache_path is not None:
        index_file = args.index_file
        if index_file is None:
            default_idx = os.path.join(model_dir, "test_indices.npy")
            if os.path.exists(default_idx):
                index_file = default_idx
                print(f"Using default index file: {index_file}")
        values_np, errors_np, observed_np, star_ids = _prepare_from_cache(
            args.cache_path,
            input_columns=input_columns_base,
            index_file=index_file,
            max_stars=args.max_stars,
            sample_mode=args.sample_mode,
            seed=args.seed,
            use_colors=use_colors,
            color_definitions=color_definitions,
            color_means=color_means,
            color_stds=color_stds,
        )
        source_name = args.cache_path
    else:
        values_np, errors_np, observed_np, star_ids = _prepare_from_obs_file(
            args.obs_file,
            input_columns=input_columns_base,
            norm_stats=norm_stats,
            id_column=args.id_column,
            max_stars=args.max_stars,
            sample_mode=args.sample_mode,
            seed=args.seed,
            use_colors=use_colors,
            color_definitions=color_definitions,
            color_means=color_means,
            color_stds=color_stds,
        )
        source_name = args.obs_file

    n_stars = values_np.shape[0]
    theta_dim = len(theta_columns)
    if n_stars == 0:
        raise ValueError("No stars selected for sampling.")
    print(
        f"Sampling from {source_name}: stars={n_stars:,}, input_nodes={values_np.shape[1]}, "
        f"theta_dim={theta_dim}, draws={args.num_samples}"
    )

    if args.output_prefix is None:
        out_prefix = os.path.join(model_dir, f"posterior_{run_name}")
    else:
        out_prefix = args.output_prefix
    out_dir = os.path.dirname(out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    norm_samples_path = out_prefix + "_samples_norm.npy"
    phys_samples_path = out_prefix + "_samples_phys.npy"
    summary_path = out_prefix + "_summary.parquet"
    meta_out_path = out_prefix + "_meta.json"

    samples_norm_mm = np.lib.format.open_memmap(
        norm_samples_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_stars, args.num_samples, theta_dim),
    )
    samples_phys_mm = None
    if args.denormalize:
        samples_phys_mm = np.lib.format.open_memmap(
            phys_samples_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_stars, args.num_samples, theta_dim),
        )

    summary_mean = np.zeros((n_stars, theta_dim), dtype=np.float32)
    summary_std = np.zeros((n_stars, theta_dim), dtype=np.float32)

    t0 = time.time()
    for start in range(0, n_stars, args.batch_size):
        end = min(start + args.batch_size, n_stars)
        vals_t = torch.from_numpy(values_np[start:end]).to(device)
        errs_t = torch.from_numpy(errors_np[start:end]).to(device)
        obs_t = torch.from_numpy(observed_np[start:end]).to(device)

        with torch.no_grad():
            samps_t = _sample_chunk(
                model,
                method=method,
                values=vals_t,
                errors=errs_t,
                observed=obs_t,
                num_samples=args.num_samples,
                steps=args.steps,
            )  # (B,S,D)

        samps_norm = samps_t.detach().cpu().numpy().astype(np.float32)
        samples_norm_mm[start:end] = samps_norm

        if args.denormalize:
            flat = samps_norm.reshape(-1, theta_dim)
            phys_flat = norm_stats.denormalize_numpy(flat, column_indices=theta_idx_full).astype(np.float32)
            samps_phys = phys_flat.reshape(end - start, args.num_samples, theta_dim)
            samples_phys_mm[start:end] = samps_phys
            samps_for_summary = samps_phys
        else:
            samps_for_summary = samps_norm

        summary_mean[start:end] = samps_for_summary.mean(axis=1)
        summary_std[start:end] = samps_for_summary.std(axis=1)

        done = end
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-6)
        print(f"  sampled {done:,}/{n_stars:,} stars ({rate:.1f} stars/s)")

    del samples_norm_mm
    if samples_phys_mm is not None:
        del samples_phys_mm

    summary_df = pd.DataFrame({"star_id": star_ids})
    for j, c in enumerate(theta_columns):
        summary_df[f"{c}_mean"] = summary_mean[:, j]
        summary_df[f"{c}_std"] = summary_std[:, j]
    summary_df.to_parquet(summary_path, index=False)

    elapsed = time.time() - t0
    meta = {
        "model_dir": model_dir,
        "run_name": run_name,
        "method": method,
        "source": source_name,
        "num_stars": int(n_stars),
        "num_samples": int(args.num_samples),
        "theta_columns": theta_columns,
        "input_columns": input_columns_model,
        "input_columns_base": input_columns_base,
        "use_colors": use_colors,
        "color_names": color_names if use_colors else [],
        "denormalize": bool(args.denormalize),
        "steps": int(args.steps) if method in ("flow_matching", "flow_matching_xattn") else None,
        "batch_size": int(args.batch_size),
        "elapsed_sec": float(elapsed),
        "outputs": {
            "samples_norm_npy": norm_samples_path,
            "samples_phys_npy": phys_samples_path if args.denormalize else None,
            "summary_parquet": summary_path,
        },
    }
    with open(meta_out_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved normalized samples: {norm_samples_path}")
    if args.denormalize:
        print(f"Saved physical samples:  {phys_samples_path}")
    print(f"Saved summary table:     {summary_path}")
    print(f"Saved metadata:          {meta_out_path}")
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
