#!/usr/bin/env python
"""
Sample from a trained SimFormer model for galaxy posterior inference.

Given observed photometric data (magnitudes, parallax, etc.), generate
posterior samples for intrinsic stellar parameters (age, metallicity,
mass, luminosity, ...) and true (noise-free) magnitudes.

Usage:
  # Sample posteriors for stars in a CSV/Parquet file
  python sample_mock_galaxy.py --model-dir ./output --run-name myrun --obs-file stars.parquet

  # More posterior draws, finer ODE integration, on GPU
  python sample_mock_galaxy.py --model-dir ./output --run-name myrun --obs-file stars.parquet \\
      --num-samples 2000 --steps 128 --batch-size 1024 --device cuda

  # Output to a specific file
  python sample_mock_galaxy.py --model-dir ./output --run-name myrun --obs-file stars.parquet \\
      --output posteriors.parquet

  # Sample directly from cached arrays (uses model-dir/test_indices.npy if present)
  python sample_mock_galaxy.py --model-dir ./test_info --run-name fixed_validation \\
      --cache-path ./test_info/build_arrays_cache.npz
"""

import argparse
import inspect
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from columns import (
    INTRINSIC_COLS, OBS_COLS, OBS_ERR_COLS, ALL_VALUE_COLS, COLOR_DEFINITIONS,
)
from prepare_data import galactic_to_unitvec
from transformer import Simformer
from inference_utils import NormStats
from encoder import ObservationEncoder
from posterior_models import ConditionalFMPosterior, ConditionalFlowPosterior
from sampling import (
    build_inference_edge_mask,
    build_inference_condition_mask,
    build_inference_node_ids,
    sample_batched_flow,
)

# Keep observed errors strictly positive so they are always treated as real
# measurements by log-error normalization.
OBS_ERROR_FLOOR = 1e-6
LOG_ERR_UNOBS = 5.0
_COLOR_BY_NAME = {name: (mag1, mag2) for (name, mag1, mag2) in COLOR_DEFINITIONS}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _simformer_kwargs_from_config(config):
    """Extract only Simformer ctor args from a possibly richer config dict."""
    sig = inspect.signature(Simformer.__init__)
    valid = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in config.items() if k in valid}

def _upgrade_legacy_binary_state_embeddings(state_dict):
    """Upgrade legacy 1-vector mask embeddings to 2-state embedding tables.

    Legacy checkpoints store:
      - tokenizer.cond_embed.condition_embedding   shape (1, 1, D)
      - tokenizer.observed_embed.observed_embedding shape (1, 1, D)

    New tokenizer expects:
      - shape (2, D), row 0=off, row 1=on
    """
    upgrades = [
        ("tokenizer.cond_embed.condition_embedding", "condition"),
        ("tokenizer.observed_embed.observed_embedding", "observed"),
    ]
    for key, name in upgrades:
        if key not in state_dict:
            continue
        w = state_dict[key]
        if w.ndim == 2 and w.shape[0] == 2:
            continue  # already new format

        on_vec = None
        if w.ndim == 3 and w.shape[0] == 1 and w.shape[1] == 1:
            on_vec = w[0, 0]
        elif w.ndim == 2 and w.shape[0] == 1:
            on_vec = w[0]

        if on_vec is None:
            continue

        new_w = torch.zeros((2, on_vec.shape[0]), dtype=on_vec.dtype, device=on_vec.device)
        new_w[1] = on_vec
        state_dict[key] = new_w
        print(f"  Upgraded legacy {name} embedding checkpoint tensor to 2-state table.")


def _load_state_dict(path, device):
    try:
        state_dict = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(path, map_location=device)
    # Handle checkpoints saved from torch.compile.
    if state_dict and all(str(k).startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k[len("_orig_mod."):]: v for k, v in state_dict.items()}
    return state_dict


def _build_posterior_model_from_config(config, input_columns):
    theta_columns = [str(c) for c in config["theta_columns"]]
    encoder = ObservationEncoder(
        input_columns=[str(c) for c in input_columns],
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


def load_model(model_dir, run_name="default", device="cpu"):
    """Load a trained model and normalization statistics.

    Args:
        model_dir: Directory containing model artifacts and normalization stats.
        run_name: Run name used during training (determines checkpoint and config filenames).
        device: Target device.

    Returns:
        model: Loaded model with weights.
        norm_stats: NormStats instance.

    Supports two artifact layouts:
      1) Legacy SimFormer training:
         - model_config_<run_name>.json
         - best_model_<run_name>.pt
         - norm_stats.npz
      2) SBI posterior training (train_sbi_posterior.py):
         - posterior_config_<run_name>.json
         - best_model_<run_name>.pt
         - posterior_norm_meta_<run_name>.npz
    """
    ckpt_path = os.path.join(model_dir, f"best_model_{run_name}.pt")
    legacy_cfg_path = os.path.join(model_dir, f"model_config_{run_name}.json")
    posterior_cfg_path = os.path.join(model_dir, f"posterior_config_{run_name}.json")

    if os.path.exists(legacy_cfg_path):
        with open(legacy_cfg_path) as f:
            config = json.load(f)
        print(f"  Model config loaded from {legacy_cfg_path}")
        model = Simformer(**_simformer_kwargs_from_config(config))
        state_dict = _load_state_dict(ckpt_path, device=device)
        _upgrade_legacy_binary_state_embeddings(state_dict)
        model.load_state_dict(state_dict)
        norm_stats = NormStats(os.path.join(model_dir, "norm_stats.npz"))
        model._sbi_artifact_kind = "legacy_simformer"
    elif os.path.exists(posterior_cfg_path):
        with open(posterior_cfg_path) as f:
            config = json.load(f)
        print(f"  Posterior config loaded from {posterior_cfg_path}")
        meta_path = os.path.join(model_dir, f"posterior_norm_meta_{run_name}.npz")
        if not os.path.exists(meta_path):
            fallback_meta = os.path.join(model_dir, "norm_stats.npz")
            if os.path.exists(fallback_meta):
                meta_path = fallback_meta
            else:
                raise FileNotFoundError(
                    "No normalization metadata found for posterior artifacts. "
                    f"Expected {meta_path} (or fallback {fallback_meta})."
                )
        norm_stats = NormStats(meta_path)
        input_columns = config.get("input_columns_with_colors")
        if input_columns is None:
            input_columns = (
                norm_stats.input_columns_meta
                if norm_stats.input_columns_meta is not None
                else config.get("input_columns")
            )
        if input_columns is None:
            raise ValueError(
                "posterior_config is missing input column metadata "
                "(input_columns or input_columns_with_colors)."
            )
        model = _build_posterior_model_from_config(config, input_columns=input_columns)
        theta_columns = [str(c) for c in config["theta_columns"]]
        state_dict = _load_state_dict(ckpt_path, device=device)
        model.load_state_dict(state_dict)
        model._sbi_artifact_kind = "direct_posterior"
        model._sbi_input_columns = [str(c) for c in input_columns]
        model._sbi_theta_columns = theta_columns
        model._sbi_theta_indices = [norm_stats.column_index(c) for c in theta_columns]
        model._sbi_full_columns = [str(c) for c in norm_stats.columns]
        model._sbi_full_dim = len(norm_stats.columns)
        model._sbi_norm_stats = norm_stats
    else:
        raise FileNotFoundError(
            f"Could not find model config for run '{run_name}' in {model_dir}. "
            f"Expected either {legacy_cfg_path} or {posterior_cfg_path}."
        )

    model.to(device)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {n_params:,} parameters from {ckpt_path}")
    return model, norm_stats


# ---------------------------------------------------------------------------
# Observation preparation
# ---------------------------------------------------------------------------

def prepare_observations(obs_df, norm_stats, device="cpu", model_columns=None):
    """Convert raw observations into model-ready tensors.

    Each row of *obs_df* represents one star. Columns should match OBS_COLS
    (with NaN for unobserved bands) and optionally OBS_ERR_COLS.

    Sky coordinates (glon, glat) are converted to 3D unit vectors
    (sky_ux, sky_uy, sky_uz) and always conditioned on.
    Other intrinsic and true-magnitude columns are left as unobserved (NaN)
    since those are what we want to infer.

    Args:
        obs_df: DataFrame with observed columns (subset of OBS_COLS / OBS_ERR_COLS).
            Must also contain 'glon' and 'glat' columns (will be converted
            to unit vectors internally).
        norm_stats: NormStats instance from training.
        device: Target device.

    Returns:
        condition_values: (N_stars, M) float tensor — normalized,
            NaN filled with 0.
        condition_mask: (N_stars, M, 1) float tensor — 1 for
            conditioned (observed) dims.
        observed_mask: (N_stars, M) float tensor — 1 for observed dims.
        errors: (N_stars, M) float tensor — log-normalized measurement
            errors with sentinels (-5 perfect, +5 unobserved).
    """
    model_columns = list(model_columns) if model_columns is not None else [str(c) for c in norm_stats.columns]
    col_to_idx = {c: i for i, c in enumerate(model_columns)}
    M = len(model_columns)
    N_stars = len(obs_df)

    # --- Value array: active model columns ---
    values_raw = np.full((N_stars, M), np.nan, dtype=np.float32)

    # Fill directly provided columns that are part of model layout.
    for col in model_columns:
        if col in obs_df.columns:
            values_raw[:, col_to_idx[col]] = obs_df[col].values.astype(np.float32)

    # Sky unit vector — use precomputed columns if available, else convert glon/glat
    sky_cols = ['sky_ux', 'sky_uy', 'sky_uz']
    present_sky_cols = [c for c in sky_cols if c in col_to_idx]
    if present_sky_cols and all(c in obs_df.columns for c in sky_cols):
        for name in sky_cols:
            if name in col_to_idx:
                values_raw[:, col_to_idx[name]] = obs_df[name].values.astype(np.float32)
    elif present_sky_cols and ('glon' in obs_df.columns and 'glat' in obs_df.columns):
        ux, uy, uz = galactic_to_unitvec(
            obs_df['glon'].values, obs_df['glat'].values
        )
        for name, vals in [('sky_ux', ux), ('sky_uy', uy), ('sky_uz', uz)]:
            if name in col_to_idx:
                values_raw[:, col_to_idx[name]] = vals.astype(np.float32)
    elif present_sky_cols:
        raise ValueError(
            "Observation file must contain either (sky_ux, sky_uy, sky_uz) "
            "or (glon, glat) columns for sky position."
        )

    # --- Masks ---
    obs_cols_in_model = [c for c in model_columns if c in OBS_COLS]
    observed_mask = np.ones((N_stars, M), dtype=np.float32)
    for c in obs_cols_in_model:
        observed_mask[:, col_to_idx[c]] = 0.0

    # Condition-mask semantics are independent from observed-mask semantics:
    # condition only on known inputs (sky + available observed measurements).
    condition_mask = np.zeros((N_stars, M), dtype=np.float32)

    for col in obs_cols_in_model:
        col_idx = col_to_idx[col]
        is_obs = ~np.isnan(values_raw[:, col_idx])
        observed_mask[:, col_idx] = is_obs.astype(np.float32)
        condition_mask[:, col_idx] = is_obs.astype(np.float32)

    # Sky unit vector components are always observed and conditioned
    for col in ['sky_ux', 'sky_uy', 'sky_uz']:
        if col in col_to_idx:
            col_idx = col_to_idx[col]
            observed_mask[:, col_idx] = 1.0
            condition_mask[:, col_idx] = 1.0

    # --- Errors array ---
    errors_raw = np.zeros((N_stars, M), dtype=np.float32)
    for c in obs_cols_in_model:
        errors_raw[:, col_to_idx[c]] = np.nan

    obs_to_err = dict(zip(OBS_COLS, OBS_ERR_COLS))
    for obs_col in obs_cols_in_model:
        err_col = obs_to_err[obs_col]
        col_idx = col_to_idx[obs_col]
        if err_col in obs_df.columns:
            err_vals = obs_df[err_col].values.astype(np.float32)
            # Keep error undefined where observation is missing.
            is_obs = ~np.isnan(values_raw[:, col_idx])
            err_vals = np.where(is_obs, err_vals, np.nan).astype(np.float32)
            # For observed entries, floor non-positive/non-finite errors so they
            # are log-scaled as real measurements (including parallax_err).
            bad = is_obs & (~np.isfinite(err_vals) | (err_vals <= 0.0))
            if bad.any():
                err_vals[bad] = OBS_ERROR_FLOOR
            errors_raw[:, col_idx] = err_vals
    # Sky unit vector errors are effectively zero (perfectly known).

    # --- Normalize errors (log-transform + standardize with sentinels) ---
    errors_norm = norm_stats.normalize_errors(errors_raw)

    # --- Normalize values ---
    values_norm = norm_stats.normalize_numpy(values_raw)
    values_norm = np.nan_to_num(values_norm, nan=0.0)

    # --- Convert to tensors ---
    condition_values = torch.tensor(values_norm, dtype=torch.float32, device=device)
    condition_mask = torch.tensor(condition_mask, dtype=torch.float32, device=device).unsqueeze(-1)
    observed_mask = torch.tensor(observed_mask, dtype=torch.float32, device=device)
    errors = torch.tensor(errors_norm, dtype=torch.float32, device=device)

    return condition_values, condition_mask, observed_mask, errors


def prepare_observations_from_cache(
    cache_path,
    indices=None,
    max_stars=None,
    expected_columns=None,
    device="cpu",
):
    """Load normalized inputs directly from build_arrays_cache.npz.

    Returns tensors ready for sample_posterior() and the selected row indices.
    """
    d = np.load(cache_path, allow_pickle=True)

    expected_columns = list(expected_columns) if expected_columns is not None else list(ALL_VALUE_COLS)
    cached_cols = list(d["columns"]) if "columns" in d else list(ALL_VALUE_COLS)
    cached_to_idx = {c: i for i, c in enumerate(cached_cols)}
    missing = [c for c in expected_columns if c not in cached_to_idx]
    if missing:
        raise ValueError(
            f"Cache {cache_path} is missing expected columns: {missing[:5]}"
            + ("..." if len(missing) > 5 else "")
        )
    col_idx = np.asarray([cached_to_idx[c] for c in expected_columns], dtype=np.int64)

    values_norm = d["values_norm"].astype(np.float32)[:, col_idx]
    errors_norm = d["errors_norm"].astype(np.float32)[:, col_idx]
    observed_mask = d["observed_mask"].astype(np.float32)[:, col_idx]

    n_total = values_norm.shape[0]
    if indices is None:
        selected = np.arange(n_total, dtype=np.int64)
    else:
        selected = np.asarray(indices, dtype=np.int64)

    if max_stars is not None:
        selected = selected[:max_stars]

    if selected.size == 0:
        raise ValueError("No rows selected from cache.")
    if selected.min() < 0 or selected.max() >= n_total:
        raise ValueError(f"Selected indices are out of bounds for cache with {n_total} rows.")

    values_norm = values_norm[selected]
    errors_norm = errors_norm[selected]
    observed_mask = observed_mask[selected]

    # Match inference semantics: condition only on observed obs-block + sky.
    condition_mask = np.zeros_like(observed_mask, dtype=np.float32)
    for i, c in enumerate(expected_columns):
        if c in OBS_COLS:
            condition_mask[:, i] = observed_mask[:, i]
    for col in ["sky_ux", "sky_uy", "sky_uz"]:
        if col in expected_columns:
            condition_mask[:, expected_columns.index(col)] = 1.0

    condition_values = torch.tensor(values_norm, dtype=torch.float32, device=device)
    condition_mask = torch.tensor(condition_mask, dtype=torch.float32, device=device).unsqueeze(-1)
    observed_mask = torch.tensor(observed_mask, dtype=torch.float32, device=device)
    errors = torch.tensor(errors_norm, dtype=torch.float32, device=device)
    return condition_values, condition_mask, observed_mask, errors, selected


# ---------------------------------------------------------------------------
# Posterior sampling
# ---------------------------------------------------------------------------

def _is_direct_posterior_model(model) -> bool:
    return (
        isinstance(model, (ConditionalFMPosterior, ConditionalFlowPosterior))
        or getattr(model, "_sbi_artifact_kind", None) == "direct_posterior"
    )


def _compute_color_feature(
    *,
    color_name,
    full_values,
    full_errors,
    full_observed,
    full_columns,
    norm_stats,
):
    if color_name not in _COLOR_BY_NAME:
        raise ValueError(f"Unknown color feature '{color_name}'.")
    if color_name not in norm_stats.color_names:
        raise ValueError(
            f"Color feature '{color_name}' is missing from normalization metadata."
        )

    mag1, mag2 = _COLOR_BY_NAME[color_name]
    full_to_idx = {str(c): i for i, c in enumerate(full_columns)}
    if mag1 not in full_to_idx or mag2 not in full_to_idx:
        raise ValueError(
            f"Cannot derive color '{color_name}' because '{mag1}' or '{mag2}' "
            "is missing from the full column layout."
        )

    idx1 = full_to_idx[mag1]
    idx2 = full_to_idx[mag2]
    col_idx = [norm_stats.column_index(mag1), norm_stats.column_index(mag2)]

    mags_norm = torch.stack([full_values[:, idx1], full_values[:, idx2]], dim=1)
    mags_raw = norm_stats.denormalize(mags_norm, column_indices=col_idx)
    color_raw = mags_raw[:, 0] - mags_raw[:, 1]

    color_meta_idx = norm_stats.color_names.index(color_name)
    color_mean = torch.as_tensor(
        float(norm_stats.color_means[color_meta_idx]),
        dtype=full_values.dtype,
        device=full_values.device,
    )
    color_std = torch.as_tensor(
        max(float(norm_stats.color_stds[color_meta_idx]), 1e-8),
        dtype=full_values.dtype,
        device=full_values.device,
    )
    color_values = (color_raw - color_mean) / color_std

    color_observed = (full_observed[:, idx1] * full_observed[:, idx2]).to(full_values.dtype)
    color_values = torch.where(
        color_observed > 0.5,
        color_values,
        torch.zeros_like(color_values),
    )

    color_errors = torch.full_like(color_values, LOG_ERR_UNOBS)
    valid = color_observed > 0.5
    if valid.any():
        log_err_std = max(float(norm_stats.log_err_std), 1e-8)
        e1_raw = torch.exp(full_errors[valid, idx1] * log_err_std + float(norm_stats.log_err_mean))
        e2_raw = torch.exp(full_errors[valid, idx2] * log_err_std + float(norm_stats.log_err_mean))
        color_err_raw = torch.sqrt(e1_raw.pow(2) + e2_raw.pow(2)).clamp_min(1e-10)
        color_errors[valid] = (
            torch.log(color_err_raw) - float(norm_stats.log_err_mean)
        ) / log_err_std

    return color_values, color_errors, color_observed


def _build_direct_posterior_inputs(
    *,
    model,
    condition_values,
    observed_mask,
    errors,
):
    input_columns = list(getattr(model, "_sbi_input_columns", []))
    full_columns = list(getattr(model, "_sbi_full_columns", []))
    norm_stats = getattr(model, "_sbi_norm_stats", None)
    if not input_columns or not full_columns or norm_stats is None:
        raise ValueError(
            "Direct posterior checkpoint is missing input metadata required for inference."
        )
    if condition_values.shape[1] != len(full_columns):
        raise ValueError(
            "Full inference tensors do not match checkpoint column layout: "
            f"got width={condition_values.shape[1]}, expected {len(full_columns)}."
        )

    full_to_idx = {str(c): i for i, c in enumerate(full_columns)}
    vals = []
    errs = []
    obs = []
    for col in input_columns:
        if col in full_to_idx:
            idx = full_to_idx[col]
            vals.append(condition_values[:, idx])
            errs.append(errors[:, idx])
            obs.append(observed_mask[:, idx])
            continue

        color_values, color_errors, color_observed = _compute_color_feature(
            color_name=col,
            full_values=condition_values,
            full_errors=errors,
            full_observed=observed_mask,
            full_columns=full_columns,
            norm_stats=norm_stats,
        )
        vals.append(color_values)
        errs.append(color_errors)
        obs.append(color_observed)

    return (
        torch.stack(vals, dim=1),
        torch.stack(errs, dim=1),
        torch.stack(obs, dim=1),
    )


def _sample_direct_posterior(
    *,
    model,
    condition_values,
    observed_mask,
    errors,
    num_samples,
    batch_size,
    steps,
    device,
):
    theta_indices = list(getattr(model, "_sbi_theta_indices", []))
    full_dim = int(getattr(model, "_sbi_full_dim", condition_values.shape[1]))
    if not theta_indices:
        raise ValueError("Direct posterior checkpoint is missing theta index metadata.")
    if condition_values.shape[1] != full_dim:
        raise ValueError(
            f"condition_values width={condition_values.shape[1]} does not match "
            f"checkpoint full_dim={full_dim}."
        )

    input_values, input_errors, input_observed = _build_direct_posterior_inputs(
        model=model,
        condition_values=condition_values,
        observed_mask=observed_mask,
        errors=errors,
    )

    n_stars = int(condition_values.shape[0])
    all_samples = (
        condition_values.detach().cpu().unsqueeze(1).repeat(1, num_samples, 1)
    )

    for start in range(0, n_stars, batch_size):
        end = min(start + batch_size, n_stars)
        print(
            f"  Sampling stars {start + 1}-{end}/{n_stars} "
            f"({num_samples} draws each)..."
        )
        vals = input_values[start:end].to(device)
        errs = input_errors[start:end].to(device)
        obs = input_observed[start:end].to(device)

        if isinstance(model, ConditionalFMPosterior):
            theta_samples = model.sample(
                values=vals,
                errors=errs,
                observed_mask=obs,
                num_samples=num_samples,
                steps=steps,
            )
        elif isinstance(model, ConditionalFlowPosterior):
            theta_samples = model.sample(
                values=vals,
                errors=errs,
                observed_mask=obs,
                num_samples=num_samples,
            )
        else:
            raise TypeError(
                f"Unsupported direct posterior model type '{type(model).__name__}'."
            )

        all_samples[start:end, :, theta_indices] = theta_samples.detach().cpu()

    return all_samples

def sample_posterior(
    model,
    condition_values,
    condition_mask,
    observed_mask,
    errors,
    num_samples=512,
    batch_size=512,
    steps=128,
    device="cpu",
):
    """Generate posterior samples for each star.

    For each of the N_stars input observations, draws *num_samples*
    independent posterior samples by running the flow from t=0 to t=1.

    Args:
        model: Trained Simformer model.
        condition_values: (N_stars, NUM_NODES) normalized values.
        condition_mask: (N_stars, NUM_NODES, 1) condition mask.
        observed_mask: (N_stars, NUM_NODES) observed mask.
        errors: (N_stars, NUM_NODES) measurement errors.
        num_samples: Number of posterior draws per star.
        batch_size: Maximum batch size for parallel sampling.
        steps: Number of Euler integration steps.
        device: Target device.

    Returns:
        all_samples: (N_stars, num_samples, NUM_NODES) tensor in normalized space.
    """
    if _is_direct_posterior_model(model):
        # Direct posterior checkpoints sample theta only; embed those draws back
        # into the full cached column layout so existing eval code can index by
        # column name without special-casing model families.
        return _sample_direct_posterior(
            model=model,
            condition_values=condition_values,
            observed_mask=observed_mask,
            errors=errors,
            num_samples=num_samples,
            batch_size=batch_size,
            steps=steps,
            device=device,
        )

    N_stars = condition_values.shape[0]
    M = condition_values.shape[1]
    all_samples = torch.zeros(N_stars, num_samples, M, device="cpu")

    for star_idx in range(N_stars):
        print(f"  Sampling star {star_idx + 1}/{N_stars} "
              f"({num_samples} draws, {steps} steps) ...")

        # Replicate this star's data num_samples times
        cv = condition_values[star_idx].unsqueeze(0).expand(num_samples, -1)  # (S, M)
        cm = condition_mask[star_idx].unsqueeze(0).expand(num_samples, -1, -1)  # (S, M, 1)
        om = observed_mask[star_idx].unsqueeze(0).expand(num_samples, -1)   # (S, M)
        er = errors[star_idx].unsqueeze(0).expand(num_samples, -1)          # (S, M)

        # Process in chunks of batch_size
        star_samples = []
        for chunk_start in range(0, num_samples, batch_size):
            chunk_end = min(chunk_start + batch_size, num_samples)
            B = chunk_end - chunk_start

            cv_chunk = cv[chunk_start:chunk_end].to(device)
            cm_chunk = cm[chunk_start:chunk_end].to(device)
            om_chunk = om[chunk_start:chunk_end].to(device)
            er_chunk = er[chunk_start:chunk_end].to(device)

            # Build inference masks
            node_ids = build_inference_node_ids(B, M, device=device)
            edge_mask = build_inference_edge_mask(B, M, observed_mask=om_chunk, device=device)

            # Run flow
            x = sample_batched_flow(
                model_fn=model,
                shape=(B,),
                condition_mask=cm_chunk,
                condition_values=cv_chunk,
                node_ids=node_ids,
                edge_masks=edge_mask,
                errors=er_chunk,
                observed_mask=om_chunk,
                steps=steps,
                device=device,
            )  # (B, M, 1)

            star_samples.append(x.squeeze(-1).cpu())  # (B, M)

        all_samples[star_idx] = torch.cat(star_samples, dim=0)  # (num_samples, M)

    return all_samples


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def samples_to_dataframe(all_samples, norm_stats, star_ids=None):
    """Convert posterior samples to a long-format DataFrame.

    Args:
        all_samples: (N_stars, num_samples, NUM_NODES) tensor in normalized space.
        norm_stats: NormStats instance for denormalization.
        star_ids: Optional list/array of star identifiers.

    Returns:
        DataFrame with columns: [star_id, sample_idx, <column_names>...]
    """
    N_stars, num_samples, M = all_samples.shape

    # Denormalize to physical units
    samples_phys = norm_stats.denormalize(all_samples.view(-1, M)).view(N_stars, num_samples, M)
    samples_np = samples_phys.numpy()

    rows = []
    for star_idx in range(N_stars):
        sid = star_ids[star_idx] if star_ids is not None else star_idx
        for sample_idx in range(num_samples):
            row = {"star_id": sid, "sample_idx": sample_idx}
            for col_idx, col_name in enumerate(norm_stats.columns):
                row[col_name] = samples_np[star_idx, sample_idx, col_idx]
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Sample posteriors from trained SimFormer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--model-dir", type=str, required=True,
                   help="Directory with model checkpoint/config and norm_stats.npz")
    p.add_argument("--run-name", type=str, default="default",
                   help="Run name (matches training --run-name for config/checkpoint)")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--obs-file", type=str, default=None,
                     help="CSV/Parquet with observed columns (_obs, _err)")
    src.add_argument("--cache-path", type=str, default=None,
                     help="Path to build_arrays_cache.npz for direct cached inference")
    p.add_argument("--index-file", type=str, default=None,
                   help="Optional .npy indices into --cache-path (e.g. test_indices.npy)")
    p.add_argument("--num-samples", type=int, default=512,
                   help="Number of posterior draws per star")
    p.add_argument("--steps", type=int, default=128,
                   help="Number of Euler ODE integration steps")
    p.add_argument("--batch-size", type=int, default=512,
                   help="Batch size for parallel sampling")
    p.add_argument("--device", type=str, default=None,
                   help="Device (auto-detect if not set)")
    p.add_argument("--output", type=str, default=None,
                   help="Output file path (default: <model-dir>/posteriors.parquet)")
    p.add_argument("--max-stars", type=int, default=None,
                   help="Limit number of stars to sample (for testing)")

    return p.parse_args()


def main():
    args = parse_args()

    # Device
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # ---- Load model ----
    print("\n--- Model ---")
    model, norm_stats = load_model(args.model_dir, run_name=args.run_name, device=device)

    # ---- Prepare inputs ----
    print("\n--- Preparing inputs ---")
    if args.cache_path is not None:
        idx = None
        if args.index_file is not None:
            idx = np.load(args.index_file)
            print(f"  Loaded {len(idx):,} row indices from {args.index_file}")
        else:
            default_idx = os.path.join(args.model_dir, "test_indices.npy")
            if os.path.exists(default_idx):
                idx = np.load(default_idx)
                print(f"  Using test indices from {default_idx} ({len(idx):,} rows)")

        condition_values, condition_mask, observed_mask, errors, selected_indices = \
            prepare_observations_from_cache(
                cache_path=args.cache_path,
                indices=idx,
                max_stars=args.max_stars,
                expected_columns=list(norm_stats.columns),
                device="cpu",
            )
        print(f"  Loaded {len(selected_indices):,} stars from cache {args.cache_path}")
        star_ids = selected_indices
    else:
        print("\n--- Observations ---")
        if args.obs_file.endswith(".parquet"):
            obs_df = pd.read_parquet(args.obs_file)
        else:
            obs_df = pd.read_csv(args.obs_file)

        if args.max_stars is not None:
            obs_df = obs_df.head(args.max_stars)

        print(f"  Loaded {len(obs_df)} stars from {args.obs_file}")

        # Check which observed columns are present
        model_obs_cols = [c for c in OBS_COLS if c in norm_stats.columns]
        model_err_cols = [dict(zip(OBS_COLS, OBS_ERR_COLS))[c] for c in model_obs_cols]
        present_obs = [c for c in model_obs_cols if c in obs_df.columns]
        present_err = [c for c in model_err_cols if c in obs_df.columns]
        print(f"  Observed columns found: {len(present_obs)}/{len(model_obs_cols)}")
        print(f"  Error columns found:    {len(present_err)}/{len(model_err_cols)}")

        if not present_obs:
            raise ValueError(
                f"No observed columns found in {args.obs_file}. "
                f"Expected columns like: {OBS_COLS[:3]}"
            )

        condition_values, condition_mask, observed_mask, errors = \
            prepare_observations(obs_df, norm_stats, device="cpu")
        # Use an identifier column if available, otherwise use row index
        star_ids = obs_df.index.values if obs_df.index.name else np.arange(len(obs_df))

    # Summary
    n_cond_per_star = condition_mask.squeeze(-1).sum(dim=1)
    print(f"  Conditioned dims per star: "
          f"min={n_cond_per_star.min():.0f}, "
          f"max={n_cond_per_star.max():.0f}, "
          f"mean={n_cond_per_star.mean():.1f}")
    num_nodes = int(condition_mask.shape[1])
    print(f"  Free dims per star (to be inferred): "
          f"{num_nodes - n_cond_per_star.mean():.1f} on average")

    # ---- Sample posteriors ----
    print(f"\n--- Sampling ({args.num_samples} draws/star, {args.steps} steps) ---")
    t0 = time.time()

    all_samples = sample_posterior(
        model=model,
        condition_values=condition_values,
        condition_mask=condition_mask,
        observed_mask=observed_mask,
        errors=errors,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        steps=args.steps,
        device=device,
    )

    elapsed = time.time() - t0
    print(f"\n  Sampling completed in {elapsed:.1f}s "
          f"({elapsed / len(star_ids):.2f}s/star)")

    # ---- Save results ----
    output_path = args.output or os.path.join(args.model_dir, "posteriors.parquet")

    print(f"\n--- Saving results ---")
    result_df = samples_to_dataframe(all_samples, norm_stats, star_ids=star_ids)
    result_df.to_parquet(output_path, index=False)
    print(f"  Posteriors saved to {output_path}")
    print(f"  Shape: {result_df.shape[0]:,} rows "
          f"({len(star_ids)} stars x {args.num_samples} samples)")

    # ---- Summary statistics ----
    print(f"\n--- Summary ---")
    print(f"  Stars:      {len(star_ids)}")
    print(f"  Samples:    {args.num_samples}/star")
    print(f"  ODE steps:  {args.steps}")
    print(f"  Total time: {elapsed:.1f}s")

    # Print mean/std of inferred intrinsic params for the first star
    if len(star_ids) > 0:
        first_star = all_samples[0]  # (num_samples, NUM_NODES)
        first_star_phys = norm_stats.denormalize(first_star)
        print(f"\n  Posterior summary for first star:")
        intrinsic_cols = [c for c in INTRINSIC_COLS if c in norm_stats.columns]
        for col in intrinsic_cols:
            i = norm_stats.column_index(col)
            vals = first_star_phys[:, i].numpy()
            print(f"    {col:>12s}: {vals.mean():10.4f} +/- {vals.std():.4f}")


if __name__ == "__main__":
    main()
