#!/usr/bin/env python
"""Shared utilities for evaluation scripts."""

from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch

from columns import ALL_VALUE_COLS, INTRINSIC_COLS, OBS_COLS


LOG_ERR_UNOBS = 5.0
DEFAULT_TARGET_COLS = INTRINSIC_COLS[3:]  # exclude sky unit vector


def auto_device(device: str | None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_float_list(raw: str) -> list[float]:
    vals = []
    for x in raw.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    if not vals:
        raise ValueError("Expected at least one float value.")
    return vals


def parse_str_list(raw: str) -> list[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one string value.")
    return vals


def column_indices(names: Sequence[str], columns: Sequence[str] | None = None) -> list[int]:
    cols = list(columns) if columns is not None else list(ALL_VALUE_COLS)
    col_to_idx = {c: i for i, c in enumerate(cols)}
    idx = []
    for n in names:
        if n not in col_to_idx:
            raise ValueError(f"Unknown column '{n}'.")
        idx.append(col_to_idx[n])
    return idx


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def load_cache_arrays(
    cache_path: str,
    index_file: str | None = None,
    max_stars: int | None = None,
    sample_mode: str = "random",
    seed: int = 42,
    expected_columns: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load selected rows from build_arrays_cache.npz.

    Returns:
        values_norm, errors_norm, observed_mask, selected_indices
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
    if index_file is not None:
        selected = np.load(index_file).astype(np.int64)
    else:
        selected = np.arange(n_total, dtype=np.int64)

    if max_stars is not None and max_stars > 0 and selected.size > max_stars:
        if sample_mode == "random":
            rng = np.random.default_rng(seed)
            selected = rng.choice(selected, size=max_stars, replace=False)
            selected.sort()
        elif sample_mode == "head":
            selected = selected[:max_stars]
        else:
            raise ValueError(f"Unsupported sample_mode '{sample_mode}'.")

    if selected.size == 0:
        raise ValueError("No rows selected from cache.")
    if selected.min() < 0 or selected.max() >= n_total:
        raise ValueError(
            f"Selected indices are out of bounds for cache with {n_total} rows."
        )

    return (
        values_norm[selected],
        errors_norm[selected],
        observed_mask[selected],
        selected,
    )


def build_condition_mask(observed_mask: np.ndarray) -> np.ndarray:
    """Build inference condition mask from observed mask.

    Conditions on:
      - observed entries in observation block
      - sky unit vector components (always)
    """
    if observed_mask.shape[1] == len(ALL_VALUE_COLS):
        columns = list(ALL_VALUE_COLS)
    else:
        raise ValueError(
            "build_condition_mask() requires explicit columns for non-default layouts; "
            "use build_condition_mask_for_columns()."
        )
    return build_condition_mask_for_columns(observed_mask, columns)


def build_condition_mask_for_columns(
    observed_mask: np.ndarray,
    columns: Sequence[str],
) -> np.ndarray:
    """Build inference condition mask from observed mask for arbitrary columns."""
    columns = list(columns)
    obs_set = set(OBS_COLS)
    condition_mask = np.zeros_like(observed_mask, dtype=np.float32)
    for i, c in enumerate(columns):
        if c in obs_set:
            condition_mask[:, i] = observed_mask[:, i]
    for c in ("sky_ux", "sky_uy", "sky_uz"):
        if c in columns:
            condition_mask[:, columns.index(c)] = 1.0
    return condition_mask


def to_input_tensors(
    values_norm: np.ndarray,
    errors_norm: np.ndarray,
    observed_mask: np.ndarray,
    condition_mask: np.ndarray | None = None,
    columns: Sequence[str] | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if condition_mask is None:
        if columns is None:
            condition_mask = build_condition_mask(observed_mask)
        else:
            condition_mask = build_condition_mask_for_columns(observed_mask, columns)

    condition_values = torch.tensor(values_norm, dtype=torch.float32, device=device)
    condition_mask_t = torch.tensor(condition_mask, dtype=torch.float32, device=device).unsqueeze(-1)
    observed_t = torch.tensor(observed_mask, dtype=torch.float32, device=device)
    errors_t = torch.tensor(errors_norm, dtype=torch.float32, device=device)
    return condition_values, condition_mask_t, observed_t, errors_t


def maybe_denormalize(
    norm_stats,
    arr: np.ndarray,
    col_idx: Sequence[int],
    denorm: bool,
) -> np.ndarray:
    if not denorm:
        return arr
    flat = arr.reshape(-1, len(col_idx))
    out = norm_stats.denormalize_numpy(flat, column_indices=col_idx)
    return out.reshape(arr.shape)


def interval_metrics(
    samples: np.ndarray,
    truth: np.ndarray,
    target_cols: Sequence[str],
    levels: Sequence[float],
) -> pd.DataFrame:
    """Compute per-parameter equal-tailed interval metrics.

    Args:
        samples: (N, S, D)
        truth: (N, D)
    """
    if samples.ndim != 3:
        raise ValueError("samples must have shape (N, S, D)")
    if truth.ndim != 2:
        raise ValueError("truth must have shape (N, D)")
    if samples.shape[0] != truth.shape[0] or samples.shape[2] != truth.shape[1]:
        raise ValueError("Shape mismatch between samples and truth.")

    rows = []
    for j, col in enumerate(target_cols):
        samp_j = samples[:, :, j]
        truth_j = truth[:, j]
        for level in levels:
            q_lo = (1.0 - level) / 2.0
            q_hi = 1.0 - q_lo
            lo = np.quantile(samp_j, q_lo, axis=1)
            hi = np.quantile(samp_j, q_hi, axis=1)
            inside = (truth_j >= lo) & (truth_j <= hi)
            width = hi - lo
            cov = float(inside.mean())
            rows.append(
                {
                    "column": col,
                    "level": float(level),
                    "coverage": cov,
                    "calibration_error": cov - float(level),
                    "mean_width": float(width.mean()),
                    "median_width": float(np.median(width)),
                }
            )
    return pd.DataFrame(rows)


def ks_uniform(u: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic against Uniform(0,1)."""
    if u.ndim != 1:
        raise ValueError("u must be 1-D")
    if u.size == 0:
        raise ValueError("u must be non-empty")
    x = np.sort(u)
    n = x.size
    cdf_hi = np.arange(1, n + 1) / n
    cdf_lo = np.arange(0, n) / n
    d_plus = np.max(cdf_hi - x)
    d_minus = np.max(x - cdf_lo)
    return float(max(d_plus, d_minus))


def projection_ranks(
    samples: np.ndarray,
    truth: np.ndarray,
    num_projections: int = 256,
    seed: int = 42,
) -> np.ndarray:
    """Compute random-projection rank statistics (TARP-style).

    Returns:
        u: (num_projections, N) array in (0, 1)
    """
    if samples.ndim != 3:
        raise ValueError("samples must have shape (N, S, D)")
    if truth.ndim != 2:
        raise ValueError("truth must have shape (N, D)")
    n, s, d = samples.shape
    if truth.shape != (n, d):
        raise ValueError("truth shape must match samples over N and D.")

    rng = np.random.default_rng(seed)
    u_all = np.empty((num_projections, n), dtype=np.float32)
    for k in range(num_projections):
        w = rng.normal(size=(d,)).astype(np.float32)
        w_norm = np.linalg.norm(w)
        if w_norm < 1e-12:
            w[0] = 1.0
            w_norm = 1.0
        w /= w_norm
        proj_samp = np.tensordot(samples, w, axes=([2], [0]))  # (N, S)
        proj_true = truth @ w                                   # (N,)
        ranks = (proj_samp <= proj_true[:, None]).sum(axis=1)  # [0, S]
        u_all[k] = (ranks + 1.0) / (s + 1.0)
    return u_all


def central_rank_coverage(u: np.ndarray, alpha: float) -> float:
    lo = (1.0 - alpha) / 2.0
    hi = 1.0 - lo
    return float(((u >= lo) & (u <= hi)).mean())


def describe_conditioning(condition_mask: np.ndarray) -> dict[str, float]:
    per_star = condition_mask.sum(axis=1)
    return {
        "cond_min": float(per_star.min()),
        "cond_max": float(per_star.max()),
        "cond_mean": float(per_star.mean()),
    }
