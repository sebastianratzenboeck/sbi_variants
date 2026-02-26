from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from columns import OBS_COLS, COLOR_DEFINITIONS
from value_transforms import apply_inverse_value_transforms_numpy


DEFAULT_THETA_COLS = ["feh", "m_init", "logAge", "rad", "logL", "logT", "logg", "Av"]
DEFAULT_INPUT_COLS = ["sky_ux", "sky_uy", "sky_uz"] + list(OBS_COLS)


def compute_colors_from_cache(
    cache: "CacheArrays",
    row_indices: np.ndarray,
    color_definitions: list[tuple[str, str, str]] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], np.ndarray, np.ndarray]:
    """Compute color features from cached magnitude values.

    Colors are computed by denormalizing magnitudes, computing differences,
    and then normalizing the resulting colors.

    Args:
        cache: CacheArrays containing normalized values and normalization params
        row_indices: Indices of rows to compute colors for
        color_definitions: List of (color_name, mag1_col, mag2_col) tuples.
                          If None, uses all COLOR_DEFINITIONS from columns.py.

    Returns:
        colors_norm: Normalized color values (N, n_colors)
        color_errors: Propagated errors for colors (N, n_colors)
        color_observed: Observed mask for colors (N, n_colors)
        color_names: List of color column names
        color_means: Mean values used for normalization
        color_stds: Std values used for normalization
    """
    if color_definitions is None:
        color_definitions = COLOR_DEFINITIONS

    if cache.means is None or cache.stds is None:
        raise ValueError("Cache must contain means and stds for color computation")

    rows = np.asarray(row_indices, dtype=np.int64)
    n_rows = len(rows)

    # Build column index lookup
    col_to_idx = {c: i for i, c in enumerate(cache.columns)}

    # Keep only definitions whose source columns are available in this cache.
    valid_defs = [
        (name, c1, c2)
        for (name, c1, c2) in color_definitions
        if c1 in col_to_idx and c2 in col_to_idx
    ]
    n_colors = len(valid_defs)
    if n_colors == 0:
        return (
            np.zeros((n_rows, 0), dtype=np.float32),
            np.zeros((n_rows, 0), dtype=np.float32),
            np.zeros((n_rows, 0), dtype=np.float32),
            [],
            np.zeros((0,), dtype=np.float32),
            np.ones((0,), dtype=np.float32),
        )

    # Initialize output arrays
    colors_raw = np.full((n_rows, n_colors), np.nan, dtype=np.float32)
    color_errors = np.full((n_rows, n_colors), np.nan, dtype=np.float32)
    color_observed = np.zeros((n_rows, n_colors), dtype=np.float32)
    color_names = []

    for i, (color_name, mag1_col, mag2_col) in enumerate(valid_defs):
        color_names.append(color_name)

        idx1 = col_to_idx[mag1_col]
        idx2 = col_to_idx[mag2_col]

        # Get normalized values
        v1_norm = cache.values_norm[rows, idx1]
        v2_norm = cache.values_norm[rows, idx2]

        # Denormalize and invert value transforms if present.
        v1_raw = v1_norm * cache.stds[idx1] + cache.means[idx1]
        v2_raw = v2_norm * cache.stds[idx2] + cache.means[idx2]
        if cache.value_transform_names is not None and cache.value_transform_params is not None:
            v1_raw = apply_inverse_value_transforms_numpy(
                v1_raw.reshape(-1, 1),
                transform_names=np.asarray([cache.value_transform_names[idx1]], dtype=object),
                transform_params=np.asarray([cache.value_transform_params[idx1]], dtype=np.float32),
            ).reshape(-1)
            v2_raw = apply_inverse_value_transforms_numpy(
                v2_raw.reshape(-1, 1),
                transform_names=np.asarray([cache.value_transform_names[idx2]], dtype=object),
                transform_params=np.asarray([cache.value_transform_params[idx2]], dtype=np.float32),
            ).reshape(-1)

        # Compute color (mag1 - mag2)
        colors_raw[:, i] = v1_raw - v2_raw

        # Get observed masks
        obs1 = cache.observed_mask[rows, idx1]
        obs2 = cache.observed_mask[rows, idx2]
        color_observed[:, i] = obs1 * obs2  # Both must be observed

        # Propagate errors: err_color = sqrt(err1^2 + err2^2)
        # Errors are stored as normalized log-errors, need to handle carefully
        e1_norm = cache.errors_norm[rows, idx1]
        e2_norm = cache.errors_norm[rows, idx2]

        if cache.log_err_mean is not None and cache.log_err_std is not None:
            # Denormalize log-errors
            e1_raw = np.exp(e1_norm * cache.log_err_std + cache.log_err_mean)
            e2_raw = np.exp(e2_norm * cache.log_err_std + cache.log_err_mean)
        else:
            # Assume errors are already in linear space
            e1_raw = e1_norm
            e2_raw = e2_norm

        # Propagate errors for difference
        color_errors[:, i] = np.sqrt(e1_raw**2 + e2_raw**2)

    # Compute normalization statistics for colors (only from observed values)
    color_means = np.zeros(n_colors, dtype=np.float32)
    color_stds = np.ones(n_colors, dtype=np.float32)

    for i in range(n_colors):
        obs_mask = color_observed[:, i] > 0.5
        if obs_mask.sum() > 10:
            valid_colors = colors_raw[obs_mask, i]
            valid_colors = valid_colors[np.isfinite(valid_colors)]
            if len(valid_colors) > 10:
                color_means[i] = np.mean(valid_colors)
                color_stds[i] = np.std(valid_colors)
                if color_stds[i] < 1e-6:
                    color_stds[i] = 1.0

    # Normalize colors
    colors_norm = (colors_raw - color_means) / color_stds

    # Handle NaNs: set to 0 and mark as unobserved
    nan_mask = ~np.isfinite(colors_norm)
    colors_norm[nan_mask] = 0.0
    color_observed[nan_mask] = 0.0

    # Normalize color errors (log-transform like other errors).
    # Use explicit unobserved sentinel (+5) so regime handling remains consistent.
    LOG_ERR_UNOBS = 5.0
    color_errors_norm = np.full_like(color_errors, LOG_ERR_UNOBS, dtype=np.float32)
    obs_mask = color_observed > 0.5
    if cache.log_err_mean is not None and cache.log_err_std is not None:
        safe = np.clip(color_errors, 1e-10, None)
        valid = obs_mask & np.isfinite(safe)
        color_errors_norm[valid] = (
            np.log(safe[valid]) - cache.log_err_mean
        ) / max(float(cache.log_err_std), 1e-8)
    else:
        valid = obs_mask & np.isfinite(color_errors)
        color_errors_norm[valid] = color_errors[valid]
    color_errors_norm[~np.isfinite(color_errors_norm)] = LOG_ERR_UNOBS

    return (
        colors_norm.astype(np.float32),
        color_errors_norm.astype(np.float32),
        color_observed.astype(np.float32),
        color_names,
        color_means,
        color_stds,
    )


@dataclass
class CacheArrays:
    values_norm: np.ndarray
    errors_norm: np.ndarray
    observed_mask: np.ndarray
    columns: list[str]
    means: np.ndarray | None
    stds: np.ndarray | None
    value_transform_names: np.ndarray | None
    value_transform_params: np.ndarray | None
    cluster_ids: np.ndarray | None
    log_err_mean: float | None
    log_err_std: float | None


@dataclass
class SBIArrays:
    inputs: np.ndarray
    input_errors: np.ndarray
    input_observed: np.ndarray
    theta: np.ndarray
    sample_weights: np.ndarray | None = None
    # Color feature metadata (populated when use_colors=True)
    color_names: list[str] | None = None
    color_means: np.ndarray | None = None
    color_stds: np.ndarray | None = None
    n_base_features: int | None = None  # Number of features before colors


def parse_column_csv(text: str) -> list[str]:
    return [c.strip() for c in text.split(",") if c.strip()]


def load_cache_arrays(cache_path: str) -> CacheArrays:
    d = np.load(cache_path, allow_pickle=True)
    required = ["values_norm", "errors_norm", "observed_mask"]
    missing = [k for k in required if k not in d]
    if missing:
        raise ValueError(f"Cache {cache_path} missing required arrays: {missing}")

    if "columns" in d:
        columns = [str(c) for c in d["columns"].tolist()]
    else:
        raise ValueError(
            f"Cache {cache_path} has no `columns` metadata. Rebuild with current training pipeline."
        )

    values_norm = d["values_norm"].astype(np.float32)
    errors_norm = d["errors_norm"].astype(np.float32)
    observed_mask = d["observed_mask"].astype(np.float32)
    if values_norm.ndim != 2:
        raise ValueError(
            f"Cache {cache_path} expected values_norm to be 2-D, got shape {values_norm.shape}."
        )
    if errors_norm.shape != values_norm.shape:
        raise ValueError(
            f"Cache {cache_path} has shape mismatch: values_norm{values_norm.shape} "
            f"vs errors_norm{errors_norm.shape}."
        )
    if observed_mask.shape != values_norm.shape:
        raise ValueError(
            f"Cache {cache_path} has shape mismatch: values_norm{values_norm.shape} "
            f"vs observed_mask{observed_mask.shape}."
        )
    if values_norm.shape[1] != len(columns):
        raise ValueError(
            f"Cache {cache_path} column metadata mismatch: values_norm width={values_norm.shape[1]} "
            f"but len(columns)={len(columns)}."
        )

    means = d["means"].astype(np.float32) if "means" in d else None
    stds = d["stds"].astype(np.float32) if "stds" in d else None
    if (means is None) != (stds is None):
        raise ValueError(
            f"Cache {cache_path} must contain both means and stds (or neither)."
        )
    if means is not None and means.shape[0] != len(columns):
        raise ValueError(
            f"Cache {cache_path} has len(means)={means.shape[0]} but len(columns)={len(columns)}."
        )
    if stds is not None and stds.shape[0] != len(columns):
        raise ValueError(
            f"Cache {cache_path} has len(stds)={stds.shape[0]} but len(columns)={len(columns)}."
        )

    has_transform_names = "value_transform_names" in d
    has_transform_params = "value_transform_params" in d
    if has_transform_names != has_transform_params:
        raise ValueError(
            f"Cache {cache_path} must contain both value_transform_names and "
            "value_transform_params (or neither)."
        )
    value_transform_names = (
        np.asarray(d["value_transform_names"], dtype=object) if has_transform_names else None
    )
    value_transform_params = (
        np.asarray(d["value_transform_params"], dtype=np.float32) if has_transform_params else None
    )
    if value_transform_names is not None and value_transform_names.shape[0] != len(columns):
        raise ValueError(
            f"Cache {cache_path} has len(value_transform_names)={value_transform_names.shape[0]} "
            f"but len(columns)={len(columns)}."
        )
    if value_transform_params is not None and value_transform_params.shape[0] != len(columns):
        raise ValueError(
            f"Cache {cache_path} has len(value_transform_params)={value_transform_params.shape[0]} "
            f"but len(columns)={len(columns)}."
        )

    cluster_ids = np.asarray(d["cluster_ids"], dtype=np.int64) if "cluster_ids" in d else None
    if cluster_ids is not None and cluster_ids.shape[0] != values_norm.shape[0]:
        raise ValueError(
            f"Cache {cache_path} has len(cluster_ids)={cluster_ids.shape[0]} but "
            f"n_rows={values_norm.shape[0]}."
        )

    has_log_err_mean = "log_err_mean" in d
    has_log_err_std = "log_err_std" in d
    if has_log_err_mean != has_log_err_std:
        raise ValueError(
            f"Cache {cache_path} must contain both log_err_mean and log_err_std (or neither)."
        )
    log_err_mean = float(d["log_err_mean"]) if has_log_err_mean else None
    log_err_std = float(d["log_err_std"]) if has_log_err_std else None

    return CacheArrays(
        values_norm=values_norm,
        errors_norm=errors_norm,
        observed_mask=observed_mask,
        columns=columns,
        means=means,
        stds=stds,
        value_transform_names=value_transform_names,
        value_transform_params=value_transform_params,
        cluster_ids=cluster_ids,
        log_err_mean=log_err_mean,
        log_err_std=log_err_std,
    )


def column_indices(columns: Sequence[str], names: Sequence[str], role: str) -> np.ndarray:
    col_to_idx = {str(c): i for i, c in enumerate(columns)}
    missing = [str(n) for n in names if str(n) not in col_to_idx]
    if missing:
        raise ValueError(
            f"{role} columns not found in cache: {missing[:5]}"
            + ("..." if len(missing) > 5 else "")
        )
    return np.asarray([col_to_idx[str(n)] for n in names], dtype=np.int64)


def load_indices(path: str | None) -> np.ndarray | None:
    if path is None:
        return None
    idx = np.load(path).astype(np.int64)
    if idx.ndim != 1:
        raise ValueError(f"Index file {path} must be 1-D; got shape {idx.shape}")
    return np.unique(idx)


def build_row_split(
    n_rows: int,
    *,
    exclude_indices: np.ndarray | None = None,
    val_split: float = 0.1,
    seed: int = 42,
    max_rows: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(n_rows, dtype=np.int64)
    if exclude_indices is not None and exclude_indices.size > 0:
        exclude_indices = np.unique(exclude_indices.astype(np.int64))
        bad = exclude_indices[(exclude_indices < 0) | (exclude_indices >= n_rows)]
        if bad.size > 0:
            print(
                f"WARNING: ignoring {bad.size} out-of-bounds exclude indices for n_rows={n_rows} "
                f"(examples: {bad[:5].tolist()})"
            )
            exclude_indices = exclude_indices[(exclude_indices >= 0) & (exclude_indices < n_rows)]
        keep = ~np.isin(rows, exclude_indices)
        rows = rows[keep]

    if rows.size < 2:
        raise ValueError("Need at least 2 rows after exclusions to build train/val split.")

    if max_rows is not None and max_rows > 0 and max_rows < rows.size:
        rng = np.random.default_rng(seed)
        rows = np.sort(rng.choice(rows, size=max_rows, replace=False).astype(np.int64))

    if not (0.0 < val_split < 1.0):
        raise ValueError(f"val_split must be in (0,1), got {val_split}")
    train_rows, val_rows = train_test_split(rows, test_size=val_split, random_state=seed)
    return train_rows.astype(np.int64), val_rows.astype(np.int64)


def build_sbi_arrays(
    cache: CacheArrays,
    *,
    row_indices: np.ndarray,
    input_columns: Sequence[str],
    theta_columns: Sequence[str],
    use_colors: bool = False,
    color_definitions: list[tuple[str, str, str]] | None = None,
    color_norm_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> SBIArrays:
    """Build SBI arrays from cache.

    Args:
        cache: CacheArrays containing normalized data
        row_indices: Indices of rows to include
        input_columns: List of input column names
        theta_columns: List of theta (target) column names
        use_colors: If True, append color features to inputs
        color_definitions: Custom color definitions, or None for defaults
        color_norm_stats: Tuple of (color_means, color_stds) from training set.
                         If None, statistics are computed from the provided rows.
                         Pass training set stats when building validation/test sets.

    Returns:
        SBIArrays containing inputs, errors, observed mask, theta, and color metadata
    """
    x_idx = column_indices(cache.columns, input_columns, role="input")
    th_idx = column_indices(cache.columns, theta_columns, role="theta")

    rows = np.asarray(row_indices, dtype=np.int64)

    # Base features
    inputs = cache.values_norm[rows][:, x_idx].astype(np.float32)
    input_errors = cache.errors_norm[rows][:, x_idx].astype(np.float32)
    input_observed = cache.observed_mask[rows][:, x_idx].astype(np.float32)
    n_base_features = inputs.shape[1]

    color_names = None
    color_means = None
    color_stds = None

    if use_colors:
        if color_definitions is None:
            base_set = set(str(c) for c in input_columns)
            color_definitions = [
                (name, m1, m2)
                for (name, m1, m2) in COLOR_DEFINITIONS
                if (m1 in base_set and m2 in base_set)
            ]
        # Compute colors
        (
            colors_norm,
            color_errors_norm,
            color_observed_mask,
            color_names,
            computed_means,
            computed_stds,
        ) = compute_colors_from_cache(cache, rows, color_definitions)

        # Use provided normalization stats or computed ones
        if color_norm_stats is not None:
            color_means, color_stds = color_norm_stats
            # Re-normalize with provided stats
            # First denormalize using computed stats, then renormalize
            colors_raw = colors_norm * computed_stds + computed_means
            colors_norm = (colors_raw - color_means) / color_stds
            colors_norm[~np.isfinite(colors_norm)] = 0.0
        else:
            color_means = computed_means
            color_stds = computed_stds

        # Concatenate colors to inputs
        inputs = np.concatenate([inputs, colors_norm], axis=1)
        input_errors = np.concatenate([input_errors, color_errors_norm], axis=1)
        input_observed = np.concatenate([input_observed, color_observed_mask], axis=1)

    return SBIArrays(
        inputs=inputs.astype(np.float32),
        input_errors=input_errors.astype(np.float32),
        input_observed=input_observed.astype(np.float32),
        theta=cache.values_norm[rows][:, th_idx].astype(np.float32),
        color_names=color_names,
        color_means=color_means,
        color_stds=color_stds,
        n_base_features=n_base_features,
    )


class SBIDataset(Dataset):
    def __init__(self, arrays: SBIArrays):
        self.inputs = torch.tensor(arrays.inputs, dtype=torch.float32)
        self.errors = torch.tensor(arrays.input_errors, dtype=torch.float32)
        self.observed = torch.tensor(arrays.input_observed, dtype=torch.float32)
        self.theta = torch.tensor(arrays.theta, dtype=torch.float32)
        if arrays.sample_weights is None:
            self.sample_weight = torch.ones(self.inputs.shape[0], dtype=torch.float32)
        else:
            self.sample_weight = torch.tensor(arrays.sample_weights, dtype=torch.float32)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": self.inputs[idx],
            "errors": self.errors[idx],
            "observed": self.observed[idx],
            "theta": self.theta[idx],
            "sample_weight": self.sample_weight[idx],
        }
