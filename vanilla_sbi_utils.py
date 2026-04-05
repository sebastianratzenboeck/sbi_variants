from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from data import build_sbi_arrays, load_cache_arrays, load_indices, parse_column_csv
from inference_utils import NormStats


def configure_sbi_env() -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp")


def build_zero_imputed_npe_arrays(
    *,
    cache_path: str,
    row_indices: np.ndarray,
    input_columns: Sequence[str],
    theta_columns: Sequence[str],
    use_colors: bool,
    color_norm_stats: tuple[np.ndarray, np.ndarray] | None = None,
):
    cache = load_cache_arrays(cache_path)
    arrays = build_sbi_arrays(
        cache,
        row_indices=np.asarray(row_indices, dtype=np.int64),
        input_columns=input_columns,
        theta_columns=theta_columns,
        use_colors=use_colors,
        color_norm_stats=color_norm_stats,
    )
    x = np.nan_to_num(arrays.inputs, nan=0.0).astype(np.float32)
    x[arrays.input_observed <= 0.5] = 0.0
    theta = arrays.theta.astype(np.float32)
    return arrays, x, theta


def maybe_subsample_indices(
    indices: np.ndarray,
    *,
    max_rows: int | None,
    seed: int,
) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if max_rows is None or max_rows <= 0 or indices.size <= max_rows:
        return np.sort(indices)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_rows, replace=False).astype(np.int64))


def build_box_prior_from_theta(
    theta: np.ndarray,
    *,
    device: str = "cpu",
    margin_frac: float = 0.05,
    min_width: float = 1e-3,
):
    from sbi.utils import BoxUniform

    lo = np.min(theta, axis=0)
    hi = np.max(theta, axis=0)
    width = hi - lo
    margin = np.maximum(width * float(margin_frac), float(min_width))
    low = torch.as_tensor(lo - margin, dtype=torch.float32, device=device)
    high = torch.as_tensor(hi + margin, dtype=torch.float32, device=device)
    return BoxUniform(low=low, high=high), low.cpu().numpy(), high.cpu().numpy()


def save_pickle(obj, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_vanilla_sbi_meta_npz(
    *,
    path: str,
    norm_stats: NormStats,
    input_columns: Sequence[str],
    theta_columns: Sequence[str],
    use_colors: bool,
    color_names: Sequence[str] | None,
    color_means: np.ndarray | None,
    color_stds: np.ndarray | None,
) -> None:
    np.savez(
        path,
        columns=np.asarray(norm_stats.columns, dtype=object),
        means=np.asarray(norm_stats.means, dtype=np.float32),
        stds=np.asarray(norm_stats.stds, dtype=np.float32),
        value_transform_names=np.asarray(norm_stats.value_transform_names, dtype=object),
        value_transform_params=np.asarray(norm_stats.value_transform_params, dtype=np.float32),
        log_err_mean=np.array(norm_stats.log_err_mean, dtype=np.float32),
        log_err_std=np.array(norm_stats.log_err_std, dtype=np.float32),
        input_columns=np.asarray(list(input_columns), dtype=object),
        theta_columns=np.asarray(list(theta_columns), dtype=object),
        use_colors=np.array(bool(use_colors), dtype=bool),
        color_names=np.asarray(list(color_names or []), dtype=object),
        color_means=np.asarray(color_means if color_means is not None else np.zeros((0,), dtype=np.float32), dtype=np.float32),
        color_stds=np.asarray(color_stds if color_stds is not None else np.zeros((0,), dtype=np.float32), dtype=np.float32),
    )
