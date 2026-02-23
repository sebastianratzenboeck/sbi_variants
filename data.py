from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from columns import OBS_COLS


DEFAULT_THETA_COLS = ["feh", "m_init", "logAge", "rad", "logL", "logT", "logg", "Av"]
DEFAULT_INPUT_COLS = ["sky_ux", "sky_uy", "sky_uz"] + list(OBS_COLS)


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
    log_err_mean: float | None
    log_err_std: float | None


@dataclass
class SBIArrays:
    inputs: np.ndarray
    input_errors: np.ndarray
    input_observed: np.ndarray
    theta: np.ndarray
    sample_weights: np.ndarray | None = None


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

    return CacheArrays(
        values_norm=d["values_norm"].astype(np.float32),
        errors_norm=d["errors_norm"].astype(np.float32),
        observed_mask=d["observed_mask"].astype(np.float32),
        columns=columns,
        means=d["means"].astype(np.float32) if "means" in d else None,
        stds=d["stds"].astype(np.float32) if "stds" in d else None,
        value_transform_names=np.asarray(d["value_transform_names"], dtype=object)
        if "value_transform_names" in d
        else None,
        value_transform_params=np.asarray(d["value_transform_params"], dtype=np.float32)
        if "value_transform_params" in d
        else None,
        log_err_mean=float(d["log_err_mean"]) if "log_err_mean" in d else None,
        log_err_std=float(d["log_err_std"]) if "log_err_std" in d else None,
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
) -> SBIArrays:
    x_idx = column_indices(cache.columns, input_columns, role="input")
    th_idx = column_indices(cache.columns, theta_columns, role="theta")

    rows = np.asarray(row_indices, dtype=np.int64)
    return SBIArrays(
        inputs=cache.values_norm[rows][:, x_idx].astype(np.float32),
        input_errors=cache.errors_norm[rows][:, x_idx].astype(np.float32),
        input_observed=cache.observed_mask[rows][:, x_idx].astype(np.float32),
        theta=cache.values_norm[rows][:, th_idx].astype(np.float32),
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
