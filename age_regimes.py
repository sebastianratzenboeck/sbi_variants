from __future__ import annotations

from typing import Sequence

import numpy as np


DEFAULT_AGE_BIN_EDGES = (7.8, 8.8)
DEFAULT_AGE_REGIME_NAMES = ("young", "mid", "old")


def validate_age_regime_spec(
    edges: Sequence[float],
    names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    edges_arr = np.asarray(edges, dtype=np.float32).reshape(-1)
    if edges_arr.ndim != 1 or edges_arr.size == 0:
        raise ValueError("edges must be a non-empty 1-D sequence")
    if not np.all(np.isfinite(edges_arr)):
        raise ValueError("edges must be finite")
    if not np.all(np.diff(edges_arr) > 0):
        raise ValueError("edges must be strictly increasing")

    if names is None:
        if len(DEFAULT_AGE_REGIME_NAMES) != edges_arr.size + 1:
            raise ValueError("default regime names do not match number of bins")
        names_list = list(DEFAULT_AGE_REGIME_NAMES)
    else:
        names_list = [str(n) for n in names]
        if len(names_list) != edges_arr.size + 1:
            raise ValueError(
                f"expected {edges_arr.size + 1} regime names for {edges_arr.size} edges, got {len(names_list)}"
            )
    return edges_arr, names_list


def regime_names(edges: Sequence[float], names: Sequence[str] | None = None) -> list[str]:
    _, names_list = validate_age_regime_spec(edges, names)
    return names_list


def logage_to_regime_index(
    logage: np.ndarray | Sequence[float],
    edges: Sequence[float] = DEFAULT_AGE_BIN_EDGES,
) -> np.ndarray:
    edges_arr, _ = validate_age_regime_spec(edges)
    vals = np.asarray(logage, dtype=np.float32)
    if np.any(~np.isfinite(vals)):
        raise ValueError("logAge values must be finite")
    idx = np.searchsorted(edges_arr, vals, side="right").astype(np.int64)
    return idx


def regime_name_from_index(
    idx: int,
    edges: Sequence[float] = DEFAULT_AGE_BIN_EDGES,
    names: Sequence[str] | None = None,
) -> str:
    _, names_list = validate_age_regime_spec(edges, names)
    if idx < 0 or idx >= len(names_list):
        raise IndexError(f"regime index out of range: {idx}")
    return names_list[int(idx)]


def regime_mask(
    logage: np.ndarray | Sequence[float],
    regime_idx: int,
    edges: Sequence[float] = DEFAULT_AGE_BIN_EDGES,
) -> np.ndarray:
    idx = logage_to_regime_index(logage, edges=edges)
    return idx == int(regime_idx)


def summarize_regime_counts(
    regime_idx: np.ndarray | Sequence[int],
    edges: Sequence[float] = DEFAULT_AGE_BIN_EDGES,
    names: Sequence[str] | None = None,
) -> dict[str, int]:
    _, names_list = validate_age_regime_spec(edges, names)
    idx = np.asarray(regime_idx, dtype=np.int64).reshape(-1)
    out: dict[str, int] = {}
    for i, name in enumerate(names_list):
        out[str(name)] = int(np.sum(idx == i))
    return out
