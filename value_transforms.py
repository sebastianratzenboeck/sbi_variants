"""Shared value-space transforms for constrained physical parameters.

These transforms let the model operate in an unconstrained space while
maintaining valid physical support after inverse-transform:
  - rad >= 0
  - Av >= 0
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch


IDENTITY = "identity"
LOG_SHIFTED_POS = "log_shifted_pos"
LOG1P_POS = "log1p_pos"


# Forward transform specs keyed by column name.
# param is interpreted as epsilon/shift where applicable.
_COLUMN_SPECS = {
    "rad": (LOG_SHIFTED_POS, 1e-6),
    "Av": (LOG1P_POS, 0.0),
}


def default_value_transform_metadata(
    columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-column transform names and parameters."""
    names = []
    params = []
    for c in columns:
        spec = _COLUMN_SPECS.get(str(c), (IDENTITY, 0.0))
        names.append(spec[0])
        params.append(float(spec[1]))
    return np.asarray(names, dtype=object), np.asarray(params, dtype=np.float32)


def _validate_shapes(
    width: int,
    transform_names: Sequence[str],
    transform_params: Sequence[float],
) -> None:
    if len(transform_names) != width:
        raise ValueError(
            f"transform_names has length {len(transform_names)}, expected {width}."
        )
    if len(transform_params) != width:
        raise ValueError(
            f"transform_params has length {len(transform_params)}, expected {width}."
        )


def apply_forward_value_transforms_numpy(
    values: np.ndarray,
    transform_names: Sequence[str],
    transform_params: Sequence[float],
) -> np.ndarray:
    """Apply forward transforms to the last axis of values (numpy)."""
    out = np.array(values, copy=True)
    width = out.shape[-1]
    _validate_shapes(width, transform_names, transform_params)

    names = np.asarray(transform_names)
    params = np.asarray(transform_params, dtype=np.float32)

    idx = np.where(names == LOG_SHIFTED_POS)[0]
    if idx.size:
        eps = params[idx].astype(out.dtype, copy=False)
        x = np.clip(out[..., idx], a_min=0.0, a_max=None)
        out[..., idx] = np.log(x + eps)

    idx = np.where(names == LOG1P_POS)[0]
    if idx.size:
        x = np.clip(out[..., idx], a_min=0.0, a_max=None)
        out[..., idx] = np.log1p(x)

    return out


def apply_inverse_value_transforms_numpy(
    values: np.ndarray,
    transform_names: Sequence[str],
    transform_params: Sequence[float],
) -> np.ndarray:
    """Apply inverse transforms to the last axis of values (numpy)."""
    out = np.array(values, copy=True)
    width = out.shape[-1]
    _validate_shapes(width, transform_names, transform_params)

    names = np.asarray(transform_names)
    params = np.asarray(transform_params, dtype=np.float32)

    idx = np.where(names == LOG_SHIFTED_POS)[0]
    if idx.size:
        eps = params[idx].astype(out.dtype, copy=False)
        out[..., idx] = np.clip(np.exp(out[..., idx]) - eps, a_min=0.0, a_max=None)

    idx = np.where(names == LOG1P_POS)[0]
    if idx.size:
        out[..., idx] = np.clip(np.expm1(out[..., idx]), a_min=0.0, a_max=None)

    return out


def apply_forward_value_transforms_torch(
    values: torch.Tensor,
    transform_names: Sequence[str],
    transform_params: Sequence[float],
) -> torch.Tensor:
    """Apply forward transforms to the last axis of values (torch)."""
    out = values.clone()
    width = out.shape[-1]
    _validate_shapes(width, transform_names, transform_params)

    names = np.asarray(transform_names)
    params = np.asarray(transform_params, dtype=np.float32)

    idx = np.where(names == LOG_SHIFTED_POS)[0]
    if idx.size:
        idx_t = torch.as_tensor(idx, device=out.device, dtype=torch.long)
        eps_t = torch.as_tensor(params[idx], device=out.device, dtype=out.dtype)
        x = out.index_select(-1, idx_t).clamp_min(0.0)
        out.index_copy_(-1, idx_t, torch.log(x + eps_t))

    idx = np.where(names == LOG1P_POS)[0]
    if idx.size:
        idx_t = torch.as_tensor(idx, device=out.device, dtype=torch.long)
        x = out.index_select(-1, idx_t).clamp_min(0.0)
        out.index_copy_(-1, idx_t, torch.log1p(x))

    return out


def apply_inverse_value_transforms_torch(
    values: torch.Tensor,
    transform_names: Sequence[str],
    transform_params: Sequence[float],
) -> torch.Tensor:
    """Apply inverse transforms to the last axis of values (torch)."""
    out = values.clone()
    width = out.shape[-1]
    _validate_shapes(width, transform_names, transform_params)

    names = np.asarray(transform_names)
    params = np.asarray(transform_params, dtype=np.float32)

    idx = np.where(names == LOG_SHIFTED_POS)[0]
    if idx.size:
        idx_t = torch.as_tensor(idx, device=out.device, dtype=torch.long)
        eps_t = torch.as_tensor(params[idx], device=out.device, dtype=out.dtype)
        x = out.index_select(-1, idx_t).exp() - eps_t
        out.index_copy_(-1, idx_t, x.clamp_min(0.0))

    idx = np.where(names == LOG1P_POS)[0]
    if idx.size:
        idx_t = torch.as_tensor(idx, device=out.device, dtype=torch.long)
        x = torch.expm1(out.index_select(-1, idx_t)).clamp_min(0.0)
        out.index_copy_(-1, idx_t, x)

    return out
