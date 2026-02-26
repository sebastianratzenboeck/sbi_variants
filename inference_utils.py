"""
Normalization utilities for SimFormer inference.

Loads the normalization statistics saved during training (norm_stats.npz)
and provides normalize/denormalize transforms for both torch tensors and
numpy arrays.
"""

import numpy as np
import torch
from typing import Optional, Sequence, Union
from value_transforms import (
    apply_forward_value_transforms_numpy,
    apply_forward_value_transforms_torch,
    apply_inverse_value_transforms_numpy,
    apply_inverse_value_transforms_torch,
)

_REQUIRED_POSITIVE_TRANSFORMS = {
    "rad": "log_shifted_pos",
    "Av": "log1p_pos",
}


class NormStats:
    """Load and apply normalization statistics saved during training.

    The training script (`train_mock_galaxy.py`) saves a ``norm_stats.npz``
    file containing per-column means, standard deviations, and column names.
    This class loads those stats and provides convenience methods for
    normalizing raw observations before sampling and denormalizing model
    outputs back to physical units.

    Usage::

        stats = NormStats("output/norm_stats.npz")

        # Normalize observed values before passing to the model
        x_norm = stats.normalize(x_raw)

        # Denormalize model output to physical units
        x_phys = stats.denormalize(samples)

        # Column name lookup
        idx = stats.column_index("logAge")
    """

    def __init__(self, norm_stats_path: str):
        """Load normalization statistics from an .npz file.

        Args:
            norm_stats_path: Path to the ``norm_stats.npz`` file produced
                by ``train_mock_galaxy.py``.
        """
        data = np.load(norm_stats_path, allow_pickle=True)
        self.means = data["means"].astype(np.float32)       # (num_nodes,)
        self.stds = data["stds"].astype(np.float32)         # (num_nodes,)
        self.columns = list(data["columns"])                 # list[str]
        self.num_nodes = len(self.means)
        self._col_to_idx = {name: i for i, name in enumerate(self.columns)}

        # Optional per-column value transforms.
        # New runs save this metadata.
        if "value_transform_names" in data and "value_transform_params" in data:
            self.value_transform_names = np.asarray(data["value_transform_names"], dtype=object)
            self.value_transform_params = np.asarray(data["value_transform_params"], dtype=np.float32)
        else:
            constrained_present = [
                c for c in _REQUIRED_POSITIVE_TRANSFORMS.keys() if c in self._col_to_idx
            ]
            if constrained_present:
                raise ValueError(
                    "Normalization metadata is missing value transforms, but constrained "
                    f"columns are present: {constrained_present}. Cannot guarantee "
                    "non-negativity for denormalized outputs. Re-export metadata from a "
                    "current training run (posterior_norm_meta_*.npz) or retrain/rebuild cache."
                )
            self.value_transform_names = np.asarray(
                ["identity"] * self.num_nodes, dtype=object
            )
            self.value_transform_params = np.zeros(self.num_nodes, dtype=np.float32)

        if self.value_transform_names.shape[0] != self.num_nodes:
            raise ValueError(
                f"len(value_transform_names)={self.value_transform_names.shape[0]} "
                f"does not match num_nodes={self.num_nodes}."
            )
        if self.value_transform_params.shape[0] != self.num_nodes:
            raise ValueError(
                f"len(value_transform_params)={self.value_transform_params.shape[0]} "
                f"does not match num_nodes={self.num_nodes}."
            )

        for name, expected in _REQUIRED_POSITIVE_TRANSFORMS.items():
            if name not in self._col_to_idx:
                continue
            idx = self._col_to_idx[name]
            got = str(self.value_transform_names[idx])
            if got != expected:
                raise ValueError(
                    f"Column '{name}' requires transform '{expected}' for positive-support "
                    f"denormalization, but metadata has '{got}'."
                )

        # Log-error normalization stats (for standardized error embedding)
        self.log_err_mean = float(data["log_err_mean"]) if "log_err_mean" in data else 0.0
        self.log_err_std = float(data["log_err_std"]) if "log_err_std" in data else 1.0

        # Optional input-column metadata for extended feature sets (e.g. colors).
        self.input_columns_base = (
            [str(c) for c in data["input_columns_base"].tolist()]
            if "input_columns_base" in data
            else None
        )
        self.input_columns_meta = (
            [str(c) for c in data["input_columns"].tolist()]
            if "input_columns" in data
            else None
        )

        self.use_colors = bool(np.asarray(data["use_colors"]).item()) if "use_colors" in data else False
        if "color_names" in data:
            self.color_names = [str(c) for c in data["color_names"].tolist()]
        else:
            self.color_names = []
        self.color_means = (
            np.asarray(data["color_means"], dtype=np.float32)
            if "color_means" in data
            else None
        )
        self.color_stds = (
            np.asarray(data["color_stds"], dtype=np.float32)
            if "color_stds" in data
            else None
        )
        if self.use_colors:
            if not self.color_names or self.color_means is None or self.color_stds is None:
                raise ValueError(
                    "Normalization metadata indicates use_colors=true but color statistics "
                    "(color_names/color_means/color_stds) are missing."
                )
            if len(self.color_names) != self.color_means.shape[0] or len(self.color_names) != self.color_stds.shape[0]:
                raise ValueError(
                    "Color metadata shape mismatch: "
                    f"len(color_names)={len(self.color_names)}, "
                    f"len(color_means)={self.color_means.shape[0]}, "
                    f"len(color_stds)={self.color_stds.shape[0]}."
                )

    # ------------------------------------------------------------------
    # Column lookup
    # ------------------------------------------------------------------

    def column_index(self, name: str) -> int:
        """Return the integer index for a column name.

        Raises KeyError if the name is not found.
        """
        return self._col_to_idx[name]

    def column_indices(self, names: Sequence[str]) -> list:
        """Return a list of integer indices for the given column names."""
        return [self._col_to_idx[n] for n in names]

    # ------------------------------------------------------------------
    # Torch operations
    # ------------------------------------------------------------------

    def _get_torch_stats(
        self,
        column_indices: Optional[Sequence[int]],
        device: Union[str, torch.device] = "cpu",
    ):
        """Return (means, stds) as torch tensors, optionally sub-selected."""
        if column_indices is not None:
            m = torch.tensor(self.means[list(column_indices)], device=device)
            s = torch.tensor(self.stds[list(column_indices)], device=device)
        else:
            m = torch.tensor(self.means, device=device)
            s = torch.tensor(self.stds, device=device)
        return m, s

    def _get_value_transform_meta(
        self,
        column_indices: Optional[Sequence[int]],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return transform metadata, optionally sub-selected by columns."""
        if column_indices is not None:
            idx = list(column_indices)
            names = self.value_transform_names[idx]
            params = self.value_transform_params[idx]
        else:
            names = self.value_transform_names
            params = self.value_transform_params
        return np.asarray(names, dtype=object), np.asarray(params, dtype=np.float32)

    def normalize(
        self,
        values: torch.Tensor,
        column_indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Normalize values: ``(x - mean) / std``.

        Args:
            values: Tensor of shape ``(..., D)`` where ``D`` is either
                ``num_nodes`` (when *column_indices* is None) or
                ``len(column_indices)``.
            column_indices: Optional subset of column indices.

        Returns:
            Normalized tensor, same shape as input.
        """
        names, params = self._get_value_transform_meta(column_indices)
        values = apply_forward_value_transforms_torch(values, names, params)
        m, s = self._get_torch_stats(column_indices, device=values.device)
        return (values - m) / s

    def denormalize(
        self,
        values: torch.Tensor,
        column_indices: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        """Denormalize values: ``x * std + mean``.

        Args:
            values: Tensor of shape ``(..., D)`` in normalized space.
            column_indices: Optional subset of column indices.

        Returns:
            Denormalized tensor in physical units.
        """
        m, s = self._get_torch_stats(column_indices, device=values.device)
        out = values * s + m
        names, params = self._get_value_transform_meta(column_indices)
        return apply_inverse_value_transforms_torch(out, names, params)

    # ------------------------------------------------------------------
    # Numpy operations
    # ------------------------------------------------------------------

    def normalize_numpy(
        self,
        values: np.ndarray,
        column_indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Normalize values: ``(x - mean) / std`` (numpy version).

        Args:
            values: Array of shape ``(..., D)``.
            column_indices: Optional subset of column indices.

        Returns:
            Normalized array.
        """
        names, params = self._get_value_transform_meta(column_indices)
        values = apply_forward_value_transforms_numpy(values, names, params)
        if column_indices is not None:
            m = self.means[list(column_indices)]
            s = self.stds[list(column_indices)]
        else:
            m, s = self.means, self.stds
        return (values - m) / s

    def denormalize_numpy(
        self,
        values: np.ndarray,
        column_indices: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Denormalize values: ``x * std + mean`` (numpy version).

        Args:
            values: Array of shape ``(..., D)`` in normalized space.
            column_indices: Optional subset of column indices.

        Returns:
            Denormalized array in physical units.
        """
        if column_indices is not None:
            m = self.means[list(column_indices)]
            s = self.stds[list(column_indices)]
        else:
            m, s = self.means, self.stds
        out = values * s + m
        names, params = self._get_value_transform_meta(column_indices)
        return apply_inverse_value_transforms_numpy(out, names, params)

    # ------------------------------------------------------------------
    # Error normalization
    # ------------------------------------------------------------------

    def normalize_errors(self, errors_raw: np.ndarray) -> np.ndarray:
        """Log-transform + standardize raw errors with sentinels.

        Matches the transform applied by ``build_arrays()`` during training:
        - ``error == 0`` (perfectly known) → -5
        - ``error > 0`` (real measurement) → ``(log(error) - mean) / std``
        - ``NaN`` (unobserved)             → +5

        Args:
            errors_raw: Array of shape ``(..., D)`` with raw error values.

        Returns:
            Array of same shape with standardized log-errors and sentinels.
        """
        LOG_ERR_PERFECT = -5.0
        LOG_ERR_UNOBS = 5.0

        has_real = (errors_raw > 0) & ~np.isnan(errors_raw)
        is_zero = (errors_raw == 0)

        errors_norm = np.full_like(errors_raw, LOG_ERR_UNOBS)
        errors_norm[has_real] = (
            np.log(errors_raw[has_real]) - self.log_err_mean
        ) / self.log_err_std
        errors_norm[is_zero] = LOG_ERR_PERFECT
        return errors_norm
