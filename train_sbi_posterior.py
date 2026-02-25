#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Sampler

# Support both:
#   python -m sbi_variants.train_sbi_posterior
#   python sbi_variants/train_sbi_posterior.py
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from sbi_variants.data import (
    CacheArrays,
    DEFAULT_INPUT_COLS,
    DEFAULT_THETA_COLS,
    SBIDataset,
    build_row_split,
    build_sbi_arrays,
    load_cache_arrays,
    load_indices,
    parse_column_csv,
)
from sbi_variants.encoder import ObservationEncoder
from sbi_variants.posterior_models import (
    ConditionalFMPosterior,
    ConditionalFlowPosterior,
)
from sbi_variants.value_transforms import apply_inverse_value_transforms_numpy
from train_mock_galaxy import (
    DEFAULT_CLUSTER_ID_COL,
    build_arrays as build_cache_arrays,
    load_data as load_raw_data,
    save_arrays as save_cache_arrays,
)

_BIN_SAMPLER_CHUNK_SIZE = 1_000_000
_REQUIRED_POSITIVE_TRANSFORMS = {
    "rad": "log_shifted_pos",
    "Av": "log1p_pos",
}


class _JointBinFirstSampler(Sampler[int]):
    """Sample bins from q_bin, then sample rows uniformly within each chosen bin."""

    def __init__(
        self,
        *,
        active_bins: np.ndarray,
        q_bin: np.ndarray,
        sorted_indices_by_bin: np.ndarray,
        bin_offsets: np.ndarray,
        num_samples: int,
        seed: int,
        chunk_size: int = _BIN_SAMPLER_CHUNK_SIZE,
    ):
        active_bins = np.asarray(active_bins, dtype=np.int64)
        bin_offsets = np.asarray(bin_offsets, dtype=np.int64)
        sorted_indices_by_bin = np.asarray(sorted_indices_by_bin, dtype=np.int64)
        q_bin = np.asarray(q_bin, dtype=np.float64)

        if active_bins.ndim != 1 or active_bins.size == 0:
            raise ValueError("active_bins must be a non-empty 1-D array.")
        if q_bin.ndim != 1:
            raise ValueError(f"q_bin must be 1-D, got shape={q_bin.shape}")
        if bin_offsets.ndim != 1 or bin_offsets.size != (q_bin.size + 1):
            raise ValueError(
                "bin_offsets must be 1-D with length n_bins+1. "
                f"Got len={bin_offsets.size}, n_bins={q_bin.size}."
            )
        if int(bin_offsets[-1]) != int(sorted_indices_by_bin.size):
            raise ValueError(
                "sorted_indices_by_bin length must match bin_offsets[-1]. "
                f"Got {sorted_indices_by_bin.size} vs {int(bin_offsets[-1])}."
            )
        if int(num_samples) <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        if int(chunk_size) <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

        q_active = q_bin[active_bins].astype(np.float64, copy=True)
        q_active_sum = float(q_active.sum())
        if q_active_sum <= 0.0:
            raise ValueError("q_bin has no positive mass on active bins.")
        q_active /= q_active_sum

        self.active_bins = active_bins
        self.q_active = q_active
        self.sorted_indices_by_bin = sorted_indices_by_bin
        self.bin_offsets = bin_offsets
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed)
        remaining = self.num_samples
        while remaining > 0:
            m = min(self.chunk_size, remaining)
            remaining -= m

            sampled_bins = rng.choice(
                self.active_bins,
                size=m,
                replace=True,
                p=self.q_active,
            )
            sampled_rows = np.empty(m, dtype=np.int64)
            unique_bins, inverse = np.unique(sampled_bins, return_inverse=True)
            for j, b in enumerate(unique_bins):
                mask = inverse == j
                n_b = int(mask.sum())
                lo = int(self.bin_offsets[int(b)])
                hi = int(self.bin_offsets[int(b) + 1])
                draw_pos = rng.integers(lo, hi, size=n_b)
                sampled_rows[mask] = self.sorted_indices_by_bin[draw_pos]

            for idx in sampled_rows:
                yield int(idx)

    def __len__(self) -> int:
        return self.num_samples


def _build_epoch_sampler(
    *,
    state: dict[str, np.ndarray | float | int],
    q_bin: np.ndarray,
    num_samples: int,
    seed: int,
) -> Sampler[int]:
    return _JointBinFirstSampler(
        active_bins=state["active_bins"],
        q_bin=q_bin,
        sorted_indices_by_bin=state["sorted_indices_by_bin"],
        bin_offsets=state["bin_offsets"],
        num_samples=num_samples,
        seed=seed,
    )

try:
    from torch.amp import GradScaler, autocast

    def _autocast_context(enabled: bool, device: str):
        return autocast("cuda", enabled=(enabled and device != "cpu"))

except ImportError:
    from torch.cuda.amp import GradScaler, autocast

    def _autocast_context(enabled: bool, device: str):
        return autocast(enabled=(enabled and device != "cpu"))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train direct SBI posterior p(theta | x_obs) using shared Simformer-style encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="Optional JSON config file. CLI flags override config values.")
    p.add_argument("--cache-path", type=str, default=None,
                   help="Path to build_arrays_cache.npz.")
    p.add_argument(
        "--data-path",
        type=str,
        default=None,
        help=(
            "Optional Parquet/CSV path used to build cache arrays when --cache-path "
            "is missing or --rebuild-cache is enabled."
        ),
    )
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory for checkpoints/configs.")
    p.add_argument("--run-name", type=str, default="sbi_variant")
    p.add_argument("--method", type=str, default="flow_matching",
                   choices=["flow_matching", "normalizing_flow", "realnvp"])

    p.add_argument("--input-columns", type=str, default=",".join(DEFAULT_INPUT_COLS),
                   help="Comma-separated input columns used as x_obs.")
    p.add_argument("--theta-columns", type=str, default=",".join(DEFAULT_THETA_COLS),
                   help="Comma-separated theta columns to model in posterior.")
    p.add_argument("--exclude-indices", type=str, default=None,
                   help="Optional .npy indices excluded from train/val (e.g. test_indices.npy).")
    p.add_argument(
        "--cluster-id-col",
        type=str,
        default=DEFAULT_CLUSTER_ID_COL,
        help=(
            "Cluster ID column used only when building cache from --data-path "
            "(stored as metadata for cluster-aware test holdout)."
        ),
    )
    p.add_argument(
        "--rebuild-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force rebuilding cache from --data-path even if cache file exists.",
    )
    p.add_argument(
        "--test-split",
        type=float,
        default=0.0,
        help=(
            "Optional test holdout fraction in [0,1). Held-out rows are excluded from train/val "
            "and saved to output_dir/test_indices.npy (ignored when --exclude-indices is provided)."
        ),
    )
    p.add_argument(
        "--test-cluster-frac",
        type=float,
        default=0.0,
        help=(
            "If >0, hold out this fraction of unique positive cluster IDs as full clusters for test "
            "(requires cache to contain cluster_ids metadata; ignored when --exclude-indices is provided)."
        ),
    )
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--max-stars", type=int, default=None,
                   help="Optional cap on train+val rows after exclusions.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--young-logage-threshold",
        type=float,
        default=7.8,
        help="Physical logAge threshold used for young-star validation subset.",
    )
    p.add_argument(
        "--young-eval-max-stars",
        type=int,
        default=0,
        help=(
            "Max stars for young-star validation loss each epoch (0 disables). "
            "Subset is sampled from validation rows."
        ),
    )
    p.add_argument(
        "--random-eval-max-stars",
        type=int,
        default=0,
        help=(
            "Max stars for random unweighted validation loss each epoch (0 disables). "
            "Subset is sampled from validation rows."
        ),
    )
    p.add_argument(
        "--train-random-eval-max-stars",
        type=int,
        default=0,
        help=(
            "Max stars for random unweighted train loss each epoch (0 disables). "
            "Subset is sampled from training rows under the natural distribution."
        ),
    )
    p.add_argument(
        "--val-curriculum-loss",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also log validation loss under the same curriculum distribution as train "
            "(standard val_loss remains natural/unweighted)."
        ),
    )
    p.add_argument(
        "--val-curriculum-epoch-size",
        type=int,
        default=0,
        help=(
            "Samples for val curriculum loss per epoch (0 => len(val)). "
            "Only used when --val-curriculum-loss is enabled."
        ),
    )

    # Joint curriculum / importance weighting
    p.add_argument(
        "--joint-curriculum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable joint (logAge,m_init) curriculum sampling "
            "q=(1-lambda)p+lambda/K (default: enabled)."
        ),
    )
    p.add_argument("--n-bins", type=int, default=25,
                   help="Number of logAge bins for joint curriculum.")
    p.add_argument("--n-mass-bins", type=int, default=12,
                   help="Number of m_init bins for joint curriculum.")
    p.add_argument("--tau-max", type=float, default=0.8,
                   help="Max tau in curriculum schedule (tau=0 => uniform bins, tau=1 => natural bins).")
    p.add_argument("--tau-warmup", type=int, default=10,
                   help="Epochs to keep tau=0 before ramping to tau-max.")
    p.add_argument("--curriculum-epoch-size", type=int, default=0,
                   help="Samples drawn per epoch when joint curriculum is enabled (0 => len(train)).")
    p.add_argument(
        "--importance-weighting",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply p/q importance correction in loss for curriculum-sampled batches (default: enabled).",
    )
    p.add_argument("--importance-weight-beta", type=float, default=1.0,
                   help=(
                       "Tempered IS exponent: w=(p/q)^beta. "
                       "1.0=full IS correction (natural-distribution gradient), "
                       "0.0=no correction (pure curriculum effect). "
                       "Values in (0,1) give a compromise."
                   ))
    p.add_argument("--importance-weight-min", type=float, default=0.5,
                   help="Safety lower clamp for importance weights (after beta tempering).")
    p.add_argument("--importance-weight-max", type=float, default=2.0,
                   help="Safety upper clamp for importance weights (after beta tempering).")
    p.add_argument("--nll-cap", type=float, default=0.0,
                   help=(
                       "Smooth upper cap (softplus-based) applied to per-sample NLL "
                       "during training for gradient stability.  0 disables the cap. "
                       "Recommended ~500 for normalizing-flow training with curriculum."
                   ))

    # Encoder
    p.add_argument("--dim-value", type=int, default=24)
    p.add_argument("--dim-id", type=int, default=24)
    p.add_argument("--dim-error", type=int, default=16)
    p.add_argument("--dim-observed", type=int, default=8)
    p.add_argument("--value-calibration-type", type=str, default="scalar_film",
                   choices=["none", "scalar_film"])
    p.add_argument("--error-embed-type", type=str, default="mlp_regime",
                   choices=["rff", "mlp_regime"])
    p.add_argument("--attn-embed-dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--widening-factor", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--use-missingness-context", action="store_true", default=False)
    p.add_argument("--missingness-context-hidden-dim", type=int, default=64)

    # Flow-matching head
    p.add_argument("--time-prior-exponent", type=float, default=0.0)
    p.add_argument("--sigma-min", type=float, default=1e-3)
    p.add_argument("--time-embed-dim", type=int, default=64)
    p.add_argument("--fm-hidden-dim", type=int, default=256)

    # Normalizing-flow head (package-backed)
    p.add_argument("--nf-hidden-dim", type=int, default=256)
    p.add_argument("--nf-backend", type=str, default="zuko", choices=["zuko", "nflows"],
                   help="Flow backend package.")
    p.add_argument("--nf-family", type=str, default="nsf", choices=["nsf", "maf", "nice"],
                   help="Flow family (for nflows backend, 'nice' is unsupported).")
    p.add_argument("--nf-num-coupling-layers", type=int, default=8)
    p.add_argument("--nf-max-scale", type=float, default=2.0,
                   help="Deprecated for package-backed flows; kept for CLI compatibility.")

    # Optimization
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--device", type=str, default=None)

    # Logging
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", type=str, default="mock-galaxy-simformer")
    return p


def _load_config_defaults(config_path: str, parser: argparse.ArgumentParser) -> dict:
    with open(config_path) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config at {config_path} must be a JSON object (dict).")
    valid = {a.dest for a in parser._actions}
    unknown = sorted(k for k in cfg.keys() if k not in valid)
    if unknown:
        raise ValueError(
            f"Unknown config keys in {config_path}: {unknown[:8]}"
            + ("..." if len(unknown) > 8 else "")
        )
    return cfg


def parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    parser = _build_parser()
    if pre_args.config:
        cfg = _load_config_defaults(pre_args.config, parser)
        parser.set_defaults(**cfg)

    args = parser.parse_args()

    if isinstance(args.cache_path, str):
        cache_norm = args.cache_path.strip().lower()
        if cache_norm in ("", "none", "null"):
            args.cache_path = None
    if isinstance(args.data_path, str):
        data_norm = args.data_path.strip().lower()
        if data_norm in ("", "none", "null"):
            args.data_path = None
    if isinstance(args.exclude_indices, str):
        norm = args.exclude_indices.strip().lower()
        if norm in ("", "none", "null"):
            args.exclude_indices = None

    missing = []
    if not args.output_dir:
        missing.append("--output-dir")
    if missing:
        parser.error(
            "Missing required arguments after applying config/CLI: "
            + ", ".join(missing)
        )
    if args.n_bins <= 0:
        parser.error(f"--n-bins must be > 0, got {args.n_bins}")
    if args.n_mass_bins <= 0:
        parser.error(f"--n-mass-bins must be > 0, got {args.n_mass_bins}")
    if args.curriculum_epoch_size < 0:
        parser.error(
            f"--curriculum-epoch-size must be >= 0, got {args.curriculum_epoch_size}"
        )
    if args.val_curriculum_epoch_size < 0:
        parser.error(
            "--val-curriculum-epoch-size must be >= 0, got "
            f"{args.val_curriculum_epoch_size}"
        )
    if args.young_eval_max_stars < 0:
        parser.error(f"--young-eval-max-stars must be >= 0, got {args.young_eval_max_stars}")
    if args.random_eval_max_stars < 0:
        parser.error(f"--random-eval-max-stars must be >= 0, got {args.random_eval_max_stars}")
    if args.train_random_eval_max_stars < 0:
        parser.error(
            f"--train-random-eval-max-stars must be >= 0, got {args.train_random_eval_max_stars}"
        )
    if not (0.0 <= args.test_split < 1.0):
        parser.error(f"--test-split must be in [0,1), got {args.test_split}")
    if not (0.0 <= args.test_cluster_frac <= 1.0):
        parser.error(f"--test-cluster-frac must be in [0,1], got {args.test_cluster_frac}")
    if args.importance_weight_min <= 0:
        parser.error(
            f"--importance-weight-min must be > 0, got {args.importance_weight_min}"
        )
    if args.importance_weight_max < args.importance_weight_min:
        parser.error(
            "--importance-weight-max must be >= --importance-weight-min "
            f"({args.importance_weight_max} < {args.importance_weight_min})"
        )
    if not (0.0 <= args.importance_weight_beta <= 1.0):
        parser.error(
            f"--importance-weight-beta must be in [0,1], got {args.importance_weight_beta}"
        )
    if args.nll_cap < 0:
        parser.error(f"--nll-cap must be >= 0, got {args.nll_cap}")
    return args


def _ensure_cache(args: argparse.Namespace) -> str:
    """Return an existing cache path or build cache from raw data if needed."""
    cache_path = args.cache_path
    if cache_path is None:
        cache_path = os.path.join(args.output_dir, "build_arrays_cache.npz")
    cache_path = os.path.abspath(cache_path)

    if os.path.exists(cache_path) and not args.rebuild_cache:
        print(f"Using cache: {cache_path}")
        return cache_path

    if args.data_path is None:
        raise ValueError(
            "Cache file is missing (or rebuild requested) but --data-path is not set. "
            "Provide --data-path to build cache, or pass an existing --cache-path."
        )

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    print(f"Building cache at {cache_path} from raw data: {args.data_path}")
    df = load_raw_data(args.data_path)
    (
        values_norm,
        errors_norm,
        observed_mask,
        means,
        stds,
        value_transform_names,
        value_transform_params,
        cluster_ids,
        log_err_mean,
        log_err_std,
    ) = build_cache_arrays(df, cluster_id_col=args.cluster_id_col)
    save_cache_arrays(
        cache_path,
        values_norm,
        errors_norm,
        observed_mask,
        means,
        stds,
        value_transform_names,
        value_transform_params,
        cluster_ids,
        log_err_mean,
        log_err_std,
    )
    return cache_path


def _compute_tau(epoch: int, total_epochs: int, tau_max: float, tau_warmup: int) -> float:
    if epoch < tau_warmup:
        return 0.0
    ramp_epochs = total_epochs - tau_warmup
    if ramp_epochs <= 0:
        return float(tau_max)
    progress = (epoch - tau_warmup) / float(ramp_epochs)
    return float(tau_max) * min(max(progress, 0.0), 1.0)


def _prepare_joint_curriculum_state(
    theta: np.ndarray,
    theta_columns: list[str],
    n_age_bins: int,
    n_mass_bins: int,
) -> dict[str, np.ndarray | float | int]:
    try:
        age_idx = theta_columns.index("logAge")
        mass_idx = theta_columns.index("m_init")
    except ValueError as e:
        raise ValueError(
            "Joint curriculum requires theta_columns to include both 'logAge' and 'm_init'."
        ) from e

    age = theta[:, age_idx]
    mass = theta[:, mass_idx]
    if not np.all(np.isfinite(age)):
        n_bad = int((~np.isfinite(age)).sum())
        raise ValueError(
            f"Joint curriculum input has {n_bad} non-finite logAge values."
        )
    if not np.all(np.isfinite(mass)):
        n_bad = int((~np.isfinite(mass)).sum())
        raise ValueError(
            f"Joint curriculum input has {n_bad} non-finite m_init values."
        )
    age_edges = np.linspace(age.min(), age.max() + 1e-6, n_age_bins + 1)
    mass_edges = np.linspace(mass.min(), mass.max() + 1e-6, n_mass_bins + 1)

    age_bin = np.digitize(age, age_edges) - 1
    mass_bin = np.digitize(mass, mass_edges) - 1
    age_bin = np.clip(age_bin, 0, n_age_bins - 1)
    mass_bin = np.clip(mass_bin, 0, n_mass_bins - 1)

    joint = (age_bin * n_mass_bins + mass_bin).astype(np.int32, copy=False)
    n_joint_bins = int(n_age_bins * n_mass_bins)
    counts = np.bincount(joint, minlength=n_joint_bins).astype(np.int64)
    active = counts > 0
    if active.sum() == 0:
        raise ValueError("Joint curriculum found no active bins.")

    p_bin = np.zeros(n_joint_bins, dtype=np.float64)
    counts_f = counts.astype(np.float64)
    p_bin[active] = counts_f[active] / counts_f[active].sum()

    active_bins = np.flatnonzero(active).astype(np.int64)
    sorted_pos = np.argsort(joint, kind="stable").astype(np.int64, copy=False)
    bin_offsets = np.zeros(n_joint_bins + 1, dtype=np.int64)
    bin_offsets[1:] = np.cumsum(counts, dtype=np.int64)

    return {
        "joint": joint,
        "counts": counts,
        "p_bin": p_bin,
        "active": active,
        "active_bins": active_bins,
        "sorted_indices_by_bin": sorted_pos,
        "bin_offsets": bin_offsets,
        "n_active": int(active.sum()),
    }


def _cache_column_physical_values(
    cache: CacheArrays, row_indices: np.ndarray, column_name: str,
) -> np.ndarray:
    """Recover one cache column in physical units for the requested rows."""
    if column_name not in cache.columns:
        raise ValueError(
            f"Column '{column_name}' not found in cache columns."
        )
    if cache.means is None or cache.stds is None:
        raise ValueError(
            "Cache is missing means/stds metadata required to recover physical "
            f"'{column_name}' values. Rebuild cache with current pipeline."
        )
    col_idx = cache.columns.index(column_name)
    std = float(cache.stds[col_idx])
    mean = float(cache.means[col_idx])
    if (not np.isfinite(std)) or (std <= 0.0):
        raise ValueError(
            f"Invalid std for column '{column_name}': std={std}. "
            "Expected finite positive std in cache metadata."
        )

    rows = np.asarray(row_indices, dtype=np.int64)
    vals = cache.values_norm[rows, col_idx].astype(np.float64, copy=False)
    vals = vals * std + mean

    if cache.value_transform_names is not None and cache.value_transform_params is not None:
        vals = apply_inverse_value_transforms_numpy(
            vals.reshape(-1, 1),
            transform_names=np.asarray([cache.value_transform_names[col_idx]], dtype=object),
            transform_params=np.asarray([cache.value_transform_params[col_idx]], dtype=np.float32),
        ).reshape(-1)
    return vals.astype(np.float32, copy=False)


def _validate_curriculum_space_assumptions(
    cache: CacheArrays, theta_columns: list[str],
) -> None:
    """Validate assumptions behind binning directly on normalized theta arrays."""
    if cache.means is None or cache.stds is None:
        raise ValueError(
            "Joint curriculum requires cache means/stds metadata for robust "
            "physical-space diagnostics. Rebuild cache with current pipeline."
        )

    if cache.value_transform_names is None:
        return

    for name in ("logAge", "m_init"):
        if name not in theta_columns:
            continue
        if name not in cache.columns:
            raise ValueError(
                f"Joint curriculum requested '{name}', but cache columns are missing it."
            )
        idx = cache.columns.index(name)
        transform_name = str(cache.value_transform_names[idx])
        if transform_name != "identity":
            raise ValueError(
                "Joint curriculum currently bins on standardized theta values, "
                f"which matches physical equal-width bins only for affine scaling. "
                f"Found non-identity transform '{transform_name}' for '{name}'."
            )


def _validate_positive_support_transforms(
    cache: CacheArrays, theta_columns: list[str],
) -> None:
    """Require constrained-parameter transforms for modeled positive thetas."""
    constrained = [c for c in theta_columns if c in _REQUIRED_POSITIVE_TRANSFORMS]
    if not constrained:
        return

    if cache.value_transform_names is None or cache.value_transform_params is None:
        raise ValueError(
            "Theta includes constrained positive parameters "
            f"{constrained}, but cache transform metadata is missing. "
            "Cannot guarantee non-negativity support for posterior samples. "
            "Rebuild cache with current preprocessing (--rebuild-cache --data-path ...)."
        )

    for name in constrained:
        if name not in cache.columns:
            raise ValueError(
                f"Theta includes '{name}', but cache columns are missing it."
            )
        idx = cache.columns.index(name)
        expected = _REQUIRED_POSITIVE_TRANSFORMS[name]
        got = str(cache.value_transform_names[idx])
        if got != expected:
            raise ValueError(
                f"Theta includes constrained parameter '{name}', expected transform "
                f"'{expected}', but cache metadata has '{got}'. "
                "This would break guaranteed non-negativity after denormalization. "
                "Rebuild cache with current preprocessing."
            )


def _compute_test_split_with_cluster_holdout(
    n_total: int,
    test_split: float,
    *,
    cluster_ids: np.ndarray | None = None,
    test_cluster_frac: float = 0.0,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (trainval_indices, test_indices, heldout_cluster_ids)."""
    all_indices = np.arange(n_total, dtype=np.int64)
    rng = np.random.RandomState(random_state)

    # Pure random test split (or no split).
    if (cluster_ids is None) or (test_cluster_frac <= 0.0):
        if test_split <= 0.0:
            return (
                all_indices.astype(np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )
        trainval_idx, test_idx = train_test_split(
            all_indices, test_size=test_split, random_state=random_state
        )
        return (
            trainval_idx.astype(np.int64),
            test_idx.astype(np.int64),
            np.array([], dtype=np.int64),
        )

    cluster_ids = np.asarray(cluster_ids)
    if cluster_ids.shape[0] != n_total:
        raise ValueError(
            f"cluster_ids length mismatch: got {cluster_ids.shape[0]}, expected {n_total}"
        )

    unique_clusters = np.unique(cluster_ids[cluster_ids > 0])
    if unique_clusters.size == 0:
        print("WARNING: no positive cluster IDs found; falling back to random test split.")
        if test_split <= 0.0:
            return (
                all_indices.astype(np.int64),
                np.array([], dtype=np.int64),
                np.array([], dtype=np.int64),
            )
        trainval_idx, test_idx = train_test_split(
            all_indices, test_size=test_split, random_state=random_state
        )
        return (
            trainval_idx.astype(np.int64),
            test_idx.astype(np.int64),
            np.array([], dtype=np.int64),
        )

    n_holdout_clusters = int(round(float(test_cluster_frac) * float(unique_clusters.size)))
    n_holdout_clusters = max(1, min(n_holdout_clusters, int(unique_clusters.size)))
    heldout_clusters = np.sort(
        rng.choice(unique_clusters, size=n_holdout_clusters, replace=False).astype(np.int64)
    )
    is_cluster_test = np.isin(cluster_ids, heldout_clusters)
    test_idx_cluster = all_indices[is_cluster_test]
    remaining_idx = all_indices[~is_cluster_test]

    desired_test_count = int(round(float(test_split) * float(n_total)))
    desired_test_count = max(desired_test_count, len(test_idx_cluster))

    n_needed = desired_test_count - len(test_idx_cluster)
    if n_needed > 0 and len(remaining_idx) > 0:
        n_needed = min(n_needed, len(remaining_idx))
        extra_test = rng.choice(remaining_idx, size=n_needed, replace=False).astype(np.int64)
        test_idx = np.concatenate([test_idx_cluster, extra_test])
    else:
        test_idx = test_idx_cluster

    test_idx = np.unique(test_idx.astype(np.int64))
    trainval_idx = all_indices[~np.isin(all_indices, test_idx)].astype(np.int64)
    return trainval_idx, test_idx, heldout_clusters


def _joint_curriculum_distributions(
    state: dict[str, np.ndarray | float | int],
    tau: float,
    *,
    importance_weighting: bool,
    importance_weight_beta: float,
    importance_weight_min: float,
    importance_weight_max: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute curriculum sampling distribution and importance weights.

    Returns:
        q_bin: Sampling distribution over joint bins.
        w_i: Per-sample importance weights (tempered, clipped, re-normalised).
        lam: Mixture parameter (1 = fully uniform bins, 0 = natural bins).
        clip_frac: Fraction of samples whose weights were clipped by the
            safety bounds.  Should stay near 0 if beta and bounds are well
            chosen; persistently high values indicate the bounds are actively
            changing the optimisation objective.
    """
    joint = state["joint"]
    p_bin = state["p_bin"]
    active = state["active"]
    n_active = int(state["n_active"])

    lam = float(np.clip(1.0 - tau, 0.0, 1.0))
    q_bin = np.zeros_like(p_bin, dtype=np.float64)
    q_bin[active] = (1.0 - lam) * p_bin[active] + lam * (1.0 / float(n_active))

    if importance_weighting:
        raw = (p_bin[joint] / q_bin[joint]).astype(np.float64)
        # Tempered IS: w = (p/q)^beta.
        #   beta=1 → full IS correction (equivalent to natural-distribution gradient).
        #   beta=0 → no correction (pure curriculum effect on gradient).
        #   0<beta<1 → partial correction (compromise).
        if importance_weight_beta != 1.0:
            raw = np.power(raw, float(importance_weight_beta))
        w_i = (raw / max(float(raw.mean()), 1e-8)).astype(np.float32)
        # Compute clip fraction *before* clipping for diagnostics.
        n_total = len(w_i)
        n_clipped = int(
            np.sum(w_i < float(importance_weight_min))
            + np.sum(w_i > float(importance_weight_max))
        )
        clip_frac = n_clipped / max(n_total, 1)
        # Safety clamp (wide bounds; should rarely trigger with tempered beta).
        w_i = np.clip(
            w_i, float(importance_weight_min), float(importance_weight_max),
        ).astype(np.float32)
        w_i /= max(float(w_i.mean()), 1e-8)
    else:
        w_i = np.ones_like(joint, dtype=np.float32)
        clip_frac = 0.0

    return q_bin.astype(np.float64), w_i, lam, clip_frac


def _build_model(args: argparse.Namespace, input_columns: list[str], theta_dim: int) -> torch.nn.Module:
    encoder = ObservationEncoder(
        input_columns=input_columns,
        dim_value=args.dim_value,
        dim_id=args.dim_id,
        value_calibration_type=args.value_calibration_type,
        dim_error=args.dim_error,
        error_embed_type=args.error_embed_type,
        dim_observed=args.dim_observed,
        attn_embed_dim=args.attn_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        widening_factor=args.widening_factor,
        dropout=args.dropout,
        use_missingness_context=args.use_missingness_context,
        missingness_context_hidden_dim=args.missingness_context_hidden_dim,
    )

    if args.method == "flow_matching":
        return ConditionalFMPosterior(
            encoder=encoder,
            theta_dim=theta_dim,
            hidden_dim=args.fm_hidden_dim,
            time_embed_dim=args.time_embed_dim,
            sigma_min=args.sigma_min,
            time_prior_exponent=args.time_prior_exponent,
            dropout=args.dropout,
        )
    if args.method in ("normalizing_flow", "realnvp"):
        return ConditionalFlowPosterior(
            encoder=encoder,
            theta_dim=theta_dim,
            backend=args.nf_backend,
            flow_family=args.nf_family,
            num_transforms=args.nf_num_coupling_layers,
            hidden_dim=args.nf_hidden_dim,
            dropout=args.dropout,
        )
    raise ValueError(f"Unsupported method: {args.method}")


def _move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _epoch_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str,
    *,
    train: bool,
    optimizer: AdamW | None = None,
    scaler: GradScaler | None = None,
    use_amp: bool = False,
    grad_clip_norm: float = 1.0,
    nll_cap: float = 0.0,
) -> dict[str, float]:
    """Run one epoch and return sample-count-weighted loss statistics.

    Returns a dict with keys:
        ``"weighted"`` – global optimisation objective assembled as an exact
            ratio of summed weighted numerators/denominators across batches
            (importance-weighted, optionally smooth-capped during training).
        ``"nll_mean"`` – true sample-mean of raw per-sample NLL/MSE values
            (unweighted, uncapped).  Comparable across different sampling
            distributions.
        ``"n_samples"`` – total number of samples processed.
    """
    if train:
        model.train()
    else:
        model.eval()

    total_weighted_num = 0.0   # accumulates sum(w * nll_capped) across batches
    total_weighted_den = 0.0   # accumulates sum(w) across batches
    total_nll = 0.0
    total_samples = 0
    # Only apply the smooth NLL cap during training (backprop stabilisation);
    # evaluation always uses raw NLL so that monitored metrics are unbiased.
    effective_cap = nll_cap if train else 0.0
    for batch in loader:
        batch = _move_batch(batch, device=device)
        if train:
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(use_amp, device):
                result = model.loss(
                    theta=batch["theta"],
                    values=batch["inputs"],
                    errors=batch["errors"],
                    observed_mask=batch["observed"],
                    sample_weights=batch.get("sample_weight"),
                    nll_cap=effective_cap,
                )
            scaler.scale(result.loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                with _autocast_context(use_amp, device):
                    result = model.loss(
                        theta=batch["theta"],
                        values=batch["inputs"],
                        errors=batch["errors"],
                        observed_mask=batch["observed"],
                        sample_weights=batch.get("sample_weight"),
                        nll_cap=effective_cap,
                    )
        # Combine per-batch SNIS numerators/denominators for exact global ratio.
        # loss = sum(w*nll_cap)/sum(w), so loss*w_sum = sum(w*nll_cap).
        total_weighted_num += float(result.loss.item()) * result.w_sum
        total_weighted_den += result.w_sum
        total_nll += float(result.nll_sum.item())
        total_samples += result.n_samples

    n = max(total_samples, 1)
    return {
        "weighted": total_weighted_num / max(total_weighted_den, 1e-8),
        "nll_mean": total_nll / n,
        "n_samples": total_samples,
    }


def _build_eval_loader_from_rows(
    *,
    cache,
    row_indices: np.ndarray,
    input_columns: list[str],
    theta_columns: list[str],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    arr = build_sbi_arrays(
        cache,
        row_indices=row_indices,
        input_columns=input_columns,
        theta_columns=theta_columns,
    )
    ds = SBIDataset(arr)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cache_path = _ensure_cache(args)
    args.cache_path = cache_path
    cache = load_cache_arrays(cache_path)
    input_columns = parse_column_csv(args.input_columns)
    theta_columns = parse_column_csv(args.theta_columns)
    if len(input_columns) == 0:
        raise ValueError("input_columns resolved to empty list.")
    if len(theta_columns) == 0:
        raise ValueError("theta_columns resolved to empty list.")
    _validate_positive_support_transforms(cache, theta_columns)

    generated_test_idx_path = None
    generated_test_cluster_ids_path = None
    if args.exclude_indices is not None:
        exclude_idx = load_indices(args.exclude_indices)
        if args.test_split > 0.0 or args.test_cluster_frac > 0.0:
            print(
                "WARNING: --exclude-indices is set; ignoring --test-split/--test-cluster-frac."
            )
    else:
        if args.test_cluster_frac > 0.0 and cache.cluster_ids is None:
            raise ValueError(
                "--test-cluster-frac requires cache metadata 'cluster_ids'. "
                "Rebuild cache from train_mock_galaxy.py with cluster_ID present, "
                "or pass --exclude-indices."
            )
        _, test_rows, heldout_clusters = _compute_test_split_with_cluster_holdout(
            n_total=cache.values_norm.shape[0],
            test_split=args.test_split,
            cluster_ids=cache.cluster_ids,
            test_cluster_frac=args.test_cluster_frac,
            random_state=args.seed,
        )
        if len(test_rows) > 0:
            generated_test_idx_path = os.path.join(args.output_dir, "test_indices.npy")
            np.save(generated_test_idx_path, test_rows.astype(np.int64))
            print(f"Saved generated test indices: {generated_test_idx_path} ({len(test_rows):,} rows)")
        if len(heldout_clusters) > 0:
            generated_test_cluster_ids_path = os.path.join(args.output_dir, "test_cluster_ids.npy")
            np.save(generated_test_cluster_ids_path, heldout_clusters.astype(np.int64))
            n_cluster_rows = int(np.isin(cache.cluster_ids, heldout_clusters).sum())
            print(
                "Cluster holdout: "
                f"{len(heldout_clusters)} clusters, {n_cluster_rows:,} rows "
                f"(saved to {generated_test_cluster_ids_path})"
            )
        # Exclude all generated test rows from train/val.
        exclude_idx = test_rows if len(test_rows) > 0 else None

    train_rows, val_rows = build_row_split(
        n_rows=cache.values_norm.shape[0],
        exclude_indices=exclude_idx,
        val_split=args.val_split,
        seed=args.seed,
        max_rows=args.max_stars,
    )
    arr_train = build_sbi_arrays(
        cache,
        row_indices=train_rows,
        input_columns=input_columns,
        theta_columns=theta_columns,
    )
    arr_val = build_sbi_arrays(
        cache,
        row_indices=val_rows,
        input_columns=input_columns,
        theta_columns=theta_columns,
    )

    train_ds = SBIDataset(arr_train)
    val_ds = SBIDataset(arr_val)
    pin_memory = (device != "cpu" and torch.cuda.is_available())
    if args.joint_curriculum:
        _validate_curriculum_space_assumptions(cache, theta_columns)
        curriculum_state = _prepare_joint_curriculum_state(
            theta=arr_train.theta,
            theta_columns=theta_columns,
            n_age_bins=args.n_bins,
            n_mass_bins=args.n_mass_bins,
        )
        n_train_samples_per_epoch = (
            int(args.curriculum_epoch_size)
            if args.curriculum_epoch_size > 0
            else int(len(train_ds))
        )
        print(
            "Joint curriculum enabled: "
            f"age_bins={args.n_bins}, mass_bins={args.n_mass_bins}, "
            f"active_bins={curriculum_state['n_active']}, "
            f"epoch_samples={n_train_samples_per_epoch}, "
            f"importance_weighting={args.importance_weighting}"
        )
        train_loader = None
    else:
        curriculum_state = None
        n_train_samples_per_epoch = int(len(train_ds))
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    eval_rng = np.random.default_rng(args.seed + 2026)
    young_val_loader = None
    young_val_count = 0
    random_val_loader = None
    train_random_loader = None
    val_curriculum_state = None
    val_curriculum_ds = None
    n_val_curr_samples_per_epoch = 0

    if args.young_eval_max_stars > 0:
        if "logAge" not in cache.columns:
            print("WARNING: logAge not in cache columns; disabling young-star validation loss.")
        else:
            logage_val = _cache_column_physical_values(
                cache, row_indices=val_rows, column_name="logAge",
            )
            young_rows = val_rows[logage_val < float(args.young_logage_threshold)]
            young_val_count = int(young_rows.size)
            print(
                "Young validation pool: "
                f"logAge<{args.young_logage_threshold}, rows={young_val_count:,}"
            )
            if young_rows.size == 0:
                print(
                    "WARNING: no validation stars below young threshold "
                    f"logAge<{args.young_logage_threshold}; disabling young-star validation loss."
                )
            else:
                n_young = min(int(args.young_eval_max_stars), int(young_rows.size))
                if n_young < int(young_rows.size):
                    young_rows = eval_rng.choice(young_rows, size=n_young, replace=False).astype(np.int64)
                young_rows = np.sort(young_rows.astype(np.int64))
                young_val_loader = _build_eval_loader_from_rows(
                    cache=cache,
                    row_indices=young_rows,
                    input_columns=input_columns,
                    theta_columns=theta_columns,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=pin_memory,
                )
                print(
                    "Young validation eval enabled: "
                    f"logAge<{args.young_logage_threshold}, rows={len(young_rows):,}"
                )
    else:
        print("Young validation eval disabled (--young-eval-max-stars <= 0).")

    if args.random_eval_max_stars > 0:
        n_rand = min(int(args.random_eval_max_stars), int(len(val_rows)))
        if n_rand <= 0:
            print("WARNING: no validation rows available for random unweighted eval.")
        else:
            rand_rows = eval_rng.choice(val_rows, size=n_rand, replace=False).astype(np.int64)
            rand_rows = np.sort(rand_rows)
            random_val_loader = _build_eval_loader_from_rows(
                cache=cache,
                row_indices=rand_rows,
                input_columns=input_columns,
                theta_columns=theta_columns,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
            print(f"Random unweighted validation eval enabled: rows={len(rand_rows):,}")

    if args.train_random_eval_max_stars > 0:
        n_train_rand = min(int(args.train_random_eval_max_stars), int(len(train_rows)))
        if n_train_rand <= 0:
            print("WARNING: no training rows available for random unweighted train eval.")
        else:
            train_rand_rows = eval_rng.choice(train_rows, size=n_train_rand, replace=False).astype(np.int64)
            train_rand_rows = np.sort(train_rand_rows)
            train_random_loader = _build_eval_loader_from_rows(
                cache=cache,
                row_indices=train_rand_rows,
                input_columns=input_columns,
                theta_columns=theta_columns,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
            print(f"Random unweighted train eval enabled: rows={len(train_rand_rows):,}")

    if args.val_curriculum_loss:
        if not args.joint_curriculum:
            print("WARNING: --val-curriculum-loss requires --joint-curriculum; disabling.")
        else:
            val_curriculum_state = _prepare_joint_curriculum_state(
                theta=arr_val.theta,
                theta_columns=theta_columns,
                n_age_bins=args.n_bins,
                n_mass_bins=args.n_mass_bins,
            )
            val_curriculum_ds = SBIDataset(arr_val)
            n_val_curr_samples_per_epoch = (
                int(args.val_curriculum_epoch_size)
                if args.val_curriculum_epoch_size > 0
                else int(len(val_curriculum_ds))
            )
            print(
                "Validation curriculum eval enabled: "
                f"epoch_samples={n_val_curr_samples_per_epoch:,}, "
                f"active_bins={val_curriculum_state['n_active']}"
            )

    model = _build_model(args, input_columns=input_columns, theta_dim=len(theta_columns))
    model.to(device)
    # Zuko/nflows flows cause many graph breaks under torch.compile (lazy
    # Distribution objects, functools.partial, generator expressions) making
    # compilation pointless — results are correct but every graph fragment
    # falls back to eager.  Only compile for flow-matching.
    use_compile = bool(args.compile and args.method == "flow_matching")
    if args.compile and not use_compile:
        print("torch.compile requested, but disabled for NF backend (graph breaks in Zuko/nflows).")
    if use_compile:
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")
    print(
        f"Dataset rows: train={len(train_ds):,}, val={len(val_ds):,}; "
        f"input_nodes={len(input_columns)}, theta_dim={len(theta_columns)}"
    )
    if generated_test_idx_path is not None:
        print(f"Generated test split saved at: {generated_test_idx_path}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    # NF log_prob/log-det is numerically fragile under autocast; keep AMP for FM only.
    use_amp_loss = bool(args.amp and args.method == "flow_matching" and device != "cpu")
    if args.amp and args.method in ("normalizing_flow", "realnvp"):
        print("AMP requested, but disabled for NF loss path for numerical stability.")

    try:
        scaler = GradScaler("cuda", enabled=use_amp_loss)
    except TypeError:
        scaler = GradScaler(enabled=use_amp_loss)

    wandb_run = None
    if args.wandb:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                **vars(args),
                "num_parameters": n_params,
                "input_columns": input_columns,
                "theta_columns": theta_columns,
                "train_rows": int(len(train_ds)),
                "val_rows": int(len(val_ds)),
                "young_val_count": int(young_val_count),
            },
        )

    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    ckpt_path = os.path.join(args.output_dir, f"best_model_{args.run_name}.pt")
    hist = []

    t0 = time.time()
    if args.joint_curriculum:
        print("Using joint bin-first sampler (sample bin -> sample row uniformly within bin).")
    for epoch in range(args.epochs):
        curriculum_log = {}
        if args.joint_curriculum:
            tau = _compute_tau(epoch, args.epochs, args.tau_max, args.tau_warmup)
            q_bin, w_i, lam, clip_frac = _joint_curriculum_distributions(
                curriculum_state,
                tau=tau,
                importance_weighting=args.importance_weighting,
                importance_weight_beta=args.importance_weight_beta,
                importance_weight_min=args.importance_weight_min,
                importance_weight_max=args.importance_weight_max,
            )
            train_ds.sample_weight = torch.from_numpy(w_i.astype(np.float32))
            sampler = _build_epoch_sampler(
                state=curriculum_state,
                q_bin=q_bin,
                num_samples=n_train_samples_per_epoch,
                seed=int(args.seed + epoch),
            )
            epoch_train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                sampler=sampler,
                shuffle=False,
                drop_last=True,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
            curriculum_log = {
                "tau": float(tau),
                "mixture_lambda": float(lam),
                "importance_weight_beta": float(args.importance_weight_beta),
                "train_importance_weight_min": float(w_i.min()),
                "train_importance_weight_mean": float(w_i.mean()),
                "train_importance_weight_max": float(w_i.max()),
                "train_weight_clip_frac": float(clip_frac),
            }
        else:
            epoch_train_loader = train_loader

        train_result = _epoch_loss(
            model,
            epoch_train_loader,
            device,
            train=True,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp_loss,
            grad_clip_norm=args.grad_clip_norm,
            nll_cap=args.nll_cap,
        )
        # nll_mean = unweighted sample mean of raw NLL over the training batch.
        # Under curriculum sampling this is E_q[NLL], NOT E_p[NLL], so it is
        # NOT directly comparable to val_loss (which is E_p[NLL]).  Use it to
        # monitor training progress, but not for model selection.
        train_nll_q = train_result["nll_mean"]
        train_loss_optim = train_result["weighted"]  # the actual backprop objective

        train_nll_p_random = None
        if train_random_loader is not None:
            train_rand_result = _epoch_loss(
                model,
                train_random_loader,
                device,
                train=False,
                use_amp=use_amp_loss,
            )
            train_nll_p_random = train_rand_result["nll_mean"]

        val_result = _epoch_loss(
            model,
            val_loader,
            device,
            train=False,
            use_amp=use_amp_loss,
        )
        val_loss = val_result["nll_mean"]

        val_loss_curriculum = None
        if val_curriculum_state is not None and val_curriculum_ds is not None:
            q_bin_val, w_val, _, _ = _joint_curriculum_distributions(
                val_curriculum_state,
                tau=tau,
                importance_weighting=args.importance_weighting,
                importance_weight_beta=args.importance_weight_beta,
                importance_weight_min=args.importance_weight_min,
                importance_weight_max=args.importance_weight_max,
            )
            val_curriculum_ds.sample_weight = torch.from_numpy(w_val.astype(np.float32))
            val_curr_sampler = _build_epoch_sampler(
                state=val_curriculum_state,
                q_bin=q_bin_val,
                num_samples=n_val_curr_samples_per_epoch,
                seed=int(args.seed + 1_000_000 + epoch),
            )
            val_curr_loader = DataLoader(
                val_curriculum_ds,
                batch_size=args.batch_size,
                sampler=val_curr_sampler,
                shuffle=False,
                drop_last=False,
                num_workers=args.num_workers,
                pin_memory=pin_memory,
            )
            val_curr_result = _epoch_loss(
                model,
                val_curr_loader,
                device,
                train=False,
                use_amp=use_amp_loss,
            )
            val_loss_curriculum = val_curr_result["weighted"]

        val_loss_young = None
        if young_val_loader is not None:
            val_young_result = _epoch_loss(
                model,
                young_val_loader,
                device,
                train=False,
                use_amp=use_amp_loss,
            )
            val_loss_young = val_young_result["nll_mean"]

        val_loss_random_unweighted = None
        if random_val_loader is not None:
            val_rand_result = _epoch_loss(
                model,
                random_val_loader,
                device,
                train=False,
                use_amp=use_amp_loss,
            )
            val_loss_random_unweighted = val_rand_result["nll_mean"]
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        hist.append(
            {
                "epoch": epoch + 1,
                "train_nll_q": float(train_nll_q),
                "train_nll_p_random": None
                if train_nll_p_random is None
                else float(train_nll_p_random),
                "train_loss_optim": float(train_loss_optim),
                "val_loss": float(val_loss),
                "val_loss_curriculum": None
                if val_loss_curriculum is None
                else float(val_loss_curriculum),
                "val_loss_young": None if val_loss_young is None else float(val_loss_young),
                "val_loss_random_unweighted": None
                if val_loss_random_unweighted is None
                else float(val_loss_random_unweighted),
                "lr": float(lr),
            }
        )
        train_rand_msg = (
            "" if train_nll_p_random is None else f" train_nll_p_random={train_nll_p_random:.6f}"
        )
        print(
            f"Epoch {epoch + 1:04d}/{args.epochs} "
            f"train_nll(q)={train_nll_q:.6f}{train_rand_msg} val_loss={val_loss:.6f} "
            f"train_optim={train_loss_optim:.6f} lr={lr:.2e}"
        )
        if args.joint_curriculum:
            print(
                f"  curriculum: tau={curriculum_log['tau']:.3f}, "
                f"lambda={curriculum_log['mixture_lambda']:.3f}, "
                f"beta={curriculum_log['importance_weight_beta']:.2f}, "
                f"w[min/mean/max]={curriculum_log['train_importance_weight_min']:.3f}/"
                f"{curriculum_log['train_importance_weight_mean']:.3f}/"
                f"{curriculum_log['train_importance_weight_max']:.3f}, "
                f"clip_frac={curriculum_log['train_weight_clip_frac']:.4f}"
            )
        extra_eval_parts = []
        if val_loss_curriculum is not None:
            extra_eval_parts.append(f"val_curriculum={val_loss_curriculum:.6f}")
        if val_loss_young is not None:
            extra_eval_parts.append(f"val_young={val_loss_young:.6f}")
        if val_loss_random_unweighted is not None:
            extra_eval_parts.append(f"val_random_unweighted={val_loss_random_unweighted:.6f}")
        if extra_eval_parts:
            print("  eval: " + ", ".join(extra_eval_parts))
        if wandb_run is not None:
            payload = {
                "epoch": epoch + 1,
                "train_nll_q": train_nll_q,
                "train_nll_p_random": float("nan")
                if train_nll_p_random is None
                else float(train_nll_p_random),
                "train_loss_optim": train_loss_optim,
                "val_loss": val_loss,
                "lr": lr,
                "val_loss_young": float("nan")
                if val_loss_young is None
                else float(val_loss_young),
            }
            if val_loss_curriculum is not None:
                payload["val_loss_curriculum"] = val_loss_curriculum
            if val_loss_random_unweighted is not None:
                payload["val_loss_random_unweighted"] = val_loss_random_unweighted
            payload.update(curriculum_log)
            wandb_run.log(payload)

        if val_loss < best_val:
            best_val = float(val_loss)
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved new best checkpoint to {ckpt_path}")
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}; best epoch={best_epoch}, best val={best_val:.6f}")
            break

    elapsed = (time.time() - t0) / 60.0
    print(f"Training finished in {elapsed:.1f} min. Best val={best_val:.6f} at epoch {best_epoch}.")

    config_out = {
        **vars(args),
        "input_columns": input_columns,
        "theta_columns": theta_columns,
        "resolved_exclude_indices": args.exclude_indices if args.exclude_indices is not None else generated_test_idx_path,
        "generated_test_indices_path": generated_test_idx_path,
        "generated_test_cluster_ids_path": generated_test_cluster_ids_path,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "num_parameters": n_params,
        "train_rows": int(len(train_ds)),
        "val_rows": int(len(val_ds)),
        "checkpoint_path": ckpt_path,
    }
    config_path = os.path.join(args.output_dir, f"posterior_config_{args.run_name}.json")
    with open(config_path, "w") as f:
        json.dump(config_out, f, indent=2)
    hist_path = os.path.join(args.output_dir, f"posterior_history_{args.run_name}.json")
    with open(hist_path, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"Saved config: {config_path}")
    print(f"Saved history: {hist_path}")

    # Save cache normalization metadata for downstream denormalization/inference.
    meta_path = os.path.join(args.output_dir, f"posterior_norm_meta_{args.run_name}.npz")
    np.savez(
        meta_path,
        columns=np.asarray(cache.columns, dtype=object),
        means=cache.means if cache.means is not None else np.zeros(len(cache.columns), dtype=np.float32),
        stds=cache.stds if cache.stds is not None else np.ones(len(cache.columns), dtype=np.float32),
        value_transform_names=cache.value_transform_names
        if cache.value_transform_names is not None
        else np.asarray(["identity"] * len(cache.columns), dtype=object),
        value_transform_params=cache.value_transform_params
        if cache.value_transform_params is not None
        else np.zeros(len(cache.columns), dtype=np.float32),
        log_err_mean=np.array(
            cache.log_err_mean if cache.log_err_mean is not None else 0.0,
            dtype=np.float32,
        ),
        log_err_std=np.array(
            cache.log_err_std if cache.log_err_std is not None else 1.0,
            dtype=np.float32,
        ),
        input_columns=np.asarray(input_columns, dtype=object),
        theta_columns=np.asarray(theta_columns, dtype=object),
    )
    print(f"Saved normalization metadata: {meta_path}")

    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = best_val
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.finish()


if __name__ == "__main__":
    main()
