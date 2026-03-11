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
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Sampler

# Support both:
#   python -m sbi_variants.train_sbi_nre
#   python train_sbi_nre.py
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from data import (
    DEFAULT_INPUT_COLS,
    DEFAULT_THETA_COLS,
    SBIDataset,
    build_row_split,
    build_sbi_arrays,
    load_cache_arrays,
    load_indices,
    parse_column_csv,
)
from encoder import ObservationEncoder
from ratio_models import ConditionalRatioEstimator
from train_mock_galaxy import (
    DEFAULT_CLUSTER_ID_COL,
    build_arrays as build_cache_arrays,
    load_data as load_raw_data,
    save_arrays as save_cache_arrays,
)
from value_transforms import apply_inverse_value_transforms_numpy

try:
    from torch.amp import GradScaler, autocast

    def _autocast_context(enabled: bool, device: str):
        return autocast("cuda", enabled=(enabled and device.startswith("cuda")))

except ImportError:
    from torch.cuda.amp import GradScaler, autocast

    def _autocast_context(enabled: bool, device: str):
        return autocast(enabled=(enabled and device.startswith("cuda")))


_BIN_SAMPLER_CHUNK_SIZE = 1_000_000


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


def _compute_tau(epoch: int, total_epochs: int, tau_max: float, tau_warmup: int) -> float:
    if tau_max <= 0.0:
        return 0.0
    if epoch < tau_warmup:
        return 0.0
    denom = max(total_epochs - tau_warmup, 1)
    progress = min(max((epoch - tau_warmup) / denom, 0.0), 1.0)
    return float(tau_max * progress)


def _prepare_joint_curriculum_state(
    *,
    theta: np.ndarray,
    theta_columns: list[str],
    n_age_bins: int,
    n_mass_bins: int,
    bin_strategy: str = "quantile",
) -> dict[str, np.ndarray | float | int]:
    try:
        age_idx = theta_columns.index("logAge")
        mass_idx = theta_columns.index("m_init")
    except ValueError as e:
        raise ValueError(
            "Joint curriculum requires theta_columns to include both 'logAge' and 'm_init'."
        ) from e

    age = theta[:, age_idx].astype(np.float64, copy=False)
    mass = theta[:, mass_idx].astype(np.float64, copy=False)

    if np.any(~np.isfinite(age)):
        n_bad = int(np.sum(~np.isfinite(age)))
        raise ValueError(f"Joint curriculum input has {n_bad} non-finite logAge values.")
    if np.any(~np.isfinite(mass)):
        n_bad = int(np.sum(~np.isfinite(mass)))
        raise ValueError(f"Joint curriculum input has {n_bad} non-finite m_init values.")

    def _build_edges(vals: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
        if strategy == "equal_width":
            vmin, vmax = float(np.min(vals)), float(np.max(vals))
            if not np.isfinite(vmin) or not np.isfinite(vmax):
                raise ValueError("Non-finite range in equal-width binning inputs.")
            if vmax <= vmin:
                vmax = vmin + 1e-6
            edges = np.linspace(vmin, vmax, n_bins + 1, dtype=np.float64)
        elif strategy == "quantile":
            edges = np.quantile(
                vals,
                np.linspace(0.0, 1.0, n_bins + 1),
            ).astype(np.float64, copy=False)
        else:
            raise ValueError(f"Unsupported bin strategy: {strategy}")

        # Enforce strict monotonicity to avoid searchsorted ambiguity.
        vmin, vmax = float(np.min(vals)), float(np.max(vals))
        eps = max(
            np.finfo(np.float64).eps * max(abs(vmin), abs(vmax), 1.0),
            1e-12,
        )
        for i in range(1, edges.size):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + eps
        if edges[-1] <= vmax:
            edges[-1] = vmax + eps
        return edges

    age_edges = _build_edges(age, n_age_bins, bin_strategy)
    mass_edges = _build_edges(mass, n_mass_bins, bin_strategy)

    age_bin = np.searchsorted(age_edges, age, side="right") - 1
    mass_bin = np.searchsorted(mass_edges, mass, side="right") - 1
    age_bin = np.clip(age_bin, 0, n_age_bins - 1)
    mass_bin = np.clip(mass_bin, 0, n_mass_bins - 1)

    joint = age_bin * n_mass_bins + mass_bin
    n_total_bins = n_age_bins * n_mass_bins
    counts = np.bincount(joint, minlength=n_total_bins).astype(np.float64)
    active = counts > 0
    n_active = int(active.sum())
    if n_active <= 0:
        raise ValueError("Joint curriculum found no active bins.")

    p_bin = np.zeros(n_total_bins, dtype=np.float64)
    p_bin[active] = counts[active] / counts[active].sum()

    active_bins = np.flatnonzero(active).astype(np.int64)
    order = np.argsort(joint, kind="mergesort")
    sorted_joint = joint[order]
    bin_offsets = np.searchsorted(
        sorted_joint, np.arange(n_total_bins + 1), side="left",
    ).astype(np.int64)

    return {
        "joint": joint.astype(np.int64),
        "p_bin": p_bin,
        "active": active,
        "active_bins": active_bins,
        "n_active": n_active,
        "n_total_bins": int(n_total_bins),
        "sorted_indices_by_bin": order.astype(np.int64),
        "bin_offsets": bin_offsets,
    }


def _joint_curriculum_distributions(
    state: dict[str, np.ndarray | float | int],
    tau: float,
    *,
    importance_weighting: bool,
    importance_weight_beta: float,
    importance_weight_min: float,
    importance_weight_max: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Compute curriculum sampling distribution and row-level weights."""
    joint = state["joint"]
    p_bin = state["p_bin"]
    active = state["active"]
    n_active = int(state["n_active"])

    lam = float(np.clip(1.0 - tau, 0.0, 1.0))
    q_bin = np.zeros_like(p_bin, dtype=np.float64)
    q_bin[active] = (1.0 - lam) * p_bin[active] + lam * (1.0 / float(n_active))

    if importance_weighting:
        raw = (p_bin[joint] / q_bin[joint]).astype(np.float64)
        if importance_weight_beta != 1.0:
            raw = np.power(raw, float(importance_weight_beta))
        w_i = (raw / max(float(raw.mean()), 1e-8)).astype(np.float32)
        n_total = len(w_i)
        n_clipped = int(
            np.sum(w_i < float(importance_weight_min))
            + np.sum(w_i > float(importance_weight_max))
        )
        clip_frac = n_clipped / max(n_total, 1)
        w_i = np.clip(
            w_i, float(importance_weight_min), float(importance_weight_max),
        ).astype(np.float32)
        w_i /= max(float(w_i.mean()), 1e-8)
    else:
        w_i = np.ones_like(joint, dtype=np.float32)
        clip_frac = 0.0

    return q_bin.astype(np.float64), w_i, lam, clip_frac


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train NRE/AMNRE ratio estimator with transformer observation encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=str, default=None,
                   help="Optional JSON config file. CLI flags override config values.")

    # Data
    p.add_argument("--cache-path", type=str, default=None,
                   help="Path to build_arrays_cache.npz.")
    p.add_argument("--data-path", type=str, default=None,
                   help="Optional Parquet/CSV path used when cache must be built.")
    p.add_argument("--rebuild-cache", action=argparse.BooleanOptionalAction, default=False,
                   help="Force rebuilding cache from --data-path.")
    p.add_argument("--cluster-id-col", type=str, default=DEFAULT_CLUSTER_ID_COL,
                   help="Cluster ID column used only when building cache.")
    p.add_argument("--output-dir", type=str, default="./output_nre")
    p.add_argument("--run-name", type=str, default="sbi_nre")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Optional resume checkpoint containing model/optimizer/scheduler/scaler state.")

    # Columns / split
    p.add_argument("--input-columns", type=str, default=",".join(DEFAULT_INPUT_COLS))
    p.add_argument("--theta-columns", type=str, default=",".join(DEFAULT_THETA_COLS))
    p.add_argument("--exclude-indices", type=str, default=None,
                   help="Optional .npy indices excluded from train/val (e.g. test_indices.npy).")
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--max-stars", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-colors", action=argparse.BooleanOptionalAction, default=False,
                   help="Append color features to inputs.")

    # Joint curriculum over (logAge, m_init) bins
    p.add_argument(
        "--joint-curriculum",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable joint (logAge,m_init) bin-first sampling. "
            "With tau=0 this strongly upsamples rare bins."
        ),
    )
    p.add_argument("--n-bins", type=int, default=25,
                   help="Number of logAge bins for joint curriculum.")
    p.add_argument("--n-mass-bins", type=int, default=12,
                   help="Number of m_init bins for joint curriculum.")
    p.add_argument(
        "--curriculum-bin-strategy",
        type=str,
        default="quantile",
        choices=["quantile", "equal_width"],
        help=(
            "Binning strategy in (logAge,m_init) curriculum space. "
            "'quantile' balances bin populations."
        ),
    )
    p.add_argument("--tau-max", type=float, default=0.0,
                   help="Max tau in q=(1-lambda)p+lambda/K, lambda=1-tau.")
    p.add_argument("--tau-warmup", type=int, default=0,
                   help="Epochs to keep tau=0 before ramping.")
    p.add_argument("--curriculum-epoch-size", type=int, default=0,
                   help="Samples drawn per epoch when curriculum is enabled (0 => len(train)).")
    p.add_argument(
        "--curriculum-importance-weighting",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply row-level p/q importance correction for curriculum sampling. "
            "Disabled by default to preserve strong rare-bin emphasis."
        ),
    )
    p.add_argument("--curriculum-importance-beta", type=float, default=1.0,
                   help="Tempering exponent for curriculum row weights.")
    p.add_argument("--curriculum-importance-min", type=float, default=0.1)
    p.add_argument("--curriculum-importance-max", type=float, default=10.0)

    # Optional young-star eval slice
    p.add_argument("--young-logage-threshold", type=float, default=7.8)
    p.add_argument("--young-eval-max-stars", type=int, default=0,
                   help="Max stars for optional young-val BCE each epoch (0 disables).")

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

    # Ratio head
    p.add_argument("--ratio-hidden-dim", type=int, default=256)
    p.add_argument("--ratio-dropout", type=float, default=0.0)
    p.add_argument(
        "--ratio-mask-mode",
        type=str,
        default="none",
        choices=["none", "amnre_k_uniform", "amnre_bernoulli"],
        help=(
            "Mask strategy over theta dimensions. "
            "'none' = standard NRE; amnre_* = AMNRE-style random parameter subsets."
        ),
    )
    p.add_argument("--mask-bernoulli-p", type=float, default=0.5,
                   help="Bernoulli p for amnre_bernoulli mask sampling.")

    # BNRE (balanced classifier regularization)
    p.add_argument("--use-balanced-loss", action=argparse.BooleanOptionalAction, default=True,
                   help="Add BNRE balancing penalty: lambda * (E_joint[d] + E_prod[d] - 1)^2.")
    p.add_argument("--bnre-lambda", type=float, default=100.0,
                   help="Balancing penalty strength (recommended around 100).")

    # K&F-style importance reweighting proxy
    p.add_argument("--importance-mode", type=str, default="none",
                   choices=["none", "kf_logit_grad"],
                   help="Per-pair importance weighting mode.")
    p.add_argument("--importance-beta", type=float, default=1.0,
                   help="Tempering exponent for importance weights.")
    p.add_argument("--importance-min", type=float, default=0.1)
    p.add_argument("--importance-max", type=float, default=10.0)
    p.add_argument("--importance-warmup-epochs", type=int, default=0,
                   help="Epochs with importance weighting disabled.")

    # Optimization
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=60)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--amp", action="store_true", default=False)
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
    parser = _build_parser()

    base = parser.parse_args()
    if base.config is not None:
        cfg = _load_config_defaults(base.config, parser)
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
        ex_norm = args.exclude_indices.strip().lower()
        if ex_norm in ("", "none", "null"):
            args.exclude_indices = None

    if not (0.0 < args.val_split < 1.0):
        raise ValueError(f"--val-split must be in (0,1), got {args.val_split}")
    if args.max_stars is not None and args.max_stars <= 1:
        raise ValueError(f"--max-stars must be >1 when provided, got {args.max_stars}")
    if args.batch_size <= 1:
        raise ValueError(f"--batch-size must be >1, got {args.batch_size}")
    if args.epochs <= 0:
        raise ValueError(f"--epochs must be >0, got {args.epochs}")
    if args.lr <= 0 or args.lr_min <= 0:
        raise ValueError("--lr and --lr-min must be >0.")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay must be >=0.")
    if args.grad_clip_norm <= 0:
        raise ValueError("--grad-clip-norm must be >0.")
    if args.patience < 0:
        raise ValueError("--patience must be >=0.")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >=0.")
    if args.n_bins <= 0:
        raise ValueError("--n-bins must be >0.")
    if args.n_mass_bins <= 0:
        raise ValueError("--n-mass-bins must be >0.")
    if args.curriculum_epoch_size < 0:
        raise ValueError("--curriculum-epoch-size must be >=0.")
    if not (0.0 <= args.tau_max <= 1.0):
        raise ValueError("--tau-max must be in [0,1].")
    if args.tau_warmup < 0:
        raise ValueError("--tau-warmup must be >=0.")
    if args.mask_bernoulli_p <= 0 or args.mask_bernoulli_p >= 1:
        raise ValueError("--mask-bernoulli-p must be in (0,1).")
    if args.bnre_lambda < 0:
        raise ValueError("--bnre-lambda must be >=0.")
    if args.importance_beta < 0:
        raise ValueError("--importance-beta must be >=0.")
    if args.importance_min <= 0:
        raise ValueError("--importance-min must be >0.")
    if args.importance_max < args.importance_min:
        raise ValueError("--importance-max must be >= --importance-min.")
    if args.importance_warmup_epochs < 0:
        raise ValueError("--importance-warmup-epochs must be >=0.")
    if args.curriculum_importance_beta < 0:
        raise ValueError("--curriculum-importance-beta must be >=0.")
    if args.curriculum_importance_min <= 0:
        raise ValueError("--curriculum-importance-min must be >0.")
    if args.curriculum_importance_max < args.curriculum_importance_min:
        raise ValueError("--curriculum-importance-max must be >= --curriculum-importance-min.")
    if args.young_eval_max_stars < 0:
        raise ValueError("--young-eval-max-stars must be >=0.")

    return args


def _ensure_cache(args: argparse.Namespace) -> str:
    cache_path = args.cache_path
    if cache_path is None:
        cache_path = os.path.join(args.output_dir, "build_arrays_cache.npz")
    cache_path = os.path.abspath(cache_path)

    if os.path.exists(cache_path) and not args.rebuild_cache:
        print(f"Using cache: {cache_path}")
        return cache_path

    if args.data_path is None:
        raise FileNotFoundError(
            f"Cache not found at {cache_path} and no --data-path was provided. "
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


def _cache_column_physical_values(
    cache,
    row_indices: np.ndarray,
    column_name: str,
) -> np.ndarray:
    if column_name not in cache.columns:
        raise ValueError(f"Column '{column_name}' not found in cache columns.")
    if cache.means is None or cache.stds is None:
        raise ValueError("Cache has no means/stds metadata.")

    col_idx = cache.columns.index(column_name)
    std = float(cache.stds[col_idx])
    mean = float(cache.means[col_idx])
    if (not np.isfinite(std)) or std <= 0.0 or (not np.isfinite(mean)):
        raise ValueError(f"Invalid cache mean/std for column '{column_name}'.")

    rows = np.asarray(row_indices, dtype=np.int64)
    vals = cache.values_norm[rows, col_idx].astype(np.float64, copy=False)
    vals = vals * std + mean

    if cache.value_transform_names is not None and cache.value_transform_params is not None:
        vals = apply_inverse_value_transforms_numpy(
            vals.reshape(-1, 1),
            transform_names=np.asarray([cache.value_transform_names[col_idx]], dtype=object),
            transform_params=np.asarray([cache.value_transform_params[col_idx]], dtype=np.float32),
        ).reshape(-1)
    return vals.astype(np.float64, copy=False)


def _sample_theta_masks(
    *,
    batch_size: int,
    theta_dim: int,
    mode: str,
    bernoulli_p: float,
    device: torch.device,
) -> torch.Tensor:
    if mode == "none":
        return torch.ones(batch_size, theta_dim, device=device, dtype=torch.float32)

    if mode == "amnre_bernoulli":
        mask = (torch.rand(batch_size, theta_dim, device=device) < bernoulli_p).to(torch.float32)
        all_zero = mask.sum(dim=1) < 0.5
        if all_zero.any():
            idx = torch.nonzero(all_zero, as_tuple=False).reshape(-1)
            forced = torch.randint(0, theta_dim, (idx.numel(),), device=device)
            mask[idx, forced] = 1.0
        return mask

    if mode == "amnre_k_uniform":
        # Sample subset size k uniformly in {1, ..., D}, then random k dimensions.
        k = torch.randint(1, theta_dim + 1, (batch_size,), device=device)
        order = torch.rand(batch_size, theta_dim, device=device).argsort(dim=1)
        rank = torch.arange(theta_dim, device=device).unsqueeze(0).expand(batch_size, -1)
        keep = rank < k.unsqueeze(1)
        mask = torch.zeros(batch_size, theta_dim, device=device, dtype=torch.float32)
        mask.scatter_(1, order, keep.to(torch.float32))
        return mask

    raise ValueError(f"Unsupported mask mode: {mode}")


def _importance_weights_from_logits(
    *,
    logits_pos: torch.Tensor,  # (B,)
    logits_neg: torch.Tensor,  # (B,)
    mode: str,
    beta: float,
    w_min: float,
    w_max: float,
) -> tuple[torch.Tensor, float]:
    if mode == "none":
        b = logits_pos.shape[0]
        return torch.ones(2 * b, device=logits_pos.device, dtype=logits_pos.dtype), 0.0

    if mode != "kf_logit_grad":
        raise ValueError(f"Unsupported importance mode: {mode}")

    # BCE with logits has dL/dz = sigmoid(z)-y, so abs gradient wrt logit is:
    #   positive pairs (y=1): 1 - sigmoid(z)
    #   negative pairs (y=0): sigmoid(z)
    p_pos = torch.sigmoid(logits_pos.detach())
    p_neg = torch.sigmoid(logits_neg.detach())
    proxy = torch.cat([1.0 - p_pos, p_neg], dim=0).clamp_min(1e-8)

    if beta != 1.0:
        proxy = proxy.pow(beta)

    w = proxy / proxy.mean().clamp_min(1e-8)
    n_clipped = ((w < w_min) | (w > w_max)).sum().item()
    clip_frac = float(n_clipped) / max(int(w.numel()), 1)

    w = w.clamp(min=w_min, max=w_max)
    w = w / w.mean().clamp_min(1e-8)
    return w, clip_frac


def _build_model(args: argparse.Namespace, input_columns: list[str], theta_dim: int) -> ConditionalRatioEstimator:
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
    return ConditionalRatioEstimator(
        encoder=encoder,
        theta_dim=theta_dim,
        hidden_dim=args.ratio_hidden_dim,
        dropout=args.ratio_dropout,
        use_mask_condition=(args.ratio_mask_mode != "none"),
    )


def _move_batch(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _save_resume_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
    epoch: int,
    best_val: float,
    best_epoch: int,
    no_improve: int,
    history: list[dict],
) -> None:
    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "best_val": float(best_val),
        "best_epoch": int(best_epoch),
        "no_improve": int(no_improve),
        "history": history,
    }
    tmp_path = f"{path}.tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _maybe_resume_training_state(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    optimizer: AdamW,
    scheduler: CosineAnnealingLR,
    scaler: GradScaler,
) -> tuple[int, float, int, int, list[dict]]:
    start_epoch = 0
    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    history: list[dict] = []

    if args.resume_from is None:
        return start_epoch, best_val, best_epoch, no_improve, history

    resume_path = str(args.resume_from)
    if not os.path.exists(resume_path):
        raise FileNotFoundError(f"--resume-from checkpoint not found: {resume_path}")

    print(f"Resuming from checkpoint: {resume_path}")
    ckpt = torch.load(resume_path, map_location="cpu")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format at {resume_path}")

    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            except Exception as e:
                print(f"WARNING: failed to load scaler state: {e}")
        start_epoch = int(ckpt.get("epoch", 0))
        best_val = float(ckpt.get("best_val", float("inf")))
        best_epoch = int(ckpt.get("best_epoch", -1))
        no_improve = int(ckpt.get("no_improve", 0))
        history_raw = ckpt.get("history", [])
        history = list(history_raw) if isinstance(history_raw, list) else []
        print(
            f"  Resume state: start_epoch={start_epoch}, "
            f"best_val={best_val:.6f}, best_epoch={best_epoch}, history_len={len(history)}"
        )
        return start_epoch, best_val, best_epoch, no_improve, history

    model.load_state_dict(ckpt, strict=True)
    print("  Loaded model weights only; optimizer/scheduler states not present.")
    return start_epoch, best_val, best_epoch, no_improve, history


def _run_epoch(
    *,
    model: ConditionalRatioEstimator,
    loader: DataLoader,
    device: str,
    train: bool,
    optimizer: AdamW | None,
    scaler: GradScaler | None,
    use_amp: bool,
    grad_clip_norm: float,
    ratio_mask_mode: str,
    mask_bernoulli_p: float,
    use_balanced_loss: bool,
    bnre_lambda: float,
    importance_mode: str,
    importance_beta: float,
    importance_min: float,
    importance_max: float,
    apply_importance: bool,
) -> dict[str, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_bce = 0.0
    total_bce_weighted = 0.0
    total_bal_pen = 0.0
    total_balance_abs = 0.0
    total_pairs = 0

    total_w = 0.0
    total_w2 = 0.0
    total_clip_num = 0.0

    for batch in loader:
        batch = _move_batch(batch, device=device)
        b = int(batch["theta"].shape[0])
        if b <= 1:
            continue

        if train:
            assert optimizer is not None
            assert scaler is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with _autocast_context(use_amp, device):
                ctx = model.encode_context(
                    batch["inputs"], batch["errors"], batch["observed"],
                )
                theta = batch["theta"]
                mask = _sample_theta_masks(
                    batch_size=b,
                    theta_dim=int(theta.shape[1]),
                    mode=ratio_mask_mode,
                    bernoulli_p=mask_bernoulli_p,
                    device=theta.device,
                )
                # BNRE/NRE negatives use p(theta)p(x) pairs. Use a random cyclic
                # shift (instead of randperm) to avoid accidental self-pairs.
                shift = int(torch.randint(1, b, (1,), device=theta.device).item())
                perm = (torch.arange(b, device=theta.device) + shift) % b
                logits_pos = model.logits_from_context(theta, ctx, mask=mask)
                logits_neg = model.logits_from_context(theta[perm], ctx, mask=mask)

                bce_pos = F.binary_cross_entropy_with_logits(
                    logits_pos, torch.ones_like(logits_pos), reduction="none",
                )
                bce_neg = F.binary_cross_entropy_with_logits(
                    logits_neg, torch.zeros_like(logits_neg), reduction="none",
                )
                bce_all = torch.cat([bce_pos, bce_neg], dim=0)

                if apply_importance:
                    w_pair, clip_frac = _importance_weights_from_logits(
                        logits_pos=logits_pos,
                        logits_neg=logits_neg,
                        mode=importance_mode,
                        beta=importance_beta,
                        w_min=importance_min,
                        w_max=importance_max,
                    )
                else:
                    w_pair = torch.ones_like(bce_all)
                    clip_frac = 0.0

                # Row-level weights come from the dataset and are used for
                # curriculum sampling correction (or are all ones by default).
                row_w = batch.get("sample_weight")
                if row_w is None:
                    row_w_pair = torch.ones_like(bce_all)
                else:
                    row_w = row_w.to(bce_all.dtype).reshape(-1)
                    row_w_pair = torch.cat([row_w, row_w], dim=0)

                w_all = row_w_pair * w_pair
                w_all = w_all / w_all.mean().clamp_min(1e-8)

                bce_weighted = (bce_all * w_all).sum() / w_all.sum().clamp_min(1e-8)

                p_pos = torch.sigmoid(logits_pos)
                p_neg = torch.sigmoid(logits_neg)
                balance_value = (p_pos.mean() + p_neg.mean() - 1.0)
                balance_pen = balance_value.pow(2)

                if use_balanced_loss:
                    loss = bce_weighted + float(bnre_lambda) * balance_pen
                else:
                    loss = bce_weighted

        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

        n_pairs = int(bce_all.numel())
        total_pairs += n_pairs
        total_loss += float(loss.detach().item()) * n_pairs
        total_bce += float(bce_all.detach().mean().item()) * n_pairs
        total_bce_weighted += float(bce_weighted.detach().item()) * n_pairs
        total_bal_pen += float(balance_pen.detach().item())
        total_balance_abs += float(balance_value.detach().abs().item())

        w_det = w_all.detach().float()
        total_w += float(w_det.sum().item())
        total_w2 += float((w_det * w_det).sum().item())
        total_clip_num += float(clip_frac) * n_pairs

    n = max(total_pairs, 1)
    n_batches = max(len(loader), 1)
    ess = (total_w * total_w) / max(total_w2, 1e-8)
    return {
        "loss": total_loss / n,
        "bce": total_bce / n,
        "bce_weighted": total_bce_weighted / n,
        "balance_penalty": total_bal_pen / n_batches,
        "balance_abs": total_balance_abs / n_batches,
        "pairs": float(total_pairs),
        "clip_frac": total_clip_num / n,
        "ess": ess,
        "ess_ratio": ess / n,
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
    use_colors: bool = False,
    color_norm_stats: tuple[np.ndarray, np.ndarray] | None = None,
) -> DataLoader:
    arr = build_sbi_arrays(
        cache,
        row_indices=row_indices,
        input_columns=input_columns,
        theta_columns=theta_columns,
        use_colors=use_colors,
        color_norm_stats=color_norm_stats,
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

    exclude_idx = load_indices(args.exclude_indices)
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
        use_colors=args.use_colors,
    )
    color_norm_stats = None
    if args.use_colors and arr_train.color_means is not None:
        color_norm_stats = (arr_train.color_means, arr_train.color_stds)
    arr_val = build_sbi_arrays(
        cache,
        row_indices=val_rows,
        input_columns=input_columns,
        theta_columns=theta_columns,
        use_colors=args.use_colors,
        color_norm_stats=color_norm_stats,
    )

    if args.use_colors and arr_train.color_names is not None:
        extended_input_columns = list(input_columns) + arr_train.color_names
    else:
        extended_input_columns = list(input_columns)

    train_ds = SBIDataset(arr_train)
    val_ds = SBIDataset(arr_val)
    pin_memory = (device != "cpu" and torch.cuda.is_available())

    curriculum_state = None
    n_train_samples_per_epoch = int(len(train_ds))
    curriculum_uses_ramp = False
    if args.joint_curriculum:
        curriculum_state = _prepare_joint_curriculum_state(
            theta=arr_train.theta,
            theta_columns=theta_columns,
            n_age_bins=args.n_bins,
            n_mass_bins=args.n_mass_bins,
            bin_strategy=args.curriculum_bin_strategy,
        )
        n_train_samples_per_epoch = (
            int(args.curriculum_epoch_size)
            if args.curriculum_epoch_size > 0
            else int(len(train_ds))
        )
        curriculum_uses_ramp = bool(args.tau_max > 0.0)
        if curriculum_uses_ramp:
            schedule_msg = (
                f"ramp(tau_warmup={args.tau_warmup}, tau_max={args.tau_max:.3f})"
            )
        else:
            schedule_msg = "fixed-uniform(tau=0)"
            if args.tau_warmup > 0:
                print(
                    "WARNING: tau_max=0.0 puts curriculum in fixed-uniform mode; "
                    "--tau-warmup has no effect."
                )
        print(
            "Joint curriculum enabled: "
            f"bin_strategy={args.curriculum_bin_strategy}, "
            f"active_bins={curriculum_state['n_active']}, "
            f"total_bins={curriculum_state['n_total_bins']}, "
            f"epoch_samples={n_train_samples_per_epoch:,}, "
            f"schedule={schedule_msg}, "
            f"row_importance_weighting={args.curriculum_importance_weighting}"
        )

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

    young_val_loader = None
    if args.young_eval_max_stars > 0:
        if "logAge" not in cache.columns:
            print("WARNING: logAge not in cache columns; disabling young-star eval.")
        else:
            logage_val = _cache_column_physical_values(cache, val_rows, "logAge")
            young_rows = val_rows[logage_val < float(args.young_logage_threshold)]
            if len(young_rows) == 0:
                print(
                    f"WARNING: no validation rows with logAge < {args.young_logage_threshold}; "
                    "disabling young-star eval."
                )
            else:
                max_young = int(args.young_eval_max_stars)
                if len(young_rows) > max_young:
                    rng = np.random.default_rng(args.seed)
                    young_rows = np.sort(rng.choice(young_rows, size=max_young, replace=False))
                young_val_loader = _build_eval_loader_from_rows(
                    cache=cache,
                    row_indices=young_rows,
                    input_columns=input_columns,
                    theta_columns=theta_columns,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    pin_memory=pin_memory,
                    use_colors=args.use_colors,
                    color_norm_stats=color_norm_stats,
                )
                print(
                    "Young-star eval enabled: "
                    f"threshold={args.young_logage_threshold}, rows={len(young_rows)}"
                )

    model = _build_model(args, extended_input_columns, theta_dim=len(theta_columns)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        "NRE model: "
        f"theta_dim={len(theta_columns)}, use_mask_condition={args.ratio_mask_mode != 'none'}, "
        f"params={n_params:,}"
    )
    print(f"Rows: train={len(train_ds):,}, val={len(val_ds):,}")

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)

    use_amp = bool(args.amp and device.startswith("cuda"))
    try:
        scaler = GradScaler("cuda", enabled=use_amp)
    except TypeError:
        scaler = GradScaler(enabled=use_amp)

    resume_ckpt_path = os.path.join(args.output_dir, f"resume_ratio_checkpoint_{args.run_name}.pt")
    (
        start_epoch,
        best_val,
        best_epoch,
        no_improve,
        hist,
    ) = _maybe_resume_training_state(
        args=args,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
    )

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
            },
        )

    ckpt_path = os.path.join(args.output_dir, f"best_ratio_model_{args.run_name}.pt")
    t0 = time.time()
    if start_epoch > 0:
        print(f"Continuing training from epoch {start_epoch + 1}/{args.epochs}")

    for epoch in range(start_epoch, args.epochs):
        apply_importance = bool(
            args.importance_mode != "none"
            and epoch >= int(args.importance_warmup_epochs)
        )
        curriculum_log = {}
        if curriculum_state is not None:
            tau = 0.0
            if curriculum_uses_ramp:
                tau = _compute_tau(epoch, args.epochs, args.tau_max, args.tau_warmup)
            q_bin, row_w, lam, row_clip_frac = _joint_curriculum_distributions(
                curriculum_state,
                tau=tau,
                importance_weighting=args.curriculum_importance_weighting,
                importance_weight_beta=args.curriculum_importance_beta,
                importance_weight_min=args.curriculum_importance_min,
                importance_weight_max=args.curriculum_importance_max,
            )
            train_ds.sample_weight = torch.from_numpy(row_w.astype(np.float32))
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
                "row_importance_weight_beta": float(args.curriculum_importance_beta),
                "row_weight_min": float(row_w.min()),
                "row_weight_mean": float(row_w.mean()),
                "row_weight_max": float(row_w.max()),
                "row_weight_clip_frac": float(row_clip_frac),
            }
        else:
            epoch_train_loader = train_loader

        train_stats = _run_epoch(
            model=model,
            loader=epoch_train_loader,
            device=device,
            train=True,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=use_amp,
            grad_clip_norm=args.grad_clip_norm,
            ratio_mask_mode=args.ratio_mask_mode,
            mask_bernoulli_p=args.mask_bernoulli_p,
            use_balanced_loss=args.use_balanced_loss,
            bnre_lambda=args.bnre_lambda,
            importance_mode=args.importance_mode,
            importance_beta=args.importance_beta,
            importance_min=args.importance_min,
            importance_max=args.importance_max,
            apply_importance=apply_importance,
        )
        val_stats = _run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            train=False,
            optimizer=None,
            scaler=None,
            use_amp=use_amp,
            grad_clip_norm=args.grad_clip_norm,
            ratio_mask_mode=args.ratio_mask_mode,
            mask_bernoulli_p=args.mask_bernoulli_p,
            use_balanced_loss=args.use_balanced_loss,
            bnre_lambda=args.bnre_lambda,
            importance_mode=args.importance_mode,
            importance_beta=args.importance_beta,
            importance_min=args.importance_min,
            importance_max=args.importance_max,
            apply_importance=False,
        )
        val_young_bce = None
        if young_val_loader is not None:
            young_stats = _run_epoch(
                model=model,
                loader=young_val_loader,
                device=device,
                train=False,
                optimizer=None,
                scaler=None,
                use_amp=use_amp,
                grad_clip_norm=args.grad_clip_norm,
                ratio_mask_mode=args.ratio_mask_mode,
                mask_bernoulli_p=args.mask_bernoulli_p,
                use_balanced_loss=args.use_balanced_loss,
                bnre_lambda=args.bnre_lambda,
                importance_mode=args.importance_mode,
                importance_beta=args.importance_beta,
                importance_min=args.importance_min,
                importance_max=args.importance_max,
                apply_importance=False,
            )
            val_young_bce = float(young_stats["bce"])

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_stats["loss"]),
            "train_bce": float(train_stats["bce"]),
            "train_bce_weighted": float(train_stats["bce_weighted"]),
            "train_balance_penalty": float(train_stats["balance_penalty"]),
            "train_balance_abs": float(train_stats["balance_abs"]),
            "train_clip_frac": float(train_stats["clip_frac"]),
            "train_ess": float(train_stats["ess"]),
            "train_ess_ratio": float(train_stats["ess_ratio"]),
            "val_loss": float(val_stats["loss"]),
            "val_bce": float(val_stats["bce"]),
            "val_bce_weighted": float(val_stats["bce_weighted"]),
            "val_balance_penalty": float(val_stats["balance_penalty"]),
            "val_balance_abs": float(val_stats["balance_abs"]),
            "val_bce_young": None if val_young_bce is None else float(val_young_bce),
            "lr": float(lr),
        }
        row.update(curriculum_log)
        hist.append(row)

        young_msg = "" if val_young_bce is None else f" val_bce_young={val_young_bce:.6f}"
        print(
            f"Epoch {epoch + 1:04d}/{args.epochs} "
            f"train_bce={train_stats['bce']:.6f} val_bce={val_stats['bce']:.6f} "
            f"train_loss={train_stats['loss']:.6f} val_loss={val_stats['loss']:.6f} "
            f"bal={train_stats['balance_abs']:.5f} "
            f"clip_frac={train_stats['clip_frac']:.4f} ess/N={train_stats['ess_ratio']:.4f} "
            f"lr={lr:.2e}{young_msg}"
        )
        if curriculum_log:
            print(
                f"  curriculum: tau={curriculum_log['tau']:.3f}, "
                f"lambda={curriculum_log['mixture_lambda']:.3f}, "
                f"row_w[min/mean/max]={curriculum_log['row_weight_min']:.3f}/"
                f"{curriculum_log['row_weight_mean']:.3f}/"
                f"{curriculum_log['row_weight_max']:.3f}, "
                f"row_clip_frac={curriculum_log['row_weight_clip_frac']:.4f}"
            )

        if wandb_run is not None:
            payload = dict(row)
            payload["importance_active"] = float(apply_importance)
            wandb_run.log(payload)

        # Model selection: keep the lowest validation BCE.
        val_key = float(val_stats["bce"])
        if val_key < best_val:
            best_val = val_key
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved new best checkpoint to {ckpt_path}")
        else:
            no_improve += 1

        _save_resume_checkpoint(
            resume_ckpt_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch + 1,
            best_val=best_val,
            best_epoch=best_epoch,
            no_improve=no_improve,
            history=hist,
        )

        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}; best epoch={best_epoch}, best val_bce={best_val:.6f}")
            break

    elapsed = (time.time() - t0) / 60.0
    print(f"Training finished in {elapsed:.1f} min. Best val_bce={best_val:.6f} at epoch {best_epoch}.")

    config_out = {
        **vars(args),
        "input_columns": input_columns,
        "input_columns_with_colors": extended_input_columns,
        "theta_columns": theta_columns,
        "color_names": arr_train.color_names if arr_train.color_names else [],
        "best_val_bce": best_val,
        "best_epoch": best_epoch,
        "num_parameters": n_params,
        "train_rows": int(len(train_ds)),
        "val_rows": int(len(val_ds)),
        "checkpoint_path": ckpt_path,
        "resume_checkpoint_path": resume_ckpt_path,
        "resumed_from": args.resume_from,
        "start_epoch": int(start_epoch),
    }
    config_path = os.path.join(args.output_dir, f"ratio_config_{args.run_name}.json")
    with open(config_path, "w") as f:
        json.dump(config_out, f, indent=2)
    hist_path = os.path.join(args.output_dir, f"ratio_history_{args.run_name}.json")
    with open(hist_path, "w") as f:
        json.dump(hist, f, indent=2)
    print(f"Saved config: {config_path}")
    print(f"Saved history: {hist_path}")

    # Save normalization metadata for downstream diagnostics/inference.
    meta_path = os.path.join(args.output_dir, f"ratio_norm_meta_{args.run_name}.npz")
    meta_dict = dict(
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
        input_columns=np.asarray(extended_input_columns, dtype=object),
        input_columns_base=np.asarray(input_columns, dtype=object),
        theta_columns=np.asarray(theta_columns, dtype=object),
        use_colors=np.array(args.use_colors, dtype=bool),
    )
    if args.use_colors and arr_train.color_names is not None:
        meta_dict["color_names"] = np.asarray(arr_train.color_names, dtype=object)
        meta_dict["color_means"] = arr_train.color_means
        meta_dict["color_stds"] = arr_train.color_stds
    np.savez(meta_path, **meta_dict)
    print(f"Saved normalization metadata: {meta_path}")

    if wandb_run is not None:
        wandb_run.summary["best_val_bce"] = best_val
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.finish()


if __name__ == "__main__":
    main()
