#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

# Support both:
#   python -m sbi_variants.train_sbi_posterior
#   python sbi_variants/train_sbi_posterior.py
if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from sbi_variants.data import (
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
    p.add_argument("--importance-weight-min", type=float, default=0.5)
    p.add_argument("--importance-weight-max", type=float, default=2.0)

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

    if isinstance(args.exclude_indices, str):
        norm = args.exclude_indices.strip().lower()
        if norm in ("", "none", "null"):
            args.exclude_indices = None

    missing = []
    if not args.cache_path:
        missing.append("--cache-path")
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
    return args


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
    age_edges = np.linspace(age.min(), age.max() + 1e-6, n_age_bins + 1)
    mass_edges = np.linspace(mass.min(), mass.max() + 1e-6, n_mass_bins + 1)

    age_bin = np.digitize(age, age_edges) - 1
    mass_bin = np.digitize(mass, mass_edges) - 1
    age_bin = np.clip(age_bin, 0, n_age_bins - 1)
    mass_bin = np.clip(mass_bin, 0, n_mass_bins - 1)

    joint = age_bin * n_mass_bins + mass_bin
    n_joint_bins = int(n_age_bins * n_mass_bins)
    counts = np.bincount(joint, minlength=n_joint_bins).astype(np.float64)
    active = counts > 0
    if active.sum() == 0:
        raise ValueError("Joint curriculum found no active bins.")

    p_bin = np.zeros_like(counts, dtype=np.float64)
    p_bin[active] = counts[active] / counts[active].sum()
    return {
        "joint": joint.astype(np.int64),
        "counts": counts,
        "p_bin": p_bin,
        "active": active,
        "n_active": int(active.sum()),
    }


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
    importance_weight_min: float,
    importance_weight_max: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    joint = state["joint"]
    counts = state["counts"]
    p_bin = state["p_bin"]
    active = state["active"]
    n_active = int(state["n_active"])

    lam = float(np.clip(1.0 - tau, 0.0, 1.0))
    q_bin = np.zeros_like(p_bin, dtype=np.float64)
    q_bin[active] = (1.0 - lam) * p_bin[active] + lam * (1.0 / float(n_active))

    q_i = q_bin[joint] / counts[joint]
    q_i = q_i / q_i.sum()

    if importance_weighting:
        w_i = (p_bin[joint] / q_bin[joint]).astype(np.float32)
        w_i /= max(float(w_i.mean()), 1e-8)
        w_i = np.clip(w_i, float(importance_weight_min), float(importance_weight_max)).astype(np.float32)
        w_i /= max(float(w_i.mean()), 1e-8)
    else:
        w_i = np.ones_like(q_i, dtype=np.float32)

    return q_i.astype(np.float64), w_i, lam


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
) -> float:
    if train:
        model.train()
    else:
        model.eval()

    total = 0.0
    n_batches = 0
    for batch in loader:
        batch = _move_batch(batch, device=device)
        if train:
            optimizer.zero_grad(set_to_none=True)
            with _autocast_context(use_amp, device):
                loss = model.loss(
                    theta=batch["theta"],
                    values=batch["inputs"],
                    errors=batch["errors"],
                    observed_mask=batch["observed"],
                    sample_weights=batch.get("sample_weight"),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            with torch.no_grad():
                with _autocast_context(use_amp, device):
                    loss = model.loss(
                        theta=batch["theta"],
                        values=batch["inputs"],
                        errors=batch["errors"],
                        observed_mask=batch["observed"],
                        sample_weights=batch.get("sample_weight"),
                    )
        total += float(loss.item())
        n_batches += 1
    return total / max(n_batches, 1)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cache = load_cache_arrays(args.cache_path)
    input_columns = parse_column_csv(args.input_columns)
    theta_columns = parse_column_csv(args.theta_columns)
    if len(input_columns) == 0:
        raise ValueError("input_columns resolved to empty list.")
    if len(theta_columns) == 0:
        raise ValueError("theta_columns resolved to empty list.")

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

    model = _build_model(args, input_columns=input_columns, theta_dim=len(theta_columns))
    model.to(device)
    if args.compile:
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
    try:
        scaler = GradScaler("cuda", enabled=(args.amp and device != "cpu"))
    except TypeError:
        scaler = GradScaler(enabled=(args.amp and device != "cpu"))

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

    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    ckpt_path = os.path.join(args.output_dir, f"best_model_{args.run_name}.pt")
    hist = []

    t0 = time.time()
    for epoch in range(args.epochs):
        curriculum_log = {}
        if args.joint_curriculum:
            tau = _compute_tau(epoch, args.epochs, args.tau_max, args.tau_warmup)
            q_i, w_i, lam = _joint_curriculum_distributions(
                curriculum_state,
                tau=tau,
                importance_weighting=args.importance_weighting,
                importance_weight_min=args.importance_weight_min,
                importance_weight_max=args.importance_weight_max,
            )
            train_ds.sample_weight = torch.from_numpy(w_i.astype(np.float32))
            sampler_gen = torch.Generator()
            sampler_gen.manual_seed(int(args.seed + epoch))
            sampler = WeightedRandomSampler(
                weights=torch.from_numpy(q_i),
                num_samples=n_train_samples_per_epoch,
                replacement=True,
                generator=sampler_gen,
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
                "train_importance_weight_min": float(w_i.min()),
                "train_importance_weight_mean": float(w_i.mean()),
                "train_importance_weight_max": float(w_i.max()),
            }
        else:
            epoch_train_loader = train_loader

        train_loss = _epoch_loss(
            model,
            epoch_train_loader,
            device,
            train=True,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=args.amp,
            grad_clip_norm=args.grad_clip_norm,
        )
        val_loss = _epoch_loss(
            model,
            val_loader,
            device,
            train=False,
            use_amp=args.amp,
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        hist.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "lr": float(lr),
            }
        )
        print(
            f"Epoch {epoch + 1:04d}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_loss={val_loss:.6f} lr={lr:.2e}"
        )
        if args.joint_curriculum:
            print(
                f"  curriculum: tau={curriculum_log['tau']:.3f}, "
                f"lambda={curriculum_log['mixture_lambda']:.3f}, "
                f"w[min/mean/max]={curriculum_log['train_importance_weight_min']:.3f}/"
                f"{curriculum_log['train_importance_weight_mean']:.3f}/"
                f"{curriculum_log['train_importance_weight_max']:.3f}"
            )
        if wandb_run is not None:
            payload = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": lr,
            }
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
