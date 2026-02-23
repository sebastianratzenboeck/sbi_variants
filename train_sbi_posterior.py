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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

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
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--max-stars", type=int, default=None,
                   help="Optional cap on train+val rows after exclusions.")
    p.add_argument("--seed", type=int, default=42)

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
    return args


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
        train_loss = _epoch_loss(
            model,
            train_loader,
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
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": lr,
                }
            )

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
