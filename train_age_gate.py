#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Sampler

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in os.sys.path:
        os.sys.path.insert(0, str(repo_root))

from age_gate_models import AgeGateClassifier
from age_regimes import DEFAULT_AGE_BIN_EDGES, regime_names, logage_to_regime_index, summarize_regime_counts
from data import build_sbi_arrays, load_cache_arrays, load_indices, parse_column_csv, DEFAULT_INPUT_COLS
from encoder import ObservationEncoder
from inference_utils import NormStats

try:
    from torch.amp import GradScaler, autocast

    def _autocast_context(enabled: bool, device: str):
        return autocast("cuda", enabled=(enabled and device.startswith("cuda")))
except ImportError:
    from torch.cuda.amp import GradScaler, autocast

    def _autocast_context(enabled: bool, device: str):
        return autocast(enabled=(enabled and device.startswith("cuda")))


class AgeGateDataset(Dataset):
    def __init__(self, *, inputs: np.ndarray, errors: np.ndarray, observed: np.ndarray, labels: np.ndarray):
        self.inputs = torch.tensor(inputs, dtype=torch.float32)
        self.errors = torch.tensor(errors, dtype=torch.float32)
        self.observed = torch.tensor(observed, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": self.inputs[idx],
            "errors": self.errors[idx],
            "observed": self.observed[idx],
            "label": self.labels[idx],
        }


class BalancedClassSampler(Sampler[int]):
    def __init__(self, labels: np.ndarray, *, epoch_size: int, seed: int):
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        counts = np.bincount(labels)
        if labels.size == 0:
            raise ValueError("labels must be non-empty")
        if np.any(counts == 0):
            raise ValueError(f"balanced sampler requires every regime present, got counts={counts.tolist()}")
        if epoch_size <= 0:
            raise ValueError(f"epoch_size must be positive, got {epoch_size}")
        self.epoch_size = int(epoch_size)
        self.seed = int(seed)
        self.num_classes = int(counts.size)
        self.class_indices = [
            torch.tensor(np.flatnonzero(labels == k), dtype=torch.long)
            for k in range(self.num_classes)
        ]
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.epoch_size

    def __iter__(self):
        gen = torch.Generator()
        gen.manual_seed(self.seed + self._epoch)
        class_draws = torch.randint(
            low=0,
            high=self.num_classes,
            size=(self.epoch_size,),
            generator=gen,
        )
        sample_idx = torch.empty(self.epoch_size, dtype=torch.long)
        for k, idx_k in enumerate(self.class_indices):
            mask = class_draws == k
            n_k = int(mask.sum().item())
            if n_k == 0:
                continue
            picked = idx_k[torch.randint(low=0, high=idx_k.numel(), size=(n_k,), generator=gen)]
            sample_idx[mask] = picked
        return iter(sample_idx.tolist())


def _parse_edges(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("age-bin-edges must contain at least one edge")
    return vals


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train an age-regime gate on top of the observation encoder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cache-path", type=str, required=True)
    p.add_argument("--norm-meta-path", type=str, required=True)
    p.add_argument("--train-indices", type=str, required=True)
    p.add_argument("--val-indices", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--run-name", type=str, default="age_gate")
    p.add_argument("--input-columns", type=str, default=",".join(DEFAULT_INPUT_COLS))
    p.add_argument("--use-colors", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--age-bin-edges", type=str, default=",".join(str(x) for x in DEFAULT_AGE_BIN_EDGES))
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--epoch-size", type=int, default=5000000)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true", default=False)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--dim-value", type=int, default=24)
    p.add_argument("--dim-id", type=int, default=24)
    p.add_argument("--dim-error", type=int, default=16)
    p.add_argument("--dim-observed", type=int, default=8)
    p.add_argument("--value-calibration-type", type=str, default="scalar_film")
    p.add_argument("--error-embed-type", type=str, default="mlp_regime")
    p.add_argument("--attn-embed-dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--widening-factor", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--use-missingness-context", action="store_true", default=True)
    p.add_argument("--missingness-context-hidden-dim", type=int, default=64)
    p.add_argument("--gate-hidden-dim", type=int, default=256)
    p.add_argument("--gate-dropout", type=float, default=0.0)
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", type=str, default="mock-galaxy-simformer")
    return p


def _build_encoder(args: argparse.Namespace, input_columns: list[str]) -> ObservationEncoder:
    return ObservationEncoder(
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


def _make_labels(cache, row_indices: np.ndarray, stats: NormStats, *, edges: list[float]) -> np.ndarray:
    logage_idx = cache.columns.index("logAge")
    logage_norm = cache.values_norm[row_indices][:, [logage_idx]].astype(np.float32)
    logage_phys = stats.denormalize_numpy(logage_norm, [logage_idx]).reshape(-1)
    return logage_to_regime_index(logage_phys, edges=edges).astype(np.int64)


def _run_epoch(model, loader, *, device: str, train: bool, optimizer=None, scaler=None, use_amp: bool, grad_clip_norm: float) -> dict[str, float]:
    model.train(train)
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    for batch in loader:
        x = batch["inputs"].to(device, non_blocking=True)
        e = batch["errors"].to(device, non_blocking=True)
        o = batch["observed"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        if train:
            optimizer.zero_grad(set_to_none=True)
        with _autocast_context(use_amp, device):
            logits = model(x, e, o)
            loss = F.cross_entropy(logits, y)
        if train:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()
        total_loss += float(loss.detach().item()) * int(y.shape[0])
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_n += int(y.shape[0])
    return {
        "loss": total_loss / max(total_n, 1),
        "acc": total_correct / max(total_n, 1),
    }


def _fit_temperature(model, loader, *, device: str) -> float:
    model.eval()
    logits_all = []
    labels_all = []
    with torch.no_grad():
        for batch in loader:
            x = batch["inputs"].to(device, non_blocking=True)
            e = batch["errors"].to(device, non_blocking=True)
            o = batch["observed"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            logits_all.append(model(x, e, o))
            labels_all.append(y)
    logits = torch.cat(logits_all, dim=0)
    labels = torch.cat(labels_all, dim=0)
    log_temp = torch.nn.Parameter(torch.zeros((), device=device))
    opt = torch.optim.LBFGS([log_temp], lr=0.1, max_iter=50)

    def closure():
        opt.zero_grad()
        temp = torch.exp(log_temp).clamp_min(1e-6)
        loss = F.cross_entropy(logits / temp, labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_temp).detach().cpu().item())


def main() -> None:
    args = _build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    edges = _parse_edges(args.age_bin_edges)
    reg_names = regime_names(edges)
    input_columns = parse_column_csv(args.input_columns)

    cache = load_cache_arrays(args.cache_path)
    stats = NormStats(args.norm_meta_path)
    train_rows = load_indices(args.train_indices)
    val_rows = load_indices(args.val_indices)
    if train_rows is None or val_rows is None:
        raise ValueError("train/val index files are required")

    arr_train = build_sbi_arrays(
        cache,
        row_indices=train_rows,
        input_columns=input_columns,
        theta_columns=["logAge"],
        use_colors=args.use_colors,
    )
    color_norm_stats = None
    if args.use_colors and arr_train.color_means is not None:
        color_norm_stats = (arr_train.color_means, arr_train.color_stds)
    arr_val = build_sbi_arrays(
        cache,
        row_indices=val_rows,
        input_columns=input_columns,
        theta_columns=["logAge"],
        use_colors=args.use_colors,
        color_norm_stats=color_norm_stats,
    )

    y_train = _make_labels(cache, train_rows, stats, edges=edges)
    y_val = _make_labels(cache, val_rows, stats, edges=edges)

    extended_input_columns = list(input_columns)
    if args.use_colors and arr_train.color_names is not None:
        extended_input_columns += list(arr_train.color_names)

    train_ds = AgeGateDataset(inputs=arr_train.inputs, errors=arr_train.input_errors, observed=arr_train.input_observed, labels=y_train)
    val_ds = AgeGateDataset(inputs=arr_val.inputs, errors=arr_val.input_errors, observed=arr_val.input_observed, labels=y_val)

    train_sampler = BalancedClassSampler(y_train, epoch_size=args.epoch_size, seed=args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=(device != "cpu" and torch.cuda.is_available()),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device != "cpu" and torch.cuda.is_available()),
    )

    encoder = _build_encoder(args, extended_input_columns)
    model = AgeGateClassifier(
        encoder=encoder,
        num_regimes=len(reg_names),
        hidden_dim=args.gate_hidden_dim,
        dropout=args.gate_dropout,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    try:
        scaler = GradScaler("cuda", enabled=bool(args.amp and device.startswith("cuda")))
    except TypeError:
        scaler = GradScaler(enabled=bool(args.amp and device.startswith("cuda")))

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                **vars(args),
                "regime_names": reg_names,
                "train_counts": summarize_regime_counts(y_train, edges=edges, names=reg_names),
                "val_counts": summarize_regime_counts(y_val, edges=edges, names=reg_names),
            },
        )

    best_val = float("inf")
    best_epoch = -1
    no_improve = 0
    history: list[dict[str, float | int]] = []
    ckpt_path = os.path.join(args.output_dir, f"best_age_gate_{args.run_name}.pt")

    print(f"Using device: {device}")
    print(f"Regimes: {reg_names}")
    print(f"Train counts: {summarize_regime_counts(y_train, edges=edges, names=reg_names)}")
    print(f"Val counts: {summarize_regime_counts(y_val, edges=edges, names=reg_names)}")
    print(f"Epoch size: {args.epoch_size:,}")

    t0 = time.time()
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_stats = _run_epoch(model, train_loader, device=device, train=True, optimizer=optimizer, scaler=scaler, use_amp=bool(args.amp), grad_clip_norm=args.grad_clip_norm)
        val_stats = _run_epoch(model, val_loader, device=device, train=False, optimizer=None, scaler=None, use_amp=bool(args.amp), grad_clip_norm=args.grad_clip_norm)
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "train_loss": float(train_stats["loss"]),
            "train_acc": float(train_stats["acc"]),
            "val_loss": float(val_stats["loss"]),
            "val_acc": float(val_stats["acc"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        print(
            f"Epoch {epoch + 1:04d}/{args.epochs} train_loss={train_stats['loss']:.6f} train_acc={train_stats['acc']:.4f} "
            f"val_loss={val_stats['loss']:.6f} val_acc={val_stats['acc']:.4f} lr={row['lr']:.2e}"
        )
        if wandb_run is not None:
            wandb_run.log(dict(row))
        if val_stats["loss"] < best_val:
            best_val = float(val_stats["loss"])
            best_epoch = epoch + 1
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved new best checkpoint to {ckpt_path}")
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}; best epoch={best_epoch}, best val_loss={best_val:.6f}")
            break

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    temperature = _fit_temperature(model, val_loader, device=device)
    elapsed = (time.time() - t0) / 60.0
    print(f"Training finished in {elapsed:.1f} min. Best val_loss={best_val:.6f} at epoch {best_epoch}.")
    print(f"Calibrated temperature: {temperature:.6f}")

    config_out = {
        **vars(args),
        "input_columns": input_columns,
        "input_columns_with_colors": extended_input_columns,
        "regime_names": reg_names,
        "age_bin_edges": edges,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "temperature": temperature,
        "train_counts": summarize_regime_counts(y_train, edges=edges, names=reg_names),
        "val_counts": summarize_regime_counts(y_val, edges=edges, names=reg_names),
        "checkpoint_path": ckpt_path,
    }
    config_path = os.path.join(args.output_dir, f"age_gate_config_{args.run_name}.json")
    hist_path = os.path.join(args.output_dir, f"age_gate_history_{args.run_name}.json")
    temp_path = os.path.join(args.output_dir, f"age_gate_temperature_{args.run_name}.json")
    meta_path = os.path.join(args.output_dir, f"age_gate_norm_meta_{args.run_name}.npz")
    with open(config_path, "w") as f:
        json.dump(config_out, f, indent=2)
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    with open(temp_path, "w") as f:
        json.dump({"temperature": temperature}, f, indent=2)
    np.savez(
        meta_path,
        columns=np.asarray(cache.columns, dtype=object),
        means=cache.means if cache.means is not None else np.zeros(len(cache.columns), dtype=np.float32),
        stds=cache.stds if cache.stds is not None else np.ones(len(cache.columns), dtype=np.float32),
        value_transform_names=cache.value_transform_names if cache.value_transform_names is not None else np.asarray(["identity"] * len(cache.columns), dtype=object),
        value_transform_params=cache.value_transform_params if cache.value_transform_params is not None else np.zeros(len(cache.columns), dtype=np.float32),
        log_err_mean=np.array(cache.log_err_mean if cache.log_err_mean is not None else 0.0, dtype=np.float32),
        log_err_std=np.array(cache.log_err_std if cache.log_err_std is not None else 1.0, dtype=np.float32),
        input_columns=np.asarray(extended_input_columns, dtype=object),
        input_columns_base=np.asarray(input_columns, dtype=object),
        use_colors=np.array(args.use_colors, dtype=bool),
        age_bin_edges=np.asarray(edges, dtype=np.float32),
        regime_names=np.asarray(reg_names, dtype=object),
    )
    print(f"Saved config: {config_path}")
    print(f"Saved history: {hist_path}")
    print(f"Saved temperature: {temp_path}")
    print(f"Saved normalization metadata: {meta_path}")
    if wandb_run is not None:
        wandb_run.summary["best_val_loss"] = best_val
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["temperature"] = temperature
        wandb_run.finish()


if __name__ == "__main__":
    main()
