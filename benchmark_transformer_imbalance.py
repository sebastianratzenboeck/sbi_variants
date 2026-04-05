#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from data import DEFAULT_INPUT_COLS, build_sbi_arrays, load_cache_arrays
from encoder import ObservationEncoder
from project_paths import REAL_CACHE_PATH, REAL_MODEL_DIR
from value_transforms import apply_inverse_value_transforms_numpy


@dataclass
class SubsetSpec:
    name: str
    pool: str
    n_rows: int | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark transformer regressors under severe logAge imbalance.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cache-path", type=str, default=str(REAL_CACHE_PATH))
    p.add_argument("--split-dir", type=str, default=str(REAL_MODEL_DIR / "age_regime_splits"))
    p.add_argument(
        "--young-index-file",
        type=str,
        default=str(REAL_MODEL_DIR / "young_test_indices_logAge_lt_7p8.npy"),
    )
    p.add_argument("--target-col", type=str, default="logAge")
    p.add_argument("--input-columns", type=str, default=",".join(DEFAULT_INPUT_COLS))
    p.add_argument("--use-colors", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--natural-train-size", type=int, default=100_000)
    p.add_argument("--balanced-train-size", type=int, default=100_000)
    p.add_argument("--val-size", type=int, default=25_000)
    p.add_argument("--natural-eval-size", type=int, default=50_000)
    p.add_argument("--balanced-eval-size", type=int, default=50_000)
    p.add_argument("--young-eval-size", type=int, default=18_000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--huber-delta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-dir", type=str, required=True)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv(raw: str) -> list[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def column_index(columns: list[str], name: str) -> int:
    return {str(c): i for i, c in enumerate(columns)}[str(name)]


def denormalize_columns(cache, values_norm: np.ndarray, columns: list[str]) -> np.ndarray:
    idx = np.asarray([column_index(list(cache.columns), c) for c in columns], dtype=np.int64)
    arr = np.asarray(values_norm, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    raw = arr * cache.stds[idx] + cache.means[idx]
    raw = apply_inverse_value_transforms_numpy(
        raw,
        transform_names=np.asarray(cache.value_transform_names[idx], dtype=object),
        transform_params=np.asarray(cache.value_transform_params[idx], dtype=np.float32),
    )
    return raw


def denormalize_target(cache, values_norm_1d: np.ndarray, target_col: str) -> np.ndarray:
    return denormalize_columns(cache, np.asarray(values_norm_1d, dtype=np.float32), [target_col]).reshape(-1)


def _build_edges(values: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if strategy == "quantile":
        edges = np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1))
    elif strategy == "equal_width":
        vmin = float(values.min())
        vmax = float(values.max())
        if vmax <= vmin:
            vmax = vmin + 1e-6
        edges = np.linspace(vmin, vmax, n_bins + 1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    eps = max(np.finfo(np.float64).eps * max(float(np.max(np.abs(values))), 1.0), 1e-12)
    edges = np.asarray(edges, dtype=np.float64)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + eps
    if edges[-1] <= edges[-2]:
        edges[-1] = edges[-2] + eps
    return edges


def prepare_age_mass_bin_state(cache, row_indices: np.ndarray, *, n_age_bins: int, n_mass_bins: int) -> dict:
    rows = np.asarray(row_indices, dtype=np.int64)
    age_mass = denormalize_columns(
        cache,
        cache.values_norm[rows][:, [column_index(list(cache.columns), "logAge"), column_index(list(cache.columns), "m_init")]],
        ["logAge", "m_init"],
    )
    age = age_mass[:, 0]
    mass = age_mass[:, 1]
    age_edges = _build_edges(age, n_age_bins, "equal_width")
    mass_edges = _build_edges(mass, n_mass_bins, "equal_width")
    age_bin = np.clip(np.searchsorted(age_edges, age, side="right") - 1, 0, n_age_bins - 1)
    mass_bin = np.clip(np.searchsorted(mass_edges, mass, side="right") - 1, 0, n_mass_bins - 1)
    joint = age_bin * n_mass_bins + mass_bin
    _, inverse, counts = np.unique(joint, return_inverse=True, return_counts=True)
    inv_counts = 1.0 / counts[inverse].astype(np.float64)
    sample_probs = inv_counts / inv_counts.sum()
    natural_probs = np.full(rows.size, 1.0 / rows.size, dtype=np.float64)
    importance = natural_probs / sample_probs
    importance /= np.mean(importance)
    return {
        "sample_probs": sample_probs.astype(np.float64),
        "importance_weights": importance.astype(np.float32),
        "joint_bins": joint.astype(np.int64),
        "n_active_bins": int(np.unique(joint).size),
    }


def choose_rows(pool_rows: np.ndarray, n_rows: int | None, seed: int) -> np.ndarray:
    pool_rows = np.asarray(pool_rows, dtype=np.int64)
    if n_rows is None or n_rows >= pool_rows.size:
        return np.sort(pool_rows.copy())
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(pool_rows, size=n_rows, replace=False).astype(np.int64))


class RegressionDataset(Dataset):
    def __init__(self, arrays, sample_weights: np.ndarray | None = None):
        self.inputs = torch.tensor(arrays.inputs, dtype=torch.float32)
        self.errors = torch.tensor(arrays.input_errors, dtype=torch.float32)
        self.observed = torch.tensor(arrays.input_observed, dtype=torch.float32)
        self.targets = torch.tensor(arrays.theta[:, 0], dtype=torch.float32)
        if sample_weights is None:
            sample_weights = np.ones(len(self.targets), dtype=np.float32)
        self.sample_weights = torch.tensor(sample_weights, dtype=torch.float32)

    def __len__(self) -> int:
        return self.inputs.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": self.inputs[idx],
            "errors": self.errors[idx],
            "observed": self.observed[idx],
            "target": self.targets[idx],
            "sample_weight": self.sample_weights[idx],
        }


class TransformerScalarRegressor(nn.Module):
    def __init__(self, input_columns: list[str], *, architecture: str = "mean"):
        super().__init__()
        self.architecture = str(architecture)
        pooling_mode = "attention" if self.architecture == "attention_pool" else "mean"
        self.encoder = ObservationEncoder(
            input_columns=input_columns,
            dim_value=24,
            dim_id=24,
            dim_error=16,
            dim_observed=8,
            attn_embed_dim=128,
            num_heads=8,
            num_layers=4,
            widening_factor=4,
            dropout=0.05,
            use_missingness_context=True,
            missingness_context_hidden_dim=64,
            pooling_mode=pooling_mode,
        )
        if self.architecture == "xattn_query":
            self.query = nn.Parameter(torch.zeros(1, 1, self.encoder.output_dim))
            nn.init.normal_(self.query, mean=0.0, std=0.02)
            self.query_norm = nn.LayerNorm(self.encoder.output_dim)
            self.ctx_norm = nn.LayerNorm(self.encoder.output_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.encoder.output_dim,
                num_heads=8,
                dropout=0.05,
                batch_first=True,
            )
            self.ff_norm = nn.LayerNorm(self.encoder.output_dim)
            self.ff = nn.Sequential(
                nn.Linear(self.encoder.output_dim, 256),
                nn.SiLU(),
                nn.Dropout(0.05),
                nn.Linear(256, self.encoder.output_dim),
            )
            self.head = nn.Sequential(
                nn.LayerNorm(self.encoder.output_dim),
                nn.Linear(self.encoder.output_dim, 256),
                nn.SiLU(),
                nn.Dropout(0.05),
                nn.Linear(256, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.LayerNorm(self.encoder.output_dim),
                nn.Linear(self.encoder.output_dim, 256),
                nn.SiLU(),
                nn.Dropout(0.05),
                nn.Linear(256, 1),
            )

    def forward(self, values: torch.Tensor, errors: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
        if self.architecture == "xattn_query":
            tokens, token_mask = self.encoder.forward_tokens(values, errors, observed)
            ctx_tokens = self.ctx_norm(tokens)
            query = self.query_norm(self.query.expand(values.shape[0], -1, -1))
            attn_out, _ = self.cross_attn(
                query=query,
                key=ctx_tokens,
                value=ctx_tokens,
                key_padding_mask=~token_mask,
                need_weights=False,
            )
            h = query + attn_out
            h = h + self.ff(self.ff_norm(h))
            return self.head(h.squeeze(1)).squeeze(-1)
        return self.head(self.encoder(values, errors, observed)).squeeze(-1)


def build_tabular_features(arrays, *, missing_value=np.nan) -> np.ndarray:
    values = np.asarray(arrays.inputs, dtype=np.float32).copy()
    errors = np.asarray(arrays.input_errors, dtype=np.float32).copy()
    observed = np.asarray(arrays.input_observed, dtype=np.float32).copy()
    missing = observed < 0.5
    values[missing] = missing_value
    errors[missing] = missing_value
    return np.concatenate([values, errors, observed], axis=1).astype(np.float32)


def fit_transformer(
    *,
    train_arrays,
    val_arrays,
    model_input_columns: list[str],
    device: str,
    batch_size: int,
    epochs: int,
    patience: int,
    lr: float,
    lr_min: float,
    weight_decay: float,
    huber_delta: float,
    sample_weights: np.ndarray | None = None,
    sampler_probs: np.ndarray | None = None,
    architecture: str = "mean",
) -> tuple[nn.Module, pd.DataFrame]:
    model = TransformerScalarRegressor(model_input_columns, architecture=architecture).to(device)
    train_ds = RegressionDataset(train_arrays, sample_weights=sample_weights)
    val_ds = RegressionDataset(val_arrays)
    sampler = None
    if sampler_probs is not None:
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sampler_probs, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)

    best_state = None
    best_val = float("inf")
    no_improve = 0
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(batch["inputs"].to(device), batch["errors"].to(device), batch["observed"].to(device))
            per_example = nn.functional.huber_loss(
                pred,
                batch["target"].to(device),
                reduction="none",
                delta=huber_delta,
            )
            weights = batch["sample_weight"].to(device)
            loss = torch.sum(per_example * weights) / torch.clamp(weights.sum(), min=1e-8)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []
        val_true = []
        val_pred = []
        with torch.no_grad():
            for batch in val_loader:
                pred = model(batch["inputs"].to(device), batch["errors"].to(device), batch["observed"].to(device))
                loss = nn.functional.huber_loss(
                    pred,
                    batch["target"].to(device),
                    reduction="mean",
                    delta=huber_delta,
                )
                val_losses.append(float(loss.detach().cpu().item()))
                val_true.append(batch["target"].cpu().numpy())
                val_pred.append(pred.cpu().numpy())
        scheduler.step()
        val_true_np = np.concatenate(val_true)
        val_pred_np = np.concatenate(val_pred)
        val_mae = float(mean_absolute_error(val_true_np, val_pred_np))
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(train_losses)),
                "val_loss": float(np.mean(val_losses)),
                "val_mae_norm": val_mae,
                "lr": float(scheduler.get_last_lr()[0]),
            }
        )
        if val_mae < best_val:
            best_val = val_mae
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(history)


def predict_transformer(model: nn.Module, arrays, *, batch_size: int, device: str) -> np.ndarray:
    ds = RegressionDataset(arrays)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    preds = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            pred = model(batch["inputs"].to(device), batch["errors"].to(device), batch["observed"].to(device))
            preds.append(pred.cpu().numpy())
    return np.concatenate(preds).astype(np.float32)


def regression_metrics(cache, target_col: str, y_true_norm: np.ndarray, y_pred_norm: np.ndarray) -> dict[str, float]:
    y_true_phys = denormalize_target(cache, y_true_norm, target_col)
    y_pred_phys = denormalize_target(cache, y_pred_norm, target_col)
    return {
        "rmse_norm": float(np.sqrt(mean_squared_error(y_true_norm, y_pred_norm))),
        "mae_norm": float(mean_absolute_error(y_true_norm, y_pred_norm)),
        "r2_norm": float(r2_score(y_true_norm, y_pred_norm)),
        "rmse_phys": float(np.sqrt(mean_squared_error(y_true_phys, y_pred_phys))),
        "mae_phys": float(mean_absolute_error(y_true_phys, y_pred_phys)),
        "r2_phys": float(r2_score(y_true_phys, y_pred_phys)),
        "pred_min_phys": float(np.min(y_pred_phys)),
        "pred_p01_phys": float(np.quantile(y_pred_phys, 0.01)),
        "pred_p05_phys": float(np.quantile(y_pred_phys, 0.05)),
        "true_p01_phys": float(np.quantile(y_true_phys, 0.01)),
        "true_p05_phys": float(np.quantile(y_true_phys, 0.05)),
        "frac_pred_below_7p8": float(np.mean(y_pred_phys < 7.8)),
        "frac_true_below_7p8": float(np.mean(y_true_phys < 7.8)),
    }


def fit_tabular_models(seed: int) -> dict[str, object]:
    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("model", RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=2, n_jobs=-1, random_state=seed)),
        ]
    )
    hgb = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_depth=8,
        max_iter=300,
        min_samples_leaf=50,
        random_state=seed,
        early_stopping=False,
    )
    return {"random_forest": rf, "hist_gbdt": hgb}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cache = load_cache_arrays(args.cache_path)
    input_columns = parse_csv(args.input_columns)
    split_dir = Path(args.split_dir)

    train_pools = {
        "natural": np.load(split_dir / "train_indices_all.npy").astype(np.int64),
        "balanced": np.load(split_dir / "train_indices_age_balanced.npy").astype(np.int64),
    }
    val_pools = {
        "natural": np.load(split_dir / "val_indices_all.npy").astype(np.int64),
        "balanced": np.load(split_dir / "val_indices_age_balanced.npy").astype(np.int64),
    }
    eval_pools = {
        "natural_test_50k": choose_rows(np.load(REAL_MODEL_DIR / "test_indices.npy").astype(np.int64), args.natural_eval_size, args.seed + 100),
        "balanced_age_50k": choose_rows(np.load(split_dir / "eval_indices_balanced_age_300k.npy").astype(np.int64), args.balanced_eval_size, args.seed + 101),
        "young_test_all": choose_rows(np.load(args.young_index_file).astype(np.int64), args.young_eval_size, args.seed + 102),
    }

    train_specs = [
        ("natural_fixed", choose_rows(train_pools["natural"], args.natural_train_size, args.seed + 1), choose_rows(val_pools["natural"], args.val_size, args.seed + 11)),
        ("age_balanced_fixed", choose_rows(train_pools["balanced"], args.balanced_train_size, args.seed + 2), choose_rows(val_pools["balanced"], args.val_size, args.seed + 12)),
    ]

    variant_defs = [
        {"name": "transformer_mean_vanilla", "architecture": "mean", "sampler_probs": None, "use_importance_weights": False},
        {"name": "transformer_mean_curriculum_only", "architecture": "mean", "sampler_probs": "curriculum", "use_importance_weights": False},
        {"name": "transformer_mean_curriculum_iw", "architecture": "mean", "sampler_probs": "curriculum", "use_importance_weights": True},
        {"name": "transformer_attnpool_vanilla", "architecture": "attention_pool", "sampler_probs": None, "use_importance_weights": False},
        {"name": "transformer_attnpool_curriculum_iw", "architecture": "attention_pool", "sampler_probs": "curriculum", "use_importance_weights": True},
        {"name": "transformer_xattn_vanilla", "architecture": "xattn_query", "sampler_probs": None, "use_importance_weights": False},
        {"name": "transformer_xattn_curriculum_iw", "architecture": "xattn_query", "sampler_probs": "curriculum", "use_importance_weights": True},
    ]

    all_results = []
    history_frames = []

    for train_name, train_rows, val_rows in train_specs:
        train_arrays = build_sbi_arrays(cache, row_indices=train_rows, input_columns=input_columns, theta_columns=[args.target_col], use_colors=args.use_colors)
        color_norm_stats = None
        if args.use_colors and train_arrays.color_means is not None:
            color_norm_stats = (train_arrays.color_means, train_arrays.color_stds)
        val_arrays = build_sbi_arrays(
            cache,
            row_indices=val_rows,
            input_columns=input_columns,
            theta_columns=[args.target_col],
            use_colors=args.use_colors,
            color_norm_stats=color_norm_stats,
        )
        model_input_columns = list(input_columns)
        if args.use_colors and train_arrays.color_names:
            model_input_columns += list(train_arrays.color_names)

        curriculum = prepare_age_mass_bin_state(cache, train_rows, n_age_bins=25, n_mass_bins=12)
        _, inverse, counts = np.unique(curriculum["joint_bins"], return_inverse=True, return_counts=True)
        sample_probs = (1.0 / counts[inverse].astype(np.float64))
        sample_probs /= sample_probs.sum()
        natural_probs = np.full(train_rows.size, 1.0 / train_rows.size, dtype=np.float64)
        importance = natural_probs / sample_probs
        importance /= np.mean(importance)

        transformer_models = {}
        for variant in variant_defs:
            model, history = fit_transformer(
                train_arrays=train_arrays,
                val_arrays=val_arrays,
                model_input_columns=model_input_columns,
                device=device,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                lr=args.lr,
                lr_min=args.lr_min,
                weight_decay=args.weight_decay,
                huber_delta=args.huber_delta,
                sample_weights=importance.astype(np.float32) if variant["use_importance_weights"] else None,
                sampler_probs=sample_probs if variant["sampler_probs"] == "curriculum" else None,
                architecture=str(variant["architecture"]),
            )
            transformer_models[variant["name"]] = model
            hist = history.copy()
            hist["train_subset"] = train_name
            hist["model"] = variant["name"]
            history_frames.append(hist)

        X_train = build_tabular_features(train_arrays, missing_value=np.nan)
        y_train = np.asarray(train_arrays.theta[:, 0], dtype=np.float32)
        tabular_models = fit_tabular_models(args.seed)
        for m in tabular_models.values():
            m.fit(X_train, y_train)

        for eval_name, eval_rows in eval_pools.items():
            eval_arrays = build_sbi_arrays(
                cache,
                row_indices=eval_rows,
                input_columns=input_columns,
                theta_columns=[args.target_col],
                use_colors=args.use_colors,
                color_norm_stats=color_norm_stats,
            )
            y_true = np.asarray(eval_arrays.theta[:, 0], dtype=np.float32)

            for model_name, model in transformer_models.items():
                y_pred = predict_transformer(model, eval_arrays, batch_size=args.batch_size, device=device)
                row = {
                    "target": args.target_col,
                    "train_subset": train_name,
                    "eval_subset": eval_name,
                    "model": model_name,
                    "n_train": int(train_rows.size),
                    "n_eval": int(eval_rows.size),
                }
                row.update(regression_metrics(cache, args.target_col, y_true, y_pred))
                all_results.append(row)

            X_eval = build_tabular_features(eval_arrays, missing_value=np.nan)
            for model_name, model in tabular_models.items():
                y_pred = np.asarray(model.predict(X_eval), dtype=np.float32)
                row = {
                    "target": args.target_col,
                    "train_subset": train_name,
                    "eval_subset": eval_name,
                    "model": model_name,
                    "n_train": int(train_rows.size),
                    "n_eval": int(eval_rows.size),
                }
                row.update(regression_metrics(cache, args.target_col, y_true, y_pred))
                all_results.append(row)

    results_df = pd.DataFrame(all_results).sort_values(["eval_subset", "train_subset", "model"])
    history_df = pd.concat(history_frames, ignore_index=True).sort_values(["train_subset", "model", "epoch"])
    results_df.to_csv(out / "benchmark_results.csv", index=False)
    history_df.to_csv(out / "benchmark_history.csv", index=False)

    summary = {
        "target": args.target_col,
        "device": device,
        "natural_train_size": args.natural_train_size,
        "balanced_train_size": args.balanced_train_size,
        "val_size": args.val_size,
        "natural_eval_size": args.natural_eval_size,
        "balanced_eval_size": args.balanced_eval_size,
        "young_eval_size": args.young_eval_size,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
