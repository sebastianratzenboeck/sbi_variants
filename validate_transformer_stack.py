#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from columns import OBS_COLS
from data import DEFAULT_INPUT_COLS
from encoder import ObservationEncoder
from sampling import build_inference_edge_mask
from transformer import (
    ErrorEmbed,
    ErrorEmbedMLP,
    ObservedEmbed,
    Simformer,
    Tokenizer,
    ValueCalibratorScalar,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Validate tokenizer/transformer infrastructure with structural and learning sanity checks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--output-dir", type=str, default="transformer_validation_outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--train-size", type=int, default=2048)
    p.add_argument("--val-size", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--quick", action="store_true", default=False,
                   help="Run a smaller/shorter sanity suite.")
    return p.parse_args()


@dataclass
class VariantSpec:
    name: str
    value_calibration_type: str
    error_embed_type: str
    use_missingness_context: bool


class SyntheticObsDataset(Dataset):
    def __init__(self, values, errors, observed, y_reg, y_cls):
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.errors = torch.as_tensor(errors, dtype=torch.float32)
        self.observed = torch.as_tensor(observed, dtype=torch.float32)
        self.y_reg = torch.as_tensor(y_reg, dtype=torch.float32)
        self.y_cls = torch.as_tensor(y_cls, dtype=torch.float32)

    def __len__(self):
        return int(self.values.shape[0])

    def __getitem__(self, idx):
        return {
            "values": self.values[idx],
            "errors": self.errors[idx],
            "observed": self.observed[idx],
            "y_reg": self.y_reg[idx],
            "y_cls": self.y_cls[idx],
        }


class EncoderHead(nn.Module):
    def __init__(self, encoder: ObservationEncoder):
        super().__init__()
        self.encoder = encoder
        self.reg_head = nn.Sequential(
            nn.Linear(encoder.output_dim, encoder.output_dim),
            nn.SiLU(),
            nn.Linear(encoder.output_dim, 1),
        )
        self.cls_head = nn.Sequential(
            nn.Linear(encoder.output_dim, encoder.output_dim),
            nn.SiLU(),
            nn.Linear(encoder.output_dim, 1),
        )

    def forward(self, values, errors, observed):
        h = self.encoder(values, errors, observed)
        reg = self.reg_head(h).squeeze(-1)
        cls = self.cls_head(h).squeeze(-1)
        return reg, cls


def _make_input_columns() -> list[str]:
    cols = list(DEFAULT_INPUT_COLS)
    if len(cols) >= 12:
        return cols[:12]
    return cols


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _synthetic_batch(n: int, input_columns: list[str], seed: int):
    rng = np.random.default_rng(seed)
    n_features = len(input_columns)

    z_age = rng.normal(size=n).astype(np.float32)
    z_mass = rng.normal(size=n).astype(np.float32)
    z_dust = rng.normal(size=n).astype(np.float32)
    z_noise = rng.normal(size=(n, n_features)).astype(np.float32)

    values = np.zeros((n, n_features), dtype=np.float32)
    for j, col in enumerate(input_columns):
        scale = 0.6 + 0.07 * j
        values[:, j] = (
            scale * z_age
            + (0.3 - 0.02 * j) * z_mass
            + 0.15 * z_dust
            + 0.3 * z_noise[:, j]
        )
        if col in OBS_COLS:
            values[:, j] += 0.1 * np.sin(z_age * (j + 1))

    observed = np.ones((n, n_features), dtype=np.float32)
    obs_cols = [j for j, c in enumerate(input_columns) if c in OBS_COLS]
    for j in obs_cols:
        p = 1.0 / (1.0 + np.exp(-(0.7 * z_age - 0.25 * z_dust + 0.08 * j)))
        keep = rng.uniform(size=n) < p
        observed[:, j] = keep.astype(np.float32)
    # keep sky coordinates always observed
    for j, c in enumerate(input_columns):
        if c not in OBS_COLS:
            observed[:, j] = 1.0

    values = np.where(observed > 0.5, values, 0.0).astype(np.float32)

    errors = np.zeros((n, n_features), dtype=np.float32)
    for j, c in enumerate(input_columns):
        if c in OBS_COLS:
            raw = np.abs(0.25 + 0.1 * rng.normal(size=n) + 0.05 * np.abs(z_dust))
            zerr = np.log(np.clip(raw, 1e-4, None)).astype(np.float32)
            errors[:, j] = np.where(observed[:, j] > 0.5, zerr, 5.0).astype(np.float32)
            perfect = (observed[:, j] > 0.5) & (rng.uniform(size=n) < 0.05)
            errors[perfect, j] = -5.0
        else:
            errors[:, j] = 0.0

    obs_idx = np.asarray(obs_cols, dtype=np.int64)
    obs_values = values[:, obs_idx]
    obs_mask = observed[:, obs_idx]
    obs_errors = errors[:, obs_idx]
    obs_count = np.maximum(obs_mask.sum(axis=1), 1.0)
    mean_obs = (obs_values * obs_mask).sum(axis=1) / obs_count
    real_err = ((obs_errors > -4.9) & (obs_errors < 4.9) & (obs_mask > 0.5)).astype(np.float32)
    real_err_sum = (np.where(real_err > 0.5, obs_errors, 0.0)).sum(axis=1)
    real_err_count = np.maximum(real_err.sum(axis=1), 1.0)
    mean_real_err = real_err_sum / real_err_count

    y_reg = (
        0.9 * z_age
        - 0.35 * z_mass
        + 0.25 * mean_obs
        - 0.12 * mean_real_err
        + 0.08 * obs_count / max(len(obs_cols), 1)
    ).astype(np.float32)
    y_cls = (z_age + 0.35 * mean_obs - 0.15 * z_dust > 0.15).astype(np.float32)
    return values, errors, observed, y_reg, y_cls


def _make_dataloaders(input_columns: list[str], train_size: int, val_size: int, batch_size: int, seed: int):
    train = SyntheticObsDataset(*_synthetic_batch(train_size, input_columns, seed))
    val = SyntheticObsDataset(*_synthetic_batch(val_size, input_columns, seed + 1))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, drop_last=False),
        DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=False),
    )


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _evaluate_model(model: EncoderHead, loader: DataLoader, device: str):
    model.eval()
    reg_pred, reg_true, cls_pred, cls_true = [], [], [], []
    with torch.no_grad():
        for batch in loader:
            values = batch["values"].to(device)
            errors = batch["errors"].to(device)
            observed = batch["observed"].to(device)
            y_reg = batch["y_reg"].to(device)
            y_cls = batch["y_cls"].to(device)
            pred_reg, pred_cls = model(values, errors, observed)
            reg_pred.append(pred_reg.cpu().numpy())
            reg_true.append(y_reg.cpu().numpy())
            cls_pred.append(torch.sigmoid(pred_cls).cpu().numpy())
            cls_true.append(y_cls.cpu().numpy())
    reg_pred = np.concatenate(reg_pred)
    reg_true = np.concatenate(reg_true)
    cls_prob = np.concatenate(cls_pred)
    cls_true = np.concatenate(cls_true)
    cls_hat = (cls_prob >= 0.5).astype(np.float32)
    return {
        "reg_mae": float(np.mean(np.abs(reg_pred - reg_true))),
        "reg_r2": float(_r2_score(reg_true, reg_pred)),
        "cls_acc": float(np.mean(cls_hat == cls_true)),
        "cls_brier": float(np.mean((cls_prob - cls_true) ** 2)),
    }


def _train_variant(spec: VariantSpec, *, input_columns: list[str], train_loader: DataLoader, val_loader: DataLoader, device: str, epochs: int):
    encoder = ObservationEncoder(
        input_columns=input_columns,
        dim_value=16,
        dim_id=16,
        value_calibration_type=spec.value_calibration_type,
        dim_error=8,
        error_embed_type=spec.error_embed_type,
        dim_observed=4,
        attn_embed_dim=64,
        num_heads=4,
        num_layers=2,
        widening_factor=2,
        dropout=0.0,
        use_missingness_context=spec.use_missingness_context,
        missingness_context_hidden_dim=32,
    )
    model = EncoderHead(encoder).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)

    best = None
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            values = batch["values"].to(device)
            errors = batch["errors"].to(device)
            observed = batch["observed"].to(device)
            y_reg = batch["y_reg"].to(device)
            y_cls = batch["y_cls"].to(device)
            pred_reg, pred_cls = model(values, errors, observed)
            loss = F.mse_loss(pred_reg, y_reg) + F.binary_cross_entropy_with_logits(pred_cls, y_cls)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        metrics = _evaluate_model(model, val_loader, device)
        if best is None or (metrics["reg_r2"] + metrics["cls_acc"]) > (best["reg_r2"] + best["cls_acc"]):
            best = metrics
    return best


def _structural_checks(device: str) -> dict:
    out: dict[str, object] = {}
    batch, num_nodes = 4, 8
    x = torch.randn(batch, num_nodes, 1, device=device)
    node_ids = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch, -1)
    cond = torch.zeros(batch, num_nodes, 1, device=device)
    obs = torch.ones(batch, num_nodes, device=device)
    obs[:, -2:] = 0.0
    errors = torch.zeros(batch, num_nodes, 1, device=device)
    errors[:, 0, 0] = -5.0
    errors[:, 1, 0] = 5.0
    errors[:, 2:, 0] = torch.randn(batch, num_nodes - 2, device=device) * 0.5

    tokenizer_variants = []
    for value_cal in ["none", "scalar_film"]:
        for err_type in ["rff", "mlp_regime"]:
            tok = Tokenizer(
                dim_value=8,
                dim_id=8,
                dim_condition=4,
                attn_embed_dim=32,
                num_nodes=num_nodes,
                value_calibration_type=value_cal,
                dim_error=8,
                use_error_embedding=True,
                error_embed_type=err_type,
                dim_observed=4,
                use_observed_embedding=True,
            ).to(device)
            tokens = tok(x, node_ids, cond, errors=errors, observed_mask=obs)
            tokenizer_variants.append(
                {
                    "value_calibration_type": value_cal,
                    "error_embed_type": err_type,
                    "shape": list(tokens.shape),
                    "finite": bool(torch.isfinite(tokens).all().item()),
                    "token_std": float(tokens.std().item()),
                }
            )
    out["tokenizer_variants"] = tokenizer_variants

    err_rff = ErrorEmbed(embed_dim=8).to(device)
    err_mlp = ErrorEmbedMLP(embed_dim=8).to(device)
    err_test = torch.tensor([[[-5.0], [0.0], [5.0]]], device=device)
    rff_out = err_rff(err_test).detach().cpu().numpy()
    mlp_out = err_mlp(err_test).detach().cpu().numpy()
    out["error_embed_checks"] = {
        "rff_shape": list(rff_out.shape),
        "mlp_shape": list(mlp_out.shape),
        "mlp_pairwise_l2": {
            "perfect_vs_real": float(np.linalg.norm(mlp_out[:, 0] - mlp_out[:, 1])),
            "real_vs_unobs": float(np.linalg.norm(mlp_out[:, 1] - mlp_out[:, 2])),
            "perfect_vs_unobs": float(np.linalg.norm(mlp_out[:, 0] - mlp_out[:, 2])),
        },
    }

    obs_embed = ObservedEmbed(4).to(device)
    obs_tok = obs_embed(torch.tensor([[0, 1]], device=device))
    out["observed_embed_diff_norm"] = float(torch.norm(obs_tok[:, 0] - obs_tok[:, 1]).item())

    calibrator = ValueCalibratorScalar(num_nodes=4).to(device)
    value_emb = torch.ones(1, 4, 6, device=device)
    cal_out = calibrator(value_emb, torch.tensor([[0, 1, 2, 3]], device=device))
    out["value_calibrator_identity_init_max_abs_diff"] = float(torch.max(torch.abs(cal_out - value_emb)).item())

    edge_mask = build_inference_edge_mask(batch_size=batch, num_nodes=num_nodes, observed_mask=obs, device=device)
    aug = Simformer._augment_edge_mask_with_context(edge_mask=edge_mask, observed_mask=obs, num_nodes=num_nodes)
    out["context_mask_checks"] = {
        "base_shape": list(edge_mask.shape),
        "aug_shape": list(aug.shape),
        "context_self_connected": bool(aug[:, -1, -1].all().item()),
        "unobserved_context_links_zero": bool((aug[:, -1, num_nodes - 2:num_nodes] == 0).all().item()),
    }

    return out


def _pass_fail_summary(structural: dict, learning_rows: list[dict]) -> dict:
    failures = []
    for row in structural["tokenizer_variants"]:
        if not row["finite"]:
            failures.append(f"non-finite tokenizer output for {row['value_calibration_type']}+{row['error_embed_type']}")
    if structural["observed_embed_diff_norm"] <= 0:
        failures.append("observed embedding did not separate observed/unobserved states")
    pairwise = structural["error_embed_checks"]["mlp_pairwise_l2"]
    if min(pairwise.values()) <= 0:
        failures.append("ErrorEmbedMLP did not separate sentinel regimes")
    if structural["value_calibrator_identity_init_max_abs_diff"] > 1e-6:
        failures.append("Value calibrator not identity-initialized")
    if not structural["context_mask_checks"]["context_self_connected"]:
        failures.append("Context token self-connection missing")

    for row in learning_rows:
        if row["reg_r2"] < 0.75:
            failures.append(f"{row['variant']} regression sanity failed (R2={row['reg_r2']:.3f})")
        if row["cls_acc"] < 0.85:
            failures.append(f"{row['variant']} classification sanity failed (acc={row['cls_acc']:.3f})")
    return {
        "ok": len(failures) == 0,
        "n_failures": len(failures),
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    train_size = 512 if args.quick else args.train_size
    val_size = 128 if args.quick else args.val_size
    epochs = 6 if args.quick else args.epochs

    input_columns = _make_input_columns()
    started = time.time()

    print(f"Using device: {device}")
    print(f"Input columns ({len(input_columns)}): {input_columns}")
    print("\n--- Structural checks ---")
    structural = _structural_checks(device)

    variants = [
        VariantSpec("scalar_film__mlp_regime__ctx", "scalar_film", "mlp_regime", True),
        VariantSpec("scalar_film__rff__noctx", "scalar_film", "rff", False),
        VariantSpec("none__mlp_regime__noctx", "none", "mlp_regime", False),
        VariantSpec("none__rff__ctx", "none", "rff", True),
    ]
    if args.quick:
        variants = variants[:2]

    print("\n--- Learning sanity checks ---")
    train_loader, val_loader = _make_dataloaders(
        input_columns=input_columns,
        train_size=train_size,
        val_size=val_size,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    learning_rows = []
    for i, spec in enumerate(variants):
        _set_seed(args.seed + 100 + i)
        metrics = _train_variant(
            spec,
            input_columns=input_columns,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=epochs,
        )
        row = {"variant": spec.name, **metrics}
        learning_rows.append(row)
        print(
            f"{spec.name}: reg_r2={row['reg_r2']:.3f} reg_mae={row['reg_mae']:.3f} "
            f"cls_acc={row['cls_acc']:.3f} cls_brier={row['cls_brier']:.3f}"
        )

    summary = _pass_fail_summary(structural, learning_rows)
    elapsed = time.time() - started
    payload = {
        "device": device,
        "input_columns": input_columns,
        "train_size": train_size,
        "val_size": val_size,
        "epochs": epochs,
        "structural_checks": structural,
        "learning_checks": learning_rows,
        "summary": summary,
        "elapsed_sec": elapsed,
    }

    summary_json = os.path.join(args.output_dir, "transformer_stack_validation_summary.json")
    learning_csv = os.path.join(args.output_dir, "transformer_stack_learning_checks.csv")
    with open(summary_json, "w") as f:
        json.dump(payload, f, indent=2)
    pd = __import__("pandas")
    pd.DataFrame(learning_rows).to_csv(learning_csv, index=False)

    print("\n--- Summary ---")
    print(f"OK: {summary['ok']}")
    if summary["failures"]:
        for msg in summary["failures"]:
            print(f"  FAIL: {msg}")
    print(f"Saved summary:  {summary_json}")
    print(f"Saved learning: {learning_csv}")


if __name__ == "__main__":
    main()
