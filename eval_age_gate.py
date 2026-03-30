#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from age_gate_models import AgeGateClassifier, apply_temperature
from age_regimes import DEFAULT_AGE_BIN_EDGES, logage_to_regime_index, regime_names
from data import build_sbi_arrays, load_cache_arrays, load_indices, parse_column_csv, DEFAULT_INPUT_COLS
from encoder import ObservationEncoder
from inference_utils import NormStats


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


def _parse_edges(raw: str | list[float]) -> list[float]:
    if isinstance(raw, str):
        vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    else:
        vals = [float(x) for x in raw]
    if not vals:
        raise ValueError("age-bin-edges must contain at least one edge")
    return vals


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate an age-gate classifier on one or more fixed index sets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cache-path", type=str, required=True)
    p.add_argument("--norm-meta-path", type=str, required=True)
    p.add_argument("--gate-config", type=str, required=True)
    p.add_argument("--gate-checkpoint", type=str, required=True)
    p.add_argument("--temperature-json", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--eval-split", action="append", default=[], help="name=/path/to/indices.npy")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--ece-bins", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    return p


def _build_encoder(config: dict, input_columns: list[str]) -> ObservationEncoder:
    return ObservationEncoder(
        input_columns=input_columns,
        dim_value=int(config["dim_value"]),
        dim_id=int(config["dim_id"]),
        value_calibration_type=str(config["value_calibration_type"]),
        dim_error=int(config["dim_error"]),
        error_embed_type=str(config["error_embed_type"]),
        dim_observed=int(config["dim_observed"]),
        attn_embed_dim=int(config["attn_embed_dim"]),
        num_heads=int(config["num_heads"]),
        num_layers=int(config["num_layers"]),
        widening_factor=int(config["widening_factor"]),
        dropout=float(config["dropout"]),
        use_missingness_context=bool(config["use_missingness_context"]),
        missingness_context_hidden_dim=int(config["missingness_context_hidden_dim"]),
    )


def _make_labels(cache, row_indices: np.ndarray, stats: NormStats, *, edges: list[float]) -> np.ndarray:
    logage_idx = cache.columns.index("logAge")
    logage_norm = cache.values_norm[row_indices][:, [logage_idx]].astype(np.float32)
    logage_phys = stats.denormalize_numpy(logage_norm, [logage_idx]).reshape(-1)
    return logage_to_regime_index(logage_phys, edges=edges).astype(np.int64)


def _build_dataset(cache, *, row_indices: np.ndarray, input_columns: list[str], use_colors: bool):
    arr = build_sbi_arrays(
        cache,
        row_indices=row_indices,
        input_columns=input_columns,
        theta_columns=["logAge"],
        use_colors=use_colors,
    )
    labels = None
    if use_colors and arr.color_names is not None:
        input_columns = list(input_columns) + list(arr.color_names)
    ds = AgeGateDataset(
        inputs=arr.inputs,
        errors=arr.input_errors,
        observed=arr.input_observed,
        labels=np.zeros(arr.inputs.shape[0], dtype=np.int64) if labels is None else labels,
    )
    return ds, input_columns


def _parse_eval_split(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"invalid --eval-split '{spec}', expected name=/path/to/file.npy")
    name, path = spec.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"invalid --eval-split '{spec}', expected non-empty name and path")
    return name, path


def _ece(probs: np.ndarray, labels: np.ndarray, *, n_bins: int) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo = edges[i]
        hi = edges[i + 1]
        if i == n_bins - 1:
            mask = (conf >= lo) & (conf <= hi)
        else:
            mask = (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        ece += float(mask.mean()) * abs(acc - avg_conf)
    return float(ece)


def _multiclass_brier(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    onehot = np.zeros((labels.shape[0], num_classes), dtype=np.float64)
    onehot[np.arange(labels.shape[0]), labels] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def _confusion_matrix(labels: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for y, p in zip(labels.tolist(), pred.tolist()):
        cm[int(y), int(p)] += 1
    return cm


def _per_class_metrics(cm: np.ndarray, names: list[str]) -> list[dict[str, float | int | str]]:
    out: list[dict[str, float | int | str]] = []
    for i, name in enumerate(names):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        support = int(cm[i, :].sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        out.append(
            {
                "class_index": i,
                "class_name": name,
                "support": support,
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
        )
    return out


def _predicted_class_distribution(pred: np.ndarray, names: list[str]) -> dict[str, int]:
    return {name: int(np.sum(pred == i)) for i, name in enumerate(names)}


def _load_temperature(path: str) -> float:
    with open(path) as f:
        payload = json.load(f)
    return float(payload["temperature"])


def _evaluate_split(
    model: AgeGateClassifier,
    loader: DataLoader,
    *,
    device: str,
    temperature: float,
    regime_names_list: list[str],
    ece_bins: int,
) -> tuple[dict[str, object], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["inputs"].to(device, non_blocking=True)
            e = batch["errors"].to(device, non_blocking=True)
            o = batch["observed"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            logits = model(x, e, o)
            logits = apply_temperature(logits, temperature)
            probs = torch.softmax(logits, dim=-1)
            logits_all.append(logits.detach().cpu().numpy())
            probs_all.append(probs.detach().cpu().numpy())
            labels_all.append(y.detach().cpu().numpy())

    logits_np = np.concatenate(logits_all, axis=0)
    probs_np = np.concatenate(probs_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0).astype(np.int64)
    pred_np = probs_np.argmax(axis=1).astype(np.int64)

    cm = _confusion_matrix(labels_np, pred_np, len(regime_names_list))
    per_class = _per_class_metrics(cm, regime_names_list)
    acc = float(np.mean(pred_np == labels_np))
    balanced_acc = float(np.mean([row["recall"] for row in per_class]))
    conf = probs_np.max(axis=1)
    correct = (pred_np == labels_np).astype(np.float64)
    nll = float(-np.mean(np.log(np.clip(probs_np[np.arange(labels_np.shape[0]), labels_np], 1e-12, 1.0))))
    summary = {
        "n_examples": int(labels_np.shape[0]),
        "accuracy": acc,
        "balanced_accuracy": balanced_acc,
        "nll": nll,
        "ece": _ece(probs_np, labels_np, n_bins=ece_bins),
        "brier_multiclass": _multiclass_brier(probs_np, labels_np, len(regime_names_list)),
        "mean_confidence": float(conf.mean()),
        "mean_correct_confidence": float(conf[correct == 1].mean()) if np.any(correct == 1) else None,
        "mean_incorrect_confidence": float(conf[correct == 0].mean()) if np.any(correct == 0) else None,
        "overconfidence_gap": float(conf.mean() - acc),
        "high_confidence_frac_90": float(np.mean(conf >= 0.9)),
        "high_confidence_accuracy_90": float(correct[conf >= 0.9].mean()) if np.any(conf >= 0.9) else None,
        "predicted_class_counts": _predicted_class_distribution(pred_np, regime_names_list),
        "true_class_counts": _predicted_class_distribution(labels_np, regime_names_list),
        "confusion_matrix": cm.tolist(),
        "per_class": per_class,
    }

    young_idx = regime_names_list.index("young") if "young" in regime_names_list else 0
    summary["young_precision"] = next(row["precision"] for row in per_class if row["class_index"] == young_idx)
    summary["young_recall"] = next(row["recall"] for row in per_class if row["class_index"] == young_idx)
    summary["young_f1"] = next(row["f1"] for row in per_class if row["class_index"] == young_idx)
    summary["young_true_mean_prob"] = float(probs_np[labels_np == young_idx, young_idx].mean()) if np.any(labels_np == young_idx) else None
    summary["young_false_positive_mean_prob"] = float(probs_np[labels_np != young_idx, young_idx].mean()) if np.any(labels_np != young_idx) else None
    return summary, probs_np, labels_np, pred_np


def main() -> None:
    args = _build_parser().parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.gate_config) as f:
        config = json.load(f)
    temperature = _load_temperature(args.temperature_json)
    edges = _parse_edges(config.get("age_bin_edges", DEFAULT_AGE_BIN_EDGES))
    reg_names = [str(x) for x in config.get("regime_names", regime_names(edges))]
    encoder_input_columns = [
        str(x)
        for x in config.get(
            "input_columns_with_colors",
            config.get("input_columns", DEFAULT_INPUT_COLS),
        )
    ]
    base_input_columns = [str(x) for x in config.get("input_columns", DEFAULT_INPUT_COLS)]
    use_colors = bool(config.get("use_colors", True))

    cache = load_cache_arrays(args.cache_path)
    stats = NormStats(args.norm_meta_path)

    encoder = _build_encoder(config, encoder_input_columns)
    model = AgeGateClassifier(
        encoder=encoder,
        num_regimes=len(reg_names),
        hidden_dim=int(config["gate_hidden_dim"]),
        dropout=float(config["gate_dropout"]),
    ).to(device)
    state = torch.load(args.gate_checkpoint, map_location=device)
    model.load_state_dict(state)

    summaries: dict[str, dict[str, object]] = {}
    for spec in args.eval_split:
        split_name, split_path = _parse_eval_split(spec)
        row_indices = load_indices(split_path)
        if row_indices is None or row_indices.size == 0:
            raise ValueError(f"split '{split_name}' has no rows: {split_path}")
        arr = build_sbi_arrays(
            cache,
            row_indices=row_indices,
            input_columns=base_input_columns,
            theta_columns=["logAge"],
            use_colors=use_colors,
        )
        labels = _make_labels(cache, row_indices, stats, edges=edges)
        ds = AgeGateDataset(
            inputs=arr.inputs,
            errors=arr.input_errors,
            observed=arr.input_observed,
            labels=labels,
        )
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=(device != "cpu" and torch.cuda.is_available()),
        )
        summary, probs_np, labels_np, pred_np = _evaluate_split(
            model,
            loader,
            device=device,
            temperature=temperature,
            regime_names_list=reg_names,
            ece_bins=args.ece_bins,
        )
        summary["index_file"] = split_path
        summaries[split_name] = summary

        np.savez_compressed(
            out_dir / f"age_gate_predictions_{split_name}.npz",
            row_indices=row_indices.astype(np.int64),
            labels=labels_np.astype(np.int64),
            pred=pred_np.astype(np.int64),
            probs=probs_np.astype(np.float32),
            regime_names=np.asarray(reg_names, dtype=object),
        )
        with open(out_dir / f"age_gate_summary_{split_name}.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"{split_name}: n={summary['n_examples']:,} acc={summary['accuracy']:.4f} "
            f"bal_acc={summary['balanced_accuracy']:.4f} young_recall={summary['young_recall']:.4f} "
            f"ece={summary['ece']:.4f} overconfidence_gap={summary['overconfidence_gap']:.4f}"
        )

    bundle = {
        "gate_checkpoint": args.gate_checkpoint,
        "gate_config": args.gate_config,
        "temperature_json": args.temperature_json,
        "temperature": temperature,
        "splits": summaries,
    }
    with open(out_dir / "age_gate_eval_bundle.json", "w") as f:
        json.dump(bundle, f, indent=2)
    print(f"Saved evaluation bundle to {out_dir / 'age_gate_eval_bundle.json'}")


if __name__ == "__main__":
    main()
