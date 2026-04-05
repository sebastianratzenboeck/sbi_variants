#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from data import build_sbi_arrays, load_cache_arrays, load_indices
from eval_subgroups import subgroup_masks
from inference_utils import NormStats
from train_sbi_nre import _build_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate an NRE model as a held-out joint-vs-product classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--cache-path", type=str, required=True)
    p.add_argument("--eval-split", action="append", required=True, help="name=/path/to/indices.npy")
    p.add_argument("--max-stars", type=int, default=8192)
    p.add_argument("--sample-mode", choices=("random", "head"), default="random")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--young-logage-threshold", type=float, default=7.8)
    p.add_argument("--min-group-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def _parse_eval_split(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"Invalid --eval-split '{spec}', expected name=/path/to/file.npy")
    name, path = spec.split("=", 1)
    return name.strip(), path.strip()


def _subset_rows(idx: np.ndarray, *, max_stars: int | None, sample_mode: str, seed: int) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    if max_stars is None or max_stars <= 0 or idx.size <= max_stars:
        return np.sort(idx)
    rng = np.random.default_rng(seed)
    if sample_mode == "head":
        return np.sort(idx[:max_stars])
    return np.sort(rng.choice(idx, size=max_stars, replace=False).astype(np.int64))


def _batch_logits(model, arrays, *, batch_size: int, device: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    values = torch.as_tensor(arrays.inputs, dtype=torch.float32, device=device)
    errors = torch.as_tensor(arrays.input_errors, dtype=torch.float32, device=device)
    observed = torch.as_tensor(arrays.input_observed, dtype=torch.float32, device=device)
    theta = torch.as_tensor(arrays.theta, dtype=torch.float32, device=device)

    rng = np.random.default_rng(seed)
    neg_perm = torch.as_tensor(rng.permutation(theta.shape[0]), dtype=torch.long, device=device)
    theta_neg = theta[neg_perm]

    pos_chunks = []
    neg_chunks = []
    for start in range(0, theta.shape[0], batch_size):
        end = min(start + batch_size, theta.shape[0])
        with torch.no_grad():
            ctx = model.encode_context(values[start:end], errors[start:end], observed[start:end])
            pos = model.logits_from_context(theta[start:end], ctx)
            neg = model.logits_from_context(theta_neg[start:end], ctx)
        pos_chunks.append(pos.detach().cpu().numpy().astype(np.float32))
        neg_chunks.append(neg.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(pos_chunks), np.concatenate(neg_chunks)


def _summary_from_logits(pos_logits: np.ndarray, neg_logits: np.ndarray) -> dict[str, float]:
    pos_logits = np.asarray(pos_logits, dtype=np.float32).reshape(-1)
    neg_logits = np.asarray(neg_logits, dtype=np.float32).reshape(-1)

    pos_t = torch.as_tensor(pos_logits)
    neg_t = torch.as_tensor(neg_logits)
    bce_pos = float(F.binary_cross_entropy_with_logits(pos_t, torch.ones_like(pos_t)).item())
    bce_neg = float(F.binary_cross_entropy_with_logits(neg_t, torch.zeros_like(neg_t)).item())
    mean_bce = 0.5 * (bce_pos + bce_neg)

    pos_acc = float((pos_logits > 0.0).mean())
    neg_acc = float((neg_logits < 0.0).mean())
    bal_acc = 0.5 * (pos_acc + neg_acc)

    y_true = np.concatenate([np.ones_like(pos_logits), np.zeros_like(neg_logits)])
    y_score = np.concatenate([pos_logits, neg_logits])
    auc = float(roc_auc_score(y_true, y_score))

    return {
        "n_examples": int(pos_logits.size),
        "bce_pos": bce_pos,
        "bce_neg": bce_neg,
        "bce_mean": mean_bce,
        "pos_acc": pos_acc,
        "neg_acc": neg_acc,
        "balanced_acc": bal_acc,
        "auc": auc,
        "pos_logit_mean": float(pos_logits.mean()),
        "neg_logit_mean": float(neg_logits.mean()),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = args.output_dir or os.path.join(args.model_dir, f"eval_{args.run_name}")
    os.makedirs(output_dir, exist_ok=True)

    config_path = os.path.join(args.model_dir, f"ratio_config_{args.run_name}.json")
    ckpt_path = os.path.join(args.model_dir, f"best_ratio_model_{args.run_name}.pt")
    meta_path = os.path.join(args.model_dir, f"ratio_norm_meta_{args.run_name}.npz")

    with open(config_path) as f:
        config = json.load(f)
    norm_stats = NormStats(meta_path)

    model_args = SimpleNamespace(**config)
    input_columns = [str(c) for c in config["input_columns"]]
    theta_columns = [str(c) for c in config["theta_columns"]]
    model = _build_model(model_args, input_columns=input_columns, theta_dim=len(theta_columns)).to(device)
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    cache = load_cache_arrays(args.cache_path)
    bundle = {}
    for i, spec in enumerate(args.eval_split):
        split_name, split_path = _parse_eval_split(spec)
        idx = load_indices(split_path)
        if idx is None or idx.size == 0:
            raise ValueError(f"Split '{split_name}' has no rows: {split_path}")
        idx = _subset_rows(idx, max_stars=args.max_stars, sample_mode=args.sample_mode, seed=args.seed + i)

        arrays = build_sbi_arrays(
            cache,
            row_indices=idx,
            input_columns=input_columns,
            theta_columns=theta_columns,
            use_colors=bool(config.get("use_colors", False)),
        )
        pos_logits, neg_logits = _batch_logits(model, arrays, batch_size=args.batch_size, device=device, seed=args.seed + i)
        summary = _summary_from_logits(pos_logits, neg_logits)

        masks = subgroup_masks(
            values_norm=cache.values_norm[idx],
            errors_norm=cache.errors_norm[idx],
            observed_mask=cache.observed_mask[idx],
            norm_stats=norm_stats,
            young_logage_threshold=args.young_logage_threshold,
        )
        subgroup_rows = []
        for group_name, mask in masks.items():
            if int(mask.sum()) < args.min_group_size:
                continue
            row = {"group": group_name, "n_stars": int(mask.sum())}
            row.update(_summary_from_logits(pos_logits[mask], neg_logits[mask]))
            subgroup_rows.append(row)

        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        pd.DataFrame(subgroup_rows).sort_values("group").to_csv(os.path.join(split_dir, "subgroup_summary.csv"), index=False)
        with open(os.path.join(split_dir, "summary.json"), "w") as f:
            json.dump(
                {
                    "split_name": split_name,
                    "index_file": split_path,
                    "model_dir": args.model_dir,
                    "run_name": args.run_name,
                    "device": device,
                    "elapsed_sec": None,
                    **summary,
                },
                f,
                indent=2,
            )
        np.savez_compressed(
            os.path.join(split_dir, "logits.npz"),
            selected_indices=idx.astype(np.int64),
            pos_logits=pos_logits,
            neg_logits=neg_logits,
        )
        bundle[split_name] = summary

    with open(os.path.join(output_dir, "nre_eval_bundle.json"), "w") as f:
        json.dump(bundle, f, indent=2)


if __name__ == "__main__":
    main()
