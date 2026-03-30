#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from age_regimes import (
    DEFAULT_AGE_BIN_EDGES,
    regime_names,
    logage_to_regime_index,
    summarize_regime_counts,
)
from data import load_indices
from inference_utils import NormStats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build age-regime train/val index splits and a balanced evaluation subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cache-path", type=str, required=True)
    p.add_argument("--test-index-file", type=str, required=True)
    p.add_argument("--norm-meta-path", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--age-bin-edges", type=str, default=",".join(str(x) for x in DEFAULT_AGE_BIN_EDGES))
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--balanced-eval-per-regime", type=int, default=100000)
    return p.parse_args()


def _parse_edges(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("age-bin-edges must contain at least one edge")
    return vals


def _select_balanced_subset(
    idx: np.ndarray,
    regime_idx: np.ndarray,
    *,
    n_per_regime: int,
    n_regimes: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    chunks = []
    for k in range(n_regimes):
        rows = idx[regime_idx == k]
        if rows.size == 0:
            continue
        take = min(int(n_per_regime), int(rows.size))
        picked = rng.choice(rows, size=take, replace=False).astype(np.int64)
        picked.sort()
        chunks.append(picked)
    if not chunks:
        raise ValueError("No rows available for balanced evaluation subset")
    out = np.concatenate(chunks).astype(np.int64)
    out.sort()
    return out


def main() -> None:
    args = parse_args()
    edges = _parse_edges(args.age_bin_edges)
    names = regime_names(edges)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = NormStats(args.norm_meta_path)
    d = np.load(args.cache_path, allow_pickle=True)
    test_idx = load_indices(args.test_index_file)
    if test_idx is None or test_idx.size == 0:
        raise ValueError("test index file is empty or missing")

    columns = [str(c) for c in d["columns"].tolist()]
    if "logAge" not in columns:
        raise ValueError("cache is missing logAge column")
    logage_idx = columns.index("logAge")

    test_logage_norm = d["values_norm"][test_idx][:, [logage_idx]].astype(np.float32)
    test_logage = stats.denormalize_numpy(test_logage_norm, [logage_idx]).reshape(-1)
    test_regime_idx = logage_to_regime_index(test_logage, edges=edges)

    summary: dict[str, object] = {
        "cache_path": args.cache_path,
        "test_index_file": args.test_index_file,
        "norm_meta_path": args.norm_meta_path,
        "age_bin_edges": list(edges),
        "regime_names": list(names),
        "test_counts": summarize_regime_counts(test_regime_idx, edges=edges, names=names),
    }

    balanced_eval_idx = _select_balanced_subset(
        test_idx,
        test_regime_idx,
        n_per_regime=args.balanced_eval_per_regime,
        n_regimes=len(names),
        seed=args.seed,
    )
    balanced_eval_regime_idx = logage_to_regime_index(
        stats.denormalize_numpy(
            d["values_norm"][balanced_eval_idx][:, [logage_idx]].astype(np.float32),
            [logage_idx],
        ).reshape(-1),
        edges=edges,
    )
    balanced_eval_path = out_dir / "eval_indices_balanced_age_300k.npy"
    np.save(balanced_eval_path, balanced_eval_idx.astype(np.int64))
    summary["balanced_eval_index_file"] = str(balanced_eval_path)
    summary["balanced_eval_counts"] = summarize_regime_counts(
        balanced_eval_regime_idx,
        edges=edges,
        names=names,
    )

    n_total = d["values_norm"].shape[0]
    all_rows = np.arange(n_total, dtype=np.int64)
    trainval_mask = np.ones(n_total, dtype=bool)
    trainval_mask[test_idx] = False
    trainval_rows = all_rows[trainval_mask]

    trainval_logage_norm = d["values_norm"][trainval_rows][:, [logage_idx]].astype(np.float32)
    trainval_logage = stats.denormalize_numpy(trainval_logage_norm, [logage_idx]).reshape(-1)
    trainval_regime_idx = logage_to_regime_index(trainval_logage, edges=edges)
    summary["trainval_counts"] = summarize_regime_counts(trainval_regime_idx, edges=edges, names=names)

    regime_files: dict[str, dict[str, str]] = {}
    for k, name in enumerate(names):
        rows = trainval_rows[trainval_regime_idx == k]
        if rows.size < 2:
            raise ValueError(f"regime '{name}' has too few train/val rows: {rows.size}")
        train_rows, val_rows = train_test_split(
            rows,
            test_size=args.val_split,
            random_state=args.seed,
        )
        train_rows = np.sort(train_rows.astype(np.int64))
        val_rows = np.sort(val_rows.astype(np.int64))

        train_path = out_dir / f"train_indices_{name}.npy"
        val_path = out_dir / f"val_indices_{name}.npy"
        np.save(train_path, train_rows)
        np.save(val_path, val_rows)
        regime_files[name] = {
            "train": str(train_path),
            "val": str(val_path),
            "train_count": int(train_rows.size),
            "val_count": int(val_rows.size),
        }

    summary["regime_files"] = regime_files

    combined_train = np.sort(
        np.concatenate([np.load(regime_files[name]["train"]) for name in names]).astype(np.int64)
    )
    combined_val = np.sort(
        np.concatenate([np.load(regime_files[name]["val"]) for name in names]).astype(np.int64)
    )
    combined_train_path = out_dir / "train_indices_all.npy"
    combined_val_path = out_dir / "val_indices_all.npy"
    np.save(combined_train_path, combined_train)
    np.save(combined_val_path, combined_val)
    summary["combined_files"] = {
        "train": str(combined_train_path),
        "val": str(combined_val_path),
        "train_count": int(combined_train.size),
        "val_count": int(combined_val.size),
    }

    summary_path = out_dir / "age_regime_split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("Saved age-regime splits.")
    print(f"Summary: {summary_path}")
    print(f"Balanced eval indices: {balanced_eval_path}")
    for name in names:
        info = regime_files[name]
        print(f"{name}: train={info['train_count']:,}, val={info['val_count']:,}")
    print("Balanced eval counts:")
    for name, count in summary["balanced_eval_counts"].items():
        print(f"  {name}: {count:,}")


if __name__ == "__main__":
    main()
