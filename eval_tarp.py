#!/usr/bin/env python
"""TARP-style calibration evaluation via random projection ranks."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from sample_mock_galaxy import load_model, sample_posterior
from eval_utils import (
    DEFAULT_TARGET_COLS,
    auto_device,
    central_rank_coverage,
    column_indices,
    ensure_dir,
    ks_uniform,
    load_cache_arrays,
    maybe_denormalize,
    parse_float_list,
    parse_str_list,
    projection_ranks,
    to_input_tensors,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate posterior calibration with TARP-style projection ranks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--run-name", type=str, default="default")
    p.add_argument("--cache-path", type=str, required=True,
                   help="Path to build_arrays_cache.npz")
    p.add_argument("--index-file", type=str, default=None,
                   help="Optional .npy row indices (e.g. test_indices.npy)")
    p.add_argument("--max-stars", type=int, default=512)
    p.add_argument("--sample-mode", choices=("random", "head"), default="random")
    p.add_argument("--num-samples", type=int, default=512,
                   help="Posterior draws per star")
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--target-cols",
        type=str,
        default=",".join(DEFAULT_TARGET_COLS),
        help="Comma-separated columns to test",
    )
    p.add_argument("--num-projections", type=int, default=256)
    p.add_argument(
        "--alpha-grid",
        type=str,
        default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated central-coverage levels for rank coverage curve",
    )
    p.add_argument("--denorm", action="store_true", default=False,
                   help="Run TARP in physical units (default: normalized units)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--tag", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = auto_device(args.device)
    target_cols = parse_str_list(args.target_cols)
    alpha_grid = parse_float_list(args.alpha_grid)
    if any(a <= 0.0 or a >= 1.0 for a in alpha_grid):
        raise ValueError("All alpha-grid values must be in (0,1).")

    output_dir = args.output_dir or os.path.join(args.model_dir, "eval")
    ensure_dir(output_dir)
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")

    print(f"Using device: {device}")
    print("\n--- Loading model ---")
    model, norm_stats = load_model(args.model_dir, run_name=args.run_name, device=device)
    target_idx = column_indices(target_cols, columns=norm_stats.columns)

    print("\n--- Loading cache rows ---")
    values_norm, errors_norm, observed_mask, selected = load_cache_arrays(
        cache_path=args.cache_path,
        index_file=args.index_file,
        max_stars=args.max_stars,
        sample_mode=args.sample_mode,
        seed=args.seed,
        expected_columns=norm_stats.columns,
    )
    cv, cm, om, er = to_input_tensors(
        values_norm,
        errors_norm,
        observed_mask,
        columns=norm_stats.columns,
        device="cpu",
    )
    print(f"  Stars selected: {len(selected):,}")

    print("\n--- Sampling ---")
    t0 = time.time()
    samples_norm = sample_posterior(
        model=model,
        condition_values=cv,
        condition_mask=cm,
        observed_mask=om,
        errors=er,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        steps=args.steps,
        device=device,
    ).cpu().numpy()
    elapsed = time.time() - t0
    print(f"  Sampling done in {elapsed:.1f}s")

    truth_norm = values_norm
    samples_t = samples_norm[:, :, target_idx]
    truth_t = truth_norm[:, target_idx]
    if args.denorm:
        samples_eval = maybe_denormalize(norm_stats, samples_t, target_idx, denorm=True)
        truth_eval = maybe_denormalize(norm_stats, truth_t, target_idx, denorm=True)
        units = "physical"
    else:
        samples_eval = samples_t
        truth_eval = truth_t
        units = "normalized"

    print("\n--- TARP-style rank test ---")
    u_all = projection_ranks(
        samples=samples_eval,
        truth=truth_eval,
        num_projections=args.num_projections,
        seed=args.seed,
    )  # (K, N)
    u_flat = u_all.reshape(-1)
    ks = ks_uniform(u_flat)

    rows = []
    for alpha in alpha_grid:
        empirical = central_rank_coverage(u_flat, alpha)
        per_proj = np.array([central_rank_coverage(u_all[k], alpha) for k in range(u_all.shape[0])])
        rows.append(
            {
                "alpha": float(alpha),
                "empirical_coverage": float(empirical),
                "calibration_error": float(empirical - alpha),
                "per_projection_mean": float(per_proj.mean()),
                "per_projection_std": float(per_proj.std(ddof=0)),
            }
        )
    curve_df = pd.DataFrame(rows).sort_values("alpha")
    ace = float(np.mean(np.abs(curve_df["calibration_error"].values)))
    mce = float(np.max(np.abs(curve_df["calibration_error"].values)))

    summary = {
        "model_dir": args.model_dir,
        "run_name": args.run_name,
        "cache_path": args.cache_path,
        "index_file": args.index_file,
        "max_stars": args.max_stars,
        "num_samples": args.num_samples,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "num_projections": args.num_projections,
        "target_cols": target_cols,
        "units": units,
        "ks_uniform": ks,
        "rank_mean": float(u_flat.mean()),
        "rank_std": float(u_flat.std(ddof=0)),
        "ace": ace,
        "mce": mce,
    }

    curve_csv = os.path.join(output_dir, f"tarp_curve_{tag}.csv")
    summary_json = os.path.join(output_dir, f"tarp_summary_{tag}.json")
    selected_npy = os.path.join(output_dir, f"tarp_selected_indices_{tag}.npy")
    curve_df.to_csv(curve_csv, index=False)
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)
    np.save(selected_npy, selected)

    print(curve_df.to_string(index=False))
    print("\n--- TARP summary ---")
    print(f"  KS(uniform) = {ks:.4f}")
    print(f"  ACE         = {ace:.4f}")
    print(f"  MCE         = {mce:.4f}")
    print(f"Saved curve CSV: {curve_csv}")
    print(f"Saved summary:   {summary_json}")


if __name__ == "__main__":
    main()
