#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

from data import load_cache_arrays, load_indices
from eval_subgroups import subgroup_masks, summarize_group
from eval_utils import (
    central_rank_coverage,
    ensure_dir,
    interval_metrics,
    ks_uniform,
    maybe_denormalize,
    projection_ranks,
)
from inference_utils import NormStats
from vanilla_sbi_utils import (
    build_zero_imputed_npe_arrays,
    configure_sbi_env,
    load_pickle,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a vanilla sbi NPE baseline on fixed cached subsets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--cache-path", type=str, required=True)
    p.add_argument("--eval-split", action="append", required=True,
                   help="name=/path/to/indices.npy")
    p.add_argument("--max-stars", type=int, default=512)
    p.add_argument("--sample-mode", choices=("random", "head"), default="random")
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--levels", type=str, default="0.5,0.8,0.9,0.95")
    p.add_argument("--coverage-level", type=float, default=0.9)
    p.add_argument("--num-projections", type=int, default=128)
    p.add_argument("--young-logage-threshold", type=float, default=7.8)
    p.add_argument("--min-group-size", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def _parse_levels(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one coverage level")
    return vals


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


def _sample_posterior_rows(
    posterior,
    x_np: np.ndarray,
    *,
    num_samples: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    n = x_np.shape[0]
    out = []
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_x = torch.as_tensor(x_np[start:end], dtype=torch.float32, device=device)
        samples_batch = []
        for i in range(batch_x.shape[0]):
            with torch.no_grad():
                s = posterior.sample((num_samples,), x=batch_x[i]).detach().cpu().numpy().astype(np.float32)
            samples_batch.append(s)
        out.append(np.stack(samples_batch, axis=0))
    return np.concatenate(out, axis=0)


def main() -> None:
    configure_sbi_env()
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = args.model_dir
    run_name = args.run_name
    output_dir = args.output_dir or os.path.join(model_dir, "eval")
    ensure_dir(output_dir)

    posterior_path = os.path.join(model_dir, f"posterior_{run_name}.pkl")
    config_path = os.path.join(model_dir, f"vanilla_sbi_config_{run_name}.json")
    meta_path = os.path.join(model_dir, f"vanilla_sbi_norm_meta_{run_name}.npz")

    posterior = load_pickle(posterior_path)
    if hasattr(posterior, "to"):
        posterior = posterior.to(device)
    with open(config_path) as f:
        config = json.load(f)
    norm_stats = NormStats(meta_path)

    input_columns = [str(c) for c in config["input_columns"]]
    theta_columns = [str(c) for c in config["theta_columns"]]
    levels = _parse_levels(args.levels)
    theta_idx_full = [norm_stats.column_index(c) for c in theta_columns]
    color_norm_stats = None
    if bool(config.get("use_colors", False)) and len(config.get("color_names", [])) > 0:
        color_norm_stats = (
            np.asarray(norm_stats.color_means, dtype=np.float32),
            np.asarray(norm_stats.color_stds, dtype=np.float32),
        )

    cache = load_cache_arrays(args.cache_path)
    bundle = {}
    for i, spec in enumerate(args.eval_split):
        split_name, split_path = _parse_eval_split(spec)
        idx = load_indices(split_path)
        if idx is None or idx.size == 0:
            raise ValueError(f"Split '{split_name}' has no rows: {split_path}")
        idx = _subset_rows(idx, max_stars=args.max_stars, sample_mode=args.sample_mode, seed=args.seed + i)
        arrays, x_np, truth_norm = build_zero_imputed_npe_arrays(
            cache_path=args.cache_path,
            row_indices=idx,
            input_columns=input_columns,
            theta_columns=theta_columns,
            use_colors=bool(config.get("use_colors", False)),
            color_norm_stats=color_norm_stats,
        )

        t0 = time.time()
        samples_norm = _sample_posterior_rows(
            posterior,
            x_np,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            device=device,
        )
        elapsed = time.time() - t0

        truth_phys = maybe_denormalize(norm_stats, truth_norm, theta_idx_full, denorm=True)
        samples_phys = maybe_denormalize(norm_stats, samples_norm, theta_idx_full, denorm=True)

        metrics_df = interval_metrics(samples_phys, truth_phys, theta_columns, levels)
        summary_by_level = (
            metrics_df.groupby("level", as_index=False)
            .agg(
                mean_coverage=("coverage", "mean"),
                mean_abs_calibration_error=("calibration_error", lambda x: float(np.mean(np.abs(x)))),
                mean_width=("mean_width", "mean"),
            )
        )
        overall_ace = float(np.mean(np.abs(metrics_df["calibration_error"].values)))

        u_all = projection_ranks(
            samples=samples_norm,
            truth=truth_norm,
            num_projections=args.num_projections,
            seed=args.seed + i,
        )
        u_flat = u_all.reshape(-1)
        tarp_rows = []
        for alpha in levels:
            empirical = central_rank_coverage(u_flat, alpha)
            tarp_rows.append(
                {
                    "alpha": float(alpha),
                    "empirical_coverage": float(empirical),
                    "calibration_error": float(empirical - alpha),
                }
            )
        tarp_curve_df = pd.DataFrame(tarp_rows).sort_values("alpha")
        tarp_ace = float(np.mean(np.abs(tarp_curve_df["calibration_error"].values)))
        tarp_mce = float(np.max(np.abs(tarp_curve_df["calibration_error"].values)))
        tarp_ks = float(ks_uniform(u_flat))

        masks = subgroup_masks(
            values_norm=cache.values_norm[idx],
            errors_norm=cache.errors_norm[idx],
            observed_mask=cache.observed_mask[idx],
            norm_stats=norm_stats,
            young_logage_threshold=args.young_logage_threshold,
        )
        summary_rows = []
        per_param_rows = []
        for j, (name, mask) in enumerate(masks.items()):
            if int(mask.sum()) < args.min_group_size:
                continue
            row, per_param = summarize_group(
                name=name,
                mask=mask,
                samples_phys=samples_phys,
                truth_phys=truth_phys,
                samples_norm=samples_norm,
                truth_norm=truth_norm,
                target_cols=theta_columns,
                coverage_level=args.coverage_level,
                num_projections=args.num_projections,
                seed=args.seed + i + j,
            )
            summary_rows.append(row)
            per_param_rows.extend(per_param)

        split_dir = os.path.join(output_dir, split_name)
        ensure_dir(split_dir)
        metrics_df.to_csv(os.path.join(split_dir, "coverage_detail.csv"), index=False)
        summary_by_level.to_csv(os.path.join(split_dir, "coverage_by_level.csv"), index=False)
        tarp_curve_df.to_csv(os.path.join(split_dir, "tarp_curve.csv"), index=False)
        pd.DataFrame(summary_rows).sort_values("group").to_csv(os.path.join(split_dir, "subgroup_summary.csv"), index=False)
        pd.DataFrame(per_param_rows).sort_values(["group", "column"]).to_csv(os.path.join(split_dir, "subgroup_per_param.csv"), index=False)
        np.save(os.path.join(split_dir, "selected_indices.npy"), idx)
        with open(os.path.join(split_dir, "summary.json"), "w") as f:
            json.dump(
                {
                    "split_name": split_name,
                    "index_file": split_path,
                    "n_examples": int(idx.size),
                    "elapsed_sec": float(elapsed),
                    "overall_ace": overall_ace,
                    "tarp_ks_uniform": tarp_ks,
                    "tarp_ace": tarp_ace,
                    "tarp_mce": tarp_mce,
                },
                f,
                indent=2,
            )
        bundle[split_name] = {
            "n_examples": int(idx.size),
            "overall_ace": overall_ace,
            "tarp_ks_uniform": tarp_ks,
            "tarp_ace": tarp_ace,
            "tarp_mce": tarp_mce,
            "summary_dir": split_dir,
        }
        print(f"{split_name}: n={idx.size:,} ACE={overall_ace:.4f} TARP_ACE={tarp_ace:.4f}")

    with open(os.path.join(output_dir, "vanilla_sbi_eval_bundle.json"), "w") as f:
        json.dump(bundle, f, indent=2)


if __name__ == "__main__":
    main()
