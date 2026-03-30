#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

try:
    from .columns import OBS_COLS
    from .eval_utils import (
        DEFAULT_TARGET_COLS,
        auto_device,
        column_indices,
        ensure_dir,
        ks_uniform,
        load_cache_arrays,
        maybe_denormalize,
        projection_ranks,
        to_input_tensors,
    )
    from .sample_mock_galaxy import load_model, sample_posterior
except ImportError:
    from columns import OBS_COLS
    from eval_utils import (
        DEFAULT_TARGET_COLS,
        auto_device,
        column_indices,
        ensure_dir,
        ks_uniform,
        load_cache_arrays,
        maybe_denormalize,
        projection_ranks,
        to_input_tensors,
    )
    from sample_mock_galaxy import load_model, sample_posterior


YOUNG_LOGAGE_THRESHOLD = 7.8
PARALLAX_COL = "parallax_obs"
PARALLAX_ERR_COL = "parallax_err"
PHOT_COLS = [c for c in OBS_COLS if c != PARALLAX_COL]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate posterior performance by observational subgroup.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--cache-path", type=str, required=True)
    p.add_argument("--index-file", type=str, default=None)
    p.add_argument("--max-stars", type=int, default=512)
    p.add_argument("--sample-mode", choices=("random", "head"), default="random")
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--target-cols", type=str, default=",".join(DEFAULT_TARGET_COLS))
    p.add_argument("--coverage-level", type=float, default=0.9)
    p.add_argument("--young-logage-threshold", type=float, default=YOUNG_LOGAGE_THRESHOLD)
    p.add_argument("--min-group-size", type=int, default=20)
    p.add_argument("--num-projections", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--tag", type=str, default=None)
    return p.parse_args()


def _parse_target_cols(raw: str) -> list[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one target column")
    return vals


def central_interval_metrics(samples: np.ndarray, truth: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    q_lo = (1.0 - level) / 2.0
    q_hi = 1.0 - q_lo
    lo = np.quantile(samples, q_lo, axis=1)
    hi = np.quantile(samples, q_hi, axis=1)
    inside = (truth >= lo) & (truth <= hi)
    return inside.mean(axis=0), (hi - lo).mean(axis=0)


def subgroup_masks(
    values_norm: np.ndarray,
    errors_norm: np.ndarray,
    observed_mask: np.ndarray,
    norm_stats,
    *,
    young_logage_threshold: float,
) -> dict[str, np.ndarray]:
    columns = list(norm_stats.columns)
    col_to_idx = {c: i for i, c in enumerate(columns)}

    phot_idx = [col_to_idx[c] for c in PHOT_COLS if c in col_to_idx]
    parallax_idx = col_to_idx[PARALLAX_COL]
    logage_idx = col_to_idx["logAge"]

    phot_count = observed_mask[:, phot_idx].sum(axis=1).astype(np.int64)
    parallax_obs = observed_mask[:, parallax_idx] > 0.5

    logage_phys = norm_stats.denormalize_numpy(values_norm[:, [logage_idx]], [logage_idx]).reshape(-1)
    young = logage_phys < float(young_logage_threshold)

    parallax_phys = norm_stats.denormalize_numpy(values_norm[:, [parallax_idx]], [parallax_idx]).reshape(-1)
    if norm_stats.log_err_std <= 0:
        raise ValueError(f"Invalid log_err_std={norm_stats.log_err_std}")
    parallax_err_norm = errors_norm[:, parallax_idx]
    parallax_err_phys = np.full_like(parallax_err_norm, np.nan, dtype=np.float32)
    real = parallax_obs & np.isfinite(parallax_err_norm) & (parallax_err_norm > -4.9) & (parallax_err_norm < 4.9)
    parallax_err_phys[real] = np.exp(
        parallax_err_norm[real] * float(norm_stats.log_err_std) + float(norm_stats.log_err_mean)
    )
    parallax_snr = np.full_like(parallax_err_phys, np.nan, dtype=np.float32)
    good = real & np.isfinite(parallax_phys) & (parallax_err_phys > 0)
    parallax_snr[good] = np.abs(parallax_phys[good]) / parallax_err_phys[good]

    masks: dict[str, np.ndarray] = {
        "all": np.ones(values_norm.shape[0], dtype=bool),
        "young": young,
        "not_young": ~young,
        "parallax_present": parallax_obs,
        "parallax_missing": ~parallax_obs,
    }

    phot_bins = [
        ("phot_0_5", (phot_count >= 0) & (phot_count <= 5)),
        ("phot_6_10", (phot_count >= 6) & (phot_count <= 10)),
        ("phot_11_15", (phot_count >= 11) & (phot_count <= 15)),
        ("phot_16_18", phot_count >= 16),
    ]
    for name, mask in phot_bins:
        masks[name] = mask

    snr_bins = [
        ("parallax_snr_lt5", good & (parallax_snr < 5.0)),
        ("parallax_snr_5_10", good & (parallax_snr >= 5.0) & (parallax_snr < 10.0)),
        ("parallax_snr_10_20", good & (parallax_snr >= 10.0) & (parallax_snr < 20.0)),
        ("parallax_snr_ge20", good & (parallax_snr >= 20.0)),
    ]
    for name, mask in snr_bins:
        masks[name] = mask

    return masks


def summarize_group(
    name: str,
    mask: np.ndarray,
    *,
    samples_phys: np.ndarray,
    truth_phys: np.ndarray,
    samples_norm: np.ndarray,
    truth_norm: np.ndarray,
    target_cols: list[str],
    coverage_level: float,
    num_projections: int,
    seed: int,
) -> tuple[dict, list[dict]]:
    s_phys = samples_phys[mask]
    t_phys = truth_phys[mask]
    s_norm = samples_norm[mask]
    t_norm = truth_norm[mask]

    post_mean_norm = s_norm.mean(axis=1)
    rmse = np.sqrt(np.mean((post_mean_norm - t_norm) ** 2, axis=0))
    mae = np.mean(np.abs(post_mean_norm - t_norm), axis=0)
    cov, width = central_interval_metrics(s_phys, t_phys, coverage_level)

    u = projection_ranks(
        samples=s_norm,
        truth=t_norm,
        num_projections=num_projections,
        seed=seed,
    ).reshape(-1)
    tarp_ace = float(np.mean(np.abs(np.sort(u) - np.linspace(1.0 / len(u), 1.0, len(u)))))
    # Keep the same directly interpretable statistic from eval_tarp.
    ks = ks_uniform(u)

    row = {
        "group": name,
        "n_stars": int(mask.sum()),
        "rmse_norm_macro": float(rmse.mean()),
        "mae_norm_macro": float(mae.mean()),
        f"coverage_{int(coverage_level * 100)}_macro": float(cov.mean()),
        f"width_{int(coverage_level * 100)}_macro": float(width.mean()),
        "tarp_ks_uniform": float(ks),
    }

    per_param = []
    for j, col in enumerate(target_cols):
        per_param.append(
            {
                "group": name,
                "column": col,
                "rmse_norm": float(rmse[j]),
                "mae_norm": float(mae[j]),
                f"coverage_{int(coverage_level * 100)}": float(cov[j]),
                f"width_{int(coverage_level * 100)}": float(width[j]),
            }
        )

    return row, per_param


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not (0.0 < args.coverage_level < 1.0):
        raise ValueError("coverage level must be in (0,1)")

    target_cols = _parse_target_cols(args.target_cols)
    device = auto_device(args.device)
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
    print(f"  Stars selected: {len(selected):,}")

    cv, cm, om, er = to_input_tensors(
        values_norm,
        errors_norm,
        observed_mask,
        columns=norm_stats.columns,
        device="cpu",
    )

    print("\n--- Sampling ---")
    t0 = time.time()
    samples_norm_full = sample_posterior(
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

    samples_norm = samples_norm_full[:, :, target_idx]
    truth_norm = values_norm[:, target_idx]
    samples_phys = maybe_denormalize(norm_stats, samples_norm, target_idx, denorm=True)
    truth_phys = maybe_denormalize(norm_stats, truth_norm, target_idx, denorm=True)

    masks = subgroup_masks(
        values_norm=values_norm,
        errors_norm=errors_norm,
        observed_mask=observed_mask,
        norm_stats=norm_stats,
        young_logage_threshold=args.young_logage_threshold,
    )

    summary_rows = []
    per_param_rows = []
    for i, (name, mask) in enumerate(masks.items()):
        n = int(mask.sum())
        if n < args.min_group_size:
            continue
        print(f"  Group {name}: {n} stars")
        row, per_param = summarize_group(
            name=name,
            mask=mask,
            samples_phys=samples_phys,
            truth_phys=truth_phys,
            samples_norm=samples_norm,
            truth_norm=truth_norm,
            target_cols=target_cols,
            coverage_level=args.coverage_level,
            num_projections=args.num_projections,
            seed=args.seed + i,
        )
        summary_rows.append(row)
        per_param_rows.extend(per_param)

    summary_df = pd.DataFrame(summary_rows).sort_values("group")
    per_param_df = pd.DataFrame(per_param_rows).sort_values(["group", "column"])

    summary_csv = os.path.join(output_dir, f"subgroup_summary_{tag}.csv")
    per_param_csv = os.path.join(output_dir, f"subgroup_per_param_{tag}.csv")
    selected_npy = os.path.join(output_dir, f"subgroup_selected_indices_{tag}.npy")
    summary_json = os.path.join(output_dir, f"subgroup_summary_{tag}.json")

    summary_df.to_csv(summary_csv, index=False)
    per_param_df.to_csv(per_param_csv, index=False)
    np.save(selected_npy, selected)
    with open(summary_json, "w") as f:
        json.dump(
            {
                "model_dir": args.model_dir,
                "run_name": args.run_name,
                "cache_path": args.cache_path,
                "index_file": args.index_file,
                "max_stars": args.max_stars,
                "num_samples": args.num_samples,
                "coverage_level": args.coverage_level,
                "young_logage_threshold": args.young_logage_threshold,
                "min_group_size": args.min_group_size,
                "artifacts": {
                    "summary_csv": summary_csv,
                    "per_param_csv": per_param_csv,
                    "selected_indices": selected_npy,
                },
            },
            f,
            indent=2,
        )

    print("\n--- Subgroup summary ---")
    print(summary_df.to_string(index=False))
    print(f"Saved subgroup summary: {summary_csv}")
    print(f"Saved subgroup per-param: {per_param_csv}")
    print(f"Saved subgroup json: {summary_json}")


if __name__ == "__main__":
    main()
