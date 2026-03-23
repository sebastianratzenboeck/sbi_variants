#!/usr/bin/env python
"""Sensitivity/robustness evaluation for SimFormer posteriors."""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

try:
    from .columns import OBS_COLS, N_INTRINSIC, N_TRUE_MAG
    from .sample_mock_galaxy import load_model, sample_posterior
    from .eval_utils import (
        DEFAULT_TARGET_COLS,
        LOG_ERR_UNOBS,
        auto_device,
        column_indices,
        ensure_dir,
        load_cache_arrays,
        parse_float_list,
        parse_str_list,
        to_input_tensors,
    )
except ImportError:
    from columns import OBS_COLS, N_INTRINSIC, N_TRUE_MAG
    from sample_mock_galaxy import load_model, sample_posterior
    from eval_utils import (
        DEFAULT_TARGET_COLS,
        LOG_ERR_UNOBS,
        auto_device,
        column_indices,
        ensure_dir,
        load_cache_arrays,
        parse_float_list,
        parse_str_list,
        to_input_tensors,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Sensitivity tests: masking, error inflation, survey ablation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model-dir", type=str, required=True)
    p.add_argument("--run-name", type=str, default="default")
    p.add_argument("--cache-path", type=str, required=True,
                   help="Path to build_arrays_cache.npz")
    p.add_argument("--index-file", type=str, default=None,
                   help="Optional .npy row indices (e.g. test_indices.npy)")
    p.add_argument("--max-stars", type=int, default=256)
    p.add_argument("--sample-mode", choices=("random", "head"), default="random")
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--target-cols",
        type=str,
        default=",".join(DEFAULT_TARGET_COLS),
        help="Comma-separated columns to evaluate",
    )
    p.add_argument("--coverage-level", type=float, default=0.9)
    p.add_argument("--dropout-rates", type=str, default="0.0,0.1,0.3,0.5")
    p.add_argument("--error-factors", type=str, default="1.0,2.0,5.0")
    p.add_argument("--survey-modes", type=str, default="full,gaia_2mass,gaia_only")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--tag", type=str, default=None)
    return p.parse_args()


def scenario_keep_obs_indices(mode: str) -> set[int]:
    keep = set()
    for i, col in enumerate(OBS_COLS):
        is_gaia = col.startswith("GAIA_GAIA3.") or col == "parallax_obs"
        is_2mass = col.startswith("2MASS_")
        if mode == "full":
            keep.add(i)
        elif mode == "gaia_2mass":
            if is_gaia or is_2mass:
                keep.add(i)
        elif mode == "gaia_only":
            if is_gaia:
                keep.add(i)
        else:
            raise ValueError(f"Unknown survey mode '{mode}'")
    return keep


def build_scenarios(
    dropout_rates: list[float],
    error_factors: list[float],
    survey_modes: list[str],
) -> list[dict]:
    scenarios = [{"name": "baseline", "kind": "baseline"}]

    for r in dropout_rates:
        if r <= 0:
            continue
        scenarios.append({"name": f"maskdrop_{r:.2f}", "kind": "maskdrop", "rate": r})

    for f in error_factors:
        if abs(f - 1.0) < 1e-12:
            continue
        scenarios.append({"name": f"errscale_{f:.2f}", "kind": "errscale", "factor": f})

    for m in survey_modes:
        if m == "full":
            continue
        scenarios.append({"name": f"survey_{m}", "kind": "survey", "mode": m})

    return scenarios


def apply_scenario(
    scenario: dict,
    values_norm: np.ndarray,
    errors_norm: np.ndarray,
    observed_mask: np.ndarray,
    log_err_std: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply perturbation scenario and return modified arrays.

    Returns:
        values_norm, errors_norm, observed_mask, condition_mask
    """
    v = values_norm.copy()
    e = errors_norm.copy()
    o = observed_mask.copy()

    obs_start = N_INTRINSIC + N_TRUE_MAG
    # Base condition mask: condition on observed obs block + sky.
    c = np.zeros_like(o, dtype=np.float32)
    c[:, obs_start:] = o[:, obs_start:]
    c[:, 0:3] = 1.0

    kind = scenario["kind"]
    if kind == "maskdrop":
        p = float(scenario["rate"])
        rng = np.random.default_rng(seed)
        obs_cond = c[:, obs_start:] > 0.5
        drop = rng.random(obs_cond.shape) < p
        to_drop = obs_cond & drop
        c[:, obs_start:][to_drop] = 0.0
        o[:, obs_start:][to_drop] = 0.0
        e[:, obs_start:][to_drop] = LOG_ERR_UNOBS

    elif kind == "errscale":
        f = float(scenario["factor"])
        if f <= 0:
            raise ValueError(f"Error scale factor must be positive, got {f}")
        if log_err_std <= 0:
            raise ValueError(f"log_err_std must be >0, got {log_err_std}")
        delta = np.log(f) / log_err_std
        is_real = (e > -4.9) & (e < 4.9)  # exclude sentinels
        e[is_real] = e[is_real] + delta

    elif kind == "survey":
        mode = scenario["mode"]
        keep = scenario_keep_obs_indices(mode)
        for i in range(len(OBS_COLS)):
            if i in keep:
                continue
            gidx = obs_start + i
            c[:, gidx] = 0.0
            o[:, gidx] = 0.0
            e[:, gidx] = LOG_ERR_UNOBS

    elif kind == "baseline":
        pass
    else:
        raise ValueError(f"Unknown scenario kind '{kind}'")

    return v, e, o, c


def central_interval_coverage(samples: np.ndarray, truth: np.ndarray, level: float) -> tuple[np.ndarray, np.ndarray]:
    q_lo = (1.0 - level) / 2.0
    q_hi = 1.0 - q_lo
    lo = np.quantile(samples, q_lo, axis=1)  # (N,D)
    hi = np.quantile(samples, q_hi, axis=1)  # (N,D)
    inside = (truth >= lo) & (truth <= hi)   # (N,D)
    return inside.mean(axis=0), (hi - lo).mean(axis=0)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not (0.0 < args.coverage_level < 1.0):
        raise ValueError("--coverage-level must be in (0,1)")

    dropout_rates = parse_float_list(args.dropout_rates)
    error_factors = parse_float_list(args.error_factors)
    survey_modes = parse_str_list(args.survey_modes)
    target_cols = parse_str_list(args.target_cols)
    scenarios = build_scenarios(dropout_rates, error_factors, survey_modes)

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
    truth = values_norm[:, target_idx]
    print(f"  Stars selected: {len(selected):,}")
    print(f"  Scenarios: {', '.join(s['name'] for s in scenarios)}")

    per_param_rows = []
    summary_rows = []
    baseline_summary = None

    for si, scenario in enumerate(scenarios):
        print(f"\n--- Scenario: {scenario['name']} ---")
        v_s, e_s, o_s, c_s = apply_scenario(
            scenario=scenario,
            values_norm=values_norm,
            errors_norm=errors_norm,
            observed_mask=observed_mask,
            log_err_std=norm_stats.log_err_std,
            seed=args.seed + si,
        )
        cv, cm, om, er = to_input_tensors(
            values_norm=v_s,
            errors_norm=e_s,
            observed_mask=o_s,
            condition_mask=c_s,
            columns=norm_stats.columns,
            device="cpu",
        )

        t0 = time.time()
        samples = sample_posterior(
            model=model,
            condition_values=cv,
            condition_mask=cm,
            observed_mask=om,
            errors=er,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            steps=args.steps,
            device=device,
        ).cpu().numpy()[:, :, target_idx]
        elapsed = time.time() - t0
        print(f"  Sampling done in {elapsed:.1f}s")

        post_mean = samples.mean(axis=1)  # (N,D)
        post_std = samples.std(axis=1)    # (N,D)

        rmse = np.sqrt(np.mean((post_mean - truth) ** 2, axis=0))
        mae = np.mean(np.abs(post_mean - truth), axis=0)
        cov, width = central_interval_coverage(samples, truth, args.coverage_level)
        std_mean = post_std.mean(axis=0)

        for j, col in enumerate(target_cols):
            per_param_rows.append(
                {
                    "scenario": scenario["name"],
                    "column": col,
                    "rmse_norm": float(rmse[j]),
                    "mae_norm": float(mae[j]),
                    f"coverage_{int(args.coverage_level * 100)}": float(cov[j]),
                    f"width_{int(args.coverage_level * 100)}": float(width[j]),
                    "posterior_std_mean": float(std_mean[j]),
                }
            )

        summary = {
            "scenario": scenario["name"],
            "n_stars": int(values_norm.shape[0]),
            "conditioned_mean": float(c_s.sum(axis=1).mean()),
            "observed_mean": float(o_s.sum(axis=1).mean()),
            "rmse_norm_macro": float(rmse.mean()),
            "mae_norm_macro": float(mae.mean()),
            f"coverage_{int(args.coverage_level * 100)}_macro": float(cov.mean()),
            f"width_{int(args.coverage_level * 100)}_macro": float(width.mean()),
            "posterior_std_mean_macro": float(std_mean.mean()),
            "elapsed_sec": float(elapsed),
        }
        summary_rows.append(summary)

        if scenario["name"] == "baseline":
            baseline_summary = summary

    summary_df = pd.DataFrame(summary_rows)
    if baseline_summary is not None:
        for metric in (
            "rmse_norm_macro",
            "mae_norm_macro",
            f"coverage_{int(args.coverage_level * 100)}_macro",
            f"width_{int(args.coverage_level * 100)}_macro",
            "posterior_std_mean_macro",
        ):
            summary_df[f"delta_{metric}"] = summary_df[metric] - float(baseline_summary[metric])

    per_param_df = pd.DataFrame(per_param_rows)

    summary_csv = os.path.join(output_dir, f"sensitivity_summary_{tag}.csv")
    per_param_csv = os.path.join(output_dir, f"sensitivity_per_param_{tag}.csv")
    selected_npy = os.path.join(output_dir, f"sensitivity_selected_indices_{tag}.npy")
    summary_json = os.path.join(output_dir, f"sensitivity_summary_{tag}.json")

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
                "steps": args.steps,
                "batch_size": args.batch_size,
                "target_cols": target_cols,
                "coverage_level": args.coverage_level,
                "scenarios": [s["name"] for s in scenarios],
                "artifacts": {
                    "summary_csv": summary_csv,
                    "per_param_csv": per_param_csv,
                    "selected_indices": selected_npy,
                },
            },
            f,
            indent=2,
        )

    print("\n--- Scenario summary ---")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary CSV:  {summary_csv}")
    print(f"Saved per-param CSV:{per_param_csv}")
    print(f"Saved summary JSON: {summary_json}")


if __name__ == "__main__":
    main()
