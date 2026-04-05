#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

try:
    from .age_gate_models import AgeGateClassifier, apply_temperature
    from .age_regimes import regime_names as default_regime_names
    from .eval_subgroups import subgroup_masks, summarize_group
    from .eval_utils import (
        DEFAULT_TARGET_COLS,
        auto_device,
        central_rank_coverage,
        column_indices,
        ensure_dir,
        interval_metrics,
        ks_uniform,
        maybe_denormalize,
        parse_float_list,
        parse_str_list,
        projection_ranks,
    )
    from .inference_utils import NormStats
    from .sample_sbi_age_mixture import (
        _assignment_frequencies,
        _build_gate_encoder,
        _gate_entropy,
        _load_temperature,
        _parse_expert_spec,
        _prepare_from_cache_loaded,
        _sample_assignments,
    )
    from .sample_sbi_posterior import (
        _build_model_from_config,
        _load_json,
        _load_state_dict,
        _resolve_color_definitions,
        _resolve_input_layout,
        _sample_chunk,
        _select_rows,
    )
    from .data import load_cache_arrays
except ImportError:
    from age_gate_models import AgeGateClassifier, apply_temperature
    from age_regimes import regime_names as default_regime_names
    from eval_subgroups import subgroup_masks, summarize_group
    from eval_utils import (
        DEFAULT_TARGET_COLS,
        auto_device,
        central_rank_coverage,
        column_indices,
        ensure_dir,
        interval_metrics,
        ks_uniform,
        maybe_denormalize,
        parse_float_list,
        parse_str_list,
        projection_ranks,
    )
    from inference_utils import NormStats
    from sample_sbi_age_mixture import (
        _assignment_frequencies,
        _build_gate_encoder,
        _gate_entropy,
        _load_temperature,
        _parse_expert_spec,
        _prepare_from_cache_loaded,
        _sample_assignments,
    )
    from sample_sbi_posterior import (
        _build_model_from_config,
        _load_json,
        _load_state_dict,
        _resolve_color_definitions,
        _resolve_input_layout,
        _sample_chunk,
        _select_rows,
    )
    from data import load_cache_arrays


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a soft age-gated mixture posterior on cached rows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gate-model-dir", type=str, required=True)
    p.add_argument("--gate-run-name", type=str, required=True)
    p.add_argument(
        "--expert",
        action="append",
        required=True,
        help="Expert spec formatted as regime_name=model_dir:run_name. Repeat once per expert.",
    )
    p.add_argument("--cache-path", type=str, required=True)
    p.add_argument("--index-file", type=str, default=None)
    p.add_argument("--max-stars", type=int, default=512)
    p.add_argument("--sample-mode", choices=("random", "head"), default="random")
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--target-cols", type=str, default=",".join(DEFAULT_TARGET_COLS))
    p.add_argument("--levels", type=str, default="0.5,0.8,0.9,0.95")
    p.add_argument("--coverage-level", type=float, default=0.9)
    p.add_argument("--young-logage-threshold", type=float, default=7.8)
    p.add_argument("--min-group-size", type=int, default=20)
    p.add_argument("--num-projections", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    p.add_argument("--tag", type=str, default=None)
    return p.parse_args()


def _load_gate_and_experts(args: argparse.Namespace, device: str):
    gate_model_dir = args.gate_model_dir
    gate_run_name = args.gate_run_name
    gate_cfg_path = os.path.join(gate_model_dir, f"age_gate_config_{gate_run_name}.json")
    gate_ckpt_path = os.path.join(gate_model_dir, f"best_age_gate_{gate_run_name}.pt")
    gate_temp_path = os.path.join(gate_model_dir, f"age_gate_temperature_{gate_run_name}.json")
    gate_meta_path = os.path.join(gate_model_dir, f"age_gate_norm_meta_{gate_run_name}.npz")

    gate_config = _load_json(gate_cfg_path)
    gate_temperature = _load_temperature(gate_temp_path)
    gate_edges = gate_config.get("age_bin_edges", [7.8, 8.8])
    regime_names = [str(x) for x in gate_config.get("regime_names", default_regime_names(gate_edges))]

    expert_specs = [_parse_expert_spec(spec) for spec in args.expert]
    experts: list[dict[str, object]] = []
    theta_columns_ref: list[str] | None = None
    for regime_name, model_dir, run_name in expert_specs:
        cfg_path = os.path.join(model_dir, f"posterior_config_{run_name}.json")
        ckpt_path = os.path.join(model_dir, f"best_model_{run_name}.pt")
        meta_path = os.path.join(model_dir, f"posterior_norm_meta_{run_name}.npz")
        config = _load_json(cfg_path)
        norm_stats = NormStats(meta_path)
        (
            input_columns_base,
            input_columns_model,
            use_colors,
            color_names,
            color_means,
            color_stds,
        ) = _resolve_input_layout(config, norm_stats)
        color_definitions = _resolve_color_definitions(color_names) if use_colors else []
        theta_columns = [str(c) for c in config["theta_columns"]]
        if theta_columns_ref is None:
            theta_columns_ref = theta_columns
        elif theta_columns != theta_columns_ref:
            raise ValueError(
                f"Expert '{regime_name}' theta columns differ from reference. "
                f"Expected {theta_columns_ref}, got {theta_columns}."
            )
        model = _build_model_from_config(config, input_columns_override=input_columns_model)
        model.load_state_dict(_load_state_dict(ckpt_path, device=device))
        model.to(device)
        model.eval()
        experts.append(
            {
                "regime_name": regime_name,
                "model_dir": model_dir,
                "run_name": run_name,
                "config": config,
                "norm_stats": norm_stats,
                "model": model,
                "method": str(config.get("method", "flow_matching")),
                "theta_columns": theta_columns,
                "input_columns_base": input_columns_base,
                "input_columns_model": input_columns_model,
                "use_colors": use_colors,
                "color_names": color_names,
                "color_means": color_means,
                "color_stds": color_stds,
                "color_definitions": color_definitions,
            }
        )

    expert_by_regime = {str(x["regime_name"]): x for x in experts}
    missing_regimes = [r for r in regime_names if r not in expert_by_regime]
    if missing_regimes:
        raise ValueError(f"Missing experts for gate regimes: {missing_regimes}")

    try:
        gate_norm_stats = NormStats(gate_meta_path)
    except ValueError as exc:
        if "color statistics" not in str(exc):
            raise
        gate_norm_stats = expert_by_regime[regime_names[0]]["norm_stats"]
    try:
        (
            gate_input_columns_base,
            gate_input_columns_model,
            gate_use_colors,
            gate_color_names,
            gate_color_means,
            gate_color_stds,
        ) = _resolve_input_layout(gate_config, gate_norm_stats)
    except ValueError as exc:
        if "color normalization stats are missing" not in str(exc):
            raise
        ref_expert = expert_by_regime[regime_names[0]]
        gate_input_columns_base = [str(c) for c in gate_config["input_columns"]]
        gate_input_columns_model = [str(c) for c in gate_config["input_columns_with_colors"]]
        gate_use_colors = bool(gate_config.get("use_colors", False))
        gate_color_names = [str(c) for c in gate_config.get("color_names", [])]
        gate_color_means = np.asarray(ref_expert["color_means"], dtype=np.float32)
        gate_color_stds = np.asarray(ref_expert["color_stds"], dtype=np.float32)
    gate_color_definitions = _resolve_color_definitions(gate_color_names) if gate_use_colors else []

    gate_encoder = _build_gate_encoder(gate_config, gate_input_columns_model)
    gate_model = AgeGateClassifier(
        encoder=gate_encoder,
        num_regimes=len(regime_names),
        hidden_dim=int(gate_config.get("gate_hidden_dim", 256)),
        dropout=float(gate_config.get("gate_dropout", 0.0)),
    ).to(device)
    gate_model.load_state_dict(_load_state_dict(gate_ckpt_path, device=device))
    gate_model.eval()

    theta_columns = theta_columns_ref or []
    theta_idx_full = [experts[0]["norm_stats"].column_index(c) for c in theta_columns]
    return {
        "gate_model": gate_model,
        "gate_temperature": gate_temperature,
        "gate_model_dir": gate_model_dir,
        "gate_run_name": gate_run_name,
        "gate_input_columns_base": gate_input_columns_base,
        "gate_use_colors": gate_use_colors,
        "gate_color_definitions": gate_color_definitions,
        "gate_color_means": gate_color_means,
        "gate_color_stds": gate_color_stds,
        "regime_names": regime_names,
        "experts": experts,
        "expert_by_regime": expert_by_regime,
        "theta_columns": theta_columns,
        "theta_idx_full": theta_idx_full,
        "norm_stats": experts[0]["norm_stats"],
    }


def _sample_mixture(
    *,
    bundle: dict,
    cache_path: str,
    index_file: str | None,
    max_stars: int | None,
    sample_mode: str,
    num_samples: int,
    steps: int,
    batch_size: int,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache = load_cache_arrays(cache_path)
    rows = _select_rows(
        cache.values_norm.shape[0],
        index_file=index_file,
        max_stars=max_stars,
        sample_mode=sample_mode,
        seed=seed,
    )

    gate_values_np, gate_errors_np, gate_observed_np = _prepare_from_cache_loaded(
        cache,
        row_indices=rows,
        input_columns=bundle["gate_input_columns_base"],
        use_colors=bool(bundle["gate_use_colors"]),
        color_definitions=bundle["gate_color_definitions"],
        color_means=bundle["gate_color_means"],
        color_stds=bundle["gate_color_stds"],
    )
    expert_inputs = {}
    for regime_name in bundle["regime_names"]:
        expert = bundle["expert_by_regime"][regime_name]
        expert_inputs[regime_name] = _prepare_from_cache_loaded(
            cache,
            row_indices=rows,
            input_columns=expert["input_columns_base"],
            use_colors=bool(expert["use_colors"]),
            color_definitions=expert["color_definitions"],
            color_means=expert["color_means"],
            color_stds=expert["color_stds"],
        )

    n_stars = rows.shape[0]
    theta_dim = len(bundle["theta_columns"])
    samples_norm = np.zeros((n_stars, num_samples, theta_dim), dtype=np.float32)
    gate_probs_out = np.zeros((n_stars, len(bundle["regime_names"])), dtype=np.float32)
    gate_entropy_out = np.zeros(n_stars, dtype=np.float32)
    assign_freq_out = np.zeros((n_stars, len(bundle["regime_names"])), dtype=np.float32)

    for start in range(0, n_stars, batch_size):
        end = min(start + batch_size, n_stars)
        vals_t = torch.from_numpy(gate_values_np[start:end]).to(device)
        errs_t = torch.from_numpy(gate_errors_np[start:end]).to(device)
        obs_t = torch.from_numpy(gate_observed_np[start:end]).to(device)
        with torch.no_grad():
            gate_logits = bundle["gate_model"](vals_t, errs_t, obs_t)
            gate_logits = apply_temperature(gate_logits, bundle["gate_temperature"])
            gate_probs_t = torch.softmax(gate_logits, dim=-1)

        expert_samples_t = []
        for regime_name in bundle["regime_names"]:
            expert = bundle["expert_by_regime"][regime_name]
            e_vals_np, e_errs_np, e_obs_np = expert_inputs[regime_name]
            e_vals_t = torch.from_numpy(e_vals_np[start:end]).to(device)
            e_errs_t = torch.from_numpy(e_errs_np[start:end]).to(device)
            e_obs_t = torch.from_numpy(e_obs_np[start:end]).to(device)
            with torch.no_grad():
                samps_t = _sample_chunk(
                    expert["model"],
                    method=str(expert["method"]),
                    values=e_vals_t,
                    errors=e_errs_t,
                    observed=e_obs_t,
                    num_samples=num_samples,
                    steps=steps,
                )
            expert_samples_t.append(samps_t)

        assign_t = _sample_assignments(
            gate_probs_t,
            num_samples=num_samples,
            seed=seed,
            offset=start,
        )
        stacked_t = torch.stack(expert_samples_t, dim=0).permute(1, 2, 0, 3)
        gather_idx = assign_t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, theta_dim)
        mixture_t = torch.gather(stacked_t, 2, gather_idx).squeeze(2)

        samples_norm[start:end] = mixture_t.detach().cpu().numpy().astype(np.float32)
        gate_probs_np = gate_probs_t.detach().cpu().numpy().astype(np.float32)
        assign_np = assign_t.detach().cpu().numpy().astype(np.int64)
        gate_probs_out[start:end] = gate_probs_np
        gate_entropy_out[start:end] = _gate_entropy(gate_probs_np).astype(np.float32)
        assign_freq_out[start:end] = _assignment_frequencies(assign_np, len(bundle["regime_names"]))

    truth_norm = cache.values_norm[rows][:, bundle["theta_idx_full"]].astype(np.float32)
    return rows, truth_norm, samples_norm, gate_probs_out, gate_entropy_out, assign_freq_out


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    target_cols = parse_str_list(args.target_cols)
    levels = parse_float_list(args.levels)
    device = auto_device(args.device)
    output_dir = args.output_dir or os.path.join(args.gate_model_dir, "eval")
    ensure_dir(output_dir)
    tag = args.tag or time.strftime("%Y%m%d_%H%M%S")

    print(f"Using device: {device}")
    print("\n--- Loading gate + experts ---")
    bundle = _load_gate_and_experts(args, device=device)
    if target_cols != bundle["theta_columns"]:
        target_idx = [bundle["theta_columns"].index(c) for c in target_cols]
    else:
        target_idx = list(range(len(target_cols)))

    print("\n--- Mixture sampling ---")
    t0 = time.time()
    selected, truth_norm_full, samples_norm_full, gate_probs, gate_entropy, assign_freq = _sample_mixture(
        bundle=bundle,
        cache_path=args.cache_path,
        index_file=args.index_file,
        max_stars=args.max_stars,
        sample_mode=args.sample_mode,
        num_samples=args.num_samples,
        steps=args.steps,
        batch_size=args.batch_size,
        seed=args.seed,
        device=device,
    )
    elapsed = time.time() - t0
    print(f"  Sampling done in {elapsed:.1f}s")

    truth_norm = truth_norm_full[:, target_idx]
    samples_norm = samples_norm_full[:, :, target_idx]
    truth_phys = maybe_denormalize(bundle["norm_stats"], truth_norm, [bundle["theta_idx_full"][i] for i in target_idx], denorm=True)
    samples_phys = maybe_denormalize(bundle["norm_stats"], samples_norm, [bundle["theta_idx_full"][i] for i in target_idx], denorm=True)

    metrics_df = interval_metrics(samples_phys, truth_phys, target_cols, levels)
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
        seed=args.seed,
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

    cache = load_cache_arrays(args.cache_path)
    values_norm_full = cache.values_norm[selected]
    errors_norm_full = cache.errors_norm[selected]
    observed_mask_full = cache.observed_mask[selected]
    masks = subgroup_masks(
        values_norm=values_norm_full,
        errors_norm=errors_norm_full,
        observed_mask=observed_mask_full,
        norm_stats=bundle["norm_stats"],
        young_logage_threshold=args.young_logage_threshold,
    )

    summary_rows = []
    per_param_rows = []
    for i, (name, mask) in enumerate(masks.items()):
        if int(mask.sum()) < args.min_group_size:
            continue
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
        row["gate_entropy_mean"] = float(gate_entropy[mask].mean())
        for j, regime_name in enumerate(bundle["regime_names"]):
            row[f"gate_p_{regime_name}_mean"] = float(gate_probs[mask, j].mean())
            row[f"assign_frac_{regime_name}_mean"] = float(assign_freq[mask, j].mean())
        summary_rows.append(row)
        per_param_rows.extend(per_param)

    summary_df = pd.DataFrame(summary_rows).sort_values("group")
    per_param_df = pd.DataFrame(per_param_rows).sort_values(["group", "column"])

    coverage_detail_csv = os.path.join(output_dir, f"age_mixture_coverage_detail_{tag}.csv")
    coverage_level_csv = os.path.join(output_dir, f"age_mixture_coverage_by_level_{tag}.csv")
    coverage_summary_json = os.path.join(output_dir, f"age_mixture_coverage_summary_{tag}.json")
    tarp_curve_csv = os.path.join(output_dir, f"age_mixture_tarp_curve_{tag}.csv")
    tarp_summary_json = os.path.join(output_dir, f"age_mixture_tarp_summary_{tag}.json")
    subgroup_summary_csv = os.path.join(output_dir, f"age_mixture_subgroup_summary_{tag}.csv")
    subgroup_per_param_csv = os.path.join(output_dir, f"age_mixture_subgroup_per_param_{tag}.csv")
    gate_summary_parquet = os.path.join(output_dir, f"age_mixture_gate_summary_{tag}.parquet")
    selected_npy = os.path.join(output_dir, f"age_mixture_selected_indices_{tag}.npy")
    meta_json = os.path.join(output_dir, f"age_mixture_eval_meta_{tag}.json")

    metrics_df.to_csv(coverage_detail_csv, index=False)
    summary_by_level.to_csv(coverage_level_csv, index=False)
    tarp_curve_df.to_csv(tarp_curve_csv, index=False)
    summary_df.to_csv(subgroup_summary_csv, index=False)
    per_param_df.to_csv(subgroup_per_param_csv, index=False)
    np.save(selected_npy, selected)

    gate_df = pd.DataFrame({"row_index": selected, "gate_entropy": gate_entropy})
    for j, regime_name in enumerate(bundle["regime_names"]):
        gate_df[f"p_{regime_name}"] = gate_probs[:, j]
        gate_df[f"assign_frac_{regime_name}"] = assign_freq[:, j]
    gate_df.to_parquet(gate_summary_parquet, index=False)

    with open(coverage_summary_json, "w") as f:
        json.dump(
            {
                "gate_model_dir": args.gate_model_dir,
                "gate_run_name": args.gate_run_name,
                "cache_path": args.cache_path,
                "index_file": args.index_file,
                "max_stars": args.max_stars,
                "num_samples": args.num_samples,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "device": device,
                "target_cols": target_cols,
                "levels": levels,
                "overall_ace": overall_ace,
                "artifacts": {
                    "detail_csv": coverage_detail_csv,
                    "level_csv": coverage_level_csv,
                    "selected_indices": selected_npy,
                },
            },
            f,
            indent=2,
        )
    with open(tarp_summary_json, "w") as f:
        json.dump(
            {
                "ks_uniform": tarp_ks,
                "ace": tarp_ace,
                "mce": tarp_mce,
                "num_projections": args.num_projections,
                "target_cols": target_cols,
                "curve_csv": tarp_curve_csv,
            },
            f,
            indent=2,
        )
    with open(meta_json, "w") as f:
        json.dump(
            {
                "gate_model_dir": args.gate_model_dir,
                "gate_run_name": args.gate_run_name,
                "experts": args.expert,
                "cache_path": args.cache_path,
                "index_file": args.index_file,
                "selected_npy": selected_npy,
                "coverage_summary_json": coverage_summary_json,
                "tarp_summary_json": tarp_summary_json,
                "subgroup_summary_csv": subgroup_summary_csv,
                "subgroup_per_param_csv": subgroup_per_param_csv,
                "gate_summary_parquet": gate_summary_parquet,
            },
            f,
            indent=2,
        )

    print("\n--- Coverage summary ---")
    print(summary_by_level.to_string(index=False))
    print(f"\nOverall ACE: {overall_ace:.4f}")
    print("\n--- TARP summary ---")
    print(f"  KS(uniform) = {tarp_ks:.4f}")
    print(f"  ACE         = {tarp_ace:.4f}")
    print(f"  MCE         = {tarp_mce:.4f}")
    print(f"\nSaved coverage summary: {coverage_summary_json}")
    print(f"Saved TARP summary:     {tarp_summary_json}")
    print(f"Saved subgroup summary: {subgroup_summary_csv}")
    print(f"Saved gate parquet:     {gate_summary_parquet}")


if __name__ == "__main__":
    main()
