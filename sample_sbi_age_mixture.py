#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence

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
    from .columns import COLOR_DEFINITIONS
    from .data import column_indices, compute_colors_from_cache, load_cache_arrays
    from .encoder import ObservationEncoder
    from .inference_utils import NormStats
    from .sample_sbi_posterior import (
        _build_model_from_config,
        _load_json,
        _load_state_dict,
        _prepare_from_obs_file,
        _resolve_color_definitions,
        _resolve_input_layout,
        _sample_chunk,
        _select_rows,
    )
except ImportError:
    from age_gate_models import AgeGateClassifier, apply_temperature
    from age_regimes import regime_names as default_regime_names
    from columns import COLOR_DEFINITIONS
    from data import column_indices, compute_colors_from_cache, load_cache_arrays
    from encoder import ObservationEncoder
    from inference_utils import NormStats
    from sample_sbi_posterior import (
        _build_model_from_config,
        _load_json,
        _load_state_dict,
        _prepare_from_obs_file,
        _resolve_color_definitions,
        _resolve_input_layout,
        _sample_chunk,
        _select_rows,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample from a soft age-gated mixture of SBI posterior experts.",
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

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--cache-path", type=str, default=None)
    src.add_argument("--obs-file", type=str, default=None)

    p.add_argument("--index-file", type=str, default=None)
    p.add_argument("--id-column", type=str, default=None)
    p.add_argument("--max-stars", type=int, default=None)
    p.add_argument("--sample-mode", type=str, default="head", choices=["head", "random"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-samples", type=int, default=512)
    p.add_argument("--steps", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output-prefix", type=str, default=None)
    p.add_argument("--denormalize", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def _parse_expert_spec(spec: str) -> tuple[str, str, str]:
    if "=" not in spec or ":" not in spec:
        raise ValueError(
            f"Invalid --expert spec '{spec}'. Expected regime_name=model_dir:run_name."
        )
    regime, rest = spec.split("=", 1)
    model_dir, run_name = rest.rsplit(":", 1)
    regime = regime.strip()
    model_dir = model_dir.strip()
    run_name = run_name.strip()
    if not regime or not model_dir or not run_name:
        raise ValueError(
            f"Invalid --expert spec '{spec}'. Expected non-empty regime, model_dir, and run_name."
        )
    return regime, model_dir, run_name


def _build_gate_encoder(config: dict, input_columns: Sequence[str]) -> ObservationEncoder:
    return ObservationEncoder(
        input_columns=[str(c) for c in input_columns],
        dim_value=int(config.get("dim_value", 24)),
        dim_id=int(config.get("dim_id", 24)),
        value_calibration_type=str(config.get("value_calibration_type", "scalar_film")),
        dim_error=int(config.get("dim_error", 16)),
        error_embed_type=str(config.get("error_embed_type", "mlp_regime")),
        dim_observed=int(config.get("dim_observed", 8)),
        attn_embed_dim=int(config.get("attn_embed_dim", 128)),
        num_heads=int(config.get("num_heads", 8)),
        num_layers=int(config.get("num_layers", 4)),
        widening_factor=int(config.get("widening_factor", 4)),
        dropout=float(config.get("dropout", 0.05)),
        use_missingness_context=bool(config.get("use_missingness_context", True)),
        missingness_context_hidden_dim=int(config.get("missingness_context_hidden_dim", 64)),
    )


def _load_temperature(path: str) -> float:
    with open(path) as f:
        payload = json.load(f)
    return float(payload["temperature"])


def _prepare_from_cache_loaded(
    cache,
    *,
    row_indices: np.ndarray,
    input_columns: Sequence[str],
    use_colors: bool,
    color_definitions: Sequence[tuple[str, str, str]],
    color_means: np.ndarray | None,
    color_stds: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    in_idx = column_indices(cache.columns, input_columns, role="input")
    values = np.nan_to_num(cache.values_norm[row_indices][:, in_idx], nan=0.0).astype(np.float32)
    errors = cache.errors_norm[row_indices][:, in_idx].astype(np.float32)
    observed = cache.observed_mask[row_indices][:, in_idx].astype(np.float32)

    if use_colors:
        if color_means is None or color_stds is None:
            raise ValueError("Color-enabled input layout requires color_means and color_stds.")
        (
            colors_norm_train,
            color_err_norm,
            color_obs,
            _color_names,
            color_means_train,
            color_stds_train,
        ) = compute_colors_from_cache(
            cache=cache,
            row_indices=row_indices,
            color_definitions=list(color_definitions),
        )
        colors_raw = colors_norm_train * color_stds_train + color_means_train
        denom = np.where(color_stds > 1e-8, color_stds, 1.0)
        colors_norm = (colors_raw - color_means) / denom
        colors_norm[~np.isfinite(colors_norm)] = 0.0

        values = np.concatenate([values, colors_norm.astype(np.float32)], axis=1)
        errors = np.concatenate([errors, color_err_norm.astype(np.float32)], axis=1)
        observed = np.concatenate([observed, color_obs.astype(np.float32)], axis=1)

    return values, errors, observed


def _gate_entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)


def _sample_assignments(probs_t: torch.Tensor, *, num_samples: int, seed: int, offset: int) -> torch.Tensor:
    gen = torch.Generator(device=probs_t.device)
    gen.manual_seed(int(seed + offset))
    return torch.multinomial(probs_t, num_samples=num_samples, replacement=True, generator=gen)


def _assignment_frequencies(assign_np: np.ndarray, num_regimes: int) -> np.ndarray:
    out = np.zeros((assign_np.shape[0], num_regimes), dtype=np.float32)
    for k in range(num_regimes):
        out[:, k] = np.mean(assign_np == k, axis=1, dtype=np.float32)
    return out


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
        print(
            f"Loaded expert '{regime_name}': method={config.get('method', 'flow_matching')}, "
            f"theta_dim={len(theta_columns)}, input_nodes={len(input_columns_model)}"
        )

    expert_by_regime = {str(x["regime_name"]): x for x in experts}
    missing_regimes = [r for r in regime_names if r not in expert_by_regime]
    extra_regimes = [str(x["regime_name"]) for x in experts if str(x["regime_name"]) not in regime_names]
    if missing_regimes:
        raise ValueError(f"Missing experts for gate regimes: {missing_regimes}")
    if extra_regimes:
        raise ValueError(f"Expert specs contain unknown regimes: {extra_regimes}")

    try:
        gate_norm_stats = NormStats(gate_meta_path)
    except ValueError as exc:
        msg = str(exc)
        if "color statistics" not in msg:
            raise
        ref_expert = expert_by_regime[regime_names[0]]
        gate_norm_stats = ref_expert["norm_stats"]
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
        msg = str(exc)
        if "color normalization stats are missing" not in msg:
            raise
        ref_expert = expert_by_regime[regime_names[0]]
        gate_input_columns_base = [str(c) for c in gate_config["input_columns"]]
        gate_input_columns_model = [str(c) for c in gate_config["input_columns_with_colors"]]
        gate_use_colors = bool(gate_config.get("use_colors", False))
        gate_color_names = [str(c) for c in gate_config.get("color_names", [])]
        if not gate_use_colors or not gate_color_names:
            raise
        if gate_input_columns_model != ref_expert["input_columns_model"]:
            raise ValueError(
                "Gate color stats are missing and expert input layout does not match gate layout, "
                "so fallback reconstruction is unsafe."
            ) from exc
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
    print(f"Loaded gate with regimes: {regime_names}")

    theta_columns = theta_columns_ref or []
    theta_dim = len(theta_columns)
    if theta_dim == 0:
        raise ValueError("No theta columns resolved from expert configs.")
    theta_idx_full = [experts[0]["norm_stats"].column_index(c) for c in theta_columns]

    if args.cache_path is not None:
        cache = load_cache_arrays(args.cache_path)
        rows = _select_rows(
            cache.values_norm.shape[0],
            index_file=args.index_file,
            max_stars=args.max_stars,
            sample_mode=args.sample_mode,
            seed=args.seed,
        )
        star_ids = rows.copy()
        source_name = args.cache_path
        gate_values_np, gate_errors_np, gate_observed_np = _prepare_from_cache_loaded(
            cache,
            row_indices=rows,
            input_columns=gate_input_columns_base,
            use_colors=gate_use_colors,
            color_definitions=gate_color_definitions,
            color_means=gate_color_means,
            color_stds=gate_color_stds,
        )
        expert_inputs = {}
        for regime_name in regime_names:
            expert = expert_by_regime[regime_name]
            expert_inputs[regime_name] = _prepare_from_cache_loaded(
                cache,
                row_indices=rows,
                input_columns=expert["input_columns_base"],
                use_colors=bool(expert["use_colors"]),
                color_definitions=expert["color_definitions"],
                color_means=expert["color_means"],
                color_stds=expert["color_stds"],
            )
    else:
        gate_values_np, gate_errors_np, gate_observed_np, star_ids = _prepare_from_obs_file(
            args.obs_file,
            input_columns=gate_input_columns_base,
            norm_stats=gate_norm_stats,
            id_column=args.id_column,
            max_stars=args.max_stars,
            sample_mode=args.sample_mode,
            seed=args.seed,
            use_colors=gate_use_colors,
            color_definitions=gate_color_definitions,
            color_means=gate_color_means,
            color_stds=gate_color_stds,
        )
        source_name = args.obs_file
        expert_inputs = {}
        for regime_name in regime_names:
            expert = expert_by_regime[regime_name]
            expert_inputs[regime_name] = _prepare_from_obs_file(
                args.obs_file,
                input_columns=expert["input_columns_base"],
                norm_stats=expert["norm_stats"],
                id_column=args.id_column,
                max_stars=args.max_stars,
                sample_mode=args.sample_mode,
                seed=args.seed,
                use_colors=bool(expert["use_colors"]),
                color_definitions=expert["color_definitions"],
                color_means=expert["color_means"],
                color_stds=expert["color_stds"],
            )[:3]

    n_stars = gate_values_np.shape[0]
    print(
        f"Mixture sampling from {source_name}: stars={n_stars:,}, "
        f"regimes={len(regime_names)}, theta_dim={theta_dim}, draws={args.num_samples}"
    )

    if args.output_prefix is None:
        out_prefix = os.path.join(gate_model_dir, f"age_mixture_{gate_run_name}")
    else:
        out_prefix = args.output_prefix
    out_dir = os.path.dirname(out_prefix) or "."
    os.makedirs(out_dir, exist_ok=True)

    samples_norm_path = out_prefix + "_samples_norm.npy"
    samples_phys_path = out_prefix + "_samples_phys.npy"
    summary_path = out_prefix + "_summary.parquet"
    gate_path = out_prefix + "_gate.parquet"
    meta_path = out_prefix + "_meta.json"

    samples_norm_mm = np.lib.format.open_memmap(
        samples_norm_path,
        mode="w+",
        dtype=np.float32,
        shape=(n_stars, args.num_samples, theta_dim),
    )
    samples_phys_mm = None
    if args.denormalize:
        samples_phys_mm = np.lib.format.open_memmap(
            samples_phys_path,
            mode="w+",
            dtype=np.float32,
            shape=(n_stars, args.num_samples, theta_dim),
        )

    summary_mean = np.zeros((n_stars, theta_dim), dtype=np.float32)
    summary_std = np.zeros((n_stars, theta_dim), dtype=np.float32)
    gate_probs_out = np.zeros((n_stars, len(regime_names)), dtype=np.float32)
    gate_entropy_out = np.zeros(n_stars, dtype=np.float32)
    assign_freq_out = np.zeros((n_stars, len(regime_names)), dtype=np.float32)

    t0 = time.time()
    for start in range(0, n_stars, args.batch_size):
        end = min(start + args.batch_size, n_stars)
        vals_t = torch.from_numpy(gate_values_np[start:end]).to(device)
        errs_t = torch.from_numpy(gate_errors_np[start:end]).to(device)
        obs_t = torch.from_numpy(gate_observed_np[start:end]).to(device)

        with torch.no_grad():
            gate_logits = gate_model(vals_t, errs_t, obs_t)
            gate_logits = apply_temperature(gate_logits, gate_temperature)
            gate_probs_t = torch.softmax(gate_logits, dim=-1)

        expert_samples_t: list[torch.Tensor] = []
        for regime_name in regime_names:
            expert = expert_by_regime[regime_name]
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
                    num_samples=args.num_samples,
                    steps=args.steps,
                )
            expert_samples_t.append(samps_t)

        assign_t = _sample_assignments(
            gate_probs_t,
            num_samples=args.num_samples,
            seed=args.seed,
            offset=start,
        )
        stacked_t = torch.stack(expert_samples_t, dim=0).permute(1, 2, 0, 3)  # (B,S,K,D)
        gather_idx = assign_t.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, theta_dim)
        mixture_t = torch.gather(stacked_t, 2, gather_idx).squeeze(2)

        mixture_norm = mixture_t.detach().cpu().numpy().astype(np.float32)
        gate_probs_np = gate_probs_t.detach().cpu().numpy().astype(np.float32)
        assign_np = assign_t.detach().cpu().numpy().astype(np.int64)

        samples_norm_mm[start:end] = mixture_norm
        gate_probs_out[start:end] = gate_probs_np
        gate_entropy_out[start:end] = _gate_entropy(gate_probs_np).astype(np.float32)
        assign_freq_out[start:end] = _assignment_frequencies(assign_np, len(regime_names))

        if args.denormalize:
            flat = mixture_norm.reshape(-1, theta_dim)
            phys_flat = experts[0]["norm_stats"].denormalize_numpy(flat, column_indices=theta_idx_full).astype(np.float32)
            mixture_phys = phys_flat.reshape(end - start, args.num_samples, theta_dim)
            samples_phys_mm[start:end] = mixture_phys
            mixture_for_summary = mixture_phys
        else:
            mixture_for_summary = mixture_norm

        summary_mean[start:end] = mixture_for_summary.mean(axis=1)
        summary_std[start:end] = mixture_for_summary.std(axis=1)

        done = end
        elapsed = time.time() - t0
        rate = done / max(elapsed, 1e-6)
        print(f"  sampled {done:,}/{n_stars:,} stars ({rate:.1f} stars/s)")

    del samples_norm_mm
    if samples_phys_mm is not None:
        del samples_phys_mm

    summary_df = pd.DataFrame({"star_id": star_ids})
    for j, col in enumerate(theta_columns):
        summary_df[f"{col}_mean"] = summary_mean[:, j]
        summary_df[f"{col}_std"] = summary_std[:, j]
    for j, regime_name in enumerate(regime_names):
        summary_df[f"gate_p_{regime_name}"] = gate_probs_out[:, j]
        summary_df[f"assign_frac_{regime_name}"] = assign_freq_out[:, j]
    summary_df["gate_entropy"] = gate_entropy_out
    summary_df["gate_top_regime"] = [regime_names[i] for i in np.argmax(gate_probs_out, axis=1)]
    summary_df.to_parquet(summary_path, index=False)

    gate_df = pd.DataFrame({"star_id": star_ids, "gate_entropy": gate_entropy_out})
    for j, regime_name in enumerate(regime_names):
        gate_df[f"p_{regime_name}"] = gate_probs_out[:, j]
        gate_df[f"assign_frac_{regime_name}"] = assign_freq_out[:, j]
    gate_df.to_parquet(gate_path, index=False)

    elapsed = time.time() - t0
    meta = {
        "gate_model_dir": gate_model_dir,
        "gate_run_name": gate_run_name,
        "gate_temperature": float(gate_temperature),
        "source": source_name,
        "num_stars": int(n_stars),
        "num_samples": int(args.num_samples),
        "theta_columns": theta_columns,
        "regime_names": regime_names,
        "experts": [
            {
                "regime_name": str(expert["regime_name"]),
                "model_dir": str(expert["model_dir"]),
                "run_name": str(expert["run_name"]),
                "method": str(expert["method"]),
            }
            for expert in experts
        ],
        "denormalize": bool(args.denormalize),
        "steps": int(args.steps),
        "batch_size": int(args.batch_size),
        "elapsed_sec": float(elapsed),
        "outputs": {
            "samples_norm_npy": samples_norm_path,
            "samples_phys_npy": samples_phys_path if args.denormalize else None,
            "summary_parquet": summary_path,
            "gate_parquet": gate_path,
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved normalized samples: {samples_norm_path}")
    if args.denormalize:
        print(f"Saved physical samples:  {samples_phys_path}")
    print(f"Saved summary table:     {summary_path}")
    print(f"Saved gate table:        {gate_path}")
    print(f"Saved metadata:          {meta_path}")
    print(f"Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
