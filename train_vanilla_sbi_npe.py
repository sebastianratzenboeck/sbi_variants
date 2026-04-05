#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from data import DEFAULT_INPUT_COLS, DEFAULT_THETA_COLS, load_indices, parse_column_csv
from inference_utils import NormStats
from vanilla_sbi_utils import (
    build_box_prior_from_theta,
    build_zero_imputed_npe_arrays,
    configure_sbi_env,
    maybe_subsample_indices,
    save_pickle,
    save_vanilla_sbi_meta_npz,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a vanilla sbi NPE baseline on zero-imputed cached inputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cache-path", type=str, required=True)
    p.add_argument("--norm-meta-path", type=str, required=True)
    p.add_argument("--train-indices", type=str, required=True)
    p.add_argument("--val-indices", type=str, default=None,
                   help="Optional bookkeeping path; not used directly by sbi training.")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--run-name", type=str, default="vanilla_sbi_npe")
    p.add_argument("--input-columns", type=str, default=",".join(DEFAULT_INPUT_COLS))
    p.add_argument("--theta-columns", type=str, default=",".join(DEFAULT_THETA_COLS))
    p.add_argument("--use-colors", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-train-sims", type=int, default=5_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--density-estimator", type=str, default="nsf", choices=["maf", "nsf", "mdn"])
    p.add_argument("--hidden-features", type=int, default=128)
    p.add_argument("--num-transforms", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--learning-rate", type=float, default=5e-4)
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--stop-after-epochs", type=int, default=20)
    p.add_argument("--max-num-epochs", type=int, default=200)
    p.add_argument("--clip-max-norm", type=float, default=5.0)
    p.add_argument("--prior-margin-frac", type=float, default=0.05)
    p.add_argument("--wandb", action="store_true", default=False)
    p.add_argument("--wandb-project", type=str, default="mock-galaxy-simformer")
    return p.parse_args()


def main() -> None:
    configure_sbi_env()
    from sbi.inference import NPE
    from sbi.neural_nets import posterior_nn

    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    input_columns = parse_column_csv(args.input_columns)
    theta_columns = parse_column_csv(args.theta_columns)
    norm_stats = NormStats(args.norm_meta_path)

    train_rows = load_indices(args.train_indices)
    if train_rows is None or train_rows.size == 0:
        raise ValueError("train-indices file is empty or missing")
    train_rows = maybe_subsample_indices(train_rows, max_rows=args.max_train_sims, seed=args.seed)

    arrays, x_np, theta_np = build_zero_imputed_npe_arrays(
        cache_path=args.cache_path,
        row_indices=train_rows,
        input_columns=input_columns,
        theta_columns=theta_columns,
        use_colors=args.use_colors,
    )
    prior, prior_low, prior_high = build_box_prior_from_theta(
        theta_np,
        device=device,
        margin_frac=args.prior_margin_frac,
    )
    density_estimator = posterior_nn(
        model=args.density_estimator,
        hidden_features=args.hidden_features,
        num_transforms=args.num_transforms,
        z_score_theta="independent",
        z_score_x="independent",
    )

    x = torch.as_tensor(x_np, dtype=torch.float32)
    theta = torch.as_tensor(theta_np, dtype=torch.float32)
    inference = NPE(prior=prior, density_estimator=density_estimator, device=device, show_progress_bars=True)

    t0 = time.time()
    inference.append_simulations(theta, x, data_device="cpu")
    density_estimator_trained = inference.train(
        training_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        stop_after_epochs=args.stop_after_epochs,
        max_num_epochs=args.max_num_epochs,
        clip_max_norm=args.clip_max_norm,
        show_train_summary=True,
    )
    posterior = inference.build_posterior(density_estimator=density_estimator_trained, prior=prior, sample_with="direct")
    elapsed_min = (time.time() - t0) / 60.0

    posterior_path = os.path.join(args.output_dir, f"posterior_{args.run_name}.pkl")
    config_path = os.path.join(args.output_dir, f"vanilla_sbi_config_{args.run_name}.json")
    meta_path = os.path.join(args.output_dir, f"vanilla_sbi_norm_meta_{args.run_name}.npz")
    train_rows_path = os.path.join(args.output_dir, f"vanilla_sbi_train_rows_{args.run_name}.npy")

    save_pickle(posterior, posterior_path)
    np.save(train_rows_path, train_rows.astype(np.int64))
    save_vanilla_sbi_meta_npz(
        path=meta_path,
        norm_stats=norm_stats,
        input_columns=input_columns,
        theta_columns=theta_columns,
        use_colors=args.use_colors,
        color_names=arrays.color_names,
        color_means=arrays.color_means,
        color_stds=arrays.color_stds,
    )

    cfg = {
        **vars(args),
        "device_used": device,
        "train_rows_used": int(train_rows.size),
        "x_dim": int(x_np.shape[1]),
        "theta_dim": int(theta_np.shape[1]),
        "input_columns": input_columns,
        "theta_columns": theta_columns,
        "color_names": list(arrays.color_names or []),
        "prior_low": prior_low.tolist(),
        "prior_high": prior_high.tolist(),
        "posterior_path": posterior_path,
        "meta_path": meta_path,
        "train_rows_path": train_rows_path,
        "elapsed_min": float(elapsed_min),
        "x_representation": "zero_imputed_inputs_only",
    }
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Training finished in {elapsed_min:.1f} min")
    print(f"Saved posterior: {posterior_path}")
    print(f"Saved config:    {config_path}")
    print(f"Saved meta:      {meta_path}")


if __name__ == "__main__":
    main()
