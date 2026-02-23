#!/usr/bin/env python
"""
Train SimFormer on galaxy photometric data.

Reads a Parquet/CSV file that already has _obs, _err columns and NaN
patterns (produced by prepare_data.py for mock data, or directly from
real survey data). Handles normalization, model creation, curriculum
scheduling, and training.

Features:
  - Curriculum data scheduling: temperature τ ramps from 0 (uniform age
    sampling) to τ_max (closer to natural age distribution) over training.
  - Cosine annealing LR schedule.
  - Mixed-precision training (AMP) and torch.compile support.
  - Observed/unobserved mask embeddings.

Usage:
  python train_mock_galaxy.py --data-path prepared_data.parquet --epochs 50
  python train_mock_galaxy.py --data-path prepared_data.parquet --epochs 1000 \\
      --tau-warmup 100 --tau-max 0.7 --amp --compile --wandb --device cuda
"""

import argparse
import inspect
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from columns import (
    INTRINSIC_COLS, TRUE_MAG_COLS, OBS_COLS, OBS_ERR_COLS,
    ALL_VALUE_COLS, NUM_NODES, N_INTRINSIC, N_TRUE_MAG,
)
from prepare_data import galactic_to_unitvec
from transformer import Simformer
from simflower import FlowMatchingTrainer
from utils import make_condition_mask_generator
from value_transforms import (
    apply_forward_value_transforms_numpy,
    default_value_transform_metadata,
)


# Observed errors that are non-finite or <=0 are floored so they still enter
# the "real measurement" log-error regime instead of becoming sentinel values.
OBS_ERROR_FLOOR = 1e-6
DEFAULT_CLUSTER_ID_COL = "cluster_ID"


def _simformer_kwargs_from_config(config):
    """Extract only Simformer ctor args from a possibly richer config dict."""
    sig = inspect.signature(Simformer.__init__)
    valid = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in config.items() if k in valid}


def _active_layout(include_true_mags):
    """Return active full-column indices and names for current run."""
    if include_true_mags:
        idx = np.arange(len(ALL_VALUE_COLS), dtype=np.int64)
    else:
        true_start = N_INTRINSIC
        true_end = N_INTRINSIC + N_TRUE_MAG
        idx = np.asarray(
            [i for i in range(len(ALL_VALUE_COLS)) if not (true_start <= i < true_end)],
            dtype=np.int64,
        )
    cols = [ALL_VALUE_COLS[i] for i in idx.tolist()]
    return idx, cols


def _compute_test_split_with_cluster_holdout(
    n_total,
    test_split,
    cluster_ids=None,
    test_cluster_frac=0.0,
    random_state=0,
):
    """Return (trainval_indices, test_indices, heldout_cluster_ids).

    If ``test_cluster_frac > 0`` and cluster IDs are available, first hold out
    that fraction of unique clusters with ID > 0. Then, if needed, fill the
    test set with a random sample from remaining stars to approximately match
    ``test_split`` (fraction or count).
    """
    from sklearn.model_selection import train_test_split

    all_indices = np.arange(n_total, dtype=np.int64)
    rng = np.random.RandomState(random_state)

    # Default random split
    if (cluster_ids is None) or (test_cluster_frac <= 0.0):
        trainval_idx, test_idx = train_test_split(
            all_indices, test_size=test_split, random_state=random_state
        )
        return trainval_idx.astype(np.int64), test_idx.astype(np.int64), np.array([], dtype=np.int64)

    cluster_ids = np.asarray(cluster_ids)
    if cluster_ids.shape[0] != n_total:
        raise ValueError(
            f"cluster_ids length mismatch: got {cluster_ids.shape[0]}, expected {n_total}"
        )

    unique_clusters = np.unique(cluster_ids[cluster_ids > 0])
    if unique_clusters.size == 0:
        print("  WARNING: no positive cluster IDs found; falling back to random test split.")
        trainval_idx, test_idx = train_test_split(
            all_indices, test_size=test_split, random_state=random_state
        )
        return trainval_idx.astype(np.int64), test_idx.astype(np.int64), np.array([], dtype=np.int64)

    n_holdout_clusters = int(round(float(test_cluster_frac) * float(unique_clusters.size)))
    n_holdout_clusters = max(1, min(n_holdout_clusters, int(unique_clusters.size)))
    heldout_clusters = np.sort(
        rng.choice(unique_clusters, size=n_holdout_clusters, replace=False).astype(np.int64)
    )
    is_cluster_test = np.isin(cluster_ids, heldout_clusters)
    test_idx_cluster = all_indices[is_cluster_test]
    remaining_idx = all_indices[~is_cluster_test]

    # Convert test_split into desired test count.
    if isinstance(test_split, float):
        desired_test_count = int(round(test_split * n_total))
    else:
        desired_test_count = int(test_split)
    desired_test_count = max(desired_test_count, len(test_idx_cluster))

    n_needed = desired_test_count - len(test_idx_cluster)
    if n_needed > 0 and len(remaining_idx) > 0:
        n_needed = min(n_needed, len(remaining_idx))
        extra_test = rng.choice(remaining_idx, size=n_needed, replace=False).astype(np.int64)
        test_idx = np.concatenate([test_idx_cluster, extra_test])
    else:
        test_idx = test_idx_cluster

    test_idx = np.unique(test_idx.astype(np.int64))
    trainval_idx = all_indices[~np.isin(all_indices, test_idx)].astype(np.int64)
    return trainval_idx, test_idx, heldout_clusters


# ---------------------------------------------------------------------------
# Data loading & array building
# ---------------------------------------------------------------------------
def load_data(data_path):
    """Load a Parquet or CSV file with expected column structure."""
    print(f'Loading data from {data_path} ...')
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)
    print(f'  Loaded {len(df):,} stars, {len(df.columns)} columns')
    return df


def build_arrays(df, cluster_id_col=DEFAULT_CLUSTER_ID_COL):
    """Build value, error, and observed mask arrays from DataFrame.

    Converts Galactic coordinates (glon, glat) to 3D unit vectors
    (sky_ux, sky_uy, sky_uz) before building arrays.

    Returns:
        values_norm: (N, NUM_NODES) normalized values, NaN filled with 0
        errors_norm: (N, NUM_NODES) standardized log-errors with sentinels:
            -5 for perfectly known (error=0), +5 for unobserved (NaN),
            ~N(0,1) for real measurements
        observed_mask: (N, NUM_NODES) binary mask (1=observed, 0=unobserved)
        means: (NUM_NODES,) per-column means (ignoring NaN)
        stds: (NUM_NODES,) per-column stds (ignoring NaN)
        value_transform_names: (NUM_NODES,) transform names per column
        value_transform_params: (NUM_NODES,) transform params per column
        cluster_ids: (N,) int64 cluster labels for split metadata only (-1 for field/unknown)
        log_err_mean: scalar, mean of log(real errors) for denormalization
        log_err_std: scalar, std of log(real errors) for denormalization
    """
    # Sky unit vector — use precomputed columns if available, else convert glon/glat
    sky_cols = ['sky_ux', 'sky_uy', 'sky_uz']
    if not all(c in df.columns for c in sky_cols):
        if 'glon' not in df.columns or 'glat' not in df.columns:
            raise ValueError(
                "DataFrame must contain either (sky_ux, sky_uy, sky_uz) "
                "or (glon, glat) columns for sky position."
            )
        ux, uy, uz = galactic_to_unitvec(df['glon'].values, df['glat'].values)
        df = df.copy()
        df['sky_ux'] = ux.astype(np.float32)
        df['sky_uy'] = uy.astype(np.float32)
        df['sky_uz'] = uz.astype(np.float32)

    values_raw = df[ALL_VALUE_COLS].values.astype(np.float32)
    value_transform_names, value_transform_params = default_value_transform_metadata(ALL_VALUE_COLS)
    values_work = apply_forward_value_transforms_numpy(
        values_raw,
        transform_names=value_transform_names,
        transform_params=value_transform_params,
    )

    # Error array: 0 for intrinsic/true mag cols, actual errors for obs cols
    errors_raw = np.zeros_like(values_work)
    num_floored_errors = 0
    for i, (obs_col, err_col) in enumerate(zip(OBS_COLS, OBS_ERR_COLS)):
        col_idx = N_INTRINSIC + N_TRUE_MAG + i
        obs_vals = df[obs_col].values.astype(np.float32)
        err_vals = df[err_col].values.astype(np.float32)
        is_obs = ~np.isnan(obs_vals)

        # Missing observations always map to unobserved sentinel (+5) via NaN.
        err_clean = np.where(is_obs, err_vals, np.nan).astype(np.float32)
        bad = is_obs & (~np.isfinite(err_clean) | (err_clean <= 0.0))
        num_floored_errors += int(bad.sum())
        if bad.any():
            err_clean[bad] = OBS_ERROR_FLOOR
        errors_raw[:, col_idx] = err_clean

    # Observed mask: 1 for intrinsic/true mag (always observed), 0/1 for obs cols
    observed_mask = np.ones_like(values_work)
    for i, obs_col in enumerate(OBS_COLS):
        col_idx = N_INTRINSIC + N_TRUE_MAG + i
        observed_mask[:, col_idx] = (~df[obs_col].isna()).values.astype(np.float32)

    # Normalize values
    means = np.nanmean(values_work, axis=0)
    stds = np.nanstd(values_work, axis=0)
    stds[stds < 1e-10] = 1.0

    values_norm = (values_work - means) / stds
    values_norm[np.isnan(values_norm)] = 0.0

    # Cluster labels are metadata for data splitting only (never model inputs).
    if (cluster_id_col is not None) and (cluster_id_col in df.columns):
        c = pd.to_numeric(df[cluster_id_col], errors="coerce").values
        cluster_ids = np.where(np.isfinite(c), c, -1).astype(np.int64)
    else:
        cluster_ids = np.full(values_norm.shape[0], -1, dtype=np.int64)

    # Log-transform + standardize errors with sentinels
    # Three regimes in normalized log-error space:
    #   perfectly known (error=0, e.g. intrinsic cols) → -5
    #   real measurements (error>0, not NaN)           → ~N(0,1)
    #   unobserved (NaN error)                         → +5
    LOG_ERR_PERFECT = -5.0
    LOG_ERR_UNOBS = 5.0

    has_real_error = (errors_raw > 0) & ~np.isnan(errors_raw)
    is_zero_error = (errors_raw == 0)

    log_errors_real = np.log(errors_raw[has_real_error])
    log_err_mean = float(log_errors_real.mean())
    log_err_std = float(log_errors_real.std())
    if log_err_std < 1e-10:
        log_err_std = 1.0

    errors_norm = np.full_like(errors_raw, LOG_ERR_UNOBS)
    errors_norm[has_real_error] = (np.log(errors_raw[has_real_error]) - log_err_mean) / log_err_std
    errors_norm[is_zero_error] = LOG_ERR_PERFECT

    print(f'  Arrays: {values_norm.shape[0]:,} stars x {values_norm.shape[1]} nodes')
    print(f'  Unobserved entries: {(observed_mask == 0).sum():,}')
    if num_floored_errors > 0:
        print(f'  Floored {num_floored_errors:,} observed non-positive/non-finite errors to {OBS_ERROR_FLOOR:g}')
    print(f'  Log-error stats: mean={log_err_mean:.3f}, std={log_err_std:.3f}')
    print(f'  Error regimes: perfect={LOG_ERR_PERFECT}, real~N(0,1), unobs={LOG_ERR_UNOBS}')
    return (
        values_norm,
        errors_norm,
        observed_mask,
        means,
        stds,
        value_transform_names,
        value_transform_params,
        cluster_ids,
        log_err_mean,
        log_err_std,
    )


def save_arrays(
    path,
    values_norm,
    errors_norm,
    observed_mask,
    means,
    stds,
    value_transform_names,
    value_transform_params,
    cluster_ids,
    log_err_mean,
    log_err_std,
):
    """Save preprocessed arrays to compressed npz for fast reloading."""
    print(f'  Saving preprocessed arrays to {path} ...')
    np.savez_compressed(path,
        values_norm=values_norm,
        errors_norm=errors_norm,
        observed_mask=observed_mask,
        means=means, stds=stds,
        value_transform_names=np.asarray(value_transform_names, dtype=object),
        value_transform_params=np.asarray(value_transform_params, dtype=np.float32),
        cluster_ids=np.asarray(cluster_ids, dtype=np.int64),
        log_err_mean=np.array(log_err_mean),
        log_err_std=np.array(log_err_std),
        columns=ALL_VALUE_COLS,
    )
    print(f'  Saved.')


def load_arrays(path):
    """Load preprocessed arrays from npz. Returns same tuple as build_arrays().

    Validates that the cached column layout matches the current column
    definitions (guards against stale caches after column changes).
    """
    print(f'  Loading cached arrays from {path} ...')
    d = np.load(path, allow_pickle=True)

    # Validate column layout matches current definitions
    if 'columns' in d:
        cached_cols = list(d['columns'])
        if cached_cols != ALL_VALUE_COLS:
            raise ValueError(
                f"Cached arrays in {path} have column layout "
                f"{cached_cols[:3]}... ({len(cached_cols)} cols) but current "
                f"column definitions expect {ALL_VALUE_COLS[:3]}... "
                f"({len(ALL_VALUE_COLS)} cols). Delete the cache and re-run."
            )

    values_norm = d['values_norm']
    errors_norm = d['errors_norm']
    observed_mask = d['observed_mask']
    means = d['means']
    stds = d['stds']
    if 'value_transform_names' in d and 'value_transform_params' in d:
        value_transform_names = np.asarray(d['value_transform_names'], dtype=object)
        value_transform_params = np.asarray(d['value_transform_params'], dtype=np.float32)
    else:
        # Legacy cache compatibility. This keeps old runs loadable but does not
        # apply rad/Av positivity transforms. Rebuild cache to enable them.
        value_transform_names = np.asarray(["identity"] * len(ALL_VALUE_COLS), dtype=object)
        value_transform_params = np.zeros(len(ALL_VALUE_COLS), dtype=np.float32)
        print(
            "  WARNING: cache has no value-transform metadata (legacy). "
            "Using identity transforms; delete cache and rebuild to enable rad/Av transforms."
        )
    if 'cluster_ids' in d:
        cluster_ids = np.asarray(d['cluster_ids'], dtype=np.int64)
    else:
        cluster_ids = None
        print(
            "  WARNING: cache has no cluster_ids metadata; cluster-based test holdout will be unavailable "
            "unless you rebuild cache from a table containing cluster_ID."
        )
    log_err_mean = float(d['log_err_mean'])
    log_err_std = float(d['log_err_std'])
    print(f'  Arrays: {values_norm.shape[0]:,} stars x {values_norm.shape[1]} nodes')
    print(f'  Unobserved entries: {(observed_mask == 0).sum():,}')
    return (
        values_norm,
        errors_norm,
        observed_mask,
        means,
        stds,
        value_transform_names,
        value_transform_params,
        cluster_ids,
        log_err_mean,
        log_err_std,
    )


# ---------------------------------------------------------------------------
# Curriculum scheduling
# ---------------------------------------------------------------------------
def compute_joint_age_mass_bin_indices(log_age, m_init, n_age_bins, n_mass_bins):
    """Assign each star to a joint (age, mass) bin."""
    age_edges = np.linspace(log_age.min(), log_age.max() + 1e-6, n_age_bins + 1)
    mass_edges = np.linspace(m_init.min(), m_init.max() + 1e-6, n_mass_bins + 1)

    age_idx = np.digitize(log_age, age_edges) - 1
    age_idx = np.clip(age_idx, 0, n_age_bins - 1)
    mass_idx = np.digitize(m_init, mass_edges) - 1
    mass_idx = np.clip(mass_idx, 0, n_mass_bins - 1)

    joint_idx = age_idx * n_mass_bins + mass_idx
    n_joint_bins = int(n_age_bins * n_mass_bins)
    joint_counts = np.bincount(joint_idx, minlength=n_joint_bins).astype(np.float64)
    return joint_idx, joint_counts


def compute_tau(epoch, total_epochs, tau_max, tau_warmup):
    """Compute temperature τ for the current epoch.

    τ=0 during warmup (uniform-over-active-joint-bins), then linearly ramps
    to τ_max (closer to natural joint-bin frequencies).
    """
    if epoch < tau_warmup:
        return 0.0
    ramp_epochs = total_epochs - tau_warmup
    if ramp_epochs <= 0:
        return tau_max
    progress = (epoch - tau_warmup) / ramp_epochs
    return tau_max * min(progress, 1.0)


def build_epoch_indices(
    bin_idx,
    bin_counts,
    tau,
    cap_per_bin,
    rng=None,
    *,
    reference_bin_count=None,
    importance_weighting=False,
    importance_weight_min=0.5,
    importance_weight_max=2.0,
):
    """Select unique star indices for one epoch via joint-bin mixture sampling.

    Let p(bin) be the natural split distribution over active bins (count > 0),
    and K be the number of active bins.  We define

      q(bin) = (1 - λ) p(bin) + λ / K

    with λ = 1 - τ, so τ=0 is uniform-over-bins and τ=1 is natural.
    Within each bin, stars are sampled uniformly without replacement.

    Args:
        bin_idx:      1-D array, bin assignment for every star in this split.
        bin_counts:   1-D array, number of stars per bin in this split.
        tau:          Curriculum temperature (0 = uniform q, 1 = natural q).
        cap_per_bin:  Per-reference-bin budget used to determine epoch size.
        rng:          numpy random Generator (for reproducibility).
        reference_bin_count: Number of reference bins that defines epoch budget.
            If None, uses number of active bins.
        importance_weighting: If True, compute per-sample weights p(bin)/q(bin),
            where p is the natural bin fraction and q is the sampled epoch fraction.
        importance_weight_min: Lower clip bound for normalized weights.
        importance_weight_max: Upper clip bound for normalized weights.

    Returns:
        (indices, sample_weights)
        indices: 1-D numpy array of shuffled global indices.
        sample_weights: Optional 1-D float32 numpy array aligned with ``indices``.
    """
    if rng is None:
        rng = np.random.default_rng()

    pop_counts = np.asarray(bin_counts, dtype=np.float64)
    active_bins = np.where(pop_counts > 0)[0]
    if active_bins.size == 0:
        return np.empty(0, dtype=np.int64), None

    p = np.zeros_like(pop_counts, dtype=np.float64)
    p[active_bins] = pop_counts[active_bins] / pop_counts[active_bins].sum()

    lam = float(np.clip(1.0 - tau, 0.0, 1.0))
    q = np.zeros_like(pop_counts, dtype=np.float64)
    q[active_bins] = (1.0 - lam) * p[active_bins] + lam * (1.0 / float(active_bins.size))

    # Epoch budget: keep the old scale (~cap_per_bin * n_age_bins) even when
    # switching to many joint bins.
    if reference_bin_count is None:
        ref_bins = int(active_bins.size)
    else:
        ref_bins = int(max(1, min(int(reference_bin_count), int(active_bins.size))))
    epoch_size = int(np.rint(float(cap_per_bin) * float(ref_bins)))
    epoch_size = int(max(int(active_bins.size), min(epoch_size, int(pop_counts.sum()))))
    target_counts = np.maximum(
        1,
        np.rint(epoch_size * q[active_bins]).astype(np.int64),
    )
    max_counts = pop_counts[active_bins].astype(np.int64)
    target_counts = np.minimum(target_counts, max_counts)

    selected = []
    selected_bins = []
    selected_counts = np.zeros_like(bin_counts, dtype=np.int64)
    for b, n_select in zip(active_bins.tolist(), target_counts.tolist()):
        if n_select <= 0:
            continue
        members = np.where(bin_idx == b)[0]
        chosen = rng.choice(members, size=n_select, replace=False)
        selected.append(chosen)
        selected_bins.append(np.full(n_select, b, dtype=np.int64))
        selected_counts[b] = n_select

    if not selected:
        return np.empty(0, dtype=np.int64), None

    indices = np.concatenate(selected).astype(np.int64)
    chosen_bins = np.concatenate(selected_bins).astype(np.int64)
    perm = rng.permutation(len(indices))
    indices = indices[perm]
    chosen_bins = chosen_bins[perm]

    sample_weights = None
    if importance_weighting:
        q_emp = selected_counts.astype(np.float64)
        q_emp = q_emp / max(q_emp.sum(), 1.0)
        per_bin_w = np.zeros_like(p)
        active = q_emp > 0
        per_bin_w[active] = p[active] / q_emp[active]
        sample_weights = per_bin_w[chosen_bins].astype(np.float32)

        # Normalize to mean 1 and clip to control variance.
        sample_weights /= max(sample_weights.mean(), 1e-8)
        sample_weights = np.clip(
            sample_weights,
            float(importance_weight_min),
            float(importance_weight_max),
        ).astype(np.float32)
        sample_weights /= max(sample_weights.mean(), 1e-8)

    return indices, sample_weights


def make_epoch_callback(bin_idx_train, bin_counts_train,
                        bin_idx_val, bin_counts_val,
                        tau_max, tau_warmup,
                        cap_per_bin=1000,
                        reference_bin_count=None,
                        use_importance_weights=False,
                        importance_weight_min=0.5,
                        importance_weight_max=2.0,
                        use_wandb=False):
    """Create the epoch callback that rebuilds train/val index arrays each epoch."""
    rng = np.random.default_rng(42)

    def epoch_callback(trainer, epoch, total_epochs):
        tau = compute_tau(epoch, total_epochs, tau_max, tau_warmup)
        lam = float(np.clip(1.0 - tau, 0.0, 1.0))

        train_indices, train_weights = build_epoch_indices(
            bin_idx_train,
            bin_counts_train,
            tau,
            cap_per_bin,
            rng,
            reference_bin_count=reference_bin_count,
            importance_weighting=use_importance_weights,
            importance_weight_min=importance_weight_min,
            importance_weight_max=importance_weight_max,
        )
        val_indices, val_weights = build_epoch_indices(
            bin_idx_val,
            bin_counts_val,
            tau,
            cap_per_bin,
            rng,
            reference_bin_count=reference_bin_count,
            importance_weighting=use_importance_weights,
            importance_weight_min=importance_weight_min,
            importance_weight_max=importance_weight_max,
        )

        trainer.set_epoch_indices(train_indices, sample_weights=train_weights)
        trainer.set_val_epoch_indices(val_indices, sample_weights=val_weights)

        n_train_steps = len(train_indices) // trainer.batch_size
        n_val_steps = len(val_indices) // trainer.batch_size
        current_lr = trainer.optimizer.param_groups[0]['lr']
        active_train_bins = int((bin_counts_train > 0).sum())
        active_val_bins = int((bin_counts_val > 0).sum())
        print(f'  [Curriculum] epoch={epoch+1}, τ={tau:.3f}, λ={lam:.3f}, '
              f'active_bins(train/val)={active_train_bins}/{active_val_bins}, '
              f'train_stars={len(train_indices):,} ({n_train_steps} steps), '
              f'val_stars={len(val_indices):,} ({n_val_steps} steps), '
              f'lr={current_lr:.2e}')
        if use_importance_weights and train_weights is not None:
            print(
                f'    importance weights (train): '
                f'min={train_weights.min():.3f}, '
                f'mean={train_weights.mean():.3f}, '
                f'max={train_weights.max():.3f}'
            )

        if use_wandb:
            import wandb
            log = {'tau': tau, 'learning_rate': current_lr,
                   'mixture_lambda': lam,
                   'epoch': epoch + 1,
                   'train_stars': len(train_indices),
                   'val_stars': len(val_indices),
                   'active_bins_train': active_train_bins,
                   'active_bins_val': active_val_bins}
            if use_importance_weights and train_weights is not None:
                log.update({
                    'train_importance_weight_min': float(train_weights.min()),
                    'train_importance_weight_mean': float(train_weights.mean()),
                    'train_importance_weight_max': float(train_weights.max()),
                })
            wandb.log(log)

    return epoch_callback


def make_post_epoch_callback(bin_idx_val, bin_counts_val,
                             tau_max, cap_per_bin=1000,
                             young_val_indices=None,
                             reference_bin_count=None,
                             use_importance_weights=False,
                             importance_weight_min=0.5,
                             importance_weight_max=2.0,
                             use_wandb=False):
    """Create post-epoch callback for secondary validation metrics.

    Runs two additional validation passes each epoch (neither affects
    early stopping):

    1. **val_loss_taumax** — validation at τ=τ_max, giving early
       visibility into final-distribution performance.
    2. **val_loss_young** — validation on young stars only
       (logAge < threshold), tracking the science-critical population.

    Args:
        young_val_indices: 1-D numpy array of val-local indices for
            young stars.  Precomputed in ``main()`` as
            ``np.where(log_age_val < 7.7)[0]``.  If None or too few
            stars (< batch_size), the young-star validation is skipped.
    """
    rng = np.random.default_rng(123)  # separate RNG from main curriculum

    def post_epoch_callback(trainer, epoch, total_epochs):
        # --- τ_max validation ---
        val_indices_taumax, val_weights_taumax = build_epoch_indices(
            bin_idx_val,
            bin_counts_val,
            tau_max,
            cap_per_bin,
            rng,
            reference_bin_count=reference_bin_count,
            importance_weighting=use_importance_weights,
            importance_weight_min=importance_weight_min,
            importance_weight_max=importance_weight_max,
        )
        trainer.set_val_epoch_indices(
            val_indices_taumax,
            sample_weights=val_weights_taumax,
        )
        val_loss_taumax = trainer.validate()

        if val_loss_taumax is not None:
            print(f'  [\u03c4_max val] val_loss_taumax = {val_loss_taumax:.6f}')

        # --- Young-star validation ---
        val_loss_young = None
        if young_val_indices is not None and len(young_val_indices) >= trainer.batch_size:
            shuffled = young_val_indices.copy()
            rng.shuffle(shuffled)
            trainer.set_val_epoch_indices(shuffled, sample_weights=None)
            val_loss_young = trainer.validate()
            if val_loss_young is not None:
                print(f'  [young val] val_loss_young = {val_loss_young:.6f}')

        if use_wandb:
            import wandb
            log = {'epoch': epoch + 1}
            if val_loss_taumax is not None:
                log['val_loss_taumax'] = val_loss_taumax
            if val_loss_young is not None:
                log['val_loss_young'] = val_loss_young
            wandb.log(log)

    return post_epoch_callback


# ---------------------------------------------------------------------------
# Model creation
# ---------------------------------------------------------------------------
def build_default_survey_obs_groups(columns):
    """Return local node-index groups for survey-level missingness summaries."""
    groups = {
        "gaia": [],
        "2mass": [],
        "wise": [],
        "ps1": [],
        "decam": [],
    }
    for idx, col in enumerate(columns):
        if col not in OBS_COLS:
            continue
        if col.startswith("GAIA_") or col == "parallax_obs":
            groups["gaia"].append(idx)
        elif col.startswith("2MASS_"):
            groups["2mass"].append(idx)
        elif col.startswith("WISE_"):
            groups["wise"].append(idx)
        elif col.startswith("PS1_"):
            groups["ps1"].append(idx)
        elif col.startswith("CTIO_DECam"):
            groups["decam"].append(idx)
    # Fixed order for stable feature layout.
    return [groups["gaia"], groups["2mass"], groups["wise"], groups["ps1"], groups["decam"]]


# Default model architecture — saved as JSON for reproducible inference
MODEL_CONFIG = dict(
    num_nodes=NUM_NODES,
    dim_value=24,
    dim_id=24,
    dim_condition=16,
    value_calibration_type='scalar_film',
    dim_error=16,
    error_embed_type='mlp_regime',
    dim_observed=8,
    use_missingness_context=True,
    obs_start_idx=N_INTRINSIC + N_TRUE_MAG,
    survey_obs_groups=build_default_survey_obs_groups(ALL_VALUE_COLS),
    missingness_context_hidden_dim=64,
    include_true_mags=True,  # training/data option (not a Simformer constructor arg)
    attn_embed_dim=128,
    num_heads=8,
    num_layers=4,
    widening_factor=4,
    time_embed_dim=64,
    dropout=0.05,
    time_prior_exponent=1.0,  # trainer option: t ~ U^(1/(1+alpha))
)


def create_model(config=None):
    """Create SimFormer model from config dict (defaults to MODEL_CONFIG)."""
    config = config or MODEL_CONFIG
    model = Simformer(**_simformer_kwargs_from_config(config))
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model created: {n_params:,} parameters')
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _build_arg_parser():
    p = argparse.ArgumentParser(
        description='Train SimFormer on galaxy data with curriculum scheduling.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--config', type=str, default=None,
                   help='Optional JSON config file for training args. '
                        'CLI flags override config values.')

    # Data
    p.add_argument('--data-path', type=str, default=None,
                   help='Path to prepared Parquet/CSV with _obs/_err columns '
                        '(optional if cache exists)')
    p.add_argument('--cache-path', type=str, default=None,
                   help='Optional path to build_arrays_cache.npz. If provided, '
                        'load/build cache at this location instead of output-dir.')
    p.add_argument('--output-dir', type=str, default='./output',
                   help='Directory for checkpoints and normalization stats')
    p.add_argument('--cluster-id-col', type=str, default=DEFAULT_CLUSTER_ID_COL,
                   help='Column name used only for cluster-aware test holdout (metadata only).')
    p.add_argument('--test-cluster-frac', type=float, default=0.0,
                   help='If >0, hold out this fraction of unique cluster IDs (>0) as full clusters in test set.')

    # Model architecture
    p.add_argument('--model-config', type=str, default=None,
                   help='Path to JSON file with Simformer architecture config. '
                        'Overrides built-in MODEL_CONFIG defaults.')

    # Training
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-3,
                   help='Initial learning rate')
    p.add_argument('--lr-min', type=float, default=1e-5,
                   help='Minimum LR for cosine annealing')
    p.add_argument('--inner-loop-size', type=int, default=500,
                   help='Training steps per epoch')
    p.add_argument('--patience', type=int, default=20,
                   help='Early stopping patience (epochs)')
    p.add_argument('--test-split', type=float, default=0.1,
                   help='Fraction of data held out as test set (saved to test_indices.npy)')
    p.add_argument('--val-split', type=float, default=0.15)
    p.add_argument('--dense-ratio', type=float, default=0.8,
                   help='Fraction of batch with fully connected edge masks (rest get random sparsity)')
    p.add_argument('--time-prior-exponent', type=float, default=None,
                   help='Alpha in t~U^(1/(1+alpha)); if unset, use model-config time_prior_exponent or 1.0')

    # Curriculum scheduling
    p.add_argument('--n-bins', type=int, default=25,
                   help='Number of logAge bins for joint (age,mass) curriculum weighting')
    p.add_argument('--n-mass-bins', type=int, default=12,
                   help='Number of mass bins for joint (age,mass) curriculum weighting')
    p.add_argument('--tau-max', type=float, default=0.8,
                   help='Max τ for q=(1-λ)p+λ/K with λ=1-τ (τ=0 uniform, τ=1 natural)')
    p.add_argument('--tau-warmup', type=int, default=10,
                   help='Epochs to stay at τ=0 before ramping')
    p.add_argument('--cap-per-bin', type=int, default=100000,
                   help='Epoch budget scale: approx cap_per_bin * n_bins stars per epoch')
    p.add_argument('--importance-weighting', action=argparse.BooleanOptionalAction, default=True,
                   help='Apply per-sample importance weights p(bin)/q(bin) for curriculum-selected batches.')
    p.add_argument('--importance-weight-min', type=float, default=0.5,
                   help='Lower clip bound for normalized curriculum importance weights.')
    p.add_argument('--importance-weight-max', type=float, default=2.0,
                   help='Upper clip bound for normalized curriculum importance weights.')

    # Performance
    p.add_argument('--amp', action='store_true', default=False,
                   help='Enable mixed-precision training (FP16). Recommended for GPU.')
    p.add_argument('--grad-clip-norm', type=float, default=1.0,
                   help='Max gradient norm for clipping (stabilizes training)')
    p.add_argument('--weight-decay', type=float, default=1e-4,
                   help='AdamW weight decay for regularization')
    p.add_argument('--compile', action='store_true', default=False,
                   help='Use torch.compile for kernel fusion (PyTorch 2.x)')

    # Run naming
    p.add_argument('--run-name', type=str, default='default',
                   help='Run name for checkpoint file (best_model_{name}.pt) and wandb run')

    # WandB
    p.add_argument('--wandb', action='store_true', default=False,
                   help='Enable WandB logging')
    p.add_argument('--wandb-project', type=str, default='mock-galaxy-simformer')

    # Device / seed
    p.add_argument('--device', type=str, default=None,
                   help='Device (auto-detect if not set)')
    p.add_argument('--seed', type=int, default=42)

    return p


def _load_config_defaults(config_path, parser):
    with open(config_path) as f:
        cfg = json.load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f'Config at {config_path} must be a JSON object (dict).')

    valid = {a.dest for a in parser._actions}
    unknown = sorted(k for k in cfg.keys() if k not in valid)
    if unknown:
        raise ValueError(
            f'Unknown config keys in {config_path}: {unknown[:8]}'
            + ('...' if len(unknown) > 8 else '')
        )
    return cfg


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=str, default=None)
    pre_args, _ = pre.parse_known_args()

    parser = _build_arg_parser()
    if pre_args.config:
        cfg = _load_config_defaults(pre_args.config, parser)
        parser.set_defaults(**cfg)
    return parser.parse_args()


def main():
    args = parse_args()

    if args.n_bins <= 0:
        raise ValueError(f'n_bins must be > 0, got {args.n_bins}')
    if args.n_mass_bins <= 0:
        raise ValueError(f'n_mass_bins must be > 0, got {args.n_mass_bins}')
    if args.importance_weight_min <= 0:
        raise ValueError(
            f'importance_weight_min must be > 0, got {args.importance_weight_min}'
        )
    if args.importance_weight_max < args.importance_weight_min:
        raise ValueError(
            f'importance_weight_max ({args.importance_weight_max}) must be >= '
            f'importance_weight_min ({args.importance_weight_min})'
        )

    # Seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    # Persist resolved device into args for logging/reproducibility.
    args.device = device
    print(f'Using device: {device}')

    # Output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load model/data config ----
    if args.model_config:
        with open(args.model_config) as f:
            model_config = json.load(f)
        print(f'Loaded model config from {args.model_config}')
    else:
        model_config = MODEL_CONFIG.copy()

    # Resolve FM time-prior exponent (alpha): CLI overrides config.
    if args.time_prior_exponent is not None:
        time_prior_exponent = float(args.time_prior_exponent)
    else:
        time_prior_exponent = float(model_config.get("time_prior_exponent", 1.0))
    if time_prior_exponent <= -1.0:
        raise ValueError(
            f"time_prior_exponent must be > -1.0, got {time_prior_exponent}"
        )
    model_config["time_prior_exponent"] = time_prior_exponent

    include_true_mags = bool(model_config.get("include_true_mags", True))
    active_full_idx, active_columns = _active_layout(include_true_mags=include_true_mags)
    print(
        f"Active node layout: {len(active_columns)} / {len(ALL_VALUE_COLS)} columns "
        f"(include_true_mags={include_true_mags})"
    )

    # ---- Load data ----
    print('\n--- Data ---')
    cache_path = args.cache_path or os.path.join(args.output_dir, 'build_arrays_cache.npz')
    if os.path.exists(cache_path):
        (
            data,
            data_errors,
            data_observed_mask,
            means,
            stds,
            value_transform_names,
            value_transform_params,
            cluster_ids,
            log_err_mean,
            log_err_std,
        ) = load_arrays(cache_path)
    else:
        if args.data_path is None:
            raise ValueError(
                "No cache found and --data-path not provided. "
                "Pass --data-path to build arrays or --cache-path to an existing cache."
            )
        df = load_data(args.data_path)
        (
            data,
            data_errors,
            data_observed_mask,
            means,
            stds,
            value_transform_names,
            value_transform_params,
            cluster_ids,
            log_err_mean,
            log_err_std,
        ) = build_arrays(df, cluster_id_col=args.cluster_id_col)
        save_arrays(
            cache_path,
            data,
            data_errors,
            data_observed_mask,
            means,
            stds,
            value_transform_names,
            value_transform_params,
            cluster_ids,
            log_err_mean,
            log_err_std,
        )

    # Optional dimensionality reduction: drop true-mag nodes from state space.
    if active_full_idx.shape[0] != data.shape[1]:
        data = data[:, active_full_idx]
        data_errors = data_errors[:, active_full_idx]
        data_observed_mask = data_observed_mask[:, active_full_idx]
        means = means[active_full_idx]
        stds = stds[active_full_idx]
        value_transform_names = np.asarray(value_transform_names, dtype=object)[active_full_idx]
        value_transform_params = np.asarray(value_transform_params, dtype=np.float32)[active_full_idx]

    # Save normalization stats (small file for standalone inference scripts)
    norm_path = os.path.join(args.output_dir, 'norm_stats.npz')
    np.savez(
        norm_path,
        means=means,
        stds=stds,
        columns=np.asarray(active_columns, dtype=object),
        value_transform_names=np.asarray(value_transform_names, dtype=object),
        value_transform_params=np.asarray(value_transform_params, dtype=np.float32),
        log_err_mean=log_err_mean,
        log_err_std=log_err_std,
    )
    print(f'  Normalization stats saved to {norm_path}')

    # ---- Hold out test set ----
    from sklearn.model_selection import train_test_split
    if not (0.0 <= args.test_cluster_frac <= 1.0):
        raise ValueError(f"--test-cluster-frac must be in [0,1], got {args.test_cluster_frac}")
    if args.test_cluster_frac > 0.0 and cluster_ids is None:
        raise ValueError(
            "Cluster-based holdout requested but cluster IDs are unavailable. "
            "Rebuild cache from data containing cluster_ID (or disable --test-cluster-frac)."
        )
    n_total = len(data)
    trainval_indices, test_indices, heldout_cluster_ids = _compute_test_split_with_cluster_holdout(
        n_total=n_total,
        test_split=args.test_split,
        cluster_ids=cluster_ids,
        test_cluster_frac=args.test_cluster_frac,
        random_state=0,
    )
    # Save test indices for later evaluation
    test_idx_path = os.path.join(args.output_dir, 'test_indices.npy')
    np.save(test_idx_path, test_indices)
    print(f'  Test set: {len(test_indices):,} stars (saved to {test_idx_path})')
    if heldout_cluster_ids.size > 0:
        holdout_path = os.path.join(args.output_dir, 'test_cluster_ids.npy')
        np.save(holdout_path, heldout_cluster_ids)
        n_cluster_stars = int(np.isin(cluster_ids, heldout_cluster_ids).sum())
        print(
            f'  Cluster holdout: {len(heldout_cluster_ids)} clusters '
            f'({n_cluster_stars:,} stars) saved to {holdout_path}'
        )

    # Keep only train+val portion
    data = data[trainval_indices]
    data_errors = data_errors[trainval_indices]
    data_observed_mask = data_observed_mask[trainval_indices]
    if cluster_ids is not None:
        cluster_ids = cluster_ids[trainval_indices]
    print(f'  Train+val: {len(data):,} stars')

    # ---- Joint age-mass bins for curriculum weighting ----
    # Recover physical logAge and m_init from normalized data.
    logage_idx = active_columns.index('logAge')
    m_init_idx = active_columns.index('m_init')
    log_age = data[:, logage_idx] * stds[logage_idx] + means[logage_idx]
    m_init = data[:, m_init_idx] * stds[m_init_idx] + means[m_init_idx]
    bin_idx_all, _ = compute_joint_age_mass_bin_indices(
        log_age=log_age,
        m_init=m_init,
        n_age_bins=args.n_bins,
        n_mass_bins=args.n_mass_bins,
    )
    n_joint_bins = int(args.n_bins * args.n_mass_bins)

    # Single authoritative train/val split — used for BOTH curriculum indices
    # and the trainer's data arrays (avoids double-split index misalignment).
    n_trainval = len(data)
    trainval_local = np.arange(n_trainval)
    train_indices, val_indices = train_test_split(
        trainval_local, test_size=args.val_split, random_state=42
    )
    bin_idx_train = bin_idx_all[train_indices]
    bin_idx_val = bin_idx_all[val_indices]
    bin_counts_train = np.bincount(bin_idx_train, minlength=n_joint_bins).astype(np.float64)
    bin_counts_val = np.bincount(bin_idx_val, minlength=n_joint_bins).astype(np.float64)
    print(
        f'  Curriculum bins: age={args.n_bins}, mass={args.n_mass_bins}, '
        f'joint={n_joint_bins} '
        f'(active train/val={(bin_counts_train > 0).sum()}/{(bin_counts_val > 0).sum()})'
    )
    print(f'  Curriculum split: train={len(train_indices):,}, val={len(val_indices):,}')

    # Split data arrays to match curriculum partition
    train_data = data[train_indices]
    val_data = data[val_indices]
    train_errors = data_errors[train_indices]
    val_errors = data_errors[val_indices]
    train_observed = data_observed_mask[train_indices]
    val_observed = data_observed_mask[val_indices]

    # ---- Model ----
    print('\n--- Model ---')
    model_config = model_config.copy()
    model_config['include_true_mags'] = include_true_mags
    model_config['active_columns'] = list(active_columns)
    model_config['active_full_indices'] = active_full_idx.tolist()
    model_config['num_nodes'] = len(active_columns)
    if model_config.get("use_missingness_context", False):
        obs_local_idx = [i for i, c in enumerate(active_columns) if c in OBS_COLS]
        obs_start = min(obs_local_idx) if obs_local_idx else len(active_columns)
        model_config['obs_start_idx'] = obs_start
        model_config['survey_obs_groups'] = build_default_survey_obs_groups(active_columns)

    model = create_model(config=model_config)
    # Save resolved config for reproducible inference
    config_path = os.path.join(args.output_dir, f'model_config_{args.run_name}.json')
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    print(f'  Model config saved to {config_path}')
    # Save resolved training args for easy reruns/reproducibility.
    args_out = vars(args).copy()
    args_out['resolved_cache_path'] = cache_path
    args_out['resolved_time_prior_exponent'] = time_prior_exponent
    args_out['resolved_include_true_mags'] = include_true_mags
    args_path = os.path.join(args.output_dir, f'training_args_{args.run_name}.json')
    with open(args_path, 'w') as f:
        json.dump(args_out, f, indent=2)
    print(f'  Training args saved to {args_path}')
    print(f'  time_prior_exponent (alpha) = {time_prior_exponent}')
    if args.compile:
        print('  Compiling model with torch.compile ...')
        model = torch.compile(model)

    # ---- Condition mask generator ----
    obs_indices = [i for i, c in enumerate(active_columns) if c in OBS_COLS]
    sky_uvec_idx = [active_columns.index(c) for c in ('sky_ux', 'sky_uy', 'sky_uz') if c in active_columns]
    cond_gen = make_condition_mask_generator(
        batch_size=args.batch_size,
        num_features=len(active_columns),
        percent=(0.0, 1.0),
        allowed_idx=obs_indices,
        always_on_idx=sky_uvec_idx,
        device=device,
    )

    # ---- Trainer ----
    print('\n--- Trainer ---')
    ckpt_path = os.path.join(args.output_dir, f'best_model_{args.run_name}.pt')
    trainer = FlowMatchingTrainer(
        model=model,
        data=train_data,
        data_errors=train_errors,
        data_observed_mask=train_observed,
        val_data=val_data,
        val_errors=val_errors,
        val_observed_mask=val_observed,
        condition_mask_generator=cond_gen,
        batch_size=args.batch_size,
        lr=args.lr,
        time_prior_exponent=time_prior_exponent,
        inner_train_loop_size=args.inner_loop_size,
        early_stopping_patience=args.patience,
        val_split=0,  # already split above
        dense_ratio=args.dense_ratio,
        use_amp=args.amp,
        grad_clip_norm=args.grad_clip_norm,
        weight_decay=args.weight_decay,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_config=vars(args),
        wandb_run_name=args.run_name,
        checkpoint_path=ckpt_path,
        device=device,
    )
    print('  Trainer initialized.')

    # ---- LR scheduler ----
    lr_scheduler = CosineAnnealingLR(
        trainer.optimizer, T_max=args.epochs, eta_min=args.lr_min
    )

    # ---- Epoch callback (curriculum) ----
    epoch_cb = make_epoch_callback(
        bin_idx_train=bin_idx_train,
        bin_counts_train=bin_counts_train,
        bin_idx_val=bin_idx_val,
        bin_counts_val=bin_counts_val,
        tau_max=args.tau_max,
        tau_warmup=args.tau_warmup,
        cap_per_bin=args.cap_per_bin,
        reference_bin_count=args.n_bins,
        use_importance_weights=args.importance_weighting,
        importance_weight_min=args.importance_weight_min,
        importance_weight_max=args.importance_weight_max,
        use_wandb=args.wandb,
    )

    # ---- Post-epoch callback (τ_max + young-star validation) ----
    # Identify young val stars (logAge < 7.7) for dedicated validation
    log_age_val = log_age[val_indices]
    young_mask = log_age_val < 7.8
    young_val_indices = np.where(young_mask)[0]  # indices into val_data
    print(f'  Young val stars (logAge < 7.8): {len(young_val_indices):,}')

    post_epoch_cb = make_post_epoch_callback(
        bin_idx_val=bin_idx_val,
        bin_counts_val=bin_counts_val,
        tau_max=args.tau_max,
        cap_per_bin=args.cap_per_bin,
        young_val_indices=young_val_indices,
        reference_bin_count=args.n_bins,
        use_importance_weights=args.importance_weighting,
        importance_weight_min=args.importance_weight_min,
        importance_weight_max=args.importance_weight_max,
        use_wandb=args.wandb,
    )

    # ---- Train ----
    print(f'\n--- Training ({args.epochs} epochs) ---')
    t0 = time.time()
    best_model = trainer.fit(
        epochs=args.epochs,
        verbose=True,
        epoch_callback=epoch_cb,
        post_epoch_callback=post_epoch_cb,
        lr_scheduler=lr_scheduler,
    )
    elapsed = time.time() - t0
    print(f'\nTraining completed in {elapsed / 60:.1f} minutes.')

    # ---- Save model ----
    final_ckpt = os.path.join(args.output_dir, f'best_model_{args.run_name}.pt')
    torch.save(best_model.state_dict(), final_ckpt)
    print(f'Best model saved to {final_ckpt}')

    # ---- Summary ----
    print('\n--- Summary ---')
    print(f'  Data:       {data.shape[0]:,} stars, {data.shape[1]} nodes')
    print(f'  Model:      {sum(p.numel() for p in best_model.parameters()):,} params')
    print(f'  Epochs:     {args.epochs}')
    print(f'  Curriculum: τ_warmup={args.tau_warmup}, τ_max={args.tau_max}')
    print(
        f'  Importance: enabled={args.importance_weighting}, '
        f'clip=[{args.importance_weight_min}, {args.importance_weight_max}]'
    )
    print(f'  LR:         {args.lr} → {args.lr_min} (cosine)')
    print(f'  AMP:        {args.amp}')
    print(f'  Compiled:   {args.compile}')
    print(f'  Output:     {args.output_dir}/')


if __name__ == '__main__':
    main()
