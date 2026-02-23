# SBI Variants (Direct `p(theta | x_obs)`)

This folder contains isolated experiments for a more classical SBI setup:

- Input: observed data tokens (`x_obs`, errors, observed-mask), including arbitrary missingness.
- Output: direct posterior over a fixed `theta` vector.
- No random condition-mask sampling during training.

This repo also keeps the original Simformer training path (`train_mock_galaxy.py` / `train_variant_simformer.py`).

Both train and sample scripts support:

- `--config /path/to/config.json`
- CLI flags override config values.
- Resolved run settings are saved to `posterior_config_<run_name>.json`.

## Implemented variants

- `flow_matching`: conditional FM in theta-space.
- `normalizing_flow`: package-backed conditional flow in theta-space.
  - backend `zuko` (recommended default)
  - backend `nflows` (supported option)

Both variants reuse existing tokenizer/transformer components from:

- `transformer.py`
- `sampling.py`
- `columns.py`

## Default column choices

- Input columns:
  - `sky_ux, sky_uy, sky_uz`
  - all `OBS_COLS` photometric/astrometric observables
- Theta columns:
  - `feh, m_init, logAge, rad, logL, logT, logg, Av`

Override via `--input-columns` and `--theta-columns` (comma-separated).

## Train example

```bash
python sbi_variants/train_sbi_posterior.py \
  --config sbi_variants/configs/train_default.json \
  --run-name sbi_fm_v2 \
  --output-dir /path/to/output \
  --device cuda
```

Equivalent explicit CLI example:

```bash
python sbi_variants/train_sbi_posterior.py \
  --cache-path /path/to/build_arrays_cache.npz \
  --exclude-indices /path/to/test_indices.npy \
  --output-dir /path/to/output \
  --run-name sbi_fm_v1 \
  --method flow_matching \
  --batch-size 4096 \
  --epochs 300 \
  --lr 8e-4 \
  --lr-min 1e-5 \
  --time-prior-exponent 0.0 \
  --use-missingness-context \
  --amp \
  --device cuda
```

Use package-backed flow training:

```bash
python sbi_variants/train_sbi_posterior.py \
  --cache-path /path/to/build_arrays_cache.npz \
  --exclude-indices /path/to/test_indices.npy \
  --output-dir /path/to/output \
  --run-name sbi_nf_v1 \
  --method normalizing_flow \
  --nf-backend zuko \
  --nf-family nsf \
  --batch-size 4096 \
  --epochs 300 \
  --lr 8e-4 \
  --lr-min 1e-5 \
  --use-missingness-context \
  --amp \
  --device cuda
```

One-line NF-theta launcher (uses `configs/train_nf_zuko_theta.json` by default):

```bash
python train_variant_sbi_nf_theta.py \
  --cache-path /path/to/build_arrays_cache.npz \
  --exclude-indices /path/to/test_indices.npy \
  --output-dir /path/to/output \
  --run-name nf_obs_to_theta
```

Compatibility note: `--method realnvp` is kept as an alias for `normalizing_flow`.

## Joint curriculum + importance correction

For rare-regime retention (e.g., young/high-mass) while keeping the natural-population target objective:

- Joint `(logAge, m_init)` curriculum sampling is enabled by default.
- `p/q` importance correction in NF/FM loss is enabled by default.
- Disable explicitly with:
  - `--no-joint-curriculum`
  - `--no-importance-weighting`

Example (NF + zuko):

```bash
python sbi_variants/train_sbi_posterior.py \
  --cache-path /path/to/build_arrays_cache.npz \
  --exclude-indices /path/to/test_indices.npy \
  --output-dir /path/to/output \
  --run-name sbi_nf_joint_curriculum \
  --method normalizing_flow \
  --nf-backend zuko \
  --nf-family nsf \
  --n-bins 25 \
  --n-mass-bins 12 \
  --tau-warmup 20 \
  --tau-max 0.8 \
  --curriculum-epoch-size 0 \
  --importance-weight-min 0.5 \
  --importance-weight-max 2.0 \
  --batch-size 4096 \
  --epochs 300 \
  --lr 8e-4 \
  --lr-min 1e-5 \
  --amp \
  --device cuda
```

## Test holdout and cluster-aware split (NF/FM trainer)

`train_sbi_posterior.py` now supports native test holdout creation:

- `--test-split`: fraction of rows held out as test
- `--test-cluster-frac`: fraction of positive cluster IDs held out as full clusters

When enabled, the trainer writes:

- `output_dir/test_indices.npy`
- `output_dir/test_cluster_ids.npy` (when cluster holdout is used)

The holdout rows are automatically excluded from train/val.

Important precedence:

- If `--exclude-indices` is provided, it overrides/ignores `--test-split` and `--test-cluster-frac`.

Example (NF + zuko, cluster-aware test holdout):

```bash
python train_sbi_posterior.py \
  --config configs/train_nf_zuko_theta.json \
  --cache-path /path/to/build_arrays_cache.npz \
  --output-dir /path/to/output \
  --run-name nf_zuko_cluster_holdout \
  --exclude-indices none \
  --test-split 0.05 \
  --test-cluster-frac 0.2 \
  --wandb \
  --wandb-project mock-galaxy-simformer \
  --device cuda
```

## Original Simformer with weighted joint curriculum

The original Simformer/FM training path now supports the same rare-regime sampling strategy:

- Joint `(logAge, m_init)` curriculum bins (`--n-bins`, `--n-mass-bins`)
- Mixture schedule: `q(bin) = (1-λ)p(bin) + λ/K` with `λ = 1-τ`
- Optional importance correction in loss via clipped `p/q` sample weights

Defaults for `train_mock_galaxy.py`:

- `n_bins=25`
- `n_mass_bins=12`
- `tau_warmup=10`
- `tau_max=0.8`
- `importance_weighting=true`
- `importance_weight_min=0.5`
- `importance_weight_max=2.0`

Run original Simformer with this scheme:

```bash
python train_variant_simformer.py \
  --data-path /path/to/galaxy_field_clusters_all_processed.parquet \
  --output-dir /path/to/output_simformer \
  --run-name simformer_joint_weighted \
  --amp \
  --device cuda
```

Useful overrides:

```bash
--n-bins 25 --n-mass-bins 12 --tau-warmup 10 --tau-max 0.8 \
--importance-weighting --importance-weight-min 0.5 --importance-weight-max 2.0
```

Disable importance correction explicitly:

```bash
--no-importance-weighting
```

## Outputs

Training writes:

- `best_model_<run_name>.pt`
- `posterior_config_<run_name>.json`
- `posterior_history_<run_name>.json`
- `posterior_norm_meta_<run_name>.npz`

The metadata file keeps normalization/column information for later denormalization and sampling scripts.

## Sampling example

From cached arrays:

```bash
python sbi_variants/sample_sbi_posterior.py \
  --config sbi_variants/configs/sample_default.json \
  --run-name sbi_fm_v1 \
  --model-dir /path/to/output \
  --cache-path /path/to/build_arrays_cache.npz \
  --device cuda
```

Equivalent explicit CLI:

```bash
python sbi_variants/sample_sbi_posterior.py \
  --model-dir /path/to/output \
  --run-name sbi_fm_v1 \
  --cache-path /path/to/build_arrays_cache.npz \
  --index-file /path/to/test_indices.npy \
  --num-samples 512 \
  --steps 128 \
  --batch-size 256 \
  --device cuda
```

From an obs CSV/Parquet:

```bash
python sbi_variants/sample_sbi_posterior.py \
  --model-dir /path/to/output \
  --run-name sbi_fm_v1 \
  --obs-file /path/to/stars.parquet \
  --id-column source_id \
  --num-samples 512 \
  --steps 128 \
  --batch-size 256 \
  --device cuda
```

Sampling writes:

- `<prefix>_samples_norm.npy` (`N x S x D`)
- `<prefix>_samples_phys.npy` (`N x S x D`, if `--denormalize`)
- `<prefix>_summary.parquet` (`star_id`, per-theta mean/std)
- `<prefix>_meta.json`
