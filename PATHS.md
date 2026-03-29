# Paths

This file inventories the concrete dataset, cache, index, model, and output paths currently referenced by notebooks, configs, and helper scripts in this repository.

## Real Paths In Use

Primary real paths currently referenced by the repo:

- Raw data parquet: `/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/mock_galaxy/galaxy_field_clusters-subset_processed_clusterID.parquet`
- Model/output directory: `/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/nf_sbi_colors`
- Cache file: `/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/nf_sbi_colors/build_arrays_cache.npz`
- Test index file: `/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/nf_sbi_colors/test_indices.npy`
- Benchmark notebook output directory: `/n/home12/sratzenboeck/code/sbi_variants/notebooks/outputs/transformer_regression_benchmark`

These paths are encoded centrally in [project_paths.py](/n/home12/sratzenboeck/code/sbi_variants/project_paths.py).

## Optional Environment Variables

Also referenced in [notebooks/transformer_regression_benchmark.ipynb](/n/home12/sratzenboeck/code/sbi_variants/notebooks/transformer_regression_benchmark.ipynb):

- `SBI_CACHE_PATH`
- `SBI_TEST_INDEX_PATH`

If set, those notebook overrides take precedence over the shared defaults.

## Notebook Outputs

From [notebooks/transformer_regression_benchmark.ipynb](/n/home12/sratzenboeck/code/sbi_variants/notebooks/transformer_regression_benchmark.ipynb):

- `OUTPUT_DIR / 'regression_results.csv'`
- `OUTPUT_DIR / 'transformer_training_history.csv'`

From [notebooks/eval_star_posteriors.ipynb](/n/home12/sratzenboeck/code/sbi_variants/notebooks/eval_star_posteriors.ipynb):

- `MODEL_DIR / 'eval_notebook_outputs'`
- `posterior_summary_selected_stars.csv`
- `tarp_curve_selected_stars.csv`
- `tarp_summary_selected_stars.json`

## Test Fixture Paths

A tiny reusable cache slice for tests lives at:

- [tests/fixtures/build_arrays_cache_tiny.npz](/n/home12/sratzenboeck/code/sbi_variants/tests/fixtures/build_arrays_cache_tiny.npz)
- [tests/fixtures/test_indices_tiny.npy](/n/home12/sratzenboeck/code/sbi_variants/tests/fixtures/test_indices_tiny.npy)

These are derived from the real cache/test split but small enough for fast local tests.

## Legacy Path

An older machine-specific repo path had existed in one notebook and has been replaced by the shared local repo root:

- `/n/home12/sratzenboeck/code/sbi_variants`
