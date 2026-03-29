from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

REAL_DATA_PATH = Path(
    "/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/mock_galaxy/galaxy_field_clusters-subset_processed_clusterID.parquet"
)
REAL_MODEL_DIR = Path(
    "/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/nf_sbi_colors"
)
REAL_CACHE_PATH = REAL_MODEL_DIR / "build_arrays_cache.npz"
REAL_TEST_INDEX_PATH = REAL_MODEL_DIR / "test_indices.npy"

NOTEBOOK_BENCHMARK_OUTPUT_DIR = REPO_ROOT / "notebooks" / "outputs" / "transformer_regression_benchmark"
EVAL_NOTEBOOK_OUTPUT_DIR = REAL_MODEL_DIR / "eval_notebook_outputs"

TEST_FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures"
TINY_CACHE_PATH = TEST_FIXTURE_DIR / "build_arrays_cache_tiny.npz"
TINY_INDEX_PATH = TEST_FIXTURE_DIR / "test_indices_tiny.npy"
