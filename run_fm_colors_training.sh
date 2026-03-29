#!/bin/bash
#SBATCH -J sbi-FM-colors
#SBATCH -p gpu
#SBATCH -t 2-00:00
#SBATCH --mem=350000
#SBATCH -c 16
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -o train_%j.out
#SBATCH -e train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sebastian.ratzenboeck@cfa.harvard.edu
#SBATCH --account=itc_lab

# ---------------------------------------------------------------------------
# Train direct-SBI Flow Matching with color features enabled.
#
# This job:
#   - uses the real processed parquet source
#   - reuses the existing large cache if available
#   - reuses the existing held-out test split if available
#   - writes FM artifacts into a separate output directory
#
# Review the variables below before submitting with:
#   sbatch run_fm_colors_training.sh
# ---------------------------------------------------------------------------

module purge
module load "${PYTHON_MODULE:-python}"
mamba activate "${CONDA_ENV:-py_gpu_cuda12.4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_PATH="${DATA_PATH:-/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/mock_galaxy/galaxy_field_clusters-subset_processed_clusterID.parquet}"
SHARED_CACHE_DIR="${SHARED_CACHE_DIR:-/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/nf_sbi_colors}"
CACHE_PATH="${CACHE_PATH:-${SHARED_CACHE_DIR}/build_arrays_cache.npz}"
TEST_INDEX_PATH="${TEST_INDEX_PATH:-${SHARED_CACHE_DIR}/test_indices.npy}"
OUTPUT_DIR="${OUTPUT_DIR:-/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/fm_sbi_colors}"
RUN_NAME="${RUN_NAME:-fm_colors_tau-0_beta-025_epochSize5M_noDropout_maxEpochs300}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/configs/train_fm_baseline.json}"
DEVICE="${DEVICE:-cuda}"
WANDB_PROJECT="${WANDB_PROJECT:-mock-galaxy-simformer}"
TEST_SPLIT="${TEST_SPLIT:-0.05}"
TEST_CLUSTER_FRAC="${TEST_CLUSTER_FRAC:-0.2}"
EPOCHS_TOTAL="${EPOCHS_TOTAL:-300}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

RESUME_CKPT="${OUTPUT_DIR}/resume_checkpoint_${RUN_NAME}.pt"

EXTRA_ARGS=()

if [[ -f "${CACHE_PATH}" ]]; then
  EXTRA_ARGS+=(--no-rebuild-cache)
  echo "Using existing cache: ${CACHE_PATH}"
else
  EXTRA_ARGS+=(--rebuild-cache)
  echo "Cache not found; will rebuild at ${CACHE_PATH}"
fi

if [[ -f "${RESUME_CKPT}" ]]; then
  EXTRA_ARGS+=(--resume-from "${RESUME_CKPT}")
  echo "Resuming from ${RESUME_CKPT}"
else
  echo "No resume checkpoint found; starting fresh."
fi

if [[ -f "${TEST_INDEX_PATH}" ]]; then
  EXTRA_ARGS+=(--exclude-indices "${TEST_INDEX_PATH}")
  echo "Reusing held-out test indices from ${TEST_INDEX_PATH}"
else
  EXTRA_ARGS+=(--test-split "${TEST_SPLIT}" --test-cluster-frac "${TEST_CLUSTER_FRAC}")
  echo "No existing test-index file found; trainer will generate a new held-out split."
fi

if [[ ! -f "${CACHE_PATH}" && -z "${DATA_PATH}" ]]; then
  echo "ERROR: cache not found at ${CACHE_PATH} and DATA_PATH is empty."
  echo "Set DATA_PATH so the cache can be built, or point CACHE_PATH to an existing cache."
  exit 1
fi

if [[ -n "${DATA_PATH}" ]]; then
  EXTRA_ARGS+=(--data-path "${DATA_PATH}")
fi

cat <<INFO
=== FM colors training job ===
REPO_DIR        = ${REPO_DIR}
DATA_PATH       = ${DATA_PATH}
CACHE_PATH      = ${CACHE_PATH}
TEST_INDEX_PATH = ${TEST_INDEX_PATH}
OUTPUT_DIR      = ${OUTPUT_DIR}
RUN_NAME        = ${RUN_NAME}
CONFIG_PATH     = ${CONFIG_PATH}
DEVICE          = ${DEVICE}
EPOCHS_TOTAL    = ${EPOCHS_TOTAL}
INFO

"${PYTHON_BIN}" train_variant_sbi_fm.py \
  --config "${CONFIG_PATH}" \
  --cache-path "${CACHE_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --epochs "${EPOCHS_TOTAL}" \
  --cluster-id-col cluster_ID \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}"

echo "Checking artifacts..."
ls -lh \
  "${OUTPUT_DIR}/best_model_${RUN_NAME}.pt" \
  "${OUTPUT_DIR}/posterior_config_${RUN_NAME}.json" \
  "${OUTPUT_DIR}/posterior_norm_meta_${RUN_NAME}.npz" \
  "${OUTPUT_DIR}/posterior_history_${RUN_NAME}.json" \
  "${OUTPUT_DIR}/resume_checkpoint_${RUN_NAME}.pt"

if [[ -f "${OUTPUT_DIR}/test_indices.npy" ]]; then
  ls -lh "${OUTPUT_DIR}/test_indices.npy"
fi
if [[ -f "${OUTPUT_DIR}/test_cluster_ids.npy" ]]; then
  ls -lh "${OUTPUT_DIR}/test_cluster_ids.npy"
fi
