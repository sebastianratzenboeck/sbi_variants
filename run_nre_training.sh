#!/bin/bash
#SBATCH -J sbi-NRE-70M
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
# Train SBI NRE (balanced) on large stellar catalogs
#
# Resources:
#   200+ GB RAM peak during cache build (DataFrame + arrays + savez)
#   16 CPUs for parquet read and compression throughput
#   1x A100 is sufficient for current NRE architecture
# ---------------------------------------------------------------------------

module purge
module load "${PYTHON_MODULE:-python}"
mamba activate "${CONDA_ENV:-py_gpu_cuda12.4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_PATH="${DATA_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/output_nre}"
CACHE_PATH="${CACHE_PATH:-${OUTPUT_DIR}/build_arrays_cache.npz}"
RUN_NAME="${RUN_NAME:-nre_balanced_tau-0_binfirst_noRowIW_maxEpochs300}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/configs/train_nre_balanced_theta.json}"
TEST_INDEX_PATH="${TEST_INDEX_PATH:-${OUTPUT_DIR}/test_indices.npy}"
TEST_SPLIT="${TEST_SPLIT:-0.05}"
TEST_CLUSTER_FRAC="${TEST_CLUSTER_FRAC:-0.2}"
DEVICE="${DEVICE:-cuda}"
WANDB_PROJECT="${WANDB_PROJECT:-mock-galaxy-simformer}"

# Total target epochs (not additional epochs)
EPOCHS_TOTAL="${EPOCHS_TOTAL:-300}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

RESUME_CKPT="${OUTPUT_DIR}/resume_ratio_checkpoint_${RUN_NAME}.pt"

EXTRA_ARGS=()
if [[ -f "${CACHE_PATH}" ]]; then
  EXTRA_ARGS+=(--no-rebuild-cache)
else
  EXTRA_ARGS+=(--rebuild-cache)
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
fi

if [[ ! -f "${CACHE_PATH}" && -z "${DATA_PATH}" ]]; then
  echo "ERROR: cache not found at ${CACHE_PATH} and DATA_PATH is empty."
  echo "Set DATA_PATH so the cache can be built, or point CACHE_PATH to an existing cache."
  exit 1
fi

if [[ -n "${DATA_PATH}" ]]; then
  EXTRA_ARGS+=(--data-path "${DATA_PATH}")
fi

"${PYTHON_BIN}" train_variant_sbi_nre_theta.py \
  --config "${CONFIG_PATH}" \
  --cache-path "${CACHE_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --epochs "${EPOCHS_TOTAL}" \
  --cluster-id-col cluster_ID \
  --ratio-mask-mode none \
  --use-balanced-loss \
  --bnre-lambda 100 \
  --joint-curriculum \
  --tau-max 0.0 \
  --no-curriculum-importance-weighting \
  --wandb \
  --wandb-project "${WANDB_PROJECT}" \
  --device "${DEVICE}" \
  "${EXTRA_ARGS[@]}"

# Required files for evaluation/diagnostics
echo "Checking artifacts..."
ls -lh \
  "${OUTPUT_DIR}/best_ratio_model_${RUN_NAME}.pt" \
  "${OUTPUT_DIR}/ratio_config_${RUN_NAME}.json" \
  "${OUTPUT_DIR}/ratio_norm_meta_${RUN_NAME}.npz" \
  "${OUTPUT_DIR}/ratio_history_${RUN_NAME}.json" \
  "${OUTPUT_DIR}/resume_ratio_checkpoint_${RUN_NAME}.pt" \
  "${OUTPUT_DIR}/test_indices.npy"

if [[ -f "${OUTPUT_DIR}/test_cluster_ids.npy" ]]; then
  ls -lh "${OUTPUT_DIR}/test_cluster_ids.npy"
fi
