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
module load python
mamba activate py_gpu_cuda12.4

DATA_PATH="/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/mock_galaxy/galaxy_field_clusters-subset_processed_clusterID.parquet"
OUTPUT_DIR="/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/nre_sbi_colors"
CACHE_PATH="${OUTPUT_DIR}/build_arrays_cache.npz"
RUN_NAME="nre_balanced_tau-0_binfirst_noRowIW_maxEpochs300"
CONFIG_PATH="configs/train_nre_balanced_theta.json"

# Total target epochs (not additional epochs)
EPOCHS_TOTAL=0

mkdir -p "${OUTPUT_DIR}"
cd "$HOME/code/sbi_variants"

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

python train_variant_sbi_nre_theta.py \
  --config "${CONFIG_PATH}" \
  --data-path "${DATA_PATH}" \
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
  --wandb-project mock-galaxy-simformer \
  --device cuda \
  "${EXTRA_ARGS[@]}"

# Required files for evaluation/diagnostics
echo "Checking artifacts..."
ls -lh \
  "${OUTPUT_DIR}/best_ratio_model_${RUN_NAME}.pt" \
  "${OUTPUT_DIR}/ratio_config_${RUN_NAME}.json" \
  "${OUTPUT_DIR}/ratio_norm_meta_${RUN_NAME}.npz" \
  "${OUTPUT_DIR}/ratio_history_${RUN_NAME}.json" \
  "${OUTPUT_DIR}/resume_ratio_checkpoint_${RUN_NAME}.pt"
