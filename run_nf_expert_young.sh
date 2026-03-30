#!/bin/bash
#SBATCH -J nf-young
#SBATCH -p gpu
#SBATCH -t 2-00:00
#SBATCH --mem=350000
#SBATCH -c 16
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -o nf_expert_young_%j.out
#SBATCH -e nf_expert_young_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sebastian.ratzenboeck@cfa.harvard.edu
#SBATCH --account=itc_lab

module purge
module load python
mamba activate py_gpu_cuda12.4

CACHE_PATH="/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/nf_sbi_colors/build_arrays_cache.npz"
DATA_PATH="/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/mock_galaxy/galaxy_field_clusters-subset_processed_clusterID.parquet"
SPLIT_DIR="/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/nf_sbi_colors/age_regime_splits"
OUTPUT_DIR="/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/age_mixture_nf_experts/young"
RUN_NAME="nf_expert_young_tau-0_epochSize5M_maxEpochs300"
CONFIG_PATH="configs/train_nf_zuko_theta.json"

mkdir -p "${OUTPUT_DIR}"
cd "$HOME/code/sbi_variants"

RESUME_CKPT="${OUTPUT_DIR}/resume_checkpoint_${RUN_NAME}.pt"

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

python -u train_variant_sbi_nf_theta.py \
  --config "${CONFIG_PATH}" \
  --data-path "${DATA_PATH}" \
  --cache-path "${CACHE_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --train-indices "${SPLIT_DIR}/train_indices_young.npy" \
  --val-indices "${SPLIT_DIR}/val_indices_young.npy" \
  --epochs 300 \
  --cluster-id-col cluster_ID \
  --curriculum-epoch-size 5000000 \
  --joint-curriculum \
  --tau-max 0.0 \
  --wandb \
  --wandb-project mock-galaxy-simformer \
  --device cuda \
  "${EXTRA_ARGS[@]}"

echo "Checking artifacts..."
ls -lh \
  "${OUTPUT_DIR}/best_model_${RUN_NAME}.pt" \
  "${OUTPUT_DIR}/posterior_config_${RUN_NAME}.json" \
  "${OUTPUT_DIR}/posterior_norm_meta_${RUN_NAME}.npz" \
  "${OUTPUT_DIR}/posterior_history_${RUN_NAME}.json" \
  "${OUTPUT_DIR}/resume_checkpoint_${RUN_NAME}.pt"
