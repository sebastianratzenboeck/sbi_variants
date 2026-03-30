#!/bin/bash
#SBATCH -J age-gate
#SBATCH -p gpu
#SBATCH -t 1-00:00
#SBATCH --mem=350000
#SBATCH -c 16
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -o age_gate_train_%j.out
#SBATCH -e age_gate_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sebastian.ratzenboeck@cfa.harvard.edu
#SBATCH --account=itc_lab

module purge
module load python
mamba activate py_gpu_cuda12.4

CACHE_PATH="/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/nf_sbi_colors/build_arrays_cache.npz"
NORM_META_PATH="/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/nf_sbi_colors/posterior_norm_meta_nf_zuko_colors_beta_tau-0_beta-025_epochSize5M_noDropout_maxEpochs5M.npz"
SPLIT_DIR="/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/nf_sbi_colors/age_regime_splits"
TRAIN_INDICES="${SPLIT_DIR}/train_indices_all.npy"
VAL_INDICES="${SPLIT_DIR}/val_indices_all.npy"
OUTPUT_DIR="/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/age_gate_colors"
RUN_NAME="age_gate_colors_balanced_epochSize5M_maxEpochs40"

mkdir -p "${OUTPUT_DIR}"
cd "$HOME/code/sbi_variants"

python -u train_age_gate.py \
  --cache-path "${CACHE_PATH}" \
  --norm-meta-path "${NORM_META_PATH}" \
  --train-indices "${TRAIN_INDICES}" \
  --val-indices "${VAL_INDICES}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-name "${RUN_NAME}" \
  --epoch-size 5000000 \
  --epochs 40 \
  --patience 10 \
  --batch-size 4096 \
  --num-workers 8 \
  --amp \
  --wandb \
  --wandb-project mock-galaxy-simformer \
  --device cuda

echo "Checking artifacts..."
ls -lh \
  "${OUTPUT_DIR}/best_age_gate_${RUN_NAME}.pt" \
  "${OUTPUT_DIR}/age_gate_config_${RUN_NAME}.json" \
  "${OUTPUT_DIR}/age_gate_history_${RUN_NAME}.json" \
  "${OUTPUT_DIR}/age_gate_temperature_${RUN_NAME}.json" \
  "${OUTPUT_DIR}/age_gate_norm_meta_${RUN_NAME}.npz"
