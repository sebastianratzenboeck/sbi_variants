#!/bin/bash
#SBATCH -J eval-age-gate
#SBATCH -p gpu
#SBATCH -t 0-12:00
#SBATCH --mem=350000
#SBATCH -c 16
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -o age_gate_eval_%j.out
#SBATCH -e age_gate_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=sebastian.ratzenboeck@cfa.harvard.edu
#SBATCH --account=itc_lab

module purge
module load python
mamba activate py_gpu_cuda12.4

CACHE_PATH="/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/nf_sbi_colors/build_arrays_cache.npz"
NORM_META_PATH="/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/nf_sbi_colors/posterior_norm_meta_nf_zuko_colors_beta_tau-0_beta-025_epochSize5M_noDropout_maxEpochs5M.npz"
SPLIT_DIR="/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/nf_sbi_colors/age_regime_splits"
MODEL_DIR="/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/age_gate_colors"
RUN_NAME="age_gate_colors_balanced_epochSize5M_maxEpochs40"
OUTPUT_DIR="${MODEL_DIR}/eval_${RUN_NAME}"

mkdir -p "${OUTPUT_DIR}"
cd "$HOME/code/sbi_variants"

python -u eval_age_gate.py \
  --cache-path "${CACHE_PATH}" \
  --norm-meta-path "${NORM_META_PATH}" \
  --gate-config "${MODEL_DIR}/age_gate_config_${RUN_NAME}.json" \
  --gate-checkpoint "${MODEL_DIR}/best_age_gate_${RUN_NAME}.pt" \
  --temperature-json "${MODEL_DIR}/age_gate_temperature_${RUN_NAME}.json" \
  --output-dir "${OUTPUT_DIR}" \
  --eval-split natural_test=/n/holystore01/LABS/itc_lab/Lab/to-Sebastian/nf_sbi_colors/test_indices.npy \
  --eval-split balanced_age_300k="${SPLIT_DIR}/eval_indices_balanced_age_300k.npy" \
  --batch-size 4096 \
  --num-workers 8 \
  --device cuda

echo "Checking artifacts..."
ls -lh \
  "${OUTPUT_DIR}/age_gate_eval_bundle.json" \
  "${OUTPUT_DIR}/age_gate_summary_natural_test.json" \
  "${OUTPUT_DIR}/age_gate_summary_balanced_age_300k.json" \
  "${OUTPUT_DIR}/age_gate_predictions_natural_test.npz" \
  "${OUTPUT_DIR}/age_gate_predictions_balanced_age_300k.npz"
