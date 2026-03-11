#!/usr/bin/env bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Flow-matching SBI training launcher
#
# Trains a conditional flow-matching posterior p(θ | x_obs) with:
#   - Quantile-based joint (logAge, m_init) curriculum
#   - Tau ramp from uniform → natural with IS correction (β=0.25)
#   - ESS monitoring, auto-widened FM clip bounds
#   - AMP + torch.compile enabled (FM is safe for both)
#
# Usage:
#   export CACHE_PATH="/path/to/build_arrays_cache.npz"
#   export OUTPUT_DIR="/path/to/output"
#   bash run_fm_training.sh
#
#   # Or override anything inline:
#   WANDB=1 EPOCHS=500 bash run_fm_training.sh
#
#   # Or append extra CLI flags:
#   bash run_fm_training.sh --batch-size 8192 --dropout 0.1
# ──────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ── Paths ────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_sbi_posterior.py"

OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/output_sbi_variants}"
RUN_NAME="${RUN_NAME:-sbi_fm_v1}"

DATA_PATH="${DATA_PATH:-}"
CACHE_PATH="${CACHE_PATH:-${OUTPUT_DIR}/build_arrays_cache.npz}"
EXCLUDE_INDICES="${EXCLUDE_INDICES:-}"
CONFIG_PATH="${CONFIG_PATH:-}"

# ── Training hyperparameters ─────────────────────────────────────────
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-300}"
BATCH_SIZE="${BATCH_SIZE:-16384}"
LR="${LR:-8e-4}"
LR_MIN="${LR_MIN:-1e-5}"
PATIENCE="${PATIENCE:-60}"

# ── Curriculum & importance weighting ────────────────────────────────
TAU_MAX="${TAU_MAX:-0.8}"
TAU_WARMUP="${TAU_WARMUP:-10}"
BETA="${BETA:-0.25}"
BIN_STRATEGY="${BIN_STRATEGY:-quantile}"
N_AGE_BINS="${N_AGE_BINS:-25}"
N_MASS_BINS="${N_MASS_BINS:-12}"

# ── Resume ───────────────────────────────────────────────────────────
RESUME_AUTO="${RESUME_AUTO:-1}"
RESUME_CKPT="${OUTPUT_DIR}/resume_checkpoint_${RUN_NAME}.pt"

# ── Logging ──────────────────────────────────────────────────────────
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-mock-galaxy-simformer}"

# ──────────────────────────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}"

ARGS=(
  --method flow_matching
  --run-name "${RUN_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --cache-path "${CACHE_PATH}"
  --seed "${SEED}"
  --device "${DEVICE}"
  # Optimisation
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --lr-min "${LR_MIN}"
  --patience "${PATIENCE}"
  --weight-decay 1e-4
  --grad-clip-norm 1.0
  # FM is safe for AMP and torch.compile
  --amp
  --compile
  # Encoder
  --attn-embed-dim 128
  --num-heads 8
  --num-layers 4
  --dropout 0.05
  --use-missingness-context
  --use-colors
  # FM head
  --fm-hidden-dim 256
  --time-embed-dim 64
  --sigma-min 1e-3
  --time-prior-exponent 0.0
  # Curriculum
  --joint-curriculum
  --curriculum-bin-strategy "${BIN_STRATEGY}"
  --n-bins "${N_AGE_BINS}"
  --n-mass-bins "${N_MASS_BINS}"
  --tau-max "${TAU_MAX}"
  --tau-warmup "${TAU_WARMUP}"
  # IS weighting (clip bounds auto-widened for FM + beta=0.25)
  --importance-weighting
  --importance-weight-beta "${BETA}"
  --importance-weight-min 0.5
  --importance-weight-max 2.0
  # Evaluation
  --val-split 0.1
  --young-logage-threshold 7.8
  --young-eval-max-stars 100000
  --random-eval-max-stars 100000
  --val-curriculum-loss
  --val-curriculum-epoch-size 100000
)

# ── Config file (if provided, CLI flags above still override) ────────
if [[ -n "${CONFIG_PATH}" ]]; then
  ARGS=(--config "${CONFIG_PATH}" "${ARGS[@]}")
  echo "Base config: ${CONFIG_PATH} (CLI flags override)"
fi

# ── Data / cache handling ────────────────────────────────────────────
if [[ -n "${DATA_PATH}" ]]; then
  ARGS+=(--data-path "${DATA_PATH}")
fi

if [[ -f "${CACHE_PATH}" ]]; then
  ARGS+=(--no-rebuild-cache)
  echo "Cache found: ${CACHE_PATH}"
else
  if [[ -z "${DATA_PATH}" ]]; then
    echo "ERROR: cache not found at ${CACHE_PATH} and DATA_PATH is empty."
    echo "  Set DATA_PATH to build cache, or point CACHE_PATH to an existing file."
    exit 1
  fi
  ARGS+=(--rebuild-cache)
  echo "Cache not found; will build from DATA_PATH."
fi

# ── Exclusion indices ────────────────────────────────────────────────
if [[ -n "${EXCLUDE_INDICES}" ]]; then
  ARGS+=(--exclude-indices "${EXCLUDE_INDICES}")
fi

# ── wandb ────────────────────────────────────────────────────────────
if [[ "${WANDB}" == "1" ]]; then
  ARGS+=(--wandb --wandb-project "${WANDB_PROJECT}")
else
  ARGS+=(--no-wandb)
fi

# ── Auto-resume ──────────────────────────────────────────────────────
if [[ "${RESUME_AUTO}" == "1" && -f "${RESUME_CKPT}" ]]; then
  ARGS+=(--resume-from "${RESUME_CKPT}")
  echo "Resuming from: ${RESUME_CKPT}"
else
  echo "Starting fresh (no resume checkpoint found or RESUME_AUTO=0)."
fi

# ── Extra CLI args passed to this script ─────────────────────────────
if [[ $# -gt 0 ]]; then
  echo "Extra CLI args: $*"
fi

# ── Launch ───────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Flow-matching training: ${RUN_NAME}"
echo "  tau_max=${TAU_MAX}, beta=${BETA}, bins=${BIN_STRATEGY}"
echo "  batch=${BATCH_SIZE}, lr=${LR}, epochs=${EPOCHS}"
echo "═══════════════════════════════════════════════════════════════"
echo ""

set -x
"${PYTHON_BIN}" "${TRAIN_SCRIPT}" "${ARGS[@]}" "$@"
set +x

echo ""
echo "Training finished. Output artifacts:"
ls -lh \
  "${OUTPUT_DIR}/best_model_${RUN_NAME}.pt" \
  "${OUTPUT_DIR}/posterior_config_${RUN_NAME}.json" \
  "${OUTPUT_DIR}/posterior_norm_meta_${RUN_NAME}.npz" \
  "${OUTPUT_DIR}/posterior_history_${RUN_NAME}.json" \
  "${OUTPUT_DIR}/resume_checkpoint_${RUN_NAME}.pt" 2>/dev/null || true
