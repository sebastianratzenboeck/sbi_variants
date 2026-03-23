#!/bin/bash

# Minimal launcher for direct p(theta | x_obs) SBI variants.
# Override variables via environment or edit below.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-${SCRIPT_DIR}}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CACHE_PATH="${CACHE_PATH:-}"
EXCLUDE_INDICES="${EXCLUDE_INDICES:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/output_sbi_variants}"
RUN_NAME="${RUN_NAME:-sbi_fm_v1}"
METHOD="${METHOD:-flow_matching}"   # flow_matching | normalizing_flow
NF_BACKEND="${NF_BACKEND:-zuko}"    # zuko | nflows
NF_FAMILY="${NF_FAMILY:-nsf}"       # nsf | maf | nice
DEVICE="${DEVICE:-cuda}"
CONFIG_PATH="${CONFIG_PATH:-${REPO_DIR}/configs/train_fm_baseline.json}"

mkdir -p "${OUTPUT_DIR}"
cd "${REPO_DIR}"

if [[ -n "${CONFIG_PATH}" ]]; then
  "${PYTHON_BIN}" train_sbi_posterior.py \
    --config "${CONFIG_PATH}" \
    --device "${DEVICE}"
else
  if [[ -z "${CACHE_PATH}" ]]; then
    echo "ERROR: set CACHE_PATH or pass CONFIG_PATH."
    exit 1
  fi

  "${PYTHON_BIN}" train_sbi_posterior.py \
    --cache-path "${CACHE_PATH}" \
    --exclude-indices "${EXCLUDE_INDICES}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-name "${RUN_NAME}" \
    --method "${METHOD}" \
    --nf-backend "${NF_BACKEND}" \
    --nf-family "${NF_FAMILY}" \
    --batch-size 4096 \
    --epochs 300 \
    --lr 8e-4 \
    --lr-min 1e-5 \
    --val-split 0.1 \
    --time-prior-exponent 0.0 \
    --use-missingness-context \
    --amp \
    --device "${DEVICE}"
fi
