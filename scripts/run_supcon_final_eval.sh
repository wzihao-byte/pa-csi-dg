#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG=${CONFIG:-configs/pa_csi_dg_supcon_paper_aligned.json}
TARGET_ENV=${TARGET_ENV:-all}
SEEDS=${SEEDS:-"42 52 62"}
EPOCHS=${EPOCHS:-200}
DEVICE=${DEVICE:-cuda:0}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_supcon_final_eval}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-pa_csi_dg_supcon_final}
LAMBDA_SUPCON=${LAMBDA_SUPCON:-0.05}
TEMPERATURE=${TEMPERATURE:-0.2}
BATCH_SIZE=${BATCH_SIZE:-16}
SAMPLER=${SAMPLER:-domain_class_balanced}
INCLUDE_SAME_DOMAIN_SAME_CLASS=${INCLUDE_SAME_DOMAIN_SAME_CLASS:-1}
SUPCON_POSITIVE_MODE=${SUPCON_POSITIVE_MODE:-all_views}
LAMBDA_SUPCON_WARMUP_EPOCHS=${LAMBDA_SUPCON_WARMUP_EPOCHS:-0}
DRY_RUN=${DRY_RUN:-0}

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config file not found: ${CONFIG}" >&2
  exit 1
fi

read -r -a SEED_ARGS <<< "${SEEDS}"

cmd=(
  python train_dg.py
  --config "${CONFIG}"
  --target-env "${TARGET_ENV}"
  --seeds "${SEED_ARGS[@]}"
  --epochs "${EPOCHS}"
  --device "${DEVICE}"
  --output-root "${OUTPUT_ROOT}"
  --batch-size "${BATCH_SIZE}"
  --sampler "${SAMPLER}"
  --lambda-supcon "${LAMBDA_SUPCON}"
  --lambda-supcon-warmup-epochs "${LAMBDA_SUPCON_WARMUP_EPOCHS}"
  --temperature "${TEMPERATURE}"
  --supcon-positive-mode "${SUPCON_POSITIVE_MODE}"
  --include-same-domain-same-class "${INCLUDE_SAME_DOMAIN_SAME_CLASS}"
  --contrastive-loss-type pairwise_supcon
  --experiment-name "${EXPERIMENT_NAME}"
)

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${cmd[@]}"
  printf '\n'
else
  "${cmd[@]}"
fi
