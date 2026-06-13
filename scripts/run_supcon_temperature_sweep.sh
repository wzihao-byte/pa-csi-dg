#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG=${CONFIG:-configs/pa_csi_dg_supcon_paper_aligned.json}
TARGET_ENV=${TARGET_ENV:-E2}
SEEDS=${SEEDS:-42}
EPOCHS=${EPOCHS:-80}
DEVICE=${DEVICE:-cuda:0}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_supcon_temperature_sweep}
LAMBDAS=${LAMBDAS:-"0.05 0.1"}
TEMPERATURES=${TEMPERATURES:-"0.07 0.1 0.2 0.3"}
BATCH_SIZE=${BATCH_SIZE:-16}
SAMPLER=${SAMPLER:-domain_class_balanced}
INCLUDE_SAME_DOMAIN_SAME_CLASS=${INCLUDE_SAME_DOMAIN_SAME_CLASS:-1}
SUPCON_POSITIVE_MODE=${SUPCON_POSITIVE_MODE:-all_views}
DRY_RUN=${DRY_RUN:-0}

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config file not found: ${CONFIG}" >&2
  exit 1
fi

lambda_tag() {
  python -c 'import sys; print(f"lam{int(round(float(sys.argv[1]) * 1000)):04d}")' "$1"
}

temperature_tag() {
  python -c 'import sys; print(f"tau{int(round(float(sys.argv[1]) * 100)):03d}")' "$1"
}

read -r -a SEED_ARGS <<< "${SEEDS}"
read -r -a LAMBDA_VALUES <<< "${LAMBDAS}"
read -r -a TEMPERATURE_VALUES <<< "${TEMPERATURES}"

for LAMBDA_SUPCON in "${LAMBDA_VALUES[@]}"; do
  LAMBDA_TAG="$(lambda_tag "${LAMBDA_SUPCON}")"
  for TEMPERATURE in "${TEMPERATURE_VALUES[@]}"; do
    TAU_TAG="$(temperature_tag "${TEMPERATURE}")"
    EXPERIMENT_NAME="pa_csi_dg_supcon_${LAMBDA_TAG}_${TAU_TAG}_bs${BATCH_SIZE}_${SAMPLER}"
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
  done
done
