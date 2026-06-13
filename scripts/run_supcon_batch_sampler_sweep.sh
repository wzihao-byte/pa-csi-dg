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
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_supcon_batch_sampler_sweep}
LAMBDAS=${LAMBDAS:-"0.05 0.1"}
BATCH_SIZES=${BATCH_SIZES:-"8 16 32"}
SAMPLERS=${SAMPLERS:-"none domain_balanced domain_class_balanced"}
TEMPERATURE=${TEMPERATURE:-0.2}
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
read -r -a BATCH_SIZE_VALUES <<< "${BATCH_SIZES}"
read -r -a SAMPLER_VALUES <<< "${SAMPLERS}"
TAU_TAG="$(temperature_tag "${TEMPERATURE}")"

for LAMBDA_SUPCON in "${LAMBDA_VALUES[@]}"; do
  LAMBDA_TAG="$(lambda_tag "${LAMBDA_SUPCON}")"
  for BATCH_SIZE in "${BATCH_SIZE_VALUES[@]}"; do
    for SAMPLER in "${SAMPLER_VALUES[@]}"; do
      EXPERIMENT_NAME="pa_csi_dg_supcon_${LAMBDA_TAG}_bs${BATCH_SIZE}_${SAMPLER}_${TAU_TAG}"
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
        --supcon-positive-mode all_views
        --include-same-domain-same-class 1
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
done
