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
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_supcon_positive_mask_sweep}
LAMBDAS=${LAMBDAS:-"0.05"}
TEMPERATURES=${TEMPERATURES:-"0.2"}
BATCH_SIZE=${BATCH_SIZE:-16}
SAMPLER=${SAMPLER:-domain_class_balanced}
INCLUDE_SAME_DOMAIN_VALUES=${INCLUDE_SAME_DOMAIN_VALUES:-"1 0"}
SUPCON_POSITIVE_MODES=${SUPCON_POSITIVE_MODES:-"all_views global_class_instance_views"}
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
read -r -a INCLUDE_VALUES <<< "${INCLUDE_SAME_DOMAIN_VALUES}"
read -r -a POSITIVE_MODES <<< "${SUPCON_POSITIVE_MODES}"

for LAMBDA_SUPCON in "${LAMBDA_VALUES[@]}"; do
  LAMBDA_TAG="$(lambda_tag "${LAMBDA_SUPCON}")"
  for TEMPERATURE in "${TEMPERATURE_VALUES[@]}"; do
    TAU_TAG="$(temperature_tag "${TEMPERATURE}")"
    for INCLUDE_SAME_DOMAIN in "${INCLUDE_VALUES[@]}"; do
      for POSITIVE_MODE in "${POSITIVE_MODES[@]}"; do
        EXPERIMENT_NAME="pa_csi_dg_supcon_${LAMBDA_TAG}_${TAU_TAG}_bs${BATCH_SIZE}_${SAMPLER}_incSame${INCLUDE_SAME_DOMAIN}_${POSITIVE_MODE}"
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
          --supcon-positive-mode "${POSITIVE_MODE}"
          --include-same-domain-same-class "${INCLUDE_SAME_DOMAIN}"
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
done
