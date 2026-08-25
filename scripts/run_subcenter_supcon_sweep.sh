#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

CONFIG=${CONFIG:-configs/pa_csi_dg_subcenter_proto_layer2_200e_k3_margin_e2.json}
TARGET_ENV=${TARGET_ENV:-E2}
SEEDS=${SEEDS:-42}
EPOCHS=${EPOCHS:-100}
DEVICE=${DEVICE:-cuda:0}
OUTPUT_ROOT=${OUTPUT_ROOT:-outputs_subcenter_supcon_sweep}
LAMBDAS=${LAMBDAS:-"0.05 0.1 0.2"}
LAMBDA_PAIR_MARGINS=${LAMBDA_PAIR_MARGINS:-"0.0 0.02 0.05 0.1"}
PROTOTYPE_NUM_SUBCENTERS=${PROTOTYPE_NUM_SUBCENTERS:-"2 3 5"}
TEMPERATURE=${TEMPERATURE:-0.2}
BATCH_SIZE=${BATCH_SIZE:-16}
SAMPLER=${SAMPLER:-domain_class_balanced}
LAMBDA_SUPCON_WARMUP_EPOCHS=${LAMBDA_SUPCON_WARMUP_EPOCHS:-0}
DRY_RUN=${DRY_RUN:-0}

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config file not found: ${CONFIG}" >&2
  exit 1
fi

lambda_tag() {
  python -c 'import sys; print(f"lam{int(round(float(sys.argv[1]) * 1000)):04d}")' "$1"
}

pair_margin_tag() {
  python -c 'import sys; print(f"pairMargin{int(round(float(sys.argv[1]) * 1000)):04d}")' "$1"
}

temperature_tag() {
  python -c 'import sys; print(f"tau{int(round(float(sys.argv[1]) * 100)):03d}")' "$1"
}

read -r -a SEED_ARGS <<< "${SEEDS}"
read -r -a LAMBDA_VALUES <<< "${LAMBDAS}"
read -r -a PAIR_MARGIN_VALUES <<< "${LAMBDA_PAIR_MARGINS}"
read -r -a SUBCENTER_VALUES <<< "${PROTOTYPE_NUM_SUBCENTERS}"
TAU_TAG="$(temperature_tag "${TEMPERATURE}")"

for LAMBDA_SUPCON in "${LAMBDA_VALUES[@]}"; do
  LAMBDA_TAG="$(lambda_tag "${LAMBDA_SUPCON}")"
  for LAMBDA_PAIR_MARGIN in "${PAIR_MARGIN_VALUES[@]}"; do
    PAIR_MARGIN_TAG="$(pair_margin_tag "${LAMBDA_PAIR_MARGIN}")"
    for NUM_SUBCENTERS in "${SUBCENTER_VALUES[@]}"; do
      EXPERIMENT_NAME="pa_csi_dg_subcenter_${LAMBDA_TAG}_${PAIR_MARGIN_TAG}_k${NUM_SUBCENTERS}_${TAU_TAG}_bs${BATCH_SIZE}"
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
        --lambda-pair-margin "${LAMBDA_PAIR_MARGIN}"
        --prototype-num-subcenters "${NUM_SUBCENTERS}"
        --temperature "${TEMPERATURE}"
        --contrastive-loss-type subcenter_prototype
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
