#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/pa_csi_dg_supcon_no_unwrap.json"
OUT="outputs_anchor_supcon_no_unwrap"
EXP="anchor_supcon_no_unwrap"
SEED=42

echo "=== Testing no_unwrap hypothesis: E1 only, seed $SEED ==="
python train_dg.py \
  --config "$CONFIG" \
  --output-dir "$OUT" \
  --experiment-name "$EXP" \
  --target-env E1 \
  --seed $SEED

echo "=== RESULT ==="
cat "$OUT/dg_loeo/$EXP/target_E1/seed_$SEED/metrics.json" | python3 -c "
import json, sys
m = json.load(sys.stdin)
print(f'test_accuracy: {m[\"test_accuracy\"]:.4f}')
print(f'test_f1_macro: {m[\"test_f1_macro\"]:.4f}')
"
