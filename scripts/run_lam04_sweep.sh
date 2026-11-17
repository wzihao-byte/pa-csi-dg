#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="outputs_supcon_lam04_sweep"
SEED=42
TARGET="E2"

CONFIGS=(
  "pa_csi_dg_supcon_lam0400_bs08_none_tau020"
  "pa_csi_dg_supcon_lam0400_bs08_domain_balanced_tau020"
  "pa_csi_dg_supcon_lam0400_bs16_none_tau020"
  "pa_csi_dg_supcon_lam0400_bs16_domain_balanced_tau020"
  "pa_csi_dg_supcon_lam0400_bs32_none_tau020"
  "pa_csi_dg_supcon_lam0400_bs32_domain_balanced_tau020"
)

echo "=== λ=0.4 sweep: 6 configs × E2 × seed42 ==="

for EXP in "${CONFIGS[@]}"; do
  echo ""
  echo "--- Running: $EXP ---"
  python train_dg.py \
    --config "configs/${EXP}.json" \
    --output-root "$OUT" \
    --experiment-name "$EXP" \
    --target-env "$TARGET" \
    --seeds $SEED

  RESULT=$(python3 -c "
import json
f = '${OUT}/dg_loeo/${EXP}/target_${TARGET}/seed_${SEED}/metrics.json'
d = json.load(open(f))
m = d.get('test_metrics', d)
print(f'acc={m[\"accuracy\"]:.4f}  f1={m[\"f1_macro\"]:.4f}')
" 2>/dev/null || echo "METRICS READ ERROR")
  echo "  RESULT: $RESULT"
done

echo ""
echo "=== FULL SUMMARY ==="
python3 - << 'PYEOF'
import json, os

base = "outputs_supcon_lam04_sweep/dg_loeo"
configs = [
    ("pa_csi_dg_supcon_lam0400_bs08_none_tau020",           8,  "none"),
    ("pa_csi_dg_supcon_lam0400_bs08_domain_balanced_tau020", 8,  "domain_balanced"),
    ("pa_csi_dg_supcon_lam0400_bs16_none_tau020",           16, "none"),
    ("pa_csi_dg_supcon_lam0400_bs16_domain_balanced_tau020",16, "domain_balanced"),
    ("pa_csi_dg_supcon_lam0400_bs32_none_tau020",           32, "none"),
    ("pa_csi_dg_supcon_lam0400_bs32_domain_balanced_tau020",32, "domain_balanced"),
]
print(f"{'bs':>4}  {'sampler':<20}  {'E2 acc':>8}")
print("-" * 40)
for exp, bs, sampler in configs:
    f = f"{base}/{exp}/target_E2/seed_42/metrics.json"
    try:
        d = json.load(open(f))
        m = d.get("test_metrics", d)
        a = m["accuracy"]
        print(f"{bs:>4}  {sampler:<20}  {a:.4f}")
    except:
        print(f"{bs:>4}  {sampler:<20}  MISSING")
PYEOF
