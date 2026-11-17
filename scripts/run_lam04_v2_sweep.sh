#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

OUT="outputs_supcon_lam04_v2_sweep"
SEED=42
TARGET="E2"

CONFIGS=(
  "pa_csi_dg_supcon_v2_lam0400_bs08_none"
  "pa_csi_dg_supcon_v2_lam0400_bs08_domain_class_balanced"
  "pa_csi_dg_supcon_v2_lam0400_bs08_domain_balanced"
  "pa_csi_dg_supcon_v2_lam0400_bs16_none"
  "pa_csi_dg_supcon_v2_lam0400_bs16_domain_class_balanced"
  "pa_csi_dg_supcon_v2_lam0400_bs16_domain_balanced"
  "pa_csi_dg_supcon_v2_lam0400_bs32_none"
  "pa_csi_dg_supcon_v2_lam0400_bs32_domain_class_balanced"
  "pa_csi_dg_supcon_v2_lam0400_bs32_domain_balanced"
)

echo "=== λ=0.4 v2 sweep (correct hyperparams): ${#CONFIGS[@]} configs × E2 × seed${SEED} ==="

for EXP in "${CONFIGS[@]}"; do
  echo ""
  echo "--- Running: $EXP ---"
  python train_dg.py \
    --config "configs/${EXP}.json" \
    --output-root "$OUT" \
    --experiment-name "$EXP" \
    --target-env "$TARGET" \
    --seeds $SEED

  python3 -c "
import json
f = '${OUT}/dg_loeo/${EXP}/target_${TARGET}/seed_${SEED}/metrics.json'
d = json.load(open(f))
m = d.get('test_metrics', d)
print(f'  RESULT: acc={m[\"accuracy\"]:.4f}  f1={m[\"f1_macro\"]:.4f}')
" 2>/dev/null || echo "  RESULT: metrics read error"
done

echo ""
echo "=== FULL SUMMARY ==="
python3 - << 'PYEOF'
import json, os, re, glob

base = "outputs_supcon_lam04_v2_sweep/dg_loeo"
print(f"{'bs':>4}  {'sampler':<26}  {'E2 acc':>8}  {'E2 f1':>8}")
print("-" * 56)
for f in sorted(glob.glob(f"{base}/*/target_E2/seed_42/metrics.json")):
    exp = f.split(os.sep)[-4]
    m_match = re.search(r"bs(\d+)_(.+)$", exp)
    if not m_match: continue
    bs, sampler = int(m_match.group(1)), m_match.group(2)
    try:
        d = json.load(open(f))
        m = d.get("test_metrics", d)
        print(f"{bs:>4}  {sampler:<26}  {m['accuracy']:.4f}    {m['f1_macro']:.4f}")
    except:
        print(f"{bs:>4}  {sampler:<26}  MISSING")
PYEOF
