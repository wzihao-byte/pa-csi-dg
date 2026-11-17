#!/usr/bin/env bash
# Phase 3: top-2 config 多 seed 验证
#
# 在 Phase 2 grid 选出 top-2 config 之后，
# 用 3env × seeds(42,52,62) 做最终验证，同时跑 matched CE baseline。
#
# 用法：
#   bash scripts/run_phase3_multiseed.sh --lam1 0.1 --bs1 8 --lam2 0.1 --bs2 16
#   bash scripts/run_phase3_multiseed.sh --lam1 0.1 --bs1 8 --lam2 0.1 --bs2 16 --device cpu

set -euo pipefail

DEVICE="cuda"
LAM1=""
BS1=""
LAM2=""
BS2=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --device) DEVICE="$2"; shift 2 ;;
    --lam1)   LAM1="$2";   shift 2 ;;
    --bs1)    BS1="$2";    shift 2 ;;
    --lam2)   LAM2="$2";   shift 2 ;;
    --bs2)    BS2="$2";    shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "$LAM1" || -z "$BS1" || -z "$LAM2" || -z "$BS2" ]]; then
  echo "Error: must provide --lam1 --bs1 --lam2 --bs2"
  echo "Example: bash scripts/run_phase3_multiseed.sh --lam1 0.1 --bs1 8 --lam2 0.1 --bs2 16"
  exit 1
fi

echo "=========================================="
echo "Phase 3: Multi-seed Validation"
echo "Config 1: lambda=$LAM1, bs=$BS1"
echo "Config 2: lambda=$LAM2, bs=$BS2"
echo "Seeds: 42 52 62  |  Envs: E1 E2 E3"
echo "Total runs: 2 configs × 3 envs × 2 new seeds = 12"
echo "  (seed42 already done in Phase 2 — reuse those results)"
echo "=========================================="

lam1_tag=$(printf "%04.0f" "$(echo "$LAM1 * 1000" | bc)")
lam2_tag=$(printf "%04.0f" "$(echo "$LAM2 * 1000" | bc)")
EXP1="final_lam${lam1_tag}_bs${BS1}"
EXP2="final_lam${lam2_tag}_bs${BS2}"

for exp_name in "$EXP1" "$EXP2"; do
  if [[ "$exp_name" == "$EXP1" ]]; then lam=$LAM1; bs=$BS1; else lam=$LAM2; bs=$BS2; fi

  echo ""
  echo "--- Config: $exp_name ---"

  # seed42 already ran in Phase 2; only add 52 and 62
  for seed in 52 62; do
    echo "  seed=$seed ..."
    python train_dg.py \
      --config configs/pa_csi_dg_supcon_paper_aligned.json \
      --target-env all \
      --seeds "$seed" \
      --device "$DEVICE" \
      --output-root "outputs_phase3_multiseed/${exp_name}" \
      --experiment-name "$exp_name" \
      --lambda-supcon "$lam" \
      --batch-size "$bs"
  done
done

# CE baseline: 3env × 3seeds
echo ""
echo "--- CE baseline (lambda=0.0, 3 seeds) ---"
for seed in 42 52 62; do
  echo "  seed=$seed ..."
  python train_dg.py \
    --config configs/pa_csi_dg_supcon_paper_aligned.json \
    --target-env all \
    --seeds "$seed" \
    --device "$DEVICE" \
    --output-root "outputs_phase3_multiseed/ce_baseline" \
    --experiment-name "ce_baseline" \
    --lambda-supcon 0.0
done

# ── 汇总 ────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "汇总 Phase 3 多 seed 结果..."
echo "=========================================="

python3 - <<PY
import csv, json
from pathlib import Path
from collections import defaultdict

def collect(root):
    rows = []
    for p in sorted(Path(root).rglob("metrics.json")):
        m = json.load(open(p))
        rc_path = p.parent / "resolved_config.json"
        rc = json.load(open(rc_path)) if rc_path.exists() else {}
        target_env = next((x.replace("target_","") for x in p.parts if x.startswith("target_")), "?")
        seed = int(p.parent.name.replace("seed_",""))
        rows.append({
            "exp": rc.get("experiment_name", "?"),
            "target_env": target_env,
            "seed": seed,
            "lam": rc.get("losses",{}).get("lambda_supcon",""),
            "bs":  rc.get("training",{}).get("batch_size",""),
            "acc": float(m.get("test_accuracy",0)),
            "f1":  float(m.get("test_f1_macro",0)),
        })
    return rows

# merge phase2 seed42 + phase3 seeds 52,62
rows = []
rows += collect("outputs_phase2_grid")
rows += collect("outputs_phase3_multiseed")

# group by (exp, lam, bs)
groups = defaultdict(list)
for r in rows:
    groups[(r["exp"], r["lam"], r["bs"])].append(r)

print(f"{'Config':<30} {'λ':>5} {'bs':>4} | {'MeanAcc':>8} {'StdAcc':>7} | {'MeanF1':>8} | {'E1':>7} {'E2':>7} {'E3':>7} | Seeds")
print("-"*100)

results = []
for (exp, lam, bs), rlist in groups.items():
    accs = [r["acc"] for r in rlist]
    f1s  = [r["f1"]  for r in rlist]
    env_acc = defaultdict(list)
    for r in rlist:
        env_acc[r["target_env"]].append(r["acc"])
    mean_acc = sum(accs)/len(accs)
    std_acc  = (sum((x-mean_acc)**2 for x in accs)/len(accs))**0.5
    mean_f1  = sum(f1s)/len(f1s)
    results.append((exp, lam, bs, mean_acc, std_acc, mean_f1, env_acc, len(rlist)))

results.sort(key=lambda x: -x[3])
for exp, lam, bs, mean_acc, std_acc, mean_f1, env_acc, n in results:
    e1 = sum(env_acc.get("E1",[0]))/max(len(env_acc.get("E1",[0])),1)
    e2 = sum(env_acc.get("E2",[0]))/max(len(env_acc.get("E2",[0])),1)
    e3 = sum(env_acc.get("E3",[0]))/max(len(env_acc.get("E3",[0])),1)
    print(f"{exp:<30} {str(lam):>5} {str(bs):>4} | {mean_acc:>8.4f} {std_acc:>7.4f} | {mean_f1:>8.4f} | {e1:>7.4f} {e2:>7.4f} {e3:>7.4f} | n={n}")

print()
print("Old anchors (reference):")
print("  旧 SupCon only seed42:  MeanAcc=0.9027  E1=0.9147  E2=0.8953  E3=0.8980")
print("  旧 CE baseline seed42:  MeanAcc=0.8787  E1=0.9007  E2=0.8547  E3=0.8807")
PY

echo ""
echo "完成。查看上方结果选择最终配置。"
