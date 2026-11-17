#!/usr/bin/env bash
# Phase 2: 3env/seed42 grid search around paper_aligned anchor
#
# Base: pa_csi_dg_supcon_paper_aligned config (bs=8, sampler=none, tau=0.2,
#       positive_mode=default, include_same_domain=True)
# Sweep: lambda_supcon × batch_size (3×3 = 9 configs × 3 envs = 27 runs)
#
# Run ONLY after run_anchor_recovery.sh passes the gate.
#
# Usage:
#   bash scripts/run_phase2_grid.sh
#   bash scripts/run_phase2_grid.sh --device cpu

set -euo pipefail

DEVICE="cuda"
for arg in "$@"; do
  case $arg in
    --device) shift; DEVICE="$1"; shift ;;
    --device=*) DEVICE="${arg#*=}" ;;
  esac
done

LAMBDAS=(0.05 0.1 0.2)
BATCH_SIZES=(8 16 32)

echo "=========================================="
echo "Phase 2: 3env/seed42 Lambda × BatchSize Grid"
echo "Device: $DEVICE"
echo "Configs: ${#LAMBDAS[@]} lambdas × ${#BATCH_SIZES[@]} batch sizes = $((${#LAMBDAS[@]} * ${#BATCH_SIZES[@]})) configs"
echo "Total runs: $((${#LAMBDAS[@]} * ${#BATCH_SIZES[@]} * 3)) (× 3 envs)"
echo "=========================================="

run_idx=0
total=$((${#LAMBDAS[@]} * ${#BATCH_SIZES[@]}))

for lam in "${LAMBDAS[@]}"; do
  for bs in "${BATCH_SIZES[@]}"; do
    run_idx=$((run_idx + 1))
    lam_tag=$(printf "%04.0f" "$(echo "$lam * 1000" | bc)")
    exp_name="grid_lam${lam_tag}_bs${bs}"
    out_root="outputs_phase2_grid/${exp_name}"

    echo ""
    echo "--- [$run_idx/$total] lambda=$lam  bs=$bs  → $exp_name ---"

    python train_dg.py \
      --config configs/pa_csi_dg_supcon_paper_aligned.json \
      --target-env all \
      --seeds 42 \
      --device "$DEVICE" \
      --output-root "$out_root" \
      --experiment-name "$exp_name" \
      --lambda-supcon "$lam" \
      --batch-size "$bs"
  done
done

# ── Aggregate all grid runs ──────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Aggregating Phase 2 grid results..."
echo "=========================================="

# Collect all per-run CSVs into one
python3 - <<'PY'
import csv, json, os
from pathlib import Path

out_root = Path("outputs_phase2_grid")
per_run_rows = []
fieldnames = None

for metrics_path in sorted(out_root.rglob("metrics.json")):
    with open(metrics_path) as f:
        m = json.load(f)
    rc_path = metrics_path.parent / "resolved_config.json"
    rc = json.load(open(rc_path)) if rc_path.exists() else {}

    row = {
        "experiment_name": rc.get("experiment_name", metrics_path.parts[-3]),
        "target_env":      metrics_path.parts[-2] if "seed_" in str(metrics_path) else "?",
        "seed":            int(metrics_path.parent.name.replace("seed_", "")),
        "lambda_supcon":   rc.get("losses", {}).get("lambda_supcon", ""),
        "batch_size":      rc.get("training", {}).get("batch_size", ""),
        "sampler_mode":    rc.get("training", {}).get("sampler", ""),
        "test_accuracy":   m.get("test_accuracy", ""),
        "test_f1_macro":   m.get("test_f1_macro", ""),
        "best_val_accuracy": m.get("best_val_accuracy", ""),
    }
    # parse target_env from path: outputs_phase2_grid/exp/dg_loeo/exp/target_E1/seed_42/
    parts = metrics_path.parts
    for p in parts:
        if p.startswith("target_"):
            row["target_env"] = p.replace("target_", "")
            break
    per_run_rows.append(row)

if not per_run_rows:
    print("No metrics.json found yet.")
else:
    fieldnames = list(per_run_rows[0].keys())
    with open("phase2_grid_per_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(per_run_rows)
    print(f"Wrote {len(per_run_rows)} rows to phase2_grid_per_run.csv")

    # Summary: mean over 3 envs per config
    from collections import defaultdict
    groups = defaultdict(list)
    for r in per_run_rows:
        key = (r["experiment_name"], r["lambda_supcon"], r["batch_size"])
        groups[key].append(r)

    print("\n3env/seed42 Grid Results (sorted by mean acc desc):")
    print(f"{'Config':<35} {'λ':>5} {'bs':>4} {'MeanAcc':>8} {'MeanF1':>8} {'E1':>7} {'E2':>7} {'E3':>7}")
    print("-" * 90)

    summary = []
    for key, rows in groups.items():
        env_acc = {r["target_env"]: float(r["test_accuracy"]) for r in rows if r["test_accuracy"]}
        env_f1  = {r["target_env"]: float(r["test_f1_macro"])  for r in rows if r["test_f1_macro"]}
        if len(env_acc) == 3:
            mean_acc = sum(env_acc.values()) / 3
            mean_f1  = sum(env_f1.values())  / 3
            summary.append((key, mean_acc, mean_f1, env_acc, env_f1))

    summary.sort(key=lambda x: -x[1])
    for key, mean_acc, mean_f1, env_acc, env_f1 in summary:
        exp, lam, bs = key
        print(f"{exp:<35} {lam:>5} {bs:>4} {mean_acc:>8.4f} {mean_f1:>8.4f} "
              f"{env_acc.get('E1',0):>7.4f} {env_acc.get('E2',0):>7.4f} {env_acc.get('E3',0):>7.4f}")

    print("\nOld anchors (reference):")
    print(f"  旧 SupCon only seed42:   MeanAcc=0.9027  MeanF1=0.8820  E1=0.9147  E2=0.8953  E3=0.8980")
    print(f"  旧 CE baseline seed42:   MeanAcc=0.8787  MeanF1=0.8515  E1=0.9007  E2=0.8547  E3=0.8807")
PY

echo ""
echo "Done. Top configs in phase2_grid_per_run.csv"
echo "Next: select top-2 configs (balanced E1/E2/E3), run scripts/run_phase3_multiseed.sh"
