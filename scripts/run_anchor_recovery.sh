#!/usr/bin/env bash
# Phase 1: Anchor recovery — paper_aligned SupCon + CE baseline
# Run on 3env / seed42. Purpose: confirm current code can reproduce old results
# before any further hyperparameter search.
#
# Gate:
#   paper_aligned SupCon mean acc >= 0.89  → proceed to Phase 2 grid
#   paper_aligned SupCon mean acc <  0.88  → STOP, debug config/code/protocol diff
#
# Usage:
#   bash scripts/run_anchor_recovery.sh
#   bash scripts/run_anchor_recovery.sh --device cpu   # for smoke test

set -euo pipefail

DEVICE="${1:-cuda}"
# Allow --device flag
for arg in "$@"; do
  case $arg in
    --device) shift; DEVICE="$1"; shift ;;
    --device=*) DEVICE="${arg#*=}" ;;
  esac
done

echo "=========================================="
echo "Phase 1: Anchor Recovery"
echo "Device: $DEVICE"
echo "=========================================="

# ── SupCon paper_aligned anchor ─────────────────────────────────────────────
echo ""
echo "[1/2] Running paper_aligned SupCon anchor (3env × seed42)..."
python train_dg.py \
  --config configs/pa_csi_dg_supcon_paper_aligned.json \
  --target-env all \
  --seeds 42 \
  --device "$DEVICE" \
  --output-root outputs_anchor_supcon_paper_aligned \
  --experiment-name anchor_supcon_paper_aligned

echo ""
echo "[1/2] Aggregating SupCon anchor results..."
python scripts/aggregate_dg_final_results.py \
  --output-root outputs_anchor_supcon_paper_aligned \
  --per-run-csv anchor_supcon_paper_aligned_per_run.csv \
  --aggregate-csv anchor_supcon_paper_aligned_aggregate.csv

# ── CE baseline anchor ───────────────────────────────────────────────────────
echo ""
echo "[2/2] Running CE baseline anchor (3env × seed42)..."
python train_dg.py \
  --config configs/pa_csi_dg_supcon_paper_aligned.json \
  --target-env all \
  --seeds 42 \
  --device "$DEVICE" \
  --output-root outputs_anchor_ce_baseline \
  --experiment-name anchor_ce_baseline \
  --lambda-supcon 0.0

echo ""
echo "[2/2] Aggregating CE baseline anchor results..."
python scripts/aggregate_dg_final_results.py \
  --output-root outputs_anchor_ce_baseline \
  --per-run-csv anchor_ce_baseline_per_run.csv \
  --aggregate-csv anchor_ce_baseline_aggregate.csv

# ── Gate check ───────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Gate Check"
echo "=========================================="
python3 - <<'PY'
import csv, sys

def read_overall(path):
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
        overall = [r for r in rows if r.get("aggregate_scope") == "overall"]
        if overall:
            r = overall[0]
            return float(r["mean_test_accuracy"]), float(r["mean_test_f1_macro"])
        # fallback: mean all rows
        accs = [float(r["mean_test_accuracy"]) for r in rows if r.get("mean_test_accuracy")]
        f1s  = [float(r["mean_test_f1_macro"])  for r in rows if r.get("mean_test_f1_macro")]
        return sum(accs)/len(accs), sum(f1s)/len(f1s)
    except FileNotFoundError:
        return None, None

sup_acc, sup_f1 = read_overall("anchor_supcon_paper_aligned_aggregate.csv")
ce_acc,  ce_f1  = read_overall("anchor_ce_baseline_aggregate.csv")

print(f"  SupCon paper_aligned:  Acc={sup_acc:.4f}  F1={sup_f1:.4f}" if sup_acc else "  SupCon: file not found")
print(f"  CE baseline:           Acc={ce_acc:.4f}   F1={ce_f1:.4f}"  if ce_acc  else "  CE:     file not found")

OLD_SUPCON_ACC = 0.9027
if sup_acc is not None:
    delta = sup_acc - OLD_SUPCON_ACC
    print(f"\n  Delta vs old SupCon only: {delta:+.4f}")
    if sup_acc >= 0.89:
        print("\n  ✓ GATE PASSED — proceed to Phase 2 grid (scripts/run_phase2_grid.sh)")
    elif sup_acc >= 0.88:
        print("\n  ⚠ MARGINAL — review per-env breakdown before proceeding")
    else:
        print("\n  ✗ GATE FAILED — do NOT continue sweep")
        print("    Debug checklist:")
        print("      1. Compare resolved_config.json with old paper_aligned run")
        print("      2. Check sampler, batch_size, positive_mode defaults")
        print("      3. Check train/val/test split reproducibility (split_manifest.json)")
        print("      4. Check data file paths and preprocessing (unwrap_phase)")
PY

echo ""
echo "Done. Check anchor_supcon_paper_aligned_aggregate.csv for per-env breakdown."
