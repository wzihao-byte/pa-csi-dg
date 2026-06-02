# Experiment Result Index

This index records the report and result artifacts intentionally preserved after cleaning abandoned side branches and large experiment outputs.

## Repository State

- Code baseline committed before adding these reports: `940949a` (`Add dynamic per-pair-margin scheduler with per-pair m_max caps (v2)`).
- Removed from the working tree before this report commit:
  - `outputs*`
  - `results`
  - `experiment_outputs`
  - `codex_isolated`
  - MMFI workspace data, scripts, configs, notes, docs, and outputs
- Preserved:
  - Report source files (`.md`, `.tex`)
  - Report PDFs
  - Small summary tables (`.csv`, `.json`)
  - MMFI weekly report files copied from the separate MMFI workspace
- Not preserved:
  - Model checkpoints (`best_model.pt`)
  - Full confusion-matrix output directories
  - Large run logs and generated intermediate outputs
  - LaTeX build intermediates (`.aux`, `.log`, `.out`)

## Primary Reports

| Artifact | Scope | Notes |
|---|---|---|
| `reports/e2_abandon_multicenter/e2_abandon_multicenter_report_en.md` | E2 multi-center retirement decision | Main technical rationale for abandoning multi-center/subcenter repair. |
| `reports/weekly_report_20260529_e2_abandonment/weekly_report_20260529_e2_abandonment.{tex,pdf}` | Weekly E2 abandonment report | Report version of the E2 stop decision. |
| `reports/monthly_report_202605_may/monthly_report_may_2026.{tex,pdf}` | May 2026 monthly report | Monthly status and experiment summary. |
| `reports/may_report_202605_interim/may_report_2026_interim.{tex,pdf}` | May interim report | Intermediate report. |
| `reports/weekly_report_20260430_subcenter_proto/weekly_report_20260430_subcenter_proto.{tex,pdf}` | Earlier subcenter prototype report | Kept for historical traceability. |
| `reports/weekly_report_loeo_ablation/weekly_report_loeo_ablation.{tex,pdf}` | LOEO ablation report | Kept for baseline context. |
| `reports/mmfi_weekly_20260525_20260531/` | MMFI weekly report | Report files only; MMFI data/scripts/outputs were deleted. |

## Result Tables

| Artifact | Scope |
|---|---|
| `reports/tables/cc_vrex_focus14_e2_smoke_summary.csv` | CC-VREx lambda 0.2 smoke result. |
| `reports/tables/cc_vrex_lam01_focus14_e2_smoke_summary.csv` | CC-VREx lambda 0.1 smoke result. |
| `reports/tables/k1_pairmargin_14_e2_smoke_summary.csv` | K1 class 1/4 pair-margin smoke result. |
| `reports/preserved_codex_isolated/codex_isolated/cc_vrex_focus14_e2_fullprep/reports/full_20260525_retry2/summary.csv` | CC-VREx full 3-seed validation table. |
| `reports/preserved_codex_isolated/codex_isolated/cc_vrex_focus14_e2_fullprep/reports/full_20260525_retry2/summary.json` | CC-VREx full 3-seed validation JSON. |
| `reports/preserved_codex_isolated/codex_isolated/cc_vrex_focus14_e2_fullprep/reports/full_20260525_retry2/result_interpretation_20260529.md` | CC-VREx full validation interpretation. |
| `reports/preserved_codex_isolated/codex_isolated/e2_14_ot_alignment_20260522/reports/` | OT alignment gate and mechanism reports. |
| `reports/preserved_codex_isolated/codex_isolated/method_result_inventory_20260525/reports/method_result_inventory_20260525.md` | E2 method/result inventory. |

## Key Experiment Conditions

### E2 Multi-Center / Subcenter Retirement

- Dataset/protocol: PA-CSI MultiEnv `dg_loeo`.
- Target environment: `E2`.
- Main comparison: K=1 source-only control vs K=3 multi-center/subcenter variants.
- Seeds used in full checks: `42`, `52`, `62`.
- Epoch budget in full checks: `200`.
- Selection discipline: source validation only; target E2 labels used for diagnostics/reporting only.
- Metrics emphasized:
  - target accuracy
  - target macro-F1
  - `min_f1_1_4`
  - class 1 -> 4 and class 4 -> 1 confusion counts
  - prototype occupancy diagnostics
- Main conclusion:
  - K=3 did not reproducibly beat K=1.
  - Target E2 class 1/4 embeddings collapsed relative to source class 1/4 separation.
  - Warmup k-means splits and trained prototype occupancy were unstable.
  - Multi-center/subcenter repair, occupancy-floor tuning, and asymmetric margin grids should be stopped.

Primary artifact: `reports/e2_abandon_multicenter/e2_abandon_multicenter_report_en.md`.

### CC-VREx Focus 1/4 Smoke

- Dataset/protocol: PA-CSI MultiEnv `dg_loeo`.
- Target environment: `E2`.
- Seed: `42`.
- Epoch budget: smoke run.
- Baseline family: K=1, no geometry, domain-balanced batch, source-only setup.
- Selection metric: `min_f1_1_4`.
- Candidate: source-only class-conditional V-REx focused on classes 1 and 4.

`lambda=0.2` smoke:

- Baseline target accuracy: `0.7840`.
- Proposed target accuracy: `0.8577`.
- Baseline macro-F1: `0.7683`.
- Proposed macro-F1: `0.8396`.
- Baseline `min_f1_1_4`: `0.4237`.
- Proposed `min_f1_1_4`: `0.6592`.
- Status: positive smoke only; not a success claim.

`lambda=0.1` smoke:

- Baseline target accuracy: `0.7953`.
- Proposed target accuracy: `0.8470`.
- Baseline macro-F1: `0.7862`.
- Proposed macro-F1: `0.8234`.
- Baseline `min_f1_1_4`: `0.5331`.
- Proposed `min_f1_1_4`: `0.5744`.
- Status: weaker positive smoke only.

Primary artifacts:

- `reports/tables/cc_vrex_focus14_e2_smoke_summary.csv`
- `reports/tables/cc_vrex_lam01_focus14_e2_smoke_summary.csv`

### CC-VREx Focus 1/4 Full Validation

- Run id: `full_20260525_retry2`.
- Dataset/protocol: PA-CSI MultiEnv `dg_loeo`.
- Target environment: `E2`.
- Seeds: `42`, `52`, `62`.
- Epoch budget: `200`.
- Baseline: `pa_csi_dg_k1_nomargin_dcb_nogeom_e2_full`.
- Candidate: `pa_csi_dg_k1_cc_vrex_focus14_nogeom_e2_full`.
- Candidate parameter: CC-VREx `lambda=0.2`.
- Selection discipline: matched baseline/candidate by seed.

Mean metrics:

| Case | Accuracy | Macro-F1 | min_f1_1_4 |
|---|---:|---:|---:|
| Baseline | `0.8779` | `0.8529` | `0.5996` |
| CC-VREx | `0.8339` | `0.8001` | `0.5530` |
| Delta | `-0.0440` | `-0.0528` | `-0.0465` |

Conclusion:

- Smoke improvement was not stable under matched full 3-seed validation.
- Seed 52 collapsed enough to make the mean negative.
- This CC-VREx configuration is not a successful main candidate.

Primary artifacts:

- `reports/preserved_codex_isolated/codex_isolated/cc_vrex_focus14_e2_fullprep/reports/full_20260525_retry2/result_interpretation_20260529.md`
- `reports/preserved_codex_isolated/codex_isolated/cc_vrex_focus14_e2_fullprep/reports/full_20260525_retry2/summary.csv`
- `reports/preserved_codex_isolated/codex_isolated/cc_vrex_focus14_e2_fullprep/reports/full_20260525_retry2/summary.json`

### K1 Pair Margin Smoke

- Dataset/protocol: PA-CSI MultiEnv `dg_loeo`.
- Target environment: `E2`.
- Seed: `42`.
- Baseline: `pa_csi_dg_k1_nomargin_e2_smoke`.
- Candidate: `pa_csi_dg_k1_pairmargin_14_e2_smoke`.
- Selection metric: `min_f1_1_4`.

Result:

- Baseline accuracy: `0.8273`.
- Candidate accuracy: `0.8447`.
- Baseline macro-F1: `0.7792`.
- Candidate macro-F1: `0.8097`.
- Baseline `min_f1_1_4`: `0.5368`.
- Candidate `min_f1_1_4`: `0.3654`.

Conclusion:

- Accuracy increased, but class 1/4 robustness worsened.
- This pair-margin direction should not be treated as a win.

Primary artifact: `reports/tables/k1_pairmargin_14_e2_smoke_summary.csv`.

### OT Alignment Reports

- Dataset/protocol: PA-CSI MultiEnv `dg_loeo`.
- Target environment: `E2`.
- Seeds in full report: `42`, `52`, `62`.
- Training boundary: target E2 labels not used for selection or loss construction.
- Result status: full-run failure for this OT formulation; retained as negative evidence.

Primary artifacts:

- `reports/preserved_codex_isolated/codex_isolated/e2_14_ot_alignment_20260522/reports/final_mechanism_report.md`
- `reports/preserved_codex_isolated/codex_isolated/e2_14_ot_alignment_20260522/reports/final_mechanism_report_full_20260524.md`
- `reports/preserved_codex_isolated/codex_isolated/e2_14_ot_alignment_20260522/reports/full_run_result_analysis_20260525.md`
- `reports/preserved_codex_isolated/codex_isolated/e2_14_ot_alignment_20260522/reports/gate_decision_20260524.md`
- `reports/preserved_codex_isolated/codex_isolated/e2_14_ot_alignment_20260522/reports/phase0_manual_review.md`

### MMFI Weekly Report

- Source workspace: `C:\Users\howie\Desktop\mmfi-pa-csi-dg-workspace`.
- Preserved files copied into this repo:
  - `reports/mmfi_weekly_20260525_20260531/MMFI_WEEKLY_REPORT_2026-05-25_to_2026-05-31.md`
  - `reports/mmfi_weekly_20260525_20260531/MMFI_WEEKLY_REPORT_2026-05-25_to_2026-05-31.tex`
  - `reports/mmfi_weekly_20260525_20260531/MMFI_WEEKLY_REPORT_2026-05-25_to_2026-05-31.pdf`
  - `reports/mmfi_weekly_20260525_20260531/MMFI_WEEKLY_REPORT_2026-05-25_to_2026-05-31_EN.tex`
  - `reports/mmfi_weekly_20260525_20260531/MMFI_WEEKLY_REPORT_2026-05-25_to_2026-05-31_EN.pdf`
- Deleted from the separate MMFI workspace:
  - data caches
  - protocol outputs
  - scripts
  - configs
  - notes/docs outside `report/`

The MMFI report is kept as documentation only. It does not restore or commit the deleted MMFI experiment workspace.

