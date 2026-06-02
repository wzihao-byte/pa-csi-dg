# Phase 0 Manual Review

## Method
- Reviewed the blinded heatmap packet first, then opened `private_key.csv` only for direction-level aggregation.
- This is an image-only CSI audit by Codex, not a human activity-rubric adjudication. I can flag obvious corruptions, visually ambiguous samples, and lack of visible label/rubric evidence, but cannot prove the real-world activity semantics from heatmaps alone.

## Outputs
- Filled review labels: `analysis\phase0_manual_review_packet\codex_manual_review_labels.csv`
- Contact sheet: `analysis\phase0_manual_review_packet\contact_sheet_blind.png`
- Original heatmaps: `analysis/phase0_manual_review_packet/heatmaps/`

## Summary
- Mismatch rows reviewed: 20
- Control rows reviewed: 20
- Obvious mislabel or semantic/rubric mismatch among mismatches: 0/20 = 0.0%
- Wilson 95% CI for obvious/semantic problem rate: [0.0%, 16.1%]
- Ambiguous mismatch rows: 3/20 = 15.0%
- Ambiguous control rows: 1/20 = 5.0%
- Clean model-error-looking mismatch rows: 17/20 = 85.0%

## Direction Summary
- `1->4`: n=10, obvious/semantic=0, ambiguous=1, model_error_like=9
- `4->1`: n=10, obvious/semantic=0, ambiguous=2, model_error_like=8

## Decision
- Observed obvious/semantic issue rate is below the 5% Phase 0 gray-zone trigger, so this audit does not block Phase 2.
- Because the CI is wide and the audit is image-only, treat this as evidence against dominant label/semantic drift, not as proof that labels are perfect.
- Proceeding to full OT is reasonable if we keep the conclusion exploratory and interpret failures as representation/alignment evidence, not as a definitive DA benchmark.

## Ambiguous Rows
- `R004` (mismatch, 1->4): low-confidence transitional heatmap; not enough to call label/rubric issue
- `R018` (mismatch, 4->1): lower-confidence smooth/transition morphology; not enough to call label/rubric issue
- `R028` (mismatch, 4->1): lowest-confidence mismatch in packet; mixed evidence, not label/rubric proof
- `R036` (control, control): control but low model confidence and mixed probabilities; mark as ambiguous control
