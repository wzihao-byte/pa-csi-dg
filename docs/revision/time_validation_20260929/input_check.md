# Time preprocessing input verification — 2026-09-29

**PASS.** Read-only CPU checks are complete. No training, target scoring, download, array modification, or row exclusion was performed.

## Coverage

- Rehashed all 28 frozen compiled artifacts, including six amplitude arrays. Each B/C variant contains 9,000 rows; each environment array is float32 `[3000, 850, 90]`, with zero nonfinite values.
- Verified all 9,000 preparation-record hashes and bindings to the preparation config, source code, compile config and prior reconstruction audit. Historical amplitude hashes also match.
- Matched historical sample IDs, order, labels, subject IDs and acquisition IDs for every source/validation/target partition. Source train/validation subject and acquisition sets remain disjoint.
- Independently reconstructed 8 selected local DAT/MAT records, including the two historical timestamp-problem records and three largest-gap records. Both B/C inputs exactly match the stored float32 rows; historical replay error is below 1e-10.

## What B and C mean

| Version | Time basis | Output rule | Interpretation |
|---|---|---|---|
| B: native_time_legacy_merge | Native uint32 DAT elapsed time | Original merge loop and reverse-tail fill | Timestamp-basis sensitivity; 850 indices are not a strict physical-time grid |
| C: native_time_linear | Same native elapsed time | Linear interpolation onto `linspace(0, duration, 850)` | Resampling-rule sensitivity; record-specific spacing, not a fixed Hz |

## Clock and gap findings

- Across 9,000 bound provenance records: zero duplicate timestamp steps and zero uint32 wraps. Record durations range from 2.990493 to 5.905944 seconds.
- 62 records contain a gap over 0.05 s; 4 over 0.1 s; 3 over 0.5 s; 2 over 1 s. These are descriptive thresholds, not exclusion criteria.
- Longest gap: **2.015493 s**, E2:0158, inside a 5.905944 s recording. C interpolates across this interval without a cutoff or mask. This validates the implementation and its limitations; it does not demonstrate recovery of the missing signal.

## Bound existing all-row comparison

| Comparison (float32 model inputs) | Exactly equal rows | Changed rows above 1e-5 element error |
|---|---:|---:|
| historical_vs_native_merge | 5256 / 9000 | 3744 / 9000 |
| historical_vs_native_linear | 0 / 9000 | 9000 / 9000 |
| native_merge_vs_native_linear | 0 / 9000 | 9000 / 9000 |

## Decision for the proposed experiment

No input-integrity blocker was found for the planned B/C global-MLP controls. Keep the frozen arrays, all rows and the matched training protocol unchanged; compare each new global model with its corresponding existing TPP model. Treat the result as preprocessing sensitivity, not a claim that native-time interpolation is universally correct.

The large gaps and the historical merge behavior must be disclosed when interpreting the difference. A gap-aware reconstruction would be a new preprocessing version and is outside the current frozen comparison.

Full checks, hashes, reconstruction examples, exact gap locations and provenance coverage are recorded in `input_check.json`.
