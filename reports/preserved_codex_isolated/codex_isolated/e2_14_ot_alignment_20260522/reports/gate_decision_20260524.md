# Gate Decision: E2 1/4 OT Alignment

## Available Results
- Phase 0 manual review found 0/20 obvious mislabel or semantic/rubric mismatches among mismatch rows, below the 5% gray-zone trigger.
- Phase 0.5 probes showed remaining E2 class 1/4 signal in at least two views:
  - source-fit LDA on projection embeddings: target E2 AUC mean 0.7196.
  - source-trained logistic on projection embeddings: target E2 AUC mean 0.7042.
  - fused-feature PCA logistic: target E2 AUC mean 0.7360.
  - raw temporal-mean probe was weak: target E2 AUC mean 0.5389.
- The 2-epoch smoke OT runs were negative on seed 42, but they are implementation smoke tests rather than the planned scientific gate.

## Failed Full-Run Attempt
- `full_20260524` did not produce metrics, checkpoints, or summaries.
- It only produced a partial `split_manifest.json` under the baseline output root.
- Cause: the PowerShell runner treated Python stderr as a terminating `NativeCommandError` before training could complete.
- This is an execution/launcher failure, not evidence that a scientific stopping condition was triggered.

## Decision
- Continue to the full matched run because Phase 0 and Phase 0.5 gate conditions are satisfied.
- Use a fresh retry namespace, `full_20260524_retry1`, to avoid overwriting the partial failed attempt.
- Do not run optional sensitivity or extra ablations until primary full-run summaries are available and reviewed.
