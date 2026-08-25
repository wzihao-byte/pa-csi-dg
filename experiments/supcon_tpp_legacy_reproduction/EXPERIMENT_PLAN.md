# SupCon + TPP historical reproduction and coexistence plan

## Status and scope

Protocol: `supcon_tpp_legacy_reproduction` v1.0.0.

Current execution status (2026-08-04):

- full data/code/config preflight: passed;
- all four historical checkpoint replays: passed with zero metric difference and exact confusion matrices;
- H11 batch32/V4 GPU forward-backward: passed;
- seed-42 sentinel matrix: complete (12/12, zero failed jobs);
- seeds-52/62 full stage: complete (24/24, zero failed jobs);
- independent anchor completion audit: passed for all 36 cells; see
  `outputs_supcon_tpp_legacy_reproduction/audit/anchor_matrix_completion.json`;
- the historical SupCon performance-tolerance gate did not pass, but is
  advisory-only and non-blocking under the recorded 2026-07-23 user
  authorization; no parameters were changed in response;
- Factorial-S32 execution: complete (36/36, zero failed attempts); all nine
  target/seed quartet barriers passed with exact split and sampler pairing;
- independent Factorial-S32 completion audit: passed, including sealed artifact
  trees, strict checkpoint reload, 200 epochs/30,000 steps, 3,000 ordered
  predictions per job, independently recomputed metrics/confusions, complete
  positive-anchor coverage, and delayed target access;
- preregistered decision: H11 satisfies coexistence and the point-estimate
  positive-interaction rule, but does not exceed both single-component arms by
  the required one percentage point; see
  `outputs_supcon_tpp_legacy_reproduction/factorial_s32/analysis/`.
- final completion audit and the visually verified two-page weekly report are
  complete; see `outputs_supcon_tpp_legacy_reproduction/audit/final_completion.json`
  and `reports/weekly_report_20260804_supcon_tpp_legacy_reproduction/`.

The execution lock binds `protocol.py`, the bundle generator, preflight,
checkpoint replay, GPU smoke test and `run_anchors.py`, together with the two
passed gate artifacts.  Any later byte change fails closed until an explicit
new lock is generated and audited.

The previous `supcon_tpp_joint` formal matrix is retained as a protocol-drift result, but it is not used to answer whether the historical SupCon and TPP implementations coexist. No previous output is deleted or overwritten.

This plan deliberately separates three questions:

1. Can the auditable historical anchors still be reproduced?
2. Can the historical model definitions be compared under one common training carrier?
3. Does a frozen candidate survive a cleaner split/phase protocol and new environments?

The three result families must never be merged into one confidence interval.

## A. Locked evidence

### A1. Code

- SupCon/CE replay commit: `7b8ec0947846daef65c6c6f37b912f9eeb5efcd6`.
- SupCon/CE execute only from a detached clean worktree at that commit.
- TPP was never committed. Its current source/config snapshot is SHA-locked, but this does not repair missing historical code provenance.
- `tpp_l124` artifacts are complete, but the current unversioned TPP files are newer than those runs. It is an artifact reference, not a bitwise code reference.

### A2. Data and labels

- Environment order is exactly `E1,E2,E3`.
- Each amplitude/phase file is `[3000,850,90] float64`; each label file is `[3000] int32`.
- Feature width is `1 Rx × 3 Tx × 30 subcarriers = 90`.
- Correct label semantics are:
  - 0 no movement;
  - 1 falling;
  - 2 walking;
  - 3 sitting down / standing up;
  - 4 turning;
  - 5 picking up a pen.
- Older TPP reports swapped the names of labels 2 and 3. Numeric metrics remain usable; old names do not.

All nine file hashes, reconstructed split hashes and reference metrics are in `generated_v1/protocol_lock.json`.

### A3. Historical-only preprocessing

Anchor replay retains `unwrap_phase=true, phase_unwrap_axis=-1` and source sample-stratified validation. This is required for compatibility with the auditable historical output. It is not endorsed as the final scientific preprocessing protocol.

Target metrics may only check a fully preregistered replay. They may not select lambda, architecture, checkpoint rule or training carrier.

## B. Anchor replay

### B1. Anchor-S: historical SupCon

- `PaCsiDGLite`, 2,788,918 trainable parameters.
- Global projection plus three Tx-masked views: `[B,4,128]`.
- Pairwise SupCon; `global_class_instance_views`; include same-domain same-class positives.
- `lambda=0.05`, `temperature=0.2`, no warm-up.
- Batch 32; domain-class-balanced; 150 steps/epoch; 200 epochs; 30,000 optimizer steps.
- Adam `1e-3`, weight decay 0, exponential per-step decay `0.9/10000`.
- Source-val accuracy checkpoint; `>=` means the last tied best epoch is retained.

Primary reference: complete 3 targets × 3 seeds, overall macro-F1 `0.8444273710435826`.

### B2. Anchor-CE and Anchor-T

- CE and TPP historical carrier: batch 8, ordinary shuffled loader, 600 steps/epoch, 120,000 optimizer steps.
- The data exposure is still 4800 samples × 200 epochs = 960,000.
- Complete TPP primary reference is plain `tpp_l124`, overall macro-F1 `0.7848034699308416`.
- Residual-0.25 TPP is the architecture candidate, but its old matrix contains only seeds 52/62. The three seed-42 cells must be completed before it can become a full anchor.

Residual-0.25 is dual-stream and dual-path:

- amplitude and phase;
- temporal and channel paths;
- max pyramid levels `[1,2,4]`;
- `global + 0.25 × projected TPP`.

### B3. Replay order and gates

1. Full preflight and historical-checkpoint re-evaluation.
2. One preregistered seed-42 sentinel for each anchor family.
3. Only after the sentinel passes, finish the complete anchor matrix.

SupCon scientific replay tolerances:

- per-run absolute accuracy difference ≤ 0.0033333;
- per-run absolute macro-F1 difference ≤ 0.005;
- per-target mean macro-F1 difference ≤ 0.003;
- overall mean macro-F1 difference ≤ 0.002.

Failure is diagnostic. It does not authorize changing a parameter or overwriting the failed attempt.

## C. Common-carrier factorial

The historical anchors use different optimization carriers, so a fair four-arm comparison cannot also be an exact replay of both. The common matrix is named `Factorial-S32`.

| Arm | Classifier representation | SupCon |
|---|---|---|
| H00 | historical global pooling | off |
| H01 | historical global pooling | historical V=4, lambda 0.05 |
| H10 | dual-stream residual-0.25 TPP | off |
| H11 | dual-stream residual-0.25 TPP | historical V=4, lambda 0.05 |

All four arms use the SupCon carrier: batch32, domain-class-balanced, 200 epochs and 30,000 steps. H10 is therefore “historical TPP architecture under the SupCon carrier”, not an exact Anchor-T replay.

Parameter counts:

- H00/H01: 2,788,918;
- H10/H11: 3,721,334;
- TPP increment: 932,416 (+33.43%);
- SupCon on/off does not change parameter count within an architecture pair.

The first H11 is the fully coupled implementation already supported by `IsolatedPaCsiDGLiteTPP`: both the full view and the three Tx views pass through TPP.

If fully coupled H11 is unhealthy, the next preregistered architecture is contrastive decoupling:

\[
h_{cls}=g+r_{TPP},\qquad z_{con}=q(g),
\]

so the TPP projection receives CE gradients but no direct SupCon gradient. This is a new model and requires a new protocol version; it cannot replace a failed H11 in place.

### C1. Factorial gates

- H11 batch32/V=4 full forward-backward must fit the 16 GiB GPU without arm-specific batch changes.
- H01/H11 positive-anchor coverage must be 100%.
- Every arm has 150 steps/epoch and 30,000 optimizer steps.
- Same `(target,seed)` blocks share the split and sampler schedule.
- Checkpoint is frozen, hashed and reloaded before target files are opened.
- Every target evaluation contains exactly 3000 ordered predictions, probabilities, a 6×6 confusion matrix and per-class metrics.

Coexistence requires H11 to be within -1 pp overall of both single components, no environment more than 2 pp below the stronger component, and pick-up-pen F1 within the locked margins. “Better” additionally requires at least +1 pp over both single components. Positive interaction is

\[
I=H11-H10-H01+H00 \ge 1\text{ pp}.
\]

## D. Scientific confirmation

E1-E3 have already been inspected repeatedly. They are suitable for replay and mechanism analysis, not a new unbiased DG claim.

After a candidate is frozen:

1. change group split only;
2. change phase preprocessing only;
3. confirm once on unseen E4/E5 or independently collected environment/subject data.

The legacy and clean-protocol results remain separate.
