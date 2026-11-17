# Three-class 4 s study (falling / walking / turning)

Requested by Prof. Ohtsuki (2026-11): compare all four configurations on falling, walking and turning alone, training and evaluating a three-class classifier with the same environment and participant splits, to test whether the gains persist when every evaluated class has the same segment duration.

## Protocol (frozen 2026-11-17, before any training)

- **Classes:** original labels 1 (falling), 2 (walking) and 4 (turning), remapped to 0, 1 and 2. Every other segment is removed from source training, source validation and the target.
- **Splits:** leave-one-environment-out over E1–E3. The main-study H-S participant-disjoint source manifests are restricted to the three classes. That gives 1,920 training and 480 validation segments over the two source environments, and 1,200 target segments. Global sample identities match the main study.
- **Arms:** G-CE, G-SC, T-CE and T-SC, the same definitions as the main study.
  - Capacity-matched global MLP or residual TPP.
  - CE: `supcon_views=1`, λ = 0.
  - Multiview SupCon: `supcon_views=4`, λ = 0.05, hybrid positives, τ = 0.2.
  - 4,785,523 parameters in every arm (the classifier has 3 outputs).
  - The shared tensors are initialized identically across arms within each seed.
- **Seeds and runs:** 42, 52 and 62, for 36 runs. The queue order is seed → target → arm, so every finished block is a complete four-arm comparison.
- **Budget:** the same 200 epochs as the main study. The domain/class-balanced sampler gives 60 batches per epoch, so each run has 12,000 updates. Batch 32 (16 per source environment); Adam at 1e-3 with step decay 0.9^(t/10⁴).
- **Selection:** the last tied-best source-validation-accuracy checkpoint; the target is scored once after the checkpoint is frozen.
- **Pre-specified criterion:** a gain persists if the paired mean is > 0 and at least 7 of 9 target–seed pairs improve. The two primary comparisons are T-SC − T-CE and T-CE − G-CE. All four comparisons are reported whatever the outcome.

## Pre-check (`probe_duration_three_class.py` → `probe_three_class.json`)

Random-forest leave-one-environment-out macro-F1 on the three classes:

| Feature | Macro-F1 |
|---|---|
| Duration only | 34.7% |
| Packet count only | 30.0% |
| Duration + packet count | 35.4% |

All three are close to the 33.3% uniform-random reference. Segment durations are 3.99–4.02 s, except a few E2 segments of up to 5.9 s.

## Files

| File | Purpose |
|---|---|
| `three_class.py` | Class restriction, split, sampler replay and dataset subsetting |
| `prepare.py` | Freezes the bundle at `D:/pa-csi-dg-revision-runtime/studies/h_three_class_4s_20261117_v1/bundle` |
| `qa.py` | Source-only gate: real-data split and sampler replay, two real optimizer steps per arm |
| `run.py` | Serial queue (`--dry-run` / `--execute`) |
| `analyze.py` | Writes `analysis/results.json` after every finished job |

The trainer and runner are the frozen main-study v2 code. The six-class invariants are changed only by the counted replacements listed in `prepare.py` (`PATCHES`).

To stop the queue after the current job, create `queue/stop_after_current.json`. To resume, rerun `run.py --execute`; finished runs are skipped.
