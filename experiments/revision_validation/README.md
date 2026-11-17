# Revision validation workspace

This directory is separate from the frozen H/Y studies. E1–E3 remain development
environments. The first new candidates use the historical amplitude arrays,
without phase or a claim of corrected physical time.

Run from the repository root, using the project's NumPy/SciPy/PyTorch runtime:

```powershell
python analysis/verify_multienv_raw_mapping.py --limit 0 --workers 6
python analysis/audit_multienv_timestamps.py
python -m experiments.revision_validation.prepare_verified_protocol
python analysis/audit_h_subject_splits.py
python -m experiments.revision_validation.run_source_smoke --target all --steps 30
python analysis/summarize_verified_revision.py
```

The protocol compiler requires all 9000 unique original-signal matches, rehashes
the nine processed files, verifies the sidecars using the existing loader
validators, and checks the frozen Y manifests' exact sample-set hashes. Source
manifests contain source paths and rows only. Target manifests are separate.

`models.py` defines four amplitude-only candidates: R00/R01 use conventional
global MLP pooling; R10/R11 use L124 residual TPP. The second bit enables
global-only SupCon. All four have 1,878,142 total parameters. The common encoder
and classifier start with identical weights for the same seed. Equal projection
parameter/MAC counts do not imply equal function classes or residual norms.

`source_data.py` memory-maps only source amplitude, uses labels from the verified
manifest, and builds deterministic, class/domain-balanced batches. Sampling
does not share RNG state with the model. A 300-update epoch is not an exhaustive
pass over the training rows.

`run_source_smoke.py` exercises all arms and records initialization agreement,
gradient activity, positive anchors, batch schedules, initial residual norms,
counted MACs, inference latency and training throughput. It uses full λ=.025 for
its short diagnostic, rather than the full-run warmup. It reports no efficacy
score and never loads target signals or labels.

The full source-only training entry point is:

```powershell
python -m experiments.revision_validation.train_source --target E1 --arm R11 --seed 401 --epochs 200 --lambda-supcon 0.025 --output outputs_revision_validation/source_E1_R11_401/attempt_001
```

This example is a command template, not evidence that the full run was executed.
The trainer implements Adam, 60,000 updates at batch 16, continuous learning-rate
decay, a 6000-update contrastive warmup, full source validation each epoch,
macro-F1 selection and earliest-tie retention. It refuses to overwrite a previous
attempt. For an execution check, set `--epochs 1` and use a separate pilot output
directory; pilot results cannot enter the formal comparison.

The completed one-epoch pilot is at
`outputs_revision_validation/pilot_E1_R11_401/attempt_001/`. Its saved checkpoint
was strictly reloaded and all 1200 source-validation predictions and metrics
were reproduced by `python analysis/verify_revision_source_pilot.py`. Repeats
must use a new attempt directory. The 12 short smoke runs are documented in
`docs/revision/SOURCE_SMOKE_REPORT.md`.

Before starting the full matrix, freeze its configurations, source-search budget,
code hashes, complete run registry and analysis rule. Keep both λ candidates
(.01/.025) under an equal source-only budget if performing a new search; do not
use target scores to select them. Use `EXPERIMENT_PLAN.md` for remaining controls.
