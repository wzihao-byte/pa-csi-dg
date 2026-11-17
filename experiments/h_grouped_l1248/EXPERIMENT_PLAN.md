# Original H L1248 grouped source validation supplement

Completed and verified on September 21, 2026: 18/18 sealed runs and 9/9 matched pairs. All signed deltas, including E1/seed42 harm, are retained. The H-only supplement is integrated into the approved 24-page H/Y PDF on September 24. No training restart is required or permitted. The execution descriptions below preserve the frozen protocol and historical recovery sequence.

Authorized by the user on 2026-09-10: retain the original H10/H11 L1248 experiment and align parameters, replacing only source train/validation membership with verified grouped membership. The user subsequently explicitly cancelled Widar experiments. This is an additional 18-run study alongside the retained 180-run MultiEnv R studies (36 core, 81 controls, 63 inputs). The old 216-run plan and its Widar-dependent completion/release gates are superseded and must not be used to declare the current scope complete.

## Fixed question and grid

Does the original SupCon increment remain under source validation that shares neither subjects nor acquisitions with source training? Run E1/E2/E3 outer targets × seeds 42/52/62 × H10/H11, in that fixed order (18 runs). H10 is original L1248 without SupCon; H11 adds the original pairwise multi-view SupCon. No new level/weight/seed search or target-driven retries.

Both arms use exactly the original generated configuration that is bound to the corresponding completed H10/H11 attempt. Excluding output routing, the only changed configuration leaves are experiment_name, job_id, protocol.version, and the two source split hashes. Extra h_grouped fields record provenance and QA. The actual scientific change is source training and validation membership together, with sample sizes fixed. The R code/model/optimizer are not substituted.

## Alignment locked before training

| Item | Frozen setting |
|---|---|
| Model | Original dual-stream gated model, temporal and channel L=[1,2,4,8], max pooling, preserve global, alpha=0.25 |
| Parameters | H10 = H11 = 4,786,294; identical initial state per paired seed |
| Input | Same original amplitude/phase arrays and hashes; original float64 phase unwrap along axis -1 then float32 conversion |
| Views | Original mode=tx, num_rx=1, num_tx=3, num_subcarriers=30: three pseudo-Tx feature-group views; no claim of three physical transmitters |
| SupCon | H10 lambda=0, one view; H11 lambda=0.05, tau=0.2, four views, original pairwise mode and masks, no warmup |
| Training | 200 epochs, batch 32, 150 steps/epoch, 30,000 optimizer steps, 960,000 ordered sample draws |
| Optimizer | Original Adam lr=0.001, weight decay=0; original continuous per-step exponential decay 0.9**(step/10000) |
| Checkpoint | Highest source validation accuracy, last tied best; patience 200, original optimizer and trainer |
| Randomness | Original Python continuous RNG sampler; both arms share the full frozen sample order and initialization. Extra H11 forwards retain their original Torch dropout randomness |
| Split | Existing verified r_core_v1 source membership used only as identity data: 4,800 train/1,200 validation, 16/4 subjects, 1,600/400 acquisitions; no overlap; fixed partition across seeds |
| Target | Original 3,000 target rows evaluated after checkpoint is frozen and strictly reloaded; original target metrics do not choose checkpoints |

Source identity manifests and sidecars, historical configurations, all nine complete sample schedules, exact legacy trainer/runner snapshots and adapter code are hashed in the new bundle. Actual old model, loss and loader dependencies retain their original provenance hashes. Each run verifies source array hashes, actual initialization, actual split and full sampler digest. Completion requires 200 epochs, exact update counts, strict checkpoint reload, 3,000 saved predictions, recomputed metrics/confusions and a sealed artifact tree. Each paired-arm barrier rechecks identical split and sampler evidence.

## QA and execution

Before formal GPU training: replay a sealed historical sampler as an anchor; independently verify all nine grouped schedules; reject parameter drift, identity overlap, mixed target/source loading and target access before checkpoint freeze; compare the historical backend and the adapter on two real full source batches per arm on CPU, including initial and post-update model/optimizer/scheduler/RNG state hashes. This QA is not experimental evidence and does not change formal settings.

Run only the frozen copies under outputs_h_grouped_l1248/v2/bundle/code. The unlaunched v1 candidate is retained as superseded evidence; v2 removes the Widar scheduling dependency without changing scientific parameters. Preparation freezes once; QA approval is bound to that manifest. The detached coordinator waits for the identified existing MultiEnv input coordinator to exit successfully, checks its bound completion/evaluation, and waits for the GPU to be idle before starting these 18 jobs. The waiting Widar coordinator has been stopped by explicit user instruction; MultiEnv training subsequently completed; these scheduling statements record the pre-training protocol.

On 2026-09-11 the original waiting process was found interrupted before H training began. Its owner record was preserved, and `analysis/recover_h_grouped_l1248_queue.py` created a new exclusive queue owner bound to the recovered `controls_to_inputs_v3` process. The recovery reverified the unchanged v2 bundle, its QA approval, and an untouched 18-job dry-run plan. The frozen `run.py` directly rechecks its original, interrupted process identity, so the recovered queue uses `analysis/execute_h_grouped_l1248_recovery.py` to replace only that prerequisite gate. The wrapper imports the unchanged frozen model, trainer, sampler, sealed-attempt runner, result finalizer, and paired analysis. It still waits for the new input coordinator's bound completion and GPU-idle gate before execution. The interruption and new process identities are recorded in `docs/revision/evidence/runtime_recovery_20260911.json` and its addenda.

Attempts are exclusive and retained. A failed or interrupted run may be restarted only from its original initialization in a new numbered attempt after checking that no previous owner/child remains active; the legacy trainer has no optimizer/RNG resume. Never warm-start from its best-model file, overwrite an attempt, change scientific settings after viewing results, or silently remove a failed seed. A failed upstream queue stops this supplement and leaves failure evidence; it is not bypassed automatically.

Outputs live in the separate outputs_h_grouped_l1248/v2 tree on C:. No original H or R output is rewritten. Existing Widar-dependent publication automation is obsolete and does not incorporate this supplement automatically. The verified supplement is now integrated into the H/Y release manuscript, with claims restricted to the available MultiEnv evidence; see docs/revision/COMPLETION_AUDIT.md for publication gates.

## Reporting and limits

Primary descriptive comparison: H11 minus H10 target macro-F1 at each matched target/seed. Also report accuracy, all seed values, per-target means and sample SD across the three training seeds, and equal-environment averages per seed. Report all signed deltas including unfavorable results. Compare each run to its exact historical H reference in a separate table. Historical versus grouped differences reflect a change in both source train and validation membership; they do not isolate validation selection alone.

These three seeds measure training randomness on one fixed partition, not uncertainty across independent subjects, split draws or environments. E1–E3 were used previously in method development; delayed checkpoint evaluation does not turn them into a new independent confirmation set. This two-arm study tests the SupCon increment within L1248. It does not by itself estimate the full 2×2 interaction, eliminate every capacity explanation, establish physical-antenna invariance or provide an independent cross-dataset confirmation. Widar is excluded by user instruction. Do not infer population p-values or confidence intervals from these nine target-seed cells.
