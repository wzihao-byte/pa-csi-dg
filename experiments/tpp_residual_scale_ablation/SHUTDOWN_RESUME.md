# Planned shutdown and resume record

The experiment was safely paused on 2026-08-12 at 21:57 JST for a planned
computer shutdown.

## Preserved state

- 10 jobs are sealed successful and will be validated and skipped.
- No job is marked failed.
- `h11_l1248_scale0750__target_E1__seed_62/attempt_001` is preserved as a
  planned interruption after 140 complete epochs.
- The interrupted trainer did not save optimizer/scheduler state.  To keep
  the locked protocol valid, this one cell must restart from epoch 1 as the
  append-only `attempt_002`; its partial `attempt_001` is retained for audit.
- The remaining 25 jobs have not started.
- The live runner lock was moved to the planned-pause evidence directory.
- No experiment Python process remains, so the computer is safe to shut down.

Authoritative evidence:

- `outputs_tpp_residual_scale_ablation/h11_l1248_v1/planned_pause_20260812/planned_pause.json`
- `outputs_tpp_residual_scale_ablation/h11_l1248_v1/planned_pause_20260812/planned_pause_intent.json`
- `outputs_tpp_residual_scale_ablation/h11_l1248_v1/planned_pause_20260812/runner_lock_snapshot.json`
- `outputs_tpp_residual_scale_ablation/h11_l1248_v1/runs/E1/seed_62/scale_0750/H11/attempt_001/planned_interruption.json`

## Resume after power-on

From the repository root, first perform a read-only recovery check:

```powershell
python experiments\tpp_residual_scale_ablation\resume_planned.py --check-only
```

It must report:

- `completed_jobs_preserved: 10`
- `planned_reruns: 1`
- `new_jobs_remaining_after_rerun: 25`
- paused job `h11_l1248_scale0750__target_E1__seed_62`
- the planned rerun uses `next_attempt: 2`

Then resume explicitly:

```powershell
python experiments\tpp_residual_scale_ablation\resume_planned.py --execute
```

Alternatively, run `resume_after_power_on.cmd` from this directory.

The resume code refuses to proceed if the bundle hash, pause record,
interruption record, existing successes, attempt numbering, or runner lock
does not match the sealed recovery plan.  It never overwrites attempt 001.
