"""Audit newly sealed H attempts without touching an active training attempt.

Use the frozen backend's CPU checkpoint probe and prediction recomputation for
new runs. Previously verified runs retain their dated records and have their
full sealed artifact trees rechecked. Pair records preserve signed differences;
partial progress is never an aggregate conclusion or a final completion gate.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HROOT = ROOT / 'outputs_h_grouped_l1248/v2'
BUNDLE = HROOT / 'bundle'
EVIDENCE = ROOT / 'docs/revision/evidence'


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['PYTHONUTF8'] = '1'
    sys.path.insert(0, str(BUNDLE / 'code'))
    import common
    import run as frozen_run

    _, registry = common.verify_bundle(BUNDLE)
    digest = common.sha(BUNDLE / 'bundle_manifest.json')
    base = frozen_run.backend()
    jobs = frozen_run.jobs_for(base, BUNDLE, registry)
    runs_dir, pairs_dir = EVIDENCE / 'h_grouped_sealed_runs', EVIDENCE / 'h_grouped_verified_pairs'
    runs_dir.mkdir(exist_ok=True)
    pairs_dir.mkdir(exist_ok=True)
    verified, new_runs, new_pairs = {}, [], []
    for job in jobs:
        sealed = sorted(job.root.glob('attempt_*/attempt_seal.json'))
        if not sealed:
            continue
        common.require(len(sealed) == 1, f'Multiple sealed attempts: {job.job_id}')
        attempt = sealed[0].parent
        completion = common.read(attempt / 'attempt_completed.json')
        completion_hash = common.sha(attempt / 'attempt_completed.json')
        seal_hash = common.sha(sealed[0])
        common.require(common.read(sealed[0])['completion_manifest_sha256'] == completion_hash, 'Seal mismatch')
        common.require(base.tree_records(attempt, excluded=('attempt_completed.json', 'attempt_seal.json'))
                       == completion['artifacts'], f'Sealed artifact tree changed: {job.job_id}')
        report = runs_dir / f'{job.job_id}_attempt_{completion["attempt"]:03d}.json'
        if report.exists():
            record = common.read(report)
            common.require(record['completion_sha256'] == completion_hash and record['seal_sha256'] == seal_hash,
                           'Existing verified record differs from sealed attempt')
            common.require(record['job_id'] == job.job_id and record['bundle_manifest_sha256'] == digest,
                           'Existing verified record scope differs')
            health = record['health']
            common.require(base.payload_sha256(health) == completion['health_sha256'], 'Verified health differs')
        else:
            state = base.scan(job, Path(sys.executable))
            common.require(state.action == 'skip' and state.successful_attempt == attempt, 'Run did not verify')
            health = next(item['health'] for item in state.prior if item['status'] == 'success')
            schedule = common.read(attempt / 'payload/run/sampler_schedule.json')
            record = {'schema': 'h-grouped-sealed-run-verified/v1', 'verified_at': now(),
                      'job_id': job.job_id, 'attempt': completion['attempt'],
                      'completed_at': completion['created_at_utc'], 'bundle_manifest_sha256': digest,
                      'completion_sha256': completion_hash, 'seal_sha256': seal_hash,
                      'sealed_artifact_count': len(completion['artifacts']), 'health': health, 'sampler': schedule,
                      'cpu_strict_checkpoint_reload_repeated': True, 'target_predictions_recomputed': 3000,
                      'frozen_protocol_verified': True, 'aggregate_conclusion_available': False,
                      'active_training_unmodified': True}
            common.write_new(report, record)
            new_runs.append(job.job_id)
        verified[job.job_id] = {'job_id': job.job_id, 'completion_sha256': completion_hash,
                              'seal_sha256': seal_hash, 'health': health,
                              'test_metrics': common.read(attempt / 'payload/run/metrics.json')['test_metrics'],
                              'completed_at': completion['created_at_utc']}

    for target, seed in sorted({(job.target, job.seed) for job in jobs}):
        pair_jobs = [job for job in jobs if job.target == target and job.seed == seed]
        if not all(job.job_id in verified for job in pair_jobs):
            continue
        automatic = list((HROOT / 'runner_invocations').glob(f'*/pair_{target}_{seed}.json'))
        if not automatic:
            continue  # The runner may still be committing this pair's barrier.
        common.require(len(automatic) == 1, 'Multiple automatic pair barriers')
        barrier = common.read(automatic[0])
        rows = {job.arm: verified[job.job_id] for job in pair_jobs}
        common.require(barrier['status'] == 'passed' and len(barrier['arms']) == 2, 'Incomplete pair barrier')
        for arm in barrier['arms']:
            common.require(arm['health'] == rows[arm['arm']]['health'], 'Pair health differs')
        for field in ('initial_state_sha256', 'sampler_schedule_sha256'):
            common.require(rows['H10']['health'][field] == rows['H11']['health'][field], 'Unpaired initialization/sampling')
        common.require(rows['H10']['health']['split_hashes'] == rows['H11']['health']['split_hashes'], 'Unpaired split')
        signed = {key: 100 * (rows['H11']['test_metrics'][key] - rows['H10']['test_metrics'][key])
                  for key in rows['H10']['test_metrics']}
        path = pairs_dir / f'{target}_seed{seed}.json'
        if path.exists():
            old = common.read(path)
            common.require(old['runs'] == rows and old['signed_H11_minus_H10_percentage_points'] == signed,
                           'Existing signed pair record differs')
            common.require(old['automatic_pair_barrier_sha256'] == common.sha(automatic[0]), 'Pair barrier changed')
            continue
        record = {'schema': 'h-grouped-sealed-pair-verified/v1', 'verified_at': now(),
                  'target': target, 'seed': seed, 'bundle_manifest_sha256': digest,
                  'automatic_pair_barrier_sha256': common.sha(automatic[0]),
                  'paired_initialization_verified': True, 'complete_paired_sampler_verified': True,
                  'source_train_val_identity_disjoint': True, 'runs': rows,
                  'signed_H11_minus_H10_percentage_points': signed, 'population_inference_performed': False,
                  'aggregate_conclusion_available': False,
                  'limitation': 'One target/seed pair; preserve signed differences and await all nine pairs.'}
        common.write_new(path, record)
        new_pairs.append({'target': target, 'seed': seed, 'signed_macro_f1_pp': signed['f1_macro']})
    print(json.dumps({'verified_runs': len(verified), 'new_runs': new_runs, 'new_pairs': new_pairs,
                      'verified_pair_records': len(list(pairs_dir.glob('*.json'))),
                      'final_completion_claimed': False}, indent=2))


if __name__ == '__main__':
    main()
