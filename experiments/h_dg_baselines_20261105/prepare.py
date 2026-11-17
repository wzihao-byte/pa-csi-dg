"""Freeze the 18-run CORAL / Mixup baseline study before any new target scoring."""
from __future__ import annotations
import copy, json, shutil, sys
from pathlib import Path
import torch
from common import (ARMS, ARM_POOL, BUNDLE, GROUP, HERE, JOBS, PAIRED_BUNDLE, PAIRED_MANIFEST_SHA256, PAIRED_STUDY,
                    PARAMETERS, SEEDS, TARGETS, base_module, now, read, require, sha, state_hash, verify_config, write)
from dg_baselines import arm_specification
# control_components is reused byte-for-byte from the paired frozen bundle (copied into this bundle below).
sys.path.append(str(PAIRED_BUNDLE / 'code'))
from control_components import make_model_class  # noqa: E402

NEW_CODE = ('common.py', 'dg_baselines.py', 'train_factorial.py', 'prepare.py', 'run.py', 'qa.py', 'analyze.py')
PAIRED_CODE = ('control_components.py', 'base_trainer.py', 'base_runner.py', 'grouped_helpers.py')
PAIRED_RECORDS = (('initialization_alignment.json', 'paired_initialization_alignment.json'),
                  ('historical_references.json', 'paired_historical_references.json'))
DECISION_RULE = {
    'purpose': 'Same-protocol generic domain-generalization baselines for the ICAIIC 2027 manuscript',
    'arms': {'GC10': 'global MLP pooling (capacity-matched ERM backbone, as G10) + CE + DomainBed CORAL (gamma 1.0) on the fused 256-d feature',
             'GX10': 'global MLP pooling + CE + DomainBed inter-domain Mixup (alpha 0.2) on amplitude and phase',
             'C10': 'TPP + CE + DomainBed CORAL (gamma 1.0) on the fused 256-d feature',
             'X10': 'TPP + CE + DomainBed inter-domain Mixup (alpha 0.2) on amplitude and phase'},
    'primary_question': 'Does TPP + multiview SupCon (HS11) exceed ERM (G10) and generic DG methods applied to the same plain backbone (GC10, GX10)?',
    'secondary_question': 'Given the TPP representation, does multiview SupCon (HS11) exceed generic DG regularizers (C10, X10)?',
    'paired_with': 'completed v2 G10/G11 and historical H-S T-CE (HS10) / T-SC (HS11) runs of the same target and seed',
    'reported': ['per-arm environment means and equal-environment means', 'nine paired differences per contrast',
                 'sign counts', '95% percentile bootstrap interval over the nine pairs (10,000 resamples, seed 20261105)'],
    'contrasts': ['GC10 - G10', 'GX10 - G10', 'HS11 - GC10', 'HS11 - GX10', 'C10 - HS10', 'X10 - HS10', 'HS11 - C10', 'HS11 - X10'],
    'interpretation': 'descriptive; no configuration of the main method is changed on the basis of these results',
    'caveat': 'Nine pairs share three development environments; intervals are descriptive, not tests of independent environments',
    'fixed_before_target_scoring': True,
}

def verify_paired_study():
    require(sha(PAIRED_BUNDLE / 'bundle_manifest.json') == PAIRED_MANIFEST_SHA256, 'Paired control bundle manifest changed')
    manifest = read(PAIRED_BUNDLE / 'bundle_manifest.json')
    for rel, expected in manifest['files_sha256'].items():
        require(sha(PAIRED_BUNDLE / rel) == expected, 'Paired control bundle file changed: ' + rel)
    status = read(PAIRED_STUDY / 'queue/status.json')
    require(status['status'] == 'complete' and status['completed_jobs'] == 54, 'Paired control queue incomplete')
    completed = read(PAIRED_STUDY / 'queue/completed_jobs.json')
    require(completed['status'] == 'complete' and len(completed['completed']) == 54, 'Paired completion record incomplete')
    attempts = {}
    for record in completed['completed']:
        attempt = Path(record['attempt'])
        require(sha(attempt / 'attempt_completed.json') == record['completion_sha256'], 'Paired completion drift')
        require(read(attempt / 'attempt_seal.json')['completion_manifest_sha256'] == record['completion_sha256'], 'Paired seal drift')
        attempts[record['job_id']] = record
    references = {}
    for r in read(PAIRED_BUNDLE / 'historical_references.json')['references']:
        attempt = Path(r['attempt'])
        require(sha(attempt / 'attempt_completed.json') == r['completion_sha256'], 'Historical reference completion drift')
        require(sha(attempt / 'payload/run/metrics.json') == r['metrics_sha256'], 'Historical reference metrics drift')
        require(sha(r['config']) == r['config_sha256'], 'Historical reference config drift')
        references[(r['target'], r['seed'], r['arm'])] = r
    require(len(references) == 18, 'Need all 18 historical H-S TPP references')
    return manifest, attempts, references

def copy_checked(source, dest, expected):
    require(sha(source) == expected, 'Copied source differs from frozen digest: ' + str(source))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    require(sha(dest) == expected, 'Copy changed bytes: ' + str(dest))
    return dest

def main():
    require(not BUNDLE.exists(), 'Preserve any previously frozen bundle; use a new version for changes')
    require(all((HERE / x).is_file() for x in NEW_CODE), 'DG-baseline implementation incomplete')
    torch.set_num_threads(2)
    paired_manifest, paired_attempts, references = verify_paired_study()
    files = paired_manifest['files_sha256']
    BUNDLE.mkdir(parents=True)
    code = BUNDLE / 'code'; code.mkdir()
    for name in NEW_CODE:
        shutil.copy2(HERE / name, code / name)
    for p in (HERE / 'tests').glob('test_*.py'):
        (code / 'tests').mkdir(exist_ok=True); shutil.copy2(p, code / 'tests' / p.name)
    for name in PAIRED_CODE:
        copy_checked(PAIRED_BUNDLE / 'code' / name, code / name, files['code/' + name])
    for rel, digest in files.items():
        if rel.startswith(('code/legacy/', 'code/tpp/')):
            copy_checked(PAIRED_BUNDLE / rel, BUNDLE / rel, digest)
    for source_name, dest_name in PAIRED_RECORDS:
        copy_checked(PAIRED_BUNDLE / source_name, BUNDLE / dest_name, files[source_name])
    initialization = read(BUNDLE / 'paired_initialization_alignment.json')
    # Re-derive the frozen TPP initial state for every seed from the bundled code (CPU, no data).
    base = base_module()
    cells, configs, jobs = {}, {}, []
    for target in TARGETS:
        manifest_rel = f'manifests/source_{target}.json'
        source_manifest = copy_checked(PAIRED_BUNDLE / manifest_rel, BUNDLE / manifest_rel, files[manifest_rel])
        for seed in SEEDS:
            schedule_rel = f'schedules/HS_{target}_{seed}.npy'
            schedule = copy_checked(PAIRED_BUNDLE / schedule_rel, BUNDLE / schedule_rel, files[schedule_rel])
            reference_paths = {}
            for arm in ('H10', 'H11'):
                rel = f'references/HS_{target}_{seed}_{arm}.json'
                reference_paths[arm] = copy_checked(PAIRED_BUNDLE / rel, BUNDLE / rel, files[rel])
            g10_rel = f'configs/HS_{target}_G10_{seed}.json'
            g10 = read(copy_checked(PAIRED_BUNDLE / g10_rel, BUNDLE / 'paired_controls' / Path(g10_rel).name, files[g10_rel]))
            tce, tsc = references[(target, seed, 'H10')], references[(target, seed, 'H11')]
            reference = read(reference_paths['H10'])
            require(reference['h_grouped']['expected_sampler_sha256'] == g10['h_controls']['expected_sampler_sha256'],
                    'Paired T-CE and G10 sampling differ')
            audits = initialization[str(seed)]
            require(audits['tpp']['model_full_state_sha256'] == reference['h_grouped']['initial_state_sha256'],
                    'Paired TPP initial state differs from the historical T-CE seed')
            require(audits['global_mlp']['model_full_state_sha256'] == g10['h_controls']['initial_state_sha256'],
                    'Paired global initial state differs from the v2 G10 run')
            require(audits['tpp']['model_common_state_sha256'] == audits['global_mlp']['model_common_state_sha256'], 'Shared tensors differ')
            modules = base.configure_imports(reference)
            for pool in ('tpp', 'global_mlp'):
                modules['train_dg'].set_seed(seed)
                model = make_model_class(modules['tpp_models'].IsolatedPaCsiDGLiteTPP, pool)(90, 6, reference['model'])
                require(sum(p.numel() for p in model.parameters()) == PARAMETERS, 'Capacity drift')
                require(state_hash(model) == audits[pool]['model_full_state_sha256'], pool + ' initial state could not be re-derived')
            key = f'{target}_{seed}'
            cells[key] = {'target': target, 'seed': seed,
                'reference_h10_sha256': sha(reference_paths['H10']), 'reference_h11_sha256': sha(reference_paths['H11']),
                'schedule_sha256': sha(schedule), 'source_manifest_sha256': sha(source_manifest),
                'expected_sampler_sha256': g10['h_controls']['expected_sampler_sha256'],
                'initial_state_sha256': {pool: audits[pool]['model_full_state_sha256'] for pool in ('tpp', 'global_mlp')},
                'common_state_sha256': audits['tpp']['model_common_state_sha256'],
                'tce_job_id': read(tce['config'])['job_id'], 'tce_attempt': tce['attempt'],
                'tce_metrics_sha256': tce['metrics_sha256'], 'tsc_job_id': read(tsc['config'])['job_id'],
                'tsc_attempt': tsc['attempt'], 'tsc_metrics_sha256': tsc['metrics_sha256'],
                'g10_job_id': g10['job_id'], 'g10_attempt': paired_attempts[g10['job_id']]['attempt'],
                'g10_config_sha256': files[g10_rel]}
    write(BUNDLE / 'paired_cells.json', cells, exclusive=True)
    # Queue order: both global-backbone arms for every cell first, then the TPP-backbone arms.
    for arms in (ARMS[:2], ARMS[2:]):
      for target in TARGETS:
        for seed in SEEDS:
            cell = cells[f'{target}_{seed}']
            reference_path = BUNDLE / f'references/HS_{target}_{seed}_H10.json'
            for arm in arms:
                pool = ARM_POOL[arm]
                config = copy.deepcopy(read(reference_path))
                config.pop('h_grouped', None)
                provenance = config['provenance']
                for name, record in provenance['legacy_modules'].items():
                    record['path'] = str(BUNDLE / 'code/legacy' / (name + '.py'))
                    require(sha(record['path']) == record['sha256'], 'Bundled legacy module differs: ' + name)
                provenance['clean_worktree'] = str(BUNDLE / 'code/legacy')
                provenance['tpp_snapshot_root'] = str(BUNDLE / 'code/tpp')
                require(sha(BUNDLE / 'code/tpp/experiments/tpp_isolated/tpp_models.py') == provenance['tpp_models_sha256'], 'Bundled TPP differs')
                job_id = f'HS_{target}_{arm}_{seed}'
                config.update(job_id=job_id, experiment_name='h_dg_baselines__' + job_id, output_dir='__ATTEMPT_OUTPUT_DIR__')
                config['factorial'].update(arm='HS_' + arm, tpp_on=(pool == 'tpp'))
                config['dg_baseline'] = arm_specification(arm)
                config['h_controls'] = {'partition': 'H-S', 'group': GROUP, 'arm': arm, 'pool_mode': pool,
                    'phase_variant': 'historical', 'reference_config': str(reference_path),
                    'reference_config_sha256': cell['reference_h10_sha256'],
                    'initial_state_sha256': cell['initial_state_sha256'][pool], 'common_state_sha256': cell['common_state_sha256'],
                    'paired_control_job_id': cell['g10_job_id'] if pool == 'global_mlp' else cell['tce_job_id'],
                    'paired_gce_job_id': cell['g10_job_id'], 'paired_gce_attempt': cell['g10_attempt'],
                    'expected_sampler_sha256': cell['expected_sampler_sha256'],
                    'sampler_global_ids_file': str(BUNDLE / f'schedules/HS_{target}_{seed}.npy'),
                    'sampler_global_ids_file_sha256': cell['schedule_sha256'],
                    'source_manifest': str(BUNDLE / f'manifests/source_{target}.json'),
                    'source_manifest_sha256': cell['source_manifest_sha256'],
                    'paired_tce_job_id': cell['tce_job_id'], 'paired_tce_attempt': cell['tce_attempt'],
                    'paired_tsc_job_id': cell['tsc_job_id'], 'paired_tsc_attempt': cell['tsc_attempt'],
                    'paired_control_bundle_manifest_sha256': PAIRED_MANIFEST_SHA256,
                    'new_hyperparameter_search_trials': 0, 'target_used_previously_for_development': True,
                    'new_environment_experiment': False,
                    'prespecified_in': 'paper/icaiic2027/README.md and docs/revision/VENUE_AND_GAP_ASSESSMENT_20261023.md section 3.2'}
                verify_config(config)
                path = BUNDLE / 'configs' / (job_id + '.json')
                write(path, config, exclusive=True)
                jobs.append({'job_id': job_id, 'partition': 'H-S', 'target': target, 'seed': seed, 'arm': 'HS_' + arm,
                             'control_arm': arm, 'group': GROUP, 'pool_mode': pool,
                             'paired_control_job_id': config['h_controls']['paired_control_job_id'],
                             'config': path.relative_to(BUNDLE).as_posix()})
                print('Derived', job_id, 'on', pool, 'paired with', config['h_controls']['paired_control_job_id'], flush=True)
    write(BUNDLE / 'registry.json', {'schema': 'h-dg-baselines-registry/v2', 'jobs': jobs, 'new_runs': JOBS,
        'ce_runs': JOBS, 'multiview_runs': 0, 'reused_paired_runs': 36, 'paired_arms': ['H10', 'H11', 'G10', 'G11'],
        'queue_order': 'global-backbone arms (GC10, GX10) for all cells, then TPP-backbone arms (C10, X10)',
        'new_environment_experiment': False, 'target_metric_tuning': False}, exclusive=True)
    write(BUNDLE / 'decision_rule.json', DECISION_RULE, exclusive=True)
    bundle_files = {p.relative_to(BUNDLE).as_posix(): sha(p) for p in BUNDLE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    write(BUNDLE / 'bundle_manifest.json', {'schema': 'h-dg-baselines-bundle/v1', 'created_at': now(),
        'files_sha256': bundle_files, 'new_runs': JOBS, 'reused_paired_runs': 36,
        'paired_bundle_manifest_sha256': PAIRED_MANIFEST_SHA256, 'formal_training_started': False,
        'new_target_scoring_started': False, 'new_environment_experiment': False}, exclusive=True)
    print(json.dumps({'status': 'frozen', 'new_runs': JOBS, 'bundle': str(BUNDLE),
                      'manifest_sha256': sha(BUNDLE / 'bundle_manifest.json')}))

if __name__ == '__main__':
    main()
