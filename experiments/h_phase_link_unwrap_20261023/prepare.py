"""Freeze the 9-run within-link unwrap control before any new target scoring."""
from __future__ import annotations
import copy, json, shutil, sys
from pathlib import Path
from common import (BUNDLE, HERE, JOBS, LINK_VARIANT, PAIRED_ARM, PAIRED_BUNDLE,
                    PAIRED_MANIFEST_SHA256, PAIRED_STUDY, SEEDS, TARGETS, now, read, require, sha,
                    verify_config, write)
# phase_inputs is reused byte-for-byte from the paired frozen bundle.
sys.path.append(str(PAIRED_BUNDLE / 'code'))
from link_inputs import audit_link_alignment

NEW_CODE = ('common.py', 'prepare.py', 'link_inputs.py', 'run.py', 'qa.py', 'analyze.py')
PAIRED_CODE = ('train_factorial.py', 'control_components.py', 'benchmark_current_gpu.py', 'phase_inputs.py',
               'base_trainer.py', 'base_runner.py', 'grouped_helpers.py')
DECISION_RULE = {
    'primary_contrast': 'N_L11 minus paired N_H11 target macro-F1 (pp), nine target-seed pairs, equal environment weights',
    'reported': ['nine paired differences', 'paired mean', 'per-environment means', 'sign count',
                 '95% percentile bootstrap interval over the nine pairs (10,000 resamples, seed 20261023)',
                 'mean within-environment seed SD of N_H11'],
    'unchanged': 'bootstrap interval includes 0 and |paired mean| is at most the mean within-environment seed SD of N_H11',
    'drop': 'bootstrap interval entirely below 0; reported as a limitation of the historical-style chain',
    'otherwise': 'reported descriptively as inconclusive without changing the frozen main representation',
    'caveat': 'Nine pairs share three development environments; the interval is descriptive, not a test of independent environments',
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
    return manifest, attempts

def copy_checked(source, dest, expected):
    require(sha(source) == expected, 'Copied source differs from frozen digest: ' + str(source))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    require(sha(dest) == expected, 'Copy changed bytes: ' + str(dest))
    return dest

def main():
    require(not BUNDLE.exists(), 'Preserve any previously frozen bundle; use a new version for changes')
    require(all((HERE / x).is_file() for x in NEW_CODE), 'Link-unwrap implementation incomplete')
    paired_manifest, paired_attempts = verify_paired_study()
    files = paired_manifest['files_sha256']
    paired_alignment = read(PAIRED_BUNDLE / 'phase_alignment.json')
    alignment = audit_link_alignment(paired_alignment)
    alignment['paired_phase_alignment_sha256'] = files['phase_alignment.json']
    BUNDLE.mkdir(parents=True)
    write(BUNDLE / 'link_alignment.json', alignment, exclusive=True)
    code = BUNDLE / 'code'
    for name in NEW_CODE:
        code.mkdir(exist_ok=True); shutil.copy2(HERE / name, code / name)
    for p in (HERE / 'tests').glob('test_*.py'):
        (code / 'tests').mkdir(exist_ok=True); shutil.copy2(p, code / 'tests' / p.name)
    for name in PAIRED_CODE:
        copy_checked(PAIRED_BUNDLE / 'code' / name, code / name, files['code/' + name])
    for rel, digest in files.items():
        if rel.startswith(('code/legacy/', 'code/tpp/', 'code/tests/')):
            copy_checked(PAIRED_BUNDLE / rel, BUNDLE / rel, digest)
    configs, jobs = {}, []
    for target in TARGETS:
        manifest_rel = f'manifests/source_{target}.json'
        source_manifest = copy_checked(PAIRED_BUNDLE / manifest_rel, BUNDLE / manifest_rel, files[manifest_rel])
        for seed in SEEDS:
            paired_rel = f'configs/HS_{target}_{PAIRED_ARM}_{seed}.json'
            paired_path = copy_checked(PAIRED_BUNDLE / paired_rel, BUNDLE / 'paired_controls' / Path(paired_rel).name, files[paired_rel])
            paired = read(paired_path)
            schedule_rel = f'schedules/HS_{target}_{seed}.npy'
            reference_rel = f'references/HS_{target}_{seed}_H11.json'
            schedule = copy_checked(PAIRED_BUNDLE / schedule_rel, BUNDLE / schedule_rel, files[schedule_rel])
            reference = copy_checked(PAIRED_BUNDLE / reference_rel, BUNDLE / reference_rel, files[reference_rel])
            paired_job = paired_attempts[paired['job_id']]
            config = copy.deepcopy(paired)
            bindings = alignment['file_records']
            for env in config['data']['env_files']:
                config['data']['env_files'][env]['phase'] = bindings[env]['phase']['path']
                config['source_file_records'][env]['phase'] = copy.deepcopy(bindings[env]['phase'])
            config['target_data']['data']['env_files'][target]['phase'] = bindings[target]['phase']['path']
            config['target_data']['file_records']['phase'] = copy.deepcopy(bindings[target]['phase'])
            job_id = f'HS_{target}_N_L11_{seed}'
            config.update(job_id=job_id, experiment_name='h_phase_link_unwrap__' + job_id, output_dir='__ATTEMPT_OUTPUT_DIR__')
            config['factorial']['arm'] = 'HS_N_L11'
            provenance = config['provenance']
            for name, record in provenance['legacy_modules'].items():
                record['path'] = str(code / 'legacy' / (name + '.py'))
                require(sha(record['path']) == record['sha256'], 'Bundled legacy module differs: ' + name)
            provenance['clean_worktree'] = str(code / 'legacy')
            provenance['tpp_snapshot_root'] = str(code / 'tpp')
            require(sha(code / 'tpp/experiments/tpp_isolated/tpp_models.py') == provenance['tpp_models_sha256'], 'Bundled TPP differs')
            c = config['h_controls']
            c.update(group='phase_unwrap_boundary', arm='N_L11', phase_variant='native_link_unwrap',
                     reference_config=str(reference), sampler_global_ids_file=str(schedule), source_manifest=str(source_manifest),
                     paired_control_config=str(paired_path), paired_control_config_sha256=sha(paired_path),
                     paired_control_job_id=paired['job_id'], paired_control_attempt=paired_job['attempt'],
                     paired_control_completion_sha256=paired_job['completion_sha256'],
                     paired_control_bundle_manifest_sha256=PAIRED_MANIFEST_SHA256,
                     prespecified_in='reports/weekly_report_20261023 Next Steps: phase check')
            verify_config_before_freeze(config)
            path = BUNDLE / 'configs' / (job_id + '.json')
            write(path, config, exclusive=True)
            jobs.append({'job_id': job_id, 'partition': 'H-S', 'target': target, 'seed': seed, 'arm': 'HS_N_L11',
                         'control_arm': 'N_L11', 'group': 'phase_unwrap_boundary', 'paired_job_id': paired['job_id'],
                         'config': path.relative_to(BUNDLE).as_posix()})
            print('Derived', job_id, 'from', paired['job_id'], flush=True)
    write(BUNDLE / 'registry.json', {'schema': 'h-phase-link-unwrap-registry/v1', 'jobs': jobs, 'new_runs': JOBS,
        'ce_runs': 0, 'multiview_runs': JOBS, 'reused_paired_runs': JOBS, 'paired_arm': PAIRED_ARM,
        'new_environment_experiment': False, 'target_metric_tuning': False}, exclusive=True)
    write(BUNDLE / 'decision_rule.json', DECISION_RULE, exclusive=True)
    bundle_files = {p.relative_to(BUNDLE).as_posix(): sha(p) for p in BUNDLE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    write(BUNDLE / 'bundle_manifest.json', {'schema': 'h-phase-link-unwrap-bundle/v1', 'created_at': now(),
        'files_sha256': bundle_files, 'new_runs': JOBS, 'reused_paired_runs': JOBS, 'variant': LINK_VARIANT,
        'paired_bundle_manifest_sha256': PAIRED_MANIFEST_SHA256, 'formal_training_started': False,
        'new_target_scoring_started': False, 'new_environment_experiment': False}, exclusive=True)
    print(json.dumps({'status': 'frozen', 'new_runs': JOBS, 'bundle': str(BUNDLE),
                      'manifest_sha256': sha(BUNDLE / 'bundle_manifest.json')}))

def verify_config_before_freeze(config):
    # verify_config reads link_alignment.json from the bundle, which is already written.
    verify_config(config)

if __name__ == '__main__':
    main()
