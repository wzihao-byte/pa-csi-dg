"""Freeze both membership protocols before any new target scoring."""
from __future__ import annotations
import copy, json, shutil, sys
from pathlib import Path
import numpy as np
import torch
from common import (ARMS, BUNDLE, HERE, HISTORICAL, PARAMETERS, ROOT, SEEDS, TARGETS,
                    base_module, grouped_helpers, now, read, require, sha,
                    state_hash, verify_config, write)
from control_components import initialization_audit, make_model_class
from phase_inputs import audit_phase_alignment, phase_input_patch

def references():
    result = {}
    for partition in ('H-S',):
        for arm, folder in (('H10', 'l1248_h10_v1'), ('H11', 'l1248_v1')):
            index_path = ROOT / ('outputs_h_grouped_l1248/v2/factorial_results.json' if partition == 'H-S'
                else 'outputs_tpp_level8_ablation/' + folder + '/level8_results.json')
            index = read(index_path)
            require(index['status'] == 'complete', 'Incomplete reference matrix')
            for row in index['rows']:
                if row['arm'] != arm:
                    continue
                attempt = Path(row['attempt_dir'])
                completed = read(attempt / 'attempt_completed.json')
                require(completed['status'] == 'success', 'Reference unsuccessful')
                require(sha(attempt / 'attempt_completed.json') == row['completion_manifest_sha256'], 'Reference completion drift')
                require(read(attempt / 'attempt_seal.json')['completion_manifest_sha256'] == row['completion_manifest_sha256'], 'Reference seal drift')
                if partition == 'H-S':
                    original = HISTORICAL / 'configs' / f"{row['target_env']}_HG_{arm}_{row['seed']}.json"
                else:
                    generated = 'tpp_level8_h10_ablation/generated_l1248_h10_v1' if arm == 'H10' else 'tpp_level8_ablation/generated_l1248_v1'
                    original = ROOT / 'experiments' / generated / 'configs' / f"tpp_l1248__target_{row['target_env']}__seed_{row['seed']}__arm_{arm}.json"
                require(sha(original) == completed['source_config_file_sha256'], 'Reference file was not the executed config')
                raw = read(original)
                executed = read(attempt / 'job_config.json')
                executed.pop('output_dir', None)
                comparable = copy.deepcopy(raw); comparable.pop('output_dir', None)
                require(comparable == executed, 'Reference configuration execution mismatch')
                sampler = read(attempt / 'payload/run/sampler_schedule.json')
                result[(partition, row['target_env'], row['seed'], arm)] = {
                    'config':raw, 'config_path':original, 'attempt':attempt, 'row':row,
                    'expected_sampler_sha256':sampler['global_sample_schedule_sha256']}
    require(len(result) == 18, 'Need all 18 completed H-S reference runs')
    return result

def arm_spec(arm):
    supcon = arm not in ('G10', 'N_H10', 'N_D10')
    return {'supcon':supcon, 'reference_arm':'H11' if supcon else 'H10',
        'pool_mode':'global_mlp' if arm.startswith('G') else 'tpp',
        'phase_variant':'native_historical' if arm.startswith('N_H') else 'native_differential' if arm.startswith('N_D') else 'historical',
        'positive_mode':'all_views' if arm == 'P_ALL' else 'hybrid_remove_other_sameclass_denominator' if arm == 'P_REMOVE' else 'global_class_instance_views',
        'group':'capacity' if arm.startswith('G') else 'positive_rule' if arm.startswith('P_') else 'phase'}

def main():
    require(not BUNDLE.exists(), 'Preserve any previously frozen bundle; use a new version for changes')
    required = ('common.py','train_factorial.py','run.py','qa.py','control_components.py','phase_inputs.py','analyze.py','benchmark_current_gpu.py')
    require(all((HERE / x).is_file() for x in required), 'Control implementation incomplete')
    torch.set_num_threads(2)
    refs = references()
    BUNDLE.mkdir(parents=True)
    code = BUNDLE / 'code'; code.mkdir()
    for name in required:
        shutil.copy2(HERE / name, code / name)
    for kind in ('trainer', 'runner'):
        shutil.copy2(HISTORICAL / 'code' / ('base_' + kind + '.py'), code / ('base_' + kind + '.py'))
    shutil.copy2(HISTORICAL / 'code/common.py', code / 'grouped_helpers.py')
    for p in (HERE / 'tests').glob('test_*.py'):
        (code / 'tests').mkdir(exist_ok=True)
        shutil.copy2(p, code / 'tests' / p.name)
    legacy_dir = code / 'legacy'; legacy_dir.mkdir()
    first = refs[('H-S','E1',42,'H11')]['config']
    frozen_provenance = copy.deepcopy(first['provenance'])
    for name, record in frozen_provenance['legacy_modules'].items():
        require(sha(record['path']) == record['sha256'], 'Historical code changed: ' + name)
        dest = legacy_dir / (name + '.py'); shutil.copy2(record['path'], dest)
        record['path'] = str(dest)
    frozen_provenance['clean_worktree'] = str(legacy_dir)
    tpp = Path(first['provenance']['tpp_snapshot_root']) / 'experiments/tpp_isolated/tpp_models.py'
    require(sha(tpp) == first['provenance']['tpp_models_sha256'], 'Historical TPP code changed')
    tpp_dir = code / 'tpp/experiments/tpp_isolated'; tpp_dir.mkdir(parents=True)
    shutil.copy2(tpp, tpp_dir / 'tpp_models.py')
    frozen_provenance['tpp_snapshot_root'] = str(code / 'tpp')
    phase_audit_path = HERE / 'phase_alignment_audit.json'
    phase_audit = read(phase_audit_path) if phase_audit_path.exists() else audit_phase_alignment()
    phase_audit['source_audit_sha256'] = sha(phase_audit_path)
    phase_audit['planned_new_runs'] = 18
    phase_audit['completed_historical_TPP_references_reused'] = 18
    phase_audit['run_scope'] = 'User selected H-S only; compare final SupCon model for both native phase chains'
    write(BUNDLE / 'phase_alignment.json', phase_audit, exclusive=True)
    if (HERE / 'history_alignment_review.json').exists():
        shutil.copy2(HERE / 'history_alignment_review.json', BUNDLE / 'history_alignment_review.json')
    helper = grouped_helpers()
    base = base_module()
    initial = {}
    for seed in SEEDS:
        cfg = refs[('H-S','E1',seed,'H11')]['config']
        modules = base.configure_imports(cfg)
        modules['train_dg'].set_seed(seed)
        reference = modules['tpp_models'].IsolatedPaCsiDGLiteTPP(90, 6, cfg['model'])
        require(state_hash(reference) == cfg['h_grouped']['initial_state_sha256'], 'Historical TPP initialization drift')
        initial[str(seed)] = {}
        for pool in ('tpp', 'global_mlp'):
            modules['train_dg'].set_seed(seed)
            model = make_model_class(modules['tpp_models'].IsolatedPaCsiDGLiteTPP, pool)(90, 6, cfg['model'])
            audit = initialization_audit(model, reference)
            require(audit['shared_state_tensors_exactly_equal'], 'Shared H initialization drift')
            require(audit['model_parameter_count'] == PARAMETERS, 'Unequal H capacity')
            initial[str(seed)][pool] = audit
    write(BUNDLE / 'initialization_alignment.json', initial, exclusive=True)
    configs, ref_records, split_records = {}, [], {}
    for partition in ('H-S',):
        for target in TARGETS:
            source_path = HISTORICAL / 'manifests' / ('source_' + target + '.json')
            dest = BUNDLE / 'manifests' / source_path.name
            dest.parent.mkdir(exist_ok=True)
            if not dest.exists(): shutil.copy2(source_path, dest)
            source = read(dest)
            dataset = helper.metadata_dataset(source)
            for seed in SEEDS:
                ref = refs[(partition,target,seed,'H11')]
                modules = base.configure_imports(ref['config'])
                if partition == 'H-S':
                    split, identity = helper.grouped_split(dataset, source, target)
                    original_schedule = HISTORICAL / 'schedules' / f'{target}_{seed}.npy'
                    expected_sampler = ref['expected_sampler_sha256']
                else:
                    split = base.source_split(dataset, seed, target, modules['dg_dataset'])
                    identity = None
                    expected_sampler, schedule = helper.replay_sampler(dataset, split, seed, modules['dg_dataset'].DomainClassBalancedBatchSampler)
                    require(expected_sampler == ref['expected_sampler_sha256'], 'Historical H sampling replay drift')
                schedule_path = BUNDLE / 'schedules' / f"{partition.replace('-','')}_{target}_{seed}.npy"
                schedule_path.parent.mkdir(exist_ok=True)
                if partition == 'H-S': shutil.copy2(original_schedule, schedule_path)
                else: np.save(schedule_path, schedule, allow_pickle=False)
                split_hashes = {part:helper.index_hash(dataset.sample_ids[split[part + '_indices']]) for part in ('train','val')}
                require(all(split_hashes[k] == ref['config']['protocol']['expected_split_hashes'][k] for k in split_hashes), 'Split replay drift')
                key = f'{partition}_{target}_{seed}'
                split_records[key] = {'split_hashes':split_hashes, 'expected_sampler_sha256':expected_sampler,
                    'source_identity':identity, 'schedule_file_sha256':sha(schedule_path)}
                for ref_arm in ('H10','H11'):
                    old = refs[(partition,target,seed,ref_arm)]
                    require(old['expected_sampler_sha256'] == expected_sampler, 'Reference arms have unequal sampling')
                    path = BUNDLE / 'references' / f"{partition.replace('-','')}_{target}_{seed}_{ref_arm}.json"
                    path.parent.mkdir(exist_ok=True); shutil.copy2(old['config_path'], path)
                    ref_records.append({'partition':partition, 'target':target,'seed':seed,'arm':ref_arm,
                        'config':str(path), 'config_sha256':sha(path), 'attempt':str(old['attempt']),
                        'completion_sha256':old['row']['completion_manifest_sha256'],
                        'metrics_sha256':old['row']['metrics_file_sha256'],
                        'predictions_sha256':old['row']['target_predictions_file_sha256']})
                for arm in ARMS:
                    spec = arm_spec(arm)
                    reference_path = BUNDLE / 'references' / f"{partition.replace('-','')}_{target}_{seed}_{spec['reference_arm']}.json"
                    raw = read(reference_path)
                    config = copy.deepcopy(raw) if spec['phase_variant'] == 'historical' else phase_input_patch(raw, spec['phase_variant'], phase_audit)
                    config.pop('h_grouped', None)
                    config['provenance'] = copy.deepcopy(frozen_provenance)
                    job_id = f"{partition.replace('-','')}_{target}_{arm}_{seed}"
                    config.update(job_id=job_id, experiment_name='main_h_controls__' + job_id, output_dir='__ATTEMPT_OUTPUT_DIR__')
                    config['factorial'].update(arm=f"{partition.replace('-','')}_{arm}", tpp_on=spec['pool_mode']=='tpp')
                    config['losses']['supcon_positive_mode'] = spec['positive_mode']
                    audit = initial[str(seed)][spec['pool_mode']]
                    config['h_controls'] = {'partition':partition, 'group':spec['group'], 'arm':arm,
                        'pool_mode':spec['pool_mode'], 'phase_variant':spec['phase_variant'],
                        'reference_config':str(reference_path), 'reference_config_sha256':sha(reference_path),
                        'initial_state_sha256':audit['model_full_state_sha256'],
                        'common_state_sha256':audit['model_common_state_sha256'],
                        'expected_sampler_sha256':expected_sampler, 'sampler_global_ids_file':str(schedule_path),
                        'sampler_global_ids_file_sha256':sha(schedule_path), 'new_hyperparameter_search_trials':0,
                        'target_used_previously_for_development':True, 'new_environment_experiment':False}
                    if partition == 'H-S':
                        config['h_controls'].update(source_manifest=str(dest), source_manifest_sha256=sha(dest))
                    verify_config(config)
                    path = BUNDLE / 'configs' / (job_id + '.json')
                    write(path, config, exclusive=True)
                    configs[(partition,target,seed,arm)] = {'job_id':job_id,'partition':partition,'target':target,
                        'seed':seed,'arm':config['factorial']['arm'], 'control_arm':arm,'group':spec['group'],
                        'config':str(path.relative_to(BUNDLE).as_posix())}
                print('Bound source split and complete sampling:', key, flush=True)
    # Priority and scope are fixed before target scoring: H-S capacity, pair rules, phase.
    jobs = []
    for partition in ('H-S',):
        for arms in (ARMS[:2], ARMS[2:4], ARMS[4:]):
            for target in TARGETS:
                for seed in SEEDS:
                    for arm in arms: jobs.append(configs[(partition,target,seed,arm)])
    write(BUNDLE / 'registry.json', {'schema':'main-h-controls-registry/v1','jobs':jobs,
        'new_runs':54,'ce_runs':9,'multiview_runs':45,'reused_reference_runs':18,
        'new_environment_experiment':False,'target_metric_tuning':False}, exclusive=True)
    write(BUNDLE / 'historical_references.json', {'references':ref_records}, exclusive=True)
    write(BUNDLE / 'source_alignment.json', {'status':'passed','partitions':split_records,
        'update_budget':30000,'batch_size':32,'source_checkpoint_rule':'source_val_accuracy_last_tied_best',
        'pseudo_tx_layout_is_software_feature_grouping':True}, exclusive=True)
    files = {p.relative_to(BUNDLE).as_posix():sha(p) for p in BUNDLE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    write(BUNDLE / 'bundle_manifest.json', {'schema':'main-h-controls-bundle/v1','created_at':now(),
        'files_sha256':files,'new_runs':54,'reused_reference_runs':18,
        'formal_training_started':False,'new_target_scoring_started':False,
        'new_environment_experiment':False}, exclusive=True)
    print(json.dumps({'status':'frozen','new_runs':54,'bundle':str(BUNDLE),'manifest_sha256':sha(BUNDLE/'bundle_manifest.json')}))

if __name__ == '__main__':
    main()
