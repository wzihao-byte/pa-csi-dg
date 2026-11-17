"""Freeze the three-class four-arm bundle before any training or target scoring.

Reuses the frozen main-study (h_main_controls_20261008_v2) trainer, runner, model
components and legacy modules unchanged, except for counted literal replacements of
the six-class / 30,000-update invariants listed in PATCHES.
"""
from __future__ import annotations
import copy, json, shutil, sys
from pathlib import Path
import numpy as np
import torch
import three_class as tc
from common import (ARMS, BUNDLE, HERE, MAIN_V2, OUTPUT, PARAMETERS, SEEDS, TARGETS, job_grid,
                    load_module, now, read, require, sha, state_hash, verify_config, write)

PATCHES = {
    'base_trainer.py': [
        ('NUM_CLASSES = 6', 'NUM_CLASSES = 3', 1),
        ('assert_equal(expected_index, 3000, "target prediction count")', 'assert_equal(expected_index, 1200, "target prediction count")', 1),
        ('assert_equal(config["protocol"]["optimizer_steps_per_epoch"], 150, "steps per epoch")', 'assert_equal(config["protocol"]["optimizer_steps_per_epoch"], 60, "steps per epoch")', 1),
        ('assert_equal(config["protocol"]["total_optimizer_steps"], 30000, "total steps")', 'assert_equal(config["protocol"]["total_optimizer_steps"], 12000, "total steps")', 1),
        ('assert_equal(split_record["train"]["count"], 4800, "train count")', 'assert_equal(split_record["train"]["count"], 1920, "train count")', 1),
        ('assert_equal(split_record["val"]["count"], 1200, "validation count")', 'assert_equal(split_record["val"]["count"], 480, "validation count")', 1),
        ('assert_equal(schedule["batches_seen"], 30000, "sampler batches")', 'assert_equal(schedule["batches_seen"], 12000, "sampler batches")', 1),
        ('assert_equal(schedule["samples_seen"], 960000, "sampler samples")', 'assert_equal(schedule["samples_seen"], 384000, "sampler samples")', 1),
        ('assert_equal(len(target_dataset), 3000, "target dataset size")', 'assert_equal(len(target_dataset), 1200, "target dataset size")', 1),
        ('"count": 3000,', '"count": 1200,', 1),
        ('"env_counts": {target_env: 3000},', '"env_counts": {target_env: 1200},', 1),
        ('"train_size": 4800,', '"train_size": 1920,', 1),
        ('"val_size": 1200,', '"val_size": 480,', 1),
        ('"test_size": 3000,', '"test_size": 1200,', 1),
        ('"optimizer_steps_per_epoch": 150,', '"optimizer_steps_per_epoch": 60,', 1),
        ('"total_optimizer_steps": 30000,', '"total_optimizer_steps": 12000,', 1),
        ('"target_prediction_count": 3000', '"target_prediction_count": 1200', 3),
    ],
    'base_runner.py': [
        ('assert_equal(config["protocol"]["total_optimizer_steps"], 30000, "config steps")', 'assert_equal(config["protocol"]["total_optimizer_steps"], 12000, "config steps")', 1),
        ('confusion = np.zeros((6, 6), dtype=np.int64)', 'confusion = np.zeros((3, 3), dtype=np.int64)', 1),
        ('probabilities.shape != (6,)', 'probabilities.shape != (3,)', 1),
        ('assert_equal(count, 3000, "prediction count")', 'assert_equal(count, 1200, "prediction count")', 1),
        ('out=np.zeros(6)', 'out=np.zeros(3)', 3),
        ('(4800, 1200, 3000)', '(1920, 480, 1200)', 2),
        ('(200, 150, 30000)', '(200, 60, 12000)', 1),
        ('(200, 30000, 960000)', '(200, 12000, 384000)', 1),
        ('stored_confusion.shape != (6, 6)', 'stored_confusion.shape != (3, 3)', 1),
        ('if job.arm in {"H01", "H11"}:', 'if job.arm in {"TC_G11", "TC_T11"}:', 1),
    ],
}
OWN = ('common.py', 'three_class.py', 'train_factorial.py', 'run.py', 'qa.py', 'analyze.py',
       'prepare.py', 'probe_duration_three_class.py', 'probe_three_class.json')


def patched(text, name):
    for old, new, count in PATCHES[name]:
        require(text.count(old) == count, f'{name}: expected {count} x {old!r}, found {text.count(old)}')
        text = text.replace(old, new)
    return text


def main():
    require(not BUNDLE.exists(), 'Preserve any frozen bundle; use a new version for changes')
    require(read(HERE / 'probe_three_class.json')['probes']['duration_only']['mean_macro_f1'] < .40,
            'Duration still separates the three classes; stop and report')
    torch.set_num_threads(2)
    code = BUNDLE / 'code'; code.mkdir(parents=True)
    src = MAIN_V2 / 'code'
    for name in ('base_trainer.py', 'base_runner.py'):
        (code / name).write_text(patched((src / name).read_text(encoding='utf-8'), name), encoding='utf-8', newline='\n')
    shutil.copy2(src / 'control_components.py', code / 'control_components.py')
    shutil.copytree(src / 'legacy', code / 'legacy', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(src / 'tpp', code / 'tpp', ignore=shutil.ignore_patterns('__pycache__'))
    for name in OWN:
        shutil.copy2(HERE / name, code / name)
    sys.path.insert(0, str(code))
    from control_components import initialization_audit, make_model_class
    base = load_module('three_class_prepare_base', code / 'base_trainer.py')

    first = read(MAIN_V2 / 'configs/HS_E1_G11_42.json')
    provenance = copy.deepcopy(first['provenance'])
    for module, record in provenance['legacy_modules'].items():
        dest = code / 'legacy' / (module + '.py')
        require(sha(dest) == record['sha256'], 'Legacy module copy differs: ' + module)
        record['path'] = str(dest)
    provenance['clean_worktree'] = str(code / 'legacy')
    require(sha(code / 'tpp/experiments/tpp_isolated/tpp_models.py') == provenance['tpp_models_sha256'], 'TPP model copy differs')
    provenance['tpp_snapshot_root'] = str(code / 'tpp')
    probe_cfg = copy.deepcopy(first); probe_cfg['provenance'] = provenance

    initial = {}
    for seed in SEEDS:
        modules = base.configure_imports(probe_cfg)
        tpp_class = modules['tpp_models'].IsolatedPaCsiDGLiteTPP
        modules['train_dg'].set_seed(seed)
        reference = make_model_class(tpp_class, 'tpp')(90, tc.NUM_CLASSES, first['model'])
        initial[str(seed)] = {}
        for pool in ('tpp', 'global_mlp'):
            modules['train_dg'].set_seed(seed)
            model = make_model_class(tpp_class, pool)(90, tc.NUM_CLASSES, first['model'])
            audit = initialization_audit(model, reference)
            require(audit['shared_state_tensors_exactly_equal'], 'Shared initialization differs across pools')
            require(audit['model_parameter_count'] == PARAMETERS, f"Capacity {audit['model_parameter_count']} != {PARAMETERS}")
            require(audit['model_full_state_sha256'] == state_hash(model), 'State-hash convention drift')
            initial[str(seed)][pool] = audit
    write(BUNDLE / 'initialization_alignment.json', initial, exclusive=True)

    sampler_class = base.configure_imports(probe_cfg)['dg_dataset'].DomainClassBalancedBatchSampler
    configs, alignment, jobs = {}, {}, []
    for target in TARGETS:
        manifest_dest = BUNDLE / 'manifests' / f'source_{target}.json'
        manifest_dest.parent.mkdir(exist_ok=True)
        shutil.copy2(MAIN_V2 / 'manifests' / f'source_{target}.json', manifest_dest)
        source = tc.filter_manifest(read(manifest_dest))
        meta = tc.metadata_dataset(source)
        split, identity = tc.grouped_split(meta, source, target)
        hashes = {part: tc.index_hash(meta.sample_ids[split[part + '_indices']]) for part in ('train', 'val')}
        ref_cfg = read(MAIN_V2 / f'configs/HS_{target}_G10_42.json')
        target_labels = np.load(ref_cfg['target_data']['data']['env_files'][target]['label'], allow_pickle=True)
        require(len(target_labels) == 3000, 'Target label file size')
        target_ids = tc.ENV_ORDER.index(target) * 3000 + np.flatnonzero(np.isin(target_labels, tc.KEEP))
        require(len(target_ids) == tc.TARGET_COUNT, 'Target restricted size')
        hashes['test'] = tc.index_hash(target_ids)
        for seed in SEEDS:
            sampler_sha, schedule, class_counts = tc.replay_sampler(meta, split, seed, sampler_class)
            schedule_path = BUNDLE / 'schedules' / f'TC_{target}_{seed}.npy'
            schedule_path.parent.mkdir(exist_ok=True)
            np.save(schedule_path, schedule, allow_pickle=False)
            alignment[f'{target}_{seed}'] = {'split_hashes': hashes, 'expected_sampler_sha256': sampler_sha,
                'schedule_file_sha256': sha(schedule_path), 'sampled_class_counts': class_counts,
                'source_identity_counts': {p: {k: len(v) for k, v in identity[p].items()} for p in identity}}
            for arm, (paper, pool, supcon) in ARMS.items():
                ref_name = f"HS_{target}_{'G11' if supcon else 'G10'}_{seed}.json"
                ref_path = BUNDLE / 'references' / ref_name
                ref_path.parent.mkdir(exist_ok=True)
                if not ref_path.exists():
                    shutil.copy2(MAIN_V2 / 'configs' / ref_name, ref_path)
                config = read(ref_path)
                config.pop('h_controls')
                config['provenance'] = copy.deepcopy(provenance)
                job_id = f'TC_{target}_{arm}_{seed}'
                config.update(job_id=job_id, experiment_name='three_class_4s__' + job_id, output_dir='__ATTEMPT_OUTPUT_DIR__')
                config['factorial'].update(arm='TC_' + arm, tpp_on=pool == 'tpp')
                config['losses']['supcon_positive_mode'] = 'global_class_instance_views'
                config['protocol'].update(version='three-class-4s/v1', expected_split_hashes=dict(hashes),
                    optimizer_steps_per_epoch=tc.STEPS_PER_EPOCH, epochs=tc.EPOCHS,
                    total_optimizer_steps=tc.TOTAL_STEPS, target_prediction_count=tc.TARGET_COUNT,
                    expected_parameter_count=PARAMETERS)
                audit = initial[str(seed)][pool]
                config['h_controls'] = {'partition': 'H-S', 'group': 'three_class_4s', 'arm': arm, 'paper_arm': paper,
                    'pool_mode': pool, 'phase_variant': 'historical',
                    'reference_config': str(ref_path), 'reference_config_sha256': sha(ref_path),
                    'initial_state_sha256': audit['model_full_state_sha256'],
                    'common_state_sha256': audit['model_common_state_sha256'],
                    'expected_sampler_sha256': sampler_sha, 'sampler_global_ids_file': str(schedule_path),
                    'sampler_global_ids_file_sha256': sha(schedule_path),
                    'source_manifest': str(manifest_dest), 'source_manifest_sha256': sha(manifest_dest),
                    'new_hyperparameter_search_trials': 0, 'target_used_previously_for_development': True}
                config['three_class'] = {'keep_original_labels': list(tc.KEEP), 'class_names': list(tc.CLASS_NAMES),
                    'label_map': {str(k): v for k, v in tc.LABEL_MAP.items()},
                    'budget_rule': 'same 200 epochs as the main study (60 updates per epoch, 12,000 updates)',
                    'success_criterion': 'gain persists if paired mean > 0 and >= 7/9 pairs improve (T-SC - T-CE, T-CE - G-CE)'}
                verify_config(config)
                path = BUNDLE / 'configs' / (job_id + '.json')
                write(path, config, exclusive=True)
                configs[(target, seed, arm)] = {'job_id': job_id, 'target': target, 'seed': seed, 'arm': 'TC_' + arm,
                    'control_arm': arm, 'paper_arm': paper, 'config': path.relative_to(BUNDLE).as_posix()}
            print('frozen', target, seed, sampler_sha[:12], class_counts, flush=True)
    jobs = [configs[key] for key in job_grid()]
    write(BUNDLE / 'registry.json', {'schema': 'three-class-4s-registry/v1', 'jobs': jobs, 'new_runs': 36,
        'ce_runs': 18, 'multiview_runs': 18, 'target_metric_tuning': False}, exclusive=True)
    write(BUNDLE / 'source_alignment.json', {'status': 'passed', 'partitions': alignment,
        'update_budget': tc.TOTAL_STEPS, 'batch_size': tc.BATCH, 'checkpoint_rule': 'source_val_accuracy_last_tied_best',
        'pseudo_tx_layout_is_software_feature_grouping': True}, exclusive=True)
    files = {p.relative_to(BUNDLE).as_posix(): sha(p) for p in BUNDLE.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    write(BUNDLE / 'bundle_manifest.json', {'schema': 'three-class-4s-bundle/v1', 'created_at': now(),
        'files_sha256': files, 'new_runs': 36, 'formal_training_started': False,
        'target_scoring_started': False}, exclusive=True)
    print(json.dumps({'status': 'frozen', 'bundle': str(BUNDLE), 'jobs': len(jobs),
                      'manifest_sha256': sha(BUNDLE / 'bundle_manifest.json')}))


if __name__ == '__main__':
    main()
