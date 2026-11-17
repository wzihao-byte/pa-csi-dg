"""Frozen 27-run phase follow-up, with comparisons kept within each time grid.

The input-study code is copied unchanged. This module only binds the missing
input conditions, audits them, coordinates the frozen engine, and reports every
predeclared result. Historical files and experiments remain read-only references.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import io
import os
from pathlib import Path
import shutil
import traceback

import numpy as np
import torch
from torch.nn import functional as F

from experiments.revision_validation.experiment_data import VerifiedInputs
from experiments.revision_validation.experiment_design import build_experiment_model, objective
from experiments.revision_validation.extended_engine import train_experiment, verify_extension
from experiments.revision_validation.freeze_inputs import INPUT_CODE
from experiments.revision_validation.modality_models import build_twin_model, historical_feature_phase
from experiments.revision_validation.source_data import epoch_schedule
from experiments.revision_validation.study_engine import acquire_lock, equal_state, restore_rng, rng_state, runtime_settings
from experiments.revision_validation.study_io import bound_path, now, read, sha256, value_hash, verify_bundle, verify_completed, write
from experiments.revision_validation.time_pooling_report import _verify_study, _verified_run, _stats, _write_text_once

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path('D:/pa-csi-dg-revision-runtime')
CORE = RUNTIME / 'studies/r_core_v1'
INPUTS = RUNTIME / 'studies/r_inputs_v1'
HIST_DATA = RUNTIME / 'data/modality_arrays_v1'
NATIVE_DATA = RUNTIME / 'data/native_histstyle_phase_v1'
HIST_VARIANT = 'historical_feature_axis_unwrap'
NATIVE_VARIANT = 'native_time_wrapped_linear_feature_unwrap_v1'
SELF = 'experiments/revision_validation/phase_followup_study.py'
HELPER = 'experiments/revision_validation/time_pooling_report.py'
SEEDS = {'E1': (401, 402, 403), 'E2': (501, 502, 503), 'E3': (601, 602, 603)}
SPECS = {
    'HIST_TWIN_AA': {'scope': 'history', 'reference': 'TWIN_AA', 'variant': HIST_VARIANT,
                     'processing': 'Frozen historical grid amplitude and historical float64 feature-axis unwrap then float32 phase'},
    'HIST_TWIN_AP': {'scope': 'history', 'reference': 'TWIN_AP', 'variant': HIST_VARIANT,
                     'processing': 'Frozen historical grid amplitude and historical float64 feature-axis unwrap then float32 phase'},
    'NATIVE_HISTSTYLE_AP': {'scope': 'native', 'reference': 'TWIN_AP', 'variant': NATIVE_VARIANT,
                          'processing': 'DAT wrapped angle arithmetic linear interpolation on native 850-point grid, then axis=-1 feature unwrap and float32; same native linear amplitude'},
}


def data_root(scope):
    return HIST_DATA if scope == 'history' else NATIVE_DATA


def configurations(scope=None, bundle=None):
    original = read(INPUTS / 'protocol.json')['jobs']
    jobs, bindings = [], []
    for control, spec in SPECS.items():
        if scope is not None and scope != spec['scope']:
            continue
        selected = [c for c in original if c['control_id'] == spec['reference']]
        if len(selected) != 9 or {(c['target_name_only'], c['seed']) for c in selected} != {(t,s) for t,seeds in SEEDS.items() for s in seeds}:
            raise RuntimeError('Reference twin matrix is incomplete')
        for old in selected:
            config = deepcopy(old)
            target, seed = old['target_name_only'], old['seed']
            config.update(job_id=f'{target}_P_{control}_{seed}', control_id=control,
                          input_variant=spec['variant'], input_processing=spec['processing'])
            directory = (bundle if bundle is not None else data_root(spec['scope']))
            directory = directory / ('manifests' if bundle is not None else spec['variant'])
            directory = directory / (spec['variant'] if bundle is not None else 'manifests')
            for key, prefix in (('source', 'source'), ('target', 'deferred_target')):
                path = directory / f'{prefix}_{target}.json'
                config[key + '_manifest_path'] = str(path)
                config[key + '_manifest_sha256'] = sha256(path)
                metadata = read(path)
                if metadata['data_version'] != spec['variant'] or metadata['outer_target'] != target:
                    raise RuntimeError('Wrong input version or outer fold')
            permitted = {'job_id', 'control_id', 'input_variant', 'input_processing',
                         'source_manifest_path', 'source_manifest_sha256', 'target_manifest_path', 'target_manifest_sha256'}
            changed = [key for key in old if config[key] != old[key]]
            if set(changed) - permitted or config['total_parameters'] != 3655286:
                raise RuntimeError('Unexpected model/training difference from frozen twin reference')
            if config['experiment']['model_kind'] != 'twin' or config['experiment']['objective'] != 'ce':
                raise RuntimeError('Require the frozen twin CE model')
            jobs.append(config)
            bindings.append({'job_id': config['job_id'], 'reference_job_id': old['job_id'],
                             'reference_config_value_sha256': value_hash(old), 'changed_fields': changed})
    return jobs, bindings


def code_check():
    evidence = {}
    for name in INPUT_CODE:
        expected = sha256(INPUTS / 'code' / name)
        if sha256(ROOT / name) != expected:
            raise RuntimeError(f'Input-study implementation changed: {name}')
        evidence[name] = expected
    for name in (SELF, HELPER):
        evidence[name] = sha256(ROOT / name)
    return evidence


def check_data(scope):
    directory = data_root(scope)
    variant = HIST_VARIANT if scope == 'history' else NATIVE_VARIANT
    manifest = read(directory / 'dataset_manifest.json')
    if not manifest.get('complete') or manifest['rows'] != 9000 or variant not in manifest['variants']:
        raise RuntimeError('Require completed 9000-record compilation')
    actual_files, identity_checks, formula_checks = {}, {}, {}
    # Historical original phase is inspected solely to verify deterministic input
    # construction. This does not run target inference or select a model.
    for target in SEEDS:
        source_path = directory / variant / 'manifests' / f'source_{target}.json'
        target_path = directory / variant / 'manifests' / f'deferred_target_{target}.json'
        reference_dir = CORE / 'manifests' if scope == 'history' else INPUTS / 'manifests/native_time_linear'
        source, target_meta = read(source_path), read(target_path)
        old_source, old_target = read(reference_dir / source_path.name), read(reference_dir / target_path.name)
        for path in (source_path, target_path):
            relative = path.relative_to(directory).as_posix()
            if sha256(path) != manifest['artifacts_sha256'][relative]:
                raise RuntimeError('Compiled partition manifest changed')
        for part in ('train', 'val'):
            if source[part] != old_source[part]:
                raise RuntimeError('Source identities or ordering differ from grid reference')
        if target_meta['test'] != old_target['test']:
            raise RuntimeError('Target identities or ordering differ from grid reference')
        current, previous = target_meta['files'][target], old_target['files'][target]
        for kind in ('amp', 'label'):
            if current['data_files'][kind] != previous['data_files'][kind] or current['actual_file_sha256'][kind] != previous['actual_file_sha256'][kind]:
                raise RuntimeError('Amplitude/labels differ from the matched time-grid reference')
        for kind in ('amp', 'phase', 'label'):
            item = current['data_files'][kind]
            path = Path(item['path'])
            digest = sha256(path)
            if digest != current['actual_file_sha256'][kind] or digest != item['sha256']:
                raise RuntimeError(f'Input bytes changed: {target}/{kind}')
            actual_files[str(path)] = digest
            values = np.load(path, mmap_mode='r', allow_pickle=False)
            if list(values.shape) != item['shape'] or str(values.dtype) != item['dtype']:
                raise RuntimeError('Input shape/dtype mismatch')
            for offset in range(0, len(values), 32):
                if not np.isfinite(values[offset:offset + 32]).all():
                    raise RuntimeError('Input contains nonfinite values')
            values._mmap.close()
        identity_checks[target] = {'train': len(source['train']), 'val': len(source['val']),
                                   'test': len(target_meta['test']), 'ordered_rows_equal': True,
                                   'amplitude_reference': 'core/R10' if scope == 'history' else 'inputs/NATIVE_LINEAR'}
        if scope == 'history':
            original_spec = previous['data_files']['phase']
            original_path = Path(original_spec['path'])
            if sha256(original_path) != previous['actual_file_sha256']['phase']:
                raise RuntimeError('Original historical phase bytes changed')
            actual_files[str(original_path)] = previous['actual_file_sha256']['phase']
            original = np.load(original_path, mmap_mode='r', allow_pickle=False)
            compiled = np.load(current['data_files']['phase']['path'], mmap_mode='r', allow_pickle=False)
            for start in range(0, 3000, 16):
                expected = historical_feature_phase(original[start:start + 16])
                if not np.array_equal(expected, compiled[start:start + 16]):
                    raise RuntimeError('Historical phase differs from original float64 unwrap-then-cast formula')
            original._mmap.close(); compiled._mmap.close()
            formula_checks[target] = {'rows': 3000, 'all_values_exact': True,
                                     'formula': 'np.unwrap(original_float64_phase, axis=-1).astype(np.float32)'}
        print(f'Input bytes and grid identity checked: {scope}/{target}', flush=True)
    compile_path = directory / 'compile_config.json'
    if sha256(compile_path) != manifest['artifacts_sha256']['compile_config.json']:
        raise RuntimeError('Compilation definition changed')
    return {'passed': True, 'scope': scope, 'variant': variant,
            'dataset_manifest_sha256': sha256(directory / 'dataset_manifest.json'),
            'compile_config_sha256': sha256(compile_path), 'verified_signal_files_sha256': actual_files,
            'identities': identity_checks, 'historical_formula_checks': formula_checks,
            'input_equivalence_only': True, 'target_model_inference_performed': False}


def source_qa(scope, output, gpu_evidence=None):
    if output.exists():
        raise FileExistsError('Preserve previous QA attempts')
    output.mkdir(parents=True)
    verify_bundle(CORE); verify_extension(INPUTS)
    code = code_check()
    data = check_data(scope)
    write(output / 'input_check.json', data)
    runtime = runtime_settings()
    jobs, _ = configurations(scope)
    if any(c['expected_runtime'] != runtime for c in jobs):
        raise RuntimeError('Runtime differs from frozen twin references')
    states, initialization = {}, []
    for config in jobs:
        model = build_experiment_model(config)
        direct = build_twin_model(config['seed'])
        if not equal_state(model.state_dict(), direct.state_dict()):
            raise RuntimeError('Twin initialization differs from frozen implementation')
        if config['seed'] in states and not equal_state(model.state_dict(), states[config['seed']]):
            raise RuntimeError('AA and AP initializations differ')
        states[config['seed']] = {k:v.clone() for k,v in model.state_dict().items()}
        initialization.append({'job_id': config['job_id'], 'twin_reference_exact': True,
                               'total_parameters': sum(p.numel() for p in model.parameters())})
        del model, direct
    del states
    first = jobs[0]
    dataset = VerifiedInputs(first['source_manifest_path'], ['amp', 'phase'])
    positions = epoch_schedule(dataset.manifest['train'], first['seed'], 0, steps=1)[0]
    inputs, labels, rows = dataset.batch('train', positions, 'cpu')
    checks = []
    for control in sorted({c['control_id'] for c in jobs}):
        config = deepcopy(next(c for c in jobs if c['control_id'] == control))
        config['device'] = 'cpu'
        model = build_experiment_model(config)
        direct = build_twin_model(config['seed'])
        state = rng_state()
        loss, _, regularizer, weight, _ = objective(model, inputs, labels, rows, 1, config)
        loss.backward()
        restore_rng(state)
        expected = F.cross_entropy(direct(inputs['amp'], secondary=inputs[config['experiment']['secondary_input']], project=False)['logits'], labels)
        expected.backward()
        exact_gradients = all((a.grad is None and b.grad is None) or
                              (a.grad is not None and b.grad is not None and torch.equal(a.grad, b.grad))
                              for a,b in zip(model.parameters(), direct.parameters()))
        active = sum(p.numel() for p in model.parameters() if p.grad is not None)
        finite = bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
        if not torch.equal(loss, expected) or not exact_gradients or not finite or active != 3556598 or float(regularizer) != 0 or weight != 0:
            raise RuntimeError('CPU source CE/gradient equivalence failed')
        checks.append({'control': control, 'device': 'cpu', 'direct_CE_exact': True, 'gradients_exact': True,
                       'finite_loss_gradients': finite, 'active_CE_parameters': active, 'total_parameters': 3655286,
                       'loss': float(loss.detach()), 'source_sample_ids': [r['sample_id'] for r in rows], 'optimizer_steps': 0})
        print(f'Source CPU twin CE and gradients exact: {control}', flush=True)
        del model,direct,loss,expected
    gpu = None
    if scope == 'history':
        if gpu_evidence is None:
            raise RuntimeError('Historical source GPU smoke evidence is required')
        gpu = read(gpu_evidence)
        if not gpu['passed'] or gpu['target_data_loaded'] or gpu['runtime'] != runtime:
            raise RuntimeError('Invalid GPU smoke evidence')
        if gpu['code_sha256'] != {k:v for k,v in code.items() if k in INPUT_CODE}:
            raise RuntimeError('GPU smoke used different scientific code')
        for record in gpu['checks']:
            config = next(c for c in jobs if c['control_id'] == record['control'] and c['seed'] == 401)
            if record['config_value_sha256'] != value_hash(config) or not record['finite_loss_gradients']:
                raise RuntimeError('GPU smoke configuration differs')
        if {r['control'] for r in gpu['checks']} != {'HIST_TWIN_AA','HIST_TWIN_AP'}:
            raise RuntimeError('Incomplete historical GPU smoke')
        shutil.copy2(gpu_evidence, output / 'gpu_smoke.json')
    prior_resume = read(INPUTS / 'input_study_qa.json')
    if not prior_resume['passed'] or not all(prior_resume['CPU_twin_resume'].values()):
        raise RuntimeError('Frozen twin engine resume evidence is invalid')
    report = {'passed': True, 'checked_at': now(), 'scope': scope, 'runtime': runtime,
              'code_sha256': code, 'input_check_sha256': sha256(output / 'input_check.json'),
              'initialization_checks': initialization, 'CPU_checks': checks,
              'GPU_smoke_sha256': sha256(output / 'gpu_smoke.json') if gpu else None,
              'GPU_note': 'Historical AA/AP source smoke passed; native new input checked on CPU to avoid concurrent GPU use.' if scope == 'native' else 'Finite GPU gradients; bitwise CUDA backward or Adam equality is not asserted.',
              'frozen_twin_resume_qa_sha256': sha256(INPUTS / 'input_study_qa.json'),
              'source_model_target_data_loaded': False, 'all_environment_input_QC_performed': True,
              'target_signals_opened_for_input_QC': True, 'target_model_inference_performed': False,
              'optimizer_steps_performed': 0,
              'full_training_completed': False}
    write(output / 'source_qa.json', report)
    print(f'{scope} source QA passed; no formal training started', flush=True)


def reference_records():
    records = []
    for role, path, verifier in (('core', CORE, verify_bundle), ('inputs', INPUTS, verify_extension)):
        _, digest = verifier(path)
        records.append({'role': role, 'bundle': str(path), 'bundle_sha256': digest,
                        'evaluation_summary_sha256': sha256(path / 'evaluation/summary.json')})
    return records


def verify_references(references):
    evidence = []
    for ref in references:
        study = _verify_study(ref['bundle'], ref['bundle_sha256'], ref['evaluation_summary_sha256'])
        arms = {'R10'} if ref['role'] == 'core' else {'NATIVE_LINEAR','TWIN_AA','TWIN_AP'}
        selected = [r for r in study['results'].values() if r['arm'] in arms]
        for result in selected:
            _verified_run(study, result)
        evidence.append({**ref, 'verified_runs': len(selected)})
    return evidence


def freeze(bundle, history_qa, native_qa):
    if bundle.exists():
        raise FileExistsError('Never overwrite a frozen study')
    code = code_check()
    for scope, path in (('history', history_qa), ('native', native_qa)):
        report = read(path)
        if not report['passed'] or report['scope'] != scope or report['source_model_target_data_loaded'] or report['target_model_inference_performed'] or report['code_sha256'] != code:
            raise RuntimeError('Source QA is stale or invalid')
        data = read(path.parent / 'input_check.json')
        if sha256(path.parent / 'input_check.json') != report['input_check_sha256'] or not data['passed']:
            raise RuntimeError('QA input evidence changed')
        if report['frozen_twin_resume_qa_sha256'] != sha256(INPUTS / 'input_study_qa.json'):
            raise RuntimeError('Frozen twin resume QA changed')
        if scope == 'history' and sha256(path.parent / 'gpu_smoke.json') != report['GPU_smoke_sha256']:
            raise RuntimeError('Historical GPU smoke evidence changed')
        if sha256(data_root(scope) / 'dataset_manifest.json') != data['dataset_manifest_sha256']:
            raise RuntimeError('Completed input dataset changed')
        if sha256(data_root(scope) / 'compile_config.json') != data['compile_config_sha256']:
            raise RuntimeError('Input compilation definition changed')
        for filename, expected in data['verified_signal_files_sha256'].items():
            if sha256(filename) != expected:
                raise RuntimeError('Input bytes changed since source QA')
    references = reference_records()
    verify_references(references)
    input_provenance = []
    for scope, path in (('history', history_qa), ('native', native_qa)):
        destination = bundle / 'qa' / scope
        destination.mkdir(parents=True)
        for name in ('source_qa.json', 'input_check.json', 'gpu_smoke.json'):
            if (path.parent / name).exists():
                shutil.copy2(path.parent / name, destination / name)
        provenance_dir = bundle / 'data_provenance' / scope
        provenance_dir.mkdir(parents=True)
        for name in ('dataset_manifest.json', 'compile_config.json'):
            shutil.copy2(data_root(scope) / name, provenance_dir / name)
        input_provenance.append({'scope': scope, 'original_directory': str(data_root(scope)),
                                 'dataset_manifest_sha256': sha256(provenance_dir / 'dataset_manifest.json'),
                                 'compile_config_sha256': sha256(provenance_dir / 'compile_config.json')})
    for scope, variant in (('history', HIST_VARIANT), ('native', NATIVE_VARIANT)):
        for path in (data_root(scope) / variant / 'manifests').glob('*.json'):
            destination = bundle / 'manifests' / variant / path.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    for name in INPUT_CODE + [SELF, HELPER]:
        source = INPUTS / 'code' / name if name in INPUT_CODE else ROOT / name
        destination = bundle / 'code' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    jobs, bindings = configurations(bundle=bundle)
    if len(jobs) != 27:
        raise RuntimeError('Require all 27 predetermined runs')
    protocol = {'schema': 'revision-extension-study/v1', 'study_id': bundle.name,
                'study_kind': 'matched_phase_followup', 'frozen_at': now(),
                'entry_module': 'experiments.revision_validation.phase_followup_study',
                'jobs': jobs, 'job_count': 27, 'paired_config_bindings': bindings, 'references': references,
                'input_provenance': input_provenance,
                'comparisons': [], 'prior_target_results_inspected': True, 'hyperparameter_search_trials': 0,
                'target_gate': 'All 27 new source runs completed and verified before any new model target inference',
                'selection': 'Earliest maximum source validation macro-F1; 4800/1200 subject-grouped partition',
                'training_budget': '200 epochs x 300 updates, batch 16, unchanged Adam/LR/sampling/runtime',
                'inference_scope': 'Previously inspected E1-E3 development evidence. Three matched seeds per environment. Descriptive differences, no population inference.',
                'reporting': 'Historical A, AA, AP; native A, AA, AP-differential, AP-historical-style. All runs and negative effects retained.',
                'limits': ['Historical feature unwrap crosses receive-link boundaries; reproduction convention, no physical calibration claim.',
                           'Native historical-style processing is a separately versioned chain diagnostic, not the original historical phase array.',
                           'Native historical-style: DAT wrapped angle arithmetic linear interpolation on native 850-point grid, feature-axis unwrap, float32. No differential phase, temporal unwrap or time centering.',
                           'AA versus AP within a grid controls twin capacity; twin versus single A changes capacity.',
                           'Do not interpret cross-grid comparisons as a single-factor phase experiment.',
                           'GPU backward can vary between repeated identical calls; same seed does not guarantee bitwise replay.']}
    write(bundle / 'protocol.json', protocol)
    files = [p for p in bundle.rglob('*') if p.is_file()]
    write(bundle / 'bundle_manifest.json', {'frozen_at': now(), 'files_sha256': {p.relative_to(bundle).as_posix():sha256(p) for p in files}})
    _, digest = verify_extension(bundle)
    print(f'Frozen 27 source runs: {bundle}; SHA256 {digest}', flush=True)


def summarize(bundle):
    new = _verify_study(bundle)
    if len(new['jobs']) != 27:
        raise RuntimeError('New phase study must contain exactly 27 prespecified runs')
    references = new['protocol']['references']
    if len(references) != 2 or {ref['role'] for ref in references} != {'core', 'inputs'}:
        raise RuntimeError('Exactly one core and one inputs reference are required')
    studies = {'new': new}
    for ref in references:
        studies[ref['role']] = _verify_study(ref['bundle'], ref['bundle_sha256'], ref['evaluation_summary_sha256'])
    cells = [('historical','A','core','R10'), ('historical','AA','new','HIST_TWIN_AA'),
             ('historical','AP_historical','new','HIST_TWIN_AP'), ('native','A','inputs','NATIVE_LINEAR'),
             ('native','AA','inputs','TWIN_AA'), ('native','AP_differential','inputs','TWIN_AP'),
             ('native','AP_historical_style','new','NATIVE_HISTSTYLE_AP')]
    rows, lookup, ids = [], {}, {}
    expected = {(t,s) for t,seeds in SEEDS.items() for s in seeds}
    for grid, cell, role, arm in cells:
        study = studies[role]
        selected = [r for r in study['results'].values() if r['arm'] == arm]
        if len(selected) != 9 or {(r['target'],r['seed']) for r in selected} != expected:
            raise RuntimeError('Incomplete cell or mismatched seeds')
        for result in sorted(selected, key=lambda r:(r['target'],r['seed'])):
            identities, status = _verified_run(study, result)
            key = (grid,result['target'])
            if key in ids and ids[key] != identities:
                raise RuntimeError('Within-grid target identities differ')
            ids[key] = identities
            row = {'grid':grid,'cell':cell,'arm':arm,'target':result['target'],'seed':result['seed'],
                   'job_id':result['job_id'],'macro_f1':result['metrics']['macro_f1'],
                   'macro_f1_pct':100*result['metrics']['macro_f1'],'source_best_epoch':status['best_epoch'],
                   'config_value_sha256':result['config_value_sha256'],'checkpoint_sha256':result['checkpoint_sha256'],
                   'result_sha256':sha256(study['bundle']/'evaluation'/result['job_id']/'result.json'),
                   'predictions_sha256':sha256(study['bundle']/'evaluation'/result['job_id']/'predictions.json')}
            rows.append(row); lookup[(grid,cell,result['target'],result['seed'])] = row
    contrasts = [('historical','AP_historical','AA','incremental_phase_at_equal_twin_capacity'),
                 ('historical','AA','A','added_encoder_capacity'), ('historical','AP_historical','A','modality_plus_capacity'),
                 ('native','AP_historical_style','AA','incremental_phase_at_equal_twin_capacity'),
                 ('native','AP_differential','AA','incremental_phase_at_equal_twin_capacity'),
                 ('native','AP_differential','AP_historical_style','whole_phase_representation_chain'),
                 ('native','AA','A','added_encoder_capacity'), ('native','AP_historical_style','A','modality_plus_capacity')]
    pairs = []
    for grid,treatment,reference,meaning in contrasts:
        for target,seed in sorted(expected):
            effect = 100*(lookup[(grid,treatment,target,seed)]['macro_f1'] - lookup[(grid,reference,target,seed)]['macro_f1'])
            pairs.append({'grid':grid,'comparison':treatment+' minus '+reference,'interpretation':meaning,
                          'target':target,'seed':seed,'effect_pp':effect})
    summaries, cell_summary = [], []
    for grid,cell,_,_ in cells:
        for target in ('ALL',*SEEDS):
            values = [r['macro_f1_pct'] for r in rows if r['grid']==grid and r['cell']==cell and (target=='ALL' or r['target']==target)]
            cell_summary.append({'grid':grid,'cell':cell,'target':target,**_stats(values)})
    for grid,treatment,reference,meaning in contrasts:
        name=treatment+' minus '+reference
        for target in ('ALL',*SEEDS):
            values=[r['effect_pp'] for r in pairs if r['grid']==grid and r['comparison']==name and (target=='ALL' or r['target']==target)]
            summaries.append({'grid':grid,'comparison':name,'target':target,**_stats(values)})
    report={'schema':'revision-phase-followup-comparison/v1','complete':True,'run_count':63,'new_run_count':27,
            'runs':rows,'cells':cell_summary,'paired_differences':pairs,'paired_summary':summaries,
            'all_predeclared_seeds_retained':True,'inference_scope':new['protocol']['inference_scope'],
            'limits':new['protocol']['limits'],
            'provenance':{role:{'bundle':str(s['bundle']),'bundle_sha256':s['bundle_sha256'],
                                'evaluation_summary_sha256':s['evaluation_summary_sha256']} for role,s in studies.items()}}
    if len(rows)!=63 or len(lookup)!=63:
        raise RuntimeError('Incomplete seven-cell evidence matrix')
    output=bundle/'comparison'; output.mkdir(exist_ok=True)
    path=output/'report.json'
    if path.exists() and read(path)!=report:
        raise RuntimeError('Existing completed report differs')
    if not path.exists(): write(path,report)
    for name in ('runs','cells','paired_differences','paired_summary'):
        stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=list(report[name][0]),lineterminator='\n'); writer.writeheader();writer.writerows(report[name])
        _write_text_once(output/(name+'.csv'),stream.getvalue())
    lines=['# Phase follow-up: all prespecified comparisons','',
           '63 runs across seven cells: 27 new runs and 36 frozen references. Equal-weight environment means; all three seeds per environment retained.','',
           '| Grid | Cell | Macro-F1 (%) |','|---|---|---:|']
    lines += [f"| {r['grid']} | {r['cell']} | {r['mean_pp']:.3f} |" for r in cell_summary if r['target']=='ALL']
    lines += ['', '| Grid | Within-grid contrast | Difference (pp) | Positive / zero / negative performance differences |','|---|---|---:|---|']
    lines += [f"| {r['grid']} | {r['comparison']} | {r['mean_pp']:+.3f} | {r['positive_pairs']} / {r['zero_pairs']} / {r['negative_pairs']} |" for r in summaries if r['target']=='ALL']
    lines += ['',*['- '+item for item in report['limits']],'']
    _write_text_once(output/'report.md','\n'.join(lines))
    return sha256(path)


def run(bundle):
    protocol,digest=verify_extension(bundle)
    if ROOT != bundle/'code': raise RuntimeError('Use this study immutable code snapshot')
    if (bundle/'queue_failure.json').exists(): raise RuntimeError('Diagnose prior recorded failure before continuing')
    lock,identity=acquire_lock(bundle)
    try:
        write(bundle/'reference_gate.json',{'passed_at':now(),'references':verify_references(protocol['references'])})
        for index,config in enumerate(protocol['jobs']):
            verify_extension(bundle)
            write(bundle/'queue_status.json',{'status':'running','job_index':index,'job_id':config['job_id'],
                  'total_jobs':27,'bundle_sha256':digest,'pid':os.getpid(),'updated_at':now()})
            train_experiment(config,bound_path(bundle,'runs/'+config['job_id']))
        write(bundle/'queue_status.json',{'status':'source_training_completed','jobs':27,'updated_at':now()})
        from experiments.revision_validation.extended_evaluate import evaluate
        evaluation_hash=evaluate(bundle)
        report_hash=summarize(bundle)
        write(bundle/'queue_status.json',{'status':'completed','jobs':27,'bundle_sha256':digest,'updated_at':now(),
              'evaluation_summary_sha256':evaluation_hash,'comparison_report_sha256':report_hash})
    except BaseException:
        write(bundle/'queue_failure.json',{'status':'failed','updated_at':now(),'error':traceback.format_exc()});raise
    finally:
        if lock.exists() and read(lock)==identity: lock.unlink()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('qa-history','qa-native','freeze','run','report'))
    parser.add_argument('--output',type=Path);parser.add_argument('--bundle',type=Path)
    parser.add_argument('--gpu-evidence',type=Path);parser.add_argument('--history-qa',type=Path);parser.add_argument('--native-qa',type=Path)
    args=parser.parse_args()
    if args.action.startswith('qa-'): source_qa(args.action.split('-')[1],args.output.resolve(),args.gpu_evidence)
    elif args.action=='freeze': freeze(args.bundle.resolve(),args.history_qa,args.native_qa)
    elif args.action=='run': run(args.bundle.resolve())
    else: print(summarize(args.bundle.resolve()))


if __name__=='__main__': main()
