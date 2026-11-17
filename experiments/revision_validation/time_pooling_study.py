"""Dated developmental follow-up: native-time inputs crossed with global MLP.

Historical snapshots and results are read-only references. Training, sampling,
checkpoint selection and evaluation reuse the exact frozen input-study code.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
import traceback

import torch
from torch.nn import functional as F

from experiments.revision_validation.experiment_data import VerifiedInputs
from experiments.revision_validation.experiment_design import build_experiment_model, objective
from experiments.revision_validation.extended_engine import train_experiment, verify_extension
from experiments.revision_validation.freeze_inputs import INPUT_CODE
from experiments.revision_validation.models import build_model
from experiments.revision_validation.source_data import epoch_schedule
from experiments.revision_validation.study_engine import acquire_lock, equal_state, restore_rng, rng_state, runtime_settings
from experiments.revision_validation.study_io import bound_path, now, read, sha256, verify_bundle, verify_completed, write

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path('D:/pa-csi-dg-revision-runtime')
CORE = RUNTIME / 'studies/r_core_v1'
INPUTS = RUNTIME / 'studies/r_inputs_v1'
CONTROLS = ('NATIVE_MERGE', 'NATIVE_LINEAR')
SELF = 'experiments/revision_validation/time_pooling_study.py'
REPORT = 'experiments/revision_validation/time_pooling_report.py'
EXPECTED_PAIRS = {(t, s, c) for t, b in (('E1', 400), ('E2', 500), ('E3', 600))
                  for s in range(b + 1, b + 4) for c in CONTROLS}


def matched_configs(bundle=None):
    originals = [c for c in read(INPUTS / 'protocol.json')['jobs'] if c['control_id'] in CONTROLS]
    if {(c['target_name_only'], c['seed'], c['control_id']) for c in originals} != EXPECTED_PAIRS or len(originals) != 18:
        raise RuntimeError('The reference must contain all eighteen predefined TPP runs')
    jobs, bindings = [], []
    for old in originals:
        config = deepcopy(old)
        control = old['control_id']
        config['job_id'] = f"{old['target_name_only']}_G_{control}_{old['seed']}"
        config['control_id'] = 'GLOBAL_' + control
        config['arm'] = 'R00'
        config['experiment']['base_arm'] = 'R00'
        if bundle is not None:
            for kind in ('source', 'target'):
                old_path = Path(old[kind + '_manifest_path'])
                config[kind + '_manifest_path'] = str(bundle / 'manifests' / old['input_variant'] / old_path.name)
                if sha256(config[kind + '_manifest_path']) != old[kind + '_manifest_sha256']:
                    raise RuntimeError('Copied manifest content differs from its matched TPP input')
        allowed = {'job_id', 'control_id', 'arm', 'experiment', 'source_manifest_path', 'target_manifest_path'}
        differences = [k for k in old if old[k] != config[k]]
        experiment = deepcopy(config['experiment'])
        experiment['base_arm'] = 'R10'
        if set(differences) - allowed or experiment != old['experiment']:
            raise RuntimeError('Unexpected paired configuration difference')
        jobs.append(config)
        bindings.append({'job_id': config['job_id'], 'reference_job_id': old['job_id'],
                         'changed_top_level_fields': differences,
                         'reference_config_sha256': sha256(INPUTS / 'runs' / old['job_id'] / 'config.json')})
    return jobs, bindings


def code_check():
    evidence = {}
    for name in INPUT_CODE:
        frozen = sha256(INPUTS / 'code' / name)
        if sha256(ROOT / name) != frozen:
            raise RuntimeError(f'Live source differs from the frozen input-study implementation: {name}')
        evidence[name] = frozen
    evidence[SELF] = sha256(ROOT / SELF)
    return evidence


def source_qa(output):
    if output.exists():
        raise FileExistsError('Preserve earlier QA attempts; choose another output directory')
    output.mkdir(parents=True)
    code = code_check()
    runtime = runtime_settings()
    jobs, _ = matched_configs()
    if any(c['expected_runtime'] != runtime for c in jobs):
        raise RuntimeError('Installed training runtime differs from the paired historical runtime')
    checks = []
    for name, device in ((name, device) for name in CONTROLS for device in ('cuda:0', 'cpu')):
        config = deepcopy(next(c for c in jobs if c['control_id'] == 'GLOBAL_' + name))
        config['device'] = device
        dataset = VerifiedInputs(config['source_manifest_path'], config['input_modalities'])
        positions = epoch_schedule(dataset.manifest['train'], config['seed'], 0, steps=1)[0]
        inputs, labels, rows = dataset.batch('train', positions, config['device'])
        candidate = build_experiment_model(config).to(config['device'])
        reference = build_model('R00', config['seed']).to(config['device'])
        initial_exact = equal_state(candidate.state_dict(), reference.state_dict())
        state = rng_state()
        actual, _, regularizer, weight, _ = objective(candidate, inputs, labels, rows, 1, config)
        actual.backward()
        active = sum(p.numel() for p in candidate.parameters() if p.grad is not None)
        finite = bool(torch.isfinite(actual)) and all(bool(torch.isfinite(p.grad).all()) for p in candidate.parameters() if p.grad is not None)
        torch.optim.Adam(candidate.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay']).step()
        restore_rng(state)
        expected = F.cross_entropy(reference(inputs['amp'], project=False)['logits'], labels)
        expected.backward()
        torch.optim.Adam(reference.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay']).step()
        check = {'control': name, 'device': device, 'initialization_exact': initial_exact,
                 'direct_CE_exact': bool(torch.equal(actual, expected)),
                 'Adam_update_exact': equal_state(candidate.state_dict(), reference.state_dict()),
                 'finite_loss_and_gradients': finite, 'active_CE_parameters': active,
                 'total_parameters': sum(p.numel() for p in candidate.parameters()),
                 'loss': float(actual.detach()), 'target_data_loaded': False,
                 'source_sample_ids': [r['sample_id'] for r in rows]}
        required = ['initialization_exact', 'direct_CE_exact', 'finite_loss_and_gradients']
        if device == 'cpu':
            required.append('Adam_update_exact')
        if not all(check[k] for k in required) or active != 1779454 or check['total_parameters'] != 1878142 or weight != 0 or float(regularizer) != 0:
            write(output / 'failed_objective_check.json', check)
            raise RuntimeError('Source objective/finite/parameter check failed')
        checks.append(check)
        print(f'Source objective checked: {name}, {device}, Adam exact={check["Adam_update_exact"]}', flush=True)
        del candidate, reference, inputs, dataset
    config = deepcopy(jobs[0])
    config.update(device='cpu', epochs=2, updates_per_epoch=2, resource_profile=False)
    continuous, resumed = output / 'continuous', output / 'resumed'
    train_experiment(config, continuous)
    train_experiment(config, resumed, pause_after_epoch=1)
    train_experiment(config, resumed)
    states = [torch.load(p / read(p / 'progress.json')['checkpoint'], map_location='cpu', weights_only=True) for p in (continuous, resumed)]
    resume = {k + '_exact': equal_state(states[0][k], states[1][k]) for k in ('model', 'optimizer', 'rng', 'history')}
    resume['selected_predictions_exact'] = read(continuous / 'selected_source_predictions.json') == read(resumed / 'selected_source_predictions.json')
    diagnostic_path = RUNTIME / 'qa/time_pooling_20260929_diagnostic/diagnostic.json'
    shutil.copy2(diagnostic_path, output / 'gpu_diagnostic.json')
    report = {'passed': all(resume.values()), 'checked_at': now(), 'objective_checks': checks, 'CPU_resume': resume,
              'resume_epochs': 2, 'resume_updates_per_epoch': 2, 'target_data_loaded': False,
              'full_training_completed': False, 'runtime': runtime, 'code_sha256': code,
              'prior_failed_QA': str(RUNTIME / 'qa/time_pooling_20260929_v1/failed_gpu_check.json'),
              'GPU_diagnostic_sha256': sha256(output / 'gpu_diagnostic.json'),
              'GPU_limit': 'Exact forward CE and initialization; finite backward. Direct-to-direct GPU backward also differs. Exact Adam equivalence is required on CPU, not asserted on CUDA; the responsible CUDA operator is unresolved.'}
    write(output / 'source_qa.json', report)
    if not report['passed']:
        raise RuntimeError('CPU interruption/resume trajectories differ')
    print('Source-only GPU equivalence and CPU resume QA passed', flush=True)


def verify_references(protocol):
    records = []
    for ref in protocol['references']:
        path = Path(ref['bundle'])
        verifier = verify_bundle if ref['role'] == 'core' else verify_extension
        p, digest = verifier(path)
        if digest != ref['bundle_sha256'] or sha256(path / 'evaluation/summary.json') != ref['evaluation_summary_sha256']:
            raise RuntimeError('Frozen reference or existing evaluation changed')
        if (path / 'queue_failure.json').exists():
            raise RuntimeError('Reference has an unresolved queue failure')
        selected = [c for c in p['jobs'] if (c['arm'] in ('R00', 'R10') if ref['role'] == 'core' else c['control_id'] in CONTROLS)]
        for config in selected:
            verify_completed(path / 'runs' / config['job_id'], config)
        summary = read(path / 'evaluation/summary.json')
        for relative, expected in summary['artifacts_sha256'].items():
            if sha256(bound_path(path, relative)) != expected:
                raise RuntimeError('Historical prediction/result artifact changed')
        records.append({**ref, 'verified_source_jobs': [c['job_id'] for c in selected]})
    return records


def freeze(bundle, qa_path, input_audit, protocol_audit):
    if bundle.exists():
        raise FileExistsError('Use a new study version; never alter an existing freeze')
    code = code_check()
    qa = read(qa_path)
    if not qa.get('passed') or qa['target_data_loaded'] or qa['code_sha256'] != code:
        raise RuntimeError('Source QA is missing, failed, or bound to different code')
    for path in (input_audit, protocol_audit):
        if not read(path).get('passed'):
            raise RuntimeError(f'Independent audit did not pass: {path}')
    references = []
    for role, path, verifier in (('core', CORE, verify_bundle), ('inputs', INPUTS, verify_extension)):
        _, digest = verifier(path)
        references.append({'role': role, 'bundle': str(path), 'bundle_sha256': digest,
                           'evaluation_summary_sha256': sha256(path / 'evaluation/summary.json')})
    verify_references({'references': references})
    old_protocol = read(INPUTS / 'protocol.json')
    selected = [c for c in old_protocol['jobs'] if c['control_id'] in CONTROLS]
    for config in selected:
        for kind in ('source', 'target'):
            path = Path(config[kind + '_manifest_path'])
            dest = bundle / 'manifests' / config['input_variant'] / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    for name in INPUT_CODE + [SELF, REPORT]:
        source = INPUTS / 'code' / name if name in INPUT_CODE else ROOT / name
        dest = bundle / 'code' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    for path, name in ((qa_path, 'source_qa.json'), (input_audit, 'input_check.json'), (protocol_audit, 'protocol_audit.json')):
        shutil.copy2(path, bundle / name)
    diagnostic = qa_path.parent / 'gpu_diagnostic.json'
    if sha256(diagnostic) != qa['GPU_diagnostic_sha256']:
        raise RuntimeError('GPU diagnosis changed after source QA')
    shutil.copy2(diagnostic, bundle / 'gpu_diagnostic.json')
    jobs, bindings = matched_configs(bundle)
    protocol = {'schema': 'revision-extension-study/v1', 'study_id': bundle.name,
                'study_kind': 'time_preprocessing_matched_global_controls', 'frozen_at': now(),
                'entry_module': 'experiments.revision_validation.time_pooling_study',
                'jobs': jobs, 'job_count': 18, 'paired_config_bindings': bindings,
                'references': references, 'comparisons': [],
                'training_budget': '18 runs, each 200 epochs x 300 updates, batch 16; matched Adam/LR/sampling/runtime',
                'selection': 'Earliest maximum source validation macro-F1; subject-grouped source validation',
                'prior_target_results_inspected': True,
                'registration_scope': 'Developmental follow-up frozen before these eighteen new runs; historical target results were already inspected',
                'target_gate': 'All eighteen new source runs must complete and verify before new target prediction/evaluation',
                'inference_scope': 'E1-E3 development evidence. Equal-environment means, seed SD, paired descriptive differences. No population p-values or confidence intervals.',
                'hyperparameter_search_trials': 0,
                'retention': 'Retain all predefined runs and negative results. No performance early stopping, seed additions or result filtering.',
                'reporting': 'All six preprocessing/pooling cells; within-version TPP-global gains; paired changes in those gains',
                'limits': ['A uses historical MAT; B/C use DAT native elapsed time. A-to-B also includes representation/origin differences.',
                           'B preserves legacy merge and reverse-tail padding; it is not a strict uniform physical-time grid.',
                           'C uses 850 linspace points per recording; variable duration means variable effective Hz. Interpolation does not recover missing motion.',
                           'Three pipelines do not form the full timestamp-by-resampling factorial.',
                           'Identical runtime settings do not guarantee bitwise deterministic CUDA TPP backward.']}
    write(bundle / 'protocol.json', protocol)
    files = [p for p in bundle.rglob('*') if p.is_file()]
    write(bundle / 'bundle_manifest.json', {'frozen_at': now(), 'files_sha256': {p.relative_to(bundle).as_posix(): sha256(p) for p in files}})
    _, digest = verify_extension(bundle)
    print(f'Frozen eighteen paired global controls: {bundle}; SHA256 {digest}', flush=True)


def run(bundle):
    protocol, digest = verify_extension(bundle)
    if ROOT != bundle / 'code':
        raise RuntimeError('Run from this study\'s immutable code snapshot')
    if (bundle / 'queue_failure.json').exists():
        raise RuntimeError('Diagnose and preserve the recorded failure before resuming')
    lock, identity = acquire_lock(bundle)
    try:
        records = verify_references(protocol)
        write(bundle / 'reference_gate.json', {'passed_at': now(), 'references': records})
        for index, config in enumerate(protocol['jobs']):
            verify_extension(bundle)
            write(bundle / 'queue_status.json', {'status': 'running', 'job_index': index, 'job_id': config['job_id'],
                  'total_jobs': len(protocol['jobs']), 'bundle_sha256': digest, 'pid': os.getpid(), 'updated_at': now()})
            train_experiment(config, bound_path(bundle, 'runs/' + config['job_id']))
        write(bundle / 'queue_status.json', {'status': 'source_training_completed', 'jobs': 18, 'updated_at': now()})
        from experiments.revision_validation.extended_evaluate import evaluate
        from experiments.revision_validation.time_pooling_report import summarize
        evaluation_hash = evaluate(bundle)
        report_hash = summarize(bundle)
        write(bundle / 'queue_status.json', {'status': 'completed', 'jobs': 18, 'bundle_sha256': digest,
              'updated_at': now(), 'evaluation_summary_sha256': evaluation_hash, 'comparison_report_sha256': report_hash})
    except BaseException:
        write(bundle / 'queue_failure.json', {'status': 'failed', 'updated_at': now(), 'error': traceback.format_exc()})
        raise
    finally:
        if lock.exists() and read(lock) == identity:
            lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('qa', 'freeze', 'run'))
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--qa-report', type=Path)
    parser.add_argument('--input-audit', type=Path)
    parser.add_argument('--protocol-audit', type=Path)
    args = parser.parse_args()
    if args.action == 'qa':
        source_qa(args.output.resolve())
    elif args.action == 'freeze':
        freeze(args.bundle.resolve(), args.qa_report, args.input_audit, args.protocol_audit)
    else:
        run(args.bundle.resolve())


if __name__ == '__main__':
    main()
