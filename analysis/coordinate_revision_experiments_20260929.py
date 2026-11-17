"""Freeze and execute the two remaining study batches sequentially on one GPU."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import psutil

RUNTIME = Path('D:/pa-csi-dg-revision-runtime')
DEFAULT = RUNTIME / 'coordinators/full_revision_20260929'
BUNDLES = [RUNTIME / 'studies/r_time_pooling_v1_20260929', RUNTIME / 'studies/r_phase_controls_v1_20260929']
MODULES = ['experiments.revision_validation.time_pooling_study', 'experiments.revision_validation.phase_followup_study']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def alive(identity):
    try:
        return abs(psutil.Process(identity['pid']).create_time() - identity['create_time']) < .1
    except psutil.NoSuchProcess:
        return False


def verify(bundle, expected=None):
    digest = sha(bundle / 'bundle_manifest.json')
    if expected is not None and digest != expected:
        raise RuntimeError('Study freeze changed')
    for name, value in read(bundle / 'bundle_manifest.json')['files_sha256'].items():
        path = (bundle / name).resolve()
        if not path.is_relative_to(bundle.resolve()) or sha(path) != value:
            raise RuntimeError('Frozen study file differs: ' + name)
    return read(bundle / 'protocol.json'), digest


def snapshot(plan):
    rows = []
    for entry in plan['studies']:
        bundle = Path(entry['bundle'])
        protocol = read(bundle / 'protocol.json')
        jobs = []
        for config in protocol['jobs']:
            path = bundle / 'runs' / config['job_id'] / 'status.json'
            value = read(path) if path.exists() else {'status': 'pending'}
            jobs.append({'job_id': config['job_id'], **{key: value.get(key) for key in ('status', 'epoch', 'epochs', 'updated_at')}})
        rows.append({'bundle': str(bundle), 'total': len(jobs),
                     'completed': sum(j['status'] == 'completed' for j in jobs),
                     'active': [j for j in jobs if j['status'] not in ('pending', 'completed')],
                     'queue': read(bundle / 'queue_status.json') if (bundle / 'queue_status.json').exists() else {'status': 'pending'}})
    return rows


def completed(bundle):
    path = bundle / 'queue_status.json'
    if not path.exists() or read(path)['status'] != 'completed':
        return False
    state = read(path)
    protocol, digest = verify(bundle)
    jobs = {c['job_id']: c for c in protocol['jobs']}
    if len(jobs) != len(protocol['jobs']) or state.get('bundle_sha256') != digest or state.get('jobs') != len(jobs):
        raise RuntimeError('Completed queue does not match its frozen study')
    if sha(bundle / 'evaluation/summary.json') != state['evaluation_summary_sha256']:
        raise RuntimeError('Completed evaluation checksum differs')
    summary = read(bundle / 'evaluation/summary.json')
    if len(summary['runs']) != len(jobs) or {r['job_id'] for r in summary['runs']} != set(jobs):
        raise RuntimeError('Completed evaluation is missing or duplicating configured jobs')
    gate = read(bundle / 'target_gate.json')
    if gate['bundle_sha256'] != digest or set(gate['source_status_sha256']) != set(jobs):
        raise RuntimeError('Target gate does not cover the frozen source jobs')
    for job_id, config in jobs.items():
        run = bundle / 'runs' / job_id
        source = read(run / 'status.json')
        expected_config = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')).hexdigest()
        if source['status'] != 'completed' or source['epochs'] != config['epochs'] or source['optimizer_updates'] != config['epochs'] * config['updates_per_epoch'] or source['config_value_sha256'] != expected_config:
            raise RuntimeError('A required source job is incomplete or has a different configuration')
        if sha(run / 'status.json') != gate['source_status_sha256'][job_id]:
            raise RuntimeError('Source completion differs from its target gate')
        for name, expected in source['artifacts_sha256'].items():
            artifact = (run / name).resolve()
            if not artifact.is_relative_to(run.resolve()) or sha(artifact) != expected:
                raise RuntimeError('Completed source artifact differs')
    for name, expected in summary['artifacts_sha256'].items():
        artifact = (bundle / name).resolve()
        if not artifact.is_relative_to(bundle.resolve()) or sha(artifact) != expected:
            raise RuntimeError('Completed evaluation artifact differs')
    if sha(bundle / 'comparison/report.json') != state['comparison_report_sha256']:
        raise RuntimeError('Final comparison report checksum differs')
    return True


def run(plan_path):
    directory = plan_path.parent
    plan = read(plan_path)
    if sha(__file__) != plan['coordinator_code_sha256']:
        raise RuntimeError('Coordinator differs from frozen code')
    lock = directory / 'coordinator.lock.json'
    identity = {'pid': os.getpid(), 'create_time': psutil.Process().create_time()}
    if lock.exists():
        if alive(read(lock)):
            raise RuntimeError('Coordinator already running')
        lock.rename(directory / f'stale_lock_{time.time_ns()}.json')
    with lock.open('x', encoding='utf-8') as f:
        json.dump(identity, f)
    try:
        for index, entry in enumerate(plan['studies']):
            bundle = Path(entry['bundle'])
            protocol, _ = verify(bundle, entry['bundle_sha256'])
            if (bundle / 'queue_failure.json').exists():
                raise RuntimeError(f'Diagnose recorded study failure: {bundle}')
            if completed(bundle):
                continue
            existing = None
            for name in ('runner.lock.json', 'launcher.json'):
                if (bundle / name).exists() and alive(read(bundle / name)):
                    existing = read(bundle / name)
                    break
            if existing:
                print(f'Waiting for the already-running study: {bundle.name}', flush=True)
                while alive(existing):
                    if (bundle / 'queue_failure.json').exists():
                        raise RuntimeError(f'Study recorded failure: {bundle}')
                    write(directory / 'status.json', {'status': 'running', 'study_index': index, 'updated_at': now(),
                                                     'pid': os.getpid(), 'studies': snapshot(plan)})
                    time.sleep(30)
                if not completed(bundle):
                    raise RuntimeError('Existing runner stopped without completing; inspect preserved logs/checkpoints')
            else:
                command = [plan['python'], '-u', '-m', protocol['entry_module'], 'run', '--bundle', str(bundle)]
                env = os.environ.copy()
                env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                env['PYTHONUNBUFFERED'] = '1'
                env.pop('PYTHONPATH', None)
                stdout = directory / f'{index}_{bundle.name}.stdout.log'
                stderr = directory / f'{index}_{bundle.name}.stderr.log'
                with stdout.open('ab') as out, stderr.open('ab') as err:
                    process = subprocess.Popen(command, cwd=bundle / 'code', env=env,
                                               stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                write(directory / f'{index}_child.json', {'pid': process.pid,
                      'create_time': psutil.Process(process.pid).create_time(), 'command': command, 'started_at': now()})
                while process.poll() is None:
                    write(directory / 'status.json', {'status': 'running', 'study_index': index, 'updated_at': now(),
                                                     'pid': os.getpid(), 'studies': snapshot(plan)})
                    time.sleep(30)
                if process.returncode != 0 or not completed(bundle):
                    raise RuntimeError(f'Study failed; no automatic numerical-failure retry: {bundle}')
            verify(bundle, entry['bundle_sha256'])
        write(directory / 'status.json', {'status': 'completed', 'updated_at': now(), 'studies': snapshot(plan)})
        print('All 45 new training jobs, deferred evaluations and study comparison reports completed.', flush=True)
    except BaseException:
        write(directory / 'failure.json', {'failed_at': now(), 'error': traceback.format_exc()})
        raise
    finally:
        if lock.exists() and read(lock) == identity:
            lock.unlink()


def prepare_and_start(directory):
    if directory.exists():
        raise FileExistsError('Preserve existing coordinator and its logs')
    entries = []
    for bundle, module, count in zip(BUNDLES, MODULES, (18, 27)):
        protocol, digest = verify(bundle)
        if protocol['entry_module'] != module or len(protocol['jobs']) != count:
            raise RuntimeError('Unexpected study module or job count')
        entries.append({'bundle': str(bundle), 'bundle_sha256': digest, 'job_count': count})
    directory.mkdir(parents=True)
    frozen_code = directory / Path(__file__).name
    frozen_code.write_bytes(Path(__file__).read_bytes())
    plan = {'created_at': now(), 'python': 'C:/ProgramData/anaconda3/python.exe',
            'studies': entries, 'new_training_runs': 45, 'execution': 'sequential; use one GPU',
            'coordinator_code_sha256': sha(frozen_code),
            'scope': 'Full shared revision plan: reuse completed comparisons; fill time preprocessing and modality gaps',
            'result_policy': 'Retain all predefined environments/seeds and negative results'}
    plan_path = directory / 'plan.json'
    write(plan_path, plan)
    with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
        process = subprocess.Popen([plan['python'], '-u', str(frozen_code), 'run', '--directory', str(directory)],
                                   cwd=directory, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    identity = {'pid': process.pid, 'create_time': psutil.Process(process.pid).create_time(), 'started_at': now()}
    write(directory / 'launcher.json', identity)
    print(json.dumps({'directory': str(directory), **identity}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'run', 'status'))
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    args = parser.parse_args()
    if args.action == 'start':
        prepare_and_start(args.directory.resolve())
    elif args.action == 'run':
        run(args.directory.resolve() / 'plan.json')
    else:
        print(json.dumps(snapshot(read(args.directory / 'plan.json')), ensure_ascii=False, indent=2))
