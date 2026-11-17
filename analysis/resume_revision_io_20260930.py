"""Diagnosed I/O recovery with unchanged frozen experiments and log-only monitoring."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import time
import traceback

import psutil

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path('D:/pa-csi-dg-revision-runtime')
OLD_COORD = RUNTIME / 'coordinators/full_revision_20260929'
NEW_COORD = RUNTIME / 'coordinators/full_revision_20260930_recovery_v1'
JOB = 'E2_G_NATIVE_MERGE_502'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def now():
    return datetime.now(timezone.utc).isoformat()


def load_base(directory):
    spec = importlib.util.spec_from_file_location('original_completion_verifier', directory / 'coordinator_base.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def alive(identity):
    try:
        return abs(psutil.Process(identity['pid']).create_time() - identity['create_time']) < .1
    except psutil.NoSuchProcess:
        return False


def check_frozen_plan(directory):
    plan = read(directory / 'plan.json')
    for name, expected in plan['code_sha256'].items():
        if sha(directory / name) != expected:
            raise RuntimeError('Recovery infrastructure code changed: ' + name)
    return plan


def tail(path, limit=32768):
    # Only call for append-only files. Never poll a live writer's replaced JSON.
    try:
        with path.open('rb') as f:
            size = f.seek(0, 2)
            f.seek(max(0, size - limit))
            return f.read().decode('utf-8', errors='replace').splitlines()
    except FileNotFoundError:
        return []


def observe(plan, protocols):
    rows = []
    for index, item in enumerate(plan['studies']):
        bundle = Path(item['bundle'])
        jobs = []
        for config in protocols[index]['jobs']:
            events = []
            for line in tail(bundle / 'runs' / config['job_id'] / 'events.jsonl'):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # Partial last append or clipped first line: next poll sees it.
            status, epoch = 'pending', 0
            if events:
                status = 'running'
                for event in events:
                    if event['event'] == 'epoch_committed':
                        epoch = max(epoch, event['epoch'])
                    elif event['event'] == 'resumed':
                        epoch = max(epoch, event['completed_epochs'])
                        status = 'running'
                    elif event['event'] == 'completed':
                        status = 'completed'
                        epoch = config['epochs']
                    elif event['event'] == 'failed':
                        status = 'failed'
            jobs.append({'job_id': config['job_id'], 'status': status, 'epoch': epoch, 'total_epochs': config['epochs']})
        rows.append({'bundle': str(bundle), 'total': len(jobs), 'completed': sum(j['status'] == 'completed' for j in jobs),
                     'active': [j for j in jobs if j['status'] not in ('pending', 'completed')],
                     'observation_source': 'append-only events; no live mutable study JSON opened'})
    return rows


def worker(directory, index):
    plan = check_frozen_plan(directory)
    item = plan['studies'][index]
    bundle = Path(item['bundle'])
    base = load_base(directory)
    protocol, _ = base.verify(bundle, item['bundle_sha256'])
    from windows_atomic_retry_20260930 import install_replace_retry
    install_replace_retry([bundle], directory / f'{index}_atomic_retry.jsonl', timeout_seconds=30)
    os.chdir(bundle / 'code')
    sys.path.insert(0, str(bundle / 'code'))
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    sys.argv = [protocol['entry_module'], 'run', '--bundle', str(bundle)]
    runpy.run_module(protocol['entry_module'], run_name='__main__')


def run(directory):
    plan = check_frozen_plan(directory)
    base = load_base(directory)
    protocols = [base.verify(Path(e['bundle']), e['bundle_sha256'])[0] for e in plan['studies']]
    from windows_atomic_retry_20260930 import install_replace_retry
    install_replace_retry([directory], directory / 'coordinator_atomic_retry.jsonl', timeout_seconds=30)
    lock = directory / 'coordinator.lock.json'
    identity = {'pid': os.getpid(), 'create_time': psutil.Process().create_time()}
    if lock.exists():
        if alive(read(lock)):
            raise RuntimeError('Recovery coordinator already active')
        lock.rename(directory / f'stale_lock_{time.time_ns()}.json')
    with lock.open('x', encoding='utf-8') as f:
        json.dump(identity, f)
    try:
        for index, item in enumerate(plan['studies']):
            bundle = Path(item['bundle'])
            # These reads occur only before a child starts or after it exits.
            if (bundle / 'runner.lock.json').exists() and alive(read(bundle / 'runner.lock.json')):
                raise RuntimeError('A study writer is already active; do not start another')
            base.verify(bundle, item['bundle_sha256'])
            if (bundle / 'queue_failure.json').exists():
                raise RuntimeError('Undiagnosed study failure remains')
            if base.completed(bundle):
                continue
            stdout, stderr = directory / f'{index}.stdout.log', directory / f'{index}.stderr.log'
            command = [plan['python'], '-u', str(directory / Path(__file__).name), 'worker', '--directory', str(directory), '--index', str(index)]
            env = os.environ.copy()
            env.pop('PYTHONPATH', None)
            env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            with stdout.open('ab') as out, stderr.open('ab') as err:
                child = subprocess.Popen(command, cwd=directory, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                         creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            write(directory / f'{index}_child.json', {'pid': child.pid, 'create_time': psutil.Process(child.pid).create_time(),
                  'command': command, 'started_at': now(), 'stdout': str(stdout), 'stderr': str(stderr)})
            while child.poll() is None:
                write(directory / 'status.json', {'status': 'running', 'updated_at': now(), 'study_index': index,
                      'pid': os.getpid(), 'studies': observe(plan, protocols)})
                time.sleep(30)
            if child.returncode != 0 or not base.completed(bundle):
                raise RuntimeError(f'Study stopped before verified completion (exit {child.returncode}): {bundle}')
        write(directory / 'status.json', {'status': 'completed', 'updated_at': now(), 'studies': observe(plan, protocols)})
    except BaseException:
        write(directory / 'failure.json', {'failed_at': now(), 'error': traceback.format_exc()})
        raise
    finally:
        if lock.exists() and read(lock) == identity:
            lock.unlink()


def prepare_and_start(audit_path):
    if NEW_COORD.exists():
        raise FileExistsError('Preserve previous recovery attempt')
    audit = read(audit_path)
    if not audit.get('passed'):
        raise RuntimeError('Independent checkpoint audit did not pass')
    old_plan = read(OLD_COORD / 'plan.json')
    for directory in (OLD_COORD, *(Path(e['bundle']) for e in old_plan['studies'])):
        for name in ('launcher.json', 'runner.lock.json', 'coordinator.lock.json'):
            if (directory / name).exists() and alive(read(directory / name)):
                raise RuntimeError('An existing coordinator/runner is still alive')
    bundle = Path(old_plan['studies'][0]['bundle'])
    run_dir = bundle / 'runs' / JOB
    failure, progress = read(run_dir / 'status.json'), read(run_dir / 'progress.json')
    if Path(audit['bundle']).resolve() != bundle.resolve() or audit['failed_job_id'] != JOB or audit['bundle_sha256'] != sha(bundle / 'bundle_manifest.json'):
        raise RuntimeError('Checkpoint audit belongs to a different frozen study/job')
    if audit['maximum_verified_recoverable_epoch'] != 136 or audit['completed_optimizer_updates'] != 40800 or audit['checkpoint_sha256'] != progress['sha256']:
        raise RuntimeError('Independent audit does not authorize this checkpoint')
    if sha(run_dir / 'status.json') != audit['failure_status_sha256'] or sha(bundle / 'queue_failure.json') != audit['queue_failure_sha256']:
        raise RuntimeError('Failed state changed after its independent audit')
    for name, expected in audit['hashes'].items():
        current = (run_dir / name).resolve()
        if not current.is_relative_to(run_dir.resolve()) or sha(current) != expected:
            raise RuntimeError('Audited runtime/configuration/history artifact changed: ' + name)
    for record in audit['verified_completed_runs']:
        if sha(bundle / 'runs' / record['job_id'] / 'status.json') != record['status_sha256']:
            raise RuntimeError('A completed source run changed after audit')
    message = failure.get('error', '')
    if failure['status'] != 'failed' or 'PermissionError: [WinError 5]' not in message or 'status.json.tmp' not in message:
        raise RuntimeError('Recovery only authorizes the diagnosed status-file replacement failure')
    if progress['completed_epochs'] != 136 or failure['optimizer_updates_attempted'] != 40800:
        raise RuntimeError('The diagnosed committed epoch changed')
    if sha(run_dir / progress['checkpoint']) != progress['sha256']:
        raise RuntimeError('Committed checkpoint changed')
    retry_qa = read(RUNTIME / 'qa/windows_atomic_retry_20260930.json')
    if not retry_qa['passed'] or retry_qa['wrapper_sha256'] != sha(ROOT / 'analysis/windows_atomic_retry_20260930.py'):
        raise RuntimeError('Atomic retry QA did not pass')
    recovery_qa = read(RUNTIME / 'qa/revision_recovery_20260930.json')
    if not recovery_qa['passed'] or recovery_qa['recovery_module_sha256'] != sha(__file__):
        raise RuntimeError('Recovery coordinator QA is missing or stale')
    NEW_COORD.mkdir(parents=True)
    for source, name in ((Path(__file__), Path(__file__).name),
                         (ROOT / 'analysis/windows_atomic_retry_20260930.py', 'windows_atomic_retry_20260930.py'),
                         (OLD_COORD / 'coordinate_revision_experiments_20260929.py', 'coordinator_base.py')):
        shutil.copy2(source, NEW_COORD / name)
    base = load_base(NEW_COORD)
    for item in old_plan['studies']:
        base.verify(Path(item['bundle']), item['bundle_sha256'])
    dossier = bundle / 'recovery/windows_status_io_20260930_01'
    dossier.mkdir(parents=True, exist_ok=False)
    files = [(run_dir / n, 'run_' + n) for n in ('status.json', 'config.json', 'runtime.json', 'progress.json', 'history.json', 'events.jsonl', progress['checkpoint'])]
    files += [(bundle / n, n) for n in ('queue_failure.json', 'queue_status.json', 'launcher.json')]
    files += [(OLD_COORD / n, 'old_coordinator_' + n) for n in ('failure.json', 'status.json', 'stdout.log', 'stderr.log')]
    launcher = read(bundle / 'launcher.json')
    files += [(Path(launcher[k]), 'old_training_' + k + '.log') for k in ('stdout', 'stderr')]
    files += [(audit_path, 'checkpoint_audit.json')]
    files += [(RUNTIME / 'qa/windows_atomic_retry_20260930.json', 'atomic_retry_qa.json'),
              (RUNTIME / 'qa/revision_recovery_20260930.json', 'recovery_coordinator_qa.json')]
    for source, name in files:
        shutil.copy2(source, dossier / name)
        if sha(source) != sha(dossier / name):
            raise RuntimeError('Failure evidence copy differs')
    decision = {'diagnosed_at': now(), 'job_id': JOB, 'committed_epoch': 136, 'optimizer_updates': 40800,
                'checkpoint_sha256': progress['sha256'], 'independent_audit_sha256': sha(audit_path),
                'cause': 'Windows denied atomic status-file replacement after a committed epoch',
                'file_holder_identified': False,
                'possible_contributor': 'Original coordinator repeatedly opened the active status file; exact holder unproven',
                'mitigation': 'Bounded retry for atomic replace plus append-only monitoring; no scientific code/config change',
                'archive_sha256': {p.name: sha(p) for p in dossier.iterdir() if p.is_file()}}
    write(dossier / 'decision.json', decision)
    # Preserve failed originals under the dossier; no fictitious paused status.
    for source, name in ((run_dir / 'status.json', 'original_failed_run_status.json'),
                         (bundle / 'queue_failure.json', 'original_queue_failure.json')):
        dest = dossier / name
        if not source.resolve().is_relative_to(bundle.resolve()) or not dest.resolve().is_relative_to(bundle.resolve()):
            raise RuntimeError('Archive path escaped the study')
        source.replace(dest)
    plan = {**old_plan, 'created_at': now(), 'original_coordinator': str(OLD_COORD),
            'original_coordinator_code_sha256': old_plan['coordinator_code_sha256'],
            'coordinator_code_sha256': sha(NEW_COORD / Path(__file__).name),
            'recovery_decision': str(dossier / 'decision.json'), 'recovery_decision_sha256': sha(dossier / 'decision.json'),
            'code_sha256': {p.name: sha(p) for p in NEW_COORD.glob('*.py')},
            'observation': 'Only append-only event/stdout logs during active training',
            'io_change': 'Only os.replace .tmp commits inside current bundle: bounded Windows sharing/access retries'}
    write(NEW_COORD / 'plan.json', plan)
    with (NEW_COORD / 'stdout.log').open('wb') as out, (NEW_COORD / 'stderr.log').open('wb') as err:
        child = subprocess.Popen([plan['python'], '-u', str(NEW_COORD / Path(__file__).name), 'run', '--directory', str(NEW_COORD)],
                                 cwd=NEW_COORD, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    identity = {'pid': child.pid, 'create_time': psutil.Process(child.pid).create_time(), 'started_at': now()}
    write(NEW_COORD / 'launcher.json', identity)
    print(json.dumps({'directory': str(NEW_COORD), **identity}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'run', 'worker', 'status'))
    parser.add_argument('--directory', type=Path, default=NEW_COORD)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--index', type=int)
    args = parser.parse_args()
    if args.action == 'start':
        prepare_and_start(args.audit.resolve())
    elif args.action == 'run':
        run(args.directory.resolve())
    elif args.action == 'worker':
        worker(args.directory.resolve(), args.index)
    else:
        p = check_frozen_plan(args.directory)
        print(json.dumps(observe(p, [read(Path(e['bundle']) / 'protocol.json') for e in p['studies']]), indent=2))
