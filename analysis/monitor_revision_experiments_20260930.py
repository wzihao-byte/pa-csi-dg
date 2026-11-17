"""Read-only health snapshot for the frozen 45-run revision queue.

This command never starts, stops, retries, repairs, or changes an experiment.
During live training it reads only immutable metadata, process identities, and
bounded tails/stat information from append-only logs. Mutable study JSON is
read by the frozen completion verifier only after every writer is confirmed
dead. A suspected stall is a diagnosis request, never permission to kill work.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import psutil

RUNTIME = Path('D:/pa-csi-dg-revision-runtime')
DEFAULT = RUNTIME / 'coordinators/full_revision_20260930_recovery_v1'
EXPECTED_STUDIES = {
    'r_time_pooling_v1_20260929': ('4fb22b1a2afb090e5dd8c64df2b6ca5748ebb1ee301368b818fecc841fab9854', 18),
    'r_phase_controls_v1_20260929': ('c98c640ac361a7888bb9a8c3a636781e8c7a9d6b117c34afeb5091a45a9a277f', 27),
}


def iso(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate(directory):
    directory = Path(directory).resolve()
    if not directory.is_relative_to((RUNTIME / 'coordinators').resolve()):
        raise ValueError('Coordinator directory must be inside the revision runtime coordinators directory')
    plan = read_json(directory / 'plan.json')
    if plan.get('new_training_runs') != 45 or len(plan.get('studies', [])) != 2:
        raise ValueError('Expected the frozen 45-run, two-study plan')
    if 'coordinator_base.py' not in plan.get('code_sha256', {}):
        raise ValueError('A recovery coordinator with a frozen completion verifier is required')
    for name, expected in plan['code_sha256'].items():
        path = (directory / name).resolve()
        if path.parent != directory or digest(path) != expected:
            raise ValueError('Frozen coordinator code changed: ' + name)
    protocols, seen = [], set()
    for item in plan['studies']:
        bundle = Path(item['bundle']).resolve()
        expected, count = EXPECTED_STUDIES.get(bundle.name, (None, None))
        if bundle.parent != (RUNTIME / 'studies').resolve() or bundle.name in seen:
            raise ValueError('Unexpected or duplicate study directory')
        if item.get('bundle_sha256') != expected or item.get('job_count') != count:
            raise ValueError('Plan does not match the original scientific study freeze')
        if digest(bundle / 'bundle_manifest.json') != expected:
            raise ValueError('Frozen study manifest changed')
        manifest = read_json(bundle / 'bundle_manifest.json')
        if digest(bundle / 'protocol.json') != manifest['files_sha256']['protocol.json']:
            raise ValueError('Frozen study protocol changed')
        protocol = read_json(bundle / 'protocol.json')
        jobs = protocol['jobs']
        if len(jobs) != count or len({j['job_id'] for j in jobs}) != count:
            raise ValueError('Frozen job coverage differs')
        for job in jobs:
            if Path(job['job_id']).name != job['job_id'] or '/' in job['job_id'] or '\\' in job['job_id']:
                raise ValueError('Invalid job identifier')
        protocols.append(protocol)
        seen.add(bundle.name)
    return plan, protocols


def identity_state(path):
    """Identity files are written once per process; PID reuse is not liveness."""
    result = {'path': str(path)}
    try:
        identity = read_json(path)
        pid, created = int(identity['pid']), float(identity['create_time'])
        result.update(pid=pid, create_time=created)
        process = psutil.Process(pid)
        current_created = process.create_time()
        if abs(current_created - created) >= .1:
            return {**result, 'state': 'pid_reused', 'current_create_time': current_created}
        if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
            return {**result, 'state': 'dead'}
        return {**result, 'state': 'alive'}
    except FileNotFoundError:
        return {**result, 'state': 'absent'}
    except (psutil.NoSuchProcess, psutil.ZombieProcess):
        return {**result, 'state': 'dead'}
    except (psutil.AccessDenied, OSError, ValueError, KeyError, TypeError) as error:
        return {**result, 'state': 'unknown', 'error': str(error)}


def identities(directory, plan):
    coordinator = [identity_state(directory / n) for n in ('launcher.json', 'coordinator.lock.json')]
    workers = []
    for index, item in enumerate(plan['studies']):
        workers.append({'study_index': index, 'identities': [
            identity_state(directory / f'{index}_child.json'),
            identity_state(Path(item['bundle']) / 'runner.lock.json')]})
    return {'coordinator': coordinator, 'workers': workers}


def scan_unregistered_writers(directory, plan):
    """Catch a child in the narrow interval before its identity file is written."""
    markers = {str(directory.resolve()).replace('\\', '/').casefold()}
    markers.update(str(Path(e['bundle']).resolve()).replace('\\', '/').casefold() for e in plan['studies'])
    found, inaccessible_python = [], []
    for process in psutil.process_iter(['pid', 'create_time', 'name', 'cmdline']):
        if process.pid == psutil.Process().pid:
            continue
        info = process.info
        name = (info.get('name') or '').casefold()
        if 'python' not in name:
            continue
        command = info.get('cmdline')
        if command is None:
            inaccessible_python.append(info['pid'])
            continue
        args = [str(a).replace('\\', '/').casefold() for a in command]
        if any(a in markers for a in args):
            # An independent read-only monitor is not an experiment writer.
            if any(Path(a).name == Path(__file__).name for a in args):
                continue
            found.append({'pid': info['pid'], 'create_time': info.get('create_time'), 'command': command})
    return {'matching_processes': found, 'inaccessible_python_pids': inaccessible_python}


def log_tail(path, limit=32768):
    """Read only complete UTF-8 lines; ignore a clipped first or partial last line."""
    result = {'path': str(path), 'exists': False, 'bytes': 0, 'modified_at': None, 'lines': []}
    try:
        with Path(path).open('rb') as stream:
            size = stream.seek(0, 2)
            start = max(0, size - limit)
            stream.seek(start)
            block = stream.read(limit)
        stat = Path(path).stat()
        pieces = block.split(b'\n')
        if start:
            pieces = pieces[1:]
        pieces = pieces[:-1]  # The last entry is empty or an uncommitted append.
        return {**result, 'exists': True, 'bytes': size, 'modified_at': iso(stat.st_mtime),
                'modified_timestamp': stat.st_mtime,
                'lines': [p.decode('utf-8', errors='replace').rstrip('\r') for p in pieces]}
    except FileNotFoundError:
        return result
    except OSError as error:
        return {**result, 'error': str(error)}


def events_for(path, config):
    log = log_tail(path)
    state, epoch, last_event = 'pending', 0, None
    last_failure, resume_audit = None, None
    for line in log['lines']:
        try:
            event = json.loads(line)
            name = event['event']
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        last_event = event
        if name == 'resumed':
            state, epoch = 'running', max(epoch, int(event['completed_epochs']))
        elif name == 'epoch_committed':
            state, epoch = 'running', max(epoch, int(event['epoch']))
        elif name == 'completed':
            state, epoch = 'source_completed', config['epochs']
        elif name == 'failed':
            state, last_failure = 'failed', event
        elif name == 'resume_state_audit':
            resume_audit = event
        elif state == 'pending':
            state = 'running'
    result = {'job_id': config['job_id'], 'event_state': state, 'epoch': epoch,
              'total_epochs': config['epochs'], 'last_event': last_event,
              'log': {k: v for k, v in log.items() if k != 'lines'}}
    if last_failure:
        result['last_failure'] = last_failure
    if resume_audit:
        result['resume_audit'] = resume_audit
    return result


def load_completion_verifier(directory):
    # No __pycache__ writes: this monitor's only output is stdout.
    previous = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        spec = importlib.util.spec_from_file_location('revision_health_frozen_verifier', directory / 'coordinator_base.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.completed
    finally:
        sys.dont_write_bytecode = previous


def all_dead(processes, scan):
    values = processes['coordinator'] + [i for w in processes['workers'] for i in w['identities']]
    return (all(i['state'] in ('dead', 'absent', 'pid_reused') for i in values)
            and not scan['matching_processes'] and not scan['inaccessible_python_pids'])


def snapshot(directory=DEFAULT, stall_seconds=1200, now=None):
    directory = Path(directory).resolve()
    now = time.time() if now is None else now
    plan, protocols = validate(directory)
    processes = identities(directory, plan)
    scan = scan_unregistered_writers(directory, plan)
    coordinator_alive = any(i['state'] == 'alive' for i in processes['coordinator'])
    live_workers = {w['study_index'] for w in processes['workers'] if any(i['state'] == 'alive' for i in w['identities'])}
    def distinct_alive(rows):
        return {(i.get('pid'), i['create_time']) for i in rows if i['state'] == 'alive'}
    duplicate_writers = (len(live_workers) > 1 or len(distinct_alive(processes['coordinator'])) > 1 or any(
        len(distinct_alive(w['identities'])) > 1 for w in processes['workers']))
    unknown = any(i['state'] == 'unknown' for i in processes['coordinator']) or any(
        i['state'] == 'unknown' for w in processes['workers'] for i in w['identities'])
    unknown = unknown or bool(scan['inaccessible_python_pids'])
    studies, logs, last_progress = [], [], None
    for index, item in enumerate(plan['studies']):
        bundle = Path(item['bundle'])
        jobs = [events_for(bundle / 'runs' / c['job_id'] / 'events.jsonl', c) for c in protocols[index]['jobs']]
        done = sum(j['event_state'] == 'source_completed' for j in jobs)
        studies.append({'study_index': index, 'bundle': str(bundle), 'source_completed': done,
                        'total': len(jobs), 'worker_alive': index in live_workers,
                        'phase': 'evaluation_or_final_verification' if done == len(jobs) else 'source_training',
                        'active_or_failed_jobs': [j for j in jobs if j['event_state'] not in ('pending', 'source_completed')]})
        for name in (f'{index}.stdout.log', f'{index}.stderr.log', f'{index}_atomic_retry.jsonl'):
            log = log_tail(directory / name)
            logs.append({**log, 'lines': log['lines'][-8:]})
            # Retry heartbeats are diagnostic, not scientific forward progress.
            if index in live_workers and 'atomic_retry' not in name and log['bytes']:
                last_progress = max(last_progress or 0, log.get('modified_timestamp', 0))
        if index in live_workers:
            for job in jobs:
                if job['log']['bytes']:
                    last_progress = max(last_progress or 0, job['log'].get('modified_timestamp', 0))
    for name in ('stdout.log', 'stderr.log', 'coordinator_atomic_retry.jsonl'):
        log = log_tail(directory / name)
        logs.append({**log, 'lines': log['lines'][-8:]})
    # Startup grace uses real process birth, never a periodically rewritten status.
    born = [i['create_time'] for w in processes['workers'] for i in w['identities'] if i['state'] == 'alive']
    if not live_workers:
        born += [i['create_time'] for i in processes['coordinator'] if i['state'] == 'alive']
    if born:
        last_progress = max(last_progress or 0, max(born))
    age = max(0, now - last_progress) if last_progress is not None else None
    completion = {'verified': False, 'attempted': False, 'studies': []}
    if all_dead(processes, scan):
        # Recheck immediately before and after the only operation allowed to read
        # mutable study JSON. A new writer cancels the verification conclusion.
        if all_dead(identities(directory, plan), scan_unregistered_writers(directory, plan)):
            completion['attempted'] = True
            verifier = load_completion_verifier(directory)
            for item in plan['studies']:
                if not all_dead(identities(directory, plan), scan_unregistered_writers(directory, plan)):
                    completion['cancelled'] = 'A writer appeared before verification'
                    break
                try:
                    verified = bool(verifier(Path(item['bundle'])))
                    completion['studies'].append({'bundle': item['bundle'], 'verified': verified})
                except Exception as error:
                    completion['studies'].append({'bundle': item['bundle'], 'verified': False, 'error': str(error)})
            completion['verified'] = (len(completion['studies']) == len(plan['studies'])
                                      and all(s['verified'] for s in completion['studies'])
                                      and all_dead(identities(directory, plan), scan_unregistered_writers(directory, plan)))
    if completion['verified']:
        state, action = 'completed', 'All frozen source jobs, target evaluations, and comparison artifacts verified; retire monitor.'
    elif duplicate_writers:
        state, action = 'duplicate_writers', 'Distinct live writers violate the sequential one-GPU queue; inspect identities and ownership before any intervention.'
    elif unknown:
        state, action = 'unknown', 'Resolve unreadable process identities before deciding whether the queue stopped; do not start a duplicate writer.'
    elif live_workers and not coordinator_alive:
        state, action = 'coordinator_lost', 'Worker is still alive; diagnose/recover orchestration without interrupting or duplicating it.'
    elif live_workers and age is not None and age >= stall_seconds:
        state, action = 'suspected_stall', 'Inspect process/GPU activity and append-only logs; a stale log alone is not grounds to kill training.'
    elif coordinator_alive and not live_workers:
        state, action = 'transitioning', 'Coordinator is alive without a registered worker; check append-only logs and process identities again shortly.'
        if age is not None and age >= stall_seconds:
            state, action = 'suspected_stall', 'Coordinator is alive without an active worker; diagnose before recovery.'
    elif coordinator_alive and live_workers:
        state, action = 'running', 'Continue monitoring quietly; no repair needed.'
    elif scan['matching_processes']:
        state, action = 'unknown', 'A matching unregistered process exists; inspect its identity before recovery.'
    else:
        state, action = 'stopped', 'Queue is incomplete and no writer is alive; preserve failure evidence, audit the last committed checkpoint, diagnose and repair before resuming.'
    return {'observed_at': iso(now), 'coordinator_directory': str(directory), 'health': state,
            'needs_attention': state not in ('running', 'transitioning', 'completed'),
            'recommended_action': action, 'read_only': True,
            'observation_policy': 'No live mutable study status/progress/queue/failure JSON is opened',
            'stall_threshold_seconds': stall_seconds, 'last_progress_at': iso(last_progress) if last_progress else None,
            'seconds_since_progress': round(age, 1) if age is not None else None,
            'source_completed': sum(s['source_completed'] for s in studies), 'source_total': 45,
            'processes': processes, 'process_scan': scan, 'studies': studies,
            'completion_verification': completion, 'append_only_logs': logs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    parser.add_argument('--stall-minutes', type=float, default=20)
    args = parser.parse_args()
    if args.stall_minutes <= 0:
        parser.error('--stall-minutes must be positive')
    try:
        result = snapshot(args.directory, args.stall_minutes * 60)
    except Exception as error:
        result = {'observed_at': iso(time.time()), 'health': 'monitor_error', 'needs_attention': True,
                  'read_only': True, 'error': str(error),
                  'recommended_action': 'Diagnose the monitor validation error; do not infer completion or start duplicate training.'}
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if result['health'] in ('running', 'transitioning', 'completed') else 2


if __name__ == '__main__':
    raise SystemExit(main())
