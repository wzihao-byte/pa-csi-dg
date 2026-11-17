"""Start or inspect the dated time-preprocessing study without a visible window."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.revision_validation.study_io import read, sha256, write

BUNDLE = Path('D:/pa-csi-dg-revision-runtime/studies/r_time_pooling_v1_20260929')
PYTHON = Path('C:/ProgramData/anaconda3/python.exe')


def live(identity):
    try:
        p = psutil.Process(identity['pid'])
        return abs(p.create_time() - identity['create_time']) < .1
    except psutil.NoSuchProcess:
        return False


def status(bundle):
    protocol = read(bundle / 'protocol.json')
    records = []
    for c in protocol['jobs']:
        path = bundle / 'runs' / c['job_id'] / 'status.json'
        records.append({'job_id': c['job_id'], **(read(path) if path.exists() else {'status': 'pending'})})
    launch = bundle / 'launcher.json'
    identity = read(launch) if launch.exists() else None
    result = {'bundle': str(bundle), 'runner_alive': live(identity) if identity else False,
              'total_jobs': len(records), 'completed_jobs': sum(r['status'] == 'completed' for r in records),
              'pending_jobs': sum(r['status'] == 'pending' for r in records),
              'active_or_failed': [{k: r.get(k) for k in ('job_id', 'status', 'epoch', 'best_epoch', 'best_source_macro_f1', 'updated_at', 'error')} for r in records if r['status'] not in ('pending', 'completed')]}
    for filename in ('queue_status.json', 'queue_failure.json'):
        if (bundle / filename).exists():
            result[filename.removesuffix('.json')] = read(bundle / filename)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def start(bundle):
    protocol = read(bundle / 'protocol.json')
    if protocol['entry_module'] != 'experiments.revision_validation.time_pooling_study' or len(protocol['jobs']) != 18:
        raise RuntimeError('Unexpected frozen study')
    if (bundle / 'queue_failure.json').exists():
        raise RuntimeError('A recorded failure needs diagnosis; refusing automatic retry')
    for name in ('launcher.json', 'runner.lock.json'):
        if (bundle / name).exists() and live(read(bundle / name)):
            raise RuntimeError('The study already has a live runner')
    if (bundle / 'queue_status.json').exists() and read(bundle / 'queue_status.json')['status'] == 'completed':
        raise RuntimeError('The study is already complete')
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    logdir = bundle / 'launcher_logs'
    logdir.mkdir(exist_ok=True)
    stdout, stderr = logdir / f'{timestamp}.stdout.log', logdir / f'{timestamp}.stderr.log'
    command = [str(PYTHON), '-u', '-m', protocol['entry_module'], 'run', '--bundle', str(bundle)]
    env = os.environ.copy()
    env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    env['PYTHONUNBUFFERED'] = '1'
    env.pop('PYTHONPATH', None)
    with stdout.open('wb') as out, stderr.open('wb') as err:
        process = subprocess.Popen(command, cwd=bundle / 'code', stdout=out, stderr=err,
                                   stdin=subprocess.DEVNULL, env=env,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    identity = {'pid': process.pid, 'create_time': psutil.Process(process.pid).create_time(),
                'started_at': timestamp, 'command': command, 'cwd': str(bundle / 'code'),
                'stdout': str(stdout), 'stderr': str(stderr),
                'bundle_sha256': sha256(bundle / 'bundle_manifest.json')}
    write(logdir / f'{timestamp}.launcher.json', identity)
    write(bundle / 'launcher.json', identity)
    print(json.dumps(identity, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('start', 'status'))
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    args = parser.parse_args()
    (start if args.action == 'start' else status)(args.bundle.resolve())
