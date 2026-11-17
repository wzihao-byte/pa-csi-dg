"""Stop the frozen v2 queue after the 18 global-backbone jobs (GC10/GX10), per the 2026-11-05 decision.

The runner checks queue/stop_after_current.json only after a job completes, so the
file must appear while job 18 is running. This watcher polls queue/status.json and
writes the stop file once 17 jobs are complete and the 18th is in progress. It
never touches the frozen bundle or any run directory.
"""
from __future__ import annotations
import json, sys, time
from datetime import datetime, timezone
from pathlib import Path

STUDY = Path(r'D:/pa-csi-dg-revision-runtime/studies/h_dg_baselines_20261105_v2')
STATUS = STUDY / 'queue/status.json'
STOP = STUDY / 'queue/stop_after_current.json'
LOG = STUDY / 'launcher/stop_watcher.log'
GLOBAL_ARM_JOBS = 18
POLL_SECONDS = 60

def log(message):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(f'{datetime.now(timezone.utc).isoformat()} {message}\n')

def main():
    log('watcher started')
    while True:
        try:
            status = json.loads(STATUS.read_text(encoding='utf-8'))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(POLL_SECONDS); continue
        state = status.get('status')
        done = int(status.get('completed_jobs', 0))
        if state in ('complete', 'failed', 'stopped_after_completed_job'):
            log(f'queue {state} with {done} completed jobs; watcher exits'); return
        if state == 'running' and done >= GLOBAL_ARM_JOBS - 1:
            if not STOP.exists():
                STOP.write_text(json.dumps({'at': datetime.now(timezone.utc).isoformat(),
                    'reason': 'User decision 2026-11-05: report only the global-backbone baselines (GC10/GX10); '
                              'the TPP-backbone arms C10/X10 stay frozen but unexecuted',
                    'written_when_completed_jobs': done, 'current_job': status.get('current_job')}, indent=2) + '\n', encoding='utf-8')
                log(f'stop file written while completed_jobs={done} current_job={status.get("current_job")}')
            if done >= GLOBAL_ARM_JOBS:
                log('18 global-backbone jobs already complete; runner will stop after the current job'); return
            log('stop file in place; waiting for the runner to honour it'); return
        time.sleep(POLL_SECONDS)

if __name__ == '__main__':
    sys.exit(main())
