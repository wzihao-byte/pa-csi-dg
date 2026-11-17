"""Retire only waiting owners and launch a provenance-bound H/WDDM recovery."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.recover_h_grouped_l1248_queue import read, sha256, verify_bundle, write_new, emit
from analysis.h_gpu_availability_v2 import verify_available

RUNTIME = Path('D:/pa-csi-dg-revision-runtime')
HROOT = ROOT / 'outputs_h_grouped_l1248/v2'
RECORD = ROOT / 'docs/revision/evidence/runtime_recovery_20260917_wddm.json'
PRIOR = ROOT / 'docs/revision/evidence/runtime_recovery_20260917.json'


def identity(process):
    return {'pid': process.pid, 'process_create_time': process.create_time(), 'command': process.cmdline()}


def launch(name, command, expected_file):
    stdout = (RUNTIME / f'logs/{name}_v5.stdout.log').open('xb')
    stderr = (RUNTIME / f'logs/{name}_v5.stderr.log').open('xb')
    try:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
    finally:
        stdout.close()
        stderr.close()
    for _ in range(40):
        if child.poll() is not None:
            raise RuntimeError(f'{name} exited {child.returncode}; inspect retained logs')
        if expected_file.is_file():
            return identity(psutil.Process(child.pid))
        time.sleep(.5)
    raise RuntimeError(f'{name} did not commit its owner/config record in time')


def main():
    assert not RECORD.exists()
    prior = read(PRIOR)
    selected = ('h_queue', 'final_evidence', 'paper_assets', 'fragment', 'manuscript', 'release')
    old = {}
    for name in selected:
        expected = prior['current_owners'][name]
        process = psutil.Process(expected['pid'])
        assert abs(process.create_time() - expected['process_create_time']) < .01
        assert process.cmdline() == expected['command']
        for child in process.children(recursive=True):
            try:
                if child.exe().replace('\\', '/').lower() == 'c:/windows/system32/conhost.exe':
                    continue
                assert child.name().lower() == 'nvidia-smi.exe', f'{name} has an active non-probe child'
                child.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass
        old[name] = identity(process)
    for name in ('inputs_coordinator', 'inputs_runner'):
        expected = prior['current_owners'][name]
        assert not psutil.pid_exists(expected['pid']), 'Input owner must have exited'
    for name in ('ready.json', 'completed.json', 'failed.json'):
        assert not (HROOT / 'queue' / name).exists()
    assert not (HROOT / 'factorial_results.json').exists()
    for path in (ROOT / 'docs/revision/evidence/multienv_revision_v2',
                 ROOT / 'paper/sensors_manuscript/revision_assets/multienv_v2',
                 RUNTIME / 'paper_assets/multienv_v2', RUNTIME / 'qa/manuscript_multienv_v5'):
        assert not path.exists(), str(path)
    for name in ('multienv_final_evidence_v5', 'multienv_paper_assets_v5', 'multienv_fragment_v5',
                 'multienv_manuscript_v5', 'multienv_release_v5'):
        assert not (RUNTIME / 'coordinators' / name).exists()
    assert sha256(HROOT / 'queue/owner.json') == prior['h_recovery']['current_owner_sha256']
    manifest, bundle_hash = verify_bundle(HROOT / 'bundle')
    assert bundle_hash == 'b30ac26dc851393ee75f58cd5dae26400930296e211bebd5384fe848b2c0696c'
    assert sha256(HROOT / 'qa/approval.json') == '6b495b82feaffd6bb25c495ebe8120b31dc27a60d917d6dc63cd51afd74cf26d'
    for name, expected in {
        'recover_h_grouped_l1248_queue.py': 'bebf1560b7fb50d66a67c44730b5f7c8ab0760a8b2ff6570876e18d5943c7670',
        'execute_h_grouped_l1248_recovery.py': '592bc6cabfe60fc077d3d4157448efb0289e31569db39b81da36024f2d91da3f',
    }.items():
        assert sha256(ROOT / 'analysis' / name) == expected
    marker = '    base = frozen_run.backend()'
    assert (ROOT / 'analysis/execute_h_grouped_l1248_recovery.py').read_text().split(marker, 1)[1] == \
           (ROOT / 'analysis/execute_h_grouped_l1248_recovery_v2.py').read_text().split(marker, 1)[1]
    gpu_report = verify_available()
    write_new(ROOT / 'docs/revision/evidence/h_gpu_preflight_20260917_v2.json', gpu_report)
    assert gpu_report['available'], 'New GPU gate blocked; preserve owners'
    input_verification = ROOT / 'docs/revision/evidence/multienv_inputs_completion_verified_20260917.json'
    assert sha256(input_verification) == 'e7d5cb71d02f415acacbe957034b11187298d9f26ccad91c97b6000f2471a859'
    dry = subprocess.run([sys.executable, '-B', '-X', 'utf8', str(HROOT / 'bundle/code/run.py'),
                          '--dry-run', '--bundle', str(HROOT / 'bundle')],
                         capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
    plan = json.loads(dry.stdout)
    assert len(plan['plan']) == 18 and all(x['action'] == 'run' for x in plan['plan'])
    retirement = {'schema': 'revision-waiting-owner-retirement/v1',
                  'at': datetime.now(timezone.utc).isoformat(), 'owners': old,
                  'reason': 'WDDM empty-compute-PID gate treats desktop applications as experiment owners',
                  'h_training_started': False, 'input_completion_verified_sha256': sha256(input_verification),
                  'h_bundle_manifest_sha256': bundle_hash, 'frozen_training_modified': False}
    retirement_path = ROOT / 'docs/revision/evidence/wddm_waiting_owners_retired_20260917.json'
    write_new(retirement_path, retirement)
    for name in reversed(selected):
        expected = old[name]
        process = psutil.Process(expected['pid'])
        assert abs(process.create_time() - expected['process_create_time']) < .01
        process.terminate()
        process.wait(timeout=10)
        if name == 'h_queue':
            events = HROOT / 'queue/events.jsonl'
        else:
            output = Path(expected['command'][expected['command'].index('--output') + 1])
            events = output / 'events.jsonl'
            write_new(output / 'retirement.json', {'reason': retirement['reason'],
                       'retirement_record_sha256': sha256(retirement_path), 'pid': expected['pid']})
        emit(events, 'retired_before_wddm_gate_recovery', pid=expected['pid'], h_training_started=False,
             retirement_record_sha256=sha256(retirement_path))
    archived_owner = HROOT / 'queue/owner.retired_wddm_20260917.json'
    assert not archived_owner.exists() and archived_owner.resolve().parent == (HROOT / 'queue').resolve()
    (HROOT / 'queue/owner.json').rename(archived_owner)
    python = sys.executable
    owners = {}
    command = [python, '-B', '-X', 'utf8', 'analysis/recover_h_grouped_l1248_queue_v2.py',
               '--inputs-output', str(RUNTIME / 'coordinators/controls_to_inputs_v4'),
               '--input-bundle', str(RUNTIME / 'studies/r_inputs_v1'), '--h-root', str(HROOT),
               '--superseded-owner', str(archived_owner)]
    owners['h_queue'] = launch('h_recovery', command, HROOT / 'queue/owner.json')
    command = [python, '-B', '-X', 'utf8', 'analysis/continue_multienv_final_evidence_v2.py',
               '--inputs-output', str(RUNTIME / 'coordinators/controls_to_inputs_v4'),
               '--source-output', str(RUNTIME / 'coordinators/formal_source_analysis_v3'),
               '--h-pid', str(owners['h_queue']['pid']), '--h-root', str(HROOT), '--runtime', str(RUNTIME),
               '--evidence-output', str(ROOT / 'docs/revision/evidence/multienv_revision_v2'),
               '--output', str(RUNTIME / 'coordinators/multienv_final_evidence_v5'),
               '--compatibility-marker', 'analysis/continue_multienv_final_evidence.py']
    owners['final_evidence'] = launch('final_evidence', command, RUNTIME / 'coordinators/multienv_final_evidence_v5/config.json')
    for name, upstream, flag in (
        ('paper_assets', 'final_evidence', '--final-evidence-pid'),
        ('fragment', 'paper_assets', '--assets-pid'),
        ('manuscript', 'fragment', '--fragment-pid'),
        ('release', 'manuscript', '--manuscript-pid'),
    ):
        command = [part.replace('_v4', '_v5') for part in old[name]['command']]
        command[command.index(flag) + 1] = str(owners[upstream]['pid'])
        output = Path(command[command.index('--output') + 1])
        owners[name] = launch(name, command, output / 'config.json')
    code_names = ['h_gpu_availability_v2.py', 'recover_h_grouped_l1248_queue_v2.py',
                  'execute_h_grouped_l1248_recovery_v2.py', 'continue_multienv_final_evidence_v2.py',
                  'launch_h_wddm_recovery_v5.py']
    record = {'schema': 'csi-wddm-operational-recovery/v1', 'recorded_at': datetime.now(timezone.utc).isoformat(),
              'retirement_record_sha256': sha256(retirement_path), 'previous_recovery_sha256': sha256(PRIOR),
              'input_completion_verification_sha256': sha256(input_verification),
              'current_owners': owners, 'archived_owner_sha256': sha256(archived_owner),
              'current_h_owner_sha256': sha256(HROOT / 'queue/owner.json'),
              'frozen_h_bundle_manifest_sha256': bundle_hash, 'frozen_h_bundle_modified': False,
              'all_18_jobs_untouched_at_recovery': True, 'scientific_protocol_changes': [],
              'source_analysis_completion_sha256': '481767f5c2b05137c83e1ee61ec0528af2b6ef81bbb8bbaa5b13aa5fc54a383b',
              'code_sha256': {name: sha256(ROOT / 'analysis' / name) for name in code_names},
              'completed_upstreams_restarted': False, 'identity_holders_used': False,
              'writing_scope': 'Original H/Y and verified grouped-H supplement; R outputs retained as technical audit evidence',
              'gpu_gate_policy': gpu_report['policy']}
    write_new(RECORD, record)
    print(json.dumps({'path': str(RECORD), 'sha256': sha256(RECORD), 'owners': owners}, indent=2))


if __name__ == '__main__':
    main()
