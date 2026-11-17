"""Compile completed studies with v2 WDDM provenance; retain the failed v5 chain."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.revision_validation.study_io import now, read, sha256, write
from analysis.continue_multienv_final_evidence_v2 import FINAL_CODE, append_event, live

RUNTIME = Path('D:/pa-csi-dg-revision-runtime')
OUTPUT = RUNTIME / 'coordinators/multienv_final_evidence_v6'
EVIDENCE = ROOT / 'docs/revision/evidence/multienv_revision_v2'


def main():
    if OUTPUT.exists() or EVIDENCE.exists():
        raise FileExistsError('Preserve all prior attempts and compiled evidence')
    failures = {}
    for name in ('final_evidence', 'paper_assets', 'fragment', 'manuscript', 'release'):
        directory = RUNTIME / f'coordinators/multienv_{name}_v5'
        config = read(directory / 'config.json')
        import psutil
        try:
            process = psutil.Process(config['pid'])
            # Identity includes command line because older configs did not save their own birth.
            if str(directory).lower() in ' '.join(process.cmdline()).lower():
                raise RuntimeError(f'Prior owner still active: {directory}')
        except psutil.NoSuchProcess:
            pass
        assert read(directory / 'status.json')['status'] == 'failed'
        failures[str(directory / 'status.json')] = sha256(directory / 'status.json')
    bindings = {}
    for path, expected in (
        (RUNTIME / 'coordinators/controls_to_inputs_v4/status.json', 'd81aa16201ae1ccaf5dff8c9bd1881ba47e9eebaed4ea924f64f8d6665449169'),
        (RUNTIME / 'coordinators/formal_source_analysis_v3/completion.json', '481767f5c2b05137c83e1ee61ec0528af2b6ef81bbb8bbaa5b13aa5fc54a383b'),
        (ROOT / 'outputs_h_grouped_l1248/v2/queue/completed.json', '69d610aa0437536e3c6a8e8a0d63324115b4a3e2488331c7ec6914532193af77'),
    ):
        assert sha256(path) == expected
        bindings[str(path)] = expected
    OUTPUT.mkdir(parents=True)
    code = OUTPUT / 'code'
    files = list(dict.fromkeys(FINAL_CODE + ['analysis/compile_multienv_revision_evidence_v3.py']))
    for name in files:
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    config = {'pid': os.getpid(), 'process_create_time': psutil.Process().create_time(),
              'started_at': now(), 'completed_upstream_bindings': bindings,
              'retained_v5_failures': failures, 'training_restarted': False,
              'identity_holders_used': False,
              'code_sha256': {name: sha256(code / name) for name in files},
              'recovery_reason': 'Compiler used v1 recovery paths from a frozen tree lacking operational scripts; v3 binds the actual v2 WDDM record and preserves scientific checks.'}
    write(OUTPUT / 'config.json', config)
    events = OUTPUT / 'events.jsonl'
    try:
        command = [sys.executable, '-B', '-X', 'utf8', str(code / 'analysis/compile_multienv_revision_evidence_v3.py'),
                   '--runtime', str(RUNTIME), '--h-root', str(ROOT / 'outputs_h_grouped_l1248/v2'), '--output', str(EVIDENCE)]
        write(OUTPUT / 'status.json', {'status': 'compiling_current_scope_evidence', **config})
        append_event(events, {'event': 'compiling_completed_studies', 'command': command})
        environment = os.environ.copy()
        environment['PA_CSI_DG_ROOT'] = str(ROOT)
        with (OUTPUT / 'compile.stdout.log').open('x') as out, (OUTPUT / 'compile.stderr.log').open('x') as err:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=out, stderr=err, check=True,
                           creationflags=subprocess.CREATE_NO_WINDOW)
        report = read(EVIDENCE / 'verified_multienv_results.json')
        assert (report['multienv_formal_run_count'], report['formal_source_analysis_count'], report['h_supplement_run_count']) == (180, 117, 18)
        result = {'status': 'completed', 'completed_at': now(), 'evidence_output': str(EVIDENCE),
                  'verified_multienv_results_sha256': sha256(EVIDENCE / 'verified_multienv_results.json'),
                  'manifest_sha256': sha256(EVIDENCE / 'manifest.json'), 'multienv_formal_runs': 180,
                  'formal_source_analyses': 117, 'h_supplement_runs': 18, 'widar_training_cancelled_by_user': True}
        write(OUTPUT / 'status.json', result)
        append_event(events, {'event': 'completed', **result})
        print(result)
    except BaseException:
        result = {'status': 'failed', 'failed_at': now(), 'error': traceback.format_exc()}
        write(OUTPUT / 'status.json', result)
        append_event(events, {'event': 'failed', **result})
        raise


if __name__ == '__main__':
    main()
