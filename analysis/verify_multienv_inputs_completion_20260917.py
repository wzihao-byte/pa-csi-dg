"""Seal input-study completion after its source and target writers exit."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.compile_formal_revision_evidence import verify_study
from experiments.revision_validation.extended_engine import verify_extension
from experiments.revision_validation.study_io import read, sha256, now


def main():
    runtime = Path('D:/pa-csi-dg-revision-runtime')
    bundle = runtime / 'studies/r_inputs_v1'
    protocol, digest, statuses, gate, summary, evaluation_hash = verify_study(bundle, verify_extension, 63)
    coordinator = runtime / 'coordinators/controls_to_inputs_v4/status.json'
    status = read(coordinator)
    assert status['status'] == 'completed' and status['input_bundle_sha256'] == digest
    assert status['input_evaluation_summary_sha256'] == evaluation_hash
    source = runtime / 'coordinators/formal_source_analysis_v3/completion.json'
    assert sha256(source) == '481767f5c2b05137c83e1ee61ec0528af2b6ef81bbb8bbaa5b13aa5fc54a383b'
    record = {'schema': 'multienv-inputs-completion-verified/v1', 'verified_at': now(),
              'completed_source_runs': len(statuses), 'completed_target_results': len(summary['runs']),
              'target_artifact_count': len(summary['artifacts_sha256']),
              'bundle_manifest_sha256': digest, 'evaluation_summary_sha256': evaluation_hash,
              'target_gate_sha256': sha256(bundle / 'target_gate.json'),
              'coordinator_status_sha256': sha256(coordinator), 'completed_at': status['completed_at'],
              'source_completion_sha256': sha256(source), 'source_analysis_count': 117,
              'verified_run_statuses': statuses, 'target_artifacts_sha256': summary['artifacts_sha256'],
              'frozen_protocol_job_count': len(protocol['jobs']), 'writing_scope': 'R technical audit evidence only; retain H/Y manuscript scope'}
    path = ROOT / 'docs/revision/evidence/multienv_inputs_completion_verified_20260917.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'path': str(path), 'sha256': sha256(path), 'source': 63, 'target': 63, 'target_artifacts': len(summary['artifacts_sha256'])}))


if __name__ == '__main__':
    main()
