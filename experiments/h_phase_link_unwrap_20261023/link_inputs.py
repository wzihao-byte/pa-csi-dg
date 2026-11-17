"""Bind the within-link unwrap arrays to the audited native historical-style inputs.

No CSI is reconstructed and no model runs. Amplitude, labels, row identity and
the native grid must be exactly those of the paired N_H11 inputs; only the phase
array differs, and by whole turns beyond the first receive link.
"""
from __future__ import annotations
import numpy as np
from common import LINK_DATA, LINK_VARIANT, TARGETS, now, read, require, sha
from phase_inputs import HISTSTYLE, PINS, check_manifest_pair

def array_record(path, expected):
    require(sha(path) == expected, 'Link-unwrap array digest differs from its manifest')
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    try:
        require(values.shape == (3000, 850, 90) and values.dtype == np.float32, 'Link-unwrap array shape/dtype differs')
        for start in range(0, len(values), 32):
            require(np.isfinite(values[start:start + 32]).all(), 'Nonfinite link-unwrap phase')
    finally:
        values._mmap.close()
    return {'path': str(path), 'sha256': expected, 'shape': [3000, 850, 90], 'dtype': 'float32', 'all_finite': True}

def audit_link_alignment(paired_alignment):
    require(paired_alignment['status'] == 'passed' and paired_alignment['rows'] == 9000, 'Paired phase audit incomplete')
    hist = paired_alignment['variants'][HISTSTYLE]
    manifest = read(LINK_DATA / 'dataset_manifest.json')
    require(manifest['complete'] and manifest['rows'] == 9000 and manifest['variants'] == [LINK_VARIANT], 'Link-unwrap dataset incomplete')
    for relative, digest in manifest['artifacts_sha256'].items():
        require(sha(LINK_DATA / relative) == digest, 'Link-unwrap artifact changed: ' + relative)
    compile_config = read(LINK_DATA / 'compile_config.json')
    require(compile_config['cross_link_reference_manifest_sha256'] == PINS['native_histstyle_phase_v1/dataset_manifest.json'],
            'Link-unwrap arrays were not checked against the paired historical-style arrays')
    require(compile_config['native_dataset_manifest_sha256'] == PINS['native_amp_arrays_v1/dataset_manifest.json'],
            'Link-unwrap native parent differs')
    qa = read(LINK_DATA / 'identity_and_grid_qa.json')
    relation = qa['cross_link_reference_relation']
    require(qa['passed'] and qa['rows'] == 9000 and relation['all_9000_rows_verified'], 'Link-unwrap identity/grid QA incomplete')
    file_records, environments = {}, {}
    for env in TARGETS:
        old = read(hist['environments'][env]['target_manifest_path'])
        new = read(LINK_DATA / LINK_VARIANT / 'manifests' / f'deferred_target_{env}.json')
        check_manifest_pair(old, new, env)
        entry = new['files'][env]
        phase = array_record(entry['data_files']['phase']['path'], entry['actual_file_sha256']['phase'])
        paired = hist['file_records'][env]
        for kind in ('amp', 'label'):
            require(entry['data_files'][kind]['path'] == paired[kind]['path'] and entry['actual_file_sha256'][kind] == paired[kind]['sha256'],
                    'Link-unwrap manifest does not share the paired ' + kind + ' file')
        require(phase['sha256'] != paired['phase']['sha256'], 'Link-unwrap phase is the paired cross-link array')
        file_records[env] = {'amp': paired['amp'], 'phase': phase, 'label': paired['label']}
        environments[env] = {'rows': 3000, 'ordered_identity_and_labels_equal_to_paired': True,
                             'amp_and_label_files_equal_to_paired': True,
                             'cross_link_relation': relation['environments'][env]}
    return {'schema': 'h-phase-link-unwrap-input-alignment/v1', 'status': 'passed', 'created_at_utc': now(), 'rows': 9000,
            'variant': LINK_VARIANT, 'paired_variant': HISTSTYLE,
            'dataset_manifest_sha256': sha(LINK_DATA / 'dataset_manifest.json'),
            'identity_and_grid_qa_sha256': sha(LINK_DATA / 'identity_and_grid_qa.json'),
            'file_records': file_records, 'environments': environments, 'cross_link_relation': relation,
            'loader_unwrap_phase': False, 'only_phase_arrays_differ_from_paired': True,
            'physical_phase_calibration_claimed': False, 'model_training_or_target_inference_performed': False}
