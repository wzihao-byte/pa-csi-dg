"""Compile the within-receive-link unwrap control of the historical-style phase.

Read-only inputs; newly versioned outputs only. Identical to the frozen
native_time_wrapped_linear_feature_unwrap_v1 chain except that the feature-axis
unwrap runs inside each 30-column receive-link group instead of across all 90
columns. It tests whether crossing the 29->30 and 59->60 boundaries matters and
is NOT a physical phase calibration.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_name, '1')

import argparse
import concurrent.futures
import copy
import hashlib
import json
import platform
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.audit_multienv_timestamps import decode_dat_csi
from analysis.prepare_native_multienv import native_elapsed, uniform_interpolation
from analysis.verify_multienv_raw_mapping import read_dat_timestamps
from experiments.revision_validation.study_io import now, read, sha256, value_hash, write

VARIANT = 'native_time_wrapped_linear_link_unwrap_v1'
REFERENCE_VARIANT = 'native_time_wrapped_linear_feature_unwrap_v1'
DEFAULT_OUTPUT = Path('D:/pa-csi-dg-revision-runtime/data/native_linkunwrap_phase_v1')
NATIVE = Path('D:/pa-csi-dg-revision-runtime/data/native_amp_arrays_v1')
PREPARED = Path('D:/pa-csi-dg-revision-runtime/data/native_amp_v1')
REFERENCE = Path('D:/pa-csi-dg-revision-runtime/data/native_histstyle_phase_v1')
REFERENCE_MANIFEST_SHA256 = '812e3394f06f21abf04d3ef2366c92ab2eeca7afb2478419da60671820da6c67'
LINKS = ((0, 30), (30, 60), (60, 90))
CODE_FILES = ('analysis/prepare_native_linkunwrap_phase_20261023.py',
              'analysis/prepare_native_multienv.py', 'analysis/audit_multienv_timestamps.py',
              'analysis/verify_multienv_raw_mapping.py', 'experiments/revision_validation/study_io.py')
DEFINITION = {
    'purpose': 'Within-receive-link unwrap control of the historical-style phase on the native-time linear amplitude grid; not physical calibration or repaired phase',
    'input': 'Permutation-corrected native DAT complex128 CSI [packet, 90], 3 Rx groups x 30 columns',
    'timestamp': 'Native uint32 microseconds, modulo-2^32 forward deltas, reject deltas >=2^31, start at zero',
    'phase_extraction': 'np.angle(csi) float64, before interpolation; angle(0+0j)=0 retained as the historical arithmetic convention',
    'duplicates': 'Arithmetic mean of wrapped angle values at identical elapsed times, independently per column; not a circular mean',
    'grid': 'np.linspace(0.0, native_elapsed[-1], 850), float64, endpoints included; identical to frozen native_time_linear amplitude',
    'resampling': 'scipy.interpolate.interp1d(kind=linear, axis=0, bounds_error=True) on wrapped angle scalars using the frozen uniform_interpolation helper',
    'large_gaps': 'Linearly bridge all packet gaps without a mask or cutoff; no row exclusion',
    'feature_unwrap': 'After interpolation apply np.unwrap(..., axis=-1) in float64 with defaults (period 2*pi, discont pi) separately to columns 0-29, 30-59 and 60-89; the 29->30 and 59->60 receive-link boundaries are NOT crossed',
    'only_difference_from_reference': 'native_time_wrapped_linear_feature_unwrap_v1 unwraps all 90 columns as one sequence; everything else is identical',
    'adjacent_frames': 'No temporal unwrap, temporal centering, phase differencing, filtering, or train-fitted normalization',
    'cast': 'Cast only the final [850,90] phase matrix to float32; loader must not unwrap again',
    'amplitude': 'Reference existing native_time_linear amplitude files and labels without copying or modifying them',
    'model_view_slicing': 'The model pseudo-Tx convention (mode=tx,num_rx=1,num_tx=3,num_subcarriers=30) slices the same three 30-column software feature groups; it does not assert three physical transmitters',
}


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def matrix_sha(value):
    return hashlib.sha256(np.asarray(value, dtype='<f4').tobytes(order='C')).hexdigest()


def link_unwrap(sampled):
    result = np.empty_like(sampled)
    for start, stop in LINKS:
        result[:, start:stop] = np.unwrap(sampled[:, start:stop], axis=-1)
    return result


def reconstruct(csi, elapsed):
    require(csi.ndim == 2 and csi.shape[1] == 90 and np.isfinite(csi).all(), 'Invalid complex CSI')
    phase = np.angle(np.asarray(csi, dtype=np.complex128))
    sampled = uniform_interpolation(phase, elapsed)
    result = link_unwrap(sampled).astype(np.float32)
    require(result.shape == (850, 90) and np.isfinite(result).all(), 'Invalid reconstructed phase')
    return result


def independent_reconstruct(csi, elapsed):
    """Independent per-column interpolation and explicit per-link unwrap formula."""
    phase = np.angle(np.asarray(csi, dtype=np.complex128))
    times = np.unique(elapsed)
    averaged = np.stack([phase[elapsed == timestamp].mean(axis=0) for timestamp in times])
    grid = np.linspace(0.0, elapsed[-1], 850)
    sampled = np.stack([np.interp(grid, times, averaged[:, col]) for col in range(90)], axis=1)
    for start, stop in LINKS:
        block = sampled[:, start:stop]
        delta = np.diff(block, axis=1)
        wrapped = (delta + np.pi) % (2 * np.pi) - np.pi
        wrapped[(wrapped == -np.pi) & (delta > 0)] = np.pi
        correction = wrapped - delta
        correction[np.abs(delta) < np.pi] = 0
        block[:, 1:] += np.cumsum(correction, axis=1)
    return sampled.astype(np.float32)


def compare_with_reference(link, reference):
    """Per-row relation to the cross-link array: link 0 equal, links 1-2 differ by 2*pi*k."""
    require(link.shape == reference.shape == (850, 90), 'Reference comparison shape differs')
    require(np.array_equal(link[:, :30], reference[:, :30]), 'First receive link differs from cross-link reference')
    shifted, max_residual = [], 0.0
    for start, stop in LINKS[1:]:
        difference = reference[:, start:stop].astype(np.float64) - link[:, start:stop].astype(np.float64)
        turns = np.round(difference / (2 * np.pi))
        residual = float(np.max(np.abs(difference - 2 * np.pi * turns)))
        require(residual < 1e-3, 'Cross-link difference is not an integer number of turns')
        require(np.all(turns == turns[:, :1]), 'Turn offset varies inside one receive link')
        max_residual = max(max_residual, residual)
        shifted.append(int(np.count_nonzero(turns[:, 0])))
    return {'link_1_frames_shifted': shifted[0], 'link_2_frames_shifted': shifted[1],
            'max_turn_residual_rad': max_residual}


def synthetic_qa():
    times = np.array([0., 1e6])
    angles = np.stack([np.full(90, 3.1), np.full(90, -3.1)])
    csi = np.exp(1j * angles)
    values = reconstruct(csi, times)
    expected = np.linspace(3.1, -3.1, 850).astype(np.float32)
    error = float(np.max(np.abs(values - expected[:, None])))
    require(error < 1e-6 and abs(float(values[424, 0])) < .01, 'Wrapped +/-pi diagnostic changed')
    duplicate_times = np.array([0., 0., 1e6, 1e6])
    opposing = np.exp(1j * np.tile(np.array([3.1, -3.1, 3.1, -3.1])[:, None], (1, 90)))
    duplicate_result = reconstruct(opposing, duplicate_times)
    require(np.array_equal(duplicate_result, np.zeros_like(duplicate_result)), 'Duplicate angle arithmetic changed')
    zeros = np.zeros((2, 90), dtype=np.complex128)
    require(np.array_equal(reconstruct(zeros, times), np.zeros((850, 90), dtype=np.float32)), 'angle(0) convention changed')
    boundary_angles = np.zeros((2, 90))
    boundary_angles[:, 30:60] = -3.1
    boundary_angles[:, :30] = 3.1
    boundary_angles[:, 60:] = 3.1
    kept = reconstruct(np.exp(1j * boundary_angles), times)
    require(abs(float(kept[0, 30]) + 3.1) < 1e-6 and abs(float(kept[0, 60]) - 3.1) < 1e-6,
            'Receive-link boundary was crossed')
    within = np.zeros((2, 90))
    within[:, 31] = 3.1
    within[:, 32] = -3.1
    unwrapped = reconstruct(np.exp(1j * within), times)
    require(abs(float(unwrapped[0, 32]) - (2*np.pi-3.1)) < 1e-6, 'Within-link unwrap missing')
    duplicated = reconstruct(np.repeat(csi, 2, axis=0), np.repeat(times, 2))
    require(np.array_equal(duplicated, values), 'Identical duplicate-time invariance failed')
    independent = independent_reconstruct(np.exp(1j * boundary_angles), times)
    require(float(np.max(np.abs(independent - kept))) < 1e-6, 'Independent synthetic reconstruction differs')
    return {
        'passed': True, 'model_training_or_scoring': False,
        'wrapped_plus_minus_3_1_linear_ramp_max_error': error,
        'wrapped_plus_minus_3_1_frame_424': float(values[424, 0]),
        'opposing_duplicate_wrapped_angles_arithmetic_mean_zero': True,
        'identical_duplicate_time_invariance_exact': True,
        'zero_amplitude_angle_zero_retained': True,
        'feature_unwrap_crosses_receive_link_boundary': False,
        'feature_unwrap_inside_receive_link': True,
        'frame_0_feature_29': float(kept[0, 29]),
        'frame_0_feature_30': float(kept[0, 30]),
        'frame_0_feature_32_within_link': float(unwrapped[0, 32]),
        'no_physical_calibration_claim': True,
    }


def process_record(task):
    identity, expected_record_sha, dat_inventory, config_sha = task
    stem = Path(identity['raw_path']).stem
    record_path = PREPARED / 'records' / (stem + '.json')
    require(sha256(record_path) == expected_record_sha, 'Native preparation record hash mismatch')
    record = read(record_path)
    require(record['verified'] and record['all_complex_csi_and_packet_ids_equal'], 'Native identity not verified')
    require(record['index'] == identity['index'] and record['dat_path'] == identity['raw_path']
            and record['dat_sha256'] == identity['raw_sha256']
            and record['mat_path'] == identity['identity_source_mat_path']
            and record['mat_sha256'] == identity['identity_source_mat_sha256']
            and record['environment'] + ':' + record['subject_id'] == identity['subject_id']
            and record['acquisition_id'] == identity['acquisition_id'], 'Native/phase identity changed')
    dat_path = PREPARED / 'dat_cache' / (stem + '.dat')
    data = dat_path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    require(digest == record['dat_sha256'], 'Pinned local DAT bytes changed')
    blob = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
    require(len(data) == dat_inventory['size'] and blob == dat_inventory['sha'] == record['dat_git_blob_sha1'],
            'DAT Git inventory binding mismatch')
    csi = decode_dat_csi(data)
    ticks, _ = read_dat_timestamps(data)
    elapsed, steps = native_elapsed(ticks)
    require(len(csi) == record['packet_count'] and float(elapsed[-1]/1e6) == record['duration_seconds']
            and float(steps.max()/1e6) == record['max_gap_seconds'], 'Native duration/packet metadata changed')
    values = reconstruct(csi, elapsed)
    grid = np.linspace(0.0, elapsed[-1], 850)
    require(grid[0] == 0 and grid[-1] == elapsed[-1] and np.all(np.diff(grid) > 0), 'Invalid native grid')
    evidence = {**identity, 'compile_config_sha256': config_sha,
        'native_preparation_record_sha256': expected_record_sha,
        'native_DAT_git_blob_sha1': blob, 'phase_float32_sha256': matrix_sha(values),
        'grid_float64_sha256': hashlib.sha256(grid.astype('<f8').tobytes()).hexdigest(),
        'grid_points': 850, 'duration_seconds': float(elapsed[-1]/1e6),
        'grid_spacing_seconds': float(grid[1]/1e6), 'packet_count': len(csi),
        'max_gap_seconds': float(steps.max()/1e6),
        'duplicate_timestamp_steps': int(np.count_nonzero(steps == 0)),
        'zero_complex_elements_angle_zero': int(np.count_nonzero(np.abs(csi) == 0)),
        'raw_complex_element_count': int(csi.size), 'phase_min': float(values.min()), 'phase_max': float(values.max()),
        'shape': [850, 90], 'dtype': 'float32', 'nonfinite_count': 0}
    return values, evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    require(output == DEFAULT_OUTPUT.resolve(), 'This operation is scoped to the new named dataset directory')
    if (output / 'dataset_manifest.json').exists():
        manifest = read(output / 'dataset_manifest.json')
        require(manifest.get('complete') and manifest['rows'] == 9000, 'Existing dataset is incomplete')
        for relative, digest in manifest['artifacts_sha256'].items():
            require(sha256(output / relative) == digest, 'Completed dataset artifact changed')
        print('Completed dataset verifies; no changes', flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    native_manifest = read(NATIVE / 'dataset_manifest.json')
    native_prep = read(PREPARED / 'summary.json')
    native_config = read(PREPARED / 'config.json')
    require(native_manifest['complete'] and native_manifest['rows'] == 9000
            and native_prep['complete_9000'] and not native_prep['failures'], 'Incomplete native parents')
    require(value_hash(native_config) == native_prep['config_value_sha256'], 'Native preparation config changed')
    require(read(NATIVE / 'compile_config.json')['preparation_summary_sha256'] == sha256(PREPARED/'summary.json'),
            'Native compilation/preparation binding changed')
    for relative, expected in native_config['code_sha256'].items():
        require(sha256(ROOT / relative) == expected, 'Frozen native preprocessing implementation changed')
    for relative, digest in native_manifest['artifacts_sha256'].items():
        if relative.startswith('native_time_linear/') or relative in ('compile_config.json', 'E1_label.npy', 'E2_label.npy', 'E3_label.npy'):
            require(sha256(NATIVE / relative) == digest, 'Native C input artifact changed: ' + relative)
    # The cross-link reference is read only to prove that the two chains differ
    # by whole turns beyond the first receive link and nowhere else.
    require(sha256(REFERENCE / 'dataset_manifest.json') == REFERENCE_MANIFEST_SHA256, 'Cross-link reference manifest changed')
    reference_manifest = read(REFERENCE / 'dataset_manifest.json')
    for env in ('E1', 'E2', 'E3'):
        relative = f'{REFERENCE_VARIANT}/{env}_phase.npy'
        require(sha256(REFERENCE / relative) == reference_manifest['artifacts_sha256'][relative], 'Cross-link reference array changed')
    config = {'version': VARIANT, 'frozen_at_operation': '2026-10-23', 'definition': DEFINITION,
              'native_compiled': str(NATIVE), 'native_prepared': str(PREPARED),
              'native_dataset_manifest_sha256': sha256(NATIVE/'dataset_manifest.json'),
              'native_preparation_summary_sha256': sha256(PREPARED/'summary.json'),
              'native_preparation_config_sha256': sha256(PREPARED/'config.json'),
              'cross_link_reference': str(REFERENCE), 'cross_link_reference_manifest_sha256': REFERENCE_MANIFEST_SHA256,
              'raw_DAT_inventory_sha256': sha256(ROOT/'docs/revision/sources/raw_dat_inventory.json'),
              'code_sha256': {name: sha256(ROOT/name) for name in CODE_FILES},
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
              'record_count': 9000, 'model_training_or_scoring': False,
              'target_records_processed_for_preprocessing_only': True, 'cohort_exclusions': 0,
              'complete_hardware_calibration': False}
    if (output/'compile_config.json').exists():
        require(read(output/'compile_config.json') == config, 'A different processing definition/config exists; preserve it')
    else:
        write(output/'compile_config.json', config)
    config_sha = sha256(output/'compile_config.json')
    for name, expected in config['code_sha256'].items():
        destination = output/'code'/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            require(sha256(destination) == expected, 'Existing code snapshot differs')
        else:
            shutil.copy2(ROOT/name, destination)
    qa = synthetic_qa()
    qa['compile_config_sha256'] = config_sha
    write(output/'synthetic_qa.json', qa)
    inventory_json = read(ROOT/'docs/revision/sources/raw_dat_inventory.json')
    require(inventory_json['commit'] == native_config['source_commit'], 'DAT source commit differs')
    inventory = {r['path']: r for r in inventory_json['files']}
    files, all_records, relation = {}, [], {}
    started = time.monotonic()
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        for env in ('E1', 'E2', 'E3'):
            reference_path = NATIVE/'native_time_linear/manifests'/f'deferred_target_{env}.json'
            reference = read(reference_path)
            require(len(reference['test']) == 3000 and [r['index'] for r in reference['test']] == list(range(3000)),
                    'C row ordering incomplete')
            final_path = output/VARIANT/f'{env}_phase.npy'
            final_path.parent.mkdir(parents=True, exist_ok=True)
            records_path = output/'environment_records'/f'{env}.json'
            if final_path.exists() and records_path.exists():
                saved = read(records_path)
                require(saved['compile_config_sha256'] == config_sha and sha256(final_path) == saved['phase_file_sha256'],
                        'Committed environment differs from frozen config/array')
                records = saved['records']
                require(len(records) == 3000, 'Incomplete committed environment')
                print(f'Reused committed {env}: 3000 rows', flush=True)
            else:
                temporary = final_path.with_suffix('.npy.partial')
                matrix = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(3000,850,90))
                tasks = [(identity, native_prep['record_sha256'][identity['identity_source_mat_path']],
                          inventory[identity['raw_path']], config_sha) for identity in reference['test']]
                records = []
                try:
                    for index, (values, evidence) in enumerate(pool.map(process_record, tasks, chunksize=4)):
                        require(evidence['index'] == index and evidence['sample_id'] == reference['test'][index]['sample_id'],
                                'Worker row ordering changed')
                        matrix[index] = values
                        records.append(evidence)
                        if (index+1) % 100 == 0:
                            matrix.flush()
                            write(output/'status.json', {'status': 'compiling', 'updated_at': now(), 'environment': env,
                                  'completed': len(all_records)+len(records), 'total': 9000,
                                  'compile_config_sha256': config_sha, 'elapsed_seconds': time.monotonic()-started})
                            print(f'{env}: {index+1}/3000; total {len(all_records)+len(records)}/9000', flush=True)
                    matrix.flush()
                finally:
                    matrix._mmap.close()
                require(len(records) == 3000, 'Incomplete generated environment')
                phase_hash = sha256(temporary)
                if final_path.exists():
                    require(sha256(final_path) == phase_hash, 'Existing completed phase array differs')
                    temporary.unlink()
                else:
                    os.replace(temporary, final_path)
                write(records_path, {'compile_config_sha256': config_sha, 'rows': 3000,
                                      'phase_file_sha256': phase_hash, 'records': records})
            # Full compiled-content verification plus the per-row whole-turn
            # relation to the frozen cross-link historical-style array.
            array = np.load(final_path, mmap_mode='r', allow_pickle=False)
            crossed = np.load(REFERENCE/REFERENCE_VARIANT/f'{env}_phase.npy', mmap_mode='r', allow_pickle=False)
            require(array.shape == crossed.shape == (3000,850,90) and array.dtype == np.float32, 'Compiled shape/dtype failed')
            env_relation = {'rows': 3000, 'rows_identical_to_cross_link': 0, 'link_1_frames_shifted': 0,
                            'link_2_frames_shifted': 0, 'max_turn_residual_rad': 0.0}
            for index, evidence in enumerate(records):
                require(np.isfinite(array[index]).all() and matrix_sha(array[index]) == evidence['phase_float32_sha256'],
                        'Compiled phase row/hash/finite mismatch')
                identity = reference['test'][index]
                require(all(evidence[k] == identity[k] for k in identity), 'Compiled identity differs from C')
                row = compare_with_reference(np.asarray(array[index]), np.asarray(crossed[index]))
                for key in ('link_1_frames_shifted', 'link_2_frames_shifted'):
                    env_relation[key] += row[key]
                env_relation['max_turn_residual_rad'] = max(env_relation['max_turn_residual_rad'], row['max_turn_residual_rad'])
                env_relation['rows_identical_to_cross_link'] += int(row['link_1_frames_shifted'] == row['link_2_frames_shifted'] == 0)
            env_relation['frames_per_link'] = 3000 * 850
            relation[env] = env_relation
            array._mmap.close()
            crossed._mmap.close()
            all_records.extend(records)
            phase_spec = {'path': str(final_path), 'sha256': sha256(final_path), 'shape': [3000,850,90],
                          'dtype': 'float32', 'nonfinite_count': 0,
                          'min': min(r['phase_min'] for r in records), 'max': max(r['phase_max'] for r in records)}
            spec = copy.deepcopy(reference['files'][env])
            spec['data_files']['phase'] = phase_spec
            spec['actual_file_sha256']['phase'] = phase_spec['sha256']
            sidecar = output/VARIANT/'subject_sidecars'/f'{env}.json'
            write(sidecar, {'env_name': env,
                'subject_ids': [r['subject_id'] for r in reference['test']],
                'acquisition_ids': [r['acquisition_id'] for r in reference['test']],
                'raw_paths': [r['raw_path'] for r in reference['test']],
                'data_file_sha256': spec['actual_file_sha256'],
                'provenance': {'variant': VARIANT, 'compile_config_sha256': config_sha,
                    'same_native_C_time_grid': True, 'reference_native_manifest_sha256': sha256(reference_path),
                    'verified_rows': 3000, 'within_link_unwrap_control_not_calibration': True}})
            spec.update(sidecar_path=str(sidecar), sidecar_sha256=sha256(sidecar))
            files[env] = spec
    require(len(all_records) == 9000 and len({r['sample_id'] for r in all_records}) == 9000, 'Incomplete unique cohort')
    for prefix in ('source', 'deferred_target'):
        for env in ('E1', 'E2', 'E3'):
            reference_path = NATIVE/'native_time_linear/manifests'/f'{prefix}_{env}.json'
            reference = read(reference_path)
            manifest = copy.deepcopy(reference)
            manifest['files'] = {name: files[name] for name in reference['files']}
            manifest.update(data_version=VARIANT, phase_available=True,
                phase_preprocessing_applied_before_loading=True, complete_hardware_calibration=False,
                phase_input_compile_config_sha256=config_sha,
                reference_native_time_manifest_sha256=sha256(reference_path),
                phase_processing_definition=DEFINITION,
                within_link_unwrap_control_not_calibration=True)
            for part in ('train', 'val', 'test'):
                if part in reference:
                    require(manifest[part] == reference[part], 'Manifest membership/labels/order changed')
            if prefix == 'source':
                for key in ('sample_id', 'subject_id', 'acquisition_id'):
                    require(not ({r[key] for r in manifest['train']} & {r[key] for r in manifest['val']}),
                            'Source split overlap')
            for name in reference['files']:
                require(manifest['files'][name]['data_files']['amp'] == reference['files'][name]['data_files']['amp'],
                        'Shared amplitude specification changed')
            write(output/VARIANT/'manifests'/f'{prefix}_{env}.json', manifest)

    selected = {r['sample_id'] for r in all_records if r['index'] == 0}
    selected.update(('E1:1108', 'E1:1715'))
    selected.update(r['sample_id'] for r in sorted(all_records, key=lambda r:r['max_gap_seconds'], reverse=True)[:3])
    sample_checks = []
    for record in all_records:
        if record['sample_id'] not in selected:
            continue
        data = (PREPARED/'dat_cache'/(Path(record['raw_path']).stem+'.dat')).read_bytes()
        require(hashlib.sha256(data).hexdigest() == record['raw_sha256'], 'Replay DAT changed')
        csi = decode_dat_csi(data)
        elapsed, _ = native_elapsed(read_dat_timestamps(data)[0])
        independent = independent_reconstruct(csi, elapsed)
        compiled = np.load(output/VARIANT/f'{record["environment"]}_phase.npy', mmap_mode='r', allow_pickle=False)
        error = float(np.max(np.abs(independent-compiled[record['index']])))
        require(error < 1e-4, 'Independent scalar phase replay differs beyond numeric tolerance')
        compiled._mmap.close()
        amp_rebuilt = uniform_interpolation(np.abs(csi), elapsed).astype(np.float32)
        amp = np.load(NATIVE/'native_time_linear'/f'{record["environment"]}_amp.npy', mmap_mode='r', allow_pickle=False)
        require(np.array_equal(amp_rebuilt, amp[record['index']]), 'Shared C amplitude grid reconstruction differs')
        amp._mmap.close()
        sample_checks.append({'sample_id': record['sample_id'], 'independent_phase_max_abs_error': error,
            'tolerance_absolute': 1e-4, 'native_C_amplitude_replay_exact': True,
            'duration_seconds': record['duration_seconds'], 'max_gap_seconds': record['max_gap_seconds']})
    require(len(sample_checks) == len(selected), 'Independent replay sample missing')
    total_frames = 9000 * 850
    qa = {'passed': True, 'completed_at': now(), 'rows': 9000, 'variant': VARIANT,
          'compile_config_sha256': config_sha, 'full_DAT_SHA_and_pinned_Git_blob_verified': True,
          'original_DAT_to_MAT_identity_preserved': True, 'all_model_input_row_hashes_verified': True,
          'all_finite_float32_shape_850_90': True, 'all_source_target_membership_labels_order_preserved': True,
          'same_native_C_amplitude_files_without_copy': True,
          'same_native_C_elapsed_endpoint_grid_definition': True,
          'source_subject_and_acquisition_disjoint': True, 'synthetic_qa_sha256': sha256(output/'synthetic_qa.json'),
          'independent_reconstruction': sample_checks,
          'cross_link_reference_relation': {
              'definition': 'link 0 bitwise equal; links 1 and 2 equal the cross-link array minus 2*pi*k with one integer k per row, frame and link',
              'all_9000_rows_verified': True, 'environments': relation,
              'link_1_fraction_frames_shifted': sum(r['link_1_frames_shifted'] for r in relation.values()) / total_frames,
              'link_2_fraction_frames_shifted': sum(r['link_2_frames_shifted'] for r in relation.values()) / total_frames,
              'rows_identical_to_cross_link': sum(r['rows_identical_to_cross_link'] for r in relation.values())},
          'zero_complex_elements_angle_zero_retained': sum(r['zero_complex_elements_angle_zero'] for r in all_records),
          'raw_complex_element_count': sum(r['raw_complex_element_count'] for r in all_records),
          'duplicate_timestamp_steps': sum(r['duplicate_timestamp_steps'] for r in all_records),
          'max_packet_gap_seconds': max(r['max_gap_seconds'] for r in all_records),
          'complete_hardware_calibration': False, 'model_training_or_scoring': False,
          'target_records_processed_for_preprocessing_only': True, 'cohort_exclusions': 0}
    write(output/'identity_and_grid_qa.json', qa)
    for name, digest in config['code_sha256'].items():
        require(sha256(ROOT/name) == digest and sha256(output/'code'/name) == digest, 'Code changed during compilation')
    artifacts = [p for p in output.rglob('*') if p.is_file() and p.name not in
                 ('dataset_manifest.json', 'status.json') and not p.name.endswith(('.partial', '.tmp'))]
    manifest = {'complete': True, 'completed_at': now(), 'rows': 9000, 'variants': [VARIANT],
        'phase_included': True, 'amplitude_copied': False,
        'original_partition_order_preserved': True, 'same_native_C_time_grid': True,
        'within_link_unwrap_control_not_calibration': True, 'model_training_or_scoring': False,
        'compile_config_sha256': config_sha,
        'external_amplitude_and_label_files': {env: {key: files[env]['data_files'][key] for key in ('amp','label')}
                                             for env in ('E1','E2','E3')},
        'artifacts_sha256': {p.relative_to(output).as_posix(): sha256(p) for p in sorted(artifacts)}}
    write(output/'dataset_manifest.json', manifest)
    write(output/'status.json', {'status': 'completed', 'updated_at': now(), 'completed': 9000, 'total': 9000,
          'dataset_manifest_sha256': sha256(output/'dataset_manifest.json'),
          'elapsed_seconds': time.monotonic()-started})
    rel = qa['cross_link_reference_relation']
    report = [
        '# Native-grid within-link unwrap phase control — 2026-10-23', '',
        f'**Completed and verified: 9,000 rows.** Variant: `{VARIANT}`.', '',
        f'Dataset directory: `{output}`. It contains three float32 `[3000,850,90]` phase arrays, six source/target manifests, three subject sidecars, complete record identities and hashes, frozen code, and synthetic/input QA. Native C amplitude and labels are referenced without duplication or modification.', '',
        '## Frozen definition', '',
    ]
    report += [f'- **{key}:** {value}' for key,value in DEFINITION.items()]
    report += ['', '## Verification and interpretation', '',
        '- Every DAT byte stream was checked against SHA-256, pinned Git blob and the native preparation identity; original DAT-to-MAT mappings, labels and partition order are retained.',
        '- All 9,000 output rows passed shape, float32, finite and per-row content-hash verification. Six manifests preserve C sample identities and source subject/acquisition separation.',
        f'- Eight input-selected records passed independent per-column interpolation and explicit per-link unwrap replay (maximum error {max(r["independent_phase_max_abs_error"] for r in sample_checks):.8g}, tolerance 1e-4); their C amplitude reconstruction is exactly equal to the frozen arrays.',
        f'- Against the frozen cross-link array (`{REFERENCE_VARIANT}`), receive link 0 is bitwise equal in all rows, and links 1 and 2 differ only by one whole-turn offset per row, frame and link (maximum residual {max(r["max_turn_residual_rad"] for r in relation.values()):.3g} rad). The cross-link unwrap shifted link 1 in {100*rel["link_1_fraction_frames_shifted"]:.2f}% and link 2 in {100*rel["link_2_fraction_frames_shifted"]:.2f}% of frames; {rel["rows_identical_to_cross_link"]} of 9,000 rows are identical in both chains.',
        '- This is a within-link unwrap control on the common native grid. It is not a physically calibrated phase. The only difference from the historical-style chain is whether the feature-axis unwrap crosses the two receive-link boundaries.',
        '- No model was trained or scored during this operation. Training must use already processed phase directly and must not apply another unwrap.', '',
        f'Dataset manifest SHA-256: `{sha256(output/"dataset_manifest.json")}`.']
    (ROOT/'docs/revision/native_linkunwrap_phase_20261023.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    print(json.dumps({'complete': True, 'rows': 9000, 'variant': VARIANT,
                      'manifest': str(output/'dataset_manifest.json'),
                      'dataset_manifest_sha256': sha256(output/'dataset_manifest.json'),
                      'cross_link_relation': rel}), flush=True)


if __name__ == '__main__':
    main()
