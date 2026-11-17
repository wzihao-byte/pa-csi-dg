"""Read-only CPU verification of frozen B/C time-preprocessing experiment inputs.

Rehash all packed inputs and preparation records; reuse the bound 9,000-row audit
and independently reconstruct a declared local subset. No training or download.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.audit_multienv_timestamps import decode_dat_csi
from analysis.prepare_native_multienv import native_elapsed, uniform_interpolation
from analysis.verify_multienv_raw_mapping import legacy_merge, read_dat_timestamps, timestamp_text15
from experiments.revision_validation.study_io import value_hash

BASE = Path('D:/pa-csi-dg-revision-runtime/data')
PREPARED = BASE / 'native_amp_v1'
COMPILED = BASE / 'native_amp_arrays_v1'
PROTOCOL = ROOT / 'experiments/revision_validation/verified_protocol'
AUDIT = ROOT / 'docs/revision/evidence/native_reconstruction_v1'
OUTPUT = ROOT / 'docs/revision/time_validation_20260929'
VARIANTS = ('native_time_legacy_merge', 'native_time_linear')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def summarize(values):
    values = np.asarray(values, dtype=np.float64)
    return {key: float(value) for key, value in zip(
        ('min', 'median', 'p95', 'max'),
        (values.min(), np.median(values), np.quantile(values, .95), values.max()))}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {'started_at': datetime.now(timezone.utc).isoformat(), 'passed': False,
              'training_or_target_scoring': False, 'downloaded_data': False,
              'historical_or_frozen_files_modified': False, 'script_sha256': sha(__file__),
              'compiled_directory': str(COMPILED), 'prepared_directory': str(PREPARED),
              'full_raw_DAT_replay_performed': False}
    manifest = read(COMPILED / 'dataset_manifest.json')
    preparation = read(PREPARED / 'summary.json')
    config = read(PREPARED / 'config.json')
    compile_config = read(COMPILED / 'compile_config.json')
    audit = read(AUDIT / 'summary.json')
    require(manifest['complete'] and manifest['rows'] == 9000, 'Incomplete compiled inputs')
    require(preparation['complete_9000'] and preparation['rows'] == 9000
            and not preparation['failures'] and preparation['all_packet_signal_matches'],
            'Incomplete or unverified preparation')
    require(value_hash(config) == preparation['config_value_sha256'], 'Preparation config changed')
    require(sha(PREPARED / 'summary.json') == compile_config['preparation_summary_sha256']
            == audit['native_preparation_summary_sha256'], 'Preparation summary binding changed')
    require(sha(COMPILED / 'dataset_manifest.json') == audit['native_dataset_manifest_sha256'],
            'Dataset and previous reconstruction audit no longer match')
    require(sha(ROOT / 'analysis/compile_revision_inputs.py') == compile_config['compiler_sha256'],
            'Compiler implementation changed')
    code_checks = {}
    for relative, expected in config['code_sha256'].items():
        actual = sha(ROOT / relative)
        require(actual == expected, 'Preparation implementation changed: ' + relative)
        code_checks[relative] = actual
    require(sha(ROOT / 'analysis/audit_native_reconstruction.py') == audit['code_sha256'],
            'Previous reconstruction audit implementation changed')
    result['source_code_sha256_verified'] = code_checks
    artifact_checks = {}
    for relative, expected in manifest['artifacts_sha256'].items():
        actual = sha(COMPILED / relative)
        require(actual == expected, 'Compiled artifact hash mismatch: ' + relative)
        artifact_checks[relative] = actual
    result['compiled_artifacts_sha256_verified'] = artifact_checks
    print(f'All {len(artifact_checks)} compiled artifact hashes verified', flush=True)

    records = {}
    for mat_relative, expected in preparation['record_sha256'].items():
        path = PREPARED / 'records' / (Path(mat_relative).stem + '.json')
        require(sha(path) == expected, 'Preparation record hash mismatch: ' + mat_relative)
        row = read(path)
        require(row['verified'] and row['all_complex_csi_and_packet_ids_equal'],
                'Unverified preparation row: ' + mat_relative)
        require(row['mat_path'] == mat_relative and row['config_value_sha256'] == value_hash(config),
                'Mixed identity/configuration: ' + mat_relative)
        records[mat_relative] = row
    require(len(records) == 9000 and len({r['dat_sha256'] for r in records.values()}) == 9000,
            'Duplicate or incomplete prepared records')
    result['preparation_record_hashes_verified'] = len(records)
    print('All 9000 preparation record hashes verified', flush=True)

    historical = {}
    array_checks, membership_checks = [], []
    for env in ('E1', 'E2', 'E3'):
        old_target = read(PROTOCOL / f'deferred_target_{env}.json')
        historical[env] = old_target
        amp_spec = old_target['files'][env]['data_files']['amp']
        old_hash = sha(amp_spec['path'])
        require(old_hash == old_target['files'][env]['actual_file_sha256']['amp']
                == audit['historical_amplitude_sha256'][env], 'Historical amplitude changed: ' + env)
        labels = np.load(COMPILED / f'{env}_label.npy', allow_pickle=False)
        require(labels.shape == (3000,) and labels.tolist() == [r['label'] for r in old_target['test']],
                'Compiled labels no longer match historical identities')
        for old in old_target['test']:
            row = records[old['raw_path']]
            require(row['environment'] == env and row['index'] == old['index']
                    and row['mat_sha256'] == old['raw_sha256']
                    and env + ':' + row['subject_id'] == old['subject_id']
                    and row['acquisition_id'] == old['acquisition_id'], 'Historical identity changed')
        for variant in VARIANTS:
            path = COMPILED / variant / f'{env}_amp.npy'
            a = np.load(path, mmap_mode='r', allow_pickle=False)
            require(a.shape == (3000, 850, 90) and a.dtype == np.float32, 'Bad packed shape/dtype')
            nonfinite, minimum, maximum = 0, float('inf'), float('-inf')
            for start in range(0, len(a), 32):
                part = np.asarray(a[start:start + 32])
                nonfinite += int(np.count_nonzero(~np.isfinite(part)))
                minimum, maximum = min(minimum, float(part.min())), max(maximum, float(part.max()))
            require(nonfinite == 0, 'Nonfinite packed model input')
            array_checks.append({'variant': variant, 'environment': env, 'path': str(path),
                                 'rows': len(a), 'shape': list(a.shape), 'dtype': str(a.dtype),
                                 'nonfinite_count': nonfinite, 'min': minimum, 'max': maximum,
                                 'sha256': artifact_checks[f'{variant}/{env}_amp.npy']})
            a._mmap.close()
            for kind in ('source', 'deferred_target'):
                previous_path = PROTOCOL / f'{kind}_{env}.json'
                previous = read(previous_path)
                current = read(COMPILED / variant / 'manifests' / previous_path.name)
                require(current['original_partition_manifest_sha256'] == sha(previous_path),
                        'Original partition manifest binding mismatch')
                for part in ('train', 'val', 'test'):
                    if part not in previous:
                        continue
                    require(len(current[part]) == len(previous[part]), 'Partition size changed')
                    for old, new in zip(previous[part], current[part]):
                        require(all(old[key] == new[key] for key in
                                    ('sample_id', 'environment', 'index', 'label', 'subject_id', 'acquisition_id')),
                                'Partition membership/order/labels changed')
                        record = records[old['raw_path']]
                        require(new['identity_source_mat_path'] == old['raw_path']
                                and new['identity_source_mat_sha256'] == old['raw_sha256']
                                and new['raw_path'] == record['dat_path']
                                and new['raw_sha256'] == record['dat_sha256'], 'DAT identity binding changed')
                    membership_checks.append({'variant': variant, 'outer_target': env,
                                              'partition': part, 'rows': len(current[part]),
                                              'historical_order_labels_subjects_preserved': True})
                if kind == 'source':
                    for key in ('sample_id', 'subject_id', 'acquisition_id'):
                        require(not ({r[key] for r in current['train']} & {r[key] for r in current['val']}),
                                'Source train-validation overlap')
        print(f'{env}: packed finite/shape checks, historical hash, membership passed', flush=True)
    result['arrays'] = array_checks
    result['membership_checks'] = membership_checks

    for relative, expected in audit['artifacts_sha256'].items():
        require(sha(AUDIT / relative) == expected, 'Bound reconstruction audit artifact changed')
    with (AUDIT / 'native_clock_statistics.csv').open(encoding='utf-8', newline='') as handle:
        clock_rows = list(csv.DictReader(handle))
    require(len(clock_rows) == 9000, 'Clock audit row count changed')
    for old_clock in clock_rows:
        old = historical[old_clock['environment']]['test'][int(old_clock['sample_id'].split(':')[1])]
        record = records[old['raw_path']]
        for key in ('duration_seconds', 'max_gap_seconds', 'packet_count', 'duplicate_timestamp_steps', 'uint32_wraps'):
            require(float(old_clock[key]) == float(record[key]), 'Clock statistics differ from bound records')
    sorted_gaps = sorted(records.values(), key=lambda r: r['max_gap_seconds'], reverse=True)
    result['clock_statistics'] = {
        'duration_seconds': summarize([r['duration_seconds'] for r in records.values()]),
        'max_gap_seconds_per_record': summarize([r['max_gap_seconds'] for r in records.values()]),
        'duplicate_timestamp_steps': sum(r['duplicate_timestamp_steps'] for r in records.values()),
        'uint32_wraps': sum(r['uint32_wraps'] for r in records.values()),
        'records_with_max_gap_gt_0_05_seconds': sum(r['max_gap_seconds'] > .05 for r in records.values()),
        'records_with_max_gap_gt_0_1_seconds': sum(r['max_gap_seconds'] > .1 for r in records.values()),
        'records_with_max_gap_gt_0_5_seconds': sum(r['max_gap_seconds'] > .5 for r in records.values()),
        'records_with_max_gap_gt_1_second': sum(r['max_gap_seconds'] > 1 for r in records.values()),
        'longest_gaps': [{key: row[key] for key in
                         ('environment', 'index', 'mat_path', 'duration_seconds', 'max_gap_seconds', 'packet_count')}
                        for row in sorted_gaps[:10]]}
    result['bound_full_reconstruction_audit_reused'] = {
        'path': str(AUDIT / 'summary.json'), 'sha256': sha(AUDIT / 'summary.json'),
        'all_artifact_hashes_verified': True,
        'comparisons': {name: value['overall'] for name, value in audit['statistics'].items()}}

    # Fixed representative IDs and both known historical timestamp-problem records;
    # select extra maximum-gap examples using inputs only, without recognition scores.
    selected = [historical[env]['test'][0]['raw_path'] for env in ('E1', 'E2', 'E3')]
    selected += ['MAT/E1/E1_S03_C04_A01_T09.mat', 'MAT/E1/E1_S05_C04_A01_T16.mat']
    selected += [r['mat_path'] for r in sorted_gaps[:3]]
    selected = list(dict.fromkeys(selected))
    reconstruction_checks = []
    inventory = read(ROOT / 'docs/revision/sources/raw_dat_inventory.json')
    require(inventory['commit'] == config['source_commit'], 'DAT source commit changed')
    inventory = {r['path']: r for r in inventory['files']}
    for relative in selected:
        row = records[relative]
        env, index = row['environment'], row['index']
        stem = Path(relative).stem
        dat_path = PREPARED / 'dat_cache' / (stem + '.dat')
        dat_bytes = dat_path.read_bytes()
        require(hashlib.sha256(dat_bytes).hexdigest() == row['dat_sha256'], 'Selected DAT changed')
        blob = hashlib.sha1(b'blob ' + str(len(dat_bytes)).encode() + b'\0' + dat_bytes).hexdigest()
        require(blob == row['dat_git_blob_sha1'] == inventory[row['dat_path']]['sha']
                and len(dat_bytes) == inventory[row['dat_path']]['size'], 'Selected pinned DAT blob changed')
        mat_path = ROOT / 'docs/revision/sources/raw_mat_cache' / (stem + '.mat')
        require(sha(mat_path) == row['mat_sha256'], 'Selected original MAT changed')
        packets = loadmat(io.BytesIO(mat_path.read_bytes()), simplify_cells=True)['data']
        native = decode_dat_csi(dat_bytes)
        ticks, counts = read_dat_timestamps(dat_bytes)
        original = np.stack([p['csi'].reshape(90) for p in packets])
        require(np.array_equal(native, original) and
                np.array_equal(counts, np.asarray([p['bfee_count'] for p in packets])),
                'Selected DAT and MAT packet values or IDs differ')
        elapsed, differences = native_elapsed(ticks)
        require(float(elapsed[-1] / 1e6) == row['duration_seconds']
                and float(differences.max() / 1e6) == row['max_gap_seconds']
                and int(np.count_nonzero(differences == 0)) == row['duplicate_timestamp_steps'],
                'Reconstructed clock differs from record')
        amplitude = np.abs(native)
        replay = (legacy_merge(amplitude, elapsed).astype(np.float32),
                  uniform_interpolation(amplitude, elapsed).astype(np.float32))
        grid = np.linspace(0, elapsed[-1], 850)
        require(len(grid) == 850 and grid[0] == 0 and grid[-1] == elapsed[-1]
                and np.all(np.diff(grid) > 0), 'Invalid native linear grid')
        sample_path = PREPARED / 'samples' / (stem + '.npz')
        require(sha(sample_path) == row['sample_sha256'], 'Selected prepared sample changed')
        with np.load(sample_path, allow_pickle=False) as sample:
            for variant, sample_key, values in zip(VARIANTS,
                    ('historical_merge_native_time', 'linear_native_time'), replay):
                packed = np.load(COMPILED / variant / f'{env}_amp.npy', mmap_mode='r', allow_pickle=False)
                require(np.array_equal(values, sample[sample_key]) and np.array_equal(values, packed[index]),
                        'Selected reconstructed input differs from frozen packed row')
                packed._mmap.close()
        old_path = historical[env]['files'][env]['data_files']['amp']['path']
        old_array = np.load(old_path, mmap_mode='r', allow_pickle=False)
        old_ticks = timestamp_text15(np.asarray([p['timestamp_low'] for p in packets], dtype=float))
        old_replay = legacy_merge(np.abs(original), old_ticks)
        old_max_error = float(np.max(np.abs(old_replay - old_array[index])))
        require(old_max_error < 1e-10, 'Selected historical replay differs from original array')
        old_array._mmap.close()
        max_gap_index = int(np.argmax(differences))
        gap_grid_points = int(np.count_nonzero((grid > elapsed[max_gap_index]) &
                                              (grid < elapsed[max_gap_index + 1])))
        reconstruction_checks.append({'environment': env, 'index': index, 'raw_mat_path': relative,
            'dat_path': str(dat_path), 'packet_count': len(native), 'native_duration_seconds': float(elapsed[-1]/1e6),
            'grid_points': len(grid), 'grid_spacing_seconds': float(grid[1]/1e6),
            'grid_is_record_specific_not_fixed_Hz': True,
            'duplicate_timestamp_steps': int(np.count_nonzero(differences == 0)),
            'max_gap_seconds': float(differences.max()/1e6),
            'largest_gap_start_seconds': float(elapsed[max_gap_index]/1e6),
            'largest_gap_end_seconds': float(elapsed[max_gap_index+1]/1e6),
            'linear_grid_points_strictly_inside_largest_gap': gap_grid_points,
            'DAT_MAT_complex_CSI_and_packet_IDs_exact': True,
            'replayed_B_and_C_exact_to_prepared_and_packed_float32': True,
            'historical_replay_max_absolute_error': old_max_error})
        print(f'Local replay passed: {stem}', flush=True)
    result['independent_local_reconstruction'] = reconstruction_checks
    result['interpretation_limits'] = [
        'B changes only the timestamp basis while retaining historical merge boundaries and reverse-tail padding; its 850 output indices are not a strict physical-time grid.',
        'C linearly interpolates native DAT amplitude onto 850 equally spaced within-record times including endpoints; spacing varies with recording duration, so it does not impose a fixed sampling frequency.',
        'Duplicate timestamps would be averaged before linear interpolation; no duplicate steps or uint32 wraps occur in the 9000 verified provenance records.',
        'Linear interpolation crosses missing-packet gaps without a gap mask or gap-duration cutoff. The longest observed gap is 2.015493 seconds; the frozen C arm tests sensitivity under that rule, not recovery of the missing physical signal.',
        'No rows are excluded due to long gaps or model performance; changing the gap policy would require a separately versioned experiment.',
        'All packed input bytes, preparation records, identities and prior audit hashes were rechecked. Raw DAT/MAT reconstruction was independently repeated only for the listed subset; the complete 9000-row reconstruction is reused through hash bindings.',
        'Historical row index and native elapsed seconds are different coordinates and must not be plotted as the same measured physical time.'
    ]
    result['passed'] = True
    result['completed_at'] = datetime.now(timezone.utc).isoformat()
    (OUTPUT / 'input_check.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    cs = result['clock_statistics']
    lines = [
        '# Time preprocessing input verification — 2026-09-29', '',
        '**PASS.** Read-only CPU checks are complete. No training, target scoring, download, array modification, or row exclusion was performed.', '',
        '## Coverage', '',
        f'- Rehashed all {len(artifact_checks)} frozen compiled artifacts, including six amplitude arrays. Each B/C variant contains 9,000 rows; each environment array is float32 `[3000, 850, 90]`, with zero nonfinite values.',
        '- Verified all 9,000 preparation-record hashes and bindings to the preparation config, source code, compile config and prior reconstruction audit. Historical amplitude hashes also match.',
        '- Matched historical sample IDs, order, labels, subject IDs and acquisition IDs for every source/validation/target partition. Source train/validation subject and acquisition sets remain disjoint.',
        f'- Independently reconstructed {len(reconstruction_checks)} selected local DAT/MAT records, including the two historical timestamp-problem records and three largest-gap records. Both B/C inputs exactly match the stored float32 rows; historical replay error is below 1e-10.', '',
        '## What B and C mean', '',
        '| Version | Time basis | Output rule | Interpretation |',
        '|---|---|---|---|',
        '| B: native_time_legacy_merge | Native uint32 DAT elapsed time | Original merge loop and reverse-tail fill | Timestamp-basis sensitivity; 850 indices are not a strict physical-time grid |',
        '| C: native_time_linear | Same native elapsed time | Linear interpolation onto `linspace(0, duration, 850)` | Resampling-rule sensitivity; record-specific spacing, not a fixed Hz |', '',
        '## Clock and gap findings', '',
        f'- Across 9,000 bound provenance records: zero duplicate timestamp steps and zero uint32 wraps. Record durations range from {cs["duration_seconds"]["min"]:.6f} to {cs["duration_seconds"]["max"]:.6f} seconds.',
        f'- {cs["records_with_max_gap_gt_0_05_seconds"]} records contain a gap over 0.05 s; {cs["records_with_max_gap_gt_0_1_seconds"]} over 0.1 s; {cs["records_with_max_gap_gt_0_5_seconds"]} over 0.5 s; {cs["records_with_max_gap_gt_1_second"]} over 1 s. These are descriptive thresholds, not exclusion criteria.',
        '- Longest gap: **2.015493 s**, E2:0158, inside a 5.905944 s recording. C interpolates across this interval without a cutoff or mask. This validates the implementation and its limitations; it does not demonstrate recovery of the missing signal.', '',
        '## Bound existing all-row comparison', '',
        '| Comparison (float32 model inputs) | Exactly equal rows | Changed rows above 1e-5 element error |',
        '|---|---:|---:|']
    for name, value in audit['statistics'].items():
        overall = value['overall']
        lines.append(f'| {name} | {overall["exact_rows"]} / 9000 | {overall["rows_with_abs_error_gt_1e_minus5"]} / 9000 |')
    lines += ['', '## Decision for the proposed experiment', '',
        'No input-integrity blocker was found for the planned B/C global-MLP controls. Keep the frozen arrays, all rows and the matched training protocol unchanged; compare each new global model with its corresponding existing TPP model. Treat the result as preprocessing sensitivity, not a claim that native-time interpolation is universally correct.', '',
        'The large gaps and the historical merge behavior must be disclosed when interpreting the difference. A gap-aware reconstruction would be a new preprocessing version and is outside the current frozen comparison.', '',
        'Full checks, hashes, reconstruction examples, exact gap locations and provenance coverage are recorded in `input_check.json`.']
    (OUTPUT / 'input_check.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'arrays': len(array_checks), 'records': len(records),
                      'local_replays': len(reconstruction_checks), 'report': str(OUTPUT / 'input_check.json')}), flush=True)


if __name__ == '__main__':
    main()
