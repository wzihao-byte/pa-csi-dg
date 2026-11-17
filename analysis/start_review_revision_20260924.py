"""Recover review evidence from sealed predictions without training or target inference.

Outputs are a new dated audit, never mutations of completed experiment bundles.
Run from the repository root: python -X utf8 analysis/start_review_revision_20260924.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.compile_formal_revision_evidence import verify_study
from analysis.compile_multienv_revision_evidence_v3 import verify_h_supplement
from analysis.revision_evidence import H_BASE, H10, H11, Y_ROOT, Y_RESULTS
from experiments.revision_validation.extended_engine import verify_extension
from experiments.revision_validation.study_io import read, sha256, verify_bundle, write, value_hash

OUT = ROOT / 'docs/revision/review_20260924'
ASSETS = ROOT / 'paper/sensors_manuscript/revision_assets/review_20260924'
RUNTIME = Path('D:/pa-csi-dg-revision-runtime/studies')
CLASSES = ['No movement', 'Falling', 'Walking', 'Sitting / standing', 'Turning', 'Picking up a pen']
SUPPORT = np.array([1200, 400, 400, 400, 400, 200])
SOURCES = {}


def bind(path, expected=None):
    path = Path(path)
    digest = sha256(path)
    if expected is not None and digest != expected:
        raise RuntimeError(f'Artifact changed: {path}')
    SOURCES[str(path)] = digest
    return digest


def save_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score(predictions, truth, predicted, id_key):
    if len(predictions) != 3000 or len({p[id_key] for p in predictions}) != 3000:
        raise RuntimeError('Incomplete or duplicate target identities')
    cm = np.zeros((6, 6), dtype=np.int64)
    for p in predictions:
        if not (0 <= p[truth] < 6 and 0 <= p[predicted] < 6):
            raise RuntimeError('Invalid class')
        cm[p[truth], p[predicted]] += 1
    if not np.array_equal(cm.sum(1), SUPPORT):
        raise RuntimeError('Unexpected class support')
    f1 = 200 * cm.diagonal() / (cm.sum(0) + cm.sum(1))
    recall = 100 * cm.diagonal() / cm.sum(1)
    precision = np.divide(100 * cm.diagonal(), cm.sum(0), out=np.zeros(6), where=cm.sum(0) > 0)
    return cm, f1, precision, recall


def add_run(runs, study, arm, env, seed, cm, f1, precision, recall, recorded):
    if abs(float(f1.mean()) - 100 * recorded) > 1e-9:
        raise RuntimeError(f'Recomputed F1 differs: {study}/{arm}/{env}/{seed}')
    runs.append(dict(study=study, arm=arm, environment=env, seed=int(seed),
                     macro_f1_percent=float(f1.mean()), accuracy_percent=100 * np.trace(cm) / cm.sum(),
                     **{f'C{i}_f1_percent': float(f1[i]) for i in range(6)},
                     **{f'C{i}_precision_percent': float(precision[i]) for i in range(6)},
                     **{f'C{i}_recall_percent': float(recall[i]) for i in range(6)}))


def formal_runs(runs):
    verified = {}
    identities = {}
    for short, folder, verifier, count in [('core', 'r_core_v1', verify_bundle, 36),
                                         ('controls', 'r_controls_v1', verify_extension, 81),
                                         ('inputs', 'r_inputs_v1', verify_extension, 63)]:
        bundle = RUNTIME / folder
        protocol, digest, statuses, gate, summary, summary_hash = verify_study(bundle, verifier, count)
        bind(bundle / 'bundle_manifest.json', digest)
        bind(bundle / 'evaluation/summary.json', summary_hash)
        configs = {c['job_id']: c for c in protocol['jobs']}
        for row in summary['runs']:
            config = configs[row['job_id']]
            directory = bundle / 'evaluation' / row['job_id']
            predictions = read(directory / 'predictions.json')
            result = read(directory / 'result.json')
            if result['config_value_sha256'] != value_hash(config):
                raise RuntimeError('Target result/config mismatch')
            bind(directory / 'predictions.json')
            bind(directory / 'result.json')
            bind(bundle / 'runs' / row['job_id'] / 'best_model.pt', result['checkpoint_sha256'])
            manifest_path = Path(config['target_manifest_path']) if 'target_manifest_path' in config else bundle / 'manifests' / f"deferred_target_{row['target']}.json"
            bind(manifest_path, config.get('target_manifest_sha256'))
            target = read(manifest_path)
            identity = [(p['sample_id'], p['subject_id'], p['acquisition_id'], p['true']) for p in predictions]
            expected = [(p['sample_id'], p['subject_id'], p['acquisition_id'], p['label']) for p in target['test']]
            if identity != expected or identities.setdefault(row['target'], identity) != identity:
                raise RuntimeError('Target identity/order mismatch across R protocols')
            cm, f1, precision, recall = score(predictions, 'true', 'predicted', 'sample_id')
            if cm.tolist() != result['metrics']['confusion_matrix']:
                raise RuntimeError('Confusion matrix mismatch')
            add_run(runs, short, row['arm'], row['target'], row['seed'], cm, f1, precision, recall, result['metrics']['macro_f1'])
        verified[short] = dict(runs=count, predictions=count * 3000, bundle_sha256=digest,
                               evaluation_summary_sha256=summary_hash, completion_and_checkpoint_hashes_rechecked=True)
        print(f'{short}: {count} completed runs and {count * 3000} predictions verified', flush=True)
    return verified


def historical_runs(runs):
    matrices = defaultdict(list)
    hrows = [r for r in read(H_BASE)['rows'] if r['arm'] in ('H00', 'H01')]
    hrows += read(H10)['rows'] + read(H11)['rows']
    for path in (H_BASE, H10, H11):
        bind(path)
    identities = {}
    for row in hrows:
        directory = Path(row['attempt_dir']) / 'payload/run'
        path = directory / 'target_predictions.jsonl'
        bind(path, row['target_predictions_file_sha256'])
        predictions = [json.loads(line) for line in path.read_text().splitlines() if line]
        identity = [(p['index'], p['label']) for p in predictions]
        if identities.setdefault(row['target_env'], identity) != identity:
            raise RuntimeError('H paired target identities differ')
        cm, f1, precision, recall = score(predictions, 'label', 'prediction', 'index')
        if not np.array_equal(cm, np.loadtxt(directory / 'confusion_matrix.csv', delimiter=',', dtype=int)):
            raise RuntimeError('H confusion mismatch')
        for name in ('confusion_matrix.csv', 'resolved_config.json', 'split_manifest.json'):
            bind(directory / name)
        add_run(runs, 'H', row['arm'], row['target_env'], row['seed'], cm, f1, precision, recall, row['test_metrics']['f1_macro'])
    bind(Y_RESULTS)
    registry = {r['job_id']: r for r in csv.DictReader((Y_ROOT / 'run_registry.csv').open(encoding='utf-8'))}
    identities = {}
    for row in read(Y_RESULTS)['results']:
        directory = Path(registry[row['job_id']]['output_dir'])
        path = directory / 'target_predictions.jsonl'
        bind(path, row['target_predictions_file_sha256'])
        predictions = [json.loads(line) for line in path.read_text().splitlines() if line]
        identity = [(p['sample_id'], p['subject_id'], p['group_id'], p['true']) for p in predictions]
        if identities.setdefault(row['target_env'], identity) != identity:
            raise RuntimeError('Y paired target identities differ')
        cm, f1, precision, recall = score(predictions, 'true', 'pred', 'sample_id')
        if cm.tolist() != row['target_metrics']['confusion_matrix']:
            raise RuntimeError('Y confusion mismatch')
        for name in ('resolved_config.json', 'split_manifest.json'):
            bind(directory / name)
        add_run(runs, 'Y', row['arm'], row['target_env'], row['seed'], cm, f1, precision, recall, row['target_metrics']['f1_macro'])
        if row['target_env'] == 'E1' and row['arm'] in ('Y10', 'Y11'):
            matrices[row['arm']].append(cm)
            np.savetxt(OUT / f"E1_{row['arm']}_seed{row['seed']}_confusion.csv", cm, delimiter=',', fmt='%d')
    print('H/Y: 72 runs and 216000 predictions recomputed', flush=True)
    return matrices


def h_grouped(runs):
    h = verify_h_supplement(ROOT / 'outputs_h_grouped_l1248/v2')
    bind(h['result_path'], h['results_sha256'])
    for row in h['result']['rows']:
        path = Path(row['attempt_dir']) / 'payload/run/target_predictions.jsonl'
        bind(path, row['target_predictions_file_sha256'])
        predictions = [json.loads(line) for line in path.read_text().splitlines() if line]
        cm, f1, precision, recall = score(predictions, 'label', 'prediction', 'index')
        add_run(runs, 'H-S', row['arm'], row['target_env'], row['seed'], cm, f1, precision, recall, row['test_metrics']['f1_macro'])
    print('H-S: 18 sealed runs and 54000 predictions verified', flush=True)


def tables(runs):
    summaries = []
    for study, arm in sorted({(r['study'], r['arm']) for r in runs}):
        row = dict(study=study, arm=arm)
        for env in ('E1', 'E2', 'E3'):
            values = [r['macro_f1_percent'] for r in runs if (r['study'], r['arm'], r['environment']) == (study, arm, env)]
            assert len(values) == 3
            row[env] = float(np.mean(values))
            row[env + '_seed_sd'] = float(np.std(values, ddof=1))
        row['equal_environment_mean'] = float(np.mean([row[e] for e in ('E1', 'E2', 'E3')]))
        summaries.append(row)
    save_csv(OUT / 'recognition_summary.csv', summaries)
    hclasses, yclasses = [], []
    for i, name in enumerate(CLASSES):
        hmeans = {a: float(np.mean([r[f'C{i}_f1_percent'] for r in runs if r['study'] == 'H' and r['arm'] == a])) for a in ('H00', 'H10', 'H11')}
        hclasses.append(dict(class_id=i, activity=name, **hmeans, tpp_ce_delta_pp=hmeans['H10']-hmeans['H00'], contrastive_tpp_delta_pp=hmeans['H11']-hmeans['H10']))
        yrow = dict(class_id=i, activity=name)
        for metric in ('f1', 'precision', 'recall'):
            means = {a: float(np.mean([r[f'C{i}_{metric}_percent'] for r in runs if (r['study'], r['arm'], r['environment']) == ('Y', a, 'E1')])) for a in ('Y10', 'Y11')}
            yrow.update({f'{a}_{metric}': v for a, v in means.items()})
            yrow[f'{metric}_delta_pp'] = means['Y11'] - means['Y10']
        yclasses.append(yrow)
    save_csv(OUT / 'H00_H10_H11_classwise.csv', hclasses)
    save_csv(OUT / 'Y_E1_classwise.csv', yclasses)
    comparisons = [('core', 'R10', 'core', 'R00'), ('core', 'R11', 'core', 'R10'),
                   ('core', 'R11', 'core', 'R01'), ('controls', 'MASK_ALL', 'controls', 'MASK_HYBRID'),
                   ('controls', 'MASK_REMOVE', 'controls', 'MASK_HYBRID'),
                   ('controls', 'MASK_ALL', 'controls', 'MASK_REMOVE'),
                   ('inputs', 'TWIN_AP', 'inputs', 'TWIN_AA'),
                   ('inputs', 'TWIN_AA', 'inputs', 'NATIVE_LINEAR'),
                   ('inputs', 'NATIVE_LINEAR', 'core', 'R10'), ('core', 'R10', 'controls', 'P1'),
                   ('H-S', 'H11', 'H-S', 'H10'), ('H', 'H11', 'H', 'H10'), ('Y', 'Y11', 'Y', 'Y10')]
    lookup = {(r['study'], r['arm'], r['environment'], r['seed']): r for r in runs}
    effects = []
    for s1, a1, s0, a0 in comparisons:
        for r in runs:
            if (r['study'], r['arm']) == (s1, a1):
                base = lookup[(s0, a0, r['environment'], r['seed'])]
                effects.append(dict(comparison=f'{s1}:{a1}-{s0}:{a0}', environment=r['environment'], seed=r['seed'],
                                    delta_pp=r['macro_f1_percent']-base['macro_f1_percent']))
    for r in runs:
        if (r['study'], r['arm']) == ('core', 'R11'):
            vals = {a: lookup[('core', a, r['environment'], r['seed'])]['macro_f1_percent'] for a in ('R00','R01','R10','R11')}
            effects.append(dict(comparison='core:interaction', environment=r['environment'], seed=r['seed'], delta_pp=vals['R11']-vals['R10']-vals['R01']+vals['R00']))
    save_csv(OUT / 'paired_differences.csv', effects)
    effect_summary = []
    for comparison in sorted({r['comparison'] for r in effects}):
        subset = [r for r in effects if r['comparison'] == comparison]
        means = {env: float(np.mean([r['delta_pp'] for r in subset if r['environment'] == env])) for env in ('E1', 'E2', 'E3')}
        effect_summary.append(dict(comparison=comparison, **means, equal_environment_mean=float(np.mean(list(means.values()))),
                                   positive_performance_differences=sum(r['delta_pp'] > 0 for r in subset), paired_cells=len(subset)))
    save_csv(OUT / 'paired_summary.csv', effect_summary)
    for filename, rows, cols, caption, label in [
        ('h_classwise.tex', hclasses, ['H00','H10','H11','tpp_ce_delta_pp','contrastive_tpp_delta_pp'],
         'Historical H class F1, averaged within each target over three seeds and then equally over targets. Differences separate pooling under CE from the contrastive increment.', 'tab:hclasses_review'),
        ('y_e1_classwise.tex', yclasses, ['Y10_f1','Y11_f1','f1_delta_pp','Y10_recall','Y11_recall'],
         'Y in E1: class F1 and recall averaged over the three matched seeds. The difference is Y11 minus Y10.', 'tab:ye1classes')]:
        headers = ['H00','H10','H11',r'$\Delta$ TPP',r'$\Delta$ contrastive'] if filename.startswith('h_') else ['Y10 F1','Y11 F1',r'$\Delta$ F1','Y10 recall','Y11 recall']
        text = [r'\begin{table}[!htbp]',f'\\caption{{{caption}\\label{{{label}}}}}',r'\centering\small',r'\begin{tabular}{lrrrrr}',r'\toprule', 'Activity & '+' & '.join(headers)+r' \\', r'\midrule']
        for row in rows:
            text.append(row['activity']+' & '+' & '.join(f'{row[k]:.2f}' for k in cols)+r' \\')
        text += [r'\bottomrule',r'\end{tabular}',r'\end{table}', '']
        (ASSETS / filename).write_text('\n'.join(text), encoding='utf-8')
    return summaries, effect_summary, hclasses, yclasses


def plot_confusions(matrices, output_stem='y_e1_confusions_readable'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    totals = {a: np.sum(values, axis=0) for a, values in matrices.items()}
    normalized = {a: 100 * cm / cm.sum(1, keepdims=True) for a, cm in totals.items()}
    delta = normalized['Y11'] - normalized['Y10']
    for a, cm in totals.items():
        np.savetxt(OUT / f'E1_{a}_pooled_confusion.csv', cm, delimiter=',', fmt='%d')
    np.savetxt(OUT / 'E1_Y11_minus_Y10_row_normalized_pp.csv', delta, delimiter=',', fmt='%.12f')
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    fig, grid_axes = plt.subplots(2, 2, figsize=(7.2, 6.6), layout='constrained')
    axes = [grid_axes[0,0],grid_axes[0,1],grid_axes[1,0]]
    grid_axes[1,1].axis('off')
    grid_axes[1,1].text(.08,.90,'Class labels',fontsize=11,weight='bold',va='top')
    grid_axes[1,1].text(.08,.78,'\n'.join(f'C{i}  {name}' for i,name in enumerate(CLASSES)),fontsize=10,va='top',linespacing=1.7)
    grid_axes[1,1].text(.08,.18,'E1, three seeds\nRows: true class; columns: prediction\nDifference: percentage points',fontsize=9,va='top',linespacing=1.5)
    for ax, data, title in zip(axes, [normalized['Y10'], normalized['Y11'], delta], ['Y10: TPP + CE', 'Y11: TPP + SupCon', 'Y11 - Y10 (pp)']):
        is_delta = ax is axes[2]
        limit = float(np.max(np.abs(delta))) if is_delta else 100
        im = ax.imshow(data, cmap='RdBu_r' if is_delta else 'Blues', vmin=-limit if is_delta else 0, vmax=limit)
        for y in range(6):
            for x in range(6):
                value = data[y, x]
                ax.text(x, y, f'{value:+.1f}' if is_delta else f'{value:.1f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(value) > limit * .58 else 'black')
        ax.set(xticks=range(6), yticks=range(6), xticklabels=[f'C{i}' for i in range(6)], yticklabels=[f'C{i}' for i in range(6)],
               xlabel='Predicted class', ylabel='True class', title=title)
        fig.colorbar(im, ax=ax, shrink=.8)
    fig.savefig(ASSETS / (output_stem+'.png'), dpi=180)
    fig.savefig(ASSETS / (output_stem+'.pdf'))
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    runs = []
    verified = formal_runs(runs)
    matrices = historical_runs(runs)
    h_grouped(runs)
    assert len(runs) == 270
    save_csv(OUT / 'recomputed_runs.csv', runs)
    summaries, effects, hclasses, yclasses = tables(runs)
    plot_confusions(matrices)
    manifest = dict(status='passed', run_count=270, prediction_rows_recomputed=810000,
                    new_training_performed=False, new_target_inference_performed=False,
                    formal_studies=verified, h_grouped_sealed_runs=18,
                    historical_prediction_runs=72, metric_aggregation='per-run F1; seed mean per environment; equal environment mean',
                    population_inference=False, source_sha256=SOURCES,
                    analysis_script_sha256=sha256(Path(__file__)),
                    output_sha256={str(p):sha256(p) for directory in (OUT, ASSETS) for p in directory.iterdir()
                                   if p.suffix in ('.csv', '.tex', '.png', '.pdf')})
    write(OUT / 'evidence_manifest.json', manifest)
    print(json.dumps({'status':'passed', 'runs':len(runs), 'prediction_rows':810000, 'paired_effects':effects,
                      'H_classes':hclasses, 'Y_E1_classes':yclasses}, indent=2))


if __name__ == '__main__':
    main()
