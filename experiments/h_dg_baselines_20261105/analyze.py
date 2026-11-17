"""Aggregate only validated runs against the frozen paired arms; never tune on target scores."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from common import ARMS, BUNDLE, JOBS, OUTPUT, PAIRED_STUDY, SEEDS, TARGETS, now, read, require, sha, write

PAIRED_ARMS = ('H10', 'H11', 'G10', 'G11')

def paired_scores():
    scores = {}
    for record in read(PAIRED_STUDY / 'queue/completed_jobs.json')['completed']:
        run = Path(record['attempt']) / 'payload/run'
        require(sha(run / 'metrics.json') == record['health']['metrics_sha256'], 'Paired metrics changed')
        arm = read(Path(record['attempt']) / 'job_config.json')['h_controls']['arm']
        if arm in ('G10', 'G11'):
            metrics = read(run / 'metrics.json')
            scores[(metrics['target_env'], metrics['seed'], arm)] = 100. * metrics['test_metrics']['f1_macro']
    for r in read(BUNDLE / 'paired_historical_references.json')['references']:
        run = Path(r['attempt']) / 'payload/run'
        require(sha(run / 'metrics.json') == r['metrics_sha256'], 'Historical reference metrics changed')
        scores[(r['target'], r['seed'], r['arm'])] = 100. * read(run / 'metrics.json')['test_metrics']['f1_macro']
    require(len(scores) == 36, 'Need all nine paired H10, H11, G10 and G11 runs')
    return scores

def bootstrap_interval(values, seed=20261105, resamples=10000):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    means = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]

def analyze(completed):
    rows = []; scores = paired_scores()
    for record in completed:
        attempt = Path(record['attempt']); run = attempt / 'payload/run'
        config = read(attempt / 'job_config.json'); metrics = read(run / 'metrics.json')
        key = (metrics['target_env'], metrics['seed'], config['h_controls']['arm'])
        require(key not in scores, 'Duplicate completed cell')
        score = 100. * metrics['test_metrics']['f1_macro']; scores[key] = score
        rows.append({'target': key[0], 'seed': key[1], 'arm': key[2], 'method': config['dg_baseline']['method'],
            'macro_f1_pct': score, 'accuracy_pct': 100. * metrics['test_metrics']['accuracy'], 'run': str(run),
            'pool_mode': config['h_controls']['pool_mode'], 'paired_control_job_id': config['h_controls']['paired_control_job_id'],
            'metrics_sha256': sha(run / 'metrics.json'), 'predictions_sha256': sha(run / 'target_predictions.jsonl')})
    summaries = {}
    for arm in ARMS + PAIRED_ARMS:
        cells = {e: [scores[(e, s, arm)] for s in SEEDS if (e, s, arm) in scores] for e in TARGETS}
        complete = all(len(x) == 3 for x in cells.values())
        summaries[arm] = {'status': 'complete' if complete else 'partial', 'completed_runs': sum(map(len, cells.values())),
            'environments': {e: {'n': len(v), 'mean': float(np.mean(v)) if v else None,
                'sample_sd': float(np.std(v, ddof=1)) if len(v) > 1 else None} for e, v in cells.items()},
            'equal_environment_mean': float(np.mean([np.mean(v) for v in cells.values()])) if complete else None}
    contrasts = {}
    for name, candidate, control in (('global_coral_minus_gce', 'GC10', 'G10'), ('global_mixup_minus_gce', 'GX10', 'G10'),
                                     ('tsc_minus_global_coral', 'H11', 'GC10'), ('tsc_minus_global_mixup', 'H11', 'GX10'),
                                     ('tce_minus_global_coral', 'H10', 'GC10'), ('tce_minus_global_mixup', 'H10', 'GX10'),
                                     ('tpp_coral_minus_tce', 'C10', 'H10'), ('tpp_mixup_minus_tce', 'X10', 'H10'),
                                     ('tsc_minus_tpp_coral', 'H11', 'C10'), ('tsc_minus_tpp_mixup', 'H11', 'X10'),
                                     ('tpp_coral_minus_global_coral', 'C10', 'GC10'), ('tpp_mixup_minus_global_mixup', 'X10', 'GX10')):
        pairs = [{'target': e, 'seed': s, 'difference_pp': scores[(e, s, candidate)] - scores[(e, s, control)]}
            for e in TARGETS for s in SEEDS if (e, s, candidate) in scores and (e, s, control) in scores]
        result = {'status': 'complete' if len(pairs) == 9 else 'partial', 'pairs': pairs}
        if len(pairs) == 9:
            values = [x['difference_pp'] for x in pairs]
            result.update(paired_mean_pp=float(np.mean(values)),
                environment_means_pp={e: float(np.mean([x['difference_pp'] for x in pairs if x['target'] == e])) for e in TARGETS},
                positive_pairs=sum(v > 0 for v in values), negative_pairs=sum(v < 0 for v in values),
                bootstrap_95_interval_pp=bootstrap_interval(values))
        contrasts[name] = result
    folder = OUTPUT / 'analysis'; folder.mkdir(exist_ok=True)
    if rows:
        with (folder / 'validated_runs.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write(folder / 'results.json', {'at': now(), 'status': 'complete' if len(rows) == JOBS else 'partial',
        'completed_new_runs': len(rows), 'total_new_runs': JOBS, 'summaries': summaries, 'contrasts': contrasts,
        'decision_rule': read(BUNDLE / 'decision_rule.json'), 'partition': 'H-S',
        'only_training_objective_differs_from_paired_backbone_run': True,
        'target_metric_tuning': False, 'new_environment_experiment': False})
