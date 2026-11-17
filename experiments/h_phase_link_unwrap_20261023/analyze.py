"""Aggregate only validated runs against the frozen paired arms; never tune on target scores."""
from __future__ import annotations
import csv
from pathlib import Path
import numpy as np
from common import BUNDLE, JOBS, OUTPUT, PAIRED_BUNDLE, PAIRED_STUDY, SEEDS, TARGETS, now, read, require, sha, write

def paired_scores():
    scores = {}
    for record in read(PAIRED_STUDY / 'queue/completed_jobs.json')['completed']:
        run = Path(record['attempt']) / 'payload/run'
        require(sha(run / 'metrics.json') == record['health']['metrics_sha256'], 'Paired metrics changed')
        config = read(Path(record['attempt']) / 'job_config.json')
        arm = config['h_controls']['arm']
        if arm in ('N_H11', 'N_D11'):
            metrics = read(run / 'metrics.json')
            scores[(metrics['target_env'], metrics['seed'], arm)] = 100. * metrics['test_metrics']['f1_macro']
    for r in read(PAIRED_BUNDLE / 'historical_references.json')['references']:
        if r['arm'] != 'H11':
            continue
        run = Path(r['attempt']) / 'payload/run'
        require(sha(run / 'metrics.json') == r['metrics_sha256'], 'Historical reference metrics changed')
        scores[(r['target'], r['seed'], 'H11')] = 100. * read(run / 'metrics.json')['test_metrics']['f1_macro']
    require(len(scores) == 27, 'Need all nine paired N_H11, N_D11 and H11 runs')
    return scores

def bootstrap_interval(values, seed=20261023, resamples=10000):
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
        rows.append({'target': key[0], 'seed': key[1], 'arm': key[2], 'macro_f1_pct': score,
            'accuracy_pct': 100. * metrics['test_metrics']['accuracy'], 'run': str(run),
            'paired_job_id': config['h_controls']['paired_control_job_id'],
            'metrics_sha256': sha(run / 'metrics.json'), 'predictions_sha256': sha(run / 'target_predictions.jsonl')})
    summaries = {}
    for arm in ('N_L11', 'N_H11', 'N_D11', 'H11'):
        cells = {e: [scores[(e, s, arm)] for s in SEEDS if (e, s, arm) in scores] for e in TARGETS}
        complete = all(len(x) == 3 for x in cells.values())
        summaries[arm] = {'status': 'complete' if complete else 'partial', 'completed_runs': sum(map(len, cells.values())),
            'environments': {e: {'n': len(v), 'mean': float(np.mean(v)) if v else None,
                'sample_sd': float(np.std(v, ddof=1)) if len(v) > 1 else None} for e, v in cells.items()},
            'equal_environment_mean': float(np.mean([np.mean(v) for v in cells.values()])) if complete else None}
    seed_sd = float(np.mean([summaries['N_H11']['environments'][e]['sample_sd'] for e in TARGETS]))
    contrasts = {}
    for name, candidate, control in (('link_unwrap_minus_cross_link', 'N_L11', 'N_H11'),
                                     ('link_unwrap_minus_historical', 'N_L11', 'H11'),
                                     ('link_unwrap_minus_native_differential', 'N_L11', 'N_D11')):
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
    primary = contrasts['link_unwrap_minus_cross_link']
    outcome = None
    if primary['status'] == 'complete':
        low, high = primary['bootstrap_95_interval_pp']
        if low <= 0 <= high and abs(primary['paired_mean_pp']) <= seed_sd:
            outcome = 'unchanged'
        elif high < 0:
            outcome = 'drop'
        else:
            outcome = 'inconclusive'
    folder = OUTPUT / 'analysis'; folder.mkdir(exist_ok=True)
    if rows:
        with (folder / 'validated_runs.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write(folder / 'results.json', {'at': now(), 'status': 'complete' if len(rows) == JOBS else 'partial',
        'completed_new_runs': len(rows), 'total_new_runs': JOBS, 'summaries': summaries, 'contrasts': contrasts,
        'n_h11_mean_within_environment_seed_sd_pp': seed_sd,
        'decision_rule': read(BUNDLE / 'decision_rule.json'), 'prespecified_outcome': outcome,
        'partition': 'H-S', 'only_phase_arrays_differ_from_paired_n_h11': True,
        'target_metric_tuning': False, 'new_environment_experiment': False})
