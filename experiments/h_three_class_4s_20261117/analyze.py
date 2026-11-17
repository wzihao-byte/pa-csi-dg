"""Summaries in the format of Tables I and II, for three-class macro-F1.

Pre-specified criterion (frozen with the bundle, before any target score):
a gain "persists" when its paired mean is > 0 AND at least 7 of 9 target-seed
pairs improve. Every comparison is reported whatever the outcome.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import three_class as tc
from common import ARMS, OUTPUT, TARGETS, SEEDS, now, read, write

COMPARISONS = [('T11', 'T10', 'multiview SupCon on TPP'), ('G11', 'G10', 'multiview SupCon on global'),
               ('T10', 'G10', 'TPP under CE'), ('T11', 'G11', 'TPP under multiview SupCon')]
CRITERION = {'paired_mean_gt': 0.0, 'min_improved_of_9': 7,
             'primary': ['T11-T10', 'T10-G10']}


def analyze(completed):
    runs = {}
    for item in completed:
        m = read(Path(item['attempt']) / 'payload/run/metrics.json')
        arm = m['factorial']['arm'].split('_', 1)[1]
        runs[(m['target_env'], int(m['seed']), arm)] = {
            'macro_f1': 100 * float(m['test_metrics']['f1_macro']),
            'class_f1': {tc.CLASS_NAMES[r['class_id']]: 100 * float(r['f1']) for r in m['per_class_metrics']}}
    arms = {}
    for a, (name, _, _) in ARMS.items():
        per_target = {}
        for t in TARGETS:
            vals = [runs[(t, s, a)]['macro_f1'] for s in SEEDS if (t, s, a) in runs]
            if vals:
                per_target[t] = {'mean': float(np.mean(vals)), 'sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else None, 'n': len(vals)}
        complete = len(per_target) == 3 and all(v['n'] == 3 for v in per_target.values())
        arms[a] = {'paper_name': name, 'per_target': per_target, 'complete': complete,
                   'mean': float(np.mean([v['mean'] for v in per_target.values()])) if complete else None}
    comparisons = {}
    for a, b, label in COMPARISONS:
        pairs = [(t, s) for t in TARGETS for s in SEEDS if (t, s, a) in runs and (t, s, b) in runs]
        diffs = [runs[(t, s, a)]['macro_f1'] - runs[(t, s, b)]['macro_f1'] for t, s in pairs]
        key = f'{a}-{b}'
        entry = {'label': label, 'pairs': len(pairs), 'complete': len(pairs) == 9}
        if diffs:
            entry.update(paired_mean=float(np.mean(diffs)), improved=int(sum(d > 0 for d in diffs)),
                         per_pair={f'{t}_{s}': float(d) for (t, s), d in zip(pairs, diffs)})
            if len(pairs) == 9:
                entry['persists_by_criterion'] = bool(entry['paired_mean'] > CRITERION['paired_mean_gt']
                                                      and entry['improved'] >= CRITERION['min_improved_of_9'])
        comparisons[key] = entry
    write(OUTPUT / 'analysis/results.json', {'at': now(), 'status': 'complete' if len(runs) == 36 else 'partial',
        'runs_completed': len(runs), 'criterion': CRITERION, 'arms': arms, 'comparisons': comparisons,
        'runs': {f'{t}_{s}_{a}': v for (t, s, a), v in sorted(runs.items())}})


if __name__ == '__main__':
    analyze(read(OUTPUT / 'queue/completed_jobs.json')['completed'])
