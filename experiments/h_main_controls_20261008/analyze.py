"""Aggregate only validated runs; never select arms from target performance."""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import numpy as np
from common import BUNDLE, OUTPUT, SEEDS, TARGETS, now, read, require, sha, write

def analyze(completed):
    rows=[]; scores={}
    for record in completed:
        attempt=Path(record['attempt']);run=attempt/'payload/run'
        config=read(attempt/'job_config.json');metrics=read(run/'metrics.json')
        c=config['h_controls'];key=(metrics['target_env'],metrics['seed'],c['arm'])
        require(key not in scores,'Duplicate completed cell')
        score=100.*metrics['test_metrics']['f1_macro'];scores[key]=score
        rows.append({'target':key[0],'seed':key[1],'arm':key[2],'macro_f1_pct':score,
            'accuracy_pct':100.*metrics['test_metrics']['accuracy'],'run':str(run),
            'metrics_sha256':sha(run/'metrics.json'),'predictions_sha256':sha(run/'target_predictions.jsonl')})
    reference=read(BUNDLE/'historical_references.json')['references']
    for r in reference:
        run=Path(r['attempt'])/'payload/run'
        require(sha(run/'metrics.json')==r['metrics_sha256'],'Historical reference metrics changed')
        scores[(r['target'],r['seed'],r['arm'])]=100.*read(run/'metrics.json')['test_metrics']['f1_macro']
    summaries={}
    for arm in ('G10','G11','P_ALL','P_REMOVE','N_H11','N_D11'):
        cells={e:[scores[(e,s,arm)] for s in SEEDS if (e,s,arm) in scores] for e in TARGETS}
        complete=all(len(x)==3 for x in cells.values())
        summaries[arm]={'status':'complete' if complete else 'partial','completed_runs':sum(map(len,cells.values())),
            'environments':{e:{'n':len(v),'mean':float(np.mean(v)) if v else None,
                'sample_sd':float(np.std(v,ddof=1)) if len(v)>1 else None} for e,v in cells.items()},
            'equal_environment_mean':float(np.mean([np.mean(v) for v in cells.values()])) if complete else None}
    contrasts={}
    for name,candidate,control in (
        ('capacity_ce_tpp_minus_global','H10','G10'),('capacity_hybrid_tpp_minus_global','H11','G11'),
        ('global_mlp_hybrid_increment','G11','G10'),('all_view_minus_hybrid','P_ALL','H11'),
        ('remove_denominator_minus_hybrid','P_REMOVE','H11'),
        ('native_histstyle_minus_historical','N_H11','H11'),('native_differential_minus_historical','N_D11','H11'),
        ('native_differential_minus_native_histstyle','N_D11','N_H11')):
        pairs=[{'target':e,'seed':s,'difference_pp':scores[(e,s,candidate)]-scores[(e,s,control)]}
            for e in TARGETS for s in SEEDS if (e,s,candidate) in scores and (e,s,control) in scores]
        contrasts[name]={'status':'complete' if len(pairs)==9 else 'partial','pairs':pairs,
            'paired_mean_pp':float(np.mean([x['difference_pp'] for x in pairs])) if len(pairs)==9 else None}
    folder=OUTPUT/'analysis';folder.mkdir(exist_ok=True)
    if rows:
        with (folder/'validated_runs.csv').open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write(folder/'results.json',{'at':now(),'status':'complete' if len(rows)==54 else 'partial',
        'completed_new_runs':len(rows),'total_new_runs':54,'summaries':summaries,'contrasts':contrasts,
        'partition':'H-S','same_native_grid_phase_comparison_is_whole_processing_chain':True,
        'target_metric_tuning':False,'new_environment_experiment':False})
