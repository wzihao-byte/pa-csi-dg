"""Measure a completed source-selected run without opening any target signal."""
from __future__ import annotations

import argparse
import copy
import shutil
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.revision_validation.experiment_data import VerifiedInputs
from experiments.revision_validation.experiment_design import build_experiment_model, objective, predict
from experiments.revision_validation.freeze_inputs import INPUT_CODE
from experiments.revision_validation.mechanism_metrics import ResidualObserver, environment_probe, feature_scatter, fixed_probe_positions, gradient_alignment
from experiments.revision_validation.models import ARMS
from experiments.revision_validation.source_data import epoch_schedule
from experiments.revision_validation.study_engine import runtime_settings
from experiments.revision_validation.study_io import now, read, sha256, value_hash, verify_completed, write


def canonical_config(config):
    result=copy.deepcopy(config)
    if 'experiment' not in result:
        result.update(experiment={'base_arm':config['arm'],'pool':'unchanged',
                                 'objective':'supcon' if ARMS[config['arm']]['supcon'] else 'ce'},
                      input_modalities=['amp'],primary_input='amp')
    return result


def collect(model,dataset,config,part,positions):
    model.eval();observer=ResidualObserver(model)
    rows=[];features=[];logits=[];projections=[]
    use_projection=config['experiment']['objective'] in ('supcon','mask')
    try:
        with torch.no_grad():
            for start in range(0,len(positions),32):
                inputs,labels,batch=dataset.batch(part,positions[start:start+32],config['device'])
                values=predict(model,inputs,config,project=use_projection)
                rows.extend(batch);features.append(values['feature'].cpu().numpy());logits.append(values['logits'].cpu().numpy())
                if use_projection:projections.append(values['projection'].cpu().numpy())
    finally:observer.close()
    norms={path:{key:np.concatenate([record[key] for record in records]) for key in ('global_norm','residual_norm','ratio','near_zero_global')}
           for path,records in observer.records.items()}
    arrays={'feature':np.concatenate(features),'logits':np.concatenate(logits)}
    if use_projection:arrays['projection']=np.concatenate(projections)
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise FloatingPointError('Nonfinite extracted source representation')
    return rows,arrays,norms


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--qa-only',action='store_true',help='Permit an explicitly labeled short completed QA run')
    args=parser.parse_args();run,output=args.run.resolve(),args.output.resolve()
    # Completion is appended after final status/artifacts. An active incomplete
    # run must never have its mutable status JSON opened by this observer.
    events=(run/'events.jsonl').read_text(encoding='utf-8').splitlines()
    if not any('"event": "completed"' in event for event in events):
        raise RuntimeError('Run has no appended completion event; do not inspect mutable training state')
    original=read(run/'config.json');status=verify_completed(run,original)
    if original['epochs']!=200 and not args.qa_only:
        raise RuntimeError('A short QA cannot be reported as a full trained-model analysis')
    if output.exists():raise FileExistsError('Preserve previous source analysis; use a new output')
    output.mkdir(parents=True)
    dependencies=INPUT_CODE+['experiments/revision_validation/mechanism_metrics.py','analysis/analyze_source_representation.py']
    code_hashes={name:sha256(ROOT/name) for name in dependencies}
    for name in dependencies:
        destination=output/'code'/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,destination)
        if sha256(destination)!=code_hashes[name]:raise RuntimeError('Analysis code changed while snapshotting')
    config=canonical_config(original);config['device']='cpu'
    runtime=runtime_settings();dataset=VerifiedInputs(config['source_manifest_path'],config['input_modalities'])
    weights=torch.load(run/'best_model.pt',map_location='cpu',weights_only=True)
    model=build_experiment_model(config);model.load_state_dict(weights,strict=True)
    probe_positions=fixed_probe_positions(dataset.manifest['train'])
    train_rows,train_arrays,train_norms=collect(model,dataset,config,'train',probe_positions)
    val_rows,val_arrays,val_norms=collect(model,dataset,config,'val',list(range(len(dataset.manifest['val']))))
    saved=read(run/'selected_source_predictions.json')
    for row,predicted in zip(val_rows,saved):
        if any(row[k]!=predicted[k] for k in ('sample_id','subject_id','acquisition_id')) or row['label']!=predicted['true']:
            raise RuntimeError('Source extraction order differs from saved prediction identity')
    predicted=val_arrays['logits'].argmax(1)
    prediction_differences=int(sum(value!=reference['predicted'] for value,reference in zip(predicted,saved)))
    labels=np.array([r['label'] for r in val_rows]);environments=np.array([r['environment'] for r in val_rows])
    scatters={key:feature_scatter(value,labels,environments) for key,value in val_arrays.items() if key in ('feature','projection')}
    probe,probe_predictions=environment_probe(train_arrays['feature'],train_rows,val_arrays['feature'],val_rows)
    if not probe['converged']:raise RuntimeError('Fixed source environment probe did not converge')
    residual_summary={}
    for path,values in val_norms.items():
        good=~values['near_zero_global'];ratios=values['ratio']
        residual_summary[path]={'rows':len(ratios),'near_zero_global_rows':int((~good).sum()),
            'all_ratios_median':float(np.median(ratios)),'all_ratios_p95':float(np.quantile(ratios,.95)),
            'nonzero_global_ratios_median':float(np.median(ratios[good])) if good.any() else None,
            'nonzero_global_ratios_p95':float(np.quantile(ratios[good],.95)) if good.any() else None}
    gradients=[]
    if config['experiment']['objective'] in ('supcon','mask','coral'):
        batches=epoch_schedule(dataset.manifest['train'],9664,0,steps=4)
        for batch_index,batch in enumerate(batches):
            model=build_experiment_model(config);model.load_state_dict(weights,strict=True);model.train()
            torch.manual_seed(9665+batch_index)
            inputs,targets,rows=dataset.batch('train',batch,'cpu')
            step=status['best_epoch']*config['updates_per_epoch']
            total,primary,regularizer,weight,diagnostics=objective(model,inputs,targets,rows,step,config)
            values=gradient_alignment(model,primary,regularizer,weight)
            gradients.append({'batch_index':batch_index,'sample_ids':[r['sample_id'] for r in rows],
                'source_selected_epoch':status['best_epoch'],'effective_schedule_step':step,
                'primary':float(primary.detach()),'regularizer':float(regularizer.detach()),'weight':weight,
                'regularizer_family':config['experiment']['objective'],'alignment':values,'objective_diagnostics':diagnostics})
    arrays={f'probe_train_{key}':value for key,value in train_arrays.items()}
    arrays.update({f'validation_{key}':value for key,value in val_arrays.items()})
    for part,parts in (('probe_train',train_norms),('validation',val_norms)):
        for path,values in parts.items():
            arrays.update({f'{part}__{path}__{key}':value for key,value in values.items()})
    np.savez_compressed(output/'source_features_and_norms.npz',**arrays)
    write(output/'selection.json',{'probe_train':train_rows,'validation':val_rows,'training_selection_rule':'32 per source environment/class, stable sample-ID hash order',
                                  'environment_probe_predictions':probe_predictions})
    write(output/'gradient_probes.json',gradients)
    report={'completed_at':now(),'qa_only':args.qa_only,'full_trained_model_analysis':original['epochs']==200 and not args.qa_only,
        'job_id':original['job_id'],'run_directory':str(run),'original_config_value_sha256':value_hash(original),
        'source_status_sha256':sha256(run/'status.json'),'source_manifest_sha256':sha256(config['source_manifest_path']),
        'checkpoint_sha256':sha256(run/'best_model.pt'),'selected_epoch':status['best_epoch'],
        'analysis_device':'cpu','runtime':runtime,'sklearn_version':sklearn.__version__,
        'target_data_loaded':False,'opened_source_paths':dataset.opened_paths,'source_prediction_differences_from_recorded_backend':prediction_differences,
        'scatter':scatters,'environment_probe':probe,'residual_summary':residual_summary,'gradient_probe_batches':len(gradients),
        'gradient_scope':'Four fixed balanced source-training batches on the selected checkpoint; post-hoc train-mode probes with fresh BN state and fixed dropout seeds, not recorded training gradients',
        'interpretation':'Descriptive source geometry and finite-input perturbation diagnostics cannot by themselves establish target mechanisms or population effects',
        'code_sha256':code_hashes,'artifacts_sha256':{p.name:sha256(p) for p in output.iterdir() if p.is_file()}}
    write(output/'summary.json',report)
    if sha256(run/'best_model.pt')!=report['checkpoint_sha256']:raise RuntimeError('Source checkpoint changed during analysis')
    print({'job_id':original['job_id'],'qa_only':args.qa_only,'source_prediction_differences':prediction_differences,
           'environment_probe_balanced_accuracy':probe['balanced_accuracy'],'gradient_batches':len(gradients)})


if __name__=='__main__':main()
