"""Check analytic variance/gradient cases and actual residual observation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.revision_validation.experiment_data import VerifiedInputs
from experiments.revision_validation.mechanism_metrics import (
    ResidualObserver, environment_probe, fixed_probe_positions, gradient_alignment, scatter_statistics,
)
from experiments.revision_validation.models import build_model
from experiments.revision_validation.source_data import epoch_schedule
from experiments.revision_validation.study_engine import equal_state, runtime_settings
from experiments.revision_validation.study_io import now, sha256, write
from experiments.tpp_isolated.tpp_models import IsolatedTemporalPyramidPool


class ToyPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_pool=IsolatedTemporalPyramidPool(1,1,[2,3],[1,2,4],dropout=0,preserve_global=True,pyramid_scale=.25)
        with torch.no_grad():
            for layer in self.temporal_pool.convs:
                layer.weight.fill_(1);layer.bias.zero_()
            self.temporal_pool.projection.weight.fill_(.1);self.temporal_pool.projection.bias.fill_(.2)
    def forward(self,x):return self.temporal_pool(x)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Keep previous mechanism QA evidence')
    runtime_settings()
    x=np.array([[-1.,-2.],[-1.,2.],[1.,-2.],[1.,2.]])
    labels=np.array([0,0,1,1]);domains=np.array([0,1,0,1])
    values=scatter_statistics(x,labels,domains)
    assert values['between_class_scatter']==1 and values['within_class_scatter']==4 and values['total_scatter']==5
    assert values['environment_scatter_within_class']==4 and values['residual_scatter_within_environment_class']==0
    assert values['equal_class_mean_domain_squared_distance']==16 and values['between_to_within_class_ratio']==.25
    translated=scatter_statistics(x+[123.,-76.],labels,domains)
    assert translated==values
    scaled=scatter_statistics(3*x,labels,domains)
    assert scaled['total_scatter']==45 and scaled['between_to_within_class_ratio']==.25
    zero=scatter_statistics(np.zeros_like(x),labels,domains)
    assert zero['within_class_near_zero'] and zero['between_to_within_class_ratio'] is None
    toy=ToyPool().eval();observer=ResidualObserver(toy)
    with torch.no_grad():toy(torch.arange(8,dtype=torch.float32).reshape(1,8,1))
    observed=observer.records['temporal_pool'][0];observer.close()
    # Independent hand calculation: branch maxima [13,18]; all fourteen bin
    # maxima sum to 160; each projected value=.1*160+.2=16.2, residual=4.05.
    expected_global=np.sqrt(493.);expected_residual=np.sqrt(2.)*4.05
    assert abs(observed['global_norm'][0]-expected_global)<1e-8
    assert abs(observed['residual_norm'][0]-expected_residual)<1e-6
    assert abs(observed['ratio'][0]-expected_residual/(expected_global+1e-8))<1e-7
    linear=nn.Linear(2,1)
    primary=sum(p.square().sum() for p in linear.parameters())
    alignment=gradient_alignment(linear,primary,-primary,.25)['all_shared']
    assert abs(alignment['cosine']+1)<1e-12 and abs(alignment['weighted_regularizer_to_primary_norm_ratio']-.25)<1e-12
    rng=np.random.default_rng(9661)
    def artificial(prefix):
        rows=[];features=[]
        for env in range(2):
            for label in range(6):
                for index in range(12):
                    rows.append({'sample_id':f'{prefix}:{env}:{label}:{index}','subject_id':f'{prefix}:subject{env}',
                        'acquisition_id':f'{prefix}:acq{env}:{label}:{index}','environment':f'E{env}','label':label})
                    features.append(np.r_[np.eye(6)[label]*5,10*env-5]+rng.normal(0,.01,7))
        return np.array(features),rows
    train,train_rows=artificial('train');val,val_rows=artificial('val')
    probe,_=environment_probe(train,train_rows,val,val_rows)
    assert probe['converged'] and probe['balanced_accuracy']==1
    bad=[dict(r) for r in val_rows];bad[0]['subject_id']=train_rows[0]['subject_id']
    overlap_rejected=False
    try:environment_probe(train,train_rows,val,bad)
    except ValueError as error:overlap_rejected='subject_id' in str(error)
    assert overlap_rejected
    source=ROOT/'experiments/revision_validation/verified_protocol/source_E1.json'
    dataset=VerifiedInputs(source)
    probe_positions=fixed_probe_positions(dataset.manifest['train'])
    assert len(probe_positions)==len(set(probe_positions))==384
    batch=epoch_schedule(dataset.manifest['train'],401,0,steps=1)[0]
    inputs,_,rows=dataset.batch('train',batch,'cpu')
    actual={}
    for arm in ('R00','R10'):
        model=build_model(arm,9662).eval();before={k:v.clone() for k,v in model.state_dict().items()}
        observer=ResidualObserver(model)
        with torch.no_grad():model(inputs['amp'],project=False)
        observer.close()
        assert equal_state(before,model.state_dict())
        actual[arm]={path:{'rows':len(records[0]['ratio']),
                          'forward_expression_max_abs_error':records[0]['forward_expression_max_abs_error'],
                          'nonfinite_ratio_count':int((~np.isfinite(records[0]['ratio'])).sum())} for path,records in observer.records.items()}
        assert all(v['nonfinite_ratio_count']==0 for v in actual[arm].values())
    report={'passed':True,'checked_at':now(),'analytic_scatter':values,'translation_and_scale_checks':True,
        'zero_scatter_ratio_undefined':True,'toy_pool_hand_calculation':{'global_norm':expected_global,'residual_norm':expected_residual},
        'opposite_gradient_check':alignment,'synthetic_environment_probe':probe,'probe_overlap_rejected':overlap_rejected,
        'actual_source_residual_observation':actual,'actual_model_state_unchanged':True,'neural_optimizer_updates':0,
        'linear_probe_fit_scope':'synthetic QA only','target_data_loaded':False,'trained_model_mechanism_results':False,
        'source_sample_ids':[r['sample_id'] for r in rows],'probe_training_sample_ids':[dataset.manifest['train'][i]['sample_id'] for i in probe_positions],
        'sklearn_version':sklearn.__version__,'code_sha256':{name:sha256(ROOT/name) for name in (
            'analysis/verify_mechanism_metrics.py','experiments/revision_validation/mechanism_metrics.py',
            'experiments/revision_validation/models.py','experiments/tpp_isolated/tpp_models.py')}}
    write(args.output,report)
    print({'passed':True,'actual_source_observation':actual,'sklearn_version':sklearn.__version__,'neural_optimizer_updates':0})


if __name__=='__main__':main()
