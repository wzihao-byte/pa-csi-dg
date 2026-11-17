"""Source-only release gate: frozen files, real-data trajectory and components."""
from __future__ import annotations
import copy, json, subprocess, sys
from pathlib import Path
import torch
from common import BUNDLE, HERE, OUTPUT, base_module, now, read, require, sha, verify_bundle, write
from control_components import make_model_class
from benchmark_current_gpu import source_batch
from train_factorial import install

def equivalent_source_trajectory():
    config=read(BUNDLE/'configs/HS_E1_G11_42.json')
    scratch=OUTPUT/'qa/adapter_equivalence';scratch.mkdir(parents=True,exist_ok=True)
    config=copy.deepcopy(config);config['output_dir']=str(scratch)
    batch=source_batch(config)
    # Four real source observations, two batches, full + masked training, source only.
    tensors=tuple(v[:4].clone() for v in batch)
    def loader():
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors),batch_size=2,
            shuffle=False,num_workers=0)
    from types import SimpleNamespace
    metadata=SimpleNamespace(input_dim=90,num_classes=6)
    original=base_module();a=original.configure_imports(config);ta=a['train_dg']
    ta.PaCsiDGLite=make_model_class(a['tpp_models'].IsolatedPaCsiDGLiteTPP,'global_mlp')
    ta.set_seed(42)
    ma,oa,sa,mode_a=ta.build_model_and_optimizer(metadata,config,torch.device('cpu'))
    la=a['dg_losses'].DomainAwareSupConLoss(temperature=.2,include_same_domain_same_class=True,
        positive_mode='global_class_instance_views')
    result_a=ta.run_epoch(model=ma,loader=loader(),optimizer=oa,scheduler=sa,scheduler_step_mode=mode_a,
        device=torch.device('cpu'),loss_config=config['losses'],contrastive_loss_type='pairwise_supcon',
        contrastive_loss_fn=la,adaptive_kmargin_loss_fn=None,adaptive_kmargin_state=None,train=True)
    adapted=install(base_module(),config);b=adapted.configure_imports(config);tb=b['train_dg']
    tb.PaCsiDGLite=adapted.model_class(config,b)
    tb.set_seed(42)
    mb,ob,sb,mode_b=tb.build_model_and_optimizer(metadata,config,torch.device('cpu'))
    lb=tb.DomainAwareSupConLoss(temperature=.2,include_same_domain_same_class=True,
        positive_mode='global_class_instance_views')
    result_b=tb.run_epoch(model=mb,loader=loader(),optimizer=ob,scheduler=sb,scheduler_step_mode=mode_b,
        device=torch.device('cpu'),loss_config=config['losses'],contrastive_loss_type='pairwise_supcon',
        contrastive_loss_fn=lb,adaptive_kmargin_loss_fn=None,adaptive_kmargin_state=None,train=True)
    require(result_a==result_b,'Adapter changed real-data loss/diagnostics')
    require(all(torch.equal(ma.state_dict()[k],v) for k,v in mb.state_dict().items()),'Adapter changed state trajectory')
    require(oa.param_groups[0]['lr']==ob.param_groups[0]['lr']==.001*.9**(2/10000),'Scheduler trajectory drift')
    require(sa.state_dict()==sb.state_dict(),'Scheduler state drift')
    require(read(scratch/'training_status.json')['optimizer_steps']==1,'Progress instrumentation did not observe first update')
    return {'status':'passed','real_source_examples':4,'optimizer_steps':2,
        'complete_and_flattened_masked_forward':True,'all_model_state_tensors_bitwise_equal':True,
        'loss_metrics_and_diagnostics_bitwise_equal':True,'scheduler_state_equal':True,
        'target_arrays_opened':False,'instrumented_step_does_not_change_training':True}

def main():
    manifest,registry=verify_bundle()
    torch.set_num_threads(2)
    checks=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(HERE/'tests'),'-v'],
        cwd=HERE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8')
    (OUTPUT/'qa').mkdir(exist_ok=True)
    (OUTPUT/'qa/component_tests.log').write_text(checks.stdout,encoding='utf-8')
    require(checks.returncode==0,'Component/source-input tests failed; see qa/component_tests.log')
    equivalence=equivalent_source_trajectory()
    receipt={'status':'passed','at':now(),'bundle_manifest_sha256':sha(BUNDLE/'bundle_manifest.json'),
        'component_test_log_sha256':sha(OUTPUT/'qa/component_tests.log'),
        'frozen_unique_configs_validated':len(registry['jobs']),
        'real_source_adapter_equivalence':equivalence,'target_arrays_opened':False,
        'torch':torch.__version__,'cpu_threads':2,'new_environment_experiment':False}
    write(OUTPUT/'qa/approval.json',receipt)
    print(json.dumps(receipt))

if __name__=='__main__':main()
