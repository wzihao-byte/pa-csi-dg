"""Measure source-only complete H steps; never open a target input."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from common import HERE, HISTORICAL, base_module, now, read, write
from control_components import make_model_class

def source_batch(config):
    ids = np.load(HISTORICAL / 'schedules/E1_42.npy', mmap_mode='r')[0, 0].copy()
    amp, phase, labels, domains = [], [], [], []
    sources = list(config['data']['env_files'])
    cache = {e:{k:np.load(p, mmap_mode='r') for k,p in files.items()} for e,files in config['data']['env_files'].items()}
    for sample_id in ids:
        env = ('E1','E2','E3')[int(sample_id)//3000]
        row = int(sample_id)%3000
        amp.append(cache[env]['amp'][row].astype(np.float32))
        phase.append(np.unwrap(cache[env]['phase'][row], axis=-1).astype(np.float32))
        labels.append(int(cache[env]['label'][row])); domains.append(sources.index(env))
    return tuple(torch.from_numpy(v) for v in (np.stack(amp),np.stack(phase),
        np.array(labels,dtype=np.int64),np.array(domains,dtype=np.int64),ids))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--timed-steps',type=int,default=5)
    p.add_argument('--output',type=Path,default=HERE/'current_gpu_benchmark.json')
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    config=read(HISTORICAL/'configs/E1_HG_H11_42.json')
    base=base_module();modules=base.configure_imports(config)
    batch=source_batch(config)
    receipts=[]
    for mode,supcon in (('tpp',False),('tpp',True),('global_mlp',False),('global_mlp',True)):
        modules['train_dg'].set_seed(42)
        model=make_model_class(modules['tpp_models'].IsolatedPaCsiDGLiteTPP,mode)(90,6,config['model']).cuda().train()
        opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=0.)
        criterion=modules['dg_losses'].DomainAwareSupConLoss(temperature=.2,
            include_same_domain_same_class=True,positive_mode='global_class_instance_views')
        times=[];losses=[]
        torch.cuda.reset_peak_memory_stats()
        for step in range(2+args.timed_steps):
            torch.cuda.synchronize();started=time.perf_counter()
            a,ph,y,domain,ids=(v.cuda() for v in batch)
            opt.zero_grad(set_to_none=True)
            out=model(a,ph)
            loss=torch.nn.functional.cross_entropy(out['logits'],y)
            if supcon:
                views=model.encode_antenna_views(a,ph)
                features=torch.cat([out['projection'].unsqueeze(1),views['projection_views']],dim=1)
                loss=loss+.05*criterion(features,labels=y,domains=domain,sample_ids=ids)
            loss.backward();opt.step()
            torch.cuda.synchronize();elapsed=time.perf_counter()-started
            if step>=2:times.append(elapsed);losses.append(float(loss.detach().cpu()))
        receipt={'pool_mode':mode,'multiview':supcon,'batch_size':32,'timed_steps':len(times),
            'seconds_per_step':float(np.mean(times)),'median_seconds_per_step':float(np.median(times)),
            'seconds_each_step':times,'losses':losses,'finite_losses':bool(np.isfinite(losses).all()),
            '30000_step_compute_hours':float(np.mean(times)*30000/3600),
            'peak_memory_mib':torch.cuda.max_memory_allocated()/1024**2,'target_input_opened':False}
        receipts.append(receipt);print(json.dumps(receipt),flush=True)
        del model,opt,criterion,out,loss;torch.cuda.empty_cache()
    write(args.output,{'at':now(),'torch':torch.__version__,'cuda':torch.version.cuda,
        'gpu':torch.cuda.get_device_name(0),'threads':args.threads,'source_only':True,
        'new_formal_runs_completed':0,'receipts':receipts})

if __name__=='__main__':main()
