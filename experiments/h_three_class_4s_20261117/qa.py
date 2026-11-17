"""Source-only release gate for the three-class bundle (no target array is opened).

1. Every frozen file and config verifies.
2. For each target, the real source loader + three-class hooks reproduce the frozen
   split hashes, and the legacy sampler on the real restricted dataset reproduces the
   frozen sampler hash for all three seeds.
3. For each arm, the installed trainer builds the model with the frozen initial state
   and takes two real optimizer steps (complete + single-link views for SupCon arms)
   with finite loss.
"""
from __future__ import annotations
import copy, gc, json
from pathlib import Path
from types import SimpleNamespace
import torch
import three_class as tc
from common import (ARMS, BUNDLE, OUTPUT, SEEDS, TARGETS, base_module, now, read, require, sha, verify_bundle, write)
from train_factorial import install


def config_for(target, arm, seed, scratch):
    config = copy.deepcopy(read(BUNDLE / f'configs/TC_{target}_{arm}_{seed}.json'))
    config['output_dir'] = str(scratch)
    return config


def source_checks(target):
    scratch = OUTPUT / 'qa/scratch' / target; scratch.mkdir(parents=True, exist_ok=True)
    config = config_for(target, 'T11', 42, scratch)
    base = install(base_module(), config)
    modules = base.configure_imports(config)
    dataset = modules['dg_dataset'].load_dataset(config['data'], modules['clean_root'])
    base.remap_global_sample_ids(dataset)
    require(len(dataset) == 2 * tc.PER_ENV and dataset.num_classes == 3, 'Restricted source dataset shape')
    split = base.source_split(dataset, 42, target, modules['dg_dataset'])
    record = base.source_manifest(dataset, split, target)
    for part in ('train', 'val'):
        require(record[part]['global_sample_id_sha256'] == config['protocol']['expected_split_hashes'][part], f'{target} {part} split hash')
    sampler = {}
    for seed in SEEDS:
        expected = read(BUNDLE / f'configs/TC_{target}_T11_{seed}.json')['h_controls']['expected_sampler_sha256']
        actual, _, counts = tc.replay_sampler(dataset, split, seed, modules['dg_dataset'].DomainClassBalancedBatchSampler)
        require(actual == expected, f'{target} seed {seed} sampler replay on real data differs')
        sampler[str(seed)] = {'sha256': actual, 'class_counts': counts}
    batch_idx = split['train_indices'][:2] + split['train_indices'][-2:]
    tensors = [torch.stack([dataset[i][k] for i in batch_idx]) for k in range(5)]
    del dataset; gc.collect()
    return {'split_hashes_match': True, 'sampler_replay': sampler, 'label_counts': record['train']['label_counts']}, tensors


def arm_step(target, arm, tensors):
    scratch = OUTPUT / 'qa/scratch' / f'{target}_{arm}'; scratch.mkdir(parents=True, exist_ok=True)
    for f in scratch.iterdir():
        f.unlink()
    config = config_for(target, arm, 42, scratch)
    base = install(base_module(), config)
    modules = base.configure_imports(config)
    trainer = modules['train_dg']
    trainer.PaCsiDGLite = base.model_class(config, modules)
    trainer.set_seed(42)
    meta = SimpleNamespace(input_dim=90, num_classes=3)
    model, optimizer, scheduler, mode = trainer.build_model_and_optimizer(meta, config, torch.device('cpu'))
    loss = trainer.DomainAwareSupConLoss(temperature=.2, include_same_domain_same_class=True,
                                        positive_mode='global_class_instance_views')
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors), batch_size=2, shuffle=False)
    result = trainer.run_epoch(model=model, loader=loader, optimizer=optimizer, scheduler=scheduler,
        scheduler_step_mode=mode, device=torch.device('cpu'), loss_config=config['losses'],
        contrastive_loss_type='pairwise_supcon', contrastive_loss_fn=loss,
        adaptive_kmargin_loss_fn=None, adaptive_kmargin_state=None, train=True)
    finite = all(torch.isfinite(p).all() for p in model.parameters())
    require(finite and result['loss_total'] == result['loss_total'], f'{arm} non-finite training step')
    supcon = ARMS[arm][2]
    require((result['loss_supcon'] > 0) == supcon, f'{arm} contrastive term presence does not match the arm')
    init = read(scratch / 'initialization_audit.json')
    require(init['initial_state_sha256'] == config['h_controls']['initial_state_sha256'], f'{arm} initial state')
    require(read(scratch / 'training_status.json')['optimizer_steps'] == 1, f'{arm} progress instrumentation')
    return {'loss_total': float(result['loss_total']), 'loss_supcon': float(result['loss_supcon']),
            'initial_state_matches_frozen': True, 'optimizer_steps': 2}


def main():
    manifest, registry = verify_bundle()
    torch.set_num_threads(4)
    sources, steps = {}, {}
    tensors = None
    for target in TARGETS:
        sources[target], t = source_checks(target)
        tensors = tensors or t
        print('source checks passed', target, flush=True)
    for arm in ARMS:
        steps[arm] = arm_step('E1', arm, tensors)
        print('arm step passed', arm, steps[arm], flush=True)
    receipt = {'status': 'passed', 'at': now(), 'bundle_manifest_sha256': sha(BUNDLE / 'bundle_manifest.json'),
        'configs_validated': len(registry['jobs']), 'source_checks': sources, 'arm_steps': steps,
        'target_arrays_opened': False}
    write(OUTPUT / 'qa/approval.json', receipt)
    print(json.dumps({'status': 'passed', 'arms': list(steps)}))


if __name__ == '__main__':
    main()
