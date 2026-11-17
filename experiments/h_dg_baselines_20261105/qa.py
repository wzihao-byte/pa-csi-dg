"""Source-only release gate: frozen files, component tests, CE replay equivalence and real-source trajectories."""
from __future__ import annotations
import copy, json, subprocess, sys
from types import SimpleNamespace
import numpy as np
import torch
from common import ARMS, BUNDLE, HERE, OUTPUT, base_module, now, read, require, sha, verify_bundle, write
from train_factorial import install

def source_batch(config, per_source=4):
    """Scheduled H-S source observations of the first batch, ``per_source`` from each source, interleaved."""
    ids = np.load(BUNDLE / f"schedules/HS_{config['target_env']}_{config['seed_list'][0]}.npy", mmap_mode='r')[0, 0].copy()
    sources = list(config['data']['env_files'])
    cache = {e: {k: np.load(p, mmap_mode='r') for k, p in files.items()} for e, files in config['data']['env_files'].items()}
    chosen = {s: [] for s in sources}
    for sample_id in ids:
        env = ('E1', 'E2', 'E3')[int(sample_id) // 3000]
        require(env in sources, 'Scheduled sample is not a source observation')
        if len(chosen[env]) < per_source:
            chosen[env].append(int(sample_id))
    require(all(len(v) == per_source for v in chosen.values()), 'First batch lacks enough samples per source')
    order = [sid for pair in zip(*[chosen[s] for s in sources]) for sid in pair]  # s0, s1, s0, s1, ...
    amp, phase, labels, domains = [], [], [], []
    for sample_id in order:
        env = ('E1', 'E2', 'E3')[sample_id // 3000]; row = sample_id % 3000
        amp.append(np.asarray(cache[env]['amp'][row], dtype=np.float32))
        phase.append(np.unwrap(np.asarray(cache[env]['phase'][row], dtype=np.float64), axis=-1).astype(np.float32))
        labels.append(int(cache[env]['label'][row])); domains.append(sources.index(env))
    return tuple(torch.from_numpy(v) for v in (np.stack(amp), np.stack(phase), np.array(labels, dtype=np.int64),
                                               np.array(domains, dtype=np.int64), np.array(order, dtype=np.int64)))

def loader_of(tensors, batch_size):
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors), batch_size=batch_size, shuffle=False, num_workers=0)

def prepared(config, scratch, spec_override=None):
    config = copy.deepcopy(config); config['output_dir'] = str(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    base = install(base_module(), config, spec_override=spec_override)
    modules = base.configure_imports(config); trainer = modules['train_dg']
    trainer.PaCsiDGLite = base.model_class(config, modules)
    trainer.set_seed(42)
    model, optimizer, scheduler, mode = trainer.build_model_and_optimizer(SimpleNamespace(input_dim=90, num_classes=6), config, torch.device('cpu'))
    return config, trainer, model, optimizer, scheduler, mode

def run_two_steps(trainer, model, optimizer, scheduler, mode, config, loader):
    loss = trainer.DomainAwareSupConLoss(temperature=.2, include_same_domain_same_class=True, positive_mode='global_class_instance_views')
    return trainer.run_epoch(model=model, loader=loader, optimizer=optimizer, scheduler=scheduler, scheduler_step_mode=mode,
        device=torch.device('cpu'), loss_config=dict(config['losses'], grad_clip_norm=0.0), contrastive_loss_type='pairwise_supcon',
        contrastive_loss_fn=loss, adaptive_kmargin_loss_fn=None, adaptive_kmargin_state=None, train=True)

def ce_replay_equivalence(config, tensors):
    """The replayed CE loop must reproduce the historical CE run_epoch bitwise on real source data."""
    pool = config['h_controls']['pool_mode']
    scratch = OUTPUT / f'qa/ce_equivalence_{pool}'
    original_base = base_module(); modules = original_base.configure_imports(dict(config, output_dir=str(scratch)))
    ta = modules['train_dg']
    from control_components import make_model_class
    ta.PaCsiDGLite = make_model_class(modules['tpp_models'].IsolatedPaCsiDGLiteTPP, pool)
    ta.set_seed(42)
    ma, oa, sa, mode_a = ta.build_model_and_optimizer(SimpleNamespace(input_dim=90, num_classes=6), config, torch.device('cpu'))
    la = modules['dg_losses'].DomainAwareSupConLoss(temperature=.2, include_same_domain_same_class=True, positive_mode='global_class_instance_views')
    result_a = ta.run_epoch(model=ma, loader=loader_of(tensors, 4), optimizer=oa, scheduler=sa, scheduler_step_mode=mode_a,
        device=torch.device('cpu'), loss_config=dict(config['losses'], grad_clip_norm=0.0), contrastive_loss_type='pairwise_supcon',
        contrastive_loss_fn=la, adaptive_kmargin_loss_fn=None, adaptive_kmargin_state=None, train=True)
    _, tb, mb, ob, sb, mode_b = prepared(config, scratch, spec_override={'method': 'ce'})
    result_b = run_two_steps(tb, mb, ob, sb, mode_b, config, loader_of(tensors, 4))
    require(result_a == result_b, 'Replayed CE epoch changed real-data loss/metrics')
    require(all(torch.equal(ma.state_dict()[k], v) for k, v in mb.state_dict().items()), 'Replayed CE epoch changed the state trajectory')
    require(oa.param_groups[0]['lr'] == ob.param_groups[0]['lr'] == .001 * .9 ** (2 / 10000), 'Scheduler trajectory drift')
    require(sa.state_dict() == sb.state_dict(), 'Scheduler state drift')
    return {'status': 'passed', 'pool_mode': pool, 'real_source_examples': 8, 'optimizer_steps': 2,
            'all_model_state_tensors_bitwise_equal': True, 'loss_metrics_bitwise_equal': True, 'scheduler_state_equal': True}

def arm_trajectory(arm, tensors):
    config = read(BUNDLE / f'configs/HS_E1_{arm}_42.json')
    scratch = OUTPUT / f'qa/source_trajectory_{arm}'
    cfg, trainer, model, optimizer, scheduler, mode = prepared(config, scratch)
    initialization = read(scratch / 'initialization_audit.json')
    require(initialization['initial_state_sha256'] == config['h_controls']['initial_state_sha256'], 'Initial state differs from the paired backbone run')
    require(initialization['pool_mode'] == config['h_controls']['pool_mode'], 'Pooling mode drift')
    before = {k: v.clone() for k, v in model.state_dict().items()}
    result = run_two_steps(trainer, model, optimizer, scheduler, mode, cfg, loader_of(tensors, 4))
    require(all(np.isfinite(v) for v in result.values() if isinstance(v, float)), 'Nonfinite source loss/diagnostic')
    require('loss_dg_regularizer' in result and result['loss_supcon'] == 0.0, 'Objective instrumentation missing')
    if arm in ('C10', 'GC10'):
        require(result['loss_dg_regularizer'] > 0.0 and result['dg_regularizer_weight'] == 1.0, 'CORAL penalty not applied')
    else:
        require(0.0 < result['mixup_mean_coefficient'] < 1.0 and result['dg_regularizer_weight'] == 0.0, 'Mixup coefficients not applied')
    changed = sum(not torch.equal(before[k], v) for k, v in model.state_dict().items() if v.dtype.is_floating_point)
    require(changed > 0, 'No parameter changed after two updates')
    require(read(scratch / 'training_status.json')['optimizer_steps'] == 1, 'Progress instrumentation did not observe first update')
    require(optimizer.param_groups[0]['lr'] == .001 * .9 ** (2 / 10000), 'Scheduler trajectory drift')
    return {'status': 'passed', 'arm': arm, 'real_source_examples': 8, 'optimizer_steps': 2, 'floating_tensors_changed': changed,
            'finite_epoch_result': {k: v for k, v in result.items() if isinstance(k, str) and isinstance(v, float)},
            'target_arrays_opened': False}

def main():
    manifest, registry = verify_bundle()
    torch.set_num_threads(2)
    checks = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(HERE / 'tests'), '-v'],
        cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    (OUTPUT / 'qa').mkdir(exist_ok=True)
    (OUTPUT / 'qa/component_tests.log').write_text(checks.stdout, encoding='utf-8')
    require(checks.returncode == 0, 'Component tests failed; see qa/component_tests.log')
    config = read(BUNDLE / 'configs/HS_E1_C10_42.json')
    tensors = source_batch(config)
    require(sorted(tensors[3].tolist()) == [0] * 4 + [1] * 4 and tensors[3].tolist()[:2] == [0, 1], 'QA batches must hold two samples per source')
    equivalence = {pool: ce_replay_equivalence(read(BUNDLE / f'configs/HS_E1_{arm}_42.json'), tensors)
                   for pool, arm in (('tpp', 'C10'), ('global_mlp', 'GC10'))}
    trajectories = {arm: arm_trajectory(arm, tensors) for arm in ARMS}
    receipt = {'status': 'passed', 'at': now(), 'bundle_manifest_sha256': sha(BUNDLE / 'bundle_manifest.json'),
        'component_test_log_sha256': sha(OUTPUT / 'qa/component_tests.log'),
        'frozen_unique_configs_validated': len(registry['jobs']), 'ce_replay_equivalence': equivalence,
        'real_source_trajectories': trajectories, 'target_arrays_opened': False,
        'torch': torch.__version__, 'cpu_threads': 2, 'new_environment_experiment': False}
    write(OUTPUT / 'qa/approval.json', receipt)
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
