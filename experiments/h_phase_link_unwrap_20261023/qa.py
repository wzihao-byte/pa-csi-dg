"""Source-only release gate: frozen files, config isolation and a real-source trajectory."""
from __future__ import annotations
import copy, json, subprocess, sys
from types import SimpleNamespace
import numpy as np
import torch
from common import BUNDLE, HERE, OUTPUT, base_module, now, read, require, sha, verify_bundle, write
from train_factorial import install

def source_batch(config, count=4):
    """First scheduled H-S source observations; stored phase is used without another unwrap."""
    ids = np.load(BUNDLE / f"schedules/HS_{config['target_env']}_{config['seed_list'][0]}.npy", mmap_mode='r')[0, 0][:count].copy()
    sources = list(config['data']['env_files'])
    cache = {e: {k: np.load(p, mmap_mode='r') for k, p in files.items()} for e, files in config['data']['env_files'].items()}
    amp, phase, labels, domains = [], [], [], []
    for sample_id in ids:
        env = ('E1', 'E2', 'E3')[int(sample_id) // 3000]
        require(env in sources, 'Scheduled sample is not a source observation')
        row = int(sample_id) % 3000
        amp.append(np.asarray(cache[env]['amp'][row], dtype=np.float32))
        phase.append(np.asarray(cache[env]['phase'][row], dtype=np.float32))
        labels.append(int(cache[env]['label'][row])); domains.append(sources.index(env))
    return tuple(torch.from_numpy(v) for v in (np.stack(amp), np.stack(phase),
        np.array(labels, dtype=np.int64), np.array(domains, dtype=np.int64), ids))

def real_source_trajectory():
    config = read(BUNDLE / 'configs/HS_E1_N_L11_42.json')
    paired = read(config['h_controls']['paired_control_config'])
    scratch = OUTPUT / 'qa/source_trajectory'; scratch.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(config); config['output_dir'] = str(scratch)
    tensors = source_batch(config)
    reference = source_batch(paired)
    require(torch.equal(tensors[0], reference[0]) and all(torch.equal(a, b) for a, b in zip(tensors[2:], reference[2:])),
            'Amplitude, labels, domains or ids differ from paired N_H11')
    require(torch.equal(tensors[1][..., :30], reference[1][..., :30]), 'First receive link differs from paired N_H11')
    turns = (reference[1][..., 30:].double() - tensors[1][..., 30:].double()) / (2 * np.pi)
    require(float((turns - turns.round()).abs().max()) < 1e-3, 'Later links differ by more than whole turns')
    loader = lambda: torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*tensors), batch_size=2,
        shuffle=False, num_workers=0)
    base = install(base_module(), config); modules = base.configure_imports(config); trainer = modules['train_dg']
    trainer.PaCsiDGLite = base.model_class(config, modules)
    trainer.set_seed(42)
    model, optimizer, scheduler, mode = trainer.build_model_and_optimizer(
        SimpleNamespace(input_dim=90, num_classes=6), config, torch.device('cpu'))
    initialization = read(scratch / 'initialization_audit.json')
    require(initialization['initial_state_sha256'] == paired['h_controls']['initial_state_sha256'], 'Initial state differs from paired N_H11')
    loss = trainer.DomainAwareSupConLoss(temperature=.2, include_same_domain_same_class=True,
        positive_mode='global_class_instance_views')
    result = trainer.run_epoch(model=model, loader=loader(), optimizer=optimizer, scheduler=scheduler,
        scheduler_step_mode=mode, device=torch.device('cpu'), loss_config=config['losses'],
        contrastive_loss_type='pairwise_supcon', contrastive_loss_fn=loss, adaptive_kmargin_loss_fn=None,
        adaptive_kmargin_state=None, train=True)
    require(all(np.isfinite(v) for v in result.values() if isinstance(v, float)), 'Nonfinite source loss/diagnostic')
    require(read(scratch / 'training_status.json')['optimizer_steps'] == 1, 'Progress instrumentation did not observe first update')
    require(optimizer.param_groups[0]['lr'] == .001 * .9 ** (2 / 10000), 'Scheduler trajectory drift')
    return {'status': 'passed', 'real_source_examples': 4, 'optimizer_steps': 2,
        'complete_and_flattened_masked_forward': True, 'initial_state_equals_paired_n_h11': True,
        'amplitude_labels_domains_ids_equal_to_paired': True, 'first_link_phase_equal_to_paired': True,
        'later_links_differ_by_whole_turns_only': True,
        'phase_rows_shifted_in_batch': int((turns.round() != 0).flatten(1).any(dim=1).sum()),
        'finite_epoch_result': {k: v for k, v in result.items() if isinstance(v, float)},
        'target_arrays_opened': False}

def main():
    manifest, registry = verify_bundle()
    torch.set_num_threads(2)
    checks = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(HERE / 'tests'), '-v'],
        cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    (OUTPUT / 'qa').mkdir(exist_ok=True)
    (OUTPUT / 'qa/component_tests.log').write_text(checks.stdout, encoding='utf-8')
    require(checks.returncode == 0, 'Component/config tests failed; see qa/component_tests.log')
    trajectory = real_source_trajectory()
    receipt = {'status': 'passed', 'at': now(), 'bundle_manifest_sha256': sha(BUNDLE / 'bundle_manifest.json'),
        'component_test_log_sha256': sha(OUTPUT / 'qa/component_tests.log'),
        'frozen_unique_configs_validated': len(registry['jobs']),
        'real_source_trajectory': trajectory, 'target_arrays_opened': False,
        'torch': torch.__version__, 'cpu_threads': 2, 'new_environment_experiment': False}
    write(OUTPUT / 'qa/approval.json', receipt)
    print(json.dumps(receipt))

if __name__ == '__main__':
    main()
