"""Three-class H-S adapter around the frozen main-study trainer (12,000 updates)."""
from __future__ import annotations
import argparse, copy, functools, os, sys, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
import numpy as np
import torch
import three_class as tc
from common import (PARAMETERS, base_module, now, read, require, sha, state_hash,
                    verify_bundle, verify_config, write)
from control_components import (ControlDomainAwareSupConLoss, make_model_class,
                                shared_state_hash, supcon_positive_pair_diagnostics)


def install(base, config):
    verify_config(config)
    c = config['h_controls']
    base.EXPECTED_PARAMETERS['isolated_pa_csi_dg_lite_tpp'] = PARAMETERS
    base.SCHEMA_VERSION = 'three-class-4s-train/v1'
    original_imports = base.configure_imports
    original_manifest = base.source_manifest
    original_remap = base.remap_global_sample_ids
    identity = {}
    require(sha(c['source_manifest']) == c['source_manifest_sha256'], 'Grouped source manifest drift')
    source = tc.filter_manifest(read(c['source_manifest']))

    def grouped_split(dataset, seed, target, _dg):
        require(seed == config['seed_list'][0], 'Seed drift')
        split, ids = tc.grouped_split(dataset, source, target)
        identity.update(ids)
        return split
    base.source_split = grouped_split

    def remap(dataset):
        if getattr(dataset, 'three_class_role', None) == 'source':
            dataset.sample_ids = dataset.three_class_global_ids.copy()
            return
        original_remap(dataset)
    base.remap_global_sample_ids = remap

    def source_manifest(dataset, split, target):
        record = original_manifest(dataset, split, target)
        record['main_control_partition'] = 'H-S'
        record['source_partition_rule'] = 'verified_subject_acquisition_grouped_holdout_three_class'
        record['group_identity'] = copy.deepcopy(identity)
        record['grouped_source_manifest_sha256'] = c['source_manifest_sha256']
        record['three_class'] = {'keep_original_labels': list(tc.KEEP), 'label_map': {str(k): v for k, v in tc.LABEL_MAP.items()}}
        return record
    base.source_manifest = source_manifest

    def model_class(cfg, modules):
        return make_model_class(modules['tpp_models'].IsolatedPaCsiDGLiteTPP, cfg['h_controls']['pool_mode'])
    base.model_class = model_class

    def configure_imports(cfg):
        modules = original_imports(cfg)
        trainer = modules['train_dg']
        trainer.DomainAwareSupConLoss = ControlDomainAwareSupConLoss
        trainer.supcon_positive_pair_diagnostics = supcon_positive_pair_diagnostics
        original_build = trainer.build_model_and_optimizer
        original_epoch = trainer.run_epoch
        original_load = modules['dg_dataset'].load_dataset
        dataset_class = modules['dg_dataset'].PaCsiArrayDataset
        run = Path(cfg['output_dir'])
        counters = {'steps': 0, 'epochs': 0}
        started = time.monotonic()

        def build(*args, **kwargs):
            result = original_build(*args, **kwargs)
            model, optimizer, scheduler, _ = result
            require(sum(p.numel() for p in model.parameters()) == PARAMETERS, 'Actual capacity drift')
            full_hash = state_hash(model)
            common_hash = shared_state_hash(model)
            require(full_hash == c['initial_state_sha256'], 'Actual initial state differs from frozen seed')
            require(common_hash == c['common_state_sha256'], 'Shared tensors differ across arms')
            require(scheduler is not None, 'Historical step scheduler missing')
            write(run / 'initialization_audit.json', {'status': 'passed', 'parameter_count': PARAMETERS,
                'initial_state_sha256': full_hash, 'common_state_sha256': common_hash,
                'pool_mode': c['pool_mode'], 'seed': cfg['seed_list'][0], 'model_class': type(model).__name__})
            original_step = optimizer.step

            @functools.wraps(original_step)
            def step(*step_args, **step_kwargs):
                value = original_step(*step_args, **step_kwargs)
                counters['steps'] += 1
                if counters['steps'] == 1 or counters['steps'] % 100 == 0:
                    write(run / 'training_status.json', {'status': 'training', 'at': now(),
                        'job_id': cfg['job_id'], 'pid': os.getpid(), 'optimizer_steps': counters['steps'],
                        'total_optimizer_steps': tc.TOTAL_STEPS, 'elapsed_seconds': time.monotonic() - started,
                        'lr_used': optimizer.param_groups[0]['lr'], 'target_opened': False})
                return value
            optimizer.step = step
            return result

        def epoch(*args, **kwargs):
            value = original_epoch(*args, **kwargs)
            if kwargs.get('train') is False and counters['epochs'] < tc.EPOCHS:
                counters['epochs'] += 1
                write(run / 'source_progress.json', {'at': now(), 'epoch': counters['epochs'],
                    'optimizer_steps': counters['steps'], 'source_val_accuracy': float(value['accuracy']),
                    'target_opened': False})
            return value

        def load(data_config, base_dir):
            environments = list(data_config['env_files'])
            if cfg['target_env'] in environments:
                require(environments == [cfg['target_env']], 'Mixed source/target loader')
                frozen = read(run / 'checkpoint_frozen.json')
                require(frozen['strict_load'] and frozen['target_data_opened'] is False, 'Premature target access')
                require(sha(run / 'best_model.pt') == frozen['checkpoint_sha256'], 'Target model was not frozen')
                dataset = tc.subset_loaded(original_load(data_config, base_dir), dataset_class, 'target')
                write(run / 'target_index_map.json', {'target_env': cfg['target_env'],
                    'prediction_index_to_main_study_global_id': dataset.three_class_global_ids.tolist(),
                    'global_ids_sha256': tc.index_hash(dataset.three_class_global_ids)})
                return dataset
            require(environments == cfg['protocol']['source_envs'], 'Source order drift')
            return tc.subset_loaded(original_load(data_config, base_dir), dataset_class, 'source')

        trainer.build_model_and_optimizer = build
        trainer.run_epoch = epoch
        modules['dg_dataset'].load_dataset = load
        modules['_control_counters'] = counters
        return modules

    base.configure_imports = configure_imports
    return base


def runtime_receipt(config):
    require(torch.cuda.is_available(), 'Formal training requires CUDA')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import sklearn
    return {'at': now(), 'pid': os.getpid(), 'python': sys.version, 'python_executable': sys.executable,
        'torch': torch.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__,
        'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
        'gpu': torch.cuda.get_device_name(0), 'torch_threads': torch.get_num_threads(),
        'cublas_workspace': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cudnn_deterministic': torch.backends.cudnn.deterministic, 'cudnn_benchmark': torch.backends.cudnn.benchmark}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--validate-checkpoint', type=Path)
    args = p.parse_args()
    config = read(args.config)
    verify_config(config)
    base = install(base_module(), config)
    if args.validate_checkpoint:
        base.validate_checkpoint(args.config.resolve(), args.validate_checkpoint.resolve())
        return
    verify_bundle()
    receipt = runtime_receipt(config)
    write(Path(config['output_dir']).parents[1] / 'training_runtime.json', receipt)
    base.train(args.config.resolve())
    run = Path(config['output_dir'])
    require(read(run / 'training_status.json')['optimizer_steps'] == tc.TOTAL_STEPS, 'Actual optimizer count drift')
    metrics = read(run / 'metrics.json')
    require(metrics['sampler_schedule_sha256'] == config['h_controls']['expected_sampler_sha256'], 'Sampling trajectory drift')
    history = metrics['history']
    best = max(float(row['val']['accuracy']) for row in history)
    last_best = max(int(row['epoch']) for row in history if row['val']['accuracy'] == best)
    require(metrics['best_val_score'] == best, 'Source selection drift')
    write(run / 'control_alignment_audit.json', {'status': 'passed', 'at': now(),
        'source_membership_and_sampler_matched': True, 'source_last_tied_best_epoch': last_best,
        'expected_sampler_sha256': config['h_controls']['expected_sampler_sha256'],
        'total_optimizer_steps': tc.TOTAL_STEPS, 'initialization': read(run / 'initialization_audit.json')})
    write(run / 'training_status.json', {'status': 'complete', 'at': now(), 'job_id': config['job_id'],
        'optimizer_steps': tc.TOTAL_STEPS, 'target_evaluated_after_source_checkpoint_freeze': True})


if __name__ == '__main__':
    main()
