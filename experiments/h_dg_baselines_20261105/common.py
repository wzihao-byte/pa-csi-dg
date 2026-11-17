"""Frozen generic-DG baseline arms (CORAL, Mixup) on the H-S four-arm backbones.

Global-backbone arms GC10/GX10 repeat the completed v2 capacity control G10
(global MLP pooling + CE) and TPP-backbone arms C10/X10 repeat the completed
H-S T-CE run (historical HS10) with only the training objective changed; the
paired G10/G11/HS10/HS11 runs are reused, not retrained.

v1 (two TPP-backbone arms) was stopped during its first job after a design
review and is preserved unchanged; v2 is the executed version.
"""
from __future__ import annotations
import copy, hashlib, importlib.util, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(r'C:/Users/howie/Desktop/pa-csi-dg')
RUNTIME = Path(r'D:/pa-csi-dg-revision-runtime')
OUTPUT = RUNTIME / 'studies/h_dg_baselines_20261105_v2'
BUNDLE = OUTPUT / 'bundle'
HISTORICAL = ROOT / 'outputs_h_grouped_l1248/v2/bundle'
PAIRED_STUDY = RUNTIME / 'studies/h_main_controls_20261008_v2'
PAIRED_BUNDLE = PAIRED_STUDY / 'bundle'
PAIRED_MANIFEST_SHA256 = '58db531376427eece15274dd3eb1dcb79d4484b284dfc499eeed6cc84cb26ac6'
TARGETS = ('E1', 'E2', 'E3')
SEEDS = (42, 52, 62)
PARAMETERS = 4_786_294
ARMS = ('GC10', 'GX10', 'C10', 'X10')   # global-backbone arms are queued first
ARM_POOL = {'GC10': 'global_mlp', 'GX10': 'global_mlp', 'C10': 'tpp', 'X10': 'tpp'}
GROUP = 'dg_baseline'
JOBS = len(TARGETS) * len(SEEDS) * len(ARMS)
PSEUDO_TX_LAYOUT = {'mode': 'tx', 'num_rx': 1, 'num_tx': 3, 'num_subcarriers': 30}

def now():
    return datetime.now(timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def write(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
        return
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name, suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

def base_module(kind='trainer'):
    local = HERE / ('base_' + kind + '.py')
    return load_module('dg_baselines_base_' + kind, local if local.exists() else PAIRED_BUNDLE / 'code' / local.name)

def grouped_helpers():
    local = HERE / 'grouped_helpers.py'
    return load_module('dg_baselines_grouped_helpers', local if local.exists() else PAIRED_BUNDLE / 'code/grouped_helpers.py')

def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()

def objective_free(config):
    """The configuration with the per-job label, objective block, control block and attempt path removed."""
    x = copy.deepcopy(config)
    for key in ('job_id', 'experiment_name', 'provenance', 'h_controls', 'h_grouped', 'dg_baseline', 'output_dir'):
        x.pop(key, None)
    x['factorial'].pop('arm')
    x['factorial'].pop('tpp_on')   # pooling is checked separately against the arm's backbone
    return x

def verify_bundle():
    require(HERE.resolve() == (BUNDLE / 'code').resolve(), 'Execute the frozen bundle/code copy')
    manifest = read(BUNDLE / 'bundle_manifest.json')
    for rel, expected in manifest['files_sha256'].items():
        require(sha(BUNDLE / rel) == expected, 'Frozen DG-baseline file changed: ' + rel)
    registry = read(BUNDLE / 'registry.json')
    expected = {('H-S', e, s, a) for e in TARGETS for s in SEEDS for a in ARMS}
    actual = [(j['partition'], j['target'], j['seed'], j['control_arm']) for j in registry['jobs']]
    require(len(actual) == JOBS and len(set(actual)) == JOBS and set(actual) == expected, 'Wrong unique H-S 18-job grid')
    for job in registry['jobs']:
        config = read(BUNDLE / job['config'])
        require(config['job_id'] == job['job_id'], 'Registry/config job mismatch')
        verify_config(config)
    return manifest, registry

def verify_config(config):
    from dg_baselines import validate_specification
    c = config['h_controls']
    require(c['partition'] == 'H-S' and c['arm'] in ARMS and c['group'] == GROUP, 'Unplanned control arm')
    pool = ARM_POOL[c['arm']]
    require(c['pool_mode'] == pool and c['phase_variant'] == 'historical', 'Arm definition drift')
    require(config['factorial']['arm'] == 'HS_' + c['arm'], 'Factorial arm drift')
    require(config['factorial']['tpp_on'] is (pool == 'tpp') and config['factorial']['supcon_on'] is False, 'Factorial switches drift')
    require(config['factorial']['lambda_supcon'] == 0.0 and config['factorial']['supcon_views'] == 1, 'Factorial contrastive drift')
    validate_specification(config['dg_baseline'], c['arm'])
    cells = read(BUNDLE / 'paired_cells.json')
    cell = cells[f"{config['target_env']}_{config['seed_list'][0]}"]
    for key in ('reference_config', 'sampler_global_ids_file', 'source_manifest'):
        require(sha(c[key]) == c[key + '_sha256'], 'Bundled control file changed: ' + key)
    require(c['reference_config_sha256'] == cell['reference_h10_sha256'], 'Reference T-CE config drift')
    require(c['sampler_global_ids_file_sha256'] == cell['schedule_sha256'], 'Schedule file drift')
    require(c['source_manifest_sha256'] == cell['source_manifest_sha256'], 'Source manifest drift')
    require(c['expected_sampler_sha256'] == cell['expected_sampler_sha256'], 'Sampling schedule drift')
    require(c['initial_state_sha256'] == cell['initial_state_sha256'][pool], 'Backbone initial state drift')
    require(c['common_state_sha256'] == cell['common_state_sha256'], 'Shared initial state drift')
    require(c['paired_tce_job_id'] == cell['tce_job_id'] and c['paired_tsc_job_id'] == cell['tsc_job_id'], 'Paired job drift')
    require(c['paired_gce_job_id'] == cell['g10_job_id'], 'Paired G10 job drift')
    require(c['paired_control_job_id'] == (cell['g10_job_id'] if pool == 'global_mlp' else cell['tce_job_id']), 'Paired backbone drift')
    require(c['new_hyperparameter_search_trials'] == 0 and c['target_used_previously_for_development'] is True
            and c['new_environment_experiment'] is False, 'Control declaration drift')
    reference = read(c['reference_config'])
    require(reference['factorial']['arm'] == 'H10' and reference['factorial']['supcon_on'] is False, 'Reference is not the T-CE arm')
    require(objective_free(config) == objective_free(reference), 'Change beyond the objective/pooling relative to the paired T-CE run')
    if pool == 'global_mlp':
        g10 = read(BUNDLE / 'paired_controls' / f"HS_{config['target_env']}_G10_{config['seed_list'][0]}.json")
        require(sha(BUNDLE / 'paired_controls' / f"HS_{config['target_env']}_G10_{config['seed_list'][0]}.json") == cell['g10_config_sha256'], 'Paired G10 config drift')
        require(objective_free(config) == objective_free(g10), 'Change beyond the objective relative to the paired G10 run')
        require(g10['h_controls']['initial_state_sha256'] == c['initial_state_sha256'], 'Global initial state differs from paired G10')
    require(config['losses']['lambda_supcon'] == 0.0 and config['losses']['lambda_supcon_warmup_epochs'] == 0, 'Contrastive coefficient drift')
    require(config['losses']['temperature'] == .2 and config['losses']['supcon_positive_mode'] == 'global_class_instance_views',
            'Historical loss block drift')
    require(config['protocol']['checkpoint_rule'] == 'source_val_accuracy_last_tied_best', 'Selection drift')
    require(config['protocol']['total_optimizer_steps'] == 30000, 'Budget drift')
    require(config['protocol']['expected_parameter_count'] == PARAMETERS, 'Capacity drift')
    require(config['training']['batch_size'] == 32 and config['training']['sampler'] == 'domain_class_balanced', 'Batch/sampler drift')
    require(config['data']['unwrap_phase'] is True and config['data']['phase_unwrap_axis'] == -1, 'Historical phase loader drift')
    require(config['provenance']['legacy_commit'] == reference['provenance']['legacy_commit'], 'Legacy commit drift')
    require(config['provenance']['tpp_models_sha256'] == reference['provenance']['tpp_models_sha256'], 'TPP code drift')
    require({k: v['sha256'] for k, v in config['provenance']['legacy_modules'].items()}
            == {k: v['sha256'] for k, v in reference['provenance']['legacy_modules'].items()}, 'Legacy module drift')
    for name, record in config['provenance']['legacy_modules'].items():
        require(Path(record['path']).resolve() == (BUNDLE / 'code/legacy' / (name + '.py')).resolve(), 'Legacy path outside bundle: ' + name)
        require(sha(record['path']) == record['sha256'], 'Bundled legacy module differs: ' + name)
    require(Path(config['provenance']['tpp_snapshot_root']).resolve() == (BUNDLE / 'code/tpp').resolve(), 'TPP snapshot outside bundle')
    require(sha(BUNDLE / 'code/tpp/experiments/tpp_isolated/tpp_models.py') == config['provenance']['tpp_models_sha256'], 'Bundled TPP differs')
    for env, files in config['data']['env_files'].items():
        require(env in config['source_file_records'], 'Unbound source environment')
        for kind, path in files.items():
            require(path == config['source_file_records'][env][kind]['path'], 'Loader/source hash receipt path mismatch')
    require(list(config['data']['env_files']) == config['protocol']['source_envs'], 'Source routing/order drift')
    require(list(config['target_data']['data']['env_files']) == [config['target_env']], 'Deferred target routing drift')
    for kind, path in config['target_data']['data']['env_files'][config['target_env']].items():
        require(path == config['target_data']['file_records'][kind]['path'], 'Loader/target hash receipt path mismatch')
    require(config['model']['antenna_layout'] == PSEUDO_TX_LAYOUT, 'Pseudo-Tx slice drift')
    return reference
