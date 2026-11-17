"""Frozen, historically aligned controls of the main H/H-S recognizer."""
from __future__ import annotations
import copy, hashlib, importlib.util, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(r'C:/Users/howie/Desktop/pa-csi-dg')
OUTPUT = Path(r'D:/pa-csi-dg-revision-runtime/studies/h_main_controls_20261008_v2')
BUNDLE = OUTPUT / 'bundle'
HISTORICAL = ROOT / 'outputs_h_grouped_l1248/v2/bundle'
TARGETS = ('E1', 'E2', 'E3')
SEEDS = (42, 52, 62)
PARAMETERS = 4_786_294
ARMS = ('G10', 'G11', 'P_ALL', 'P_REMOVE', 'N_H11', 'N_D11')

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
    return load_module('main_h_base_' + kind, local if local.exists() else HISTORICAL / 'code' / local.name)

def grouped_helpers():
    local = HERE / 'grouped_helpers.py'
    return load_module('main_h_grouped_helpers', local if local.exists() else HISTORICAL / 'code/common.py')

def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()

def verify_bundle():
    require(HERE.resolve() == (BUNDLE / 'code').resolve(), 'Execute the frozen bundle/code copy')
    manifest = read(BUNDLE / 'bundle_manifest.json')
    for rel, expected in manifest['files_sha256'].items():
        require(sha(BUNDLE / rel) == expected, 'Frozen control file changed: ' + rel)
    registry = read(BUNDLE / 'registry.json')
    expected = {('H-S', e, s, a) for e in TARGETS for s in SEEDS for a in ARMS}
    actual = [(j['partition'],j['target'],j['seed'],j['control_arm']) for j in registry['jobs']]
    require(len(actual) == 54 and len(set(actual)) == 54 and set(actual) == expected, 'Wrong unique H-S 54-job grid')
    for job in registry['jobs']:
        config = read(BUNDLE / job['config'])
        require(config['job_id'] == job['job_id'], 'Registry/config job mismatch')
        verify_config(config)
    return manifest, registry

def verify_config(config):
    c = config['h_controls']
    reference = read(c['reference_config'])
    require(sha(c['reference_config']) == c['reference_config_sha256'], 'Reference config changed')
    require(config['training'] == reference['training'], 'Historical training settings changed')
    require(config['split'] == reference['split'], 'Historical split settings changed')
    require(config['seed_list'] == reference['seed_list'], 'Seed drift')
    require(config['target_env'] == reference['target_env'], 'Target drift')
    require(config['protocol']['expected_split_hashes'] == reference['protocol']['expected_split_hashes'], 'Membership drift')
    require(config['protocol']['checkpoint_rule'] == 'source_val_accuracy_last_tied_best', 'Selection drift')
    require(config['protocol']['total_optimizer_steps'] == 30000, 'Budget drift')
    require(config['model'] == reference['model'], 'Backbone/fusion/TPP specification drift')
    expected_losses = copy.deepcopy(reference['losses'])
    expected_positive = ('all_views' if c['arm'] == 'P_ALL' else
        'hybrid_remove_other_sameclass_denominator' if c['arm'] == 'P_REMOVE' else 'global_class_instance_views')
    expected_losses['supcon_positive_mode'] = expected_positive
    require(config['losses'] == expected_losses, 'Unexpected loss change')
    require(config['losses']['temperature'] == .2, 'Temperature drift')
    require(config['losses']['lambda_supcon_warmup_epochs'] == 0, 'Warm-up drift')
    expected_lambda = .05 if config['factorial']['supcon_on'] else 0.
    require(config['losses']['lambda_supcon'] == expected_lambda, 'Contrastive coefficient drift')
    require(c['pool_mode'] in ('tpp', 'global_mlp'), 'Unsupported pooling control')
    require(c['partition'] == 'H-S', 'Only H-S is authorized for this frozen execution')
    require(c['arm'] in ARMS, 'Unplanned control arm')
    expected_pool = 'global_mlp' if c['arm'].startswith('G') else 'tpp'
    expected_phase = 'native_historical' if c['arm'] == 'N_H11' else 'native_differential' if c['arm'] == 'N_D11' else 'historical'
    require(c['pool_mode'] == expected_pool and c['phase_variant'] == expected_phase, 'Arm definition drift')
    require(config['factorial']['supcon_on'] == (c['arm'] != 'G10'), 'Arm coefficient/forward-exposure drift')
    require(config['losses']['include_same_domain_same_class'] is True and config['losses']['contrastive_loss_type'] == 'pairwise_supcon', 'Historical contrastive type drift')
    require(config['protocol']['expected_parameter_count'] == PARAMETERS, 'Capacity drift')
    if c['phase_variant'] == 'historical':
        require(config['data'] == reference['data'] and config['target_data'] == reference['target_data'], 'Historical input drift')
        require(config['source_file_records'] == reference['source_file_records'], 'Input hash drift')
    else:
        phase = read(BUNDLE / 'phase_alignment.json')
        require(phase['status'] == 'passed' and phase['rows'] == 9000, 'Native identity audit incomplete')
        bindings = phase['variants'][phase['variant_aliases'][c['phase_variant']]]['file_records']
        require(config['data']['unwrap_phase'] is False and config['target_data']['data']['unwrap_phase'] is False,
                'Native phase must not be unwrapped twice')
        for candidate, old in ((config['data'], reference['data']),
                               (config['target_data']['data'], reference['target_data']['data'])):
            for key in old:
                if key not in ('env_files','unwrap_phase'):
                    require(candidate[key] == old[key], 'Non-input loader setting changed: ' + key)
        require(list(config['data']['env_files']) == config['protocol']['source_envs'], 'Source routing/order drift')
        require(list(config['target_data']['data']['env_files']) == [config['target_env']], 'Deferred target routing drift')
        for env, records in config['source_file_records'].items():
            require(records == bindings[env], 'Native source receipt differs from audited arrays')
        require(config['target_data']['file_records'] == bindings[config['target_env']], 'Native target receipt differs from audited arrays')
    for env, files in config['data']['env_files'].items():
        require(env in config['source_file_records'], 'Unbound source environment')
        for kind, path in files.items():
            require(path == config['source_file_records'][env][kind]['path'], 'Loader/source hash receipt path mismatch')
    require(set(config['source_file_records']) == set(config['data']['env_files']), 'Source receipt environment mismatch')
    for kind, path in config['target_data']['data']['env_files'][config['target_env']].items():
        require(path == config['target_data']['file_records'][kind]['path'], 'Loader/target hash receipt path mismatch')
    require(config['model']['antenna_layout'] == {'mode':'tx','num_rx':1,'num_tx':3,'num_subcarriers':30}, 'Pseudo-Tx slice drift')
    return reference
