"""Frozen within-receive-link unwrap control of the H-S native historical-style model.

Each N_L11 job is the completed N_H11 job of h_main_controls_20261008_v2 with only
the phase arrays replaced; the paired N_H11 runs are reused, not retrained.
"""
from __future__ import annotations
import copy, hashlib, importlib.util, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path(r'C:/Users/howie/Desktop/pa-csi-dg')
RUNTIME = Path(r'D:/pa-csi-dg-revision-runtime')
# v1 failed preflight QA before any training: phase_free also compared the
# per-attempt output_dir. It is preserved unchanged; v2 is the executed version.
OUTPUT = RUNTIME / 'studies/h_phase_link_unwrap_20261023_v2'
BUNDLE = OUTPUT / 'bundle'
HISTORICAL = ROOT / 'outputs_h_grouped_l1248/v2/bundle'
PAIRED_STUDY = RUNTIME / 'studies/h_main_controls_20261008_v2'
PAIRED_BUNDLE = PAIRED_STUDY / 'bundle'
PAIRED_MANIFEST_SHA256 = '58db531376427eece15274dd3eb1dcb79d4484b284dfc499eeed6cc84cb26ac6'
LINK_DATA = RUNTIME / 'data/native_linkunwrap_phase_v1'
LINK_VARIANT = 'native_time_wrapped_linear_link_unwrap_v1'
TARGETS = ('E1', 'E2', 'E3')
SEEDS = (42, 52, 62)
PARAMETERS = 4_786_294
ARMS = ('N_L11',)
PAIRED_ARM = 'N_H11'
JOBS = len(TARGETS) * len(SEEDS) * len(ARMS)

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
    return load_module('link_unwrap_base_' + kind, local if local.exists() else PAIRED_BUNDLE / 'code' / local.name)

def grouped_helpers():
    local = HERE / 'grouped_helpers.py'
    return load_module('link_unwrap_grouped_helpers', local if local.exists() else PAIRED_BUNDLE / 'code/grouped_helpers.py')

def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()

def phase_free(config):
    """The configuration with every phase binding, per-job label and attempt path removed."""
    x = copy.deepcopy(config)
    for key in ('job_id', 'experiment_name', 'provenance', 'h_controls', 'output_dir'):
        x.pop(key)
    x['factorial'].pop('arm')
    for files in list(x['data']['env_files'].values()) + list(x['target_data']['data']['env_files'].values()):
        files.pop('phase')
    for records in x['source_file_records'].values():
        records.pop('phase')
    x['target_data']['file_records'].pop('phase')
    return x

def verify_bundle():
    require(HERE.resolve() == (BUNDLE / 'code').resolve(), 'Execute the frozen bundle/code copy')
    manifest = read(BUNDLE / 'bundle_manifest.json')
    for rel, expected in manifest['files_sha256'].items():
        require(sha(BUNDLE / rel) == expected, 'Frozen link-unwrap file changed: ' + rel)
    registry = read(BUNDLE / 'registry.json')
    expected = {('H-S', e, s, a) for e in TARGETS for s in SEEDS for a in ARMS}
    actual = [(j['partition'], j['target'], j['seed'], j['control_arm']) for j in registry['jobs']]
    require(len(actual) == JOBS and len(set(actual)) == JOBS and set(actual) == expected, 'Wrong unique H-S 9-job grid')
    for job in registry['jobs']:
        config = read(BUNDLE / job['config'])
        require(config['job_id'] == job['job_id'], 'Registry/config job mismatch')
        verify_config(config)
    return manifest, registry

def verify_config(config):
    c = config['h_controls']
    require(c['partition'] == 'H-S' and c['arm'] in ARMS, 'Unplanned control arm')
    require(c['pool_mode'] == 'tpp' and c['phase_variant'] == 'native_link_unwrap', 'Arm definition drift')
    require(config['factorial']['arm'] == 'HS_N_L11' and config['factorial']['supcon_on'] is True, 'Factorial arm drift')
    # The paired configuration was frozen and executed by the v2 control study.
    require(sha(c['paired_control_config']) == c['paired_control_config_sha256'], 'Paired N_H11 config changed')
    paired = read(c['paired_control_config'])
    p = paired['h_controls']
    require(p['arm'] == PAIRED_ARM and p['phase_variant'] == 'native_historical', 'Paired control is not N_H11')
    require(paired['target_env'] == config['target_env'] and paired['seed_list'] == config['seed_list'], 'Paired cell drift')
    require(phase_free(config) == phase_free(paired), 'Change beyond the phase arrays relative to paired N_H11')
    for key in ('reference_config_sha256', 'initial_state_sha256', 'common_state_sha256', 'expected_sampler_sha256',
                'sampler_global_ids_file_sha256', 'source_manifest_sha256', 'new_hyperparameter_search_trials',
                'target_used_previously_for_development', 'new_environment_experiment', 'pool_mode', 'partition'):
        require(c[key] == p[key], 'Paired control field drift: ' + key)
    for key in ('reference_config', 'sampler_global_ids_file', 'source_manifest'):
        require(sha(c[key]) == c[key + '_sha256'], 'Bundled control file changed: ' + key)
    reference = read(c['reference_config'])
    require(config['training'] == reference['training'] and config['split'] == reference['split'], 'Historical training/split drift')
    require(config['model'] == reference['model'], 'Backbone/fusion/TPP specification drift')
    require(config['protocol']['expected_split_hashes'] == reference['protocol']['expected_split_hashes'], 'Membership drift')
    require(config['protocol']['checkpoint_rule'] == 'source_val_accuracy_last_tied_best', 'Selection drift')
    require(config['protocol']['total_optimizer_steps'] == 30000, 'Budget drift')
    require(config['protocol']['expected_parameter_count'] == PARAMETERS, 'Capacity drift')
    require(config['losses']['lambda_supcon'] == .05 and config['losses']['temperature'] == .2, 'Contrastive coefficient drift')
    require(config['losses']['supcon_positive_mode'] == 'global_class_instance_views', 'Positive rule drift')
    require(config['losses']['lambda_supcon_warmup_epochs'] == 0, 'Warm-up drift')
    require(config['data']['unwrap_phase'] is False and config['target_data']['data']['unwrap_phase'] is False,
            'Preprocessed phase must not be unwrapped again')
    alignment = read(BUNDLE / 'link_alignment.json')
    require(alignment['status'] == 'passed' and alignment['rows'] == 9000 and alignment['variant'] == LINK_VARIANT,
            'Link-unwrap input audit incomplete')
    bindings = alignment['file_records']
    require(list(config['data']['env_files']) == config['protocol']['source_envs'], 'Source routing/order drift')
    for env, records in config['source_file_records'].items():
        require(records == bindings[env], 'Source receipt differs from audited link-unwrap arrays')
        require(config['data']['env_files'][env] == {k: records[k]['path'] for k in ('amp', 'phase', 'label')},
                'Loader/source hash receipt path mismatch')
    target = config['target_env']
    require(list(config['target_data']['data']['env_files']) == [target], 'Deferred target routing drift')
    require(config['target_data']['file_records'] == bindings[target], 'Target receipt differs from audited link-unwrap arrays')
    require(config['target_data']['data']['env_files'][target] == {k: bindings[target][k]['path'] for k in ('amp', 'phase', 'label')},
            'Loader/target hash receipt path mismatch')
    for env in list(config['source_file_records']) + [target]:
        old = paired['source_file_records'].get(env) or paired['target_data']['file_records']
        require({k: old[k] for k in ('amp', 'label')} == {k: bindings[env][k] for k in ('amp', 'label')},
                'Amplitude/label receipts differ from paired N_H11')
    require(config['provenance']['legacy_commit'] == paired['provenance']['legacy_commit'], 'Legacy commit drift')
    require(config['provenance']['tpp_models_sha256'] == paired['provenance']['tpp_models_sha256'], 'TPP code drift')
    require({k: v['sha256'] for k, v in config['provenance']['legacy_modules'].items()}
            == {k: v['sha256'] for k, v in paired['provenance']['legacy_modules'].items()}, 'Legacy module drift')
    require(config['model']['antenna_layout'] == {'mode':'tx','num_rx':1,'num_tx':3,'num_subcarriers':30}, 'Pseudo-Tx slice drift')
    return reference
