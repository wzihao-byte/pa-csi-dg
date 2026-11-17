"""Three-class (falling/walking/turning) four-arm study on the main H-S split."""
from __future__ import annotations
import copy, hashlib, importlib.util, json, os, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

import three_class as tc

HERE = Path(__file__).resolve().parent
ROOT = Path(r'C:/Users/howie/Desktop/pa-csi-dg')
OUTPUT = Path(r'D:/pa-csi-dg-revision-runtime/studies/h_three_class_4s_20261117_v1')
BUNDLE = OUTPUT / 'bundle'
MAIN_V2 = Path(r'D:/pa-csi-dg-revision-runtime/studies/h_main_controls_20261008_v2/bundle')
TARGETS = ('E1', 'E2', 'E3')
SEEDS = (42, 52, 62)
# arm code -> (paper name, pool mode, multiview SupCon on); queue order within a target-seed block
ARMS = {'G10': ('G-CE', 'global_mlp', False), 'G11': ('G-SC', 'global_mlp', True),
        'T10': ('T-CE', 'tpp', False), 'T11': ('T-SC', 'tpp', True)}
PARAMETERS = 4_785_523   # 4,786,294 - (256*6+6) + (256*3+3); checked by instantiation in prepare.py


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
    return load_module('three_class_base_' + kind, HERE / ('base_' + kind + '.py'))


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def job_grid():
    """Queue order: every target-seed block runs its four arms together, so partial
    results are always complete pairs."""
    return [(t, s, a) for s in SEEDS for t in TARGETS for a in ARMS]


def verify_bundle():
    require(HERE.resolve() == (BUNDLE / 'code').resolve(), 'Execute the frozen bundle/code copy')
    manifest = read(BUNDLE / 'bundle_manifest.json')
    for rel, expected in manifest['files_sha256'].items():
        require(sha(BUNDLE / rel) == expected, 'Frozen file changed: ' + rel)
    registry = read(BUNDLE / 'registry.json')
    actual = [(j['target'], j['seed'], j['control_arm']) for j in registry['jobs']]
    require(actual == job_grid(), 'Wrong 36-job grid or order')
    for job in registry['jobs']:
        config = read(BUNDLE / job['config'])
        require(config['job_id'] == job['job_id'], 'Registry/config job mismatch')
        verify_config(config)
    return manifest, registry


def verify_config(config):
    c = config['h_controls']; t = config['three_class']
    reference = read(c['reference_config'])
    require(sha(c['reference_config']) == c['reference_config_sha256'], 'Main-study reference config changed')
    arm_name, pool, supcon = ARMS[c['arm']]
    require(config['training'] == reference['training'], 'Training settings differ from the main study')
    require(config['model'] == reference['model'], 'Model specification differs from the main study')
    require(config['seed_list'] == reference['seed_list'] and config['target_env'] == reference['target_env'], 'Seed/target drift')
    require(config['data'] == reference['data'] and config['target_data'] == reference['target_data'], 'Input drift')
    require(config['source_file_records'] == reference['source_file_records'], 'Input hash drift')
    expected_losses = copy.deepcopy(reference['losses'])
    expected_losses['supcon_positive_mode'] = 'global_class_instance_views'
    expected_losses['lambda_supcon'] = .05 if supcon else 0.
    require(config['losses'] == expected_losses, 'Unexpected loss change')
    require(config['factorial']['supcon_on'] == supcon and config['factorial']['supcon_views'] == (4 if supcon else 1),
            'Arm objective/view definition drift')
    require(config['factorial']['tpp_on'] == (pool == 'tpp') and c['pool_mode'] == pool, 'Arm pooling drift')
    require(c['phase_variant'] == 'historical' and c['partition'] == 'H-S', 'Only original inputs on H-S')
    p = config['protocol']
    require(p['epochs'] == tc.EPOCHS and p['optimizer_steps_per_epoch'] == tc.STEPS_PER_EPOCH
            and p['total_optimizer_steps'] == tc.TOTAL_STEPS, 'Budget drift')
    require(p['target_prediction_count'] == tc.TARGET_COUNT, 'Target size drift')
    require(p['expected_parameter_count'] == PARAMETERS, 'Capacity drift')
    require(p['checkpoint_rule'] == 'source_val_accuracy_last_tied_best', 'Selection drift')
    require(t['keep_original_labels'] == list(tc.KEEP) and t['class_names'] == list(tc.CLASS_NAMES), 'Class set drift')
    require(config['model']['antenna_layout'] == {'mode': 'tx', 'num_rx': 1, 'num_tx': 3, 'num_subcarriers': 30},
            'Pseudo-Tx slice drift')
    return reference
