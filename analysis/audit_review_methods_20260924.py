"""Check manuscript architecture and mask semantics against bound experiment code."""
from pathlib import Path
import json
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.revision_validation.study_io import read, sha256, write
from experiments.tpp_isolated.tpp_models import IsolatedPaCsiDGLiteTPP
from experiments.revision_validation.models import build_model
from experiments.revision_validation.control_components import mask_control_loss

OUT = ROOT / 'docs/revision/review_20260924'


def main():
    torch.set_num_threads(4)
    torch.manual_seed(23)
    lock = read(ROOT / 'outputs_h_grouped_l1248/v2/bundle/bundle_manifest.json')
    checked = {}
    for filename in ('dg_models.py', 'dg_losses.py', 'experiments/tpp_isolated/tpp_models.py'):
        digest = sha256(ROOT / filename)
        matches = [(p, h) for p, h in lock['external_code_sha256'].items() if Path(p).name == Path(filename).name]
        assert len(matches) == 1 and digest == matches[0][1]
        checked[filename] = digest
    for filename in ('dg_models.py', 'experiments/tpp_isolated/tpp_models.py', 'experiments/revision_validation/models.py', 'experiments/revision_validation/control_components.py'):
        study = 'r_controls_v1' if filename.endswith('control_components.py') else 'r_core_v1'
        frozen = Path('D:/pa-csi-dg-revision-runtime/studies') / study / 'code' / filename
        assert sha256(frozen) == sha256(ROOT / filename), filename
        checked[str(frozen)] = sha256(frozen)
    result = read(ROOT / 'outputs_tpp_level8_ablation/l1248_v1/level8_results.json')['rows'][0]
    config_path = Path(result['attempt_dir']) / 'payload/run/resolved_config.json'
    config = read(config_path)
    model = IsolatedPaCsiDGLiteTPP(90, 6, config['model']).eval()
    shapes = {}
    handles = []
    for name, module in model.named_modules():
        if name in ('amp_encoder.position_encoding', 'amp_encoder.temporal_encoder',
                    'amp_encoder.temporal_pool', 'amp_encoder.channel_encoder', 'amp_encoder.channel_pool',
                    'amp_encoder', 'phase_encoder', 'fusion', 'classifier', 'projection_head') or name.startswith(('amp_encoder.temporal_pool.convs.', 'amp_encoder.channel_pool.convs.')):
            def hook(mod, inputs, output, name=name):
                shapes[name] = {'input': [list(v.shape) for v in inputs if isinstance(v, torch.Tensor)], 'output': list(output.shape)}
            handles.append(module.register_forward_hook(hook))
    x, phase = torch.randn(2,850,90), torch.randn(2,850,90)
    with torch.no_grad():
        model(x, phase)
    for handle in handles:
        handle.remove()
    assert shapes['amp_encoder.channel_encoder']['input'] == [[2,425,30]]
    assert shapes['amp_encoder.channel_pool']['input'] == [[2,425,30]]
    assert shapes['amp_encoder.channel_pool.convs.0']['output'] == [2,16,424]
    assert shapes['amp_encoder.channel_pool.convs.1']['output'] == [2,16,422]
    assert sum(p.numel() for p in model.parameters()) == 4786294
    with torch.no_grad():
        amp, ph = torch.randn(2,288), torch.randn(2,288)
        fusion = model.fusion
        a, p = fusion.amp_proj(amp), fusion.phase_proj(ph)
        gate = torch.sigmoid(fusion.gate_proj(fusion.norm(torch.nn.functional.elu(a+p))))
        assert torch.equal(fusion(amp, ph), gate*a+(1-gate)*p)
    counts = {}
    for arm in ('R00','R01','R10','R11'):
        m = build_model(arm, 401).eval()
        m(x, project=False)['logits'].sum().backward()
        counts[arm] = {'total_parameters':sum(p.numel() for p in m.parameters()),
                       'classification_graph_parameters':sum(p.numel() for p in m.parameters() if p.grad is not None),
                       'excluded_training_only_projection':sum(p.numel() for p in m.projection_head.parameters())}
    assert len({v['classification_graph_parameters'] for v in counts.values()}) == 1
    labels = torch.tensor([0,0,1])
    embeddings = torch.randn(3,4,8, requires_grad=True)
    checks = {}
    for mode in ('hybrid','all_view','hybrid_remove_same_class_nonpositives'):
        loss, diagnostics, positive, allowed = mask_control_loss(embeddings, labels, mode)
        # Independent enumerated definition: instance-major (sample, view) order.
        expected_positive = torch.zeros(12,12,dtype=torch.bool)
        expected_allowed = torch.zeros_like(expected_positive)
        for i in range(12):
            for j in range(12):
                s,v = divmod(i,4)
                t,u = divmod(j,4)
                same_class = labels[s] == labels[t]
                hybrid = i != j and (s == t or (same_class and v == 0 and u == 0))
                expected_positive[i,j] = i != j and same_class if mode == 'all_view' else hybrid
                expected_allowed[i,j] = i != j and not (mode == 'hybrid_remove_same_class_nonpositives' and same_class and not hybrid)
        assert torch.equal(positive.cpu(), expected_positive)
        assert torch.equal(allowed.cpu(), expected_allowed)
        assert torch.isfinite(loss)
        loss.backward(retain_graph=True)
        assert torch.isfinite(embeddings.grad).all()
        checks[mode] = diagnostics
    controls = read(Path('D:/pa-csi-dg-revision-runtime/studies/r_controls_v1/protocol.json'))['jobs']
    config_comparisons = 0
    for baseline in [c for c in controls if c['control_id']=='MASK_HYBRID']:
        for mode in ('MASK_ALL','MASK_REMOVE'):
            changed = next(c for c in controls if c['control_id']==mode and c['seed']==baseline['seed'] and c['target_name_only']==baseline['target_name_only'])
            def common_config(c):
                c = json.loads(json.dumps(c))
                c.pop('job_id'); c.pop('control_id'); c['experiment'].pop('mask')
                return c
            assert common_config(changed) == common_config(baseline)
            config_comparisons += 1
    write(OUT / 'method_checks.json', dict(status='passed', input='synthetic tensors only; no training updates',
          h_config_path=str(config_path), h_config_sha256=sha256(config_path), code_sha256=checked,
          shapes=shapes, channel_pool_axis='temporal response length; not feature/subcarrier axis',
          fusion_equation_exact=True, r_parameter_counts=counts, enumerated_mask_checks=checks,
          mask_config_pair_comparisons=config_comparisons, mask_config_only_scientific_difference='experiment.mask',
          script_sha256=sha256(Path(__file__))))
    print(json.dumps({'status':'passed','shapes':shapes,'R_counts':counts,'masks':checks}, indent=2))


if __name__ == '__main__':
    main()
