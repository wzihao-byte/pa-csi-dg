"""Enforce the September 14 H/Y writing scope independently of R technical QA."""
import hashlib
import json
from pathlib import Path
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root: Path, evidence: Path, assets: Path):
    values = json.loads((assets / 'formal_narrative_values.json').read_text())
    assert values['schema'] == 'hy-writing-scope/v1'
    assert values['r_scores_in_manuscript'] is False
    summary = json.loads((evidence / 'h_results.json').read_text())['summary']
    assert values['h_summary'] == summary
    lookup = {(r['target'],r['metric'],r['arm']):r for r in summary}
    pairs = values['all_signed_paired_deltas']
    assert len(pairs) == 9 and len(values['all_signed_historical_deltas']) == 36
    assert {(r['target'],r['seed']) for r in pairs} == {(e,s) for e in ('E1','E2','E3') for s in (42,52,62)}
    fragment = (assets / 'formal_results.tex').read_text(encoding='utf-8')
    for row in pairs:
        for metric in ('accuracy','f1_macro'):
            expected = lookup[row['target'],metric,'H11_minus_H10'][f"seed_{row['seed']}"]
            assert abs(row[metric] - expected) < 1e-10
            assert f'{expected:+.2f}' in fragment
    for row in values['all_signed_historical_deltas']:
        assert f"{float(row['grouped_minus_historical_pp']):+.2f}" in fragment
    paper = root / 'paper/sensors_manuscript'
    visited = set()
    def expand(path):
        path = path.resolve()
        assert path.is_relative_to(paper.resolve())
        if path in visited:
            return ''
        visited.add(path)
        text = path.read_text(encoding='utf-8')
        active = '\n'.join(re.split(r'(?<!\\)%', line)[0] for line in text.splitlines())
        for relative in re.findall(r'\\input\{([^}]+)\}', active):
            child = paper / relative
            if not child.suffix:
                child = child.with_suffix('.tex')
            active += '\n' + expand(child)
        return active
    prose = expand(paper / 'manuscript.tex')
    reply = (root / 'docs/revision/REVIEW_RESPONSE_DRAFT.md').read_text(encoding='utf-8')
    for forbidden in ('R00', 'R01', 'R10', 'R11', 'tab:formalcore', 'tab:formalcontrols', 'tab:formalinputs',
                      'formal_factorials', 'formal_control_effects', '48.80', '54.88', '54.60', '57.49'):
        assert forbidden not in prose + reply, f'Withdrawn R material in writing: {forbidden}'
    assert (assets / 'formal_results.tex').resolve() in visited
    for name in ('manuscript.tex','revision_body.tex'):
        text = (paper / name).read_text()
        for token in ('88.02','89.83','1.81'):
            assert token in text
    assert 'pseudo-Tx' in prose and 'not three physical transmitters' in prose
    assert not values['population_inference'] and not values['independent_confirmation']
    return {'writing_scope':'H/Y plus verified grouped-H; R technical audit only',
            'all_signed_paired_deltas':9,'all_signed_historical_metric_deltas':36,
            'scope_instruction_sha256':sha(root/'docs/revision/WRITING_SCOPE_UPDATE_20260914.md'),
            'scope_verifier_sha256':sha(Path(__file__)),
            'included_tex_sha256':{str(p.relative_to(root)).replace('\\','/'):sha(p) for p in sorted(visited)},
            'review_response_sha256':sha(root/'docs/revision/REVIEW_RESPONSE_DRAFT.md')}
