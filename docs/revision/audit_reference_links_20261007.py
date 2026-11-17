from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'paper/submission_versions_20261007'
VARIANTS = ['sensors', 'ieice_transactions', 'research_meeting', 'comex', 'ipsj89', 'ieice_general']

def expand(path, seen):
    path = path.resolve()
    if path in seen:
        return ''
    seen.add(path)
    text = path.read_text(encoding='utf-8-sig')
    text = re.sub(r'(?<!\\)%[^\n]*', '', text)
    def include(match):
        child = path.parent / match.group(1)
        if not child.suffix:
            child = child.with_suffix('.tex')
        if not child.is_file():
            return match.group(0)
        return expand(child, seen)
    return re.sub(r'\\(?:input|include)\{([^}]+)\}', include, text)

records = []
for variant in VARIANTS:
    seen = set()
    source = expand(BASE / variant / 'manuscript.tex', seen)
    cited = [k.strip() for group in re.findall(r'\\cite\w*\*?(?:\[[^]]*\])*\{([^}]+)\}', source) for k in group.split(',')]
    keys = re.findall(r'\\bibitem(?:\[[^]]*\])?\{([^}]+)\}', source)
    items = re.findall(r'\\bibitem(?:\[[^]]*\])?\{([^}]+)\}([\s\S]*?)(?=\\bibitem|\\end\{thebibliography\})', source)
    records.append({
        'variant': variant, 'references': len(keys), 'citation_keys': sorted(set(cited)),
        'missing_bibliography_keys': sorted(set(cited) - set(keys)),
        'uncited_bibliography_keys': sorted(set(keys) - set(cited)),
        'duplicate_bibliography_keys': sorted({k for k in keys if keys.count(k) > 1}),
        'bibliography': [{'key': k, 'entry': re.sub(r'\s+', ' ', v).strip()} for k, v in items],
        'sources_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(seen)}
    })
out = ROOT / 'docs/revision/reference_audit_20261007_links.json'
out.write_text(json.dumps({'checked_date': '2026-10-07', 'scope': 'Active LaTeX manuscript input graph, excluding unused template examples and unused supporting files.', 'variants': records}, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps([{k: r[k] for k in ['variant', 'references', 'missing_bibliography_keys', 'uncited_bibliography_keys', 'duplicate_bibliography_keys']} for r in records], indent=2))
