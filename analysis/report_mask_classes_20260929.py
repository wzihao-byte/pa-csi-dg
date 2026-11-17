"""Recover classwise contrasts for all three frozen masked-view rules."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'docs/revision/review_20260924'
OUTPUT = ROOT / 'docs/revision/mask_classwise_20260929'
ARMS = ('MASK_HYBRID', 'MASK_ALL', 'MASK_REMOVE')
NAMES = ('No movement', 'Falling', 'Walking', 'Sitting / standing', 'Turning', 'Picking up a pen')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(name, rows):
    with (OUTPUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    with (SOURCE / 'recomputed_runs.csv').open(encoding='utf-8', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['study'] == 'controls' and r['arm'] in ARMS]
    expected = {(t, str(s)) for t, b in (('E1', 400), ('E2', 500), ('E3', 600)) for s in range(b + 1, b + 4)}
    by_arm = {a: {(r['environment'], r['seed']): r for r in rows if r['arm'] == a} for a in ARMS}
    if len(rows) != 27 or any(set(v) != expected for v in by_arm.values()):
        raise RuntimeError('Need all 27 prespecified runs and paired seeds')
    for row in rows:
        if abs(statistics.mean(float(row[f'C{i}_f1_percent']) for i in range(6)) - float(row['macro_f1_percent'])) > 1e-10:
            raise RuntimeError('Class mean does not reproduce the run macro-F1')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paired, means = [], []
    contrasts = [('MASK_ALL', 'MASK_HYBRID'), ('MASK_REMOVE', 'MASK_HYBRID'), ('MASK_ALL', 'MASK_REMOVE')]
    for treatment, control in contrasts:
        for target, seed in sorted(expected):
            for c, name in enumerate(NAMES):
                row = {'comparison': treatment + '-' + control, 'environment': target, 'seed': int(seed), 'class_id': c, 'activity': name}
                for metric in ('f1', 'precision', 'recall'):
                    row[metric + '_difference_pp'] = float(by_arm[treatment][target, seed][f'C{c}_{metric}_percent']) - float(by_arm[control][target, seed][f'C{c}_{metric}_percent'])
                paired.append(row)
        for target in ('E1', 'E2', 'E3', 'equal_environment_mean'):
            for c, name in enumerate(NAMES):
                group = [r for r in paired if r['comparison'] == treatment + '-' + control and r['class_id'] == c and (target == 'equal_environment_mean' or r['environment'] == target)]
                row = {'comparison': treatment + '-' + control, 'environment': target, 'class_id': c, 'activity': name,
                       'paired_runs': len(group), 'positive_f1_differences': sum(r['f1_difference_pp'] > 0 for r in group)}
                row.update({metric + '_mean_difference_pp': statistics.mean(r[metric + '_difference_pp'] for r in group) for metric in ('f1', 'precision', 'recall')})
                means.append(row)
    save_csv('all_paired_class_differences.csv', paired)
    save_csv('class_difference_summary.csv', means)
    table = ['| 活动 | All−Hybrid | Remove−Hybrid | All−Remove |', '|---|---:|---:|---:|']
    for c, name in enumerate(NAMES):
        values = [next(r['f1_mean_difference_pp'] for r in means if r['class_id'] == c and r['environment'] == 'equal_environment_mean' and r['comparison'] == a + '-' + b) for a, b in contrasts]
        table.append('| ' + name + ' | ' + ' | '.join(f'{v:+.2f}' for v in values) + ' |')
    text = '''# 三种 masked-view 规则：类别变化补充

从已有27次运行的逐运行类别指标恢复，无新增训练或目标前向推理。下表为类别F1的配对变化（百分点），先按每个环境内三个种子平均，再对三个环境等权平均；完整162条类别配对差和各环境结果见CSV，负差未删除。

''' + '\n'.join(table) + '''

相对Hybrid，All-view在坐下/站起（+8.70）和摔倒（+5.72）的平均改善最大；Remove-unselected在捡笔（+4.80）和坐下/站起（+4.62）的改善最大。All-view与Remove-unselected之间没有逐类别一致的排序：前者在坐下/站起和摔倒更高，而后者在捡笔、不活动、行走和转身更高。六类平均差全正不表示每个环境或每个种子的差都为正；逐对正差计数单独保存在汇总CSV。

这些结果描述匹配规则下的识别变化，不把“从分母排除”或“增加正样本”直接解释为已被证明的目标域机制。不使用H/Y分数来排序规则，也不从总体差推断某个具体样本流向。

可用于英文结果节的文字：

Under the matched four-view protocol, all-view same-class positives improved classwise F1 over the hybrid rule most for sitting/standing (+8.70 percentage points) and falling (+5.72). Excluding unselected same-class pairs from the denominator produced its largest improvements for picking up a pen (+4.80) and sitting/standing (+4.62). The two alternatives did not show a uniform classwise ordering: all-view positives were better for sitting/standing and falling, whereas the exclusion rule was better for the other four classes. These are descriptive paired changes averaged over the predefined seeds and development environments; they do not establish the mechanism of target-domain improvement.
'''
    (OUTPUT / 'classwise_report.md').write_text(text, encoding='utf-8')
    report = {'passed': True, 'run_count': 27, 'paired_class_count': len(paired), 'new_training': False, 'new_target_inference': False,
              'source_files_sha256': {str(SOURCE / n): digest(SOURCE / n) for n in ('recomputed_runs.csv', 'H00_H10_H11_classwise.csv', 'evidence_manifest.json')},
              'script_sha256': digest(Path(__file__)), 'artifacts_sha256': {p.name: digest(p) for p in OUTPUT.iterdir() if p.is_file() and p.name != 'manifest.json'}}
    (OUTPUT / 'manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'passed': True, 'paired_classes': len(paired), 'output': str(OUTPUT)}))


if __name__ == '__main__':
    main()
