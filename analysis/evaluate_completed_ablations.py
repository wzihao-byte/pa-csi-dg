"""Read-only verification and descriptive review of completed R core/controls.

Never starts training, opens target signal arrays, or selects a new model.
Only writes a separate review artifact; unfinished studies are excluded.
"""
from __future__ import annotations

import argparse
import json
import statistics as stats
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.compile_formal_revision_evidence import verify_study
from experiments.revision_validation.evaluate_study import metrics_for
from experiments.revision_validation.extended_engine import verify_extension
from experiments.revision_validation.study_io import now, read, sha256, value_hash, verify_bundle

COMPARISONS = [
    ("R10", "R00"), ("R11", "R01"), ("R01", "R00"), ("R11", "R10"),
    ("R10", "P1"), ("R10", "NO_P1"), ("R10", "ATTENTION"),
    ("MASK_HYBRID", "R11"), ("MASK_ALL", "MASK_HYBRID"),
    ("MASK_REMOVE", "MASK_HYBRID"), ("MASK_ALL", "MASK_REMOVE"),
    ("MASK_ALL", "R10"), ("CORAL", "R00"), ("MIXUP", "R00"),
    ("TEMPORAL_AUG", "R10"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path("D:/pa-csi-dg-revision-runtime"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the previous review; use a new output directory")
    rows, evidence, configurations = [], {}, {}
    for name, verifier, count in [("r_core_v1", verify_bundle, 36), ("r_controls_v1", verify_extension, 81)]:
        bundle = args.runtime / "studies" / name
        protocol, digest, statuses, gate, summary, summary_hash = verify_study(bundle, verifier, count)
        configs = {c["job_id"]: c for c in protocol["jobs"]}
        for row in summary["runs"]:
            config = configs[row["job_id"]]
            directory = bundle / "evaluation" / row["job_id"]
            assert row == read(directory / "result.json")
            assert row["config_value_sha256"] == value_hash(config)
            assert row["checkpoint_sha256"] == sha256(bundle / "runs" / row["job_id"] / "best_model.pt")
            predictions = read(directory / "predictions.json")
            assert len(predictions) == row["sample_count"] == 3000
            assert metrics_for(predictions) == row["metrics"]
            assert row["source_predictions_reproduced"]
            assert statuses[row["job_id"]]["optimizer_updates"] == 60000
            rows.append(row)
            configurations.setdefault(row["arm"], config)
        evidence[name] = {"bundle": str(bundle), "bundle_sha256": digest,
                          "evaluation_summary_sha256": summary_hash,
                          "verified_jobs": count, "predictions_recomputed": count,
                          "target_gate_sha256": sha256(bundle / "target_gate.json")}
        print(f"Verified {name}: {count} runs and saved target predictions", flush=True)
    lookup = {(r["arm"], r["target"], r["seed"]): r for r in rows}
    assert len(lookup) == len(rows) == 117
    arms = list(dict.fromkeys(r["arm"] for r in rows))
    scores = []
    for arm in arms:
        selected = [r for r in rows if r["arm"] == arm]
        environments = {}
        for target in ("E1", "E2", "E3"):
            values = [r["metrics"]["macro_f1"] * 100 for r in selected if r["target"] == target]
            assert len(values) == 3
            environments[target] = {"mean_percent": stats.mean(values), "seed_sd_pp": stats.stdev(values)}
        scores.append({"arm": arm, "environments": environments,
                      "mean_percent": stats.mean(e["mean_percent"] for e in environments.values()),
                      "accuracy_percent": stats.mean(r["metrics"]["accuracy"] * 100 for r in selected),
                      "source_macro_f1_percent": stats.mean(r["source_metrics"]["macro_f1"] * 100 for r in selected)})
    effects = []
    for treatment, reference in COMPARISONS:
        pairs = []
        for row in rows:
            if row["arm"] != treatment:
                continue
            other = lookup[(reference, row["target"], row["seed"])]
            pairs.append({"target": row["target"], "seed": row["seed"],
                          "delta_pp": 100 * (row["metrics"]["macro_f1"] - other["metrics"]["macro_f1"])})
        assert len(pairs) == 9
        effects.append({"treatment": treatment, "reference": reference,
                        "mean_pp": stats.mean(p["delta_pp"] for p in pairs),
                        "positive_pairs": sum(p["delta_pp"] > 0 for p in pairs),
                        "environment_mean_pp": {e: stats.mean(p["delta_pp"] for p in pairs if p["target"] == e)
                                                for e in ("E1", "E2", "E3")}, "pairs": pairs})
    interaction = []
    for row in rows:
        if row["arm"] == "R00":
            key = (row["target"], row["seed"])
            v = {a: lookup[(a, *key)]["metrics"]["macro_f1"] for a in ("R00", "R01", "R10", "R11")}
            interaction.append({"target": key[0], "seed": key[1],
                                "delta_pp": 100 * (v["R11"] - v["R10"] - v["R01"] + v["R00"])})
    report = {"recorded_at": now(), "scope": "Completed amplitude-only R L124 core and controls; excludes incomplete inputs and H L1248",
              "evidence": evidence, "scores": scores, "effects": effects,
              "interaction_mean_pp": stats.mean(p["delta_pp"] for p in interaction), "interaction_pairs": interaction,
              "population_inference": False,
              "configuration_examples": configurations,
              "runs": [{"job_id": r["job_id"], "arm": r["arm"], "target": r["target"], "seed": r["seed"],
                        "macro_f1_percent": r["metrics"]["macro_f1"] * 100} for r in rows]}
    lines = ["# 已完成消融实验评估（R 核心 + 控制）", "", f"核验时间：{report['recorded_at']}", "",
             "核验 36+81 次训练、冻结配置/代码、最佳 checkpoint、目标评估门和评估文件哈希，并从保存的 3000 条/次目标预测重算指标。未重新评估模型或读取目标信号。",
             "", "范围：幅度单流、TPP L124、200×300 更新、源验证 macro-F1 最早最优；不混入历史 H/Y、未完成输入实验或 H L1248。",
             "", "## 结果", "", "各环境为三种子均值 ± 样本 SD；总均值对三个环境等权。单位为百分数，SD 单位为百分点。", "",
             "| Arm | E1 | E2 | E3 | 等权均值 |", "|---|---:|---:|---:|---:|"]
    for score in scores:
        cells = [f"{score['environments'][e]['mean_percent']:.2f} ± {score['environments'][e]['seed_sd_pp']:.2f}" for e in ("E1", "E2", "E3")]
        lines.append(f"| {score['arm']} | " + " | ".join(cells) + f" | {score['mean_percent']:.2f} |")
    lines += ["", "## 配对差异", "", "同目标环境、同种子相减；正配对数仅为描述，不是独立样本显著性检验。", "",
              "| 比较（前者减后者） | 平均百分点 | E1 | E2 | E3 | 正配对/9 |", "|---|---:|---:|---:|---:|---:|"]
    for effect in effects:
        cells = [f"{effect['environment_mean_pp'][e]:+.2f}" for e in ("E1", "E2", "E3")]
        lines.append(f"| {effect['treatment']} − {effect['reference']} | {effect['mean_pp']:+.2f} | " + " | ".join(cells) + f" | {effect['positive_pairs']}/9 |")
    lines += ["", "## 解释", "",
              "- TPP 在 CE 和 global-only SupCon 下均有 9/9 正配对，对等参数 global MLP 的收益较一致；也优于本次等参数 attention 对照。不能推广为优于所有 attention 实现或纯时间机制已获证明。",
              "- global-only SupCon 的增量不稳定。正的差分交互不等于 SupCon 在 TPP 上存在正收益，也不足以支持稳定协同。",
              "- 完整 L124 明显优于仅 P1，但 P1-only 参数更少；去除金字塔 P1 后整体差距小、完整 L124 仅 3/9 正配对。去 P1 仍保留全局 residual，不能称为去除全部全局信息。",
              "- 四视图 MASK_ALL 和 MASK_REMOVE 均优于 MASK_HYBRID，均为 8/9 正配对；这不支持 hybrid 必要性或优越性。MASK_ALL 与 MASK_REMOVE 的差距小，应报告逐环境差异。",
              "- 三个 mask 臂共享相同视图、两次 forward 和 BatchNorm 暴露，可解释为固定配置下的 mask 比较。与 R11/R10 比较同时改变视图与训练暴露，不能单独归因于 mask 或对比损失。",
              "- pseudo-Tx 是三个连续 30 维软件特征组，不是三根物理 Tx。",
              "- CORAL/Mixup 比较基线为 R00，随机循环平移比较基线为 R10；仅代表本次固定强度、无新增搜索的实现。循环平移下降不证明时间不变性成立。",
              "- 三个种子仅反映固定划分上的训练随机性，E1–E3 曾用于开发；不报告总体 p 值/置信区间，也不将结果宣称为独立跨数据集确认。不得以目标域表中最佳项反向选择方法并称源域选择。",
              "", f"2×2 平均差分交互：{report['interaction_mean_pp']:+.3f} 个百分点。", "",
              "## 证据", ""]
    for name, ev in evidence.items():
        lines += [f"- {name}: `{ev['bundle']}`", f"  - bundle SHA-256: `{ev['bundle_sha256']}`",
                  f"  - evaluation summary SHA-256: `{ev['evaluation_summary_sha256']}`"]
    args.output.mkdir(parents=True)
    (args.output / "review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / "review.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"scores": scores, "effects": [{k: v for k, v in e.items() if k != "pairs"} for e in effects],
                      "interaction_mean_pp": report["interaction_mean_pp"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
