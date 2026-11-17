"""Recompute the two development matrices without modifying their frozen artifacts.

This analysis reports environment means and within-environment seed dispersion.
It intentionally performs no hypothesis test of an unseen-environment population.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/revision/evidence"
ASSETS = ROOT / "paper/sensors_manuscript/revision_assets"
H_BASE = ROOT / "outputs_supcon_tpp_legacy_reproduction/factorial_s32/factorial_results.json"
H10 = ROOT / "outputs_tpp_level8_ablation/l1248_h10_v1/level8_results.json"
H11 = ROOT / "outputs_tpp_level8_ablation/l1248_v1/level8_results.json"
Y_ROOT = ROOT / "outputs_supcon_tpp_joint/formal_interim_v12"
Y_RESULTS = Y_ROOT / "formal_results.json"
DECISION = ROOT / "outputs_supcon_tpp_joint/interim_decision_v12/interim_decision.json"
SUPPORT = np.array([1200, 400, 400, 400, 400, 200])


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def macro_f1(cm):
    cm = np.asarray(cm, dtype=np.int64)
    assert cm.shape == (6, 6) and (cm >= 0).all()
    assert np.array_equal(cm.sum(axis=1), SUPPORT), "Unexpected target class support"
    denominator = cm.sum(axis=0) + cm.sum(axis=1)
    return float(np.mean(np.divide(2 * np.diag(cm), denominator, out=np.zeros(6), where=denominator != 0)))


def historical_rows():
    rows = [row for row in read_json(H_BASE)["rows"] if row["arm"] in ("H00", "H01")]
    rows += read_json(H10)["rows"] + read_json(H11)["rows"]
    output = []
    for row in rows:
        cm_path = Path(row["attempt_dir"]) / "payload/run/confusion_matrix.csv"
        score = macro_f1(np.loadtxt(cm_path, delimiter=",", dtype=np.int64))
        assert abs(score - row["test_metrics"]["f1_macro"]) < 1e-12
        output.append(dict(protocol="H_historical", environment=row["target_env"], seed=int(row["seed"]),
                           arm=row["arm"], macro_f1_percent=100 * score, evidence=str(cm_path.relative_to(ROOT))))
    return output


def grouped_rows():
    results = read_json(Y_RESULTS)
    assert results["completed_job_count"] == 36
    registry = {r["job_id"]: r for r in csv.DictReader((Y_ROOT / "run_registry.csv").open(encoding="utf-8"))}
    output, audits, identities = [], [], {}
    for row in results["results"]:
        run = registry[row["job_id"]]
        assert run["status"] == "completed"
        folder = Path(run["output_dir"])
        prediction_path = folder / "target_predictions.jsonl"
        assert sha256(prediction_path) == row["target_predictions_file_sha256"]
        predictions = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line]
        assert len(predictions) == 3000
        cm = np.zeros((6, 6), dtype=np.int64)
        identity = []
        for pred in predictions:
            assert pred["environment"] == row["target_env"]
            assert 0 <= pred["true"] < 6 and 0 <= pred["pred"] < 6
            cm[pred["true"], pred["pred"]] += 1
            identity.append((pred["sample_id"], pred["true"], pred["group_id"], pred["subject_id"]))
        assert len({p[0] for p in identity}) == 3000
        env = row["target_env"]
        assert identities.setdefault(env, identity) == identity, "Unpaired target samples"
        assert np.array_equal(cm, row["target_metrics"]["confusion_matrix"])
        score = macro_f1(cm)
        assert abs(score - row["target_metrics"]["f1_macro"]) < 1e-12
        split = read_json(folder / "split_manifest.json")
        config = read_json(folder / "resolved_config.json")
        audits.append(dict(environment=env, arm=row["arm"], seed=row["seed"],
                           split=split["inner_validation"], subject_sidecars_complete=split["subject_sidecars_complete"],
                           phase_unwrap=config["data"]["unwrap_phase"],
                           prediction_sha256=sha256(prediction_path), elapsed_seconds=float(run["elapsed_seconds"])))
        output.append(dict(protocol="Y_grouped", environment=env, seed=int(row["seed"]), arm=row["arm"],
                           macro_f1_percent=100 * score, evidence=str(prediction_path.relative_to(ROOT))))
    return output, audits


def summarize(rows):
    environment, effects = [], []
    for protocol in sorted({r["protocol"] for r in rows}):
        subset = [r for r in rows if r["protocol"] == protocol]
        arms = sorted({r["arm"] for r in subset})
        assert len(arms) == 4 and len(subset) == 36
        for env in ("E1", "E2", "E3"):
            for arm in arms:
                values = [r["macro_f1_percent"] for r in subset if r["environment"] == env and r["arm"] == arm]
                assert len(values) == 3
                environment.append(dict(protocol=protocol, environment=env, arm=arm, n_seeds=3,
                                        mean_percent=float(np.mean(values)), seed_sd_pp=float(np.std(values, ddof=1))))
            seeds = sorted({r["seed"] for r in subset if r["environment"] == env})
            for seed in seeds:
                values = {r["arm"]: r["macro_f1_percent"] for r in subset if r["environment"] == env and r["seed"] == seed}
                assert set(values) == set(arms)
                a, b, c, d = (values[arm] for arm in arms)
                effects.append(dict(protocol=protocol, environment=env, seed=seed,
                                    tpp_with_ce_pp=c-a, tpp_with_contrastive_pp=d-b,
                                    contrastive_on_control_pp=b-a, contrastive_on_tpp_pp=d-c,
                                    interaction_pp=d-c-b+a))
    return environment, effects


def render_outputs(rows, environment, effects, audits):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ASSETS.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 16, "axes.spines.top": False, "axes.spines.right": False})
    colors = ["#637185", "#4775a8", "#c07827", "#387a65"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), sharey=True)
    for ax, protocol, title in zip(axes, ("H_historical", "Y_grouped"), ("(a) Historical H protocol", "(b) Grouped Y protocol")):
        arms = sorted({r["arm"] for r in rows if r["protocol"] == protocol})
        for index, arm in enumerate(arms):
            stats = [r for r in environment if r["protocol"] == protocol and r["arm"] == arm]
            x = np.arange(3) + (index - 1.5) * 0.15
            ax.errorbar(x, [r["mean_percent"] for r in stats], yerr=[r["seed_sd_pp"] for r in stats],
                        fmt="o", markersize=5, capsize=3, color=colors[index], label=arm)
        ax.set(xticks=range(3), xticklabels=["E1", "E2", "E3"], title=title, ylim=(45, 100), xlabel="Development environment")
        ax.grid(axis="y", alpha=.2)
        ax.legend(ncol=2, fontsize=15)
    axes[0].set_ylabel("Macro-F1 (%)\nMean and seed SD")
    fig.tight_layout()
    fig.savefig(ASSETS / "protocol_comparison.png", dpi=350)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.5))
    for ax, key, title in zip(axes, ("contrastive_on_tpp_pp", "interaction_pp"),
                              ("(a) Contrastive objective on TPP", "(b) Factorial interaction")):
        for index, protocol in enumerate(("H_historical", "Y_grouped")):
            means = [np.mean([r[key] for r in effects if r["protocol"] == protocol and r["environment"] == env]) for env in ("E1", "E2", "E3")]
            ax.plot(range(3), means, "o-", label=protocol[0] + " protocol", color=colors[index*2])
        ax.axhline(0, color="#777777", linewidth=.8)
        ax.set(title=title, xticks=range(3), xticklabels=["E1", "E2", "E3"], ylabel="Paired difference (pp)")
        ax.legend()
        ax.grid(axis="y", alpha=.2)
    fig.tight_layout()
    fig.savefig(ASSETS / "protocol_effects.png", dpi=350)
    plt.close(fig)
    tex = [r"\begin{table}[tbp]", r"\caption{Separate development protocols: macro-F1 (\%), environment mean $\pm$ seed SD. The overall column is the equal-environment mean, without population inference.\label{tab:revisionresults}}",
           r"\small", r"\begin{tabularx}{\textwidth}{lXXXX}", r"\toprule", r"Arm & E1 & E2 & E3 & Overall\\", r"\midrule"]
    for arm in sorted({r["arm"] for r in rows}):
        stats = [r for r in environment if r["arm"] == arm]
        cells = [f"${r['mean_percent']:.2f}\\pm{r['seed_sd_pp']:.2f}$" for r in stats]
        tex.append(arm + " & " + " & ".join(cells) + f" & {np.mean([r['mean_percent'] for r in stats]):.2f}" + r"\\")
        if arm == "H11":
            tex.append(r"\midrule")
    tex += [r"\bottomrule", r"\end{tabularx}", r"\end{table}"]
    (ASSETS / "protocol_results.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    lines = ["# 修订证据审计", "", "本报告由 `python analysis/revision_evidence.py` 从冻结记录重算。H/Y 是不同开发协议，不合并结果，也不以跨协议差值归因。", "",
             "- H：36 个目标混淆矩阵重算 macro-F1。", "- Y：36 份目标预测（108,000 行）核对 SHA256、样本身份、标签、分组、混淆矩阵和 macro-F1。",
             "- 每环境的三个 seed 只描述固定测试集上的优化波动；没有环境总体 p 值或置信区间。", "",
             "| Protocol | Arm | E1 mean ± seed SD | E2 | E3 | Equal-environment mean |", "|---|---|---|---|---|---|"]
    for arm in sorted({r["arm"] for r in rows}):
        stats = [r for r in environment if r["arm"] == arm]
        cells = [f"{r['mean_percent']:.2f} ± {r['seed_sd_pp']:.2f}" for r in stats]
        lines.append(f"| {stats[0]['protocol']} | {arm} | " + " | ".join(cells) + f" | {np.mean([r['mean_percent'] for r in stats]):.2f} |")
    lines += ["", "| Protocol | Environment | TPP with CE (pp) | Contrastive on TPP (pp) | Interaction (pp) |", "|---|---|---:|---:|---:|"]
    for protocol in ("H_historical", "Y_grouped"):
        for env in ("E1", "E2", "E3", "equal_environment_mean"):
            group = [r for r in effects if r["protocol"] == protocol and (env == "equal_environment_mean" or r["environment"] == env)]
            lines.append(f"| {protocol} | {env} | " + " | ".join(f"{np.mean([r[key] for r in group]):+.2f}" for key in ("tpp_with_ce_pp", "contrastive_on_tpp_pp", "interaction_pp")) + " |")
    decision = read_json(DECISION)
    lines += ["", "## 停止记录与解释", "", f"Y 的保存判定为 `{decision['decision']}`；触发项为 `{', '.join(decision['analysis']['triggered_condition_ids'])}`。",
              "该停止规则是固定三环境、给定模型假设下的内部决策，不是未知环境总体的显著性检验。保持原停止记录，不补 seed 寻找正结果。",
              "两协议同时改变了 phase、split、结构、loss/view、batch 与训练步数；不能把分数差归因于其中一项。TPP 相对各自对照的方向一致，正交互并不跨协议保持。",
              "", "## 身份核验与后续证据", ""]
    identity_path = OUT / "verified_subject_audit.json"
    if identity_path.exists():
        identity = read_json(identity_path)
        assert identity["all_9000_verified"] and identity["Y_run_count"] == 36
        lines += ["- 全量 9000 行原始信号映射已核验。另行核对 Y 的 36 个冻结 split：样本集合哈希一致，真实 subject 和 acquisition 分组不交叉。",
                  "- 这是事后身份核验；原 manifest 仍保持 inferred_group_block_holdout，未回写历史记录，也未改变开发环境的性质。"]
    else:
        lines += ["- 真实 sample/recording/subject 映射尚待全量核验；Y 原始记录为 inferred_group_block_holdout。"]
    lines += ["- 时间戳精度、相位处理仍需独立验证。原始路径可重放不等于物理校准；关闭 unwrap 也不构成校准。",
              "- 同新协议的常规小模型 ERM、模态对照、常规容量控制、外部 DG 基线。",
              "- 冻结方法之后的新目标环境。", "", f"Y 运行登记平均耗时：{np.mean([a['elapsed_seconds'] for a in audits])/3600:.2f} h/run；不是新模型的保证预算。"]
    (OUT.parent / "EVIDENCE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def audit_arrays():
    config = read_json(ROOT / "experiments/supcon_tpp_joint/generated_formal_interim_v12/formal_interim_configs/formal_interim__target_E1__arm_Y11__seed_101.json")
    paths = dict(config["data"]["env_files"])
    paths["E1"] = config["target_data"]["env_files"]
    output = {"scope": "previously inspected MultiEnv development data only", "subject_identity_verified": False,
              "raw_to_processed_mapping_verified": False, "physical_phase_calibration_verified": False, "environments": {}}
    for env in sorted(paths):
        records = {}
        labels = np.load(paths[env]["label"], allow_pickle=False)
        assert labels.shape == (3000,) and np.isin(labels, np.arange(6)).all()
        for block in range(10):
            assert np.array_equal(np.bincount(labels[block*300:(block+1)*300].astype(int), minlength=6), SUPPORT//10)
        for modality in ("amp", "phase", "label"):
            path = Path(paths[env][modality])
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            minimum, maximum, invalid = float("inf"), -float("inf"), 0
            for start in range(0, len(array), 32):
                batch = np.asarray(array[start:start+32])
                invalid += int(np.count_nonzero(~np.isfinite(batch)))
                minimum, maximum = min(minimum, float(np.nanmin(batch))), max(maximum, float(np.nanmax(batch)))
            records[modality] = {"path": str(path), "sha256": sha256(path), "shape": list(array.shape),
                                 "dtype": str(array.dtype), "nonfinite_count": invalid, "min": minimum, "max": maximum}
            assert invalid == 0
        output["environments"][env] = {"files": records, "contiguous_label_template_verified": True,
                                        "groups": 10, "samples_per_group": 300, "subject_mapping": None}
    write_json(OUT / "array_audit.json", output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-arrays", action="store_true", help="Stream the previously inspected MultiEnv files for hashes and basic validity.")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    y_rows, audits = grouped_rows()
    rows = historical_rows() + y_rows
    class_means = {}
    for arm in ("H00", "H11"):
        class_scores = []
        for row in rows:
            if row["arm"] == arm:
                cm = np.loadtxt(ROOT / row["evidence"], delimiter=",")
                class_scores.append(200 * np.diag(cm) / (cm.sum(axis=0) + cm.sum(axis=1)))
        class_means[arm] = np.mean(class_scores, axis=0).tolist()
    write_json(OUT / "h_class_f1.json", {"aggregation":"Mean per-run class F1 over the balanced target/seed matrix; descriptive only",
                                        "arm_means_percent":class_means,
                                        "H11_minus_H00_pp":(np.array(class_means["H11"])-class_means["H00"]).tolist()})
    level_path = ROOT / "outputs_tpp_level8_ablation/l1248_v1/analysis/level8_analysis.json"
    scale_path = ROOT / "outputs_tpp_residual_scale_ablation/h11_l1248_v1/analysis/residual_scale_analysis.json"
    level, scale = read_json(level_path), read_json(scale_path)
    write_json(OUT / "historical_secondary_summaries.json", {
        "scope":"Extracted descriptive summaries of saved historical analyses; not additional independently rechecked runs in the 72-run audit.",
        "source_hashes":{str(p.relative_to(ROOT)):sha256(p) for p in (level_path,scale_path)},
        "level8_parameters":level["parameters"], "level8_means":level["overall_arm_summary"],
        "level8_environment_deltas":level["environment_macro_f1_deltas"], "coefficient_curve":scale["scale_curve"]})
    environment, effects = summarize(rows)
    write_csv(OUT / "per_run.csv", rows)
    write_csv(OUT / "environment_summary.csv", environment)
    write_csv(OUT / "paired_effects.csv", effects)
    write_json(OUT / "verification.json", {"historical_confusion_matrices_checked": 36, "grouped_prediction_files_checked": 36,
                                          "grouped_prediction_rows_checked": 108000, "grouped_runs": audits,
                                          "input_hashes": {str(p.relative_to(ROOT)): sha256(p) for p in (H_BASE,H10,H11,Y_RESULTS,DECISION)}})
    render_outputs(rows, environment, effects, audits)
    if args.audit_arrays:
        audit_arrays()
    print(f"Verified 72 runs; descriptive evidence written to {OUT.parent}")


if __name__ == "__main__":
    main()
