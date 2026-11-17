"""Read-only audit of reusable R evidence for the 18 missing time/global controls.

No signal array is opened, no optimizer step is taken, and no old artifact is
modified. Synthetic CPU forward/backward checks count active CE parameters.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDIES = Path("D:/pa-csi-dg-revision-runtime/studies")
CORE = STUDIES / "r_core_v1"
INPUTS = STUDIES / "r_inputs_v1"
OUT = ROOT / "docs/revision/time_validation_20260929"
# Import the already frozen, hash-verified scientific implementation.
sys.dont_write_bytecode = True
sys.path.insert(0, str(INPUTS / "code"))

import torch
from torch.nn import functional as F

from experiments.revision_validation.experiment_design import build_experiment_model, objective
from experiments.revision_validation.extended_engine import verify_extension
from experiments.revision_validation.models import build_model
from experiments.revision_validation.source_data import epoch_schedule, schedule_hash
from experiments.revision_validation.study_io import now, read, sha256, value_hash, verify_bundle, verify_completed, write

IDENTITY_KEYS = ("sample_id", "environment", "index", "label", "subject_id", "acquisition_id")
FIXED_KEYS = (
    "stage", "target_name_only", "seed", "model", "epochs", "updates_per_epoch", "batch_size", "device",
    "optimizer", "learning_rate", "weight_decay", "scheduler", "lambda_supcon", "temperature", "warmup_steps",
    "checkpoint_metric", "tie_rule", "expected_runtime", "target_data_loaded", "hyperparameter_search_trials",
    "development_environments", "phase_loaded",
)
PROJECTION_PREFIXES = ("encoder.temporal_pool.projection.", "encoder.channel_pool.projection.")


def identities(rows):
    return [{key: row[key] for key in IDENTITY_KEYS} for row in rows]


def main():
    torch.set_num_threads(4)
    core, core_hash = verify_bundle(CORE)
    inputs, inputs_hash = verify_extension(INPUTS)
    code_checks = {}
    scientific_files = [
        "dg_dataset.py", "dg_models.py", "dg_losses.py", "experiments/tpp_isolated/tpp_models.py",
        "experiments/revision_validation/models.py", "experiments/revision_validation/source_data.py",
        "experiments/revision_validation/study_engine.py", "experiments/revision_validation/study_io.py",
        "experiments/revision_validation/train_source.py", "experiments/revision_validation/experiment_data.py",
        "experiments/revision_validation/experiment_design.py", "experiments/revision_validation/extended_engine.py",
    ]
    for relative in scientific_files:
        hashes = {label: sha256(base / relative) for label, base in (
            ("core", CORE / "code"), ("inputs", INPUTS / "code"), ("working_tree", ROOT)
        ) if (base / relative).exists()}
        code_checks[relative] = {"sha256": hashes, "identical": len(set(hashes.values())) == 1}
        assert code_checks[relative]["identical"], relative

    by_core = {(c["target_name_only"], c["seed"], c["arm"]): c for c in core["jobs"]}
    by_input = {(c["target_name_only"], c["seed"], c["control_id"]): c for c in inputs["jobs"]}
    pairs, verified_runs, init_checks = [], {}, []
    ordered_identity_checks, synthetic = {}, {}
    for target, base in (("E1", 400), ("E2", 500), ("E3", 600)):
        hist_source = read(CORE / "manifests" / f"source_{target}.json")
        hist_target = read(CORE / "manifests" / f"deferred_target_{target}.json")
        identity_record = {}
        for control, variant in (("NATIVE_MERGE", "native_time_legacy_merge"), ("NATIVE_LINEAR", "native_time_linear")):
            native_source = read(INPUTS / "manifests" / variant / f"source_{target}.json")
            native_target = read(INPUTS / "manifests" / variant / f"deferred_target_{target}.json")
            assert native_source["data_version"] == variant
            assert native_source["source_environments"] == hist_source["source_environments"]
            for partition in ("train", "val"):
                assert identities(native_source[partition]) == identities(hist_source[partition])
                for old, new in zip(hist_source[partition], native_source[partition]):
                    assert old["raw_path"] == new["identity_source_mat_path"]
                    assert old["raw_sha256"] == new["identity_source_mat_sha256"]
            assert identities(native_target["test"]) == identities(hist_target["test"])
            identity_record[control] = {
                "source_manifest_sha256": sha256(INPUTS / "manifests" / variant / f"source_{target}.json"),
                "target_manifest_sha256": sha256(INPUTS / "manifests" / variant / f"deferred_target_{target}.json"),
                "ordered_train_val_test_identities_equal": True,
                "original_MAT_identity_mapping_verified": True,
                "counts": {"train": len(native_source["train"]), "val": len(native_source["val"]), "test": len(native_target["test"])},
            }
        ordered_identity_checks[target] = identity_record
        for seed in range(base + 1, base + 4):
            configs = [by_core[(target, seed, "R00")], by_core[(target, seed, "R10")],
                       by_input[(target, seed, "NATIVE_MERGE")], by_input[(target, seed, "NATIVE_LINEAR")]]
            histories = []
            for config in configs:
                bundle = INPUTS if "control_id" in config else CORE
                run = bundle / "runs" / config["job_id"]
                assert read(run / "config.json") == config
                status = verify_completed(run, config)
                verified_runs[config["job_id"]] = {
                    "bundle": str(bundle), "config_value_sha256": value_hash(config),
                    "status_sha256": sha256(run / "status.json"), "optimizer_updates": status["optimizer_updates"],
                    "best_epoch": status["best_epoch"], "artifacts_and_earliest_source_max_verified": True,
                }
                histories.append(read(run / "history.json"))
            for config in configs[1:]:
                for key in FIXED_KEYS:
                    assert config[key] == configs[0][key], (target, seed, key)
            regenerated = []
            for epoch in range(200):
                expected = schedule_hash(epoch_schedule(hist_source["train"], seed, epoch, steps=300))
                assert all(history[epoch]["schedule_sha256"] == expected for history in histories)
                assert all(history[epoch]["optimizer_steps"] == (epoch + 1) * 300 for history in histories)
                assert len({history[epoch]["learning_rate_after_epoch"] for history in histories}) == 1
                regenerated.append(expected)
            # No optimizer step. Model construction stays on CPU.
            global_model, tpp_model = build_model("R00", seed), build_model("R10", seed)
            left, right = global_model.state_dict(), tpp_model.state_dict()
            common = [key for key in right if not key.startswith(PROJECTION_PREFIXES)]
            assert all(torch.equal(left[key], right[key]) for key in common)
            assert sum(p.numel() for p in global_model.parameters()) == sum(p.numel() for p in tpp_model.parameters()) == 1_878_142
            init_checks.append({"target": target, "seed": seed, "common_tensors_exactly_equal": len(common),
                                "total_parameters_each": 1_878_142, "projection_head_parameters_each": 98_688,
                                "classification_parameters_each": 1_779_454})
            for config in configs[2:]:
                candidate = copy.deepcopy(config)
                candidate["arm"] = "R00"
                candidate["experiment"]["base_arm"] = "R00"
                candidate["control_id"] += "_GLOBAL"
                candidate["job_id"] = f"{target}_TIME_{candidate['control_id']}_{seed}"
                candidate_model = build_experiment_model(candidate)
                assert all(torch.equal(candidate_model.state_dict()[key], value) for key, value in left.items())
                pairs.append({"target": target, "seed": seed, "input_variant": config["input_variant"],
                              "reference_job": config["job_id"], "reference_config_value_sha256": value_hash(config),
                              "fixed_keys_equal_to_core_R00": list(FIXED_KEYS),
                              "all_200_epoch_schedules_regenerated_and_equal": True,
                              "schedule_sequence_sha256": value_hash(regenerated),
                              "proposed_model_exactly_equals_R00_initialization": True})
            print(f"Verified {target}/{seed}: four completed runs, 200 paired epoch schedules, common initialization", flush=True)

    # Independent objective/gradient check on synthetic CPU input. Preserve the
    # RNG state so the direct CE reference sees precisely the same dropout.
    reference_config = copy.deepcopy(by_input[("E1", 401, "NATIVE_LINEAR")])
    rows = [{"environment": "E2"}, {"environment": "E3"}]
    generator = torch.Generator().manual_seed(99029)
    x = torch.randn(2, 850, 90, generator=generator)
    y = torch.tensor([0, 1])
    for arm in ("R00", "R10"):
        config = copy.deepcopy(reference_config)
        config["arm"] = config["experiment"]["base_arm"] = arm
        model = build_experiment_model(config)
        direct = build_model(arm, config["seed"])
        torch.manual_seed(401 + 17)
        rng = torch.get_rng_state()
        loss, ce, regularizer, weight, _ = objective(model, {"amp": x}, y, rows, 1, config)
        torch.set_rng_state(rng)
        expected = F.cross_entropy(direct(x, project=False)["logits"], y)
        assert torch.equal(loss, expected) and torch.equal(ce, expected)
        assert float(regularizer) == 0.0 and weight == 0.0
        loss.backward()
        active = sum(p.numel() for p in model.parameters() if p.grad is not None)
        assert active == 1_779_454
        assert all(p.grad is None for p in model.projection_head.parameters())
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        synthetic[arm] = {"direct_CE_exact": True, "active_CE_parameters": active, "unused_projection_parameters": 98_688,
                          "finite_gradients": True, "optimizer_steps": 0, "device": "cpu", "signal_data_loaded": False}

    recommendations = [
        "Create a new study; never add runs or mutate protocols in r_core_v1 or r_inputs_v1.",
        "Copy the input-study frozen scientific code. Use extended_engine.train_experiment and the VerifiedInputs adapter.",
        "Set BOTH arm and experiment.base_arm to R00; pool remains unchanged, objective ce, model_kind single. No head_kind field is consumed.",
        "Copy each B/C source and deferred-target manifest without modifying contents. Bind their SHA256 in each new configuration.",
        "Keep total_parameters 1878142. Verify active CE/inference classification count 1779454; the 98688 projection-head parameters are unused by CE.",
        "Keep all nine matched fold/seed combinations, 200 epochs x 300 updates, batch 16, fixed Adam/LR, and earliest source-validation macro-F1 checkpoint.",
        "Sampler is fixed at 16 samples: eight per source environment, every class per source, epoch-seeded independently of model initialization.",
        "Do not replace cached native arrays or cast/resample them again. VerifiedInputs hashes the actual files and casts batches to float32.",
        "Resource profiling uses a separate model and no optimizer updates. Training RNG resets to seed+17 after constructing the training model.",
        "Use a separate new run directory per configuration. A resume requires exact config/runtime/model/optimizer/RNG hashes; recorded failures must be diagnosed.",
        "Require every one of the 18 new source jobs to finish and verify before new target evaluation. Existing E1-E3 scores have been observed, so this is development extension evidence.",
        "The stock extended evaluator pairs only same-study or core-reference results; it does not directly resolve historical input-study TPP references. Add explicit cross-bundle keyed pairing after verifying all reference hashes.",
        "Historical A uses MAT provenance; native B/C use DAT provenance. A-to-B includes representation/origin differences and is not a pure isolated timestamp manipulation.",
        "B-to-C shares native clock and record identities but changes merge/interpolation processing; primary inference compares TPP-minus-global within each input version.",
        "CUDA adaptive max-pool backward remains nondeterministic. Matched seeds and common initialization do not imply bitwise replay on GPU.",
    ]
    report = {"schema": "time-validation-protocol-audit/v1", "checked_at": now(), "passed": True,
              "audit_script_sha256": sha256(Path(__file__)), "target_signal_loaded": False,
              "source_signal_loaded": False, "optimizer_steps_performed": 0,
              "core_bundle_sha256": core_hash, "inputs_bundle_sha256": inputs_hash,
              "scientific_code_checks": code_checks, "ordered_identity_checks": ordered_identity_checks,
              "verified_completed_runs": verified_runs, "proposed_pairs": pairs,
              "shared_initialization": init_checks, "synthetic_CE_checks": synthetic,
              "recommendations_and_limits": recommendations}
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "protocol_audit.json", report)
    lines = ["# 时间预处理补充实验：协议独立核查", "", "检查日期：2026-09-29", "",
             "结论：18 个 native global-MLP 补充训练可复用已冻结 R00 模型及 inputs 引擎。核查通过；本检查未启动训练、未打开源/目标信号数组、未修改历史文件。", "",
             "## 已验证证据", "",
             "- 核对 core、inputs 的完整冻结文件哈希，36 个相关已完成 run 的配置、结果附件和源验证最优 epoch。",
             "- 3 个环境、9 个配对种子，四个历史分支各 200 epoch 的采样哈希均一致；独立重算全部 1,800 个 schedule，与记录一致。",
             "- 历史与 native 的 train/val/test 样本身份、标签、环境、索引、被试和采集顺序一致；native 回指原 MAT 文件及哈希验证通过。",
             "- 每折 4,800 个源训练样本、1,200 个源验证样本、3,000 个目标样本。",
             "- 9 个种子的非池化投影公共张量完全一致；新 global 配置构造结果与 R00 完全一致。",
             "- R00/TPP 总参数均为 1,878,142；CE 有梯度分类参数均为 1,779,454；98,688 个训练用 projection-head 参数在 CE 中未使用。",
             "- CPU 合成输入检查：extension objective 与直接 CE 完全相同，梯度有限，优化器步数为 0。", "",
             "## 实施边界", ""]
    lines.extend(f"{i}. {text}" for i, text in enumerate(recommendations, 1))
    lines += ["", "## 新增矩阵", "", "|输入版本|模型|种子|新增训练|", "|---|---|---|---:|",
              "|native_time_legacy_merge|R00 / global-MLP + CE|E1:401–403; E2:501–503; E3:601–603|9|",
              "|native_time_linear|R00 / global-MLP + CE|同上|9|", "",
              "这份审计证明方案可配对，不预判训练结果；全部环境和全部种子均应保留。", ""]
    (OUT / "protocol_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"passed": True, "verified_runs": len(verified_runs), "proposed_new_runs": len(pairs),
                      "schedule_epochs_regenerated": 1800, "report": str(OUT / "protocol_audit.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
