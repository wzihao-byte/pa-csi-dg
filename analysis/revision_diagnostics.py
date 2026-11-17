"""Read saved Y diagnostics and measure trained TPP norms on source data only.

No training, target inference, model selection, or frozen-artifact modification.
The fixed source validation sample is descriptive and is not an independent test.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.revision_evidence import OUT, Y_ROOT, read_json, sha256, write_csv, write_json


def summarize_gradients(registry):
    rows, summaries, provenance = [], [], []
    for run in registry:
        if run["arm"] not in ("Y01", "Y11"):
            continue
        folder = Path(run["output_dir"])
        metrics_path = folder / "metrics.json"
        assert sha256(metrics_path) == run["metrics_file_sha256"]
        history = read_json(folder / "history.json")
        metrics = read_json(metrics_path)
        assert history == metrics["resources"]["history"]
        epochs = [row for row in history if "gradient_diagnostic" in row]
        assert [row["epoch"] for row in epochs] == list(range(5, 201, 5))
        provenance.append(dict(job_id=run["job_id"], history_sha256=sha256(folder / "history.json"),
                               metrics_sha256=run["metrics_file_sha256"],
                               fixed_source_batch_ids=metrics["resources"]["batch_schedule_audit"]["first_batch_sample_ids"]))
        for row in epochs:
            report = row["gradient_diagnostic"]
            for block in ("amplitude_encoder", "phase_encoder", "fusion"):
                value = report["blocks"][block]
                cosine = value["gradient_cosine"]
                assert cosine is not None and np.isfinite(cosine) and abs(cosine) <= 1 + 1e-8
                rows.append(dict(arm=run["arm"], held_out_environment=run["target_env"], seed=int(run["seed"]),
                                 epoch=row["epoch"], block=block, lambda_supcon=report["lambda_supcon"],
                                 gradient_cosine=cosine, weighted_supcon_to_ce_ratio=value["weighted_supcon_to_ce_ratio"],
                                 ce_grad_norm=value["ce_grad_norm"], supcon_grad_norm=value["supcon_grad_norm"]))
        for block in ("amplitude_encoder", "phase_encoder", "fusion"):
            values = [r for r in rows if r["arm"] == run["arm"] and r["held_out_environment"] == run["target_env"]
                      and r["seed"] == int(run["seed"]) and r["block"] == block]
            summaries.append(dict(arm=run["arm"], held_out_environment=run["target_env"], seed=int(run["seed"]),
                                  block=block, observations=len(values),
                                  median_cosine=float(np.median([r["gradient_cosine"] for r in values])),
                                  negative_fraction=float(np.mean([r["gradient_cosine"] < 0 for r in values])),
                                  median_weighted_ratio=float(np.median([r["weighted_supcon_to_ce_ratio"] for r in values]))))
    assert len(rows) == 18 * 40 * 3
    write_csv(OUT / "gradient_observations.csv", rows)
    write_csv(OUT / "gradient_run_summary.csv", summaries)
    write_json(OUT / "gradient_provenance.json", provenance)
    return summaries


def measure_residuals(registry, device):
    import torch
    from experiments.supcon_tpp_joint.joint_models import build_joint_model

    torch.set_num_threads(4)
    arrays, rows, provenance = {}, [], []
    for run in registry:
        if run["arm"] not in ("Y10", "Y11"):
            continue
        folder = Path(run["output_dir"])
        checkpoint = folder / "best_model.pt"
        assert sha256(checkpoint) == run["best_model_file_sha256"]
        config = read_json(folder / "resolved_config.json")
        split = read_json(folder / "split_manifest.json")
        assert config["data"]["source_only"] and not config["data"]["unwrap_phase"]
        assert run["target_env"] not in config["data"]["env_files"]
        model = build_joint_model(run["arm"], 90, 6, config["model"], initialization_seed=int(run["seed"]))
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model.to(device).eval()
        selected_ids = []
        with torch.inference_mode():
            for env, files in sorted(config["data"]["env_files"].items()):
                if env not in arrays:
                    arrays[env] = {key: np.load(files[key], mmap_mode="r", allow_pickle=False) for key in ("amp", "label")}
                ids = []
                for group in split["validation_group_ids"][env]:
                    block = int(group.split(":B")[1])
                    block_ids = np.arange(block * 300, (block + 1) * 300)
                    for label in range(6):
                        ids.extend(block_ids[arrays[env]["label"][block_ids] == label][:2].tolist())
                assert len(ids) == 24 and len(set(ids)) == 24
                selected_ids.extend(f"{env}:{index:04d}" for index in ids)
                for start in range(0, len(ids), 8):
                    indices = ids[start:start + 8]
                    amplitude = torch.from_numpy(np.asarray(arrays[env]["amp"][indices], dtype=np.float32)).to(device)
                    path_inputs = model.amp_encoder.encode_pool_inputs(amplitude)
                    for name, pool in model.amp_encoder.residual_pools().items():
                        global_feature, pyramid = pool.extract_residual_inputs(path_inputs[name])
                        residual = 0.25 * pool.projection(pyramid)
                        # Check the measured expression against the real eval-mode module.
                        assert torch.allclose(pool(path_inputs[name]), global_feature + residual, rtol=1e-5, atol=1e-5)
                        global_norm = torch.linalg.vector_norm(global_feature, dim=1).double().cpu().numpy()
                        residual_norm = torch.linalg.vector_norm(residual, dim=1).double().cpu().numpy()
                        for index, g_norm, r_norm in zip(indices, global_norm, residual_norm):
                            assert np.isfinite(g_norm) and np.isfinite(r_norm)
                            rows.append(dict(arm=run["arm"], held_out_environment=run["target_env"], seed=int(run["seed"]),
                                             source_environment=env, source_array_index=index, label=int(arrays[env]["label"][index]),
                                             path=name, global_norm=float(g_norm), scaled_residual_norm=float(r_norm),
                                             residual_to_global_ratio=float(r_norm / (g_norm + 1e-12))))
        provenance.append(dict(job_id=run["job_id"], checkpoint_sha256=run["best_model_file_sha256"],
                               resolved_config_sha256=sha256(folder / "resolved_config.json"),
                               split_manifest_sha256=sha256(folder / "split_manifest.json"), selected_source_ids=selected_ids,
                               source_file_hashes={env:split["file_sha256"][env] for env in config["data"]["env_files"]}))
        print(f"Measured {run['arm']} / {run['target_env']} / {run['seed']}: {len(selected_ids)} source validation samples", flush=True)
        del model
    summaries = []
    for run in registry:
        if run["arm"] not in ("Y10", "Y11"):
            continue
        for path in ("temporal", "channel"):
            values = [r for r in rows if r["arm"] == run["arm"] and r["held_out_environment"] == run["target_env"]
                      and r["seed"] == int(run["seed"]) and r["path"] == path]
            assert len(values) == 48
            ratios = [r["residual_to_global_ratio"] for r in values]
            summaries.append(dict(arm=run["arm"], held_out_environment=run["target_env"], seed=int(run["seed"]),
                                  path=path, samples=48, median_ratio=float(np.median(ratios)),
                                  p10_ratio=float(np.quantile(ratios, .1)), p90_ratio=float(np.quantile(ratios, .9)),
                                  fraction_above_one=float(np.mean(np.asarray(ratios) > 1)),
                                  zero_global_norm_count=sum(r["global_norm"] <= 1e-12 for r in values)))
    write_csv(OUT / "residual_sample_norms.csv", rows)
    write_csv(OUT / "residual_run_summary.csv", summaries)
    write_json(OUT / "residual_provenance.json", {"epsilon":1e-12, "device":str(device),
               "sampling":"First two rows of each class in each of two recorded source-validation blocks, per source environment; 48 per checkpoint.",
               "target_inference":False, "runs":provenance})
    return summaries


def write_report(gradients, residuals):
    lines = ["# 保存模型的机制诊断", "", "诊断为事后描述，不用来选择模型或改写已有停止判定。", "",
             "## 已保存的梯度诊断", "", "核对 18 个对比学习 run 的 history 与已哈希 metrics 完全一致。每 run 在固定的 16 个源训练样本上，每 5 epoch 测一次，共 40 次；eval 模式关闭 dropout。时间点不是独立重复。下表先求每 run 的时间中位数，再报告 9 个 run 中位数的范围。负梯度占比也先在 run 内计算。", "",
             "| Arm | 参数模块 | Run 内余弦中位数范围 | Run 内负余弦比例范围 | 加权 SC/CE 梯度范数比中位数范围 |", "|---|---|---|---|---|"]
    for arm in ("Y01", "Y11"):
        for block in ("amplitude_encoder", "phase_encoder", "fusion"):
            subset = [r for r in gradients if r["arm"] == arm and r["block"] == block]
            def span(key):
                values = [r[key] for r in subset]
                return f"{min(values):.3f}–{max(values):.3f}"
            lines.append(f"| {arm} | {block} | {span('median_cosine')} | {span('negative_fraction')} | {span('median_weighted_ratio')} |")
    lines += ["", "分类头、投影头没有共同损失梯度，空的 shared_backbone 桶也不产生有意义的余弦；均未纳入表。固定源 batch 上的梯度冲突不能单独解释目标环境退化。", "", "## 已训练 TPP 的实际残差", "",
              "严格加载并核对 18 份 Y10/Y11 checkpoint 的 SHA256。每模型仅使用两个源环境的验证块，每块每类取前 2 条，合计 48 条。输入、权重和 alpha 均不改变；逐路径核验实际 eval forward 等于 g + 0.25(Wz+b)。下表为 9 个 run 的样本范数比中位数范围。", "",
              "| Arm | 路径 | Run 中位数范围 | Run 内 r > 1 比例范围 |", "|---|---|---|---|"]
    for arm in ("Y10", "Y11"):
        for path in ("temporal", "channel"):
            subset = [r for r in residuals if r["arm"] == arm and r["path"] == path]
            ratios = [r["median_ratio"] for r in subset]
            fractions = [r["fraction_above_one"] for r in subset]
            lines.append(f"| {arm} | {path} | {min(ratios):.3f}–{max(ratios):.3f} | {min(fractions):.3f}–{max(fractions):.3f} |")
    lines += ["", "r = ||0.25(Wz+b)||₂ / (||g||₂ + 1e-12)。该量是实际已训练 Y 模型的测量，不能代替 H 模型的结果；不是全数据分布估计、范数上界或 alpha 最优值证明。", "",
              "逐样本、逐 run 表与选择索引、模型/配置哈希保存在 evidence/residual_*.csv/json；梯度原始观测和汇总在 evidence/gradient_*.csv/json。"]
    (OUT.parent / "MECHANISM_DIAGNOSTICS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    registry = list(csv.DictReader((Y_ROOT / "run_registry.csv").open(encoding="utf-8")))
    assert len(registry) == 36 and all(r["status"] == "completed" for r in registry)
    gradients = summarize_gradients(registry)
    residuals = measure_residuals(registry, args.device)
    write_report(gradients, residuals)


if __name__ == "__main__":
    main()
