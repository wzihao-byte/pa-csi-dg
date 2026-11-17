"""Measure all four historical H TPP paths using saved models and source inputs."""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.revision_evidence import H10, H11, OUT, read_json, sha256, write_csv, write_json
from experiments.tpp_isolated.tpp_models import IsolatedPaCsiDGLiteTPP


def main():
    torch.set_num_threads(4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rows, provenance, datasets = [], [], {}
    runs = read_json(H10)["rows"] + read_json(H11)["rows"]
    for run in runs:
        folder = Path(run["attempt_dir"]) / "payload/run"
        checkpoint = folder / "best_model.pt"
        assert sha256(checkpoint) == run["checkpoint_sha256"]
        config = read_json(folder / "resolved_config.json")
        assert run["target_env"] not in config["data"]["env_files"]
        assert config["data"]["unwrap_phase"] and config["data"]["phase_unwrap_axis"] == -1
        model = IsolatedPaCsiDGLiteTPP(90, 6, config["model"])
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model.to(device).eval()
        context, selected = {}, []

        def make_hook(modality, path):
            def hook(pool, inputs, output):
                assert pool.preserve_global and pool.pyramid_scale == .25 and pool.levels == [1,2,4,8]
                parts, globals_ = [], []
                for conv in pool.convs:
                    response = pool.activation(conv(inputs[0].transpose(1,2)))
                    globals_.append(F.adaptive_max_pool1d(response,1).flatten(start_dim=1))
                    parts.extend(pool.pool_response(response,level).flatten(start_dim=1) for level in pool.levels)
                g = torch.cat(globals_,dim=1)
                residual = .25 * pool.projection(torch.cat(parts,dim=1))
                assert torch.allclose(output,g+residual,rtol=1e-5,atol=1e-5)
                gn = torch.linalg.vector_norm(g,dim=1).double().cpu().numpy()
                rn = torch.linalg.vector_norm(residual,dim=1).double().cpu().numpy()
                for index,label,gval,rval in zip(context["indices"],context["labels"],gn,rn):
                    assert np.isfinite(gval) and np.isfinite(rval)
                    rows.append(dict(arm=run["arm"],held_out_environment=run["target_env"],seed=int(run["seed"]),
                                     source_environment=context["env"],source_array_index=int(index),label=int(label),
                                     modality=modality,path=path,global_norm=float(gval),scaled_residual_norm=float(rval),
                                     residual_to_global_ratio=float(rval/(gval+1e-12))))
            return hook

        handles = [getattr(encoder,path+"_pool").register_forward_hook(make_hook(modality,path))
                   for modality,encoder in (("amplitude",model.amp_encoder),("phase",model.phase_encoder))
                   for path in ("temporal","channel")]
        with torch.inference_mode():
            for env,files in sorted(config["data"]["env_files"].items()):
                if env not in datasets:
                    arrays = {kind:np.load(files[kind],mmap_mode="r",allow_pickle=False) for kind in ("amp","phase","label")}
                    rng = np.random.default_rng(20260903)
                    indices = np.concatenate([np.sort(rng.choice(np.flatnonzero(arrays["label"]==label),size=4,replace=False)) for label in range(6)])
                    datasets[env] = (arrays,indices)
                arrays,indices = datasets[env]
                selected.extend(f"{env}:{index:04d}" for index in indices)
                for start in range(0,len(indices),8):
                    batch = indices[start:start+8]
                    context.update(env=env,indices=batch,labels=arrays["label"][batch])
                    amp = np.asarray(arrays["amp"][batch],dtype=np.float32)
                    # Historical loader unwraps float64 features BEFORE float32 conversion.
                    phase = np.unwrap(np.asarray(arrays["phase"][batch]),axis=-1).astype(np.float32)
                    model.amp_encoder(torch.from_numpy(amp).to(device))
                    model.phase_encoder(torch.from_numpy(phase).to(device))
        for handle in handles:
            handle.remove()
        provenance.append(dict(arm=run["arm"],target=run["target_env"],seed=run["seed"],checkpoint_sha256=run["checkpoint_sha256"],
                               resolved_config_sha256=sha256(folder/"resolved_config.json"),
                               source_file_records=config["source_file_records"],selected_source_ids=selected))
        print(f"Measured H residuals: {run['arm']} / {run['target_env']} / {run['seed']}",flush=True)
        del model
    summaries = []
    for run in runs:
        for modality in ("amplitude","phase"):
            for path in ("temporal","channel"):
                subset = [r for r in rows if r["arm"]==run["arm"] and r["held_out_environment"]==run["target_env"]
                          and r["seed"]==run["seed"] and r["modality"]==modality and r["path"]==path]
                assert len(subset)==48
                ratios = [r["residual_to_global_ratio"] for r in subset]
                nonzero_ratios = [r["residual_to_global_ratio"] for r in subset if r["global_norm"]>1e-12]
                summaries.append(dict(arm=run["arm"],held_out_environment=run["target_env"],seed=run["seed"],modality=modality,path=path,
                                      samples=48,median_ratio=float(np.median(ratios)),p10_ratio=float(np.quantile(ratios,.1)),
                                      p90_ratio=float(np.quantile(ratios,.9)),fraction_above_one=float(np.mean(np.array(ratios)>1)),
                                      median_ratio_nonzero_global=float(np.median(nonzero_ratios)) if nonzero_ratios else None,
                                      nonzero_global_samples=len(nonzero_ratios),
                                      zero_global_norm_count=sum(r["global_norm"]<=1e-12 for r in subset)))
    assert len(rows)==18*48*4
    write_csv(OUT/"h_residual_sample_norms.csv",rows)
    write_csv(OUT/"h_residual_run_summary.csv",summaries)
    write_json(OUT/"h_residual_provenance.json",dict(sampling="Four uniformly sampled rows per class per source environment, numpy default_rng seed 20260903; source train and validation combined; not an independent test.",
                                                  target_inference=False,epsilon=1e-12,device=str(device),runs=provenance))
    lines = ["# 历史 H 模型的实际残差", "", "18 个 H10/H11 保存模型均严格加载且哈希一致。每 run 使用两个源环境，每环境每类固定随机抽 4 条（NumPy seed 20260903），共 48 条；范围含源 train/validation。仅作事后模型诊断，不用于选模型，不推断未见环境。H phase 输入按原 loader 先在 float64 上沿末轴 unwrap，再转 float32。", "",
             "逐路径 hook 重建全局特征和 alpha(Wz+b)，与实际 forward 输出核对一致。alpha=0.25，epsilon=1e-12。下表范围跨同一 arm 的 9 个 run；没有合并 H/Y 样本。", "",
             "部分 H10 相位路径的全局 ReLU 特征范数为零，含 epsilon 的比值因此很大，不能解释成正常残差相对强度。原始比值保留在 CSV，下表仅对非零全局特征计算 run 中位数；若整 run 全零则该 run 不贡献中位数。零全局范数数量单列。", "",
             "| Arm | 模态 | 路径 | 非零全局样本的 run 中位数范围 | 每 run 零全局范数数 / 48 |", "|---|---|---|---|---|"]
    for arm in ("H10","H11"):
        for modality in ("amplitude","phase"):
            for path in ("temporal","channel"):
                subset = [r for r in summaries if r["arm"]==arm and r["modality"]==modality and r["path"]==path]
                medians = [r["median_ratio_nonzero_global"] for r in subset if r["median_ratio_nonzero_global"] is not None]
                zeros = [r["zero_global_norm_count"] for r in subset]
                lines.append(f"| {arm} | {modality} | {path} | {min(medians):.3f}–{max(medians):.3f} | {min(zeros)}–{max(zeros)} |")
    (OUT.parent/"H_RESIDUAL_DIAGNOSTICS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__ == "__main__":
    main()
