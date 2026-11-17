"""Measure candidate implementation health/resources on verified source data.

No target loader, target predictions, model selection or reported efficacy score.
This smoke run is not a replacement for full source-selected baseline training.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from dg_losses import DomainAwareSupConLoss, supcon_positive_pair_diagnostics
from experiments.revision_validation.models import (
    ARMS, MODEL_CONFIG, build_model, count_matmul_conv_macs, shared_initialization_audit,
)
from experiments.revision_validation.prepare_verified_protocol import ROOT, OUT, sha256, write
from experiments.revision_validation.source_data import VerifiedSources, epoch_schedule, schedule_hash


def synchronize(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def initial_residual_norms(model, amplitude):
    cache, handles = {}, []
    for name in ("temporal_pool", "channel_pool"):
        pool = getattr(model.encoder, name)
        handles.append(pool.projection.register_forward_hook(
            lambda module, inputs, output, key=name, scale=pool.pyramid_scale:
            cache.__setitem__(key + ":residual", output.detach() * scale)))
        handles.append(pool.dropout.register_forward_pre_hook(
            lambda module, inputs, key=name: cache.__setitem__(key + ":combined", inputs[0].detach())))
    model.eval()
    try:
        with torch.no_grad():
            model(amplitude, project=False)
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for name in ("temporal_pool", "channel_pool"):
        residual = cache[name + ":residual"].norm(dim=1)
        global_norm = (cache[name + ":combined"] - cache[name + ":residual"]).norm(dim=1)
        nonzero = global_norm > 1e-12
        ratio = residual[nonzero] / global_norm[nonzero]
        result[name] = {"zero_global_count": int((~nonzero).sum()),
                        "median_residual_norm": float(residual.median()),
                        "median_global_norm": float(global_norm.median()),
                        "median_ratio_nonzero_global": float(ratio.median()) if ratio.numel() else None}
    return result


def run_fold(target, output, steps, device):
    seed = {"E1": 401, "E2": 501, "E3": 601}[target]
    manifest_path = OUT / f"source_{target}.json"
    dataset = VerifiedSources(manifest_path)
    schedule = epoch_schedule(dataset.manifest["train"], seed, 0)
    assert np.array_equal(schedule, epoch_schedule(dataset.manifest["train"], seed, 0))
    assert not np.array_equal(schedule, epoch_schedule(dataset.manifest["train"], seed, 1))
    batches = [dataset.batch("train", indices, device) for indices in schedule[:steps]]
    np.save(output / f"{target}_epoch0_schedule.npy", schedule, allow_pickle=False)
    full_schedule_checks = []
    for epoch in range(200):
        current = epoch_schedule(dataset.manifest["train"], seed, epoch)
        full_schedule_checks.append(schedule_hash(current))
    write(output / f"{target}_schedule_audit.json", {
        "seed": seed, "epoch_hashes": full_schedule_checks, "optimizer_updates": 60000,
        "batch_size": 16, "each_batch_has_all_six_classes_per_source": True,
        "duplicate_sample_in_any_batch": False,
        "note": "Schedules audited for 200 epochs; training executed only for the smoke steps recorded separately."})
    reports = []
    for arm, spec in ARMS.items():
        torch.manual_seed(seed)
        model = build_model(arm, seed).to(device)
        norms = initial_residual_norms(model, batches[0][0])
        macs = count_matmul_conv_macs(model, batches[0][0][:1])
        model.eval()
        with torch.no_grad():
            for _ in range(5):
                model(batches[0][0], project=False)
            synchronize(device)
            timings = []
            for _ in range(20):
                started = time.perf_counter()
                model(batches[0][0], project=False)
                synchronize(device)
                timings.append((time.perf_counter() - started) * 1000)
        model.train()
        torch.manual_seed(seed + 17)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
        loss_sc = DomainAwareSupConLoss(temperature=0.2, positive_mode="all_views")
        if str(device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
        losses, activity, positive_reports = [], [], []
        for step, (amplitude, labels, rows) in enumerate(batches):
            synchronize(device)
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            result = model(amplitude, project=spec["supcon"])
            ce = F.cross_entropy(result["logits"], labels)
            sc = loss_sc(result["projection"], labels) if spec["supcon"] else ce.new_zeros(())
            # Full candidate strength deliberately exercises the contrastive path
            # during a short smoke; a full protocol would use a recorded warmup.
            loss = ce + 0.025 * sc
            assert torch.isfinite(loss), (target, arm, step)
            loss.backward()
            assert torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all()
            used = sum(p.numel() for p in model.parameters() if p.grad is not None)
            nonzero = sum(p.numel() for p in model.parameters() if p.grad is not None and torch.count_nonzero(p.grad))
            activity.append({"with_gradient_tensor": used, "in_tensors_with_nonzero_gradient": nonzero})
            if spec["supcon"]:
                positive = supcon_positive_pair_diagnostics(result["projection"], labels)
                assert positive["supcon_frac_anchors_with_positive"] == 1.0
                positive_reports.append(positive)
            optimizer.step()
            synchronize(device)
            losses.append({"step": step + 1, "ce": float(ce.detach()), "supcon": float(sc.detach()),
                           "total": float(loss.detach()), "optimizer_step_seconds": time.perf_counter() - started})
        total = sum(p.numel() for p in model.parameters())
        head = sum(p.numel() for p in model.projection_head.parameters())
        assert all(item["with_gradient_tensor"] == total - (0 if spec["supcon"] else head) for item in activity)
        report = {"target_name_only": target, "arm": arm, "seed": seed, "steps": steps,
                  "opened_data_paths": dataset.opened_paths, "phase_files_opened": False,
                  "target_signals_or_labels_loaded": False, "source_manifest_sha256": sha256(manifest_path),
                  "batch_schedule_sha256": schedule_hash(schedule),
                  "first_batch_sample_ids": [r["sample_id"] for r in batches[0][2]],
                  "parameters_total": total, "training_only_projection_parameters": head,
                  "gradient_activity": activity, "positive_anchor_diagnostics": positive_reports,
                  "inference_counted_macs": macs, "initial_residual_norms": norms,
                  "inference_batch_size": 16, "inference_batch_latency_ms_median": float(np.median(timings)),
                  "inference_batch_latency_ms_p95": float(np.percentile(timings, 95)),
                  "training_step_seconds_median": float(np.median([r["optimizer_step_seconds"] for r in losses[5:]])),
                  "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else None,
                  "preprocessing": "Original amplitude array cast float32; no phase, no fitted input normalization",
                  "smoke_lambda": 0.025 if spec["supcon"] else 0.0, "smoke_uses_warmup": False,
                  "loss_trace": losses, "efficacy_evaluation_performed": False}
        reports.append(report)
        write(output / f"{target}_{arm}.json", report)
        print(f"{target} {arm}: {steps} finite updates, {total:,} parameters, "
              f"{report['training_step_seconds_median']:.3f} s/update", flush=True)
        del optimizer, model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--target", choices=["all", "E1", "E2", "E3"], default="all")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/revision/evidence/amplitude_source_smoke")
    args = parser.parse_args()
    assert 10 <= args.steps <= 300
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    assert all((OUT / f"source_{env}.json").exists() for env in ("E1", "E2", "E3")), "Verified source manifests required"
    args.output.mkdir(parents=True, exist_ok=True)
    initialization = shared_initialization_audit()
    reports = []
    for target in ([args.target] if args.target != "all" else ["E1", "E2", "E3"]):
        reports.extend(run_fold(target, args.output, args.steps, args.device))
    summary = {"status": "source_smoke_complete_not_full_training", "model_config": MODEL_CONFIG,
               "initialization_audit": initialization, "arm_runs": len(reports),
               "total_optimizer_updates_executed": sum(r["steps"] for r in reports),
               "device": args.device, "gpu": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
               "torch_version": torch.__version__, "candidate_lambda": 0.025,
               "all_arms_equal_total_parameters": len({r["parameters_total"] for r in reports}) == 1,
               "all_arms_equal_counted_inference_macs": len({r["inference_counted_macs"]["total_counted_macs"] for r in reports}) == 1,
               "source_only": True, "full_training_completed": False,
               "timing_scope": "Preloaded batches; excludes loading, source validation, checkpoint writes and full-run variability",
               "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in (
                   Path(__file__), Path(__file__).with_name("models.py"), Path(__file__).with_name("source_data.py"),
                   ROOT / "dg_models.py", ROOT / "dg_losses.py", ROOT / "experiments/tpp_isolated/tpp_models.py")}}
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
