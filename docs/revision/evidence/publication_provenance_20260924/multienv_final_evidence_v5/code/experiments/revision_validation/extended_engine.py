"""Resumable source training for frozen controls and new input versions.

Core R v1 continues to use its immutable original engine. This version uses the
same epoch checkpoint contract, with explicit experiment/input adapters.
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from experiments.revision_validation.experiment_data import VerifiedInputs
from experiments.revision_validation.experiment_design import build_experiment_model, objective, predict, score_partition
from experiments.revision_validation.source_data import epoch_schedule, schedule_hash
from experiments.revision_validation.study_engine import (
    acquire_lock, equal_state, restore_rng, rng_state, runtime_settings, save_tensor,
)
from experiments.revision_validation.study_io import bound_path, event, now, read, sha256, value_hash, verify_completed, write


def verify_extension(bundle):
    bundle = Path(bundle).resolve()
    manifest = read(bundle / "bundle_manifest.json")
    for relative, expected in manifest["files_sha256"].items():
        if sha256(bound_path(bundle, relative)) != expected:
            raise RuntimeError(f"Extension freeze changed: {relative}")
    protocol = read(bundle / "protocol.json")
    if protocol["schema"] != "revision-extension-study/v1":
        raise ValueError("Unsupported extension study schema")
    return protocol, sha256(bundle / "bundle_manifest.json")


def resource_profile(config, dataset, output):
    path = output / "resource_profile.json"
    if path.exists():
        if read(path)["config_value_sha256"] != value_hash(config):
            raise RuntimeError("Resource profile configuration mismatch")
        return
    device = config["device"]
    model = build_experiment_model(config).to(device)
    batch = epoch_schedule(dataset.manifest["train"], config["seed"], 0, steps=1)[0]
    inputs, labels, rows = dataset.batch("train", batch, device)
    inference_times, backward_times = [], []
    def synchronize():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    model.eval()
    with torch.no_grad():
        for index in range(7):
            synchronize(); start = time.perf_counter()
            predict(model, inputs, config)
            synchronize()
            if index >= 2:
                inference_times.append(time.perf_counter() - start)
    model.train()
    for index in range(7):
        model.zero_grad(set_to_none=True)
        synchronize(); start = time.perf_counter()
        loss, _, _, _, _ = objective(model, inputs, labels, rows, 6000, config)
        loss.backward()
        synchronize()
        if index >= 2:
            backward_times.append(time.perf_counter() - start)
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise FloatingPointError("Resource preflight produced nonfinite loss/gradients")
    report = {"config_value_sha256": value_hash(config), "source_sample_ids": [r["sample_id"] for r in rows],
              "total_parameters": sum(p.numel() for p in model.parameters()),
              "parameters_with_gradient": sum(p.numel() for p in model.parameters() if p.grad is not None),
              "batch_size": len(labels), "inference_median_seconds": float(np.median(inference_times)),
              "forward_backward_median_seconds": float(np.median(backward_times)),
              "timed_repeats": 5, "warmup_repeats": 2, "optimizer_steps_performed": 0,
              "supcon_profile_weight": "full fixed lambda at step 6000, if enabled",
              "training_model_independent": True, "target_data_loaded": False,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0}
    write(path, report)


def train_experiment(config, output, *, pause_after_epoch=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_hash = value_hash(config)
    if (output / "config.json").exists():
        if read(output / "config.json") != config:
            raise RuntimeError("Changed configuration cannot resume an experiment")
    else:
        if any(output.iterdir()):
            raise RuntimeError("Unrecognized nonempty experiment directory")
        write(output / "config.json", config)
    old = read(output / "status.json") if (output / "status.json").exists() else None
    if old and old["status"] == "completed":
        return verify_completed(output, config)
    if old and old["status"] == "failed":
        raise RuntimeError("Recorded failure requires diagnosis; no silent numerical-failure retry")
    runtime = runtime_settings()
    if runtime != config["expected_runtime"]:
        raise RuntimeError("Runtime differs from frozen experiment configuration")
    if (output / "runtime.json").exists() and read(output / "runtime.json") != runtime:
        raise RuntimeError("Runtime changed across resume")
    write(output / "runtime.json", runtime)
    if sha256(config["source_manifest_path"]) != config["source_manifest_sha256"]:
        raise RuntimeError("Source manifest hash mismatch")
    dataset = VerifiedInputs(config["source_manifest_path"], config["input_modalities"])
    if dataset.target != config["target_name_only"] or len(dataset.manifest["val"]) != config["validation_sample_count"]:
        raise RuntimeError("Wrong source fold/validation partition size")
    if config.get("resource_profile", False):
        resource_profile(config, dataset, output)
    device = config["device"]
    model = build_experiment_model(config).to(device)
    if sum(p.numel() for p in model.parameters()) != config["total_parameters"]:
        raise RuntimeError("Model parameter count differs from frozen specification")
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    torch.manual_seed(config["seed"] + 17)
    random.seed(config["seed"] + 17)
    np.random.seed(config["seed"] + 17)
    history, best, best_epoch, best_state, best_predictions = [], -1.0, None, None, None
    step, elapsed_before = 0, 0.0
    if (output / "progress.json").exists():
        progress = read(output / "progress.json")
        checkpoint = bound_path(output, progress["checkpoint"])
        if sha256(checkpoint) != progress["sha256"]:
            raise RuntimeError("Resume checkpoint hash mismatch")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["config_value_sha256"] != config_hash:
            raise RuntimeError("Checkpoint configuration binding differs")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        checks = {"model_exact": equal_state(model.state_dict(), state["model"]),
                  "optimizer_exact": equal_state(optimizer.state_dict(), state["optimizer"]),
                  "rng_exact": equal_state(rng_state(), state["rng"])}
        if not all(checks.values()):
            raise RuntimeError(f"Resume state mismatch: {checks}")
        event(output, "resume_state_audit", **checks)
        history, best, best_epoch = state["history"], state["best"], state["best_epoch"]
        best_state, best_predictions = state["best_state"], state["best_predictions"]
        step, elapsed_before = state["step"], state["elapsed_seconds"]
        if step != len(history) * config["updates_per_epoch"]:
            raise RuntimeError("Checkpoint step/epoch counts differ")
        event(output, "resumed", completed_epochs=len(history), checkpoint_sha256=progress["sha256"])
    else:
        event(output, "started", config_value_sha256=config_hash)
    started = time.perf_counter()
    write(output / "status.json", {"status": "running", "epoch": len(history), "pid": os.getpid(),
          "updated_at": now(), "target_data_loaded": False})
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    try:
        for epoch in range(len(history), config["epochs"]):
            model.train()
            schedule = epoch_schedule(dataset.manifest["train"], config["seed"], epoch, steps=config["updates_per_epoch"])
            losses, diagnostics = [], []
            active_parameters = None
            for batch in schedule:
                step += 1
                inputs, labels, rows = dataset.batch("train", batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss, primary, regularizer, weight, diagnostic = objective(model, inputs, labels, rows, step, config)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite objective at step {step}")
                loss.backward()
                gradients = [p for p in model.parameters() if p.grad is not None]
                if not gradients or not torch.stack([torch.isfinite(p.grad).all() for p in gradients]).all():
                    raise FloatingPointError(f"Missing/nonfinite gradient at step {step}")
                if active_parameters is None:
                    active_parameters = sum(p.numel() for p in gradients)
                optimizer.step()
                for group in optimizer.param_groups:
                    group["lr"] = config["learning_rate"] * 0.9 ** (step / 10000)
                losses.append([float(primary.detach()), float(regularizer.detach()), float(loss.detach())])
                diagnostics.append(diagnostic)
            metrics, predictions = score_partition(model, dataset, config)
            average_diagnostics = {key: float(np.mean([d[key] for d in diagnostics])) for key in diagnostics[0]} if diagnostics else {}
            history.append({"epoch": epoch + 1, "optimizer_steps": step, "source_validation": metrics,
                            "mean_primary_regularizer_total": np.mean(losses, axis=0).tolist(),
                            "regularizer_weight_at_epoch_end": weight, "objective_diagnostics_mean": average_diagnostics,
                            "parameters_with_gradient_at_first_batch": active_parameters,
                            "schedule_sha256": schedule_hash(schedule), "learning_rate_after_epoch": optimizer.param_groups[0]["lr"]})
            if metrics["macro_f1"] > best:
                best, best_epoch = metrics["macro_f1"], epoch + 1
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                best_predictions = predictions
            checkpoint = output / f"resume_epoch_{epoch + 1:04d}.pt"
            state = {"config_value_sha256": config_hash, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                     "rng": rng_state(), "history": history, "best": best, "best_epoch": best_epoch,
                     "best_state": best_state, "best_predictions": best_predictions, "step": step,
                     "elapsed_seconds": elapsed_before + time.perf_counter() - started}
            save_tensor(checkpoint, state)
            write(output / "progress.json", {"checkpoint": checkpoint.name, "sha256": sha256(checkpoint),
                                             "completed_epochs": epoch + 1, "updated_at": now()})
            write(output / "history.json", history)
            write(output / "status.json", {"status": "running", "epoch": epoch + 1, "best_epoch": best_epoch,
                  "best_source_macro_f1": best, "pid": os.getpid(), "updated_at": now(), "target_data_loaded": False})
            event(output, "epoch_committed", epoch=epoch + 1, source_macro_f1=metrics["macro_f1"])
            obsolete_epoch = epoch - 1
            if obsolete_epoch > 1 and obsolete_epoch % 10 != 0:
                obsolete = bound_path(output, f"resume_epoch_{obsolete_epoch:04d}.pt")
                if obsolete.is_file():
                    obsolete.unlink()
            print(f"{config['job_id']}: epoch {epoch + 1}/{config['epochs']}, source macro-F1 {metrics['macro_f1']:.6f}", flush=True)
            if pause_after_epoch is not None and epoch + 1 == pause_after_epoch:
                result = {"status": "paused", "epoch": epoch + 1, "target_data_loaded": False}
                write(output / "status.json", result)
                event(output, "intentional_QA_pause", epoch=epoch + 1)
                return result
        save_tensor(output / "best_model.pt", best_state)
        write(output / "selected_source_predictions.json", best_predictions)
        write(output / "history.json", history)
        artifacts = ["config.json", "runtime.json", "history.json", "best_model.pt", "selected_source_predictions.json"]
        if config.get("resource_profile", False):
            artifacts.append("resource_profile.json")
        result = {"status": "completed", "completed_at": now(), "epochs": config["epochs"], "optimizer_updates": step,
                  "best_epoch": best_epoch, "best_source_macro_f1": best, "config_value_sha256": config_hash,
                  "target_data_loaded": False, "phase_loaded": "phase" in config["input_modalities"],
                  "opened_data_paths": dataset.opened_paths, "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                  "peak_cuda_allocated_bytes_this_process": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0,
                  "artifacts_sha256": {name: sha256(output / name) for name in artifacts}}
        write(output / "status.json", result)
        event(output, "completed", optimizer_updates=step)
        return verify_completed(output, config)
    except Exception:
        write(output / "status.json", {"status": "failed", "optimizer_updates_attempted": step,
              "updated_at": now(), "error": traceback.format_exc(), "target_data_loaded": False})
        event(output, "failed", optimizer_updates_attempted=step)
        raise


def wait_for_source_dependency(dependency, waiting_bundle, expected_hash):
    """Wait in a background process; never restart or interrupt another runner."""
    from experiments.revision_validation.study_io import verify_bundle
    dependency = Path(dependency)
    protocol, bundle_hash = verify_bundle(dependency)
    if bundle_hash != expected_hash:
        raise RuntimeError("Dependency differs from the extension's frozen reference study")
    previous = None
    while True:
        if (dependency / "queue_failure.json").exists():
            raise RuntimeError("Dependency recorded a failure; inspect it before continuing")
        completed = []
        for config in protocol["jobs"]:
            path = dependency / "runs" / config["job_id"]
            if (path / "status.json").exists() and read(path / "status.json")["status"] == "completed":
                completed.append(config["job_id"])
        if len(completed) == len(protocol["jobs"]):
            verify_bundle(dependency)
            for config in protocol["jobs"]:
                verify_completed(dependency / "runs" / config["job_id"], config)
            summary_path = dependency / "evaluation/summary.json"
            if not summary_path.exists():
                # The core's existing deferred evaluator runs before this process
                # takes the GPU for controls. It independently rechecks its gate.
                with (dependency / "evaluation.stdout.log").open("a", encoding="utf-8") as stdout, (dependency / "evaluation.stderr.log").open("a", encoding="utf-8") as stderr:
                    subprocess.run([sys.executable, "-m", "experiments.revision_validation.evaluate_study", "--bundle", str(dependency)],
                                   cwd=dependency / "code", stdout=stdout, stderr=stderr, check=True)
            evaluation = read(summary_path)
            if {r["job_id"] for r in evaluation["runs"]} != {c["job_id"] for c in protocol["jobs"]}:
                raise RuntimeError("Dependency evaluation does not cover its frozen jobs")
            for relative, expected in evaluation["artifacts_sha256"].items():
                if sha256(bound_path(dependency, relative)) != expected:
                    raise RuntimeError("Dependency evaluation artifact hash mismatch")
            write(waiting_bundle / "dependency_gate.json", {"passed_at": now(), "dependency": str(dependency),
                  "dependency_bundle_sha256": bundle_hash, "verified_source_jobs": completed,
                  "dependency_evaluation_sha256": sha256(summary_path)})
            return
        if len(completed) != previous:
            previous = len(completed)
            write(waiting_bundle / "queue_status.json", {"status": "waiting_for_source_dependency", "dependency": str(dependency),
                  "completed_dependency_jobs": len(completed), "required_dependency_jobs": len(protocol["jobs"]),
                  "updated_at": now(), "pid": os.getpid()})
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    protocol, bundle_hash = verify_extension(bundle)
    if Path(__file__).resolve().parents[2] != bundle / "code":
        raise RuntimeError("Run from the immutable extension code snapshot")
    lock, identity = acquire_lock(bundle)
    try:
        if protocol.get("source_training_dependency"):
            wait_for_source_dependency(protocol["source_training_dependency"], bundle, protocol["dependency_bundle_sha256"])
        for index, config in enumerate(protocol["jobs"]):
            verify_extension(bundle)
            write(bundle / "queue_status.json", {"status": "running", "job_index": index, "job_id": config["job_id"],
                  "total_jobs": len(protocol["jobs"]), "bundle_sha256": bundle_hash, "pid": os.getpid(), "updated_at": now()})
            train_experiment(config, bound_path(bundle, "runs/" + config["job_id"]))
        write(bundle / "queue_status.json", {"status": "source_training_completed", "jobs": len(protocol["jobs"]),
              "bundle_sha256": bundle_hash, "updated_at": now(), "target_evaluation_performed": False})
        from experiments.revision_validation.extended_evaluate import evaluate
        summary_hash = evaluate(bundle)
        write(bundle / "queue_status.json", {"status": "completed", "jobs": len(protocol["jobs"]),
              "bundle_sha256": bundle_hash, "updated_at": now(), "target_evaluation_performed": True,
              "evaluation_summary_sha256": summary_hash})
    except Exception:
        write(bundle / "queue_failure.json", {"status": "failed", "updated_at": now(), "error": traceback.format_exc()})
        raise
    finally:
        if lock.exists() and read(lock) == identity:
            lock.unlink()


if __name__ == "__main__":
    main()
