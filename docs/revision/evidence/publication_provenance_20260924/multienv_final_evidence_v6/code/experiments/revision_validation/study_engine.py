"""Resumable source-only training and sequential execution of a frozen study.

No target loader is imported. A committed epoch checkpoint includes optimizer,
RNG, history and the best source-selected weights/predictions. An interrupted
epoch is replayed. Known CUDA adaptive-max-pool nondeterminism is disclosed;
restoring state does not promise bitwise equality on every CUDA installation.
"""
from __future__ import annotations

import argparse
import os
import random
import time
import traceback
from pathlib import Path

# Configure cuBLAS before CUDA is initialized. cuDNN's deterministic switch
# alone does not control GEMM reductions used by attention and linear layers.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from dg_losses import DomainAwareSupConLoss
from experiments.revision_validation.models import ARMS, build_model
from experiments.revision_validation.source_data import VerifiedSources, epoch_schedule, schedule_hash
from experiments.revision_validation.train_source import score_source
from experiments.revision_validation.study_io import (
    bound_path, event, now, read, sha256, value_hash, verify_bundle, verify_completed, write,
)


def runtime_settings():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Strict mode raises on the existing adaptive_max_pool1d CUDA backward.
    torch.use_deterministic_algorithms(False)
    return {"torch_version": str(torch.__version__), "numpy_version": np.__version__,
            "cuda_version": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "num_threads": 4, "cudnn_benchmark": False, "cudnn_deterministic": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
            "strict_deterministic_algorithms": False,
            "limitation": "Existing CUDA adaptive-max-pool backward lacks a deterministic implementation"}


def rng_state():
    np_state = np.random.get_state()
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "python": random.getstate(),
            "numpy": [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])]}


def restore_rng(state):
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))


def equal_state(left, right):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.detach().cpu(), right.detach().cpu())
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal_state(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(equal_state(a, b) for a, b in zip(left, right))
    return left == right


def save_tensor(path, value):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def train_job(config, output, *, pause_after_epoch=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_hash = value_hash(config)
    if (output / "config.json").exists():
        if read(output / "config.json") != config:
            raise RuntimeError("Cannot resume under a changed configuration")
    else:
        if any(output.iterdir()):
            raise RuntimeError("Run directory contains unrecognized artifacts")
        write(output / "config.json", config)
    if (output / "status.json").exists():
        old = read(output / "status.json")
        if old["status"] == "completed":
            return verify_completed(output, config)
        if old["status"] == "failed":
            raise RuntimeError("Recorded failure requires diagnosis; automatic numerical-failure retry is disabled")
    runtime = runtime_settings()
    if config.get("expected_runtime") and runtime != config["expected_runtime"]:
        raise RuntimeError("Runtime changed since study freeze")
    if (output / "runtime.json").exists() and read(output / "runtime.json") != runtime:
        raise RuntimeError("Runtime changed across a resume")
    write(output / "runtime.json", runtime)
    manifest_path = Path(config["source_manifest_path"])
    if sha256(manifest_path) != config["source_manifest_sha256"]:
        raise RuntimeError("Source manifest hash mismatch")
    dataset = VerifiedSources(manifest_path)
    if dataset.target != config["target_name_only"]:
        raise RuntimeError("Wrong source fold")
    if len(dataset.manifest["val"]) != 1200:
        raise RuntimeError("Wrong source validation size")
    device = config["device"]
    model = build_model(config["arm"], config["seed"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=0.0)
    criterion = DomainAwareSupConLoss(temperature=config["temperature"], positive_mode="all_views")
    torch.manual_seed(config["seed"] + 17)
    random.seed(config["seed"] + 17)
    np.random.seed(config["seed"] + 17)
    history, best, best_epoch, best_state, best_predictions = [], -1.0, None, None, None
    step, elapsed_before = 0, 0.0
    if (output / "progress.json").exists():
        progress = read(output / "progress.json")
        checkpoint = bound_path(output, progress["checkpoint"])
        if sha256(checkpoint) != progress["sha256"]:
            raise RuntimeError("Resume checkpoint checksum mismatch")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if state["config_value_sha256"] != config_hash:
            raise RuntimeError("Resume checkpoint configuration mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
        audit = {"model_exact": equal_state(model.state_dict(), state["model"]),
                 "optimizer_exact": equal_state(optimizer.state_dict(), state["optimizer"]),
                 "rng_exact": equal_state(rng_state(), state["rng"])}
        if not all(audit.values()):
            raise RuntimeError(f"Resume state did not restore exactly: {audit}")
        event(output, "resume_state_audit", **audit)
        history, best, best_epoch = state["history"], state["best"], state["best_epoch"]
        best_state, best_predictions = state["best_state"], state["best_predictions"]
        step, elapsed_before = state["step"], state["elapsed_seconds"]
        if step != len(history) * config["updates_per_epoch"]:
            raise RuntimeError("Checkpoint epoch/update count mismatch")
        event(output, "resumed", completed_epochs=len(history), checkpoint_sha256=progress["sha256"])
    else:
        event(output, "started", config_value_sha256=config_hash)
    write(output / "status.json", {"status": "running", "epoch": len(history),
          "pid": os.getpid(), "updated_at": now(), "target_data_loaded": False})
    started = time.perf_counter()
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)
    use_sc = ARMS[config["arm"]]["supcon"]
    try:
        for epoch in range(len(history), config["epochs"]):
            model.train()
            schedule = epoch_schedule(dataset.manifest["train"], config["seed"], epoch,
                                      steps=config["updates_per_epoch"])
            losses = []
            for batch in schedule:
                step += 1
                amplitude, labels, _ = dataset.batch("train", batch, device)
                optimizer.zero_grad(set_to_none=True)
                values = model(amplitude, project=use_sc)
                ce = F.cross_entropy(values["logits"], labels)
                sc = criterion(values["projection"], labels) if use_sc else ce.new_zeros(())
                weight = config["lambda_supcon"] * min(1.0, step / config["warmup_steps"]) if use_sc else 0.0
                loss = ce + weight * sc
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Nonfinite loss at step {step}")
                loss.backward()
                if not torch.stack([torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]).all():
                    raise FloatingPointError(f"Nonfinite gradient at step {step}")
                optimizer.step()
                for group in optimizer.param_groups:
                    group["lr"] = config["learning_rate"] * 0.9 ** (step / 10000)
                losses.append([float(ce.detach()), float(sc.detach()), float(loss.detach())])
            metrics, predictions = score_source(model, dataset, device)
            history.append({"epoch": epoch + 1, "optimizer_steps": step, "source_validation": metrics,
                            "mean_ce_supcon_total": np.mean(losses, axis=0).tolist(),
                            "schedule_sha256": schedule_hash(schedule), "lambda_at_epoch_end": weight,
                            "learning_rate_after_epoch": optimizer.param_groups[0]["lr"]})
            if metrics["macro_f1"] > best:
                best, best_epoch = metrics["macro_f1"], epoch + 1
                best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                best_predictions = predictions
            checkpoint = output / f"resume_epoch_{epoch + 1:04d}.pt"
            state = {"config_value_sha256": config_hash, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "rng": rng_state(), "history": history,
                     "best": best, "best_epoch": best_epoch, "best_state": best_state,
                     "best_predictions": best_predictions, "step": step,
                     "elapsed_seconds": elapsed_before + time.perf_counter() - started}
            save_tensor(checkpoint, state)
            write(output / "progress.json", {"checkpoint": checkpoint.name, "sha256": sha256(checkpoint),
                                             "completed_epochs": epoch + 1, "updated_at": now()})
            write(output / "history.json", history)
            write(output / "status.json", {"status": "running", "epoch": epoch + 1,
                  "best_epoch": best_epoch, "best_source_macro_f1": best,
                  "updated_at": now(), "pid": os.getpid(), "target_data_loaded": False})
            event(output, "epoch_committed", epoch=epoch + 1, source_macro_f1=metrics["macro_f1"])
            # Keep epoch 1, each tenth epoch and two recent checkpoints. Only a
            # specific obsolete file belonging to this run may be removed.
            obsolete_epoch = epoch - 1
            if obsolete_epoch > 1 and obsolete_epoch % 10 != 0:
                obsolete = bound_path(output, f"resume_epoch_{obsolete_epoch:04d}.pt")
                if obsolete.is_file():
                    obsolete.unlink()
            print(f"{config['job_id']}: epoch {epoch + 1}/{config['epochs']}, "
                  f"source macro-F1 {metrics['macro_f1']:.6f}, best {best:.6f}", flush=True)
            if pause_after_epoch is not None and epoch + 1 == pause_after_epoch:
                result = {"status": "paused", "epoch": epoch + 1, "target_data_loaded": False}
                write(output / "status.json", result)
                event(output, "intentional_QA_pause", epoch=epoch + 1)
                return result
        save_tensor(output / "best_model.pt", best_state)
        write(output / "selected_source_predictions.json", best_predictions)
        write(output / "history.json", history)
        artifacts = ["config.json", "runtime.json", "history.json", "best_model.pt", "selected_source_predictions.json"]
        result = {"status": "completed", "completed_at": now(), "epochs": config["epochs"],
                  "optimizer_updates": step, "best_epoch": best_epoch, "best_source_macro_f1": best,
                  "config_value_sha256": config_hash, "target_data_loaded": False, "phase_loaded": False,
                  "opened_data_paths": dataset.opened_paths,
                  "elapsed_seconds": elapsed_before + time.perf_counter() - started,
                  "peak_cuda_allocated_bytes_this_process": torch.cuda.max_memory_allocated(device) if str(device).startswith("cuda") else 0,
                  "artifacts_sha256": {name: sha256(output / name) for name in artifacts}}
        write(output / "status.json", result)
        event(output, "completed", optimizer_updates=step)
        return verify_completed(output, config)
    except Exception:
        write(output / "status.json", {"status": "failed", "optimizer_updates_attempted": step,
              "error": traceback.format_exc(), "updated_at": now(), "target_data_loaded": False})
        event(output, "failed", optimizer_updates_attempted=step)
        raise


def acquire_lock(bundle):
    path = bundle / "runner.lock.json"
    identity = {"pid": os.getpid(), "create_time": psutil.Process().create_time(), "acquired_at": now()}
    while True:
        try:
            with path.open("x", encoding="utf-8") as stream:
                import json
                json.dump(identity, stream)
            return path, identity
        except FileExistsError:
            old = read(path)
            try:
                alive = abs(psutil.Process(old["pid"]).create_time() - old["create_time"]) < 0.1
            except psutil.NoSuchProcess:
                alive = False
            if alive:
                raise RuntimeError(f"Study runner already active: PID {old['pid']}")
            path.rename(bundle / f"stale_runner_lock_{time.time_ns()}.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    protocol, bundle_hash = verify_bundle(bundle)
    expected_root = bundle / "code"
    if Path(__file__).resolve().parents[2] != expected_root:
        raise RuntimeError("Launch the runner from its frozen code snapshot")
    lock, identity = acquire_lock(bundle)
    try:
        for number, config in enumerate(protocol["jobs"]):
            verify_bundle(bundle)
            write(bundle / "queue_status.json", {"status": "running", "job_index": number,
                  "job_id": config["job_id"], "total_jobs": len(protocol["jobs"]),
                  "bundle_sha256": bundle_hash, "pid": os.getpid(), "updated_at": now()})
            train_job(config, bound_path(bundle, "runs/" + config["job_id"]))
        write(bundle / "queue_status.json", {"status": "source_training_completed", "jobs": len(protocol["jobs"]),
              "bundle_sha256": bundle_hash, "updated_at": now(), "target_evaluation_performed": False})
        print("All frozen source runs completed; deferred target gate is now eligible.", flush=True)
    except Exception:
        write(bundle / "queue_failure.json", {"status": "failed", "updated_at": now(), "error": traceback.format_exc()})
        raise
    finally:
        if lock.exists() and read(lock) == identity:
            lock.unlink()


if __name__ == "__main__":
    main()
