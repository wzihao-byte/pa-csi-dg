"""Execution freeze, serial GPU runner and source-only timing for frozen WiMANS jobs.

`register` binds the frozen configs, source QA receipts, code and inputs into one
study registry before any formal run. `run` executes one registered stage at a
time on cuda:0, resuming an interrupted run from its last committed epoch.
Target evaluation is a separate stage that requires every four-arm source
checkpoint to be frozen first; CORAL search runs never evaluate target.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .adapter import file_sha256, read_json, require, sampling_schedule
from .models import ARMS, make_model, objective
from .training import atomic_json, code_receipt, inputs_for, now, prepare_runtime, validate_config

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ROOMS = ("classroom", "meeting_room", "empty_room")
SEEDS = (42, 52, 62)
FOUR_ARMS = ("W00", "W01", "W10", "W11")
GAMMAS = (0.1, 1.0, 10.0)
STAGES = ("four_arm_source", "four_arm_target", "coral_search_source")


def registered_jobs():
    jobs = []
    for target in ROOMS:
        for seed in SEEDS:
            for arm in FOUR_ARMS:
                jobs.append({"job_id": f"{arm}_{target}_{seed}", "stage": "four_arm_source", "arm": arm,
                             "target_environment": target, "seed": seed, "gamma": 0.0})
    for target in ROOMS:
        for gamma in GAMMAS:
            jobs.append({"job_id": f"WC_search_{target}_gamma{gamma}_20261014", "stage": "coral_search_source",
                         "arm": "WC", "target_environment": target, "seed": 20261014, "gamma": gamma})
    for job in jobs:
        job["config"] = f"frozen_configs/{job['job_id']}.json"
        job["run_dir"] = f"runs/{job['arm']}/{job['target_environment']}/{job['job_id']}"
    return jobs


def register(study):
    study = Path(study)
    require(not (study / "registry.json").exists(), "Preserve the existing registry; use a new study version")
    receipts, qa_receipts = {}, {}
    for job in registered_jobs():
        path = study / job["config"]
        config = read_json(path)
        validate_config(config, formal=True)
        require(all(config[key] == job[key] for key in ("arm", "target_environment", "seed", "gamma")), "Frozen config identity differs: " + job["job_id"])
        if job["stage"] == "coral_search_source":
            require("coral_selection_receipt" not in config["execution_lock"], "Search configs carry no gamma selection")
        receipts[job["job_id"]] = {"config_file_sha256": file_sha256(path), "config_sha256": config["execution_lock"]["config_sha256"]}
        qa_receipts[config["execution_lock"]["source_qa_receipt"]] = config["execution_lock"]["source_qa_sha256"]
    require(len(qa_receipts) == 3, "Expect one real source QA receipt per fold")
    inputs = inputs_for(read_json(study / registered_jobs()[0]["config"]))
    registry = {"schema": "wimans-execution-registry/v1", "created_at": now(), "status": "frozen",
                "jobs": [{**job, **receipts[job["job_id"]]} for job in registered_jobs()],
                "stage_order": list(STAGES), "four_arm_source_runs": 36, "coral_search_runs": 9,
                "code_files_sha256": code_receipt(), "execution_code_sha256": file_sha256(Path(__file__)),
                "source_qa_receipts": qa_receipts, "input_audit_sha256": read_json(study / registered_jobs()[0]["config"])["input_audit_sha256"],
                "manifest_receipt": inputs.manifest.receipt(),
                "target_access_rule": "four_arm_target only after all 36 four-arm source checkpoints are frozen; search seed never opens target",
                "wc_formal": "built after source-only gamma selection from the nine search runs; not part of this registry",
                "target_metric_tuning": False, "formal_training_started": False}
    atomic_json(study / "registry.json", registry)
    print(json.dumps({"status": "registered", "jobs": len(registry["jobs"]), "registry_sha256": file_sha256(study / "registry.json")}))


def verify(study):
    study = Path(study)
    registry = read_json(study / "registry.json")
    require(registry["status"] == "frozen" and registry["code_files_sha256"] == code_receipt(), "Registered implementation drift")
    require(registry["execution_code_sha256"] == file_sha256(Path(__file__)), "Execution runner drift")
    require([{k: j[k] for k in ("job_id", "stage", "arm", "target_environment", "seed", "gamma", "config", "run_dir")}
             for j in registry["jobs"]] == registered_jobs(), "Registered job grid drift")
    for job in registry["jobs"]:
        require(file_sha256(study / job["config"]) == job["config_file_sha256"], "Frozen config file changed: " + job["job_id"])
    for path, digest in registry["source_qa_receipts"].items():
        require(file_sha256(path) == digest, "Source QA receipt changed: " + path)
    return registry


def frozen_receipt(run_dir, job):
    frozen = read_json(run_dir / "checkpoint_frozen.json")
    require(frozen["optimizer_updates"] == 30000 and frozen["strict_reloaded"] is True and frozen["target_data_opened"] is False,
            "Incomplete source checkpoint: " + job["job_id"])
    require(frozen["config_sha256"] == job["config_sha256"] and frozen["checkpoint_sha256"] == file_sha256(run_dir / "best_model.pt"),
            "Source checkpoint binding drift: " + job["job_id"])
    return frozen


def other_training_processes():
    import psutil
    found = []
    for process in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            command = " ".join(process.info["cmdline"] or [])
            if process.pid != os.getpid() and "python" in (process.info["name"] or "").lower() and \
                    ("--config" in command or "train-source" in command or "evaluate-target" in command):
                found.append(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return found


def run(study, stage):
    study = Path(study).resolve()
    require(stage in STAGES, "Unknown stage")
    registry = verify(study)
    jobs = [j for j in registry["jobs"] if j["stage"] == ("four_arm_source" if stage == "four_arm_target" else stage)]
    if stage == "four_arm_target":
        for job in jobs:
            frozen_receipt(study / job["run_dir"], job)
    require(not other_training_processes(), "Another training/evaluation worker is active")
    lock = study / ".runner.lock"
    descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    os.write(descriptor, json.dumps({"pid": os.getpid(), "stage": stage, "at": now()}).encode())
    os.close(descriptor)
    invocation = uuid.uuid4().hex
    status_path = study / "queue" / f"{stage}_status.json"
    done, job = [], None
    try:
        for index, job in enumerate(jobs):
            verify(study)
            run_dir = study / job["run_dir"]
            atomic_json(status_path, {"status": "running", "stage": stage, "at": now(), "pid": os.getpid(),
                        "invocation": invocation, "current_job": job["job_id"], "position": index + 1,
                        "completed_jobs": len(done), "total_jobs": len(jobs)})
            command = [sys.executable, "-B", "-m", "experiments.wimans_20261014.training"]
            if stage == "four_arm_target":
                if not (run_dir / "target_metrics.json").exists():
                    command += ["evaluate-target", "--run-dir", str(run_dir), "--device", "cuda:0"]
                else:
                    command = None
            elif not (run_dir / "checkpoint_frozen.json").exists():
                command += ["train-source", "--config", str(study / job["config"]), "--run-dir", str(run_dir), "--device", "cuda:0"]
                if (run_dir / "latest.pt").exists():
                    command.append("--resume")
            else:
                command = None
            if command:
                logs = study / "logs"
                logs.mkdir(exist_ok=True)
                with (logs / f"{job['job_id']}.{stage}.log").open("ab") as handle:
                    handle.write(f"\n=== {now()} invocation {invocation}: {' '.join(command)}\n".encode())
                    handle.flush()
                    completed = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
                require(completed.returncode == 0, f"{job['job_id']} {stage} returned {completed.returncode}")
            frozen = frozen_receipt(run_dir, job)
            record = {"job_id": job["job_id"], "run_dir": str(run_dir), "checkpoint_sha256": frozen["checkpoint_sha256"],
                      "source_validation_score": frozen["source_validation_score"]}
            if stage == "four_arm_target":
                metrics = read_json(run_dir / "target_metrics.json")
                require(metrics["checkpoint_sha256"] == frozen["checkpoint_sha256"] and metrics["target_prediction_count"] == 594 and
                        metrics["prediction_sha256"] == file_sha256(run_dir / "target_predictions.json"), "Target metric binding drift")
                record.update(target_macro_f1=metrics["f1_macro"], target_accuracy=metrics["accuracy"])
            else:
                require(not (run_dir / "target_access.json").exists(), "Source stage must not open target")
            done.append(record)
            atomic_json(study / "queue" / f"{stage}_completed.json", {"stage": stage, "at": now(), "completed": done,
                        "total_jobs": len(jobs), "status": "complete" if len(done) == len(jobs) else "partial"})
            if (study / "queue" / "stop_after_current.json").exists():
                atomic_json(status_path, {"status": "stopped_after_completed_job", "stage": stage, "at": now(),
                            "completed_jobs": len(done), "total_jobs": len(jobs)})
                return
        atomic_json(status_path, {"status": "complete", "stage": stage, "at": now(), "completed_jobs": len(done), "total_jobs": len(jobs)})
    except BaseException as error:
        atomic_json(status_path, {"status": "failed", "stage": stage, "at": now(), "current_job": job and job["job_id"],
                    "completed_jobs": len(done), "total_jobs": len(jobs), "error": repr(error)})
        raise
    finally:
        lock.unlink(missing_ok=True)


def benchmark(study, output, timed_steps=5):
    """Source-only GPU seconds/update per arm on the classroom fold; target never opened."""
    study, output = Path(study), Path(output)
    require(not output.exists(), "Preserve existing benchmark receipts")
    require(not other_training_processes(), "Benchmark requires an otherwise idle GPU")
    registry = verify(study)
    device = prepare_runtime("cuda:0")
    receipts = []
    for arm in ("W00", "W01", "W10", "W11", "WC"):
        job = next(j for j in registry["jobs"] if j["arm"] == arm and j["target_environment"] == "classroom")
        config = read_json(study / job["config"])
        inputs = inputs_for(config)
        train = inputs.dataset("classroom", "train")
        schedule = sampling_schedule(train, config["seed"])
        model = make_model(ARMS[arm]["pool"], config["seed"]).to(device).train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
        torch.cuda.reset_peak_memory_stats(device)
        seconds = []
        for step, batch in enumerate(DataLoader(train, batch_sampler=schedule[:2 + timed_steps].tolist(), num_workers=0)):
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            batch = tuple(value.to(device) for value in batch)
            optimizer.zero_grad(set_to_none=True)
            loss, _parts = objective(model, batch, arm, config["gamma"])
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize(device)
            if step >= 2:
                seconds.append(time.perf_counter() - started)
        require(not any(part == "test" for part, _id in inputs.signal_access), "Benchmark opened target signals")
        receipts.append({"arm": arm, "job_id": job["job_id"], "timed_steps": len(seconds),
                         "seconds_per_update": float(np.mean(seconds)), "median_seconds_per_update": float(np.median(seconds)),
                         "seconds_each_update": seconds, "estimated_30000_update_compute_hours": float(np.mean(seconds) * 30000 / 3600),
                         "peak_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20})
        print(json.dumps(receipts[-1]), flush=True)
        del model, optimizer
        torch.cuda.empty_cache()
    atomic_json(output, {"schema": "wimans-exclusive-gpu-timing/v1", "at": now(), "gpu": torch.cuda.get_device_name(device),
                         "torch": torch.__version__, "cuda": torch.version.cuda, "exclusive_gpu": True,
                         "target_signal_rows_read": 0, "registry_sha256": file_sha256(Path(study) / "registry.json"),
                         "receipts": receipts})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("register", "verify"):
        commands.add_parser(name).add_argument("--study", type=Path, required=True)
    runner = commands.add_parser("run")
    runner.add_argument("--study", type=Path, required=True)
    runner.add_argument("--stage", choices=STAGES, required=True)
    timing = commands.add_parser("benchmark")
    timing.add_argument("--study", type=Path, required=True)
    timing.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "register":
        register(args.study)
    elif args.command == "verify":
        registry = verify(args.study)
        print(json.dumps({"status": "verified", "jobs": len(registry["jobs"])}))
    elif args.command == "run":
        run(args.study, args.stage)
    else:
        benchmark(args.study, args.output)


if __name__ == "__main__":
    main()
