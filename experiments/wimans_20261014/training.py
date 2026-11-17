"""Separate source training, checkpoint freeze and offline target evaluation.

Formal runs fail closed until full input QA and a real source QA receipt pass.
No command combines source training with target evaluation. Resume replays any
unfinished virtual epoch from its last committed epoch boundary, restoring
optimizer and all RNG states; target data never determine update budgets.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from .adapter import (Manifest, ProcessedInput, file_sha256, read_json, require,
                      require_formal_input, sampling_schedule, schedule_hash)
from .frozen_models.control_components import shared_state_hash
from .metrics import confusion_metrics
from .models import ARMS, EXPECTED_PARAMETERS, MODEL_CONFIG, make_model, objective

SCHEMA = "wimans-source-train/v1"
HERE = Path(__file__).resolve().parent
OWN_CODE = ("adapter.py", "models.py", "metrics.py", "training.py", "prepare_configs.py",
            "frozen_models/dg_models.py", "frozen_models/tpp_models.py", "frozen_models/control_components.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def code_receipt():
    return {name: file_sha256(HERE / name) for name in OWN_CODE}


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def atomic_torch(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    os.close(fd)
    try:
        torch.save(value, temporary)
        with open(temporary, "r+b") as handle:
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def validate_config(config, *, formal=False):
    require(config.get("schema") == SCHEMA, "Unsupported training config")
    require(config.get("arm") in ARMS and config.get("target_environment") in {"classroom", "meeting_room", "empty_room"}, "Unregistered arm/fold")
    require(config.get("seed") in (42, 52, 62, 20261014), "Unregistered seed")
    require(config.get("seed") != 20261014 or config.get("arm") == "WC", "Search seed is reserved for CORAL source searches")
    require(config.get("model") == MODEL_CONFIG and config.get("expected_parameters") == EXPECTED_PARAMETERS, "Model specification drift")
    require(config.get("training") == {"epochs": 200, "updates_per_epoch": 150, "batch_size": 32,
            "optimizer": "adam", "lr": 0.001, "weight_decay": 0.0, "sampler": "domain_class_balanced",
            "lr_decay_rate": 0.9, "lr_decay_steps": 10000, "grad_clip_norm": 0.0,
            "checkpoint_rule": "source_val_accuracy_last_tied_best"}, "Training protocol drift")
    require(config.get("losses") == {**ARMS[config["arm"]], "temperature": 0.2,
            "positive_mode": "global_class_instance_views", "include_same_domain_same_class": True,
            "warmup_updates": 0}, "Loss specification drift")
    gamma = config.get("gamma")
    require(gamma in (0.1, 1.0, 10.0) if config["arm"] == "WC" else gamma == 0.0, "CORAL gamma not registered")
    require(config.get("masked_forward_exposure") is False, "Exposure is a separately registered MultiEnv control")
    if formal:
        lock = config.get("execution_lock", {})
        require(lock.get("status") == "frozen", "Formal training blocked: candidate configuration is not frozen")
        unsigned = {k: v for k, v in config.items() if k != "execution_lock"}
        require(lock.get("config_sha256") == canonical_hash(unsigned), "Frozen config drift")
        require(lock.get("code_files_sha256") == code_receipt(), "Frozen implementation drift")
        qa = read_json(lock["source_qa_receipt"])
        require(file_sha256(lock["source_qa_receipt"]) == lock["source_qa_sha256"], "Source QA receipt drift")
        require(qa.get("status") == "passed" and qa.get("synthetic") is False and qa.get("target_signal_rows_read") == 0,
                "Formal training requires real source-only QA")
        require(qa.get("code_files_sha256") == code_receipt() and qa.get("input_audit_sha256") == config["input_audit_sha256"] and
                qa.get("manifest_receipt") == config["manifest_receipt"] and qa.get("target_environment") == config["target_environment"],
                "Source QA binding differs from frozen data/fold/code")
        require(qa.get("source_short_train_updates", 0) >= 2 and qa.get("validation_rows") == 396 and
                qa.get("source_train_batch_size") == 32 and qa.get("input_shape") == [850, 90] and
                qa.get("checked_model_arm") == "W11" and qa.get("expected_parameters") == EXPECTED_PARAMETERS and
                qa.get("checkpoint_strict_reload_logits_exact") is True,
                "Real source QA did not verify full training/batch/validation/reload contract")
        if config["arm"] == "WC" and config["seed"] != 20261014:
            choice = read_json(lock["coral_selection_receipt"])
            require(file_sha256(lock["coral_selection_receipt"]) == lock["coral_selection_sha256"], "CORAL selection drift")
            require(choice["target_environment"] == config["target_environment"] and choice["selected_gamma"] == gamma and
                    choice["target_evaluated"] is False, "CORAL must be chosen by source validation")
            require(choice.get("input_audit_sha256") == config["input_audit_sha256"] and
                    choice.get("manifest_receipt") == config["manifest_receipt"] and choice.get("code_files_sha256") == code_receipt(),
                    "CORAL selection data/fold/code binding drift")
            candidates = choice.get("candidates", [])
            require(len(candidates) == 3 and {x["gamma"] for x in candidates} == {0.1, 1.0, 10.0}, "CORAL selection grid incomplete")
            selected = sorted(candidates, key=lambda row: (-row["source_validation_accuracy"], row["gamma"]))[0]
            require(selected["gamma"] == gamma and all(x["input_audit_sha256"] == config["input_audit_sha256"] and
                    x["manifest_receipt"] == config["manifest_receipt"] and x["code_files_sha256"] == code_receipt()
                    for x in candidates), "CORAL source selection rule or candidate bindings drift")


def inputs_for(config, *, candidate_source_qa=False):
    manifest = Manifest(config["subset_manifest"], config["fold_manifest"])
    require(manifest.receipt() == config["manifest_receipt"], "Manifest binding drift")
    require(file_sha256(Path(config["processed_dir"]) / "input_audit.json") == config["input_audit_sha256"], "Input audit receipt drift")
    return ProcessedInput(config["processed_dir"], manifest, candidate_source_qa=candidate_source_qa)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        require(torch.cuda.is_available(), "CUDA checkpoint cannot resume on CPU")
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def runtime_receipt(device):
    return {"python": os.sys.version, "numpy": np.__version__, "torch": torch.__version__,
            "device": str(device), "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "threads": torch.get_num_threads(), "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "strict_deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}


def prepare_runtime(device):
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return torch.device(device)


def evaluate_source(model, dataset, device):
    labels, predictions = [], []
    model.eval()
    with torch.no_grad():
        for amp, phase, target, _domains, _ids in DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0):
            logits = model(amp.to(device), phase.to(device))["logits"]
            require(torch.isfinite(logits).all().item(), "Non-finite source logits")
            labels.extend(target.tolist())
            predictions.extend(logits.argmax(1).cpu().tolist())
    return confusion_metrics(np.asarray(labels), np.asarray(predictions))


def choose_last_best(history):
    require(bool(history), "No source validation history")
    score = max(row["validation"]["accuracy"] for row in history)
    return max(row["epoch"] for row in history if row["validation"]["accuracy"] == score), score


def train_source(config_path, run_dir, *, device="cuda:0", resume=False):
    config = read_json(config_path)
    validate_config(config, formal=True)
    device, run_dir = prepare_runtime(device), Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    require(resume or not any(run_dir.iterdir()), "Use a new run directory or explicit --resume")
    require(not (run_dir / "target_access.json").exists(), "Source training cannot resume after target evaluation has opened signals")
    inputs = inputs_for(config)
    target, seed = config["target_environment"], config["seed"]
    train, validation = inputs.dataset(target, "train"), inputs.dataset(target, "validation")
    schedule = sampling_schedule(train, seed)
    sampler_sha = schedule_hash(train, schedule)
    require(sampler_sha == config["sampling_schedule_sha256"], "Sampling sequence drift")
    seed_all(seed)
    model = make_model(config["losses"]["pool"], seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    runtime = runtime_receipt(device)
    fingerprint = canonical_hash(config)
    history, best_state, best_accuracy, start_epoch = [], None, -1.0, 0
    if resume:
        checkpoint = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
        require(checkpoint["config_sha256"] == fingerprint and checkpoint["sampler_sha256"] == sampler_sha,
                "Resume config/sampling contract changed")
        require(checkpoint["runtime"] == runtime, "Resume requires identical runtime/device")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, best_state, best_accuracy = checkpoint["history"], checkpoint["best_model"], checkpoint["best_accuracy"]
        start_epoch = checkpoint["completed_epochs"]
        require(checkpoint["optimizer_updates"] == start_epoch * 150 and len(history) == start_epoch,
                "Resume boundary/history drift")
        restore_rng(checkpoint["rng"])
    else:
        # The epoch-zero boundary allows --resume even if the first epoch is interrupted.
        atomic_torch(run_dir / "latest.pt", {"schema": SCHEMA, "config_sha256": fingerprint,
                     "sampler_sha256": sampler_sha, "runtime": runtime, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "rng": rng_state(), "history": [],
                     "completed_epochs": 0, "optimizer_updates": 0, "best_model": None, "best_accuracy": -1.0})
        atomic_json(run_dir / "initialization_audit.json", {"parameters": EXPECTED_PARAMETERS,
                    "common_state_sha256": shared_state_hash(model), "seed": seed})
        atomic_json(run_dir / "resolved_config.json", config)
    for epoch in range(start_epoch, 200):
        model.train()
        totals = {"loss": 0.0, "ce": 0.0, "supcon": 0.0, "coral": 0.0}
        loader = DataLoader(train, batch_sampler=schedule[epoch * 150:(epoch + 1) * 150].tolist(), num_workers=0)
        for offset, batch in enumerate(loader):
            step = epoch * 150 + offset
            for group in optimizer.param_groups:
                group["lr"] = 0.001 * 0.9 ** (step / 10000)
            batch = tuple(value.to(device) for value in batch)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = objective(model, batch, config["arm"], config["gamma"])
            require(torch.isfinite(loss).item(), "Non-finite source training objective")
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for key in ("ce", "supcon", "coral"):
                totals[key] += parts[key]
            if (step + 1) % 100 == 0:
                atomic_json(run_dir / "training_status.json", {"status": "source_training", "at": now(),
                            "optimizer_updates": step + 1, "target_signal_rows_read": 0})
        metrics = evaluate_source(model, validation, device)
        history.append({"epoch": epoch + 1, "optimizer_updates": (epoch + 1) * 150,
                        "train": {k: v / 150 for k, v in totals.items()}, "validation": metrics})
        if metrics["accuracy"] >= best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        atomic_torch(run_dir / "latest.pt", {"schema": SCHEMA, "config_sha256": fingerprint,
                     "sampler_sha256": sampler_sha, "runtime": runtime, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "rng": rng_state(), "history": history,
                     "completed_epochs": epoch + 1, "optimizer_updates": (epoch + 1) * 150,
                     "best_model": best_state, "best_accuracy": best_accuracy})
        atomic_json(run_dir / "source_history.json", history)
    best_epoch, score = choose_last_best(history)
    atomic_torch(run_dir / "best_model.pt", best_state)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True), strict=True)
    require(not any(part == "test" for part, _id in inputs.signal_access), "Target signal read during source training")
    atomic_json(run_dir / "checkpoint_frozen.json", {"schema": SCHEMA, "at": now(), "target_environment": target,
                "checkpoint_sha256": file_sha256(run_dir / "best_model.pt"), "config_sha256": fingerprint,
                "code_files_sha256": code_receipt(), "input_audit_sha256": config["input_audit_sha256"],
                "manifest_receipt": config["manifest_receipt"], "sampler_sha256": sampler_sha,
                "optimizer_updates": 30000, "source_validation_score": score, "source_last_tied_best_epoch": best_epoch,
                "strict_reloaded": True, "target_data_opened": False, "target_signal_rows_read": 0})
    atomic_json(run_dir / "training_status.json", {"status": "source_complete_checkpoint_frozen", "at": now(),
                "optimizer_updates": 30000, "target_signal_rows_read": 0})


def evaluate_target(run_dir, *, device="cuda:0"):
    run_dir, device = Path(run_dir), prepare_runtime(device)
    require(not (run_dir / "target_metrics.json").exists(), "Target evaluation already exists")
    config = read_json(run_dir / "resolved_config.json")
    validate_config(config, formal=True)
    frozen = read_json(run_dir / "checkpoint_frozen.json")
    require(frozen["config_sha256"] == canonical_hash(config) and frozen["code_files_sha256"] == code_receipt(), "Frozen evaluator configuration/code drift")
    require(frozen["optimizer_updates"] == 30000 and frozen["checkpoint_sha256"] == file_sha256(run_dir / "best_model.pt"), "Incomplete or changed source checkpoint")
    require(config["seed"] in (42, 52, 62), "Source-search runs must not evaluate target")
    model = make_model(config["losses"]["pool"], config["seed"]).to(device)
    model.load_state_dict(torch.load(run_dir / "best_model.pt", map_location=device, weights_only=True), strict=True)
    # Model is strict reloaded before target signal rows can be retrieved.
    inputs = inputs_for(config)
    target = inputs.dataset(config["target_environment"], "test", {**frozen, "checkpoint_path": str(run_dir / "best_model.pt")})
    atomic_json(run_dir / "target_access.json", {"at": now(), "status": "opened_for_offline_evaluation",
                "checkpoint_sha256": frozen["checkpoint_sha256"], "sample_ids": target.ids,
                "target_environment": config["target_environment"]})
    model.eval()
    rows, labels, predictions = [], [], []
    with torch.no_grad():
        for amp, phase, batch_labels, _domains, ids in DataLoader(target, batch_size=32, shuffle=False, num_workers=0):
            probabilities = model(amp.to(device), phase.to(device))["logits"].softmax(1).cpu()
            require(torch.isfinite(probabilities).all().item(), "Non-finite target probabilities")
            batch_predictions = probabilities.argmax(1)
            labels.extend(batch_labels.tolist())
            predictions.extend(batch_predictions.tolist())
            for i, numeric_id in enumerate(ids.tolist()):
                sample_id = inputs.manifest.ids[numeric_id]
                require(sample_id == target.ids[len(rows)], "Ordered target identity drift")
                rows.append({"sample_id": sample_id, "label": int(batch_labels[i]),
                             "prediction": int(batch_predictions[i]), "probabilities": probabilities[i].tolist()})
    require(len(rows) == 594, "Target prediction count drift")
    atomic_json(run_dir / "target_predictions.json", rows)
    atomic_json(run_dir / "target_metrics.json", {**confusion_metrics(np.asarray(labels), np.asarray(predictions)),
                "checkpoint_sha256": frozen["checkpoint_sha256"],
                "prediction_sha256": file_sha256(run_dir / "target_predictions.json"), "target_prediction_count": len(rows)})


def source_qa(config_path, output_dir, *, device="cpu", updates=2):
    """Real source QA is explicitly distinct from synthetic implementation QA."""
    config = read_json(config_path)
    validate_config(config)
    require(config["arm"] == "W11" and updates >= 2, "Real source QA exercises W11 for at least two updates")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    require(not any(output_dir.iterdir()), "Source QA needs an empty output directory")
    device = prepare_runtime(device)
    inputs = inputs_for(config, candidate_source_qa=True)
    train = inputs.dataset(config["target_environment"], "train")
    validation = inputs.dataset(config["target_environment"], "validation")
    schedule = sampling_schedule(train, config["seed"])
    require(schedule_hash(train, schedule) == config["sampling_schedule_sha256"], "Source QA sampler drift")
    seed_all(config["seed"])
    model = make_model("tpp", config["seed"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0)
    started = time.perf_counter()
    losses = []
    model.train()
    for step, batch in enumerate(DataLoader(train, batch_sampler=schedule[:updates].tolist(), num_workers=0)):
        optimizer.zero_grad(set_to_none=True)
        optimizer.param_groups[0]["lr"] = 0.001 * 0.9 ** (step / 10000)
        loss, parts = objective(model, tuple(x.to(device) for x in batch), "W11")
        require(torch.isfinite(loss).item(), "Non-finite real source QA loss")
        loss.backward()
        require(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()), "Non-finite source gradient")
        optimizer.step()
        losses.append({"loss": float(loss.detach()), **parts})
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    metrics = evaluate_source(model, validation, device)
    # A strict reload must preserve actual source logits for the checkpoint.
    amp, phase, _labels, _domain, _id = validation[0]
    with torch.no_grad():
        before = model(amp.unsqueeze(0).to(device), phase.unsqueeze(0).to(device))["logits"].cpu()
    atomic_torch(output_dir / "source_qa_checkpoint.pt", {k: v.detach().cpu() for k, v in model.state_dict().items()})
    reloaded = make_model("tpp", config["seed"]).to(device)
    reloaded.load_state_dict(torch.load(output_dir / "source_qa_checkpoint.pt", map_location=device, weights_only=True), strict=True)
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(amp.unsqueeze(0).to(device), phase.unsqueeze(0).to(device))["logits"].cpu()
    require(torch.equal(before, after), "Source checkpoint reload changed logits")
    target_reads = sum(part == "test" for part, _id in inputs.signal_access)
    require(target_reads == 0, "Target signal read in source QA")
    atomic_json(output_dir / "source_qa.json", {"schema": "wimans-real-source-qa/v1", "at": now(), "status": "passed",
                "synthetic": False, "target_environment": config["target_environment"], "target_signal_rows_read": target_reads,
                "code_files_sha256": code_receipt(), "input_audit_sha256": config["input_audit_sha256"],
                "manifest_receipt": config["manifest_receipt"], "source_short_train_updates": updates,
                "source_train_batch_size": 32, "validation_rows": len(validation), "input_shape": [850, 90],
                "checked_model_arm": "W11", "expected_parameters": EXPECTED_PARAMETERS, "source_validation": metrics,
                "source_short_train_seconds": training_seconds, "runtime": runtime_receipt(device),
                "losses": losses, "checkpoint_sha256": file_sha256(output_dir / "source_qa_checkpoint.pt"),
                "checkpoint_strict_reload_logits_exact": True,
                "physical_axis_claim": "storage order only" if inputs.audit.get("representation_version") == "storage_order_v2" else "bound axis export audit"})


def source_shape_qa(config_path, output_dir):
    """Low-memory batch-one real-source CPU eval; never authorizes freeze."""
    config = read_json(config_path)
    validate_config(config)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    require(not any(output_dir.iterdir()), "Shape QA needs a new output directory")
    prepare_runtime("cpu")
    inputs = inputs_for(config, candidate_source_qa=True)
    source = inputs.dataset(config["target_environment"], "train")
    model = make_model("tpp", config["seed"])
    model.eval()
    amp, phase, label, domain, numeric_id = source[0]
    with torch.no_grad():
        output = model(amp.unsqueeze(0), phase.unsqueeze(0))
        require(output["logits"].shape == (1, 9) and torch.isfinite(output["logits"]).all().item(), "Source eval output invalid")
    atomic_torch(output_dir / "source_shape_checkpoint.pt", model.state_dict())
    reloaded = make_model("tpp", config["seed"])
    reloaded.load_state_dict(torch.load(output_dir / "source_shape_checkpoint.pt", map_location="cpu", weights_only=True), strict=True)
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(amp.unsqueeze(0), phase.unsqueeze(0))["logits"]
    require(torch.equal(output["logits"], after), "Source shape checkpoint reload changed logits")
    atomic_json(output_dir / "source_shape_qa.json", {"schema": "wimans-real-source-shape-qa/v1", "status": "passed",
                "synthetic": False, "real_source_training_qa_passed": False, "execution_freeze_authorized": False,
                "source_short_train_updates": 0, "source_eval_batch_size": 1, "input_shape": [850, 90],
                "target_environment": config["target_environment"], "sample_id": inputs.manifest.ids[numeric_id],
                "label": label, "domain_id": domain, "target_signal_rows_read": 0,
                "checkpoint_strict_reload_logits_exact": True,
                "checkpoint_sha256": file_sha256(output_dir / "source_shape_checkpoint.pt"),
                "code_files_sha256": code_receipt(), "input_audit_sha256": config["input_audit_sha256"],
                "manifest_receipt": config["manifest_receipt"], "runtime": runtime_receipt(torch.device("cpu"))})


def recorded_call(destination, command, callback, *, config_path=None, device="cpu", resume=False):
    """Append each attempt outside the run's empty-directory gate, including failures."""
    destination = Path(destination).resolve()
    receipt_path = destination.parent / "attempts" / (destination.name + "." + str(uuid.uuid4()) + ".json")
    receipt = {"schema": "wimans-execution-attempt/v1", "started_at": now(), "pid": os.getpid(),
               "command": command, "destination": str(destination), "device_requested": device,
               "resume_requested": resume, "config_path": str(Path(config_path).resolve()) if config_path else None,
               "status": "running", "code_files_sha256": code_receipt(), "start_boundary_optimizer_updates": None}
    if resume and (destination / "latest.pt").exists():
        boundary = torch.load(destination / "latest.pt", map_location="cpu", weights_only=False)
        receipt["start_boundary_optimizer_updates"] = boundary.get("optimizer_updates")
        del boundary
    elif command == "train-source":
        receipt["start_boundary_optimizer_updates"] = 0
    atomic_json(receipt_path, receipt)
    started = time.perf_counter()
    try:
        result = callback()
    except BaseException as error:
        receipt.update({"status": "failed_or_interrupted", "finished_at": now(),
                        "elapsed_seconds": time.perf_counter() - started,
                        "error": type(error).__name__ + ": " + str(error)})
        atomic_json(receipt_path, receipt)
        raise
    receipt.update({"status": "completed", "finished_at": now(), "elapsed_seconds": time.perf_counter() - started})
    atomic_json(receipt_path, receipt)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train-source")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--run-dir", type=Path, required=True)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--device", default="cuda:0")
    evaluate = commands.add_parser("evaluate-target")
    evaluate.add_argument("--run-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="cuda:0")
    qa = commands.add_parser("source-qa")
    qa.add_argument("--config", type=Path, required=True)
    qa.add_argument("--output-dir", type=Path, required=True)
    qa.add_argument("--device", default="cpu")
    qa.add_argument("--updates", type=int, default=2)
    shape = commands.add_parser("source-shape-qa")
    shape.add_argument("--config", type=Path, required=True)
    shape.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train-source":
        recorded_call(args.run_dir, args.command,
                      lambda: train_source(args.config, args.run_dir, device=args.device, resume=args.resume),
                      config_path=args.config, device=args.device, resume=args.resume)
    elif args.command == "evaluate-target":
        recorded_call(args.run_dir, args.command, lambda: evaluate_target(args.run_dir, device=args.device), device=args.device)
    elif args.command == "source-qa":
        recorded_call(args.output_dir, args.command,
                      lambda: source_qa(args.config, args.output_dir, device=args.device, updates=args.updates),
                      config_path=args.config, device=args.device)
    else:
        recorded_call(args.output_dir, args.command, lambda: source_shape_qa(args.config, args.output_dir), config_path=args.config)


if __name__ == "__main__":
    main()
