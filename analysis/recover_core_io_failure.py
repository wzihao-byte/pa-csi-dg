"""Diagnose the committed-epoch Windows status replacement failure before resume.

This permits only the specific non-numerical failure after an intact committed
checkpoint. It preserves failed state and never changes frozen code/configuration.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.revision_validation.study_io import bound_path, now, read, sha256, value_hash, verify_bundle, write


def all_finite(value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(all_finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(all_finite(v) for v in value)
    if isinstance(value, float):
        return bool(np.isfinite(value))
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core-bundle", type=Path, required=True)
    parser.add_argument("--controls-bundle", type=Path, required=True)
    args = parser.parse_args()
    bundle, controls = args.core_bundle.resolve(), args.controls_bundle.resolve()
    protocol, bundle_hash = verify_bundle(bundle)
    queue = read(bundle / "queue_status.json")
    config = next(c for c in protocol["jobs"] if c["job_id"] == queue["job_id"])
    run = bound_path(bundle, "runs/" + config["job_id"])
    status, failure = read(run / "status.json"), read(bundle / "queue_failure.json")
    message = status["error"]
    if status["status"] != "failed" or "PermissionError: [WinError 5]" not in message or "status.json.tmp" not in message or 'write(output / "status.json"' not in message:
        raise RuntimeError("This recovery applies only to the diagnosed Windows status replacement failure")
    if (bundle / "runner.lock.json").exists() or (controls / "runner.lock.json").exists():
        raise RuntimeError("A runner lock remains; establish that processes exited before recovery")
    progress = read(run / "progress.json")
    checkpoint = bound_path(run, progress["checkpoint"])
    if sha256(checkpoint) != progress["sha256"]:
        raise RuntimeError("Last committed checkpoint hash differs")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["config_value_sha256"] != value_hash(config) or read(run / "config.json") != config:
        raise RuntimeError("Resume configuration differs from the frozen job")
    epochs = len(state["history"])
    if state["step"] != status["optimizer_updates_attempted"] or state["step"] != epochs*config["updates_per_epoch"] or epochs != progress["completed_epochs"]:
        raise RuntimeError("The I/O failure was not after exactly the last complete epoch")
    if state["history"] != read(run / "history.json") or not all_finite(state):
        raise RuntimeError("Committed history or numeric checkpoint state is invalid")
    best_index = max(range(epochs), key=lambda i:state["history"][i]["source_validation"]["macro_f1"])
    if state["best_epoch"] != best_index+1 or state["best"] != state["history"][best_index]["source_validation"]["macro_f1"]:
        raise RuntimeError("Saved best selection differs from the frozen earliest-source maximum")
    source = read(config["source_manifest_path"])
    if len(state["best_predictions"]) != len(source["val"]):
        raise RuntimeError("Saved source predictions are incomplete")
    for actual, predicted in zip(source["val"], state["best_predictions"]):
        if predicted["true"] != actual["label"] or any(predicted[k] != actual[k] for k in ("sample_id", "subject_id", "acquisition_id")):
            raise RuntimeError("Saved source predictions have inconsistent identities")
    dossier = bound_path(bundle, "recovery/windows_status_io_001")
    if dossier.exists():
        raise FileExistsError("Recovery already recorded; do not erase or repeat it")
    dossier.mkdir(parents=True)
    saved = {}
    for original, name in ((run / "status.json", "failed_run_status.json"), (bundle / "queue_failure.json", "core_queue_failure.json"),
                           (bundle / "runner.stdout.log", "core_stdout.log"), (bundle / "runner.stderr.log", "core_stderr.log"),
                           (run / "progress.json", "committed_progress.json"), (run / "history.json", "committed_history.json"),
                           (controls / "queue_failure.json", "dependent_queue_failure.json"),
                           (controls / "runner.stderr.log", "dependent_stderr.log")):
        shutil.copy2(original, dossier / name)
        saved[name] = sha256(dossier / name)
    decision = {"diagnosed_at": now(), "job_id": config["job_id"], "cause": "Windows denied atomic status-file replacement after checkpoint/progress/history writes",
                "file_holder_identified": False, "possible_contributor": "Dependency observer polls the same status file without FILE_SHARE_DELETE; its next failure timestamp follows this event by 30 seconds",
                "bundle_sha256_unchanged": bundle_hash, "checkpoint_sha256": progress["sha256"], "committed_epochs": epochs,
                "committed_updates": state["step"], "updates_to_replay": 0, "all_numeric_state_finite": True,
                "source_identity_and_earliest_best_verified": True, "target_data_loaded": False,
                "action": "Archive failed flags after diagnosis, then resume the unmodified frozen engine from its committed checkpoint",
                "mitigation": "Use Windows shared-delete observer reads; defer the original control runner until the core process exits successfully",
                "files_sha256": saved}
    write(dossier / "decision.json", decision)
    # Retain the original failed flags under the dossier, rather than replacing
    # them with a fictitious paused status. The existing progress enables resume.
    for original, name, allowed in ((run / "status.json", "original_failed_run_status.json", bundle),
                                    (bundle / "queue_failure.json", "original_core_queue_failure.json", bundle),
                                    (controls / "queue_failure.json", "original_dependent_queue_failure.json", controls)):
        if not original.resolve().is_relative_to(allowed):
            raise RuntimeError("Recovery path escaped its explicitly named study directory")
        original.replace(dossier / name)
    if verify_bundle(bundle)[1] != bundle_hash:
        raise RuntimeError("Frozen bundle changed during recovery")
    print(decision)


if __name__ == "__main__":
    main()
