"""Small, atomic and hash-bound storage helpers for prospective revision runs."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def event(directory, kind, **fields):
    with (Path(directory) / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"time": now(), "event": kind, **fields}, ensure_ascii=False) + "\n")
        stream.flush()


def bound_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f"Path is not a child of the study directory: {relative}")
    return path


def verify_bundle(bundle):
    bundle = Path(bundle).resolve()
    lock = read(bundle / "bundle_manifest.json")
    for relative, expected in lock["files_sha256"].items():
        if sha256(bound_path(bundle, relative)) != expected:
            raise RuntimeError(f"Frozen bundle hash mismatch: {relative}")
    protocol = read(bundle / "protocol.json")
    if protocol["schema"] != "revision-core-study/v1":
        raise ValueError("Unsupported study protocol")
    return protocol, sha256(bundle / "bundle_manifest.json")


def verify_completed(directory, config):
    directory = Path(directory)
    status = read(directory / "status.json")
    if status["status"] != "completed":
        raise RuntimeError(f"Run is not completed: {directory}")
    if status["config_value_sha256"] != value_hash(config):
        raise RuntimeError("Completed run belongs to a different configuration")
    if status["optimizer_updates"] != config["epochs"] * config["updates_per_epoch"]:
        raise RuntimeError("Wrong number of optimizer updates")
    for relative, expected in status["artifacts_sha256"].items():
        if sha256(bound_path(directory, relative)) != expected:
            raise RuntimeError(f"Completed artifact hash mismatch: {relative}")
    history = read(directory / "history.json")
    if len(history) != config["epochs"]:
        raise RuntimeError("Incomplete validation history")
    selected = max(range(len(history)), key=lambda i: history[i]["source_validation"]["macro_f1"])
    if status["best_epoch"] != selected + 1:
        raise RuntimeError("Checkpoint does not use the earliest source-validation maximum")
    if status["best_source_macro_f1"] != history[selected]["source_validation"]["macro_f1"]:
        raise RuntimeError("Selected source metric mismatch")
    return status
