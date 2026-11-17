"""Checks shared by the original-H grouped-validation supplement.

The frozen legacy loader, model, loss, optimizer and sampler remain authoritative.
The R manifests are used only for verified sample/subject membership.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import struct
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(r"C:/Users/howie/Desktop/pa-csi-dg")
OUTPUT = ROOT / "outputs_h_grouped_l1248/v2"
BUNDLE = OUTPUT / "bundle"
TARGETS = ("E1", "E2", "E3")
SEEDS = (42, 52, 62)
ARMS = ("H10", "H11")
PARAMETERS = 4_786_294


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def legacy(kind="trainer"):
    local = HERE / f"base_{kind}.py"
    filename = "train_factorial.py" if kind == "trainer" else "run_factorial.py"
    path = local if local.exists() else ROOT / "experiments/supcon_tpp_legacy_reproduction" / filename
    return load_module(f"h_grouped_base_{kind}", path)


def index_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def state_hash(model):
    h = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()


def changed_paths(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result = set()
        for key in set(a) | set(b):
            path = f"{prefix}.{key}" if prefix else key
            result |= {path} if key not in a or key not in b else changed_paths(a[key], b[key], path)
        return result
    return {prefix} if a != b else set()


def verify_config(config):
    supplement = config["h_grouped"]
    reference_path = Path(supplement["reference_config"])
    require(sha(reference_path) == supplement["reference_config_sha256"], "Historical configuration changed")
    reference = read(reference_path)
    comparison_reference = copy.deepcopy(reference)
    comparison_reference.pop("output_dir", None)
    candidate = copy.deepcopy(config)
    candidate.pop("h_grouped")
    candidate.pop("output_dir", None)
    allowed = {"experiment_name", "job_id", "protocol.version",
               "protocol.expected_split_hashes.train", "protocol.expected_split_hashes.val"}
    require(changed_paths(comparison_reference, candidate) == allowed, "Unexpected scientific configuration change")
    require(config["model"]["tpp_levels"] == [1, 2, 4, 8], "Not the H L1248 model")
    require(config["protocol"]["expected_parameter_count"] == PARAMETERS, "Wrong parameter count")
    require(config["protocol"]["checkpoint_rule"] == "source_val_accuracy_last_tied_best", "Selection rule changed")
    require(config["factorial"]["arm"] in ARMS and config["seed_list"][0] in SEEDS, "Unknown arm/seed")
    require(sha(supplement["source_manifest"]) == supplement["source_manifest_sha256"], "Grouped membership changed")
    return reference


def metadata_dataset(source):
    environments = list(source["source_environments"])
    rows = {r["sample_id"]: r for part in ("train", "val") for r in source[part]}
    ordered = [rows[f"{e}:{i:04d}"] for e in environments for i in range(3000)]
    return SimpleNamespace(env_names=environments, labels=np.array([r["label"] for r in ordered]),
                           env_ids=np.repeat(np.arange(2), 3000),
                           sample_ids=np.array([TARGETS.index(r["environment"]) * 3000 + r["index"] for r in ordered]))


def grouped_split(dataset, source, target):
    require(source["outer_target"] == target and source["inner_validation"] == "subject_grouped_holdout", "Wrong source partition")
    require(list(dataset.env_names) == source["source_environments"] and target not in dataset.env_names, "Source environments changed")
    positions = {int(sample_id): i for i, sample_id in enumerate(dataset.sample_ids)}
    parts, identity = {}, {}
    for name, count in (("train", 4800), ("val", 1200)):
        rows = source[name]
        require(len(rows) == count, "Wrong grouped partition size")
        ids = [TARGETS.index(r["environment"]) * 3000 + int(r["index"]) for r in rows]
        require(len(set(ids)) == count, "Duplicate grouped sample")
        require(all(r["environment"] != target for r in rows), "Target row in source partition")
        indices = np.sort([positions[x] for x in ids])
        require(all(int(dataset.labels[positions[x]]) == int(r["label"]) for x, r in zip(ids, rows)), "Grouped labels differ from legacy loader")
        parts[name] = indices
        identity[name] = {key: sorted({r[key] for r in rows}) for key in ("sample_id", "subject_id", "acquisition_id")}
    for key in ("sample_id", "subject_id", "acquisition_id"):
        require(not set(identity["train"][key]) & set(identity["val"][key]), f"Train/validation {key} overlap")
    require(set(parts["train"]) | set(parts["val"]) == set(range(6000)), "Incomplete source inventory")
    split = {"mode": "dg_loeo", "target_env": target, "source_envs": list(dataset.env_names),
             "train_indices": parts["train"].tolist(), "val_indices": parts["val"].tolist(),
             "test_indices": parts["val"].tolist()}
    return split, identity


def replay_sampler(dataset, split, seed, sampler_class):
    """Replay the original continuous Python RNG; never substitute the R sampler."""
    prior = random.getstate()
    try:
        random.seed(seed)
        indices = split["train_indices"]
        domains = {d: [i for i in indices if int(dataset.env_ids[i]) == d] for d in range(2)}
        sampler = sampler_class(domains, dataset.labels, batch_size=32)
        require(len(sampler) == 150, "Legacy sampler update budget changed")
        digest = hashlib.sha256()
        schedule = np.empty((200, 150, 32), dtype="<i8")
        for epoch in range(200):
            digest.update(struct.pack("<Q", epoch + 1))
            for step, batch in enumerate(sampler):
                require(Counter(int(dataset.env_ids[i]) for i in batch) == {0: 16, 1: 16}, "Unbalanced batch")
                ids = np.asarray(dataset.sample_ids[batch], dtype="<i8")
                schedule[epoch, step] = ids
                digest.update(ids.tobytes())
        return digest.hexdigest(), schedule
    finally:
        random.setstate(prior)


def verify_bundle(bundle=BUNDLE, require_qa=True):
    bundle = Path(bundle)
    manifest = read(bundle / "bundle_manifest.json")
    for relative, expected in manifest["files_sha256"].items():
        require(sha(bundle / relative) == expected, f"Frozen supplement changed: {relative}")
    for path, expected in manifest["external_code_sha256"].items():
        require(sha(path) == expected, f"Historical dependency changed: {path}")
    registry = read(bundle / "registry.json")
    expected = [(e, s, a) for e in TARGETS for s in SEEDS for a in ARMS]
    require([(j["target"], j["seed"], j["arm"]) for j in registry["jobs"]] == expected, "Wrong 18-job grid")
    for job in registry["jobs"]:
        verify_config(read(bundle / job["config"]))
    if require_qa:
        qa_path = bundle.parent / "qa/approval.json"
        qa = read(qa_path)
        require(qa["status"] == "passed" and qa["bundle_manifest_sha256"] == sha(bundle / "bundle_manifest.json"), "Supplement QA not approved for this bundle")
        require(qa["qa_source_sha256"] == manifest["files_sha256"]["code/qa.py"], "QA implementation binding changed")
        require(qa["external_code_sha256"] == manifest["external_code_sha256"], "QA historical dependency binding changed")
        require(qa["target_arrays_opened"] is False and len(qa["grouped_schedules"]) == 9, "Incomplete source-only QA")
        require(set(qa["cpu_training_equivalence"]) == set(ARMS), "Missing real-data equivalence for an arm")
    return manifest, registry
