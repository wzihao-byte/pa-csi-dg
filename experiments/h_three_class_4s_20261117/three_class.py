"""Three-class (falling / walking / turning) restriction of the main H-S study.

Everything here is a pure function of the frozen main-study membership (the H-S
source manifests) and the legacy loader output.  Classes keep their order:
original label 1 (falling) -> 0, 2 (walking) -> 1, 4 (turning) -> 2.
Source samples keep their main-study global identity, ENV_ORDER.index(env) * 3000
+ within-environment index, so split hashes are comparable with the main study.
"""
from __future__ import annotations

import hashlib
import random
import struct
from collections import Counter
from types import SimpleNamespace

import numpy as np

ENV_ORDER = ("E1", "E2", "E3")
KEEP = (1, 2, 4)
LABEL_MAP = {1: 0, 2: 1, 4: 2}
CLASS_NAMES = ("falling", "walking", "turning")
NUM_CLASSES = 3
PER_ENV = 1200            # 3 classes x 400 segments
TRAIN_COUNT = 1920        # 2 source environments x 960
VAL_COUNT = 480           # 2 source environments x 240
TARGET_COUNT = 1200
EPOCHS = 200
BATCH = 32
STEPS_PER_EPOCH = 60      # ceil(960 / 16)
TOTAL_STEPS = EPOCHS * STEPS_PER_EPOCH   # 12,000
SAMPLES_SEEN = TOTAL_STEPS * BATCH        # 384,000


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def index_hash(values):
    return hashlib.sha256(np.asarray(values, dtype="<i8").tobytes()).hexdigest()


def remap_labels(labels):
    labels = np.asarray(labels)
    out = np.full(labels.shape, -1, dtype=np.int64)
    for old, new in LABEL_MAP.items():
        out[labels == old] = new
    require(np.all(out >= 0), "Label outside the three kept classes")
    return out


def filter_manifest(source):
    """Copy of an H-S source manifest restricted to the three classes (labels remapped)."""
    out = {k: v for k, v in source.items() if k not in ("train", "val")}
    for part in ("train", "val"):
        rows = [dict(r) for r in source[part] if int(r["label"]) in KEEP]
        for r in rows:
            r["original_label"] = int(r["label"])
            r["label"] = LABEL_MAP[int(r["label"])]
        out[part] = rows
    return out


def global_id(row):
    return ENV_ORDER.index(row["environment"]) * 3000 + int(row["index"])


def metadata_dataset(source_filtered):
    """Array-free stand-in for the loaded, restricted source dataset (same order)."""
    envs = list(source_filtered["source_environments"])
    rows = sorted((r for p in ("train", "val") for r in source_filtered[p]),
                  key=lambda r: (envs.index(r["environment"]), int(r["index"])))
    require(len(rows) == 2 * PER_ENV, "Restricted source inventory is not 2 x 1200")
    return SimpleNamespace(env_names=envs,
                           labels=np.array([r["label"] for r in rows], dtype=np.int64),
                           env_ids=np.array([envs.index(r["environment"]) for r in rows], dtype=np.int64),
                           sample_ids=np.array([global_id(r) for r in rows], dtype=np.int64))


def grouped_split(dataset, source_filtered, target):
    require(source_filtered["outer_target"] == target, "Wrong source partition")
    require(source_filtered["inner_validation"] == "subject_grouped_holdout", "Not the grouped H-S split")
    require(list(dataset.env_names) == source_filtered["source_environments"] and target not in dataset.env_names,
            "Source environments changed")
    positions = {int(s): i for i, s in enumerate(dataset.sample_ids)}
    require(len(positions) == len(dataset.sample_ids), "Duplicate global sample id")
    parts, identity = {}, {}
    for name, count in (("train", TRAIN_COUNT), ("val", VAL_COUNT)):
        rows = source_filtered[name]
        require(len(rows) == count, f"Restricted {name} size {len(rows)} != {count}")
        ids = [global_id(r) for r in rows]
        require(len(set(ids)) == count, "Duplicate grouped sample")
        require(all(r["environment"] != target for r in rows), "Target row in source partition")
        require(all(int(dataset.labels[positions[x]]) == int(r["label"]) for x, r in zip(ids, rows)),
                "Restricted labels differ from loader")
        parts[name] = np.sort([positions[x] for x in ids])
        identity[name] = {k: sorted({r[k] for r in rows}) for k in ("sample_id", "subject_id", "acquisition_id")}
    for k in ("sample_id", "subject_id", "acquisition_id"):
        require(not set(identity["train"][k]) & set(identity["val"][k]), f"Train/validation {k} overlap")
    require(set(parts["train"]) | set(parts["val"]) == set(range(len(dataset.sample_ids))), "Incomplete source inventory")
    split = {"mode": "dg_loeo", "target_env": target, "source_envs": list(dataset.env_names),
             "train_indices": parts["train"].tolist(), "val_indices": parts["val"].tolist(),
             "test_indices": parts["val"].tolist()}
    return split, identity


def replay_sampler(dataset, split, seed, sampler_class):
    """Replay the legacy Python-RNG domain/class-balanced sampler for 200 epochs."""
    prior = random.getstate()
    try:
        random.seed(seed)
        indices = split["train_indices"]
        domains = {d: [i for i in indices if int(dataset.env_ids[i]) == d] for d in range(2)}
        sampler = sampler_class(domains, dataset.labels, batch_size=BATCH)
        require(len(sampler) == STEPS_PER_EPOCH, f"Sampler gives {len(sampler)} batches, expected {STEPS_PER_EPOCH}")
        digest = hashlib.sha256()
        schedule = np.empty((EPOCHS, STEPS_PER_EPOCH, BATCH), dtype="<i8")
        class_counts = Counter()
        for epoch in range(EPOCHS):
            digest.update(struct.pack("<Q", epoch + 1))
            for step, batch in enumerate(sampler):
                require(Counter(int(dataset.env_ids[i]) for i in batch) == {0: 16, 1: 16}, "Unbalanced batch")
                class_counts.update(int(dataset.labels[i]) for i in batch)
                ids = np.asarray(dataset.sample_ids[batch], dtype="<i8")
                schedule[epoch, step] = ids
                digest.update(ids.tobytes())
        return digest.hexdigest(), schedule, dict(sorted(class_counts.items()))
    finally:
        random.setstate(prior)


def subset_loaded(dataset, dataset_class, role):
    """Restrict a legacy PaCsiArrayDataset to the three classes.

    role='source': sample_ids become main-study global ids (kept by the patched remap).
    role='target': sample_ids become 0..1199 in original order (the trainer requires an
    ordered target); the original within-environment indices are kept on the object.
    """
    keep_parts, global_parts = [], []
    for local_env, env_name in enumerate(dataset.env_names):
        env_idx = np.flatnonzero(dataset.env_ids == local_env)
        mask = np.isin(dataset.labels[env_idx], KEEP)
        keep_parts.append(env_idx[mask])
        global_parts.append(ENV_ORDER.index(env_name) * 3000 + np.arange(len(env_idx))[mask])
    keep = np.concatenate(keep_parts)
    global_ids = np.concatenate(global_parts).astype(np.int64)
    require(len(keep) == PER_ENV * len(dataset.env_names), "Restricted size is not 1200 per environment")
    labels = remap_labels(dataset.labels[keep])
    if role == "source":
        sample_ids = global_ids
    elif role == "target":
        require(len(dataset.env_names) == 1, "Target loader must hold one environment")
        sample_ids = np.arange(len(keep), dtype=np.int64)
    else:
        raise ValueError(role)
    out = dataset_class(dataset.amplitude[keep], dataset.phase[keep], labels,
                        dataset.env_ids[keep], dataset.env_names, sample_ids)
    out.three_class_global_ids = global_ids
    out.three_class_role = role
    return out
