"""Lazy amplitude-only source batches with a reproducible class/domain schedule."""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.revision_validation.prepare_verified_protocol import sha256


class VerifiedSources:
    def __init__(self, manifest_path):
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        self.target = self.manifest["outer_target"]
        self.sources = self.manifest["source_environments"]
        assert self.target not in self.sources and len(self.sources) == 2
        assert self.manifest["inner_validation"] == "subject_grouped_holdout"
        assert set(self.manifest["files"]) == set(self.sources)
        self.arrays = {}
        self.opened_paths = []
        for env, files in self.manifest["files"].items():
            path = files["data_files"]["amp"]["path"]
            assert sha256(path) == files["actual_file_sha256"]["amp"]
            self.arrays[env] = np.load(path, mmap_mode="r", allow_pickle=False)
            self.opened_paths.append(path)
        for part in ("train", "val"):
            assert all(row["environment"] in self.sources for row in self.manifest[part])
        assert not ({row["subject_id"] for row in self.manifest["train"]} &
                    {row["subject_id"] for row in self.manifest["val"]})
        assert not ({row["acquisition_id"] for row in self.manifest["train"]} &
                    {row["acquisition_id"] for row in self.manifest["val"]})

    def batch(self, part, positions, device):
        rows = [self.manifest[part][int(i)] for i in positions]
        amplitude = np.stack([self.arrays[row["environment"]][row["index"]] for row in rows]).astype(np.float32)
        labels = np.array([row["label"] for row in rows], dtype=np.int64)
        return torch.from_numpy(amplitude).to(device), torch.from_numpy(labels).to(device), rows


def epoch_schedule(rows, seed, epoch, steps=300):
    """Each 16-item batch has eight/source, all six labels/source, no duplicates.

    Private RNG and explicit epoch seeding make the schedule independent of model
    initialization, dropout, arm order and prior execution. Sampling intentionally
    balances classes; an epoch is 300 updates rather than an exhaustive pass.
    """
    rng = random.Random(int(seed) * 1_000_003 + int(epoch))
    environments = sorted({row["environment"] for row in rows})
    assert len(environments) == 2
    pools = {(env, label): [index for index, row in enumerate(rows)
                           if row["environment"] == env and row["label"] == label]
             for env in environments for label in range(6)}
    assert all(pools.values())
    for pool in pools.values():
        rng.shuffle(pool)
    offsets = {key: 0 for key in pools}
    schedule = []
    for _ in range(steps):
        batch = []
        for env in environments:
            classes = list(range(6))
            rng.shuffle(classes)
            for label in classes + classes[:2]:
                key = env, label
                pool = pools[key]
                while True:
                    if offsets[key] == len(pool):
                        rng.shuffle(pool)
                        offsets[key] = 0
                    index = pool[offsets[key]]
                    offsets[key] += 1
                    if index not in batch:
                        batch.append(index)
                        break
        rng.shuffle(batch)
        assert len(batch) == len(set(batch)) == 16
        assert all(value >= 2 for value in Counter(rows[i]["label"] for i in batch).values())
        schedule.append(batch)
    return np.asarray(schedule, dtype="<i8")


def schedule_hash(schedule):
    return hashlib.sha256(np.asarray(schedule, dtype="<i8").tobytes()).hexdigest()
