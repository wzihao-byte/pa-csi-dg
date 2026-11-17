"""Independent identity-driven WiMANS adapter; no MultiEnv size assumptions.

Array headers and identity/label metadata can cover the entire subset. Source
datasets only retrieve source rows. Target signal retrieval requires a frozen
checkpoint receipt and is performed by the separate evaluation command.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ("nothing", "walk", "rotation", "jump", "wave", "lie_down",
           "pick_up", "sit_down", "stand_up")
ROOMS = ("classroom", "meeting_room", "empty_room")
PHASE_REPRESENTATION = "native_historical_feature_unwrap_v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity_hash(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class Manifest:
    def __init__(self, subset_path, folds_path):
        self.subset_path, self.folds_path = Path(subset_path), Path(folds_path)
        with self.subset_path.open(encoding="utf-8-sig", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        self.ids = [row["sample_id"] for row in self.rows]
        self.by_id = dict(zip(self.ids, self.rows))
        require(len(self.rows) == len(self.by_id) == 1782, "Expected 1782 unique subset identities")
        for row in self.rows:
            require(row["environment"] in ROOMS and row["wifi_band"] == "5", "Subset environment/band drift")
            require(CLASSES[int(row["class_id"])] == row["activity"], "Nine-class order drift")
        payload = read_json(self.folds_path)
        self.folds = {f["target_environment"]: f for f in payload["folds"]}
        require(set(self.folds) == set(ROOMS), "Expected three fixed room folds")
        for target, fold in self.folds.items():
            require(set(fold["source_environments"]) == set(ROOMS) - {target}, "Source room roster drift")
            parts = fold["sample_ids"]
            require(set(parts) == {"train", "validation", "test"}, "Unexpected fold partitions")
            members = [set(parts[key]) for key in ("train", "validation", "test")]
            require(all(len(parts[key]) == len(members[i]) for i, key in enumerate(("train", "validation", "test"))), "Duplicate fold identity")
            require(not any(a & b for i, a in enumerate(members) for b in members[i + 1:]), "Overlapping fold sample identity")
            require(set.union(*members) == set(self.ids), "Fold does not cover exactly the subset")
            for part, count, per_class in (("train", 792, 88), ("validation", 396, 44), ("test", 594, 66)):
                require(len(parts[part]) == count, "Fixed fold count drift")
                rows = [self.by_id[x] for x in parts[part]]
                require([sum(int(r["class_id"]) == c for r in rows) for c in range(9)] == [per_class] * 9,
                        "Fixed fold class count drift")
                expected_rooms = {target} if part == "test" else set(fold["source_environments"])
                expected_users = set(fold["target_users"] if part == "test" else
                                     fold["source_training_users"] if part == "train" else fold["source_validation_users"])
                require({r["environment"] for r in rows} == expected_rooms, "Fold room membership drift")
                require({int(r["subject_id"]) for r in rows} == expected_users, "Fold user membership drift")
            group_sets = [{self.by_id[x]["acquisition_group"] for x in parts[key]} for key in ("train", "validation", "test")]
            require(not any(a & b for i, a in enumerate(group_sets) for b in group_sets[i + 1:]), "Fold acquisition overlap")
            require(not set(fold["source_training_users"]) & set(fold["source_validation_users"]), "Source subject overlap")

    def receipt(self):
        return {"subset_manifest_sha256": file_sha256(self.subset_path),
                "fold_manifest_sha256": file_sha256(self.folds_path),
                "fold_identity_sha256": {room: {part: identity_hash(ids) for part, ids in fold["sample_ids"].items()}
                                         for room, fold in self.folds.items()}}


def require_formal_input(audit):
    require(audit.get("status") in {"passed", "passed_storage_order_v2"} and audit.get("full_input_qa_passed") is True,
            "Formal training blocked: full input QA has not passed")
    require(audit.get("numerical_input_qa_passed") is True and
            audit.get("expected_rows") == audit.get("candidate_rows") == audit.get("completed_rows") == 1782 and
            audit.get("preview") is False and audit.get("failures") == [], "Full-subset numerical acceptance is incomplete")
    if audit.get("status") == "passed_storage_order_v2":
        binding = audit.get("axis_storage_protocol", {})
        require("path" in binding and "sha256" in binding, "Missing registered storage-order protocol binding")
        require(file_sha256(binding["path"]) == binding["sha256"], "Registered storage-order protocol drift")
        protocol = read_json(binding["path"])
        require(protocol.get("status") == "registered" and protocol.get("representation_version") == "storage_order_v2" and
                protocol.get("target_model_scores_accessed_before_registration") is False and
                protocol.get("perm_reapplied") is False and protocol.get("physical_tx_index_confirmed") is False,
                "Invalid storage-order fallback registration")
        storage_expected = {"selected_first_stored_axis_index": 0, "retained_second_stored_axis_indices": [0, 1, 2],
                            "retained_last_axis_indices": list(range(30)), "flatten_order": "C",
                            "feature_group_slices": [[0, 30], [30, 60], [60, 90]], "raw_layout_required": [3, 3, 30],
                            "metadata_Ntx_Nrx_required": [3, 3], "input_phase_representation": PHASE_REPRESENTATION,
                            "phase_unwrap_axis": -1, "loader_unwrap_phase": False, "physical_rx_mapping_confirmed": False}
        require(all(protocol.get(key) == value for key, value in storage_expected.items()), "Registered storage-order slicing drift")
        require(audit.get("representation_version") == "storage_order_v2" and
                audit.get("axis_export_verification", {}).get("status") == "storage_order_protocol_registered",
                "Storage-order fallback audit/version mismatch")
    else:
        require(audit.get("axis_export_verification", {}).get("status") == "passed",
                "Formal training blocked: MAT axis/export order is unresolved")
    require(audit.get("phase_representation") == PHASE_REPRESENTATION, "Unregistered phase representation")
    require(audit.get("loader_unwrap_phase") is False, "Phase must not be unwrapped again by the loader")


class ProcessedInput:
    def __init__(self, directory, manifest: Manifest, *, candidate_source_qa=False):
        self.directory, self.manifest = Path(directory), manifest
        self.audit = read_json(self.directory / "input_audit.json")
        if candidate_source_qa:
            require(self.audit.get("numerical_input_qa_passed") is True,
                    "Candidate source QA requires complete numerical input audit")
            require(self.audit.get("phase_representation") == PHASE_REPRESENTATION, "Candidate phase representation drift")
        else:
            require_formal_input(self.audit)
        require(self.audit.get("manifest_sha256") == file_sha256(manifest.subset_path) and
                self.audit.get("fold_manifest_sha256") == file_sha256(manifest.folds_path), "Input audit manifest binding drift")
        require(self.audit.get("loader_unwrap_phase") is False, "Processed phase must not be unwrapped twice")
        self.candidate_source_qa = candidate_source_qa
        for name in ("amp.npy", "phase.npy", "label.npy", "sample_ids.json"):
            record = self.audit["processed_files"][name]
            expected = record["sha256"] if isinstance(record, dict) else record
            require(file_sha256(self.directory / name) == expected, "Processed file drift: " + name)
        self.ids = read_json(self.directory / "sample_ids.json")
        require(isinstance(self.ids, list) and len(self.ids) == len(set(self.ids)) == 1782,
                "Processed identities must be 1782 unique IDs")
        require(set(self.ids) == set(manifest.ids), "Processed identity membership drift")
        self.id_to_row = {sample_id: i for i, sample_id in enumerate(self.ids)}
        self.amp = np.load(self.directory / "amp.npy", mmap_mode="r", allow_pickle=False)
        self.phase = np.load(self.directory / "phase.npy", mmap_mode="r", allow_pickle=False)
        self.labels = np.load(self.directory / "label.npy", mmap_mode="r", allow_pickle=False)
        require(self.amp.shape == self.phase.shape == (1782, 850, 90), "Processed signal shape drift")
        require(self.amp.dtype == self.phase.dtype == np.float32, "Signals must be float32")
        require(self.labels.shape == (1782,) and self.labels.dtype == np.int64, "Labels must be int64 [1782]")
        require(np.array_equal(self.labels, [int(manifest.by_id[x]["class_id"]) for x in self.ids]), "Label identity drift")
        self.signal_access = []

    def dataset(self, target, partition, frozen_receipt=None):
        require(target in self.manifest.folds, "Unknown target room")
        require(partition in {"train", "validation", "test"}, "Unknown partition")
        if partition == "test":
            require(not self.candidate_source_qa, "Candidate source QA cannot open target")
            require(frozen_receipt and frozen_receipt.get("strict_reloaded") is True and
                    frozen_receipt.get("target_data_opened") is False and
                    frozen_receipt.get("target_environment") == target and
                    frozen_receipt.get("optimizer_updates") == 30000 and
                    frozen_receipt.get("manifest_receipt") == self.manifest.receipt(),
                    "Target signals require strict reloaded frozen checkpoint")
            require("checkpoint_path" in frozen_receipt and
                    file_sha256(frozen_receipt["checkpoint_path"]) == frozen_receipt.get("checkpoint_sha256"),
                    "Target signals require an unchanged checkpoint file")
        return PartitionDataset(self, target, partition)


class PartitionDataset(Dataset):
    def __init__(self, inputs, target, partition):
        self.inputs, self.target, self.partition = inputs, target, partition
        self.ids = inputs.manifest.folds[target]["sample_ids"][partition]
        self.rows = [inputs.id_to_row[x] for x in self.ids]
        self.labels = np.array([int(inputs.manifest.by_id[x]["class_id"]) for x in self.ids], dtype=np.int64)
        self.domains = np.array([ROOMS.index(inputs.manifest.by_id[x]["environment"]) for x in self.ids], dtype=np.int64)
        self.numeric_ids = np.array([inputs.manifest.ids.index(x) for x in self.ids], dtype=np.int64)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        row = self.rows[index]
        self.inputs.signal_access.append((self.partition, self.ids[index]))
        # Own arrays avoid non-writable memmap tensors; phase is already unwrapped.
        amp = torch.from_numpy(np.array(self.inputs.amp[row], copy=True))
        phase = torch.from_numpy(np.array(self.inputs.phase[row], copy=True))
        require(torch.isfinite(amp).all().item() and torch.isfinite(phase).all().item(), "Non-finite selected source/target row")
        return amp, phase, int(self.labels[index]), int(self.domains[index]), int(self.numeric_ids[index])


def sampling_schedule(dataset, seed, *, epochs=200, updates_per_epoch=150):
    """Registered WiMANS domain_class_balanced convention, independent RNG.

Fresh shuffled class pools each virtual epoch, 16 draws per source: each class
once plus seven classes twice, chosen by a new class permutation per batch.
Exhausted class pools reshuffle. Final batch order shuffles. Pool insertion
order follows the manifest and the RNG is Python Random(seed), never target.
"""
    domains = sorted(set(map(int, dataset.domains)))
    require(len(domains) == 2, "Training sampler requires two source rooms")
    rng, batches = random.Random(int(seed)), []
    for _epoch in range(epochs):
        pools = {d: {c: [i for i in range(len(dataset)) if dataset.domains[i] == d and dataset.labels[i] == c]
                     for c in range(9)} for d in domains}
        require(all(pool for classes in pools.values() for pool in classes.values()), "Missing source domain/class")
        positions = {d: {c: 0 for c in range(9)} for d in domains}
        for classes in pools.values():
            for pool in classes.values():
                rng.shuffle(pool)
        for _step in range(updates_per_epoch):
            batch = []
            for d in domains:
                order = list(range(9))
                rng.shuffle(order)
                for draw in range(16):
                    c = order[draw % 9]
                    if positions[d][c] == len(pools[d][c]):
                        rng.shuffle(pools[d][c])
                        positions[d][c] = 0
                    batch.append(pools[d][c][positions[d][c]])
                    positions[d][c] += 1
            rng.shuffle(batch)
            batches.append(batch)
    return np.asarray(batches, dtype=np.int64)


def schedule_hash(dataset, schedule):
    return hashlib.sha256(np.asarray(dataset.numeric_ids[schedule], dtype="<i8").tobytes()).hexdigest()
