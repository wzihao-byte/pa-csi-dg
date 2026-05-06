from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import BatchSampler, Dataset


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def ensure_time_feature_layout(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3:
        return array.astype(np.float32, copy=False)
    if array.ndim == 4:
        n, time_steps = array.shape[0], array.shape[1]
        return array.reshape(n, time_steps, -1).astype(np.float32, copy=False)
    if array.ndim == 2:
        return array[:, None, :].astype(np.float32, copy=False)
    raise ValueError(f"Expected 2D, 3D, or 4D CSI arrays, received shape {array.shape}.")


def unwrap_phase_array(array: np.ndarray, axis: int) -> np.ndarray:
    return np.unwrap(array, axis=axis)


class PaCsiArrayDataset(Dataset):
    def __init__(
        self,
        amplitude: np.ndarray,
        phase: np.ndarray,
        labels: np.ndarray,
        env_ids: np.ndarray,
        env_names: Sequence[str],
        sample_ids: np.ndarray,
    ) -> None:
        self.amplitude = ensure_time_feature_layout(amplitude)
        self.phase = ensure_time_feature_layout(phase)
        self.labels = labels.astype(np.int64, copy=False)
        self.env_ids = env_ids.astype(np.int64, copy=False)
        self.env_names = list(env_names)
        self.sample_ids = sample_ids.astype(np.int64, copy=False)

        if len(self.amplitude) != len(self.phase) or len(self.phase) != len(self.labels):
            raise ValueError("Amplitude, phase, and label arrays must have the same sample count.")
        if len(self.env_ids) != len(self.labels) or len(self.sample_ids) != len(self.labels):
            raise ValueError("Environment ids and sample ids must match the label array length.")
        if self.amplitude.shape[1] != self.phase.shape[1]:
            raise ValueError("Amplitude and phase inputs must share the same time dimension.")
        if self.amplitude.shape[2] != self.phase.shape[2]:
            raise ValueError("Amplitude and phase inputs must share the same feature width.")

        self.num_classes = int(np.unique(self.labels).size)
        self.input_dim = int(self.amplitude.shape[-1])
        self.time_steps = int(self.amplitude.shape[1])

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        amp = torch.from_numpy(self.amplitude[index]).float()
        phase = torch.from_numpy(self.phase[index]).float()
        label = torch.tensor(self.labels[index], dtype=torch.long)
        env_id = torch.tensor(self.env_ids[index], dtype=torch.long)
        sample_id = torch.tensor(self.sample_ids[index], dtype=torch.long)
        return amp, phase, label, env_id, sample_id

    def env_name(self, env_id: int) -> str:
        return self.env_names[int(env_id)]

    def indices_by_env(self, indices: Sequence[int]) -> Dict[int, List[int]]:
        grouped: Dict[int, List[int]] = defaultdict(list)
        for index in indices:
            grouped[int(self.env_ids[index])].append(int(index))
        return dict(grouped)

    def label_counts(self, indices: Sequence[int]) -> Dict[int, int]:
        counter = Counter(int(self.labels[index]) for index in indices)
        return dict(sorted(counter.items()))

    def env_counts(self, indices: Sequence[int]) -> Dict[str, int]:
        counter = Counter(self.env_name(int(self.env_ids[index])) for index in indices)
        return dict(sorted(counter.items()))


def load_env_npy_triplets(data_config: Mapping[str, Any], base_dir: Path) -> PaCsiArrayDataset:
    env_files = data_config.get("env_files", {})
    if not env_files:
        raise ValueError("data.env_files must map environment names to amp/phase/label files.")

    amplitude_parts: List[np.ndarray] = []
    phase_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    env_parts: List[np.ndarray] = []
    sample_parts: List[np.ndarray] = []
    env_names = list(env_files.keys())
    running_sample_id = 0
    unwrap_phase = bool(data_config.get("unwrap_phase", True))
    phase_unwrap_axis = int(data_config.get("phase_unwrap_axis", -1))

    for env_id, env_name in enumerate(env_names):
        file_group = env_files[env_name]
        amp_path = resolve_path(base_dir, file_group["amp"])
        phase_path = resolve_path(base_dir, file_group["phase"])
        label_path = resolve_path(base_dir, file_group["label"])

        amp = np.load(amp_path, allow_pickle=True)
        phase = np.load(phase_path, allow_pickle=True)
        labels = np.load(label_path, allow_pickle=True).astype(np.int64, copy=False)

        if unwrap_phase:
            phase = unwrap_phase_array(np.asarray(phase), axis=phase_unwrap_axis)

        amp = ensure_time_feature_layout(np.asarray(amp))
        phase = ensure_time_feature_layout(np.asarray(phase))

        if len(amp) != len(phase) or len(phase) != len(labels):
            raise ValueError(
                f"Mismatched sample counts for environment {env_name}: "
                f"amp={len(amp)} phase={len(phase)} labels={len(labels)}."
            )

        sample_ids = np.arange(running_sample_id, running_sample_id + len(labels), dtype=np.int64)
        running_sample_id += len(labels)

        amplitude_parts.append(amp)
        phase_parts.append(phase)
        label_parts.append(labels)
        env_parts.append(np.full(len(labels), env_id, dtype=np.int64))
        sample_parts.append(sample_ids)

    amplitude = np.concatenate(amplitude_parts, axis=0)
    phase = np.concatenate(phase_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    env_ids = np.concatenate(env_parts, axis=0)
    sample_ids = np.concatenate(sample_parts, axis=0)
    return PaCsiArrayDataset(amplitude, phase, labels, env_ids, env_names, sample_ids)


def load_dataset(data_config: Mapping[str, Any], base_dir: Path) -> PaCsiArrayDataset:
    data_format = data_config.get("format", "env_npy_triplets")
    if data_format != "env_npy_triplets":
        raise ValueError(
            f"Unsupported data format '{data_format}'. "
            "This DG path currently supports only 'env_npy_triplets'."
        )
    return load_env_npy_triplets(data_config, base_dir)


def safe_stratified_split(
    indices: np.ndarray,
    labels: np.ndarray,
    test_ratio: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if test_ratio <= 0.0 or len(indices) < 2:
        return indices.copy(), np.empty(0, dtype=np.int64)

    unique_labels, counts = np.unique(labels, return_counts=True)
    requested_test = max(1, int(round(len(indices) * test_ratio)))
    can_stratify = unique_labels.size > 1 and np.all(counts >= 2) and requested_test >= unique_labels.size

    if can_stratify:
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        train_pos, test_pos = next(splitter.split(np.zeros(len(indices)), labels))
    else:
        rng = np.random.default_rng(seed)
        shuffled = indices.copy()
        rng.shuffle(shuffled)
        split_at = max(1, len(indices) - requested_test)
        train_part, test_part = shuffled[:split_at], shuffled[split_at:]
        return np.sort(train_part), np.sort(test_part)

    return np.sort(indices[train_pos]), np.sort(indices[test_pos])


def build_split_indices(
    dataset: PaCsiArrayDataset,
    mode: str,
    seed: int,
    val_ratio: float,
    test_ratio: float = 0.2,
    target_env: Optional[str] = None,
) -> Dict[str, Any]:
    all_indices = np.arange(len(dataset), dtype=np.int64)

    if mode == "random_split":
        train_val_indices, test_indices = safe_stratified_split(
            all_indices,
            dataset.labels,
            test_ratio=test_ratio,
            seed=seed,
        )
        train_indices, val_indices = safe_stratified_split(
            train_val_indices,
            dataset.labels[train_val_indices],
            test_ratio=val_ratio,
            seed=seed + 1,
        )
        split = {
            "mode": mode,
            "target_env": None,
            "source_envs": list(dataset.env_names),
            "train_indices": train_indices.tolist(),
            "val_indices": val_indices.tolist(),
            "test_indices": test_indices.tolist(),
        }
        validate_split_integrity(dataset, split)
        return split

    if mode != "dg_loeo":
        raise ValueError(f"Unsupported split mode '{mode}'.")
    if target_env is None:
        raise ValueError("target_env must be set when mode='dg_loeo'.")
    if target_env not in dataset.env_names:
        raise ValueError(f"Unknown target environment '{target_env}'. Available: {dataset.env_names}")

    target_env_id = dataset.env_names.index(target_env)
    test_indices = all_indices[dataset.env_ids == target_env_id]

    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []
    source_envs: List[str] = []

    for env_id, env_name in enumerate(dataset.env_names):
        if env_id == target_env_id:
            continue
        source_envs.append(env_name)
        env_indices = all_indices[dataset.env_ids == env_id]
        env_train, env_val = safe_stratified_split(
            env_indices,
            dataset.labels[env_indices],
            test_ratio=val_ratio,
            seed=seed + env_id + 1,
        )
        train_parts.append(env_train)
        val_parts.append(env_val)

    train_indices = np.sort(np.concatenate(train_parts, axis=0)) if train_parts else np.empty(0, dtype=np.int64)
    val_indices = np.sort(np.concatenate(val_parts, axis=0)) if val_parts else np.empty(0, dtype=np.int64)

    split = {
        "mode": mode,
        "target_env": target_env,
        "source_envs": source_envs,
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
        "test_indices": test_indices.tolist(),
    }
    validate_split_integrity(dataset, split)
    return split


def validate_split_integrity(dataset: PaCsiArrayDataset, split: Mapping[str, Any]) -> None:
    train_set = set(int(index) for index in split["train_indices"])
    val_set = set(int(index) for index in split["val_indices"])
    test_set = set(int(index) for index in split["test_indices"])

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Train/val/test indices overlap.")

    target_env = split.get("target_env")
    if split["mode"] == "dg_loeo" and target_env is not None:
        target_env_id = dataset.env_names.index(target_env)
        if any(int(dataset.env_ids[index]) == target_env_id for index in train_set | val_set):
            raise ValueError("Target environment leaked into train or validation.")
        if any(int(dataset.env_ids[index]) != target_env_id for index in test_set):
            raise ValueError("DG test split contains non-target environments.")


def build_split_manifest(dataset: PaCsiArrayDataset, split: Mapping[str, Any]) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "mode": split["mode"],
        "target_env": split.get("target_env"),
        "source_envs": list(split.get("source_envs", [])),
        "dataset_size": len(dataset),
    }
    for subset_name in ("train_indices", "val_indices", "test_indices"):
        indices = split[subset_name]
        manifest[subset_name.replace("_indices", "")] = {
            "count": len(indices),
            "env_counts": dataset.env_counts(indices),
            "label_counts": dataset.label_counts(indices),
        }
    return manifest


class DomainBalancedBatchSampler(BatchSampler):
    def __init__(self, domain_to_indices: Mapping[int, Sequence[int]], batch_size: int) -> None:
        self.domain_to_indices = {int(domain): [int(index) for index in indices] for domain, indices in domain_to_indices.items()}
        self.domains = sorted(self.domain_to_indices.keys())
        if not self.domains:
            raise ValueError("DomainBalancedBatchSampler requires at least one source domain.")

        self.batch_size = int(batch_size)
        self.samples_per_domain = max(1, self.batch_size // len(self.domains))
        self.domain_batch_sizes = [self.samples_per_domain for _ in self.domains]
        remainder = self.batch_size - self.samples_per_domain * len(self.domains)
        for offset in range(remainder):
            self.domain_batch_sizes[offset] += 1

        max_domain_size = max(len(indices) for indices in self.domain_to_indices.values())
        self.num_batches = max(1, math.ceil(max_domain_size / self.samples_per_domain))

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterable[List[int]]:
        shuffled = {
            domain: random.sample(indices, len(indices))
            for domain, indices in self.domain_to_indices.items()
        }

        for batch_id in range(self.num_batches):
            batch: List[int] = []
            for domain, take_count in zip(self.domains, self.domain_batch_sizes):
                domain_indices = shuffled[domain]
                start = batch_id * self.samples_per_domain
                end = start + take_count
                selected = domain_indices[start:end]
                if len(selected) < take_count:
                    if not domain_indices:
                        raise ValueError(f"Domain {domain} has no indices to sample from.")
                    selected = list(selected) + random.choices(domain_indices, k=take_count - len(selected))
                batch.extend(selected)
            random.shuffle(batch)
            yield batch


class DomainClassBalancedBatchSampler(BatchSampler):
    def __init__(
        self,
        domain_to_indices: Mapping[int, Sequence[int]],
        labels: np.ndarray,
        batch_size: int,
    ) -> None:
        self.labels = labels
        self.domain_to_indices = {int(domain): [int(index) for index in indices] for domain, indices in domain_to_indices.items()}
        self.domains = sorted(self.domain_to_indices.keys())
        if not self.domains:
            raise ValueError("DomainClassBalancedBatchSampler requires at least one source domain.")

        self.classes = sorted({int(labels[index]) for indices in self.domain_to_indices.values() for index in indices})
        if not self.classes:
            raise ValueError("DomainClassBalancedBatchSampler requires at least one class.")

        self.batch_size = int(batch_size)
        self.samples_per_domain = max(1, self.batch_size // len(self.domains))
        self.domain_batch_sizes = [self.samples_per_domain for _ in self.domains]
        remainder = self.batch_size - self.samples_per_domain * len(self.domains)
        for offset in range(remainder):
            self.domain_batch_sizes[offset] += 1

        max_domain_size = max(len(indices) for indices in self.domain_to_indices.values())
        self.num_batches = max(1, math.ceil(max_domain_size / self.samples_per_domain))

    def __len__(self) -> int:
        return self.num_batches

    def _build_class_pools(self) -> Dict[int, Dict[int, List[int]]]:
        pools: Dict[int, Dict[int, List[int]]] = {}
        for domain, indices in self.domain_to_indices.items():
            class_to_indices: Dict[int, List[int]] = defaultdict(list)
            for index in indices:
                class_to_indices[int(self.labels[index])].append(int(index))
            for class_id in class_to_indices:
                random.shuffle(class_to_indices[class_id])
            pools[domain] = dict(class_to_indices)
        return pools

    def __iter__(self) -> Iterable[List[int]]:
        pools = self._build_class_pools()
        positions = {
            domain: {class_id: 0 for class_id in pools[domain].keys()}
            for domain in self.domains
        }

        def take_from_class(domain: int, class_id: int) -> int:
            class_pool = pools[domain][class_id]
            if not class_pool:
                raise ValueError(f"Domain {domain} class {class_id} has no indices to sample from.")
            position = positions[domain][class_id]
            if position >= len(class_pool):
                random.shuffle(class_pool)
                position = 0
            index = class_pool[position]
            positions[domain][class_id] = position + 1
            return index

        for _ in range(self.num_batches):
            batch: List[int] = []
            for domain, take_count in zip(self.domains, self.domain_batch_sizes):
                available_classes = [class_id for class_id, class_pool in pools[domain].items() if class_pool]
                if not available_classes:
                    raise ValueError(f"Domain {domain} has no class pools to sample from.")

                class_order = available_classes.copy()
                random.shuffle(class_order)
                domain_batch: List[int] = []
                cursor = 0
                while len(domain_batch) < take_count:
                    class_id = class_order[cursor % len(class_order)]
                    domain_batch.append(take_from_class(domain, class_id))
                    cursor += 1
                batch.extend(domain_batch)
            random.shuffle(batch)
            yield batch
