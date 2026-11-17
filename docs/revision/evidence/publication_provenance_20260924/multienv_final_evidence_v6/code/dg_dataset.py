from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import BatchSampler, Dataset


PA_CSI_ENV_SAMPLE_COUNT = 3000
PA_CSI_GROUP_SIZE = 300
PA_CSI_GROUPS_PER_ENV = 10
PA_CSI_GROUP_LABEL_COUNTS = {0: 120, 1: 40, 2: 40, 3: 40, 4: 40, 5: 20}
PA_CSI_F0_SPLIT_SEED = 20260714
PA_CSI_F0_VALIDATION_BLOCKS: Dict[str, tuple[str, str]] = {
    "E1": ("B01", "B07"),
    "E2": ("B00", "B06"),
    "E3": ("B00", "B04"),
}


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


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_optional_id(value: Any) -> Optional[str]:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return None
    return text


def _validate_and_cast_integer_labels(labels: Any, context: str) -> np.ndarray:
    array = np.asarray(labels)
    if np.iscomplexobj(array) or not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
        or np.issubdtype(array.dtype, np.bool_)
    ):
        raise ValueError(f"{context} must be a finite numeric array of integer-valued labels.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{context} contains NaN or infinite labels.")
    if not np.all(array == np.floor(array)):
        raise ValueError(f"{context} contains non-integer label values.")
    int64_info = np.iinfo(np.int64)
    if array.size and (np.any(array < int64_info.min) or np.any(array > int64_info.max)):
        raise ValueError(f"{context} contains labels outside the int64 range.")
    return array.astype(np.int64, copy=False)


def _namespace_id(env_name: str, value: Any, kind: str) -> str:
    identifier = _normalise_optional_id(value)
    if identifier is None:
        raise ValueError(f"{kind} for environment {env_name} cannot be null or empty.")
    if ":" in identifier:
        prefix = identifier.split(":", 1)[0]
        if prefix != env_name:
            raise ValueError(
                f"{kind} '{identifier}' is namespaced to '{prefix}', not environment '{env_name}'."
            )
        return identifier
    return f"{env_name}:{identifier}"


def _build_and_validate_group_ids(labels: np.ndarray, env_name: str) -> np.ndarray:
    if labels.ndim != 1:
        raise ValueError(f"Labels for environment {env_name} must be one-dimensional, got {labels.shape}.")
    if len(labels) != PA_CSI_ENV_SAMPLE_COUNT:
        raise ValueError(
            f"Environment {env_name} must contain exactly {PA_CSI_ENV_SAMPLE_COUNT} samples "
            f"({PA_CSI_GROUPS_PER_ENV} contiguous groups x {PA_CSI_GROUP_SIZE}); got {len(labels)}."
        )

    group_ids = np.empty(len(labels), dtype=object)
    for block_index in range(PA_CSI_GROUPS_PER_ENV):
        start = block_index * PA_CSI_GROUP_SIZE
        end = start + PA_CSI_GROUP_SIZE
        block_labels = labels[start:end]
        counts = Counter(int(label) for label in block_labels.tolist())
        actual_counts = dict(sorted(counts.items()))
        if actual_counts != PA_CSI_GROUP_LABEL_COUNTS:
            raise ValueError(
                f"Environment {env_name} block B{block_index:02d} label counts must be "
                f"{PA_CSI_GROUP_LABEL_COUNTS}; got {actual_counts}."
            )
        group_ids[start:end] = f"{env_name}:B{block_index:02d}"
    return group_ids


def _ordered_block_sidecar_values(values: Mapping[str, Any], env_name: str) -> List[Any]:
    ordered: List[Any] = []
    for block_index in range(PA_CSI_GROUPS_PER_ENV):
        candidates: tuple[Any, ...] = (
            f"{env_name}:B{block_index:02d}",
            f"B{block_index:02d}",
            str(block_index),
            block_index,
        )
        matching_key = next((key for key in candidates if key in values), None)
        if matching_key is None:
            raise ValueError(
                f"Subject sidecar for {env_name} is missing block B{block_index:02d}."
            )
        value = values[matching_key]
        if isinstance(value, Mapping):
            if "subject_id" not in value:
                raise ValueError(
                    f"Subject sidecar entry {matching_key!r} for {env_name} must contain subject_id."
                )
            value = value["subject_id"]
        ordered.append(value)
    return ordered


def _read_subject_sidecar(path: Path, env_name: str) -> tuple[List[Any], Dict[str, Any]]:
    suffix = path.suffix.lower()
    metadata: Dict[str, Any] = {}

    if suffix == ".npy":
        values = np.load(path, allow_pickle=True).reshape(-1).tolist()
    elif suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            values = payload
        elif isinstance(payload, Mapping):
            metadata = {
                key: payload[key]
                for key in ("env_name", "data_file_sha256", "file_sha256")
                if key in payload
            }
            declared_env = payload.get("env_name")
            if declared_env is not None and str(declared_env) != env_name:
                raise ValueError(
                    f"Subject sidecar {path} declares env_name={declared_env!r}, expected {env_name!r}."
                )
            if "subject_ids" in payload:
                values = list(payload["subject_ids"])
            elif "subjects" in payload:
                values = list(payload["subjects"])
            elif "block_subject_ids" in payload:
                block_values = payload["block_subject_ids"]
                if isinstance(block_values, Mapping):
                    values = _ordered_block_sidecar_values(block_values, env_name)
                else:
                    values = list(block_values)
            else:
                reserved = {"env_name", "data_file_sha256", "file_sha256"}
                block_values = {key: value for key, value in payload.items() if key not in reserved}
                values = _ordered_block_sidecar_values(block_values, env_name)
        else:
            raise ValueError(f"Unsupported JSON subject sidecar payload in {path}.")
    elif suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows or "subject_id" not in rows[0]:
            raise ValueError(f"CSV subject sidecar {path} must contain a subject_id column.")
        if "group_id" in rows[0] and len(rows) == PA_CSI_GROUPS_PER_ENV:
            values = _ordered_block_sidecar_values(
                {str(row["group_id"]): row["subject_id"] for row in rows},
                env_name,
            )
        else:
            values = [row["subject_id"] for row in rows]
    elif suffix in {".txt", ".tsv"}:
        with path.open("r", encoding="utf-8") as handle:
            values = [line.strip().split("\t")[-1] for line in handle if line.strip()]
    else:
        raise ValueError(
            f"Unsupported subject sidecar format '{path.suffix}'. Use .json, .npy, .csv, .txt, or .tsv."
        )

    return list(values), metadata


def _validate_sidecar_hash_binding(
    metadata: Mapping[str, Any],
    actual_hashes: Mapping[str, str],
    env_name: str,
) -> str:
    expected = metadata.get("data_file_sha256", metadata.get("file_sha256"))
    if expected is None:
        return "unbound"
    if not isinstance(expected, Mapping):
        raise ValueError(f"Subject sidecar hashes for {env_name} must be a mapping.")

    aliases = {
        "amp": ("amp", "amplitude"),
        "phase": ("phase", "angle"),
        "label": ("label", "labels"),
    }
    missing: List[str] = []
    for canonical_name, keys in aliases.items():
        declared = next((expected[key] for key in keys if key in expected), None)
        if declared is None:
            missing.append(canonical_name)
            continue
        if str(declared).lower() != actual_hashes[canonical_name].lower():
            raise ValueError(
                f"Subject sidecar hash mismatch for {env_name} {canonical_name}: "
                f"declared={declared}, actual={actual_hashes[canonical_name]}."
            )
    return "bound" if not missing else "partially_bound"


def _expand_and_validate_subject_ids(
    values: Sequence[Any],
    env_name: str,
) -> tuple[np.ndarray, List[str]]:
    if len(values) == PA_CSI_GROUPS_PER_ENV:
        block_values = list(values)
    elif len(values) == PA_CSI_ENV_SAMPLE_COUNT:
        block_values = []
        for block_index in range(PA_CSI_GROUPS_PER_ENV):
            start = block_index * PA_CSI_GROUP_SIZE
            end = start + PA_CSI_GROUP_SIZE
            identifiers = {
                _normalise_optional_id(value)
                for value in values[start:end]
            }
            if None in identifiers or len(identifiers) != 1:
                raise ValueError(
                    f"Subject sidecar for {env_name} block B{block_index:02d} must contain "
                    "one non-null subject id for all 300 samples."
                )
            block_values.append(next(iter(identifiers)))
    else:
        raise ValueError(
            f"Subject sidecar for {env_name} must contain either {PA_CSI_GROUPS_PER_ENV} block ids "
            f"or {PA_CSI_ENV_SAMPLE_COUNT} per-sample ids; got {len(values)}."
        )

    block_subject_ids = [
        _namespace_id(env_name, value, "subject_id")
        for value in block_values
    ]
    if len(set(block_subject_ids)) != PA_CSI_GROUPS_PER_ENV:
        raise ValueError(
            f"Subject sidecar for {env_name} must map its 10 blocks one-to-one to 10 subjects; "
            f"got {len(set(block_subject_ids))} unique ids."
        )
    expanded = np.repeat(np.asarray(block_subject_ids, dtype=object), PA_CSI_GROUP_SIZE)
    return expanded, block_subject_ids


def _hash_indices(indices: Sequence[int]) -> str:
    array = np.asarray([int(index) for index in indices], dtype="<i8")
    return hashlib.sha256(array.tobytes()).hexdigest()


class PaCsiArrayDataset(Dataset):
    def __init__(
        self,
        amplitude: np.ndarray,
        phase: np.ndarray,
        labels: np.ndarray,
        env_ids: np.ndarray,
        env_names: Sequence[str],
        sample_ids: np.ndarray,
        group_ids: Optional[Sequence[Any]] = None,
        subject_ids: Optional[Sequence[Any]] = None,
        provenance: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.amplitude = ensure_time_feature_layout(amplitude)
        self.phase = ensure_time_feature_layout(phase)
        self.labels = _validate_and_cast_integer_labels(labels, "Dataset labels")
        self.env_ids = env_ids.astype(np.int64, copy=False)
        self.env_names = list(env_names)
        self.sample_ids = sample_ids.astype(np.int64, copy=False)
        self.group_ids = np.asarray(
            [None] * len(self.labels) if group_ids is None else list(group_ids),
            dtype=object,
        )
        self.subject_ids = np.asarray(
            [None] * len(self.labels) if subject_ids is None else list(subject_ids),
            dtype=object,
        )
        self.provenance = dict(provenance or {})

        if len(self.amplitude) != len(self.phase) or len(self.phase) != len(self.labels):
            raise ValueError("Amplitude, phase, and label arrays must have the same sample count.")
        if len(self.env_ids) != len(self.labels) or len(self.sample_ids) != len(self.labels):
            raise ValueError("Environment ids and sample ids must match the label array length.")
        if len(self.group_ids) != len(self.labels) or len(self.subject_ids) != len(self.labels):
            raise ValueError("Group ids and subject ids must match the label array length.")
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

    @staticmethod
    def _identifier_counts(values: np.ndarray, indices: Sequence[int]) -> Dict[str, int]:
        counter = Counter(
            identifier
            for index in indices
            if (identifier := _normalise_optional_id(values[int(index)])) is not None
        )
        return dict(sorted(counter.items()))

    def group_counts(self, indices: Sequence[int]) -> Dict[str, int]:
        return self._identifier_counts(self.group_ids, indices)

    def subject_counts(self, indices: Sequence[int]) -> Dict[str, int]:
        return self._identifier_counts(self.subject_ids, indices)

    def has_complete_group_ids(self, indices: Optional[Sequence[int]] = None) -> bool:
        selected = range(len(self)) if indices is None else indices
        return all(_normalise_optional_id(self.group_ids[int(index)]) is not None for index in selected)

    def has_complete_subject_ids(self, indices: Optional[Sequence[int]] = None) -> bool:
        selected = range(len(self)) if indices is None else indices
        return all(_normalise_optional_id(self.subject_ids[int(index)]) is not None for index in selected)


def load_env_npy_triplets(data_config: Mapping[str, Any], base_dir: Path) -> PaCsiArrayDataset:
    env_files = data_config.get("env_files", {})
    if not env_files:
        raise ValueError("data.env_files must map environment names to amp/phase/label files.")

    amplitude_parts: List[np.ndarray] = []
    phase_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    env_parts: List[np.ndarray] = []
    sample_parts: List[np.ndarray] = []
    group_parts: List[np.ndarray] = []
    subject_parts: List[np.ndarray] = []
    env_names = list(env_files.keys())
    running_sample_id = 0
    unwrap_phase = bool(data_config.get("unwrap_phase", True))
    phase_unwrap_axis = int(data_config.get("phase_unwrap_axis", -1))
    top_level_sidecars = data_config.get("subject_sidecars", {})
    if top_level_sidecars is None:
        top_level_sidecars = {}
    if not isinstance(top_level_sidecars, Mapping):
        raise ValueError("data.subject_sidecars must be a mapping from environment name to sidecar path.")
    has_any_sidecar = any(
        env_files[env_name].get("subject_sidecar", top_level_sidecars.get(env_name)) is not None
        for env_name in env_names
    )
    strict_group_protocol = bool(data_config.get("strict_group_protocol", False)) or has_any_sidecar

    data_file_records: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {}
    sidecar_status: Dict[str, Dict[str, Any]] = {}

    for env_id, env_name in enumerate(env_names):
        file_group = env_files[env_name]
        amp_path = resolve_path(base_dir, file_group["amp"])
        phase_path = resolve_path(base_dir, file_group["phase"])
        label_path = resolve_path(base_dir, file_group["label"])

        actual_hashes: Dict[str, Optional[str]] = {
            "amp": sha256_file(amp_path) if strict_group_protocol else None,
            "phase": sha256_file(phase_path) if strict_group_protocol else None,
            "label": sha256_file(label_path) if strict_group_protocol else None,
        }
        data_file_records[str(env_name)] = {
            "amp": {"path": str(amp_path), "sha256": actual_hashes["amp"]},
            "phase": {"path": str(phase_path), "sha256": actual_hashes["phase"]},
            "label": {"path": str(label_path), "sha256": actual_hashes["label"]},
        }

        amp = np.load(amp_path, allow_pickle=True)
        phase = np.load(phase_path, allow_pickle=True)
        labels = _validate_and_cast_integer_labels(
            np.load(label_path, allow_pickle=True),
            f"Labels for environment {env_name}",
        )

        if unwrap_phase:
            phase = unwrap_phase_array(np.asarray(phase), axis=phase_unwrap_axis)

        amp = ensure_time_feature_layout(np.asarray(amp))
        phase = ensure_time_feature_layout(np.asarray(phase))

        if len(amp) != len(phase) or len(phase) != len(labels):
            raise ValueError(
                f"Mismatched sample counts for environment {env_name}: "
                f"amp={len(amp)} phase={len(phase)} labels={len(labels)}."
            )

        group_ids = (
            _build_and_validate_group_ids(labels, str(env_name))
            if strict_group_protocol
            else np.full(len(labels), None, dtype=object)
        )

        sidecar_spec = file_group.get("subject_sidecar", top_level_sidecars.get(env_name))
        if sidecar_spec is None:
            subject_ids = np.full(len(labels), None, dtype=object)
            sidecar_status[str(env_name)] = {
                "status": (
                    "absent_inferred_groups"
                    if strict_group_protocol
                    else "absent_group_protocol_disabled"
                ),
                "present": False,
                "verified": False,
                "hash_binding": "absent",
                "path": None,
                "sha256": None,
                "subject_count": 0,
            }
        else:
            if isinstance(sidecar_spec, Mapping):
                if "path" not in sidecar_spec:
                    raise ValueError(
                        f"Subject sidecar mapping for {env_name} must contain a 'path' field."
                    )
                sidecar_value = sidecar_spec["path"]
            else:
                sidecar_value = sidecar_spec
            sidecar_path = resolve_path(base_dir, str(sidecar_value))
            sidecar_values, sidecar_metadata = _read_subject_sidecar(sidecar_path, str(env_name))
            hash_binding = _validate_sidecar_hash_binding(
                sidecar_metadata,
                {
                    file_kind: str(file_hash)
                    for file_kind, file_hash in actual_hashes.items()
                    if file_hash is not None
                },
                str(env_name),
            )
            candidate_subject_ids, block_subject_ids = _expand_and_validate_subject_ids(
                sidecar_values,
                str(env_name),
            )
            sidecar_verified = hash_binding == "bound"
            subject_ids = (
                candidate_subject_ids
                if sidecar_verified
                else np.full(len(labels), None, dtype=object)
            )
            sidecar_status[str(env_name)] = {
                "status": f"validated_{hash_binding}",
                "present": True,
                "verified": sidecar_verified,
                "hash_binding": hash_binding,
                "path": str(sidecar_path),
                "sha256": sha256_file(sidecar_path),
                "subject_count": len(block_subject_ids) if sidecar_verified else 0,
                "subject_ids": block_subject_ids if sidecar_verified else [],
                "candidate_subject_count_unverified": (
                    0 if sidecar_verified else len(block_subject_ids)
                ),
                "candidate_subject_ids_unverified": (
                    [] if sidecar_verified else block_subject_ids
                ),
            }

        sample_ids = np.arange(running_sample_id, running_sample_id + len(labels), dtype=np.int64)
        running_sample_id += len(labels)

        amplitude_parts.append(amp)
        phase_parts.append(phase)
        label_parts.append(labels)
        env_parts.append(np.full(len(labels), env_id, dtype=np.int64))
        sample_parts.append(sample_ids)
        group_parts.append(group_ids)
        subject_parts.append(subject_ids)

    amplitude = np.concatenate(amplitude_parts, axis=0)
    phase = np.concatenate(phase_parts, axis=0)
    labels = np.concatenate(label_parts, axis=0)
    env_ids = np.concatenate(env_parts, axis=0)
    sample_ids = np.concatenate(sample_parts, axis=0)
    group_ids = np.concatenate(group_parts, axis=0)
    subject_ids = np.concatenate(subject_parts, axis=0)
    provenance = {
        "data_files": data_file_records,
        "sidecar_status": sidecar_status,
        "subject_sidecars_complete": all(
            bool(status.get("verified")) for status in sidecar_status.values()
        ),
        "group_protocol": {
            "enabled": strict_group_protocol,
            "source": "contiguous_blocks" if strict_group_protocol else None,
            "samples_per_environment": PA_CSI_ENV_SAMPLE_COUNT,
            "groups_per_environment": PA_CSI_GROUPS_PER_ENV,
            "group_size": PA_CSI_GROUP_SIZE,
            "group_label_counts": dict(PA_CSI_GROUP_LABEL_COUNTS),
        },
    }
    return PaCsiArrayDataset(
        amplitude,
        phase,
        labels,
        env_ids,
        env_names,
        sample_ids,
        group_ids=group_ids,
        subject_ids=subject_ids,
        provenance=provenance,
    )


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


def _normalise_validation_group_id(env_name: str, value: Any) -> str:
    if isinstance(value, (int, np.integer)):
        block_name = f"B{int(value):02d}"
    else:
        block_name = str(value).strip()
        if block_name.isdigit():
            block_name = f"B{int(block_name):02d}"
    if ":" in block_name:
        prefix = block_name.split(":", 1)[0]
        if prefix != env_name:
            raise ValueError(
                f"Validation group '{block_name}' does not belong to environment '{env_name}'."
            )
        return block_name
    return f"{env_name}:{block_name}"


def _resolve_f0_split_seed(split_seed: Optional[int]) -> int:
    resolved = PA_CSI_F0_SPLIT_SEED if split_seed is None else int(split_seed)
    if resolved != PA_CSI_F0_SPLIT_SEED:
        raise ValueError(
            f"Grouped F0 split_seed is protocol-locked to {PA_CSI_F0_SPLIT_SEED}; got {resolved}. "
            "The seed is provenance only and never triggers resampling."
        )
    return resolved


def _validate_group_protocol_dataset(dataset: PaCsiArrayDataset, require_subjects: bool) -> None:
    if not dataset.has_complete_group_ids():
        raise ValueError("Grouped holdout requires a non-null group_id for every sample.")
    if require_subjects:
        if not dataset.has_complete_subject_ids():
            raise ValueError(
                "subject_grouped_holdout requires validated subject sidecars for every environment; "
                "use inferred_group_block_holdout when subject ids are unavailable."
            )
        status_by_env = dataset.provenance.get("sidecar_status", {})
        if not status_by_env or any(
            not bool(status_by_env.get(env_name, {}).get("verified"))
            for env_name in dataset.env_names
        ):
            raise ValueError(
                "subject_grouped_holdout requires sidecar_status.verified=true for every environment."
            )

    for env_id, env_name in enumerate(dataset.env_names):
        env_indices = np.where(dataset.env_ids == env_id)[0]
        if len(env_indices) != PA_CSI_ENV_SAMPLE_COUNT:
            raise ValueError(
                f"Environment {env_name} must contain {PA_CSI_ENV_SAMPLE_COUNT} samples for grouped holdout; "
                f"got {len(env_indices)}."
            )
        env_groups = dataset.group_counts(env_indices)
        if len(env_groups) != PA_CSI_GROUPS_PER_ENV:
            raise ValueError(
                f"Environment {env_name} must contain {PA_CSI_GROUPS_PER_ENV} groups; got {len(env_groups)}."
            )
        for group_id, count in env_groups.items():
            if count != PA_CSI_GROUP_SIZE:
                raise ValueError(
                    f"Group {group_id} must contain {PA_CSI_GROUP_SIZE} samples; got {count}."
                )
            group_indices = [
                int(index)
                for index in env_indices
                if _normalise_optional_id(dataset.group_ids[int(index)]) == group_id
            ]
            if dataset.label_counts(group_indices) != PA_CSI_GROUP_LABEL_COUNTS:
                raise ValueError(
                    f"Group {group_id} label counts must be {PA_CSI_GROUP_LABEL_COUNTS}; "
                    f"got {dataset.label_counts(group_indices)}."
                )
            if require_subjects:
                subjects = dataset.subject_counts(group_indices)
                if len(subjects) != 1 or next(iter(subjects.values())) != PA_CSI_GROUP_SIZE:
                    raise ValueError(f"Group {group_id} must map one-to-one to one verified subject.")

        if require_subjects:
            env_subjects = dataset.subject_counts(env_indices)
            if len(env_subjects) != PA_CSI_GROUPS_PER_ENV:
                raise ValueError(
                    f"Environment {env_name} must contain 10 unique verified subjects; "
                    f"got {len(env_subjects)}."
                )


def _build_group_holdout_split(
    dataset: PaCsiArrayDataset,
    mode: str,
    target_env: str,
    inner_validation: str,
    validation_group_ids: Optional[Mapping[str, Sequence[Any]]],
    split_seed: Optional[int],
    target_not_loaded: bool = False,
) -> Dict[str, Any]:
    require_subjects = inner_validation == "subject_grouped_holdout"
    _validate_group_protocol_dataset(dataset, require_subjects=require_subjects)

    if target_not_loaded:
        if target_env in dataset.env_names:
            raise ValueError(
                f"Source-only grouped split requires outer target '{target_env}' to be absent "
                "from the loaded dataset."
            )
        target_env_id: Optional[int] = None
        source_envs = list(dataset.env_names)
    else:
        target_env_id = dataset.env_names.index(target_env)
        source_envs = [
            env_name
            for env_id, env_name in enumerate(dataset.env_names)
            if env_id != target_env_id
        ]
    table = PA_CSI_F0_VALIDATION_BLOCKS if validation_group_ids is None else validation_group_ids
    if not isinstance(table, Mapping):
        raise ValueError("validation_group_ids must map each source environment to two explicit groups.")

    resolved_validation_groups: Dict[str, List[str]] = {}
    val_group_set: set[str] = set()
    for env_name in source_envs:
        if env_name not in table:
            raise ValueError(
                f"Explicit F0 validation table has no entry for source environment '{env_name}'."
            )
        raw_group_ids = list(table[env_name])
        if len(raw_group_ids) != 2:
            raise ValueError(
                f"Source environment {env_name} must have exactly two explicit validation groups; "
                f"got {raw_group_ids}."
            )
        resolved = [_normalise_validation_group_id(env_name, value) for value in raw_group_ids]
        if len(set(resolved)) != 2:
            raise ValueError(f"Validation groups for {env_name} must be distinct; got {resolved}.")

        env_id = dataset.env_names.index(env_name)
        available = set(dataset.group_counts(np.where(dataset.env_ids == env_id)[0]).keys())
        missing = sorted(set(resolved) - available)
        if missing:
            raise ValueError(
                f"Explicit validation groups {missing} are unavailable in environment {env_name}; "
                f"available={sorted(available)}."
            )
        resolved_validation_groups[env_name] = resolved
        val_group_set.update(resolved)

    all_indices = np.arange(len(dataset), dtype=np.int64)
    if target_not_loaded:
        test_indices = np.empty(0, dtype=np.int64)
        source_mask = np.ones(len(dataset), dtype=bool)
    else:
        if target_env_id is None:
            raise AssertionError("target_env_id must be resolved for a loaded-target split.")
        test_indices = all_indices[dataset.env_ids == target_env_id]
        source_mask = dataset.env_ids != target_env_id
    val_mask = np.asarray(
        [
            bool(source_mask[index])
            and _normalise_optional_id(dataset.group_ids[index]) in val_group_set
            for index in all_indices
        ],
        dtype=bool,
    )
    train_mask = source_mask & ~val_mask
    train_indices = np.sort(all_indices[train_mask])
    val_indices = np.sort(all_indices[val_mask])

    split = {
        "mode": mode,
        "inner_validation": inner_validation,
        "target_env": target_env,
        "source_envs": source_envs,
        "train_envs": source_envs,
        "val_envs": source_envs,
        "group_fold": "F0" if validation_group_ids is None else "explicit",
        "split_seed": _resolve_f0_split_seed(split_seed),
        "validation_group_ids": resolved_validation_groups,
        "subject_sidecars_verified": require_subjects,
        "target_not_loaded": bool(target_not_loaded),
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
        "test_indices": np.sort(test_indices).tolist(),
    }
    validate_split_integrity(dataset, split)
    return split


def build_source_only_group_split(
    dataset: PaCsiArrayDataset,
    target_env: str,
    inner_validation: str = "inferred_group_block_holdout",
    validation_group_ids: Optional[Mapping[str, Sequence[Any]]] = None,
    split_seed: Optional[int] = PA_CSI_F0_SPLIT_SEED,
) -> Dict[str, Any]:
    """Build the locked F0 source train/val split without loading the outer target.

    The dataset must contain exactly the two canonical source environments. The returned
    test split is empty and ``target_not_loaded`` is true, so source-only screening cannot
    accidentally inspect target arrays or statistics.
    """
    target_env = str(target_env)
    if target_env not in PA_CSI_F0_VALIDATION_BLOCKS:
        raise ValueError(
            f"Unknown logical outer target '{target_env}'. "
            f"Expected one of {sorted(PA_CSI_F0_VALIDATION_BLOCKS)}."
        )
    expected_sources = set(PA_CSI_F0_VALIDATION_BLOCKS) - {target_env}
    loaded_sources = set(dataset.env_names)
    if loaded_sources != expected_sources or len(dataset.env_names) != 2:
        raise ValueError(
            f"Source-only split for outer target {target_env} must load exactly "
            f"{sorted(expected_sources)} and no target data; got {dataset.env_names}."
        )
    if inner_validation not in {"subject_grouped_holdout", "inferred_group_block_holdout"}:
        raise ValueError(
            "Source-only grouped split supports only subject_grouped_holdout or "
            "inferred_group_block_holdout."
        )
    return _build_group_holdout_split(
        dataset=dataset,
        mode="dg_loeo_source_only",
        target_env=target_env,
        inner_validation=inner_validation,
        validation_group_ids=validation_group_ids,
        split_seed=split_seed,
        target_not_loaded=True,
    )


def build_split_indices(
    dataset: PaCsiArrayDataset,
    mode: str,
    seed: int,
    val_ratio: float,
    test_ratio: float = 0.2,
    target_env: Optional[str] = None,
    inner_validation: str = "stratified_holdout",
    loso_validation_env: Optional[str] = None,
    validation_group_ids: Optional[Mapping[str, Sequence[Any]]] = None,
    split_seed: Optional[int] = None,
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

    source_envs = [
        env_name
        for env_id, env_name in enumerate(dataset.env_names)
        if env_id != target_env_id
    ]
    inner_validation = str(inner_validation or "stratified_holdout")

    if inner_validation in {"subject_grouped_holdout", "inferred_group_block_holdout"}:
        return _build_group_holdout_split(
            dataset=dataset,
            mode=mode,
            target_env=target_env,
            inner_validation=inner_validation,
            validation_group_ids=validation_group_ids,
            split_seed=split_seed,
        )

    if inner_validation == "loso":
        if len(source_envs) < 2:
            raise ValueError("LOSO inner validation requires at least two source environments.")
        if loso_validation_env is None:
            validation_env_name = source_envs[-1]
        else:
            validation_env_name = str(loso_validation_env)
            if validation_env_name not in source_envs:
                raise ValueError(
                    f"split.loso_validation_env='{validation_env_name}' is not a source environment "
                    f"for target '{target_env}'. Available sources: {source_envs}"
                )
        validation_env_id = dataset.env_names.index(validation_env_name)
        train_parts = [
            all_indices[dataset.env_ids == env_id]
            for env_id, env_name in enumerate(dataset.env_names)
            if env_id != target_env_id and env_id != validation_env_id
        ]
        train_indices = (
            np.sort(np.concatenate(train_parts, axis=0))
            if train_parts
            else np.empty(0, dtype=np.int64)
        )
        val_indices = np.sort(all_indices[dataset.env_ids == validation_env_id])
        train_envs = [env_name for env_name in source_envs if env_name != validation_env_name]

        split = {
            "mode": mode,
            "inner_validation": inner_validation,
            "target_env": target_env,
            "source_envs": source_envs,
            "train_envs": train_envs,
            "val_envs": [validation_env_name],
            "loso_val_env": validation_env_name,
            "train_indices": train_indices.tolist(),
            "val_indices": val_indices.tolist(),
            "test_indices": test_indices.tolist(),
        }
        validate_split_integrity(dataset, split)
        return split

    if inner_validation not in {"stratified_holdout", "source_stratified_holdout"}:
        raise ValueError(
            f"Unsupported DG inner validation mode '{inner_validation}'. "
            "Choose from ['stratified_holdout', 'source_stratified_holdout', 'loso', "
            "'subject_grouped_holdout', 'inferred_group_block_holdout']."
        )

    train_parts: List[np.ndarray] = []
    val_parts: List[np.ndarray] = []

    for env_id, env_name in enumerate(dataset.env_names):
        if env_id == target_env_id:
            continue
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
        "inner_validation": inner_validation,
        "target_env": target_env,
        "source_envs": source_envs,
        "train_envs": source_envs,
        "val_envs": source_envs,
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
        "test_indices": test_indices.tolist(),
    }
    validate_split_integrity(dataset, split)
    return split


def validate_split_integrity(dataset: PaCsiArrayDataset, split: Mapping[str, Any]) -> None:
    all_sample_ids = [int(sample_id) for sample_id in dataset.sample_ids.tolist()]
    if len(all_sample_ids) != len(set(all_sample_ids)):
        raise ValueError("Dataset sample_ids must be globally unique before splitting.")

    subset_lists = {
        "train": [int(index) for index in split["train_indices"]],
        "val": [int(index) for index in split["val_indices"]],
        "test": [int(index) for index in split["test_indices"]],
    }
    for subset_name, indices in subset_lists.items():
        if len(indices) != len(set(indices)):
            raise ValueError(f"{subset_name} split contains duplicate indices.")
        if any(index < 0 or index >= len(dataset) for index in indices):
            raise ValueError(f"{subset_name} split contains an out-of-range index.")

    train_set = set(subset_lists["train"])
    val_set = set(subset_lists["val"])
    test_set = set(subset_lists["test"])

    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError("Train/val/test indices overlap.")
    if train_set | val_set | test_set != set(range(len(dataset))):
        raise ValueError("Train/val/test indices must partition the complete dataset.")

    subset_sample_ids = {
        subset_name: {int(dataset.sample_ids[index]) for index in indices}
        for subset_name, indices in subset_lists.items()
    }
    if (
        subset_sample_ids["train"] & subset_sample_ids["val"]
        or subset_sample_ids["train"] & subset_sample_ids["test"]
        or subset_sample_ids["val"] & subset_sample_ids["test"]
    ):
        raise ValueError("Train/val/test sample_ids overlap.")

    target_env = split.get("target_env")
    target_not_loaded = bool(split.get("target_not_loaded", False))
    dg_mode = split.get("mode") in {"dg_loeo", "dg_loeo_source_only"}
    if target_not_loaded and split.get("mode") != "dg_loeo_source_only":
        raise ValueError("target_not_loaded is only valid for mode='dg_loeo_source_only'.")

    if dg_mode and target_env is not None:
        if target_not_loaded:
            if target_env in dataset.env_names:
                raise ValueError("Logical outer target must not be present in a source-only dataset.")
            if test_set:
                raise ValueError("Source-only grouped split must have an empty test split.")
            declared_sources = {str(env_name) for env_name in split.get("source_envs", [])}
            if declared_sources != set(dataset.env_names):
                raise ValueError(
                    "Source-only split source_envs must exactly match the loaded environments."
                )
        else:
            if target_env not in dataset.env_names:
                raise ValueError(f"Loaded-target DG split is missing target environment '{target_env}'.")
            target_env_id = dataset.env_names.index(target_env)
            if any(int(dataset.env_ids[index]) == target_env_id for index in train_set | val_set):
                raise ValueError("Target environment leaked into train or validation.")
            if any(int(dataset.env_ids[index]) != target_env_id for index in test_set):
                raise ValueError("DG test split contains non-target environments.")

            if split.get("inner_validation") == "loso":
                val_env_ids = {int(dataset.env_ids[index]) for index in val_set}
                if len(val_env_ids) != 1:
                    raise ValueError("LOSO validation split must contain exactly one source environment.")

                val_env_id = next(iter(val_env_ids))
                val_env_name = dataset.env_name(val_env_id)
                source_envs = set(str(env_name) for env_name in split.get("source_envs", []))
                if val_env_name not in source_envs:
                    raise ValueError("LOSO validation environment is not a source environment.")
                if any(int(dataset.env_ids[index]) == val_env_id for index in train_set):
                    raise ValueError("LOSO validation environment leaked into training.")

                expected_val_set = {
                    int(index)
                    for index in np.where(dataset.env_ids == val_env_id)[0].tolist()
                }
                if val_set != expected_val_set:
                    raise ValueError(
                        "LOSO validation split must contain all samples from its held-out source."
                    )

    if dg_mode and split.get("inner_validation") in {
        "subject_grouped_holdout",
        "inferred_group_block_holdout",
    }:
        require_subjects = split.get("inner_validation") == "subject_grouped_holdout"
        _validate_group_protocol_dataset(dataset, require_subjects=require_subjects)

        group_sets = {
            subset_name: set(dataset.group_counts(indices).keys())
            for subset_name, indices in subset_lists.items()
        }
        if (
            group_sets["train"] & group_sets["val"]
            or group_sets["train"] & group_sets["test"]
            or group_sets["val"] & group_sets["test"]
        ):
            raise ValueError("Train/val/test group ids overlap.")

        for subset_name, indices in subset_lists.items():
            index_set = set(indices)
            for group_id in group_sets[subset_name]:
                complete_group = {
                    int(index)
                    for index in range(len(dataset))
                    if _normalise_optional_id(dataset.group_ids[index]) == group_id
                }
                if not complete_group.issubset(index_set):
                    raise ValueError(f"Group {group_id} is only partially assigned to {subset_name}.")

        source_envs = [str(env_name) for env_name in split.get("source_envs", [])]
        validation_table = split.get("validation_group_ids", {})
        for source_env in source_envs:
            if source_env not in dataset.env_names:
                raise ValueError(f"Split declares unloaded source environment '{source_env}'.")
            source_env_id = dataset.env_names.index(source_env)
            train_env_indices = [
                index for index in subset_lists["train"]
                if int(dataset.env_ids[index]) == source_env_id
            ]
            val_env_indices = [
                index for index in subset_lists["val"]
                if int(dataset.env_ids[index]) == source_env_id
            ]
            train_groups = set(dataset.group_counts(train_env_indices).keys())
            val_groups = set(dataset.group_counts(val_env_indices).keys())
            if len(train_groups) != 8 or len(val_groups) != 2:
                raise ValueError(
                    f"Grouped holdout for {source_env} requires 8 train groups and 2 val groups; "
                    f"got {len(train_groups)} and {len(val_groups)}."
                )
            expected_val_groups = {
                str(group_id) for group_id in validation_table.get(source_env, [])
            }
            if val_groups != expected_val_groups:
                raise ValueError(
                    f"Validation groups for {source_env} do not match the explicit F0 table: "
                    f"actual={sorted(val_groups)}, expected={sorted(expected_val_groups)}."
                )

        expected_test_groups = 0 if target_not_loaded else PA_CSI_GROUPS_PER_ENV
        if len(group_sets["test"]) != expected_test_groups:
            raise ValueError(
                f"Grouped holdout test split must contain {expected_test_groups} target groups."
            )

        if require_subjects:
            subject_sets = {
                subset_name: set(dataset.subject_counts(indices).keys())
                for subset_name, indices in subset_lists.items()
            }
            if (
                subject_sets["train"] & subject_sets["val"]
                or subject_sets["train"] & subject_sets["test"]
                or subject_sets["val"] & subject_sets["test"]
            ):
                raise ValueError("Train/val/test verified subject ids overlap.")


def build_split_manifest(dataset: PaCsiArrayDataset, split: Mapping[str, Any]) -> Dict[str, Any]:
    split_indices = {
        subset_name: [int(index) for index in split[f"{subset_name}_indices"]]
        for subset_name in ("train", "val", "test")
    }

    def identifier_set(values: np.ndarray, indices: Sequence[int]) -> set[str]:
        return {
            identifier
            for index in indices
            if (identifier := _normalise_optional_id(values[int(index)])) is not None
        }

    sample_sets = {
        subset_name: {int(dataset.sample_ids[index]) for index in indices}
        for subset_name, indices in split_indices.items()
    }
    group_sets = {
        subset_name: identifier_set(dataset.group_ids, indices)
        for subset_name, indices in split_indices.items()
    }
    subject_sets = {
        subset_name: identifier_set(dataset.subject_ids, indices)
        for subset_name, indices in split_indices.items()
    }

    def pairwise_overlap(sets: Mapping[str, set[Any]]) -> Dict[str, List[Any]]:
        return {
            "train_val": sorted(sets["train"] & sets["val"]),
            "train_test": sorted(sets["train"] & sets["test"]),
            "val_test": sorted(sets["val"] & sets["test"]),
        }

    sample_overlap = pairwise_overlap(sample_sets)
    group_overlap = pairwise_overlap(group_sets)
    subject_overlap = pairwise_overlap(subject_sets)
    group_checked = any(group_sets.values())
    subject_checked = any(subject_sets.values())

    data_files = dataset.provenance.get("data_files", {})
    file_sha256 = {
        str(env_name): {
            str(file_kind): (
                str(record.get("sha256")) if record.get("sha256") is not None else None
            )
            for file_kind, record in records.items()
        }
        for env_name, records in data_files.items()
    }
    manifest: Dict[str, Any] = {
        "mode": split["mode"],
        "inner_validation": split.get("inner_validation"),
        "target_env": split.get("target_env"),
        "target_not_loaded": bool(split.get("target_not_loaded", False)),
        "source_envs": list(split.get("source_envs", [])),
        "train_envs": list(split.get("train_envs", [])),
        "val_envs": list(split.get("val_envs", [])),
        "loso_val_env": split.get("loso_val_env"),
        "dataset_size": len(dataset),
        "group_fold": split.get("group_fold"),
        "split_seed": split.get("split_seed"),
        "validation_group_ids": {
            str(env_name): [str(group_id) for group_id in group_ids]
            for env_name, group_ids in split.get("validation_group_ids", {}).items()
        },
        "file_sha256": file_sha256,
        "data_files": data_files,
        "sidecar_status": dataset.provenance.get("sidecar_status", {}),
        "subject_sidecars_complete": bool(
            dataset.provenance.get("subject_sidecars_complete", False)
        ),
        "group_protocol": dataset.provenance.get("group_protocol"),
    }
    for subset_name, indices in split_indices.items():
        group_counts = dataset.group_counts(indices)
        subject_counts = dataset.subject_counts(indices)
        manifest[subset_name] = {
            "count": len(indices),
            "env_counts": dataset.env_counts(indices),
            "label_counts": dataset.label_counts(indices),
            "group_count": len(group_counts),
            "group_ids": sorted(group_counts),
            "group_counts": group_counts,
            "group_ids_complete": dataset.has_complete_group_ids(indices),
            "subject_count": len(subject_counts),
            "subject_ids": sorted(subject_counts),
            "subject_counts": subject_counts,
            "subject_ids_complete": dataset.has_complete_subject_ids(indices),
            "indices_sha256": _hash_indices(indices),
        }

    canonical_indices = json.dumps(
        split_indices,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    manifest["index_sha256"] = hashlib.sha256(canonical_indices).hexdigest()
    manifest["overlap_audit"] = {
        "sample_ids": sample_overlap,
        "group_ids": group_overlap,
        "group_ids_checked": group_checked,
        "subject_ids": subject_overlap,
        "subject_ids_checked": subject_checked,
        "all_clear": not any(sample_overlap.values())
        and not any(group_overlap.values())
        and (not subject_checked or not any(subject_overlap.values())),
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
