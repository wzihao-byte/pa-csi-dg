"""Deterministic, protocol-facing utilities for the joint experiment runner.

The legacy training code relies on process-global ``random`` state for its
balanced sampler.  That is convenient for exploratory runs, but it makes it
too easy for model-construction changes to alter the batch sequence.  The
helpers here keep sampler and model randomness explicitly separated and also
centralize the audit calculations required by the locked protocol.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import BatchSampler


def _update_hash_with_text(digest: "hashlib._Hash", value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
    digest.update(encoded)


def sha256_int_sequence(values: Iterable[int]) -> str:
    """Hash an integer sequence without depending on platform integer width."""

    array = np.asarray([int(value) for value in values], dtype="<i8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def sha256_nested_int_sequences(values: Iterable[Iterable[int]]) -> str:
    """Hash a sequence of batches, preserving both batch and item boundaries."""

    digest = hashlib.sha256()
    for batch in values:
        array = np.asarray([int(value) for value in batch], dtype="<i8")
        digest.update(len(array).to_bytes(8, byteorder="little", signed=False))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def trainable_parameter_hash(model: torch.nn.Module) -> str:
    """Return a stable hash of names, shapes, dtypes, and trainable values."""

    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        if not parameter.requires_grad:
            continue
        value = parameter.detach().cpu().contiguous()
        _update_hash_with_text(digest, name)
        _update_hash_with_text(digest, str(value.dtype))
        _update_hash_with_text(digest, json.dumps(list(value.shape)))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))


class DeterministicDomainClassBatchSampler(BatchSampler):
    """Yield an exact, reproducible domain/class-balanced batch schedule.

    For the locked E1--E3 protocol this produces 300 batches per epoch, eight
    samples from each of two source domains, and at least one example of every
    class in each domain.  Pools cycle independently when a minority class is
    exhausted.  All random choices use a sampler-local ``random.Random``
    instance derived solely from ``seed`` and ``epoch``.
    """

    def __init__(
        self,
        domain_to_indices: Mapping[int, Sequence[int]],
        labels: np.ndarray,
        *,
        batch_size: int,
        num_batches: int,
        seed: int,
    ) -> None:
        self.domain_to_indices = {
            int(domain): [int(index) for index in indices]
            for domain, indices in domain_to_indices.items()
        }
        self.domains = tuple(sorted(self.domain_to_indices))
        if not self.domains:
            raise ValueError("At least one source domain is required.")

        self.labels = np.asarray(labels)
        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.epoch = 0
        if self.batch_size <= 0 or self.num_batches <= 0:
            raise ValueError("batch_size and num_batches must both be positive.")
        if self.batch_size % len(self.domains) != 0:
            raise ValueError(
                "The locked sampler requires an equal integer number of samples "
                "from every source domain."
            )
        self.samples_per_domain = self.batch_size // len(self.domains)

        classes_by_domain: Dict[int, Tuple[int, ...]] = {}
        for domain, indices in self.domain_to_indices.items():
            if not indices:
                raise ValueError(f"Source domain {domain} has no training indices.")
            classes_by_domain[domain] = tuple(sorted({int(self.labels[index]) for index in indices}))
        class_sets = set(classes_by_domain.values())
        if len(class_sets) != 1:
            raise ValueError(f"Source domains do not expose identical classes: {classes_by_domain}")
        self.classes = next(iter(class_sets))
        if not self.classes:
            raise ValueError("At least one class is required.")
        if self.samples_per_domain < len(self.classes):
            raise ValueError(
                f"samples_per_domain={self.samples_per_domain} cannot cover "
                f"all {len(self.classes)} classes."
            )

        self._base_pools: Dict[int, Dict[int, Tuple[int, ...]]] = {}
        for domain, indices in self.domain_to_indices.items():
            domain_pools: Dict[int, Tuple[int, ...]] = {}
            for class_id in self.classes:
                pool = tuple(index for index in indices if int(self.labels[index]) == class_id)
                if not pool:
                    raise ValueError(f"Source domain {domain} has no samples for class {class_id}.")
                domain_pools[class_id] = pool
            self._base_pools[domain] = domain_pools

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _epoch_seed(self) -> int:
        # A fixed arithmetic map avoids Python's process-randomized hash().
        return self.seed + self.epoch * 1_000_003

    def __iter__(self) -> Iterable[List[int]]:
        rng = random.Random(self._epoch_seed())
        pools: Dict[int, Dict[int, List[int]]] = {
            domain: {class_id: list(indices) for class_id, indices in domain_pools.items()}
            for domain, domain_pools in self._base_pools.items()
        }
        positions: Dict[int, Dict[int, int]] = {
            domain: {class_id: 0 for class_id in self.classes} for domain in self.domains
        }
        for domain in self.domains:
            for class_id in self.classes:
                rng.shuffle(pools[domain][class_id])

        def take(domain: int, class_id: int) -> int:
            pool = pools[domain][class_id]
            position = positions[domain][class_id]
            if position >= len(pool):
                rng.shuffle(pool)
                position = 0
            positions[domain][class_id] = position + 1
            return int(pool[position])

        for _ in range(self.num_batches):
            batch: List[int] = []
            for domain in self.domains:
                # Cover all classes first.  Randomly rotated/permuted extras
                # prevent the two surplus positions from favoring fixed IDs.
                class_order = list(self.classes)
                rng.shuffle(class_order)
                chosen = list(class_order)
                extra_count = self.samples_per_domain - len(self.classes)
                while extra_count > 0:
                    extras = list(self.classes)
                    rng.shuffle(extras)
                    take_count = min(extra_count, len(extras))
                    chosen.extend(extras[:take_count])
                    extra_count -= take_count
                batch.extend(take(domain, class_id) for class_id in chosen)
            rng.shuffle(batch)
            yield batch

    def schedule_hash(self) -> str:
        return sha256_nested_int_sequences(iter(self))


def filter_data_config_to_environments(
    data_config: Mapping[str, Any],
    environment_names: Sequence[str],
) -> Dict[str, Any]:
    """Copy a data config while retaining only explicitly authorized env files."""

    filtered = copy.deepcopy(dict(data_config))
    env_files = data_config.get("env_files")
    if not isinstance(env_files, Mapping):
        raise ValueError("data.env_files must be a mapping.")
    requested = [str(name) for name in environment_names]
    missing = [name for name in requested if name not in env_files]
    if missing:
        raise ValueError(f"Missing environment files for: {missing}")
    filtered["env_files"] = {name: copy.deepcopy(env_files[name]) for name in requested}
    return filtered


def assert_phase_unwrap_disabled(
    raw_config: Mapping[str, Any],
    resolved_config: Optional[Mapping[str, Any]] = None,
) -> None:
    """Fail unless both raw and resolved configs explicitly disable unwrap."""

    for label, config in (("raw", raw_config), ("resolved", resolved_config)):
        if config is None:
            continue
        data_config = config.get("data")
        if not isinstance(data_config, Mapping):
            raise ValueError(f"{label} config has no data mapping.")
        if data_config.get("unwrap_phase") is not False:
            raise ValueError(f"{label} config must explicitly set data.unwrap_phase=false.")
        protocol_config = config.get("protocol")
        if isinstance(protocol_config, Mapping) and protocol_config.get("phase_unwrap") is not False:
            raise ValueError(f"{label} config protocol.phase_unwrap must be false.")


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fixed_label_metrics(
    labels: Sequence[int] | np.ndarray,
    predictions: Sequence[int] | np.ndarray,
    *,
    num_classes: int = 6,
) -> Dict[str, Any]:
    """Compute aggregate and per-class metrics without dropping absent labels."""

    labels_array = np.asarray(labels, dtype=np.int64)
    predictions_array = np.asarray(predictions, dtype=np.int64)
    if labels_array.shape != predictions_array.shape:
        raise ValueError("labels and predictions must have the same shape.")
    class_ids = np.arange(num_classes, dtype=np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_array,
        predictions_array,
        labels=class_ids,
        zero_division=0,
    )
    matrix = confusion_matrix(labels_array, predictions_array, labels=class_ids)
    result: Dict[str, Any] = {
        "accuracy": float(accuracy_score(labels_array, predictions_array)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "per_class": {
            str(class_id): {
                "precision": float(precision[class_id]),
                "recall": float(recall[class_id]),
                "f1": float(f1[class_id]),
                "support": int(support[class_id]),
            }
            for class_id in range(num_classes)
        },
        "confusion_matrix": matrix.astype(np.int64).tolist(),
    }
    if num_classes > 4:
        support_1 = int(matrix[1].sum())
        support_4 = int(matrix[4].sum())
        one_to_four = int(matrix[1, 4])
        four_to_one = int(matrix[4, 1])
        result["c1_c4_pair_confusion"] = {
            "c1_to_c4_count": one_to_four,
            "c4_to_c1_count": four_to_one,
            "bidirectional_count": one_to_four + four_to_one,
            "c1_to_c4_rate": float(one_to_four / support_1) if support_1 else 0.0,
            "c4_to_c1_rate": float(four_to_one / support_4) if support_4 else 0.0,
        }
    return result


DEFAULT_GRADIENT_BLOCKS: Mapping[str, Tuple[str, ...]] = {
    "amplitude_encoder": ("amp_encoder.",),
    "phase_encoder": ("phase_encoder.",),
    "fusion": ("fusion.",),
    "shared_backbone": ("shared_backbone.", "backbone."),
    "classifier": ("classifier.",),
    "projection_head": ("projection_head.",),
}


def _gradient_summary(
    named_parameters: Sequence[Tuple[str, torch.nn.Parameter]],
    gradients_a: Mapping[str, Optional[torch.Tensor]],
    gradients_b: Mapping[str, Optional[torch.Tensor]],
    prefixes: Sequence[str],
) -> Tuple[float, float, float]:
    norm_a_sq = 0.0
    norm_b_sq = 0.0
    dot = 0.0
    for name, _ in named_parameters:
        if not any(name.startswith(prefix) for prefix in prefixes):
            continue
        grad_a = gradients_a.get(name)
        grad_b = gradients_b.get(name)
        if grad_a is not None:
            detached_a = grad_a.detach().double()
            norm_a_sq += float(torch.sum(detached_a * detached_a).item())
        if grad_b is not None:
            detached_b = grad_b.detach().double()
            norm_b_sq += float(torch.sum(detached_b * detached_b).item())
        if grad_a is not None and grad_b is not None:
            dot += float(torch.sum(grad_a.detach().double() * grad_b.detach().double()).item())
    return math.sqrt(norm_a_sq), math.sqrt(norm_b_sq), dot


def ce_supcon_gradient_diagnostics(
    model: torch.nn.Module,
    ce_loss: torch.Tensor,
    supcon_loss: torch.Tensor,
    *,
    lambda_supcon: float,
    blocks: Mapping[str, Sequence[str]] = DEFAULT_GRADIENT_BLOCKS,
    epsilon: float = 1e-12,
) -> Dict[str, Any]:
    """Measure separate CE/SupCon gradients from one shared forward graph.

    The function never calls ``backward`` and therefore does not populate
    ``parameter.grad`` or modify optimizer state.  Classifier and projection
    heads report raw norms only, as required by the protocol.
    """

    named_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [parameter for _, parameter in named_parameters]
    ce_values = torch.autograd.grad(
        ce_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    supcon_values = torch.autograd.grad(
        supcon_loss,
        parameters,
        retain_graph=False,
        allow_unused=True,
    )
    ce_gradients = {name: gradient for (name, _), gradient in zip(named_parameters, ce_values)}
    supcon_gradients = {name: gradient for (name, _), gradient in zip(named_parameters, supcon_values)}

    report: Dict[str, Any] = {}
    head_blocks = {"classifier", "projection_head"}
    for block_name, prefixes in blocks.items():
        ce_norm, supcon_norm, dot = _gradient_summary(
            named_parameters,
            ce_gradients,
            supcon_gradients,
            prefixes,
        )
        if not all(math.isfinite(value) for value in (ce_norm, supcon_norm, dot)):
            raise FloatingPointError(f"Non-finite gradient diagnostic in block {block_name}.")
        block_report: Dict[str, Any] = {
            "ce_grad_norm": ce_norm,
            "supcon_grad_norm": supcon_norm,
            "weighted_supcon_grad_norm": float(lambda_supcon) * supcon_norm,
        }
        if block_name not in head_blocks:
            denominator = ce_norm + float(epsilon)
            block_report["weighted_supcon_to_ce_ratio"] = (
                float(lambda_supcon) * supcon_norm / denominator
            )
            if ce_norm > 0.0 and supcon_norm > 0.0:
                block_report["gradient_cosine"] = dot / (ce_norm * supcon_norm)
            else:
                block_report["gradient_cosine"] = None
        report[block_name] = block_report
    return report


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")

