from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class AdaptiveKMarginConfig:
    enabled: bool = False
    warmup_epochs: int = 10
    recalibrate_interval: int = 5
    beta_softmax: float = 10.0
    gap_threshold_tau: float = 0.0
    rho0: float = 0.65
    rho_min: float = 0.45
    rho_max: float = 0.85
    rho_entropy_scale: float = 0.15
    rho_instability_scale: float = 0.15
    m_min: float = 0.02
    m_max: float = 0.20
    alpha_h: float = 1.0
    beta_v: float = 1.0
    gamma_kappa: float = 1.0
    delta_u: float = 1.0
    lambda_akm: float = 0.1
    eps: float = 1e-8

    @classmethod
    def from_mapping(cls, payload: Optional[Mapping[str, Any]]) -> "AdaptiveKMarginConfig":
        values = dict(payload or {})
        if "rho_entropy_scale_a" in values and "rho_entropy_scale" not in values:
            values["rho_entropy_scale"] = values["rho_entropy_scale_a"]
        if "rho_instability_scale_b" in values and "rho_instability_scale" not in values:
            values["rho_instability_scale"] = values["rho_instability_scale_b"]
        valid_keys = set(cls.__dataclass_fields__.keys())
        return cls(**{key: values[key] for key in values if key in valid_keys})


def _safe_minmax(values: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = torch.zeros_like(values)
    if not mask.any():
        return normalized
    selected = values[mask]
    denom = (selected.max() - selected.min()).clamp_min(eps)
    normalized[mask] = (values[mask] - selected.min()) / denom
    return normalized


def _tensor_mean(values: torch.Tensor) -> float:
    return float(values.mean().item()) if values.numel() > 0 else 0.0


class AdaptiveKMarginRecalibrator:
    def __init__(self, config: AdaptiveKMarginConfig, num_classes: int, device: torch.device) -> None:
        self.config = config
        self.num_classes = int(num_classes)
        self.device = device
        self.state: Optional[Dict[str, torch.Tensor]] = None
        self.debug_stats: Dict[str, Any] = self._empty_debug()

    @torch.no_grad()
    def recalibrate(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        artifact_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        was_training = model.training
        model.eval()
        embeddings: List[torch.Tensor] = []
        labels_parts: List[torch.Tensor] = []
        domains_parts: List[torch.Tensor] = []

        for amplitude, phase, labels, domains, _sample_ids in loader:
            outputs = model(amplitude.to(self.device), phase.to(self.device))
            embeddings.append(F.normalize(outputs["projection"].detach(), dim=-1).cpu())
            labels_parts.append(labels.detach().cpu().long())
            domains_parts.append(domains.detach().cpu().long())

        if was_training:
            model.train()

        if not embeddings:
            self.state = None
            self.debug_stats = self._empty_debug()
            if artifact_path is not None:
                self.save_debug_json(artifact_path)
            return self.result()

        tables = self._build_tables(
            z=torch.cat(embeddings, dim=0),
            labels=torch.cat(labels_parts, dim=0),
            domains=torch.cat(domains_parts, dim=0),
        )
        self.debug_stats = tables["debug_stats"]
        self.state = {
            key: value.to(self.device)
            for key, value in tables.items()
            if isinstance(value, torch.Tensor)
        }

        if artifact_path is not None:
            self.save_artifact_json(artifact_path, tables)
        return self.result()

    def _build_tables(self, z: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor) -> Dict[str, Any]:
        cfg = self.config
        num_classes = self.num_classes
        feature_dim = int(z.size(1))
        domain_ids = [int(value) for value in torch.unique(domains).tolist()]

        global_prototypes = torch.zeros(num_classes, feature_dim, dtype=z.dtype)
        prototype_valid_mask = torch.zeros(num_classes, dtype=torch.bool)
        for class_id in range(num_classes):
            class_mask = labels == class_id
            if class_mask.any():
                global_prototypes[class_id] = F.normalize(z[class_mask].mean(dim=0), dim=0)
                prototype_valid_mask[class_id] = True

        h_sum = torch.zeros(num_classes, num_classes, dtype=z.dtype)
        h_sq_sum = torch.zeros_like(h_sum)
        v_sum = torch.zeros_like(h_sum)
        pair_count = torch.zeros(num_classes, num_classes, dtype=torch.long)
        kappa_sum = torch.zeros(num_classes, dtype=z.dtype)
        kappa_count = torch.zeros(num_classes, dtype=torch.long)

        for domain_id in domain_ids:
            heldout_mask = domains == domain_id
            loso_source_mask = ~heldout_mask
            loso_prototypes = torch.zeros(num_classes, feature_dim, dtype=z.dtype)
            loso_valid = torch.zeros(num_classes, dtype=torch.bool)

            for class_id in range(num_classes):
                proto_mask = loso_source_mask & (labels == class_id)
                if proto_mask.any():
                    loso_prototypes[class_id] = F.normalize(z[proto_mask].mean(dim=0), dim=0)
                    loso_valid[class_id] = True

            if int(loso_valid.sum().item()) < 2:
                continue

            heldout_z = z[heldout_mask]
            heldout_labels = labels[heldout_mask]
            sims = heldout_z @ loso_prototypes.T

            for class_id in range(num_classes):
                sample_mask = heldout_labels == class_id
                if not sample_mask.any() or not loso_valid[class_id]:
                    continue

                class_sims = sims[sample_mask]
                true_scores = class_sims[:, class_id]
                kappa_sum[class_id] += ((true_scores + 1.0) * 0.5).mean()
                kappa_count[class_id] += 1

                valid_negatives = loso_valid.clone()
                valid_negatives[class_id] = False
                if not valid_negatives.any():
                    continue

                neg_indices = torch.arange(num_classes)[valid_negatives]
                neg_probs = torch.softmax(class_sims[:, valid_negatives] * cfg.beta_softmax, dim=1)
                for local_index, neg_class in enumerate(neg_indices.tolist()):
                    neg_scores = class_sims[:, neg_class]
                    hardness = neg_probs[:, local_index].mean()
                    violation = (true_scores - neg_scores <= cfg.gap_threshold_tau).float().mean()
                    h_sum[class_id, neg_class] += hardness
                    h_sq_sum[class_id, neg_class] += hardness * hardness
                    v_sum[class_id, neg_class] += violation
                    pair_count[class_id, neg_class] += 1

        pair_valid = pair_count > 0
        h = torch.zeros_like(h_sum)
        v = torch.zeros_like(v_sum)
        u = torch.zeros_like(h_sum)
        h[pair_valid] = h_sum[pair_valid] / pair_count[pair_valid].to(z.dtype)
        v[pair_valid] = v_sum[pair_valid] / pair_count[pair_valid].to(z.dtype)
        h_mean_sq = torch.zeros_like(h_sum)
        h_mean_sq[pair_valid] = h_sq_sum[pair_valid] / pair_count[pair_valid].to(z.dtype)
        u[pair_valid] = (h_mean_sq[pair_valid] - h[pair_valid].pow(2)).clamp_min(0.0).sqrt()

        kappa = torch.zeros(num_classes, dtype=z.dtype)
        kappa_valid = kappa_count > 0
        kappa[kappa_valid] = kappa_sum[kappa_valid] / kappa_count[kappa_valid].to(z.dtype)

        h_tilde = _safe_minmax(h, pair_valid, cfg.eps)
        v_tilde = _safe_minmax(v, pair_valid, cfg.eps)
        u_tilde = _safe_minmax(u, pair_valid, cfg.eps)
        kappa_tilde = _safe_minmax(kappa, kappa_valid, cfg.eps)
        kappa_pair = kappa_tilde[:, None].expand_as(h_tilde)

        pair_scores = torch.zeros_like(h)
        pair_scores[pair_valid] = (
            h_tilde[pair_valid]
            * v_tilde[pair_valid]
            * (1.0 - u_tilde[pair_valid])
            * kappa_pair[pair_valid]
        )

        pair_margins = torch.zeros_like(h)
        margin_logits = (
            cfg.alpha_h * h_tilde
            + cfg.beta_v * v_tilde
            + cfg.gamma_kappa * kappa_pair
            - cfg.delta_u * u_tilde
        )
        pair_margins[pair_valid] = cfg.m_min + (cfg.m_max - cfg.m_min) * torch.sigmoid(margin_logits[pair_valid])

        selected_pairs = torch.zeros(num_classes, num_classes, dtype=torch.bool)
        pair_weights = torch.zeros_like(h)
        class_k = torch.zeros(num_classes, dtype=torch.long)
        rho_by_class = torch.zeros(num_classes, dtype=z.dtype)

        for class_id in range(num_classes):
            selectable = pair_valid[class_id] & (pair_scores[class_id] > 0)
            if not selectable.any():
                continue

            row_h = h[class_id, selectable]
            h_distribution = row_h / row_h.sum().clamp_min(cfg.eps)
            entropy = -(h_distribution * torch.log(h_distribution.clamp_min(cfg.eps))).sum()
            if h_distribution.numel() > 1:
                entropy = entropy / torch.log(torch.tensor(float(h_distribution.numel()), dtype=z.dtype))
            mean_u = u_tilde[class_id, selectable].mean()
            rho = torch.clamp(
                torch.tensor(cfg.rho0, dtype=z.dtype)
                + cfg.rho_entropy_scale * entropy
                - cfg.rho_instability_scale * mean_u,
                min=cfg.rho_min,
                max=cfg.rho_max,
            )
            rho_by_class[class_id] = rho

            neg_indices = torch.arange(num_classes)[selectable]
            scores = pair_scores[class_id, selectable]
            order = torch.argsort(scores, descending=True)
            cumulative = torch.cumsum(scores[order] / scores.sum().clamp_min(cfg.eps), dim=0)
            selected_count = int((cumulative < rho).sum().item()) + 1
            chosen = neg_indices[order[:selected_count]]
            selected_pairs[class_id, chosen] = True
            class_k[class_id] = selected_count
            weights = pair_scores[class_id, chosen]
            pair_weights[class_id, chosen] = weights / weights.sum().clamp_min(cfg.eps)

        debug_stats = {
            "num_calibration_samples": int(z.size(0)),
            "num_source_domains": int(len(domain_ids)),
            "classes_with_global_prototypes": int(prototype_valid_mask.sum().item()),
            "mean_effective_k": _tensor_mean(class_k[class_k > 0].float()),
            "mean_margin": _tensor_mean(pair_margins[selected_pairs]),
            "fraction_valid_pairs": float(pair_valid.float().mean().item()),
            "fraction_selected_pairs": float(selected_pairs.float().mean().item()),
            "effective_k_by_class": {str(i): int(class_k[i].item()) for i in range(num_classes)},
            "rho_by_class": {str(i): float(rho_by_class[i].item()) for i in range(num_classes)},
        }

        return {
            "class_k": class_k,
            "pair_scores": pair_scores,
            "pair_margins": pair_margins,
            "pair_weights": pair_weights,
            "selected_pairs": selected_pairs,
            "cached_prototypes": global_prototypes,
            "prototype_valid_mask": prototype_valid_mask,
            "debug_stats": debug_stats,
        }

    def _empty_debug(self) -> Dict[str, Any]:
        return {
            "num_calibration_samples": 0,
            "num_source_domains": 0,
            "classes_with_global_prototypes": 0,
            "mean_effective_k": 0.0,
            "mean_margin": 0.0,
            "fraction_valid_pairs": 0.0,
            "fraction_selected_pairs": 0.0,
            "effective_k_by_class": {},
            "rho_by_class": {},
        }

    def result(self) -> Dict[str, Any]:
        return {"state": self.state, "debug_stats": self.debug_stats}

    def save_debug_json(self, path: Path) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.debug_stats, handle, indent=2)

    def save_artifact_json(self, path: Path, tables: Mapping[str, Any]) -> None:
        import json

        payload: Dict[str, Any] = {"debug_stats": dict(self.debug_stats)}
        for key in (
            "class_k",
            "pair_scores",
            "pair_margins",
            "pair_weights",
            "selected_pairs",
            "prototype_valid_mask",
            "cached_prototypes",
        ):
            value = tables.get(key)
            if isinstance(value, torch.Tensor):
                payload[key] = value.cpu().tolist()

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
