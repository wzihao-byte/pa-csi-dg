from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dg_models import AntennaViewBuilder, GatedResidualFusion, PaCsiStreamEncoder  # noqa: E402


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class SharedPrivateDecomposition(nn.Module):
    """Split a PA-CSI feature into SimMMDG-style shared and private components."""

    def __init__(
        self,
        input_dim: int,
        shared_dim: int,
        private_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.shared_encoder = _make_mlp(input_dim, hidden_dim, shared_dim, dropout)
        self.private_encoder = _make_mlp(input_dim, hidden_dim, private_dim, dropout)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.shared_encoder(feature), self.private_encoder(feature)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        return self.net(feature)


class SharedFeatureTranslator(nn.Module):
    """A small shared-space translator for optional cross-view regularization."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, shared_feature: torch.Tensor) -> torch.Tensor:
        return self.net(shared_feature)


class PaCsiSimMMDG(nn.Module):
    """PA-CSI DG model with an isolated SimMMDG-inspired shared/private head."""

    def __init__(self, input_dim: int, num_classes: int, model_config: Mapping[str, Any]) -> None:
        super().__init__()
        self.amp_encoder = PaCsiStreamEncoder(input_dim, model_config)
        self.phase_encoder = PaCsiStreamEncoder(input_dim, model_config)

        fusion_dim = int(model_config.get("fusion_dim", 256))
        self.fusion = GatedResidualFusion(self.amp_encoder.output_dim, fusion_dim)

        shared_dim = int(model_config.get("shared_dim", model_config.get("projection_dim", 128)))
        private_dim = int(model_config.get("private_dim", shared_dim))
        decomposition_hidden_dim = int(model_config.get("decomposition_hidden_dim", fusion_dim))
        dropout = float(model_config.get("dropout", 0.1))

        self.decomposition = SharedPrivateDecomposition(
            input_dim=fusion_dim,
            shared_dim=shared_dim,
            private_dim=private_dim,
            hidden_dim=decomposition_hidden_dim,
            dropout=dropout,
        )
        self.classifier = nn.Linear(shared_dim, num_classes)
        self.projection_head = ProjectionHead(
            input_dim=shared_dim,
            hidden_dim=int(model_config.get("projection_hidden_dim", max(shared_dim, fusion_dim))),
            output_dim=int(model_config.get("projection_dim", 128)),
            dropout=dropout,
        )
        self.shared_translator = SharedFeatureTranslator(
            dim=shared_dim,
            hidden_dim=int(model_config.get("translation_hidden_dim", max(shared_dim, fusion_dim))),
            dropout=float(model_config.get("translation_dropout", dropout)),
        )

        antenna_layout = model_config.get("antenna_layout")
        self.view_builder = AntennaViewBuilder(input_dim, antenna_layout) if antenna_layout else None
        self.fusion_dim = fusion_dim
        self.shared_dim = shared_dim
        self.private_dim = private_dim
        self.num_classes = num_classes

    def encode_features(self, amplitude: torch.Tensor, phase: torch.Tensor) -> Dict[str, torch.Tensor]:
        amp_feature = self.amp_encoder(amplitude)
        phase_feature = self.phase_encoder(phase)
        fused_feature = self.fusion(amp_feature, phase_feature)
        return {
            "amp_feature": amp_feature,
            "phase_feature": phase_feature,
            "fused_feature": fused_feature,
        }
    def decompose_feature(self, fused_feature: torch.Tensor) -> Dict[str, torch.Tensor]:
        original_shape = fused_feature.shape
        flat_feature = fused_feature.reshape(-1, original_shape[-1])
        shared_feature, private_feature = self.decomposition(flat_feature)
        projection = F.normalize(self.projection_head(shared_feature), dim=-1)

        shared_feature = shared_feature.reshape(*original_shape[:-1], -1)
        private_feature = private_feature.reshape(*original_shape[:-1], -1)
        projection = projection.reshape(*original_shape[:-1], -1)
        return {
            "shared_feature": shared_feature,
            "private_feature": private_feature,
            "projection": projection,
        }

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encode_features(amplitude, phase)
        decomposed = self.decompose_feature(features["fused_feature"])
        logits = self.classifier(decomposed["shared_feature"])
        features.update(decomposed)
        features["logits"] = logits
        return features

    def encode_antenna_views(self, amplitude: torch.Tensor, phase: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.view_builder is None:
            raise RuntimeError("Antenna views requested but model_config.antenna_layout is not configured.")

        amp_views = self.view_builder(amplitude)
        phase_views = self.view_builder(phase)
        batch_size, num_views, time_steps, input_dim = amp_views.shape

        flat_amp = amp_views.reshape(batch_size * num_views, time_steps, input_dim)
        flat_phase = phase_views.reshape(batch_size * num_views, time_steps, input_dim)
        encoded = self.encode_features(flat_amp, flat_phase)
        fused_views = encoded["fused_feature"].reshape(batch_size, num_views, -1)
        decomposed = self.decompose_feature(fused_views)

        return {
            "amp_views": amp_views,
            "phase_views": phase_views,
            "fused_views": fused_views,
            "shared_views": decomposed["shared_feature"],
            "private_views": decomposed["private_feature"],
            "projection_views": decomposed["projection"],
        }
