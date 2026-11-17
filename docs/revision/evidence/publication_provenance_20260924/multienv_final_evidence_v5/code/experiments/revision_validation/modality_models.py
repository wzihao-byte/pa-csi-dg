"""Separately specified phase and two-stream capacity controls for revision R.

These are new R controls. They do not recreate the historical H/Y fusion model.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments.revision_validation.models import build_model


def historical_feature_phase(values):
    """Exactly the H loader order: unwrap the stored float64 values, then cast."""
    values = np.asarray(values)
    if values.ndim != 3 or values.shape[1:] != (850, 90) or values.dtype != np.float64:
        raise ValueError("Historical control requires original float64 [batch,850,90] phase")
    return np.unwrap(values, axis=-1).astype(np.float32)


class TwinStreamRevisionModel(nn.Module):
    def __init__(self, seed):
        super().__init__()
        primary, secondary = build_model("R10", seed), build_model("R10", seed + 100000)
        self.encoder, self.secondary_encoder = primary.encoder, secondary.encoder
        d = self.encoder.output_dim
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 200000)
            fusion_projection = nn.Linear(2 * d, 256)
        self.representation = nn.Sequential(fusion_projection, primary.representation[1], primary.representation[2])
        self.classifier = primary.classifier
        self.projection_head = primary.projection_head

    def forward(self, primary, *, secondary, project=True):
        encoded = torch.cat([self.encoder(primary), self.secondary_encoder(secondary)], dim=-1)
        feature = self.representation(encoded)
        output = {"feature": feature, "logits": self.classifier(feature)}
        if project:
            output["projection"] = F.normalize(self.projection_head(feature), dim=-1)
        return output


def build_twin_model(seed):
    # Input routing lives in the frozen experiment configuration. Amp+amp and
    # amp+phase therefore instantiate precisely the same model and initial state.
    return TwinStreamRevisionModel(seed)
