"""Amplitude-only candidates for a new, explicitly separate revision protocol.

The conventional global MLP and L124 residual TPP have identical projection
parameter counts. This does not assert identical function classes or residual
norms. Historical H/Y models and their checkpoints are not modified.
"""
from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F

from dg_models import ProjectionHead
from experiments.tpp_isolated.tpp_models import IsolatedPaCsiStreamEncoderTPP


MODEL_CONFIG = {
    "time_downsample": 2, "max_positions": 500, "position_kernels": 10,
    "mcat_layers": 5, "mcat_heads": 9, "dropout": 0.1,
    "temporal_kernel_num": 128, "temporal_kernel_sizes": [20, 40],
    "channel_group_width": 30, "channel_encoder_layers": 1,
    "channel_encoder_heads": 6, "channel_kernel_num": 16,
    "channel_kernel_sizes": [2, 4], "tpp_levels": [1, 2, 4],
    "tpp_pool_type": "max", "tpp_preserve_global": True,
    "tpp_pyramid_scale": 0.25,
}
ARMS = {
    "R00": {"pool": "global_mlp", "supcon": False},
    "R01": {"pool": "global_mlp", "supcon": True},
    "R10": {"pool": "tpp", "supcon": False},
    "R11": {"pool": "tpp", "supcon": True},
}


class GlobalMlpPool(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.convs = reference.convs
        self.activation = reference.activation
        self.dropout = reference.dropout
        self.output_dim = reference.output_dim
        self.pyramid_scale = reference.pyramid_scale
        assert reference.segment_count == 7 and self.output_dim % 2 == 0
        d = self.output_dim
        # TPP: Linear(7d,d,bias=True) has 7d^2+d parameters.
        # MLP: d -> 7d/2 (no bias) -> d (bias) has the same count and MACs.
        self.projection = nn.Sequential(nn.Linear(d, 7 * d // 2, bias=False),
                                        nn.ReLU(), nn.Linear(7 * d // 2, d, bias=True))
        assert sum(p.numel() for p in self.projection.parameters()) == sum(
            p.numel() for p in reference.projection.parameters())

    def forward(self, x):
        responses = [self.activation(conv(x.transpose(1, 2))) for conv in self.convs]
        global_feature = torch.cat([response.amax(dim=-1) for response in responses], dim=1)
        return self.dropout(global_feature + self.pyramid_scale * self.projection(global_feature))


class AmplitudeRevisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = IsolatedPaCsiStreamEncoderTPP(90, deepcopy(MODEL_CONFIG))
        self.representation = nn.Sequential(nn.Linear(self.encoder.output_dim, 256),
                                            nn.ELU(), nn.LayerNorm(256))
        self.classifier = nn.Linear(256, 6)
        self.projection_head = ProjectionHead(256, 256, 128)

    def forward(self, amplitude, *, project=True):
        feature = self.representation(self.encoder(amplitude))
        output = {"feature": feature, "logits": self.classifier(feature)}
        if project:
            output["projection"] = F.normalize(self.projection_head(feature), dim=-1)
        return output


def build_model(arm, seed):
    specification = ARMS[arm]
    # Build the common encoder and both heads first, so constructing a different
    # pooling projection cannot shift the random initialization of shared layers.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = AmplitudeRevisionModel()
        if specification["pool"] == "global_mlp":
            model.encoder.temporal_pool = GlobalMlpPool(model.encoder.temporal_pool)
            model.encoder.channel_pool = GlobalMlpPool(model.encoder.channel_pool)
    return model


def shared_initialization_audit(seed=401):
    models = {arm: build_model(arm, seed) for arm in ARMS}
    states = {arm: model.state_dict() for arm, model in models.items()}
    excluded = ("encoder.temporal_pool.projection.", "encoder.channel_pool.projection.")
    common = [key for key in states["R10"] if not key.startswith(excluded)]
    assert all(torch.equal(states["R10"][key], state[key]) for state in states.values() for key in common)
    assert all(torch.equal(states["R00"][key], states["R01"][key]) for key in states["R00"])
    assert all(torch.equal(states["R10"][key], states["R11"][key]) for key in states["R10"])
    counts = {arm: sum(p.numel() for p in model.parameters()) for arm, model in models.items()}
    assert len(set(counts.values())) == 1
    return {"seed": seed, "shared_state_tensors_exactly_equal": len(common),
            "same_pool_CE_SupCon_all_initial_tensors_equal": True, "total_parameters": counts}


def count_matmul_conv_macs(model, amplitude):
    """Count Conv1d, linear, attention and position-embedding MACs by shape.

    Pool comparisons, softmax, normalization, activations, bias additions and
    data movement are excluded, so this is not a total-operation/FLOP measure.
    """
    from dg_models import GaussianRelativeEncoding
    counters = {"conv1d": 0, "linear": 0, "attention": 0, "position_embedding": 0}
    handles = []
    batch = amplitude.size(0)

    def hook(module, inputs, output):
        if isinstance(module, nn.Conv1d):
            counters["conv1d"] += output.numel() * (module.in_channels // module.groups) * module.kernel_size[0]
        elif isinstance(module, nn.Linear):
            counters["linear"] += output.numel() * module.in_features
        elif isinstance(module, nn.MultiheadAttention):
            q, k, v = inputs[:3]
            assert module.batch_first and q.shape == k.shape == v.shape
            b, t, d = q.shape
            counters["attention"] += b * (4 * t * d * d + 2 * t * t * d)
        elif isinstance(module, GaussianRelativeEncoding):
            counters["position_embedding"] += inputs[0].size(1) * module.num_kernels * module.feature_dim

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear, nn.MultiheadAttention, GaussianRelativeEncoding)):
            handles.append(module.register_forward_hook(hook))
    old_fastpath = torch.backends.mha.get_fastpath_enabled()
    previous_mode = model.training
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        model.eval()
        with torch.no_grad():
            model(amplitude, project=False)
    finally:
        torch.backends.mha.set_fastpath_enabled(old_fastpath)
        model.train(previous_mode)
        for handle in handles:
            handle.remove()
    # Gaussian positional weights are computed once and shared across the batch.
    # Use batch=1 for an unambiguous per-sample inference count.
    assert batch == 1
    return {**counters, "total_counted_macs": sum(counters.values()),
            "excludes": "pool comparisons, softmax, normalization, activations, bias additions, data movement",
            "projection_head_used": False}
