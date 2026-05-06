from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianRelativeEncoding(nn.Module):
    def __init__(self, feature_dim: int, max_positions: int = 2048, num_kernels: int = 10) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.max_positions = max_positions
        self.num_kernels = num_kernels

        self.embedding = nn.Parameter(torch.empty(num_kernels, feature_dim))
        nn.init.xavier_uniform_(self.embedding)

        interval = max_positions / max(1, num_kernels)
        initial_mu = torch.linspace(0.0, max_positions - interval, steps=num_kernels)
        self.mu = nn.Parameter(initial_mu)
        self.log_sigma = nn.Parameter(torch.full((num_kernels,), math.log(50.0)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)
        mu = self.mu.to(dtype=x.dtype).unsqueeze(0)
        sigma = self.log_sigma.exp().to(dtype=x.dtype).unsqueeze(0).clamp_min(1.0)
        weights = torch.softmax(-((positions - mu) ** 2) / (2.0 * sigma**2), dim=1)
        pos_encoding = torch.matmul(weights, self.embedding.to(dtype=x.dtype))
        return x + pos_encoding.unsqueeze(0)


class HarCNNFeedForward(nn.Module):
    def __init__(self, feature_dim: int, kernels: list[int], dropout: float = 0.1) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(feature_dim, feature_dim, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm1d(feature_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for kernel_size in kernels
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        outputs = [branch(x_t) for branch in self.branches]
        merged = torch.stack(outputs, dim=0).mean(dim=0)
        return merged.transpose(1, 2)


class MCATBlock(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(feature_dim)
        self.self_attention = nn.MultiheadAttention(feature_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn_norm = nn.LayerNorm(feature_dim)
        self.feed_forward = HarCNNFeedForward(feature_dim, kernels=[1, 3, 5], dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(x)
        attn_output, _ = self.self_attention(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.dropout(attn_output)
        ffn_input = self.ffn_norm(x)
        return x + self.dropout(self.feed_forward(ffn_input))


class MCATEncoder(nn.Module):
    def __init__(self, feature_dim: int, num_layers: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [MCATBlock(feature_dim, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


class MultiScaleTemporalPool(nn.Module):
    def __init__(self, input_dim: int, output_channels: int, kernel_sizes: list[int], dropout: float = 0.1) -> None:
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Conv1d(input_dim, output_channels, kernel_size=kernel_size) for kernel_size in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.output_dim = output_channels * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        pooled = []
        for conv in self.convs:
            features = self.activation(conv(x_t))
            pooled.append(F.max_pool1d(features, kernel_size=features.size(-1)).squeeze(-1))
        return self.dropout(torch.cat(pooled, dim=1))


def compatible_heads(feature_dim: int, requested_heads: int) -> int:
    for candidate in range(requested_heads, 0, -1):
        if feature_dim % candidate == 0:
            return candidate
    return 1


class ChannelContextEncoder(nn.Module):
    def __init__(self, feature_dim: int, num_layers: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=compatible_heads(feature_dim, num_heads),
            dim_feedforward=feature_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(x))


class PaCsiStreamEncoder(nn.Module):
    def __init__(self, input_dim: int, model_config: Mapping[str, Any]) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.time_downsample = int(model_config.get("time_downsample", 2))
        self.position_encoding = GaussianRelativeEncoding(
            input_dim,
            max_positions=int(model_config.get("max_positions", 2048)),
            num_kernels=int(model_config.get("position_kernels", 10)),
        )
        self.temporal_encoder = MCATEncoder(
            input_dim,
            num_layers=int(model_config.get("mcat_layers", 2)),
            num_heads=compatible_heads(input_dim, int(model_config.get("mcat_heads", 6))),
            dropout=float(model_config.get("dropout", 0.1)),
        )
        self.temporal_pool = MultiScaleTemporalPool(
            input_dim=input_dim,
            output_channels=int(model_config.get("temporal_kernel_num", 128)),
            kernel_sizes=list(model_config.get("temporal_kernel_sizes", [20, 40])),
            dropout=float(model_config.get("dropout", 0.1)),
        )

        group_width = int(model_config.get("channel_group_width", 30))
        self.channel_group_width = group_width
        self.num_groups = input_dim // group_width if input_dim % group_width == 0 else 0
        self.use_channel_path = self.num_groups > 0
        if self.use_channel_path:
            self.channel_encoder = ChannelContextEncoder(
                feature_dim=group_width,
                num_layers=int(model_config.get("channel_encoder_layers", 1)),
                num_heads=int(model_config.get("channel_encoder_heads", 6)),
                dropout=float(model_config.get("dropout", 0.1)),
            )
            self.channel_pool = MultiScaleTemporalPool(
                input_dim=group_width,
                output_channels=int(model_config.get("channel_kernel_num", 16)),
                kernel_sizes=list(model_config.get("channel_kernel_sizes", [2, 4])),
                dropout=float(model_config.get("dropout", 0.1)),
            )
            self.output_dim = self.temporal_pool.output_dim + self.channel_pool.output_dim
        else:
            self.output_dim = self.temporal_pool.output_dim

    def temporal_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.time_downsample <= 1:
            return x
        trimmed_steps = (x.size(1) // self.time_downsample) * self.time_downsample
        if trimmed_steps == 0:
            raise ValueError(
                f"Input time steps {x.size(1)} are shorter than time_downsample={self.time_downsample}."
            )
        x = x[:, :trimmed_steps, :]
        batch_size, _, feature_dim = x.shape
        return x.reshape(batch_size, trimmed_steps // self.time_downsample, self.time_downsample, feature_dim).mean(dim=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.temporal_reduce(x)
        x = self.position_encoding(x)
        temporal_features = self.temporal_encoder(x)
        pooled_features = [self.temporal_pool(temporal_features)]

        if self.use_channel_path:
            channel_context = x.reshape(x.size(0), x.size(1), self.num_groups, self.channel_group_width).mean(dim=2)
            channel_context = self.channel_encoder(channel_context)
            pooled_features.append(self.channel_pool(channel_context))

        return torch.cat(pooled_features, dim=1)


class GatedResidualFusion(nn.Module):
    def __init__(self, input_dim: int, fusion_dim: int) -> None:
        super().__init__()
        self.amp_proj = nn.Linear(input_dim, fusion_dim)
        self.phase_proj = nn.Linear(input_dim, fusion_dim)
        self.gate_proj = nn.Linear(fusion_dim, fusion_dim)
        self.norm = nn.LayerNorm(fusion_dim)

    def forward(self, amp_feature: torch.Tensor, phase_feature: torch.Tensor) -> torch.Tensor:
        amp_hidden = self.amp_proj(amp_feature)
        phase_hidden = self.phase_proj(phase_feature)
        hidden = self.norm(F.elu(amp_hidden + phase_hidden))
        gate = torch.sigmoid(self.gate_proj(hidden))
        return gate * amp_hidden + (1.0 - gate) * phase_hidden


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AntennaViewBuilder(nn.Module):
    def __init__(self, input_dim: int, layout_config: Mapping[str, Any]) -> None:
        super().__init__()
        self.mode = str(layout_config.get("mode", "rx")).lower()
        self.num_rx = int(layout_config.get("num_rx", 3))
        self.num_tx = int(layout_config.get("num_tx", 3))
        self.num_subcarriers = int(layout_config.get("num_subcarriers", 30))
        expected_dim = self.num_rx * self.num_tx * self.num_subcarriers
        if input_dim != expected_dim:
            raise ValueError(
                f"NARC antenna layout expects input_dim={expected_dim} from "
                f"{self.num_rx}x{self.num_tx}x{self.num_subcarriers}, received {input_dim}."
            )
        self.input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, time_steps, _ = x.shape
        structured = x.reshape(batch_size, time_steps, self.num_rx, self.num_tx, self.num_subcarriers)
        views = []

        if self.mode == "rx":
            for rx_index in range(self.num_rx):
                masked = torch.zeros_like(structured)
                masked[:, :, rx_index, :, :] = structured[:, :, rx_index, :, :]
                views.append(masked.reshape(batch_size, time_steps, self.input_dim))
        elif self.mode == "tx":
            for tx_index in range(self.num_tx):
                masked = torch.zeros_like(structured)
                masked[:, :, :, tx_index, :] = structured[:, :, :, tx_index, :]
                views.append(masked.reshape(batch_size, time_steps, self.input_dim))
        elif self.mode == "link":
            for rx_index in range(self.num_rx):
                for tx_index in range(self.num_tx):
                    masked = torch.zeros_like(structured)
                    masked[:, :, rx_index, tx_index, :] = structured[:, :, rx_index, tx_index, :]
                    views.append(masked.reshape(batch_size, time_steps, self.input_dim))
        else:
            raise ValueError(f"Unsupported antenna view mode '{self.mode}'. Choose from ['rx', 'tx', 'link'].")

        return torch.stack(views, dim=1)


class PaCsiDGLite(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, model_config: Mapping[str, Any]) -> None:
        super().__init__()
        self.amp_encoder = PaCsiStreamEncoder(input_dim, model_config)
        self.phase_encoder = PaCsiStreamEncoder(input_dim, model_config)

        fusion_dim = int(model_config.get("fusion_dim", 256))
        self.fusion = GatedResidualFusion(self.amp_encoder.output_dim, fusion_dim)
        self.classifier = nn.Linear(fusion_dim, num_classes)
        self.projection_head = ProjectionHead(
            input_dim=fusion_dim,
            hidden_dim=int(model_config.get("projection_hidden_dim", fusion_dim)),
            output_dim=int(model_config.get("projection_dim", 128)),
        )

        antenna_layout = model_config.get("antenna_layout")
        self.view_builder = AntennaViewBuilder(input_dim, antenna_layout) if antenna_layout else None
        self.fusion_dim = fusion_dim
        self.num_classes = num_classes

        self.projection_dim = int(model_config.get("projection_dim", 128))
        prototype_num_subcenters = int(model_config.get("prototype_num_subcenters", 0))
        self.prototype_num_subcenters = prototype_num_subcenters
        if prototype_num_subcenters > 0:
            prototypes = torch.empty(num_classes, prototype_num_subcenters, self.projection_dim)
            nn.init.xavier_uniform_(prototypes)
            self.class_prototypes = nn.Parameter(prototypes)
        else:
            self.class_prototypes = None

    def has_class_prototypes(self) -> bool:
        return isinstance(self.class_prototypes, nn.Parameter) and self.prototype_num_subcenters > 0

    def encode_features(self, amplitude: torch.Tensor, phase: torch.Tensor) -> Dict[str, torch.Tensor]:
        amp_feature = self.amp_encoder(amplitude)
        phase_feature = self.phase_encoder(phase)
        fused_feature = self.fusion(amp_feature, phase_feature)
        return {
            "amp_feature": amp_feature,
            "phase_feature": phase_feature,
            "fused_feature": fused_feature,
        }

    def forward(self, amplitude: torch.Tensor, phase: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.encode_features(amplitude, phase)
        fused_feature = features["fused_feature"]
        logits = self.classifier(fused_feature)
        projection = F.normalize(self.projection_head(fused_feature), dim=-1)
        features.update({"logits": logits, "projection": projection})
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
        projection_views = F.normalize(self.projection_head(fused_views.reshape(batch_size * num_views, -1)), dim=-1)
        projection_views = projection_views.reshape(batch_size, num_views, -1)
        return {
            "amp_views": amp_views,
            "phase_views": phase_views,
            "fused_views": fused_views,
            "projection_views": projection_views,
        }
