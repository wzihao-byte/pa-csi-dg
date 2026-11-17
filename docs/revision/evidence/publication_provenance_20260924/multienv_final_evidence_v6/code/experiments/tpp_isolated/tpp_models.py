from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from dg_models import (
    AntennaViewBuilder,
    ChannelContextEncoder,
    GaussianRelativeEncoding,
    GatedResidualFusion,
    MCATEncoder,
    ProjectionHead,
    compatible_heads,
)


class IsolatedTemporalPyramidPool(nn.Module):
    """Conv1D responses followed by configurable temporal pyramid pooling.

    The default settings reproduce the original isolated TPP implementation.
    C3-recovery experiments can either preserve the shift-invariant global-max
    feature as a residual or blend max and average statistics inside each bin.
    """

    def __init__(
        self,
        input_dim: int,
        output_channels: int,
        kernel_sizes: Sequence[int],
        levels: Sequence[int],
        dropout: float = 0.1,
        pool_type: str = "max",
        mixed_max_weight: float = 0.5,
        preserve_global: bool = False,
        pyramid_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if not kernel_sizes:
            raise ValueError("kernel_sizes must contain at least one value.")
        if not levels:
            raise ValueError("levels must contain at least one value.")

        self.kernel_sizes = [int(kernel_size) for kernel_size in kernel_sizes]
        self.levels = [int(level) for level in levels]
        if any(kernel_size <= 0 for kernel_size in self.kernel_sizes):
            raise ValueError(f"All kernel sizes must be positive, received {self.kernel_sizes}.")
        if any(level <= 0 for level in self.levels):
            raise ValueError(f"All TPP levels must be positive, received {self.levels}.")

        self.pool_type = str(pool_type).lower()
        if self.pool_type not in {"max", "avg", "mixed"}:
            raise ValueError(
                f"Unsupported TPP pool_type '{pool_type}'. Choose from ['max', 'avg', 'mixed']."
            )
        self.mixed_max_weight = float(mixed_max_weight)
        if not 0.0 <= self.mixed_max_weight <= 1.0:
            raise ValueError(
                "mixed_max_weight must be in [0, 1], "
                f"received {self.mixed_max_weight}."
            )
        self.preserve_global = bool(preserve_global)
        self.pyramid_scale = float(pyramid_scale)
        if self.pyramid_scale < 0.0:
            raise ValueError(f"pyramid_scale must be non-negative, received {self.pyramid_scale}.")

        self.convs = nn.ModuleList(
            [nn.Conv1d(input_dim, output_channels, kernel_size=kernel_size) for kernel_size in self.kernel_sizes]
        )
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.segment_count = sum(self.levels)
        self.pyramid_dim = output_channels * len(self.kernel_sizes) * self.segment_count
        self.output_dim = output_channels * len(self.kernel_sizes)
        self.projection = nn.Linear(self.pyramid_dim, self.output_dim)

    def pool_response(self, response: torch.Tensor, level: int) -> torch.Tensor:
        if self.pool_type == "max":
            return F.adaptive_max_pool1d(response, output_size=level)
        if self.pool_type == "avg":
            return F.adaptive_avg_pool1d(response, output_size=level)
        max_pooled = F.adaptive_max_pool1d(response, output_size=level)
        avg_pooled = F.adaptive_avg_pool1d(response, output_size=level)
        return self.mixed_max_weight * max_pooled + (1.0 - self.mixed_max_weight) * avg_pooled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        pooled = []
        global_max = []
        for conv in self.convs:
            response = self.activation(conv(x_t))
            if self.preserve_global:
                global_max.append(F.adaptive_max_pool1d(response, output_size=1).flatten(start_dim=1))
            for level in self.levels:
                pooled.append(self.pool_response(response, level).flatten(start_dim=1))

        pyramid = torch.cat(pooled, dim=1)
        pyramid_feature = self.projection(pyramid)
        if self.preserve_global:
            # Keep the original global-max detector as the primary feature and
            # let the position-sensitive pyramid act only as a correction.
            pyramid_feature = torch.cat(global_max, dim=1) + self.pyramid_scale * pyramid_feature
        return self.dropout(pyramid_feature)


class IsolatedPaCsiStreamEncoderTPP(nn.Module):
    """Copy of the stream encoder structure with local TPP replacing global temporal pooling."""

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

        tpp_levels = list(model_config.get("tpp_levels", [1, 2, 4]))
        tpp_pool_type = str(model_config.get("tpp_pool_type", "max"))
        tpp_mixed_max_weight = float(model_config.get("tpp_mixed_max_weight", 0.5))
        tpp_preserve_global = bool(model_config.get("tpp_preserve_global", False))
        tpp_pyramid_scale = float(model_config.get("tpp_pyramid_scale", 1.0))
        self.temporal_pool = IsolatedTemporalPyramidPool(
            input_dim=input_dim,
            output_channels=int(model_config.get("temporal_kernel_num", 128)),
            kernel_sizes=list(model_config.get("temporal_kernel_sizes", [20, 40])),
            levels=tpp_levels,
            dropout=float(model_config.get("dropout", 0.1)),
            pool_type=tpp_pool_type,
            mixed_max_weight=tpp_mixed_max_weight,
            preserve_global=tpp_preserve_global,
            pyramid_scale=tpp_pyramid_scale,
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
            self.channel_pool = IsolatedTemporalPyramidPool(
                input_dim=group_width,
                output_channels=int(model_config.get("channel_kernel_num", 16)),
                kernel_sizes=list(model_config.get("channel_kernel_sizes", [2, 4])),
                levels=list(model_config.get("channel_tpp_levels", tpp_levels)),
                dropout=float(model_config.get("dropout", 0.1)),
                pool_type=str(model_config.get("channel_tpp_pool_type", tpp_pool_type)),
                mixed_max_weight=float(
                    model_config.get("channel_tpp_mixed_max_weight", tpp_mixed_max_weight)
                ),
                preserve_global=bool(
                    model_config.get("channel_tpp_preserve_global", tpp_preserve_global)
                ),
                pyramid_scale=float(
                    model_config.get("channel_tpp_pyramid_scale", tpp_pyramid_scale)
                ),
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


class IsolatedPaCsiDGLiteTPP(nn.Module):
    """Isolated PaCsiDGLite variant whose stream encoders use TPP."""

    def __init__(self, input_dim: int, num_classes: int, model_config: Mapping[str, Any]) -> None:
        super().__init__()
        self.amp_encoder = IsolatedPaCsiStreamEncoderTPP(input_dim, model_config)
        self.phase_encoder = IsolatedPaCsiStreamEncoderTPP(input_dim, model_config)

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
