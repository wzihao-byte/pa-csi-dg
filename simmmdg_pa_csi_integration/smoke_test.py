from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dg_losses import DomainAwareSupConLoss  # noqa: E402
from simmmdg_losses import SharedPrivateSeparationLoss, cross_view_translation_loss  # noqa: E402
from simmmdg_models import PaCsiSimMMDG  # noqa: E402


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = {
        "time_downsample": 2,
        "max_positions": 128,
        "position_kernels": 4,
        "mcat_layers": 1,
        "mcat_heads": 2,
        "temporal_kernel_num": 8,
        "temporal_kernel_sizes": [5, 9],
        "channel_group_width": 30,
        "channel_encoder_layers": 1,
        "channel_encoder_heads": 2,
        "channel_kernel_num": 4,
        "channel_kernel_sizes": [2, 4],
        "fusion_dim": 32,
        "shared_dim": 16,
        "private_dim": 16,
        "decomposition_hidden_dim": 32,
        "projection_hidden_dim": 32,
        "projection_dim": 12,
        "translation_hidden_dim": 32,
        "dropout": 0.0,
        "antenna_layout": {
            "mode": "tx",
            "num_rx": 1,
            "num_tx": 3,
            "num_subcarriers": 30
        }
    }
    model = PaCsiSimMMDG(input_dim=90, num_classes=6, model_config=config).to(device)
    amplitude = torch.randn(6, 80, 90, device=device)
    phase = torch.randn(6, 80, 90, device=device)
    labels = torch.tensor([0, 0, 1, 1, 2, 2], device=device)
    domains = torch.tensor([0, 1, 0, 1, 0, 1], device=device)
    sample_ids = torch.arange(6, device=device)

    outputs = model(amplitude, phase)
    view_outputs = model.encode_antenna_views(amplitude, phase)
    supcon_views = torch.cat([outputs["projection"].unsqueeze(1), view_outputs["projection_views"]], dim=1)

    ce_loss = F.cross_entropy(outputs["logits"], labels)
    supcon_loss = DomainAwareSupConLoss(temperature=0.2)(supcon_views, labels, domains, sample_ids)
    separation_loss = SharedPrivateSeparationLoss()(outputs["shared_feature"], outputs["private_feature"])
    translation_loss = cross_view_translation_loss(model.shared_translator, view_outputs["shared_views"])

    payload = {
        "device": str(device),
        "logits_shape": list(outputs["logits"].shape),
        "shared_shape": list(outputs["shared_feature"].shape),
        "private_shape": list(outputs["private_feature"].shape),
        "view_shared_shape": list(view_outputs["shared_views"].shape),
        "ce_loss": float(ce_loss.item()),
        "supcon_loss": float(supcon_loss.item()),
        "separation_loss": float(separation_loss.item()),
        "translation_loss": float(translation_loss.item())
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
