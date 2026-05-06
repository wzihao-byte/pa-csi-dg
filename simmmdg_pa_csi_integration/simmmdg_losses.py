from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class SharedPrivateSeparationLoss(nn.Module):
    """Orthogonality-style separation between shared and private components."""

    def forward(self, shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
        if shared.shape[:-1] != private.shape[:-1]:
            raise ValueError(
                "Shared and private features must share all leading dimensions; "
                f"received {shared.shape} and {private.shape}."
            )

        flat_shared = shared.reshape(-1, shared.shape[-1])
        flat_private = private.reshape(-1, private.shape[-1])
        if flat_shared.size(0) == 0:
            return shared.new_zeros(())

        if flat_shared.size(1) == flat_private.size(1):
            shared_norm = F.normalize(flat_shared, dim=-1)
            private_norm = F.normalize(flat_private, dim=-1)
            return (shared_norm * private_norm).sum(dim=-1).pow(2).mean()

        if flat_shared.size(0) < 2:
            return shared.new_zeros(())
        shared_centered = flat_shared - flat_shared.mean(dim=0, keepdim=True)
        private_centered = flat_private - flat_private.mean(dim=0, keepdim=True)
        covariance = torch.matmul(shared_centered.T, private_centered) / float(flat_shared.size(0) - 1)
        return covariance.pow(2).mean()


def cross_view_translation_loss(
    translator: Callable[[torch.Tensor], torch.Tensor],
    shared_views: torch.Tensor,
    detach_target: bool = False,
) -> torch.Tensor:
    """Translate each natural view's shared feature into every other view."""

    if shared_views.dim() != 3:
        raise ValueError(f"Expected shared_views with shape [batch, views, dim], received {shared_views.shape}.")
    if shared_views.size(1) < 2:
        return shared_views.new_zeros(())

    losses = []
    num_views = shared_views.size(1)
    for source_index in range(num_views):
        source = shared_views[:, source_index, :]
        predicted = F.normalize(translator(source), dim=-1)
        for target_index in range(num_views):
            if target_index == source_index:
                continue
            target = shared_views[:, target_index, :]
            if detach_target:
                target = target.detach()
            target = F.normalize(target, dim=-1)
            losses.append(torch.norm(predicted - target, p=2, dim=-1).mean())

    if not losses:
        return shared_views.new_zeros(())
    return torch.stack(losses).mean()
