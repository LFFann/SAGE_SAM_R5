from __future__ import annotations

import torch

from .correlation_propagation import correlation_propagation_loss, propagate_correlation_targets


@torch.no_grad()
def propagate_foreground_correlation_targets(
    feature_fusion: torch.Tensor,
    prob_fusion: torch.Tensor,
    targets: dict,
    sam_shape: torch.Tensor | None = None,
    resolution: int = 16,
    topk: int = 8,
    temperature: float = 0.2,
    min_weight: float = 0.15,
):
    prob_fg = prob_fusion.detach().clone()
    if prob_fg.shape[1] > 1:
        prob_fg[:, 0] = 0.0
    seed = targets.get("foreground_seed_mask")
    if seed is None:
        seed = targets.get("singleton_mask", None)
    propagated = propagate_correlation_targets(
        feature_fusion,
        prob_fg,
        sam_shape=sam_shape,
        reliable_mask=seed,
        resolution=resolution,
        topk=topk,
        temperature=temperature,
        min_weight=min_weight,
    )
    foreground = propagated["propagated_label"] > 0
    propagated["expanded_reliable_mask"] = propagated["expanded_reliable_mask"] & foreground
    propagated["propagated_weight"] = propagated["propagated_weight"] * foreground.float()
    return propagated


def foreground_correlation_loss(logits: torch.Tensor, propagated: dict):
    return correlation_propagation_loss(logits, propagated)


def masked_locality_proxy_loss(logits: torch.Tensor, targets: dict, rank_margin: float = 0.5):
    from r5.losses.tri_state_pseudo_loss import tri_state_pseudo_supervision_loss

    losses = tri_state_pseudo_supervision_loss(logits, targets, rank_margin)
    return losses["loss_fuzzy"] + losses["loss_set"]
