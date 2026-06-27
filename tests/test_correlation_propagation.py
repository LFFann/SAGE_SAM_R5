from __future__ import annotations

import torch

from r6.ssl.correlation_propagation import correlation_propagation_loss, propagate_correlation_targets
from r6.ssl.foreground_correlation_locality import build_masked_locality_view, expand_targets_with_correlation


def test_correlation_propagation_returns_dense_training_signal():
    feature = torch.randn(2, 4, 8, 8)
    prob = torch.softmax(torch.randn(2, 3, 32, 32), dim=1)
    reliable = torch.zeros(2, 32, 32, dtype=torch.bool)
    reliable[:, 8:16, 8:16] = True
    sam_shape = torch.ones(2, 1, 32, 32)

    propagated = propagate_correlation_targets(
        feature,
        prob,
        sam_shape=sam_shape,
        reliable_mask=reliable,
        resolution=8,
        topk=4,
        min_weight=0.05,
    )

    assert propagated["propagated_label"].shape == (2, 32, 32)
    assert propagated["propagated_weight"].shape == (2, 32, 32)
    assert propagated["expanded_reliable_mask"].shape == (2, 32, 32)
    assert propagated["propagated_weight"].mean() > 0


def test_correlation_propagation_loss_backward():
    logits = torch.randn(1, 3, 16, 16, requires_grad=True)
    propagated = {
        "propagated_label": torch.zeros(1, 16, 16, dtype=torch.long),
        "propagated_weight": torch.ones(1, 16, 16),
        "expanded_reliable_mask": torch.ones(1, 16, 16, dtype=torch.bool),
    }

    loss = correlation_propagation_loss(logits, propagated)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None


def test_masked_locality_view_masks_only_foreground_seed_pixels():
    image = torch.ones(1, 3, 16, 16)
    seed = torch.zeros(1, 16, 16, dtype=torch.bool)
    seed[:, 4:12, 4:12] = True

    masked, stats = build_masked_locality_view(image, seed, mask_ratio=1.0, patch_size=4, fill="zero")

    changed = (masked != image).any(dim=1)
    assert changed.sum() > 0
    assert torch.all(seed[changed])
    assert stats["masked_locality_ratio"] > 0.0
    assert stats["foreground_masked_ratio"] > 0.0


def test_masked_locality_view_no_seed_is_noop():
    image = torch.randn(1, 3, 8, 8)
    masked, stats = build_masked_locality_view(image, torch.zeros(1, 8, 8, dtype=torch.bool))

    assert torch.equal(masked, image)
    assert stats["masked_locality_ratio"] == 0.0


def test_correlation_expands_foreground_candidate_set():
    candidate_set = torch.zeros(1, 3, 4, 4, dtype=torch.bool)
    targets = {
        "candidate_set": candidate_set,
        "candidate_weight": torch.zeros(1, 4, 4),
        "ambiguous_mask": torch.zeros(1, 4, 4, dtype=torch.bool),
        "singleton_mask": torch.zeros(1, 4, 4, dtype=torch.bool),
        "negative_set": torch.zeros(1, 3, 4, 4, dtype=torch.bool),
        "conflict_mask": torch.zeros(1, 4, 4, dtype=torch.bool),
        "teacher_only_soft_target": torch.full((1, 3, 4, 4), 1.0 / 3.0),
        "sam_train_gate": torch.zeros(1, 4, 4, dtype=torch.bool),
        "structure_gate": torch.zeros(1, 4, 4, dtype=torch.bool),
        "sam_weight": torch.zeros(1, 4, 4),
        "structure_weight": torch.zeros(1, 4, 4),
        "stats": {},
    }
    propagated = {
        "propagated_label": torch.ones(1, 4, 4, dtype=torch.long),
        "propagated_weight": torch.ones(1, 4, 4),
        "expanded_reliable_mask": torch.ones(1, 4, 4, dtype=torch.bool),
    }

    expanded = expand_targets_with_correlation(targets, propagated, min_weight=0.15)

    assert expanded["candidate_set"][:, 1].all()
    assert expanded["fuzzy_region"].any()
    assert expanded["sam_region_gate"].any()
    assert expanded["stats"]["foreground_propagated_ratio"] == 1.0
