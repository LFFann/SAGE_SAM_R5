from __future__ import annotations

import torch

from .foreground_participation_controller import apply_foreground_budget
from .sam_structural_support import build_sam_structural_support


def _threshold_vec(calibrator, attr: str, default: float, num_classes: int, device, dtype):
    value = getattr(calibrator, attr, None)
    if value is None:
        return torch.full((num_classes,), float(default), device=device, dtype=dtype)
    value = value.to(device=device, dtype=dtype)
    if value.numel() != num_classes:
        return torch.full((num_classes,), float(default), device=device, dtype=dtype)
    return value


def _foreground_classes(config: dict, num_classes: int) -> list[int]:
    return [int(c) for c in config.get("foreground_classes", list(range(1, num_classes))) if 0 < int(c) < num_classes]


def build_foreground_safe_targets(teacher_out: dict, sam_out: dict | None, calibrator, config: dict):
    teacher_prob = teacher_out["mean_prob"].detach()
    if teacher_prob.ndim != 4:
        raise ValueError(f"teacher mean_prob must be BCHW, got {tuple(teacher_prob.shape)}")
    device = teacher_prob.device
    dtype = teacher_prob.dtype
    bsz, num_classes, height, width = teacher_prob.shape
    fg_classes = _foreground_classes(config, num_classes)
    iter_now = int(config.get("_iteration", 0))
    disable_bg_until = int(config.get("disable_background_unsup_until", config.get("foreground_grounding_start", 1200)))
    use_background_hard = iter_now >= disable_bg_until

    teacher_thresh = _threshold_vec(calibrator, "teacher_q", config.get("min_teacher_confidence", 0.5), num_classes, device, dtype)
    sam_thresh = _threshold_vec(calibrator, "sam_q", config.get("min_sam_confidence", 0.5), num_classes, device, dtype)
    sam_struct = build_sam_structural_support(
        sam_out,
        teacher_prob,
        foreground_classes=fg_classes,
        min_support=float(config.get("min_sam_structural_support", 0.0)),
    )
    sam_support = sam_struct["support"]
    sam_valid = bool(sam_struct["valid"])
    fg_support = sam_struct["foreground_support"]

    teacher_candidate, teacher_low = calibrator.prediction_sets(teacher_prob)
    candidate_set = torch.zeros_like(teacher_candidate, dtype=torch.bool)
    foreground_score = teacher_prob * sam_support

    min_sam_conf = float(config.get("min_sam_confidence", 0.5))
    min_teacher_conf = float(config.get("min_teacher_confidence", 0.5))
    min_fg_score = float(config.get("min_foreground_score", 0.02))
    for cls in fg_classes:
        teacher_fg = teacher_candidate[:, cls] | (teacher_prob[:, cls] >= min_teacher_conf)
        if sam_valid:
            sam_fg = sam_support[:, cls] >= min_sam_conf
            candidate_set[:, cls] = teacher_fg | sam_fg
        else:
            candidate_set[:, cls] = teacher_fg

    reliable_fg = torch.zeros_like(candidate_set)
    for cls in fg_classes:
        if sam_valid:
            reliable_fg[:, cls] = (
                (teacher_prob[:, cls] >= torch.maximum(teacher_thresh[cls], teacher_prob.new_tensor(min_teacher_conf)))
                & (sam_support[:, cls] >= torch.maximum(sam_thresh[cls], teacher_prob.new_tensor(min_sam_conf)))
                & (foreground_score[:, cls] >= min_fg_score)
            )
        else:
            reliable_fg[:, cls] = teacher_prob[:, cls] >= min_teacher_conf

    reliable_fg_any = reliable_fg[:, 1:].any(dim=1) if num_classes > 1 else torch.zeros((bsz, height, width), device=device, dtype=torch.bool)
    fg_score_max, fg_label = foreground_score[:, 1:].max(dim=1) if num_classes > 1 else (teacher_prob.new_zeros((bsz, height, width)), torch.zeros((bsz, height, width), device=device, dtype=torch.long))
    fg_label = fg_label + 1
    teacher_conf, teacher_label = teacher_prob.max(dim=1)
    fallback_fg_label = torch.where(teacher_label > 0, teacher_label, fg_label)
    singleton_label = torch.where(reliable_fg_any, fallback_fg_label, teacher_label)

    fg_low = float(config.get("sam_foreground_low", 0.15))
    bg_thresh = float(config.get("background_confidence", config.get("min_teacher_confidence", 0.5)))
    has_fg_candidate = candidate_set[:, 1:].any(dim=1) if num_classes > 1 else torch.zeros_like(teacher_label, dtype=torch.bool)
    reliable_background = (
        use_background_hard
        & (teacher_prob[:, 0] >= bg_thresh)
        & (fg_support < fg_low)
        & ~has_fg_candidate
        & ~reliable_fg_any
    )
    candidate_set[:, 0] = reliable_background
    singleton_label = torch.where(reliable_background, torch.zeros_like(singleton_label), singleton_label)
    singleton_mask = reliable_fg_any | reliable_background

    max_set = int(config.get("max_candidate_set_size", 2))
    if max_set > 0 and num_classes > 1:
        score_for_topk = teacher_prob.clone()
        support_boost = float(config.get("sam_fuzzy_support_weight", 0.25))
        score_for_topk[:, 1:] = torch.maximum(score_for_topk[:, 1:], sam_support[:, 1:] * support_boost)
        _, topi = score_for_topk.topk(k=min(max_set, num_classes), dim=1)
        top_candidate = torch.zeros_like(candidate_set)
        top_candidate.scatter_(1, topi, True)
        candidate_set = candidate_set & top_candidate
        candidate_set[:, 0] = candidate_set[:, 0] & reliable_background

    empty = candidate_set.sum(dim=1) == 0
    if empty.any():
        candidate_set[:, 0] = candidate_set[:, 0] | empty
    candidate_count = candidate_set.sum(dim=1)
    ambiguous_mask = ((candidate_count > 1) | (~singleton_mask & has_fg_candidate) | teacher_low) & (candidate_count > 0)
    ambiguous_mask = ambiguous_mask & ~reliable_fg_any
    conflict_mask = (teacher_label == 0) & has_fg_candidate & ~reliable_background

    singleton_label, singleton_mask, candidate_set, ambiguous_mask, budget_stats = apply_foreground_budget(
        singleton_label=singleton_label,
        singleton_mask=singleton_mask,
        candidate_set=candidate_set,
        ambiguous_mask=ambiguous_mask,
        foreground_score=torch.maximum(foreground_score, teacher_prob * candidate_set.float()),
        teacher_prob=teacher_prob,
        config=config,
    )

    negative_thresh = float(config.get("safe_negative_threshold", 0.05))
    negative_sam_thresh = float(config.get("safe_negative_sam_threshold", negative_thresh))
    negative_set = torch.zeros_like(candidate_set, dtype=torch.bool)
    for cls in fg_classes:
        negative_set[:, cls] = (teacher_prob[:, cls] < negative_thresh) & (sam_support[:, cls] < negative_sam_thresh) & ~candidate_set[:, cls]
    negative_mask = negative_set.any(dim=1) | conflict_mask

    soft_score = teacher_prob.clone()
    soft_score[:, 1:] = torch.maximum(soft_score[:, 1:], sam_support[:, 1:] * float(config.get("sam_fuzzy_support_weight", 0.25)))
    soft_score = soft_score * candidate_set.float()
    soft_empty = soft_score.sum(dim=1, keepdim=True) <= 1e-6
    soft_score = torch.where(soft_empty, teacher_prob, soft_score)
    soft_target = soft_score / soft_score.sum(dim=1, keepdim=True).clamp_min(1e-6)
    teacher_only_soft_target = teacher_prob / teacher_prob.sum(dim=1, keepdim=True).clamp_min(1e-6)

    candidate_weight = torch.maximum(teacher_conf, fg_support).clamp(0.05, 1.0)
    safe_negative_weight = negative_mask.float().clamp(0.0, 1.0)
    foreground_seed = reliable_fg.bool()
    foreground_seed_mask = foreground_seed[:, 1:].any(dim=1) if num_classes > 1 else torch.zeros_like(singleton_mask)
    fuzzy_region = ambiguous_mask & (candidate_set[:, 1:].any(dim=1) if num_classes > 1 else ambiguous_mask)
    semantic_gate = singleton_mask | ambiguous_mask
    sam_train_gate = foreground_seed_mask & bool(sam_valid)
    structure_gate = (fg_support >= fg_low) | foreground_seed_mask | fuzzy_region
    sam_weight = (foreground_score[:, 1:].max(dim=1).values if (sam_valid and num_classes > 1) else teacher_prob.new_zeros((bsz, height, width))).clamp(0.0, 1.0)
    structure_weight = torch.maximum(sam_weight, fg_support).clamp(0.0, 1.0)

    per_class_participation = [0.0 for _ in range(num_classes)]
    per_class_foreground_participation = [0.0 for _ in range(num_classes)]
    per_class_safe_negative = [0.0 for _ in range(num_classes)]
    for cls in fg_classes:
        fg_participates = foreground_seed[:, cls] | (fuzzy_region & candidate_set[:, cls])
        per_class_foreground_participation[cls] = float(fg_participates.float().mean().detach())
        if sam_valid:
            per_class_participation[cls] = float(((sam_support[:, cls] >= min_sam_conf) & fg_participates).float().mean().detach())
        per_class_safe_negative[cls] = float(negative_set[:, cls].float().mean().detach())

    stats = {
        "singleton_ratio": float(singleton_mask.float().mean().detach()),
        "singleton_pixel_ratio": float(singleton_mask.float().mean().detach()),
        "ambiguous_ratio": float(ambiguous_mask.float().mean().detach()),
        "ambiguous_pixel_ratio": float(ambiguous_mask.float().mean().detach()),
        "conflict_ratio": float(conflict_mask.float().mean().detach()),
        "negative_ratio": float(negative_mask.float().mean().detach()),
        "safe_negative_pixel_ratio": float(negative_set.any(dim=1).float().mean().detach()),
        "per_class_safe_negative_ratio": per_class_safe_negative,
        "avg_set_size": float(candidate_set.float().sum(dim=1).mean().detach()),
        "sam_semantic_gate_ratio": float(semantic_gate.float().mean().detach()),
        "sam_structure_gate_ratio": float(structure_gate.float().mean().detach()),
        "sam_train_gate_ratio": float(sam_train_gate.float().mean().detach()),
        "sam_soft_weight_mean": float(sam_weight.mean().detach()),
        "sam_soft_weight_p25": float(torch.quantile(sam_weight.detach().float().reshape(-1).cpu(), 0.25)),
        "sam_soft_weight_p50": float(torch.quantile(sam_weight.detach().float().reshape(-1).cpu(), 0.50)),
        "sam_soft_weight_p75": float(torch.quantile(sam_weight.detach().float().reshape(-1).cpu(), 0.75)),
        "sam_participation_ratio": float(sam_train_gate.float().mean().detach()),
        "per_class_sam_participation_ratio": per_class_participation,
        "per_class_foreground_participation_ratio": per_class_foreground_participation,
        "sam_teacher_agreement": float(((teacher_label == singleton_label) | ambiguous_mask).float().mean().detach()),
        "sam_foreground_support_ratio": float((fg_support >= fg_low).float().mean().detach()),
        **budget_stats,
    }
    for cls in fg_classes:
        stats[f"safe_negative_ratio_class{cls}"] = per_class_safe_negative[cls]

    return {
        "singleton_label": singleton_label.detach(),
        "singleton_mask": singleton_mask.detach(),
        "candidate_set": candidate_set.detach(),
        "candidate_weight": candidate_weight.detach(),
        "ambiguous_mask": ambiguous_mask.detach(),
        "fuzzy_region": fuzzy_region.detach(),
        "conflict_mask": conflict_mask.detach(),
        "negative_set": negative_set.detach(),
        "safe_negative_set": negative_set.detach(),
        "negative_mask": negative_mask.detach(),
        "safe_negative_weight": safe_negative_weight.detach(),
        "semantic_gate": semantic_gate.detach(),
        "sam_train_gate": sam_train_gate.detach(),
        "structure_gate": structure_gate.detach(),
        "sam_weight": sam_weight.detach(),
        "teacher_weight": teacher_conf.detach(),
        "semantic_weight": candidate_weight.detach(),
        "structure_weight": structure_weight.detach(),
        "teacher_reliable_mask": foreground_seed_mask.detach(),
        "foreground_seed": foreground_seed.detach(),
        "foreground_seed_mask": foreground_seed_mask.detach(),
        "sam_support": sam_support.detach(),
        "sam_foreground_support": fg_support.detach(),
        "sam_boundary": sam_struct["boundary"].detach(),
        "reliable_background_mask": reliable_background.detach(),
        "soft_target": soft_target.detach(),
        "teacher_only_soft_target": teacher_only_soft_target.detach(),
        "stats": stats,
    }


def build_set_valued_targets(teacher_out: dict, sam_out: dict | None, calibrator, config: dict):
    return build_foreground_safe_targets(teacher_out, sam_out, calibrator, config)
