from __future__ import annotations

import torch
import torch.nn.functional as F


def _resize_like(tensor: torch.Tensor, ref: torch.Tensor, mode: str = "bilinear") -> torch.Tensor:
    if tensor.shape[-2:] == ref.shape[-2:]:
        return tensor
    if mode == "nearest":
        return F.interpolate(tensor, size=ref.shape[-2:], mode=mode)
    return F.interpolate(tensor, size=ref.shape[-2:], mode=mode, align_corners=False)


def _class_scores_to_spatial(scores: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError(f"class scores must be shaped BxC, got {tuple(scores.shape)}")
    return scores[:, :, None, None].expand(-1, -1, height, width)


def build_sam_structural_support(
    sam_out: dict | None,
    teacher_prob: torch.Tensor,
    foreground_classes: list[int] | tuple[int, ...] | None = None,
    min_support: float = 0.0,
) -> dict:
    """Convert SAM outputs into foreground-only structural support.

    SAM is deliberately not allowed to create a background channel here.  The
    returned support tensor is C-channel for shape compatibility, but channel 0
    is always zero and downstream code must derive background from the task
    teacher plus foreground exclusion.
    """

    if teacher_prob.ndim != 4:
        raise ValueError(f"teacher_prob must be BCHW, got {tuple(teacher_prob.shape)}")
    device = teacher_prob.device
    dtype = teacher_prob.dtype
    bsz, num_classes, height, width = teacher_prob.shape
    fg_classes = list(foreground_classes or range(1, num_classes))
    support = teacher_prob.new_zeros((bsz, num_classes, height, width))
    boundary = teacher_prob.new_zeros((bsz, 1, height, width))
    valid = bool(sam_out and sam_out.get("valid") and sam_out.get("sam_prob") is not None)
    if not valid:
        return {
            "valid": False,
            "support": support,
            "foreground_support": support[:, 1:].max(dim=1).values if num_classes > 1 else support[:, 0],
            "boundary": boundary,
        }

    sam_prob = sam_out["sam_prob"].detach().to(device=device, dtype=dtype)
    if sam_prob.ndim != 4:
        raise ValueError(f"sam_prob must be BCHW, got {tuple(sam_prob.shape)}")
    sam_prob = _resize_like(sam_prob, teacher_prob).clamp(0.0, 1.0)
    usable_classes = min(num_classes, sam_prob.shape[1])
    for cls in fg_classes:
        if 0 < cls < usable_classes:
            support[:, cls] = sam_prob[:, cls]

    prompt_quality = sam_out.get("prompt_quality")
    if prompt_quality is not None:
        prompt_quality = prompt_quality.detach().to(device=device, dtype=dtype)
        if prompt_quality.ndim == 2:
            support = support * _class_scores_to_spatial(prompt_quality[:, :num_classes].clamp(0.0, 1.0), height, width)

    sam_iou = sam_out.get("sam_iou")
    if sam_iou is not None:
        sam_iou = sam_iou.detach().to(device=device, dtype=dtype)
        if sam_iou.ndim == 2:
            support = support * _class_scores_to_spatial(sam_iou[:, :num_classes].clamp(0.0, 1.0), height, width)

    if min_support > 0.0:
        support = torch.where(support >= float(min_support), support, torch.zeros_like(support))
    support[:, 0] = 0.0

    sam_boundary = sam_out.get("sam_boundary")
    if sam_boundary is not None:
        boundary = sam_boundary.detach().to(device=device, dtype=dtype)
        if boundary.ndim == 3:
            boundary = boundary.unsqueeze(1)
        boundary = _resize_like(boundary, teacher_prob[:, :1]).clamp(0.0, 1.0)

    fg_support = support[:, 1:].max(dim=1).values if num_classes > 1 else support[:, 0]
    return {
        "valid": True,
        "support": support,
        "foreground_support": fg_support,
        "boundary": boundary,
    }
