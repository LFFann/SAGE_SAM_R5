from __future__ import annotations

import torch

from r6.calibration.prompt_reliability_calibrator import PromptReliabilityCalibrator
from r6.ssl.target_builder import build_set_valued_targets


def test_target_builder_always_returns_set_and_safe_negative_shapes():
    cal = PromptReliabilityCalibrator(3, min_pixels_per_class=1, use_soft_gate=True, min_participation_ratio=0.25)
    cal.teacher_q = torch.tensor([0.95, 0.95, 0.95])
    cal.sam_q = torch.tensor([0.95, 0.95, 0.95])
    teacher_prob = torch.full((2, 3, 5, 5), 0.01)
    teacher_prob[:, 0] = 0.98
    sam_prob = torch.full_like(teacher_prob, 0.01)
    sam_prob[:, 1] = 0.98

    targets = build_set_valued_targets(
        {"mean_prob": teacher_prob},
        {"valid": True, "sam_prob": sam_prob},
        cal,
        {"max_candidate_set_size": 2, "safe_negative_threshold": 0.05, "min_teacher_confidence": 0.5},
    )

    assert targets["candidate_set"].shape == teacher_prob.shape
    assert targets["safe_negative_set"].shape == teacher_prob.shape
    assert targets["candidate_set"].sum(dim=1).min() >= 1
    assert targets["sam_train_gate"].any()
    assert torch.isfinite(targets["candidate_weight"]).all()
    assert "safe_negative_pixel_ratio" in targets["stats"]
    assert len(targets["stats"]["per_class_safe_negative_ratio"]) == 3


def test_calibrator_coverage_fallback_keeps_soft_participation_nonzero():
    cal = PromptReliabilityCalibrator(
        2,
        min_pixels_per_class=1,
        use_soft_gate=True,
        min_participation_ratio=0.50,
        coverage_target=0.50,
        temperature=0.05,
    )
    cal.teacher_q = torch.tensor([1.0, 1.0])
    cal.sam_q = torch.tensor([1.0, 1.0])
    prob = torch.tensor([[[[0.60, 0.55], [0.50, 0.45]], [[0.40, 0.45], [0.50, 0.55]]]])

    gates = cal.gates(prob, prob)

    assert float(gates["sam_train_weight"].mean()) > 0.05
    assert float(gates["sam_train_gate"].float().mean()) >= 0.50


def test_r6_sam_foreground_support_does_not_create_background_hard_label():
    cal = PromptReliabilityCalibrator(3, min_pixels_per_class=1, use_soft_gate=True)
    cal.teacher_q = torch.tensor([0.50, 0.50, 0.50])
    cal.sam_q = torch.tensor([0.50, 0.50, 0.50])
    teacher_prob = torch.full((1, 3, 4, 4), 0.05)
    teacher_prob[:, 0] = 0.90
    sam_prob = torch.full_like(teacher_prob, 0.01)
    sam_prob[:, 1] = 0.95

    targets = build_set_valued_targets(
        {"mean_prob": teacher_prob},
        {"valid": True, "sam_prob": sam_prob},
        cal,
        {
            "_iteration": 1500,
            "foreground_grounding_start": 1200,
            "disable_background_unsup_until": 1200,
            "foreground_classes": [1, 2],
            "min_teacher_confidence": 0.5,
            "min_sam_confidence": 0.5,
        },
    )

    assert targets["stats"]["background_hard_ratio"] == 0.0
    assert targets["candidate_set"][:, 0].sum() == 0
    assert targets["candidate_set"][:, 1].sum() > 0
    assert targets["sam_train_gate"].any()


def test_r6_emergency_mode_disables_background_when_foreground_absent():
    cal = PromptReliabilityCalibrator(3, min_pixels_per_class=1, use_soft_gate=True)
    cal.teacher_q = torch.tensor([0.50, 0.50, 0.50])
    cal.sam_q = torch.tensor([0.50, 0.50, 0.50])
    teacher_prob = torch.full((1, 3, 4, 4), 0.01)
    teacher_prob[:, 0] = 0.98
    sam_prob = torch.full_like(teacher_prob, 0.01)
    sam_prob[:, 0] = 0.98

    targets = build_set_valued_targets(
        {"mean_prob": teacher_prob},
        {"valid": True, "sam_prob": sam_prob},
        cal,
        {
            "_iteration": 1500,
            "foreground_grounding_start": 1200,
            "disable_background_unsup_until": 1200,
            "foreground_classes": [1, 2],
            "min_teacher_confidence": 0.5,
            "min_sam_confidence": 0.5,
            "disable_bg_if_no_fg": True,
        },
    )

    assert targets["stats"]["emergency_mode"] == 1.0
    assert targets["stats"]["background_hard_ratio"] == 0.0
    assert targets["singleton_mask"].sum() == 0
    assert targets["candidate_set"][:, 0].sum() == 0
    assert targets["candidate_set"].sum() == 0


def test_r6_rank_negative_keeps_unreliable_pixels_useful_without_sam_veto():
    cal = PromptReliabilityCalibrator(3, min_pixels_per_class=1, use_soft_gate=True)
    teacher_prob = torch.tensor(
        [[
            [[0.65, 0.65], [0.65, 0.65]],
            [[0.30, 0.30], [0.30, 0.30]],
            [[0.05, 0.05], [0.05, 0.05]],
        ]]
    )
    sam_prob = torch.full_like(teacher_prob, 1.0 / 3.0)

    targets = build_set_valued_targets(
        {"mean_prob": teacher_prob},
        {"valid": True, "sam_prob": sam_prob},
        cal,
        {
            "_iteration": 1500,
            "foreground_classes": [1, 2],
            "disable_bg_if_no_fg": True,
            "empty_candidate_topk_foreground": 1,
            "safe_negative_rank_low": 1,
            "safe_negative_max_prob": 0.35,
        },
    )

    assert targets["candidate_set"][:, 1].any()
    assert targets["safe_negative_set"][:, 2].any()
    assert targets["stats"]["safe_negative_pixel_ratio"] > 0.0
