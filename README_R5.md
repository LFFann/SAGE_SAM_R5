# SAGE-SAM R5

Foreground-Calibrated SAM-Guided Semi-Supervised Segmentation.

R5 keeps the R4 single-loop training and dual-fusion deploy student, but changes SAM's role. SAM is no longer treated as a full semantic pseudo-label teacher. It is used as a foreground structural verifier; class semantics come from the student/EMA teachers, and background hard labels are allowed only after foreground exclusion and budget capping.

## Core Changes From R4

- `r5/ssl/foreground_safe_target_builder.py`: builds foreground-safe targets. SAM channel 0 is ignored; background is derived only from teacher confidence plus foreground exclusion.
- `r5/ssl/foreground_participation_controller.py`: enforces minimum foreground participation, caps hard background supervision, and triggers emergency mode when foreground participation collapses.
- `r5/losses/tri_state_pseudo_loss.py`: uses hard foreground/background-after-cap, fuzzy candidate positives, rank loss, and safe negative loss.
- `r5/losses/foreground_safe_kd.py`: SAM KD and SAM unsupervised consistency run only on calibrated foreground pixels.
- `r5/ssl/foreground_correlation_locality.py`: correlation/locality losses propagate foreground candidates only and never convert foreground candidates into background hard labels.

## Training Schedule

```text
0 - 1200 iter:
  supervised student + supervised SAM adapter/prompt only
  no SAM unsupervised CE
  no background unsupervised hard CE

1200 - 3000 iter:
  class-conditional SAM foreground grounding
  tri-state pseudo supervision
  background cap active

3000 - 6000 iter:
  foreground-only correlation propagation
  masked-locality proxy via strong masked views

6000+ iter:
  self-reliance decay for SAM SSL/KD
  labeled SAM adapter supervision remains active
```

## Data Format

```text
<root>/<dataset_name>/
  labeled/image
  labeled/mask
  unlabeled/image
  val/image
  val/mask
  test/image
  test/mask
```

Masks must contain integer ids in `0..num_classes-1` or `ignore_index`.

## Commands

CPU smoke:

```bash
python SAGE_SAM_R5/train_r5.py --config SAGE_SAM_R5/configs/r5_smoke_cpu.yaml --dry-run
python SAGE_SAM_R5/train_r5.py --config SAGE_SAM_R5/configs/r5_smoke_cpu.yaml
```

Server data/SAM checks:

```bash
python SAGE_SAM_R5/tools/validate_dataset.py --config SAGE_SAM_R5/configs/r5_3class_v100_tuned.yaml
python SAGE_SAM_R5/tools/verify_real_sam.py --config SAGE_SAM_R5/configs/r5_3class_v100_tuned.yaml
```

V100 tuned training:

```bash
bash SAGE_SAM_R5/scripts/train_r5_v100_tuned.sh
```

Validation/test/export after training:

```bash
bash SAGE_SAM_R5/scripts/test_r5_v100_tuned.sh
```

Key diagnostics to watch in `metrics.jsonl`:

```text
per_class_sam_participation_ratio
hard_fg_ratio_class1 / hard_fg_ratio_class2
soft_fg_ratio_class1 / soft_fg_ratio_class2
background_hard_ratio
ambiguous_ratio
safe_negative_pixel_ratio
fast_slow_agreement
sam_foreground_support_ratio
emergency_mode
```

R5 is healthy only if foreground participation is nonzero after the grounding stage and `background_hard_ratio` stays capped instead of saturating the unsupervised loss.
