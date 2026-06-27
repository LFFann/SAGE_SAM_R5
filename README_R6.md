# SAGE-SAM R6

SAM-Calibrated Set-valued Self-training with Structure Propagation.

R6 keeps the single-loop dual-fusion deploy student, but changes SAM's role. SAM is not a hard foreground pseudo-label judge. It provides structure, boundary, region support, and calibration signals while class semantics remain controlled by the student/EMA teachers and the labeled medical data.

## Core Changes From R5

- `r6/ssl/foreground_safe_target_builder.py`: builds calibrated candidate label sets. Empty candidates are not silently converted to background; weak foreground evidence is recovered as top-k foreground candidates or left ignored.
- `r6/ssl/foreground_participation_controller.py`: enforces foreground participation, caps hard background supervision, and removes background candidates when emergency mode disables background hard CE.
- `r6/losses/tri_state_pseudo_loss.py`: trains singleton sets, fuzzy candidate sets, rank separation, and U2PL-style probability-rank negative labels.
- `r6/ssl/foreground_correlation_locality.py`: uses SAM/candidate foreground regions as propagation seeds and writes propagated foreground labels back into the candidate set before SSL losses.
- `r6/engine/trainer.py`: SAM KD, SAM unsupervised consistency, relation, and locality use `sam_region_gate`, which includes candidate foreground, fuzzy, conflict, propagated, and uncertain boundary regions instead of only hard foreground seeds.

## Training Schedule

```text
0 - 800 iter:
  supervised student + supervised SAM adapter/prompt only
  no background unsupervised hard CE
  no foreground conformal threshold updates

800 - 2000 iter:
  class-conditional conformal candidate sets
  fuzzy foreground supervision
  SAM shape/boundary support without hard veto
  background cap active

2000 - 5000 iter:
  SAM-anchored correlation propagation
  U2PL-style rank negative learning
  conflict and bias review

5000+ iter:
  self-reliance decay for SAM SSL/KD
  SAM boundary/shape regularization remains active
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
python train_r6.py --config configs/r6_smoke_cpu.yaml --dry-run
python train_r6.py --config configs/r6_smoke_cpu.yaml
```

Server data/SAM checks:

```bash
python tools/validate_dataset.py --config configs/r6_3class_v100_tuned.yaml
python tools/verify_real_sam.py --config configs/r6_3class_v100_tuned.yaml
```

V100 tuned training:

```bash
bash scripts/train_r6_v100_tuned.sh
```

Validation/test/export after training:

```bash
bash scripts/test_r6_v100_tuned.sh
```

Key diagnostics to watch in `metrics.jsonl`:

```text
per_class_sam_participation_ratio
hard_fg_ratio_class1 / hard_fg_ratio_class2
soft_fg_ratio_class1 / soft_fg_ratio_class2
background_hard_ratio
ambiguous_ratio
empty_candidate_ratio
candidate_foreground_ratio
foreground_propagated_ratio
safe_negative_pixel_ratio
fast_slow_agreement
sam_train_gate_ratio
sam_foreground_support_ratio
masked_locality_ratio
foreground_masked_ratio
emergency_mode
```

R6 is healthy only if foreground participation remains nonzero after the grounding stage, `sam_train_gate_ratio` does not collapse to zero, and `background_hard_ratio` stays capped instead of saturating the unsupervised loss.
