#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python SAGE_SAM_R5/validate_r5.py \
  --config outputs/SAGE_SAM_R5_3Class_V100_Tuned/resolved_config.yaml \
  --checkpoint outputs/SAGE_SAM_R5_3Class_V100_Tuned/checkpoints/best_val_dice.pth

python SAGE_SAM_R5/test_r5.py \
  --config outputs/SAGE_SAM_R5_3Class_V100_Tuned/resolved_config.yaml \
  --checkpoint outputs/SAGE_SAM_R5_3Class_V100_Tuned/checkpoints/best_val_dice.pth \
  --save-pred

python SAGE_SAM_R5/export_deploy_checkpoint.py \
  --checkpoint outputs/SAGE_SAM_R5_3Class_V100_Tuned/checkpoints/best_val_dice.pth \
  --output outputs/SAGE_SAM_R5_3Class_V100_Tuned/checkpoints/deploy_student.pth

