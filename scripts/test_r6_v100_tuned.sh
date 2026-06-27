#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python validate_r6.py \
  --config outputs/SAGE_SAM_R6_3Class_V100_Tuned/resolved_config.yaml \
  --checkpoint outputs/SAGE_SAM_R6_3Class_V100_Tuned/checkpoints/best_val_dice.pth

python test_r6.py \
  --config outputs/SAGE_SAM_R6_3Class_V100_Tuned/resolved_config.yaml \
  --checkpoint outputs/SAGE_SAM_R6_3Class_V100_Tuned/checkpoints/best_val_dice.pth \
  --save-pred

python export_deploy_checkpoint.py \
  --checkpoint outputs/SAGE_SAM_R6_3Class_V100_Tuned/checkpoints/best_val_dice.pth \
  --output outputs/SAGE_SAM_R6_3Class_V100_Tuned/checkpoints/deploy_student.pth
