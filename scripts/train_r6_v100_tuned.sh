#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

python tools/validate_dataset.py \
  --config configs/r6_3class_v100_tuned.yaml

python tools/verify_real_sam.py \
  --config configs/r6_3class_v100_tuned.yaml

mkdir -p outputs/SAGE_SAM_R6_3Class_V100_Tuned
python train_r6.py \
  --config configs/r6_3class_v100_tuned.yaml \
  2>&1 | tee outputs/SAGE_SAM_R6_3Class_V100_Tuned/stdout.log
