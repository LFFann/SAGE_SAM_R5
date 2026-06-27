#!/usr/bin/env bash
set -euo pipefail
python SAGE_SAM_R5/train_r5.py --config SAGE_SAM_R5/configs/r5_3class_v100.yaml
