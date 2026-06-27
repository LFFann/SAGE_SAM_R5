#!/usr/bin/env bash
set -euo pipefail
python SAGE_SAM_R5/test_r5.py --config outputs/SAGE_SAM_R5_3Class/resolved_config.yaml --checkpoint outputs/SAGE_SAM_R5_3Class/checkpoints/best_val_dice.pth --save-pred --split test

