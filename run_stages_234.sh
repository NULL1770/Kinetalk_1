#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/kinetalk_b0_residual_train
PYTHON=/root/miniconda3/bin/python
CONFIG=configs/train.yaml

echo "[$(date '+%F %T')] automatic pipeline started"
echo "[$(date '+%F %T')] stage2 factors"
PYTHONPATH=. "$PYTHON" -u train.py --config "$CONFIG" --stage factors 2>&1 | tee logs/stage2.log
echo "[$(date '+%F %T')] stage2 completed"

echo "[$(date '+%F %T')] stage3 audio_emotion"
PYTHONPATH=. "$PYTHON" -u train.py --config "$CONFIG" --stage audio_emotion 2>&1 | tee logs/stage3.log
echo "[$(date '+%F %T')] stage3 completed"

echo "[$(date '+%F %T')] stage4 generator"
PYTHONPATH=. "$PYTHON" -u train.py --config "$CONFIG" --stage generator 2>&1 | tee logs/stage4.log
echo "[$(date '+%F %T')] stage4 completed"

echo "[$(date '+%F %T')] automatic pipeline completed"
