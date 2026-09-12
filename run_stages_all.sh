#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/train.yaml}"
mkdir -p logs outputs_b0_residual

echo "[$(date '+%F %T')] pipeline started"
echo "[$(date '+%F %T')] checking stage contracts"
PYTHONPATH=. "$PYTHON" -u scripts/check_pipeline_contract.py --config "$CONFIG" 2>&1 | tee logs/contract_check.log

echo "[$(date '+%F %T')] stage1 neutral"
PYTHONPATH=. "$PYTHON" -u train.py --config "$CONFIG" --stage neutral 2>&1 | tee logs/stage1.log
echo "[$(date '+%F %T')] stage1 completed"

echo "[$(date '+%F %T')] stage2 factors"
PYTHONPATH=. "$PYTHON" -u train.py --config "$CONFIG" --stage factors 2>&1 | tee logs/stage2.log
echo "[$(date '+%F %T')] stage2 completed"

echo "[$(date '+%F %T')] stage3 audio_emotion"
PYTHONPATH=. "$PYTHON" -u train.py --config "$CONFIG" --stage audio_emotion 2>&1 | tee logs/stage3.log
echo "[$(date '+%F %T')] stage3 completed"

echo "[$(date '+%F %T')] stage4 generator"
PYTHONPATH=. "$PYTHON" -u train.py --config "$CONFIG" --stage generator 2>&1 | tee logs/stage4.log
echo "[$(date '+%F %T')] stage4 completed"
echo "[$(date '+%F %T')] pipeline completed"
