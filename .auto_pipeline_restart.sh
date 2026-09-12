#!/usr/bin/env bash
set -u
cd /root/autodl-tmp/kinetalk_b0_residual_train
log=outputs_b0_residual/auto_pipeline.log
printf 'auto_pipeline_restart %s\n' "$(date -Is)" >> "$log"
for spec in 'factors stage2_factors' 'audio_emotion stage3_audio_emotion' 'generator stage4_generator'; do
  set -- $spec; stage=$1; name=$2
  printf 'starting_%s %s\n' "$stage" "$(date -Is)" >> "$log"
  /root/miniconda3/bin/python3 train.py --stage "$stage" --config configs/train.yaml >> "outputs_b0_residual/${name}.log" 2>&1 || { printf 'failed_%s %s\n' "$stage" "$(date -Is)" >> "$log"; exit 1; }
done
printf 'auto_pipeline_complete %s\n' "$(date -Is)" >> "$log"
