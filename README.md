# KineTalk: restore-based residual facial animation

This repository contains the cleaned four-stage KineTalk experiment built on
the validated restore-style DLP plus residual Flow-Matching DiT pipeline.
Source code, training configuration, evaluation tools, architecture notes, and
data-contract checks are included. Datasets and model checkpoints are kept out
of Git; configure their local paths before training.

This project keeps the restore baseline's validated DLP + residual Flow-Matching DiT and changes the Style path to use a reference residual:

```text
B0_target = DLP(target_audio_content)
B0_ref = DLP(reference_audio_content)
S_ref = StyleEncoder(reference_BS - B0_ref)
M_hat = B0_target + ResidualDiT(target_audio, target_affect, S_ref, noise)
```

The reference audio and BS can contain different content from the target. They only need to be aligned to each other on their own time axis. The reference audio is not used as a target and is not mixed into target content.

## Pipeline

The stages are run in order after setting paths in `configs/train.yaml`:

```text
python scripts/00_check_data.py --config configs/train.yaml
python train.py --config configs/train.yaml --stage neutral
python train.py --config configs/train.yaml --stage factors
python train.py --config configs/train.yaml --stage audio_emotion
python train.py --config configs/train.yaml --stage generator
```

The active path is:

```text
target audio/content -> B0 target
reference audio + reference BS -> reference residual style
B0 target + target affect + style -> residual DiT -> final BS
```

The reference audio and BS may contain different content from the target.
Style is computed from the reference BS minus the DLP output for that same
reference clip, so reference mouth timing is not copied into the target.

Evaluation must use an explicit held-out split. The CSV exporter and full
stage evaluator default to `val`; use `--split test` only for the final frozen
report after model selection. The previous no-argument CSV command used the
training split and is therefore only a qualitative training diagnostic.

```text
python scripts/export_blender_semantic_csv.py --split val --output-dir artifacts/blender/semantic_val
python scripts/02_eval_all_stages.py --split val --per-emotion 0
python scripts/05_eval_stage1_fidelity.py --split val --output artifacts/eval/stage1_val.json
python scripts/08_eval_stage4_fidelity.py --split val --output artifacts/eval/stage4_val.json
python scripts/09_eval_stage4_semantics.py --split val --output artifacts/eval/stage4_semantics_val.json
```

After freezing the protocol and checkpoint selection, repeat the same reports
with `--split test`; test results are never used for tuning.

## Repository layout

- `kinetalk_b0/`: datasets, losses, encoders, DLP, residual DiT, and stage models
- `scripts/`: data checks, audits, evaluation, and Blender CSV export
- `configs/`: reproducible training configurations
- `docs/`: architecture, execution plan, and literature/data-contract notes
- `deploy.py`: deployment-facing inference entry point

For Blender evaluation, `scripts/export_blender_semantic_csv.py` writes the
model's ARKit-52 output by name into the project's 51-channel Blender order.

The active tree contains only the restore-based model path. Historical
prototype/calibration implementations are kept outside the source tree in the
ignored `artifacts/` archive and are not importable by training code.
