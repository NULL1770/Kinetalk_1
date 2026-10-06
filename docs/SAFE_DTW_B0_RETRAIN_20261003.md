# Safe-DTW B0 retraining protocol

The retraining run uses the audited native-time DTW release at
`safe_supervision_v1/teacher_manifest_gated.jsonl`.

For each source clip, the articulation stage reads `neutral_teacher_on_source`
and `native_teacher_mask` from the HuBERT DTW pair artifact. This target is on
the same source motion clock as the native content feature. The emotional
`canonical_motion` field is never used as a B0 target. Clean neutral identity
clips are added as exact native anchors; all other stages continue to use the
complete approved train split.

The run is bound to the safe manifest SHA256 in its recipe. It must pass the
Stage1 native overfit and development mouth timing gates before sealed-test
evaluation and Blender rendering are accepted.
