# Restore-based KineTalk architecture

The active model keeps the validated restore baseline: an audio-only DLP-like neutral prior followed by a residual Flow-Matching DiT.

```text
target audio/content -> DLP -> B0_target
reference audio/content -> DLP -> B0_ref
reference BS - B0_ref -> residual Style Encoder -> S_ref
B0_target + target affect + S_ref + noise -> residual DiT -> DeltaM
M_hat = B0_target + DeltaM
```

Stage 1 trains the audio-only neutral prior with masked reconstruction, velocity and acceleration losses, then freezes it. Reference inputs never alter `B0_target`.

Stage 2 learns a global execution Style from the reference residual. The reference sentence may differ from the target sentence; subtracting its own DLP output removes content-driven mouth motion before temporal pooling. Style training uses same-speaker crop/segment consistency, cross-speaker separation, and behavior statistics. It does not use cross-sentence framewise targets.

Stage 3 trains the residual DiT with target audio affect, emotion/intensity auxiliary supervision, velocity/endpoint terms, and Style swap consistency. The DiT receives the full 52-D prior and produces a full 52-D residual.

Stage 4 is deployment and causal validation. Inference requires target audio and a reference audio+BS pair. Fixed-noise swaps test that changing Style changes execution behavior while preserving target content and mouth timing; changing target audio tests affect/content control at fixed Style.

The source tree has one active style coordinate (`style`) and one active
restore-based path. Old prototype/calibration checkpoints are incompatible;
all four stages must be retrained under the current protocol.
