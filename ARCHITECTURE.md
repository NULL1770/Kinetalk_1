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

During Stage 4 training, supervision is split into two paths:

- **Self-style reconstruction** uses the query clip's own Style and the aligned
  query residual as a valid framewise flow-matching target. This branch remains
  dominant and protects fidelity to the frozen B0 baseline.
- **Cross-style intervention** uses a same-sentence emotion donor as a
  counterfactual Style source. It is not compared frame by frame with query
  motion, because a different Style has no unique target trajectory. It
  instead matches target global/local affect, recovers donor Style, and
  preserves lower-face velocity direction/timing.

The paired neutral Style view in Stage 2 is built as
`neutral_motion - Stage1(neutral_content)`. `b0_gt` is lower-face-only by
contract, so using it for this view would erase upper-face execution signals
(eyes, brows, cheeks) and teach view consistency to suppress them. Stage 1
still owns only lower-face/content channels; upper-face expression and
execution behavior remain in the residual Style/affect path.

The source tree has one active style coordinate (`style`) and one active
restore-based path. Old prototype/calibration checkpoints are incompatible;
all four stages must be retrained under the current protocol.
