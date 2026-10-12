# Matched dual-timescale pilot, 2026-10-12

## Material Passport

- Origin: user-authorized continued optimization and explanation of content, emotion, dynamics and style.
- Inputs: raw-direct parent SHA `310fe9fa348ecb7f6210fc854f3c867dc055e470db00be1d2dc38722016df702`, fixed MEAD TRAIN internal fold, unchanged independent references and frozen B0, validation development reports only.
- Scope: bounded implementation hypothesis and descriptive development comparison; no validated novelty, semantic disentanglement, calibrated uncertainty or paper-readiness claim.

## Corrected recovery

Remote server is idle before this new queue. The completed 16-update smoke was numeric-only: its script incorrectly used `fit_sids` (speaker identifiers) as clip indices, repeated one batch, and omitted the complete style/execution/rollout objective. It is retained in `dual_timescale_smoke_numeric_only.json` and does not qualify as a formal-pilot gate. No formal dual-timescale training was dispatched by the interrupted prior turn.

Do not suppress checkpoint configuration mismatches by discarding keys. The prior mismatch came from copying an old v4 dependency snapshot; current v4 dependencies must be hashed and included with the candidate snapshot. Existing Phase53/v4 experiments are preserved.

## Hypothesis and implementation boundary

Audio supplies continuous expression coordinates and event context. Motion teacher guides only supported, low-frequency expression coordinates when its detached TRAIN target error is below the student's. Its prediction is not a valence/arousal annotation or psychological ground truth. Direct supported-envelope supervision remains available when no teacher advantage exists.

9 native frames at 25 fps span .32 s from first to last center; this fixed smoothing is a timescale bias, not an established emotional cutoff. Filtering must not cross invalid frames or timestamp gaps. Unsupported teacher/target coordinates must not influence the envelope loss or reliability target. Clip reliability is the fraction of supported frames with teacher advantage, not `any(frame)` and not calibrated uncertainty.

The decoder retains a joint inherited residual, plus zero-start low/high hidden-state adapters. Both adapters see a mixed content/emotion/style/event representation. They cannot yet be called exclusively phonetic and emotional; the legacy internal names are hypotheses. All 52 output channels remain learnable. No emotion category selects a fixed channel mask.

## Matched arms

1. v4 continuation control: same raw-direct source and legacy complete objective.
2. v5 direct envelope: added state/decoder adapters, supported-envelope direct loss, no teacher envelope distillation or reliability supervision.
3. v5 guided envelope: identical v5 architecture/direct objective plus accepted low-frequency teacher guidance and fractional reliability supervision.

Each uses seed47, exactly256 updates, batch12, lr5e-5, same data/reference order, direct posture weight1 and sampled rollout-mean weight.5. Reset RNG after model creation. Baseline-versus-v5 changes architecture and objectives; direct-versus-guided tests the guidance module. Sampling seeds47/91/137 are not independent training seeds.

Corrected16-update smoke uses indices from the real TRAIN fold, full objective, independent reference contract, B0 digest and finite new-head gradients. Formal arms start afresh from the parent, not from smoke weights. No HDTF in this isolating pilot; HDTF binding remains233TRAIN sources/145people/6739filtered windows, with unknown emotion/intensity labels masked.

## Evaluation and gates

For each arm: full1367 validation clips with Phase53 descriptive comparison, fixed128 development dynamics with all3sampling seeds,32reference/control clips,128geometry clips, and64cross-person target-style pairs. Export angry/happy render inputs on original clocks. Query motion is used only for training and scoring.

Smoke acceptance: finite loss/gradients, unchanged B0, independent TRAIN references and nonzero adapter gradients. Promotion is disabled. Longer training is justified only if lip and upper-face errors, closure, event distribution/direction and reference-transfer behavior jointly warrant it. Pipeline stop-on-failure and45-minute owned-stage timeouts; no automatic retry or favorable-checkpoint/seed selection. Preserve missing/failure receipts.

## Current measured status

Raw-direct full-development sample/clip_all: lip mean3.07597mm, max-vertex LVE5.90062mm, EVE2.42609mm, closure proxyF1.53121, whole-clip probeF1.76259; Phase53 values3.26256/6.13245/2.23756/.40102/.67681. Gains are descriptive and training budgets differ. Angry display remains corner-deficient. Dynamics event execution is not emotion timing correctness. Existing style statistics show partial donor-direction transfer on64pairs/3development speakers, with content/emotion interference still unresolved.
