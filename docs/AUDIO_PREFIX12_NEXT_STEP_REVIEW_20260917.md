# Audio Prefix12: Two Conditional Next Steps

Date: 2026-09-17. Design review only. No training, source-code modification, sealed-test access, new curve retrieval, or remote operation was performed. The remote connection was unavailable when this review was requested. Recommendations below are hypotheses, not measured results.

Evidence comes from `artifacts/audio_prefix_adaptation_20260917/audio_prefix12/metrics_review.md` and `.json`, `NEXT_AUDIO_CONDITION_REVIEW_20260917.md`, and `NEXT_ORIGIN_AUDIO_AUDIT_20260917.md`. The next experimental protocol must be locked separately before implementation.

The immediate priority is useful audio timing on development clips, not additional movement or another mean-offset improvement. Keep the successful clean-prefix continuation mechanism and separate offline DC output. Do not combine both proposed modifications in the first experiment.

## 1. Limit Temporal Adaptation Capacity Before Replacing the Audio Representation

**Evidence.** Joint local-trunk adaptation improves development correlation in all three inference seeds, but the average gain is small: brows .0437 -> .0796, eyes .0742 -> .0818. Fit seed42 correlation changes much more: .0811/.1396 -> .2792/.2101, compared with development .0488/.0707 -> .0782/.0794. Eye displacement MSE worsens in all three seeds; centered ES worsens in both regions. This is compatible with excessive adaptation capacity or representation drift, but the different fit/development populations and one training seed do not prove either cause.

Local-static is an especially useful warning. On adapted development seed42 it lowers centered MSE from .001892/.001935 to .001538/.001738, while correlation drops from .0782/.0794 to .0256/.0179. Time-varying audio supplies some correct timing and also extra variation that harms other scores. More conditioning strength cannot be accepted as the objective.

**Prerequisite diagnosis.** Before another generator run, establish these two points on available or later retrieved data:

- Compare full and local-static on the *same clips, noise, and masks*, stratified by emotion and sentence. Separate centered amplitude, target correlation, velocity, and per-clip score differences. The present aggregate result cannot establish that every clip has excessive movement. Preserve the raw domain and separately inspect renderer clipping; coefficient motion outside the usable rig range can look static after display.
- Check whether trainable-local feature change and generation gain concentrate on sentences used in adaptation. Use a predeclared split inside the fit data for the next adaptation experiment and report incremental-adaptation transfer. A local encoder already trained on all fit labels must not be called an unseen-sentence encoder merely because a new probe head uses out-of-fold fitting. Fully clean cross-modal OOF evidence would require every motion-supervised component under study to exclude that fold. Existing low-pass ridge results are weak evidence, not an information-theoretic upper bound.

These are pending diagnoses, not completed measurements. If the primary failure is renderer saturation, data/clock/tracker problems, or absence of conditional information, a small adapter is not the remedy.

**Minimal change.** Freeze the existing original local encoder. Replace full `input/blocks/local_head` unfreezing with one small residual feature adapter after its 64D local output, restricted to temporal deviations:

```text
l0(t) = frozen_original_local(audio)(t)
d(t)  = center_valid(l0)(t)
r(t)  = W_up SiLU(W_down d(t))
l(t)  = l0(t) + center_valid(r)(t)
```

Fix rank 8 in advance, with bias-free 64->8->64 maps (1,024 parameters), and initialize `W_up` to zero. The second centering is needed because the nonlinear transform can reintroduce a time mean. This keeps the mean local feature unchanged and step0 equivalent to the original receiver, while allowing a limited learned temporal feature correction. It introduces no new motion label, per-frame 9D control, activity scalar, output gain, smoothing, or loss. It cannot reconstruct information discarded by the frozen encoder, which is a clear limitation rather than a claimed benefit.

Train the adapter and the same upper receiver with the existing unknown-only FM, old internal static coordinates, and strict-past training. Global audio, identity, state, mouth, and final DC stay fixed. Use the same source, upper initialization, clip order, noise/time, optimizer policy, and 12-epoch budget as the recorded frozen-local control. Existing control results are reusable only if exact recipe/random-draw equivalence is established; otherwise run a matched control. The adapter initialization must not perturb the recorded upper initialization or training random stream. Verify zero-change step0; `W_down` can legitimately have zero gradient at the first step because `W_up` starts at zero, so check its gradient after the adapter begins updating.

**Acceptance and stop.** This tests restricted feature adaptation against full-trunk adaptation, not the already answered question of whether any local gradient exists. Require meaningful held-out incremental transfer and consistent paired/distribution evidence against frozen-local, with no renewed seam or eye-speed regression. Include local-only static/reverse. Lower fit loss, stronger feature response, or bigger RMS is insufficient. If fit gain remains large and development gain weak, stop increasing capacity or epochs and revisit alignment and conditional information. If useful time-varying information is absent, the adapter should not be promoted.

This is an engineering hypothesis about where adaptation should occur. By itself it is not a novel paper contribution or a guarantee of success.

## 2. Expose the Existing Audio Origin Only if Initialization Affects Later Shape

**Evidence.** DC composition reduces adapted raw error by 26.20%/30.13% but cannot improve centered correlation or velocity. The internal flow target subtracts a frozen audio-derived static origin that is not explicitly supplied to the flow as its construction-state vector. The flow may infer related information from global/local features, so absence of an explicit input is not proof of missing information. GT-history resets improve absolute reconstruction, but repeated resets mix initialization and continual target assistance.

**Prerequisite diagnosis.** Run one clearly isolated oracle when inference access returns: supply only the initial eight native frames of true past, then generate every later frame recursively with no more target resets. Compare with normal generation using the same native positions, valid masks, audio, noise indexing, and receiver calls. Exclude supplied frames from scoring both arms, and score first generated frames plus later blocks in both raw and centered coordinates. Do not treat the existing every-block GT-history oracle as this test. The true initial history is unavailable at deployment, so this is only a mechanism diagnosis.

If this one-time initialization materially helps *later centered shape and timing*, an explicit origin input deserves a matched test. If it only improves the mean, DC already isolates that benefit and this proposal has low priority for dynamics. If later timing remains weak even after a correct start, focus on proposal 1 or audio/data information instead. None of these outcomes can be inferred from the currently downloaded summary files alone.

**Minimal change.** Pass the same frozen audio four-state valid-frame mean used to construct `static_upper` into the existing global conditioning sum through a zero-initialized bias-free `Linear(4, 192)`. Save and verify that this vector is exactly the source of the static origin; do not read the adapted local copy's stale state/global heads. Keep the old target/known normalization and both raw/DC outputs unchanged. Compare the origin projection enabled versus disabled from one common source, holding local-training policy fixed. Do not add the low-rank adapter and the origin projection together in this comparison.

This is a clip-level audio coordinate input, not GT clip mean, VA, a target-like 96x9 trajectory, or a window activity label. It supplies existing information at the receiver rather than fitting an additional near-motion condition. It does not justify online causality: the current static mean and final DC depend on the complete clip.

**Acceptance and stop.** Evaluate first-block state, later centered trajectories, local-only ablations, seeds, ES/VS, and exact 43-channel preservation. Improvement limited to raw mean is reported as origin conditioning only. No timing or expression-diversity claim follows. If the same mean input is redundant with the existing global features, retain the simpler receiver.

## Scope Relative to Prior Work

These are alternatives with different causal questions, not a new collection of auxiliary losses. The 4D spline plus hard P/Q residual, soft-state injection, deterministic aligned-trajectory plus residual, and white-versus-AR1 centered prior have already been tried. Reintroducing them under new names is not the next step. Neither proposal copies a VA plus window-activity control stack; neither obtains novelty merely by differing from SubtleTalk.

The current evidence supports keeping prefix continuity and explicit mean/shape accounting. It does not yet support claiming an expressive, generally accurate audio-to-upper-face model. Mouth preservation and the identical 91.44% motion-teacher emotion readout remain inherited/nonindependent evidence; new visual and external quality checks are still separate.
