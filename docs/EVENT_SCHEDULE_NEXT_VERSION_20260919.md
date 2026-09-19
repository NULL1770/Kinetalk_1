# Event schedule / mouth residual experiment

Status: both fixed-budget formal experiments completed; neither audio timing gate passed. No model was promoted. This is an inner-development experiment with inherited upstream exposure, not a sealed test or an external-method ranking. Results: `EVENT_SCHEDULE_RESULTS_20260919.md`.

## Motivation and architecture

The bounded prior improved motion distribution, while the ordered-audio branch and a nonlinear window-summary probe did not establish a timing advantage over independently trained static controls. Existing systems already contain temporal networks. This version changes the learned intermediate variable, rather than claiming to add temporal modeling for the first time.

The factorization is `p(event schedule | ordered audio, global emotion, identity, audio baseline) × p(motion | event schedule, global emotion, identity, audio baseline)`. Four groups describe brow raising, brow lowering, squinting and eye widening. Each event has an onset and sampled duration; active/phase/duration form a 12-dimensional schedule. The schedule contains no measured motion coefficient amplitudes or query target mean. It is a weak motion-derived label, not a semantic ground truth.

The train-only excursion teacher requires positive local prominence and complete temporal support. A constant high eyebrow pose is not an event. Gaps and missing channels split supervision; incomplete excursions are unknown. Prediction clocks use native audio support, never target-known support. Duration bins are 5/10/15/25/40/60/100 frames at 25 fps; quantization error and overlength events are explicitly reported.

The receiver starts from the existing latent prior with zero initialized condition injection and a frozen motion AE. Two copies train with identical initialization, examples, noise and budget: event schedule versus null schedule. The original source checkpoint is immutable. The audio event model is a temporal convolution predictor of onset hazard and categorical duration. Sampling suppresses within-group overlapping events. Three seeds each train ordered audio and an independently trained matched-static control. Reverse audio and within-model static interventions are additional diagnostics. Training uses one joint event-process likelihood; it does not add several independently weighted motion losses.

## Fixed budget and evidence

- Same source-bound 613 fit / 206 sentence-held-out inner validation clips. Historical outer tensors are not indexed; existing upstream exposure is disclosed.
- Receiver: 2 arms × 2000 updates, batch 24, paired seed 2026091909.
- Predictor: 3 seeds (42/123/2026) × 2 arms × 1500 updates, batch 12, 180-frame crops. Artificial crop boundaries exclude 15 frames of target likelihood on each affected side; audio context remains available.
- Receiver controls and generated-motion diagnostics: 24 clips selected only by metadata, 4 predeclared draws each. Audio predictor metrics use all supported inner validation clips.
- Receiver engineering gate: oracle activity Brier below both null and empty conditions on identical target-known support, visible-condition response numerically above 1e-5 mean coefficient difference for empty and shifted schedules, and exact protection of the other 43 channels. Predicted unknown events do not remove score positions. This gate alone does not certify naturalness.
- Audio gate: all seeds must improve onset Brier and joint process NLL versus the independently trained static arm with sentence-bootstrap 95% upper bound below zero; joint NLL must also beat reverse audio. Duration-only NLL is reported separately. Only five validation sentences are available, so uncertainty evidence is limited.
- Fixed final checkpoints and the first declared seed produce diagnostic candidates even when a gate fails. No best-seed selection and no automatic default replacement.

Every full-face evaluation exports shared ARKit-MBE, official-mask LBE, signed/absolute FDD and supplemental upper/brow diagnostics. FDD is order invariant and cannot certify audio timing. AV confidence/offset and external distribution metrics remain pending their proper evaluator dependencies. Source hashes, teacher coverage, source checkpoint/statistics bindings, failures, and status are saved beside outputs.

## Separate mouth repair

`train_mouth_residual_candidate.py` learns a zero-initialized bounded residual on mouth channels 14:41 only. The other 25 channels remain bitwise unchanged. Ordered-audio and independently trained static controls share the dynamic frozen baseline, so any incremental ordered-audio claim is relative to an already audio-driven baseline. Three seeds × two arms × 1500 updates use the same inner split. MBE/LBE, mouth velocity/amplitude and out-of-range diagnostics are exported; rendering and AV checks are still required.

The mouth candidate and eyebrow receiver are evaluated separately first. Combining them requires an explicitly named combined arm and a new full-face evaluation. Global emotion and identity conditions remain inputs; this round does not establish or promise unchanged perceptual quality merely from architecture.

## Formal-run interpretation and completed prosody ablation

The formal run passed the receiver engineering control gate but failed the audio-predictor gate on all three seeds. Audio minus matched-static onset Brier was +0.00546/+0.00492/+0.00610 and joint-process NLL was +0.09881/+0.08862/+0.09315. The receiver responds to an event schedule, while the current 1540-dimensional direct TCN does not establish that audio predicts the schedule better than global/static conditions. The oracle-versus-empty activity Brier confidence interval crosses zero. Moreover, even the oracle schedule has a pooled, within-run centered, training-scale-normalized brow-up RMS ratio of only 0.2413, with 40.25% raw brow-up values outside [0,1]. Therefore the evidence does not localize failure solely to audio prediction, or certify the receiver's motion quality.

The completed v2 ablation kept the event teacher, split and duration bins fixed. A static event prior receives normalized context202 and four training-normalized clip-mean prosody features. A frozen copy is augmented by a zero-initialized TCN residual driven by ten derived local prosody channels, centered over each native audio run before cropping. The residual is bounded as tanh(f(condition))-tanh(f(zeros)), so a null local condition reproduces the frozen static model exactly even after training. Reversed conditions are recomputed from reversed raw prosody. Uniform clip/run/crop sampling is retained; there is no event oversampling or target-derived inference clock.

Each of three seeds trained static and residual stages for 1500 updates. All three failed: sentence-paired residual-minus-static joint NLL was +0.00381/+0.00388/+0.00388, with positive confidence intervals. On seed42, training joint NLL improved from 0.08829 to 0.06866, while validation worsened from 0.08562 to 0.08938. This is evidence of overfitting in this model/target/data setup, not proof that audio contains no useful information or that all audio-to-brow timing is impossible. Failed predictor weights were not used for a new generation candidate or promoted.

The completed `audit_event_receiver_bottleneck.py` compares the frozen baseline, deterministic AE reconstruction, original source prior and saved event variants on the exact same24 validation clips. It checks source hashes, masks and four declared generation seeds, and reports raw mean/bias, out-of-range values, centered RMS and shared ARKit metrics. It does not retrain or modify original results. The SSH connection recovered after the initial report. A follow-up all206 audit found brow-up RMS ratios 0.9757 for AE reconstruction, 0.7943 for the source prior and 0.6320 for event oracle. Thus the representation preserves motion while event adaptation reduces generated amplitude. Euler/Heun resolution comparisons changed amplitude only slightly. A support audit found event training retained only348/613 clips and14898/66272 source-supervised frames. The next controlled repair restores full source supervision with all other null-receiver training settings held fixed, to measure the effect before any new audio training. See the results document for exact populations and limitations.

## Execution

The sequential supervisor runs the event pipeline then the mouth candidate with separate logs and exit codes. A failed event stage is recorded and does not suppress the independent mouth experiment. Fresh output paths are required. GPU smoke must pass before the full fixed-budget run is launched.
