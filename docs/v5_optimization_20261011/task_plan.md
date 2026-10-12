# v5 optimization: demonstrated fixes and contribution validation

## Material Passport

- Origin: authorized continuation of KineTalk architecture experiments; ARS experiment/reproducibility discipline and planning-with-files used inline.
- Inputs: immutable v4 seed47 checkpoint, Phase53 retained baseline, MEAD TRAIN/development bindings, existing local/remote HDTF raw-motion/audio/feature products.
- Evidence: v4 completed review and source code; diagnostics and future experiments must distinguish deployable conditions from GT oracles.
- Authority: user authorizes code changes, processing, training, evaluation and continued optimization. Preserve original assets and sealed test boundary.

## Active phases

1. [complete] Frozen v4 diagnosis on predeclared balanced development subset plus fixed display clips:134clips/15conditions; saved `diagnostic_report.json`. Detected geometry supervision mask bug, not an established teacher-state improvement.
2. [complete] Repair demonstrated fixed-rig numerical dependency support, refit TRAIN geometry scales, and compare matched repair/continued-training/rollout-mean pilots; engineering fix and mean loss are not established contributions.
3. [complete] Bind full available HDTF quality/source-window/features and source-time-disjoint single references.233TRAIN sources/145people/6739windows verified; emotion/intensity/neutral targets masked. Domain-weighted training still pending phase4.
4. [complete] Verify mixed-data 32-step smoke, then run matched MEAD-only versus MEAD+HDTF 256-update arms with fixed development and 3-seed dynamic evaluation; decide whether evidence supports longer training.
5. [complete] Render fixed angry/happy native-clock clips for prior controls and all data routes; inspect geometry and masks/hashes. Complete full1367/factor32 controls for MEAD/all-factor/annotation-aware routes. All routes fail overall promotion; contribution validation remains pending.
6. [in_progress] Isolate the measured sustained-state bias: implement a zero-start audio pooled-posture adapter plus TRAIN-only, teacher-advantage-gated posture guidance. Compare equal-budget continuation, adapter/direct-posture, and adapter/privileged-posture arms from the identical MEAD checkpoint. Preserve stochastic event likelihood/execution and report posture versus centered error separately. Freeze running code snapshots; queue new pilots after existing scoring.
7. [complete] Independently audit shared-rig width/lift/aperture/depth at B0, geometry head, proposal, projection and coefficient clipping on128balanced development clips. PID9406 completed four frozen checkpoints; upstream residual/head width remains deficient, projection recovers part and clipping effect is negligible. Saved720rows/receipt. Preserve model digests and score-only query targets.
8. [in_progress] Test whether frozen h0/B0 discards acoustically timed mouth geometry: add a zero-start geometry residual receiver from native HuBERT+prosody, with affect/style conditions and no query motion. Compare equal-capacity coarse-h0 receiver versus native-HuBERT receiver and retained no-receiver256update control. A fourth receiver arm adds only sampled-rollout geometry-mean supervision to test observed/sample mismatch. New pilots wait for8413, use immutable fresh snapshot, same source/sequence/optimizer budget, complete full/dynamics/control/geometry evaluation. This receiver is a bounded architecture hypothesis, not a novel contribution by itself; do not assume better mouth or style.
9. [in_progress] Freeze four candidates and measure target-style direction on64predeclared cross-person development utterance pairs matching exact sentence/emotion/intensity. Generation uses query audio/B0 + one independent neutral reference; target donor motion enters clip-statistic scoring only after generation. CPU worker10476, no training; record target-statistic improvement and frozen emotion/timing preservation. Three development people limit generalization; no factor disentanglement claim from sensitivity alone.

## Acceptance boundaries

## Continuation 2026-10-12

- [complete] Resume the immutable acoustic geometry queue after the prior SSH bridge became unresponsive. The coarse arm completed its full 1,367-clip report; the native HuBERT arm completed training and entered scoring.
- [complete] Collect native HuBERT and coarse-receiver reports; rollout-mean arm failed before producing a checkpoint, was recorded without retry. Export matched full-development, dynamics, controls, geometry and style CSVs with common clip/B0 binding in `acoustic_tables_provenance.json`.
- [in_progress] Finish matched state-coordinate intervention: compare raw-direct, decoder-conditioned and teacher-guided decoder-conditioned state under 256 updates, then run the fixed development/state/dynamics/control/geometry audits. This is a diagnosis of a conditioning hypothesis, not a novelty claim.
- [complete] Render and inspect acoustic-receiver angry/happy at five fixed native timestamps; full decode/audio/rig verified. Receiver helps the happy display clip but angry remains underexpressed. State-arm native renders remain pending collection.
- [in_progress] Collect matched state arms with source/checkpoint hashes, export full/state/dynamics/geometry tables, and inspect fixed state-arm renders.
- [in_progress] Run fixed-schedule frozen decoder-bottleneck diagnostic from predeclared raw-direct checkpoint after state queue; detached PID3781, fresh tools only, smoke then128+display clips. Distinguish input-state response from geometric/output response before selecting further training changes.
- [pending] Promote no candidate until full development, independent training seeds, dynamic calibration and style/content factor tests pass.

## Decoder bottleneck result 2026-10-12

- [complete] Collect frozen decoder-bottleneck smoke/full reports from the predeclared raw-direct checkpoint. The run completed 134 development clips, fixed native event schedules, no deployment query motion, and no test targets.
- [complete] Export paired clip summaries and descriptive bootstrap intervals in `decoder_bottleneck_summary.csv`, `decoder_bottleneck_paired_deltas.csv`, and `decoder_bottleneck_summary_provenance.json`; intervals resample clips only and are not significance or multi-seed evidence.
- [complete] Render fixed angry/happy counterfactuals for GT/B0/audio/teacher-state/mean-state/geometry-oracle and inspect uniform native timestamps. Teacher/raw state improves the visible sustained mouth pose; mean-state removes much of that gain; geometry oracle improves lip shape but produces out-of-support affect collapse.
- [in_progress] Reframe the next trained candidate as a dual-timescale, uncertainty-gated affect-state distillation: keep audio for phonetic timing and high-frequency state, use the motion teacher only for a low-frequency supported affect envelope with confidence gating, and decode phonetic residual and affect residual through separate heads before a bounded composition. This is an engineering hypothesis until matched training, independent seeds, dynamics and style/content tests pass.
- [pending] Run a small matched smoke and equal-budget MEAD pilot for the dual-timescale candidate against raw-direct and decoder-direct; stop if finite/B0/reference/hash gates or predefined lip/EVE/dynamics gates fail.

Recovery note: one malformed bridge command and one unavailable tool lookup failed before changing server state; a fresh pinned bridge was used. A guessed report filename was replaced by a directory listing. No running job, checkpoint, raw data or sealed/test target was modified.

- Dynamics schedule execution is not GT emotional dynamics; report event rates/duration/amplitude, expression direction and seed-induced mouth drift independently.
- Geometric projection/decoder must improve mouth shape without losing closure/content; score all eight classes and angular/width/aperture errors.
- Style must transfer across query contents/emotions from one reference, with non-target factors held; output sensitivity alone is insufficient.
- Controlled inference switches diagnose a trained model; architectural contribution claims require matched trained ablations and independent seeds.
- Record every run directory, source binding, seed, checkpoint/hash, dataset scope and oracle status. No cherry-picked seed or test-set optimization.

## Preflight

2026-10-11: remote RTX4090 idle, system free≈11GB, data free≈4.2GB; v4 prior train/eval complete. SSH via pinned-host bridge, password never persisted. Fresh output directories only.

## Resume audit

2026-10-11 22:34 CST: bridge83717 alive; pilot follow-up5630 scores mixed full development (840/1367), route6827 trains step161/256, route follow-up7017 waits. Summary's prior1367/1367 statement was not a completed report and is superseded by live state. No duplicated jobs. A Windows `rg` literal wildcard caused an invalid-path message; replaced by directory plus `-g '*.py'` searches. Root planning milestone checkboxes need synchronization to the detailed plan.
# Completed dual-timescale pilot and archive, 2026-10-12

- Corrected full-objective smoke and control/direct/guided256-update runs completed.
- All1367development/128×3dynamics/32control/128geometry/64style reports SHA-collected.
- New-module mouth effect negligible versus matched control; guided dynamic direction worse in all3samplingseeds. No promotion.
- Angry82/happy103valid-native-frame videos with audio completed and inspected; rigerror0,25fps, full decode.
- Source/evidence/video archive prepared on user-specified existing GitHub branch; verify push before reporting upload.
- Latest evidence: `dual_timescale_review_20261012.md`. Historical plans below are superseded by this result.

