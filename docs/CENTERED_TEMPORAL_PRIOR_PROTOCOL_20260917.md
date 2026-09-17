# Mean-preserving dynamics and temporal-prior comparison

This is an exploratory follow-up selected after inspecting repair12 development
results. The existing 405 clips remain internal development, never an untouched
test. No default is replaced and sealed test data is not accessed.

## Evidence and hypothesis

Alignment improves temporal correlation but changes static expression means.
Free absolute flow and residual flow fail joint acceptance. A new train-pool
sentence split diagnostic finds only small linear predictability of brows and
eyes; smoothing from 4 to 16 frames does not fix it. This is not a nonlinear
predictability upper bound. There is no confirmed fixed clock bug.

The next experiment explicitly removes clip mean from the generated dynamics
and preserves the frozen original-local Stage3 renderer mean. It compares
independent versus temporally correlated Gaussian starting noise. The latter
uses nine train-only lag-one correlations, clamped to [0,.995], as stationary
AR(1) priors. This is an established stochastic prior, not a novelty claim or
a claim to reproduce SubtleTalk. No new loss or GT inference condition is used.

## Fixed training

- Same 2315 fit and 405 internal dev, 96 native frames at 25Hz.
- Both arms: 12 epochs, seed61, batch16, AdamW lr1e-4, wd1e-5, gradclip1.
- Same model initialization, batch order, raw Gaussian draws and flow times.
- Target: valid-frame centered raw nine-channel upper motion, normalized by
  fit-only dynamic RMS (floor .005). All temporal frequencies remain free.
- Velocity, noise, ODE state and decoded dynamics are centered on valid frames.
  Correlated-prior centered variance is normalized analytically from masks/rho,
  not from each sample's realized amplitude.
- Independent local audio trunk copies completed aligned trunk; local head
  starts at zero, then learns jointly with the flow. Global/audio classifiers,
  B0, identity, whole-face renderer and aligned adapter are frozen.
- Whole-face frozen baseline uses original Stage4 audio local in Stage3
  renderer, with seed42/123/2026 and 12 Euler steps. Final upper = baseline
  valid-frame mean + centered generated dynamics. Other43 channels copy exactly.
  This removes old upper dynamics; it does not add a second dynamic trajectory.
- Fixed final checkpoint, no best-epoch or seed selection. Full development
  scores after training; no data clock, output gain or label gate tuning.

## Evaluation

Both arms report raw/centered error, temporal correlation, RMS, velocity energy,
lag-one coherence, probabilistic trajectory/variogram scores, out-of-range rate,
generated emotion teacher readout (nonindependent), and frozen hashes. Check
upper means numerically and non-upper values bitwise. Baseline and mean-preserved
aligned local are controls. Reverse/static interventions affect both flow local
and h0, leaving baseline, global and identity unchanged; this tests the whole
temporal audio condition. No independent identity or lip-sync success claim.

The paired comparison isolates prior covariance within this new parameterization;
comparison against earlier free flow changes centering and normalization too.
Stronger motion alone is insufficient: coherence, input response and preservation
must all be inspected, and failure remains a reported outcome.

## Completion and separately identified exploratory follow-up

Both arms completed 12 epochs/1740 steps in 193.3 seconds combined. Matched
initial parameter and raw random-stream hashes passed. Mean error was below
2e-7 and all other43 channels were unchanged. White improved movement coherence
and amplitude but not accurate event timing. AR1 did not outperform white.

After viewing these development outcomes, evaluated a fixed, untrained composition:
the window-average frozen Stage4 audio four-state prediction is lifted relative
to independent neutral anchors to supply static upper means. Aligned or white-flow
centered dynamics supply temporal changes; original-local baseline supplies all
remaining43 channels. No GT inference, sample-specific selection, gain or lag fit.
This is post-development exploration, not part of the paired prior comparison.

The state-aligned candidate lowers brow/eye raw MSE to .017733/.006593 while
mouth stays .009348; correlations remain .132/.188. The state-white candidate is
more coherent but correlations remain .100/.118. State replacement increases
brow out-of-range rate to about22% and lowers nonindependent generated emotion
readout from92.9% to91.4%. Not a joint success and no default replacement.

Full conclusions and fixed videos are in
artifacts/temporal_repair_20260917/centered_prior/state_mean/RESULTS.md and index.html.
