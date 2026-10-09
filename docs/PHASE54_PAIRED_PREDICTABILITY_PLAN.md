# Phase54: paired expression dynamics predictability

2026-10-09, pre-implementation snapshot c669865 pushed. Diagnostic only; no new deployable model or reconstruction gradient to the student.

Question: does the frozen 772D emotion/prosody prior retain predictable native expression dynamics, and how does their predictability compare with neutral-B0/alignment differences? Phase45 tested existing latent targets; Phase50 decomposed energies. This audit adds prediction of the physical paired components, not a repeat of either.

Use the same approved safe manifest, TRAIN fit2024 emotional pairs, internal speaker281 and sentence278 pairs. No external development query inference or sealed data. Original native source clocks, observation/channel support, mouth event masks and adjacent25fps gap checks. No new DTW or interpolation of targets. Aligned neutral is an imperfect counterfactual, so call GT-neutral an expression difference, not pure emotion ground truth.

Targets: GT-neutral, neutral-B0 and GT-B0, jointly reported under exactly the same mask; within-clip centered trajectories and adjacent-frame displacements separately. Center both inputs and targets using each channel's own observed support. Difference only adjacent valid frames with .04s intervals; never bridge a gap.

Fixed linear ridge .001 on TRAIN only, equal weight per supported clip/channel. Features: frozen predicted u16, prior hidden128 on the same native stride/interpolation as u, standardized emotion/prosody772 and diagnostic B0 control52. Input RMS scaling and target energy normalization from TRAIN only. No regularizer/epoch search or fit on held data. Unequal input capacity means probe ranking is association, not causal disentanglement. Physical MSE, TRAIN-energy-normalized MSE, zero baseline and held energy reduction are reported together, per region/emotion; near-zero TRAIN target channels are excluded only from normalized scores and disclosed. Coefficients are diagnostic files, never injected into the model.

Tests before dispatch: channel-specific centering/weights, unequal clip lengths, missing channels/NaNs, temporal gaps, exact component sum, analytic ridge agreement and frozen train-only normalization. Preserve exact B0/parent checkpoints; verify content-NaN isolation and hidden-to-u mapping on real inputs. Fixed24clip smoke then full2583clip audit, fresh outputs, no overwrite/retry of completed jobs. Maximum outputs32MiB and bounded runtime; no raw waveform/motion exports. Store source/checkpoint/manifest hashes and per-clip errors. Original packed runtime loads development metadata but does not infer or fit development queries.

Decision: if current u is less predictive than hidden/audio on both held splits, investigate its target/mean head or temporal representation; if physical differences themselves remain unpredictable, quantify alignment/one-to-many limits before adding student capacity. Stronger B0 association alone is not leakage proof. Do not automatically choose a new loss, hard mouth mask, gain correction or receiver variant from this audit.

## Verified dispatch

Implementation5128310 pushed.21local/21remote tests pass;24clip real GPU smoke passed and parent/B0 states exact. Worker18700 full audit active at /root/kinetalk_phase54_paired_predictability_20261009. Preflight200 original files4873972bytes independently SHA-backed locally. No neural model updates; no new generated F1/MBE. Initial test detected float32 clip-weight rounding against independent NumPy; fixed explicit float64 weights and regularizer before remote dispatch. Existing expression internal holds do not erase the older B0 training exposure to the original TRAIN safe-neutral targets; the B0 predictor is a descriptive control, not a clean causal instrument.
