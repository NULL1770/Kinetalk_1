# Phase43 implementation and execution record

2026-10-08. Read this and PHASE43_PLAN.md before resuming. The user permits several changes in a round, requires a Git snapshot before changes, all eight emotion videos at review, comparison tables, and actual trained ablations. Pre-change snapshots 7fcad7e and 88dd7e9 are pushed. No sealed-set tuning or automatic model promotion.

## Implemented

- `ResponseConfig.center_local=False` preserves historical inference. When enabled, mean or sampled local latents are interpolated on the unchanged native clock, then their observed-frame mean is subtracted. The global latent is unchanged. No new parameters or loss. The diagonal Gaussian KL still concerns the original latent distribution, not its projected output. A constant shift in u cannot change the centered condition; a generated mouth mean need not be zero.
- `--reference-training single` remains the historical default: each query twice, using support A/B. `mixed` uses AB plus A on even optimizer steps, AB plus B on odd steps. Both still reconstruct 32 query views per full batch of 16. A missing second support has both masks false and no aggregation weight. Both references remain independent of the query clip and sentence. Inference still uses both references.
- Checkpoints/protocols bind both options. Resume rejects architecture or reference-schedule changes. Step parity resumes from the saved optimizer step. No HuBERT/h0 input, mouth mask, extra labels, or identity table was added. B0 remains the same frozen neutral model.
- Same learned prior variance as Phase41. Reuse completed Phase41 as 00; train A, B, AB from seed47 for 24 epochs, 16,368 updates each, on the same 10,903 fit clips and internal held-out folds. Validation is report-only and no checkpoint selection is performed.

## Verification and execution

Local complete suite: 427 passed / 1 skipped. Additional native-clock, centering, reference-view, default replay, and checkpoint tests pass. The remote legacy checkpoint gives exactly the original predictions for both mean and sampled inference. All arms and the old default have exactly equal initialized parameters. Real GPU preflight checks the longest 16 fit clips for finite learning, correct gradient routes, B0 freezing, content-input isolation, and exact optimizer/RNG restoration. Three concurrent 120-update small-data fits and 16-clip smoke evaluations must pass before formal launch.

Remote directory: `/root/kinetalk_phase43_factorial_20261008`; arms `a`, `b`, `ab`; one shared frozen `code` directory with per-arm hash bindings. Initial packaging preflight omitted an old diagnosis script; dependency was added and bound. Original failure/log preserved. Use `preflight_state_v2.json` and `preflight_v2.log` for the corrected preflight; do not rerun the original deployment script.

At this record: preflight running, formal training not yet launched. Read the latest launch receipt and append-only updates below before reporting status.

After each arm: full 1,367-clip raw/clip metrics, four fixed probes, per-class/per-speaker scores, native dynamics and range, frozen reference/time interventions, original-SHA backup, eight fixed-clock videos. The collector then updates `docs/PHASE43_VIDEO_GALLERY.md` and `docs/PHASE43_EXPERIMENT_TABLES.md`. Trained ablations and inference interventions stay in separate tables. These are finite job continuations, not model selection.

## Interpretation limits

Phase42 improved geometry but worsened F1 and jaw range; it is not the selected new default. Phase43 has no performance result yet. Neutral supports may only reveal habitual neutral articulation; they do not uniquely identify an actor's full emotional style. The posterior sees GT residuals and may still encode B0 errors. Centering and reference exposure do not prove content disentanglement, correct style transfer, or audio-GT dynamic correspondence. Independent phoneme readout, target-style validation, matched-budget baselines, multiple seeds and public-benchmark evidence remain needed for paper claims.

## Preflight completed

All three 120-update fits passed (last/first reconstruction ratios 0.228, 0.230, 0.228). All three longest-16-clip/432-frame gradient and exact-resume checks passed. GPU allocation peaks approximately 3.4–3.5 GB per process; all three 16-clip evaluations passed. Formal launch follows source snapshot and concurrency checks. Small-data fit is a correctness check, not improved validation performance.
