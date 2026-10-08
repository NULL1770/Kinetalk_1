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

## Current candidate architecture and data

| Part | Inputs and model | Learning responsibility |
|---|---|---|
| Neutral B0 | Original frozen neutral checkpoint; speech content features | Supplies the neutral articulation sequence, with no gradient updates in this experiment |
| Audio prior p | emotion2vec768 + prosody4 only; 128-wide encoder, 4 temporal convolution blocks + 2 Transformer layers (4 heads); global32, local16 at stride2 | Global emotion/intensity supervision and normalized q/p Gaussian KL; query-motion reconstruction cannot backpropagate into p |
| Motion posterior q | Observed GT residual, B0, adjacent residual displacement, channel support, detached style64; 128-wide encoder, 3 temporal blocks + 2 Transformer layers | Training-only inference from motion; stochastic q reconstruction, KL, and global semantic supervision; not an emotion-only oracle |
| Reference style | Two independent same-person neutral supports; residual/B0/channel support156; 128-wide 3 temporal blocks, projection64, mean across observed references then 2-layer MLP | Learns through same-person query reconstruction; no speaker embedding table, no new identity classification loss |
| Response decoder | B0 52 to hidden192; 4 modulated temporal convolution blocks; conditions global32 + local16 + style64; full52 output and style bias | Produces expression and mouth-amplitude changes on the original clock; all observed mouth channels remain available |

Total newly trained parameters: 2,176,948 in each arm. B0 is external and frozen. q, p, style and decoder train together from fresh initialization; the prior receives KL/semantic gradients, while an additional detached-prior reconstruction term starts after epoch4 to train the decoder for deployment inputs. No new objective was introduced in Phase43.

The existing TRAIN set has 12,536 query clips; the fixed internal split uses 10,903 fit clips, 743 held-speaker clips and 890 held-sentence clips. The external development set has 1,367 clips. There are 50 independent enrollment clips (two for each identity); query/enrollment clip and sentence overlap is zero in the audited data. Native query motion supervises expression response; neutral B0 supervision is not changed. No new pair-exchange training was added, and low-quality pairs are not newly admitted. All new normalization uses the fit split only. This is a development comparison, not the final all-data paper training.

## Preflight completed

All three 120-update fits passed (last/first reconstruction ratios 0.228, 0.230, 0.228). All three longest-16-clip/432-frame gradient and exact-resume checks passed. GPU allocation peaks approximately 3.4–3.5 GB per process; all three 16-clip evaluations passed. Formal launch follows source snapshot and concurrency checks. Small-data fit is a correctness check, not improved validation performance.

## Formal launch dispatched

Implementation snapshot b34c7edd67ed9600a5dc6154c88447a918ba8598 pushed. Remote PIDs A=6344, B=6345, AB=6346. Local finite collector PID39796. The source-bound preflight/launch backup has 208 members, all original SHA verified. Root free at launch 2,335,264,768 bytes; conservative sum of longest-batch GPU allocation peaks 10,418,202,624 bytes on 25,757,220,864 bytes total GPU. Each arm uses its own checkpoint/log/binding and shared immutable code. At dispatch models first cache frozen B0; do not treat pipeline status alone as proof of optimizer progress.

The local table formatter has only an LF CSV line-ending change after the remote code freeze; it does not participate in training/evaluation. Remote source SHA bindings, not the Git commit alone, identify exact executed files. The old preflight failure is preserved in the backup.

## Optimizer progress verified

All three arms reached at least 101 actual optimizer updates (epoch1), with finite losses/gradients. Initial model digest is identical in all three formal runs: `346bde357868122a52e3aab963b2278c128798d1259f5daa977ba491dd6d5356`. At this observation A/B/AB cost 0.471/0.559/0.506 seconds per update; total GPU usage 9,183 MiB and utilization 99%. Current estimate: 3–4 hours of training, with extra allowance for the prior reconstruction term after epoch4; full postprocessing follows. These speeds are early estimates, not a completion guarantee or a quality result. Latest original snapshot is saved at `final_experiment/evaluation/diagnostics/phase43_factorial_20261008/last_remote_snapshot.json`.

## 2026-10-08 SSH restart recovery

The user reported a server restart and requested all three arms be resumed. The GPU was idle and no old training workers survived. Each last.pt was intact at epoch3/step2046, with finite model and optimizer state, complete RNG state, and matching source/config bindings. Epoch4 partial updates were not checkpointed and must replay. Original logs/states/checkpoints were preserved remotely under restart_v1 and backed up locally (19 original-SHA-verified files). The frozen model/trainer/evaluator sources and all budgets remain unchanged.

New workers A=1839, B=1840, AB=1841 use the existing trainer with --resume, continuing epochs4–24. Recovery worker SHA e2708aedd77e39be3fe7bcc83202a624ec233e50e440f3fd54e3c478cb84d2e6. Local collector39796 and its arm-A child130328 survived; reused without duplicate launch. All jobs first rebuild frozen B0 caches. Verify actual post-restart updates with .codex-finalizer/phase43_verify_resume.py; its first resumed update is compared with the original epoch4 first update. Do not rerun phase43_resume_v1.py: it refuses the existing recovery directory.

Recovery verified: all three resumed actual optimization at epoch4/step2047. Every logged loss component and gradient norm of that first update matches the original pre-reboot step2047 exactly (timing excluded). Optimizer/RNG restoration is therefore verified on the real formal trajectory, not just a smoke test. GPU utilization99%, three workers alive. Remaining training estimate approximately3hours, allow3–4hours before reviewing final metrics/videos. Verification receipt: restart_v1/runtime_verification.json.

## Completion supersedes launch/restart status

All three arms completed24epochs/16368updates/full1367 on2026-10-08.576 original-SHA files and24videos verified. Read PHASE43_RESULTS.md for tradeoffs and PHASE43_EXPERIMENT_TABLES.md for all nine method rows. No joint winner/default promotion/new training. Do not rerun completed pipelines or collectors.
