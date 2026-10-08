# Phase51: diverse neutral support and compact posture/response style

2026-10-09. User-authorized optimization; freeze neutral B0 and772D emotion/prosody prior. Pre-change snapshot3568e58 already pushed; this plan/results archive must be pushed before model changes.

User clarified during training: the objective is a scientifically appropriate, actually learned reference-driven speaker motion style, not a required posture/response split. The two arms remain bounded implementation candidates for that same objective. Temporal may be preferred if it works better; no paper contribution or successful disentanglement is implied by the split. Updated definition/training/evidence plan is in STYLE_DEFINITION_AND_EVALUATION.md. Neither arm changes the neutral-only reference information budget. Do not add a new module before evaluating what the current training actually learns.

## Rationale and prespecified arms

Phase48 shows partial directed transfer but same-person A/B variability76.4%ofcross-person. Phase49 shows20fixed enrollment identities are nearly memorized while held codes vary. Phase50 shows full residual mixes expression, posture and neutral-B0 error.620 legitimate TRAIN-fit neutral clips allow varied same-person supports without dev fitting.

Two fixed seed47,8epoch, batch16, lr1e-4 arms:

1. `temporal`: original3block64D reference encoder; train style+decoder using diverse TRAIN neutral references and frozen p_mean conditions.
2. `statistics`: small reference encoder with32D posture code from supported GT/B0 means and32D response code from centered GT/B0 RMS and covariance. Aggregate independent support statistics before two small MLPs. Posture code enters only the explicit52D static bias; response code enters the four decoder modulations together withg/u. No temporal order/content token encoder in style. Features are descriptive, not provably pure identity.

Each target uses two independently sampled same-person TRAIN-fit neutral clips; exclude target clip and target sentence. Neutral supports are drawn only from the original TRAIN-fit fold, never internal-held or external query data. Original enrollment supports remain unchanged for all evaluations. Order/reference RNGs independent and saved for resume. No reference-derived per-target GT supervision at deployment.

Train only style+decoder. Freeze B0/prior/posterior/emotion/intensity tensors and gradients. Use existing normalized position +.5 adjacent displacement objective, all supported mouth channels open. No extra semantic/style consistency loss or manual gain. Fixed parent Phase45-u; do not reuse Phase47 correction fitted in the old style coordinates. Both arm budgets include inherited training and analytic fit. The inherited motion posterior is not recalibrated to the new style code: oracle rows, if produced, are uncalibrated diagnostics and must not support conclusions.

## Gates

- Local/remote tests: statistics padding/missing-channel/permutation invariance; static-versus-response decoder isolation; sampled-reference actor/clip/sentence disjointness; unchanged legacy load; gradient responsibility; deployment content-NaN isolation.
- GPU balanced small-data smoke120updates using query32 and independent reference pool, require loss last10<.9first10; frozen modules exact and no gradient escape. Preserve failed roots; no numeric gate relaxation.
- Formal fixed8epochs; evaluate original743speaker/890sentence holds (compact256each per epoch for progress only, no checkpoint selection). Final model fixed. No automatic training extension.
- If smoke passes, start both formal runs sequentially with resumable last.pt. Report measured ETA after first epoch. Do not pretend internal fitting equals development gains.
- After finals: full1367 unchanged native/raw+clip/four-probe/rig scoring, GT/B0/source replay, all emotion/intensity/speaker tables, eight-emotion videos per arm, same-content/emotion style-swap and ownA/B. Compare frozen Phase45/47 historical results; geometry/timing/style jointly, no F1-only success. No sealed tuning.
- Separate two changes only through arm comparison: baseline versus temporal confounds varied supports+style training; temporal versus statistics confounds encoder+style integration. Do not attribute to one isolated module. More ablations needed for paper.

Future paper figures remain deferred per user. Fixed examples/timestamps, high-resolution individual images, style mean/std and neutral asymmetry comparisons are recorded for later.

## Implementation and initialization

Implemented in d8d544e, pushed and independently verified. `ResponseConfig.reference_encoder` defaults to the unchanged temporal path; old checkpoints still load. Statistical branch uses156→128→32 posture and208→128→32 response MLPs, each one SiLU hidden layer. Features are channel-supported GT/B0 means, centered RMS and normalized same-channel covariance with fixed denominator floor1e-6. Equal support weights; absent supports/channels omitted. These are speaking-reference descriptors, not static scans or causal identity ground truth.

Statistical branch inherits B0 receiver input/4temporal blocks/output andg/u modulation columns, initializes new style MLPs normally, style modulation columns and static bias to zero. Temporal control starts from all original weights. Initialization differences are part of the architecture comparison, not a single isolated-module ablation. No Phase47 latent correction reused.

52local checks pass. First remote root stopped before any GPU/model update because the inherited source snapshot omitted an existing legacy test file. Failure retained. Fresh `/root/kinetalk_phase51_reference_response_v2_20261009`,worker49242, binds186source files and passes52remote tests; GPU smoke underway. Read-only status helper `.codex-finalizer/phase51_status.py`. Do not relaunch completed or active worker.

## Verified running state, 2026-10-09 01:43:06 China time

Both120update smokes complete/pass: temporal .260430→.076666(first/last10),statistics .532788→.116705; loss ratios .2944/.2190. Frozen B0/expression exact and HuBERT-NaN isolation pass. Formal temporal has completed7epochs and is in epoch8 at5375updates;146–151seconds/epoch. Statistics follows automatically. Worker is alive;training/cache/internal-check ETA20–30minutes remaining, approximate. Current pipeline stops at training_complete; full evaluation and rendering need subsequent dispatch and storage arrangement.

205 immutable preflight files,50,179,858bytes,186sources are locally SHAverified under `final_experiment/evaluation/diagnostics/phase51_reference_response_v2_20261009/preflight`, including both smoke checkpoint pairs. Transient SSH reset during concurrent transfer recovered with one connection; original sources/checkpoints never changed. Root/data free317,939,712/93,532,160bytes. Final results remain pending; no model promotion. Early temporal TRAIN loss decreases .339823→.329486 but internal-speaker prior position .456846→.486064 worsens; these diagnostic normalized losses must not be mislabeled MBE or F1. Continue fixed budget, report final without selecting an early favorable checkpoint.
