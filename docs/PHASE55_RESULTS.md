# Phase55 conditional receiver probe — complete, rejected

2026-10-09. Git5642ff9 pushed before dispatch.23local/23remote checks,24clip GPU smoke and full2024 TRAIN-fit +281 internal speaker +278 internal sentence diagnostic complete. No neural model training, development query inference or sealed data. Worker22650 complete; do not relaunch.

Original report SHA58c0ec636ac04dd33ea0c1b04720c93390c1b74a0d7e52981bb0fd81ab3b3c66.17original files15500594bytes and191source files2061122bytes SHA-verified locally. Full diagnostic output14272677bytes. Parent/B0 frozen, content-NaN isolation and hidden-to-u mapping pass.24saved B0 control arrays (errors, supports, weights, energies, counts) are exactly equal to Phase54, plus all control summary rows.

## Fixed experiment

Receiver features: normalized centered B052 plus B0 multiplied by audio-prior emotion8 and intensity4 probabilities (676D). Same fixedridge.001, clip/channel equal weighting, TRAIN-only scaling, approved pair masks, native25fps gaps, centered/adjacent targets as Phase54. Only base and conditional features rerun; previous other controls retained. No labels/GT are input to the feature map. No content enters the student. Probabilities derive from frozen prior global mean. Probe coefficients never modify a model.

## Result

Mouth expression-difference prediction relative to zero dynamic prediction. Values are MSE/zero-MSE (lower is better,1 means no gain), not final model metrics or causal variance fractions.

|Split|B0 centered|Conditional centered|B0 displacement|Conditional displacement|
|---|---:|---:|---:|---:|
|TRAIN|.854760|.796780|.877920|.865240|
|Held speaker|.976310|11.938841|.984830|4.976862|
|Held sentence|.822090|11.316447|.842860|5.919779|

Conditional held speaker centered physical MSE=.04944092 vszero=.00414118; displacement=.00380425 vs.00076439. Held sentence centered=.07451207 vs.00658440; displacement=.00899648 vs.00151973. TRAIN improvement does not generalize.

Failure is strongly concentrated in contempt: its centered ratio82.985/80.198 and displacement28.335/42.357 (speaker/sentence). Other emotion ratios vary: held speaker centered expression .949–2.395 excluding contempt; held sentence happy2.358 while other non-contempt classes .761–.882. These subgroup results describe failure; do not remove contempt or tune a gain/regularizer on holds to hide it.

B0 neutral-base displacement reduction remains30.79%/20.65%, replayed exactly. Its stronger association with this target partly reflects that B0 is itself subtracted in the target; it is not an independent causal instrument and does not prove current student leakage. The older B0 was trained with original TRAIN safe neutral targets, including later internal holds.

## Decision and limits

Reject this unbounded conditional linear feature as an immediate model change. Capacity, correlated probability interactions, TRAIN/held distribution shifts and fixed regularization can cause unstable extrapolation; this experiment has not isolated their contributions. It does not establish that every bounded nonlinear emotion-by-content response must fail. The current neural receiver already conditions on B0/g/u; this was a diagnostic, not proof that it lacks all interactions.

Together Phase54/55 show weak linearly recoverable paired expression dynamics and an unsuccessful conditional probe. They do not establish that simply swapping the teacher target to GT-alignedneutral will solve the task. That target contains alignment and independent-performance differences; directly assuming it is pure emotion would repeat the same error in a different coordinate system.

Next priority is a bounded reliability audit of the proposed paired dynamics target against timing/alignment quality and frequency. Compare stable expression amplitude/envelope statistics with native fast differences, using TRAIN/internal holds only, without selecting a smoothing gain from development. Retain prior Phase45/50 evidence and Phase37/39 named-state failures; do not repeat them as new discoveries. A new teacher/student objective should follow a demonstrated reliable target, with no B0 emotion supervision, no content input to student, and no full-motion gradient into student. No new neural run or model promotion is justified by these probes alone.

## Verification scope

Results ANALYZED, artifacts and exact controls VERIFIED; no independent neural rerun or significance claim.11/11 interpretation risks checked: emotion group variation retained; no individual-level inference from averages; approved-pair selection and conditioning limit generalization; per-emotion base rates retained; no regression-to-mean improvement claim; failures retained; no feature/regularizer search; follow-up identified as new; association not causation; B0-containing targets and reverse-direction ambiguity disclosed.

Implementation notes: draft conditional code originally shadowed the B0 module with a feature tensor; fixed before dispatch. Replaced a tautological shape test with synthetic held-sequence conditional-response recovery and exact probability-block checks. No failed draft was used to generate these results. One local status-print NameError happened after a successful remote read; it did not affect worker or reports. Git relay failed repeatedly; direct HTTPS push with proxy cleared succeeded. Planning/source/report snapshots retained; unrelated third_party/voca_reference untouched.
