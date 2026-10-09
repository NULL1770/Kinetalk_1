# Phase56 paired target sensitivity — complete

2026-10-09. Diagnostic code1378447 pushed before dispatch.33local/33remote tests,24clip smoke and full2024 TRAIN-fit +281 internal speaker +278 internal sentence audit complete. Worker2361 finished in about65seconds including smoke/testing. No neural training, no B0 forward predictions, no query audio reading, no development query inference or sealed data.

Original report SHAac424e3017b0e69642071381508ced588538e6ffa7e04497c4117c1215195f9f.206original/source files10764854bytes SHA-verified locally.193source bindings; pair/event hashes match Phase54. Neutral B0 state exact. Original per-clip/channel observation and adjacency counts match Phase50; centered and displacement expression energies agree within4.17e-17 over2583clips×3regions. This is independent arithmetic/provenance verification, not a new model gain.

## What was tested

The safe manifest references an MFCC alternative alignment, but all2583 selected alternative artifacts are absent. The initial suggestion that both saved alignment outputs could be compared was corrected before execution. Available path-p95 fields describe aligner disagreement, not annotated ground-truth timing error. Boundary/viseme labels are missing in the inventoried release.

Use existing approved neutral teacher on native source clocks. Expression difference=queryGT−alignedneutral. Hold query fixed and shift only the neutral teacher by±1native frame (40ms); require common original teacher/source/event/channel support and continuous native intervals. This is a stress test, NOT an estimate of actual alignment uncertainty/noise.

Fixed1/5/11frame windows (40/200/440ms sample supports), no smoothing selection. Every sample and interval in a window must be valid. Center compared signals per channel on identical support to exclude static posture. Moving-average and residual energies are nonorthogonal; they do not partition causal expression or spectral power. Local standard deviation is amplitude variability, not emotion semantics.

## Coverage: mouth support fragments matter

Mean per-clip coverage, percentage of the original safe observed channels/frames:

|Split/region|5frames|11frames|
|---|---:|---:|
|TRAIN mouth|41.10%|11.14%|
|Held speaker mouth|49.92%|17.11%|
|Held sentence mouth|39.20%|9.48%|
|TRAIN brows/eyes|94.82%|87.79%|
|Held speaker brows/eyes|94.94%|87.92%|
|Held sentence brows/eyes|94.39%|86.50%|

Original mouth safe coverage across the full native clip is53.53%/57.98%/53.32% (TRAIN/speaker/sentence). After strict5frame windows it is23.03%/30.54%/21.62%; after11frames6.66%/11.23%/5.57%. These are clip-equal averages of direct fractions, not the product of separately averaged percentages. Existing masks are unchanged and no invalid frames are filled.

These masks define the approved pairing support; they are NOT a hard mouth mask in the deployed renderer. Full native expression/receiver training has its own original observation mask. This audit does not show that final generated mouths are being masked.

## Sensitivity: squared error ratios, not noise shares

Mean per-clip ratio of average±1-frame change to original signal energy on matching support. Smaller is less sensitive; ratios can exceed1, especially with small original energy. They are not causal percentages. Full tables retain absolute errors, valid n and medians.

|Split/region|Native position ratio|Native displacement ratio|5frame displacement ratio|11frame displacement ratio|
|---|---:|---:|---:|---:|
|TRAIN mouth|.1461|.4547|.2227|.1896|
|Held speaker mouth|.2268|.6031|.3935|.4327|
|Held sentence mouth|.1540|.4876|.2334|.1709|
|TRAIN brows|.0553|.3210|.1056|.0639|
|TRAIN eyes|.0783|.4019|.1352|.0823|

Mean5frame local-std shift ratio: mouth .1142/.2082/.1258 versus brows .0583/.0596/.0602, eyes .0832/.0789/.0715 across TRAIN/speaker/sentence. Larger windows have different retained clips/support; their lower ratios cannot be attributed entirely to smoothing. Mouth11frame centered position ratios are unstable (mean.944/1.223/.943 versus medians.142/.214/.135), so do not report only favorable mean/std/displacement rows.

5frame centered-energy ratios on matching support: TRAIN mouth.7332, brows.8335, eyes.8230. The ratio is descriptive and does not show that removed motion was noise. Longer smoothing also discards valid expression dynamics.

## Gate checks do not supply a simple quality threshold

TRAIN whole-clip mouth event gate passes190/2024, with native coverage72.92% versus51.52% for the1834nonpassing clips. Both still use the original local event mask. The native displacement shift ratio is higher in the passing group (.558 versus.444); on held speakers .807 versus.524. This comparison is confounded by signal amplitude, emotion and existing sample selection; do not call the passed pairs worse or the failed pairs better.

Spearman correlation between path-p95 and normalized native position shift sensitivity is−.242/−.049/−.042 across TRAIN/speaker/sentence. Original event agreement correlation is+.166/+.303/+.151. This is not evidence that worse alignment helps; it shows these already-selected, amplitude-dependent diagnostic ratios cannot be used as a simple new quality ranking. Per-emotion/fold/gate tables are retained.

## Decision

Do not impose a shared long smoothing window across the mouth and upper face, loosen event masks, or drop more classes to improve statistics. Do not directly treat every paired fast mouth difference as precise emotion supervision. Conversely, these results do NOT prove that the current u target is corrupted: current posterior training uses native GT−B0, not this directly aligned-neutral expression difference. The audit evaluates a proposed alternative target and its limitations.

The justified next design direction is to keep content timing/B0 correction at the receiving side, while testing an emotion/prosody-only student's ability to predict stable expression descriptors under actual available support. Pairing can provide bounded evidence for expression amplitude but cannot furnish dense universal framewise emotion truth. Native real motion and independent neutral references must remain available for regions/frames without valid paired supervision. A replacement objective needs a controlled learnability check and comparison to existing latent/student targets; it is not justified merely by these smoothness ratios. No repeated widening-u/676Dinteraction/named4state/gain sweep is queued.

Latest actual generation remains Phase53 style-only mainF1.676805/MBE.763407 and joint.690150/.823704. Phase56 has no new deployment metrics, no checkpoint promotion or new video rendering; the existing24Phase53 emotion/style videos remain the current outputs. No SOTA claim. Figure generation remains deferred.

## Artifacts and recovery

Full data: final_experiment/evaluation/diagnostics/phase56_target_sensitivity_20261009. Per-clip statistics are compressedJSON, not raw motion exports. Tables: final_experiment/paper_tables/phase56_target_audit_20261009/target_sensitivity_compact.csv and target_sensitivity_all_groups.csv. No running training/audit remains.

Validation status: results ANALYZED; original hashes/frozen state and independent physical replay VERIFIED.11/11 statistical interpretation risks considered: all group/fold differences retained; no individual-level or causal inference; approved-pair/collider selection disclosed; base-rate/subgroup counts retained; no regression-to-mean improvement claim; rejected outcomes retained; no parameter search or post-hoc threshold; associations and reverse dependence on signal energy disclosed. No significance claim.

Draft errors fixed before dispatch: 1D/2D mask mismatch, uncentered static energy in retention, incorrect hand-calculated rank-test expectation. Initial metadata inspection overestimated alternative file availability; missing files disclosed. A local verification draft repeatedly decompressed NPZ arrays and was interrupted; a cached-array rerun verified all physical controls in under2seconds, preserved as original_physical_replay_v2.json. Source data/checkpoints unchanged.
