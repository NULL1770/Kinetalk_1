# Phase57: local expression descriptor prediction

2026-10-09. Complete: 43 local and 43 remote tests, 24-clip smoke, then the same 2024 TRAIN-fit / 281 internal-speaker / 278 internal-sentence approved emotional pairs. Implementation c80ec82 was pushed before dispatch. Worker5974 finished in about92seconds including tests/smoke. No neural training, B0 forward pass, external development inference, sealed test or generated video.

## Result and decision

Five-frame pair descriptors do not provide a demonstrated strong replacement for the current teacher target. Existing u contains modest linearly recoverable mouth amplitude variation; this is not zero information, but it does not recover the full dynamic target. Higher-dimensional readouts overfit. Do not add these descriptors as a universal smoothing loss or replace the teacher on this evidence. Do not infer that nonlinear audio prediction is impossible.

Every number below is a diagnostic error reduction against a per-clip constant baseline after centering on the original supported native slots. Negative means worse than that baseline. This is NOT F1, MBE, generated-motion improvement or a fraction of causal emotion explained.

|Feature / mouth target|TRAIN fit|Held speaker|Held sentence|
|---|---:|---:|---:|
|u16 / local mean|2.43%|-1.94%|1.99%|
|u16 / local std|4.54%|1.80%|5.38%|
|hidden128 / local mean|4.47%|-4.69%|-0.16%|
|hidden128 / local std|7.37%|-1.92%|3.58%|
|audio772 / local mean|17.48%|-14.06%|-7.73%|
|audio772 / local std|20.98%|-13.53%|-3.61%|
|prosody4 / local mean|0.23%|0.07%|0.10%|
|prosody4 / local std|0.54%|0.20%|0.77%|

u16 upper-face reductions on held speaker/sentence: brow mean2.95%/0.72%, brow std1.71%/0.26%; eye mean0.73%/-0.01%, eye std-0.17%/-0.41%. Mouth u std is negative in two held-speaker emotion groups; all seven emotional groups remain in the report. No significance or all-emotion success claim.

On identical forward/reverse support, u mouth local-std forward MSE is87.75%/89.59% of reversed MSE (speaker/sentence). The corresponding local-mean ratios are95.05%/94.60%. Temporal order therefore affects this fitted readout. Beating reversal is insufficient by itself: audio772 also beats reversal while losing to the constant baseline. These are two different comparisons and must not be conflated.

## Support and protocol

Targets: signed mean and population std of queryGT minus aligned neutral in fully observed five-frame native windows. Both features and targets are centered per clip/channel on the same support. Predictors are frozen u16, prior hidden128, normalized emotion2vec/prosody772, or its last4 prosody channels, each averaged over five audio-valid native slots. Fixed clip-equal ridge .001; only TRAIN-fit contributes coefficients/scales. No B0, HuBERT, identity label, query GT or emotion/intensity label enters the predictors.

Original teacher/event hashes equal Phase56. Original per-region observation counts match every one of2583 clips. Excluding channels with fewer than two valid descriptor centers removes162 mouth channel-frames total. Effective mouth clips are2013/280/277; all original clips remain listed and unsupported targets are explicitly absent, not filled. Brow/eye counts remain2024/281/278.

|Split|Mouth descriptor coverage of original safe support|Coverage of full native clip|
|---|---:|---:|
|TRAIN|41.09%|23.03%|
|Held speaker|49.91%|30.54%|
|Held sentence|39.18%|21.62%|

Upper-face safe-support coverage is94.82%/94.94%/94.40%. This asymmetry limits any proposed shared target. Pair differences still contain alignment and independent-performance differences; neither smoothing nor a good readout would make them pure emotional ground truth. The actual existing posterior uses nativeGT-B0, so these results do not establish contamination of the existing teacher.

Reverse controls mirror full native indices within each clip's actual length, retain gaps, and intersect forward/reverse feature masks before scoring both against the same unmodified target. They do not retime generated speech or fit an alternative predictor. Parent and neutral B0 state hashes remain exact; HuBERT-NaN isolation passes.

## Implication for optimization

Stop this paired-fast-difference / fixed-descriptor replacement family here. Phase54–57 do not support simply widening u, adding the failed B0×emotion676D interaction, or universally smoothing motion. No additional loss/module or generator training is queued from these readouts.

The practical remaining issues are still the Phase53 native mouth timing/amplitude tradeoff and insufficient semantic/dynamic generalization across speakers. A future candidate must improve the actual generator, retain independent neutral content supervision and reference-style checks, and compare against the original teacher/student objective. Any nonlinear learnability experiment must be described as new evidence, not justified by treating this linear failure as an impossibility theorem. No particular new architecture is yet validated or promised to succeed.

Current actual generated metrics remain Phase53: style-only MBE.763407/F1.676805; joint MBE.823704/F1.690150. There is no new video or deployment metric this round. Paper-wide SOTA, matching baseline budgets, multi-seed evidence and the final figure set remain incomplete.

## Files and reproducibility

Remote: /root/kinetalk_phase57_descriptor_predictability_20261009; completed, do not restart. Local originals: final_experiment/evaluation/diagnostics/phase57_descriptor_predictability_20261009.214 original files31,536,754bytes SHA-verified,195 source bindings. Report SHA386deee1259965a00d477d56fe29c60cdfa826149905352fa061db4b5bed51fa. Source/smoke/formal reports, coefficients and per-clip errors are preserved.2,016 summary fields replayed from saved float32 error arrays with declared rounding tolerances; support independently replayed against Phase56.

Tables: final_experiment/paper_tables/phase57_descriptor_audit_20261009,2304 all-group rows and96 compact rows. Descriptive diagnostic/appendix material only, not a new method ranking. No active training remains. Latest observed root free space60,903,424bytes; recheck and SHA-archive obsolete artifacts before any future training.

Operational errors: a read initially used the parent directory; a Windows glob was passed literally to rg and then corrected to -g; the first support replay used the wrong JSON key and was fixed before writing its successful receipt. None altered the remote experiment or its results.
