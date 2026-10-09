# Phase54 paired dynamics predictability — complete

2026-10-09. Implementation5128310;21local/21remote tests,24clip GPU smoke, full2024 TRAIN-fit +281 internal speaker-held +278 internal sentence-held complete. Fixedridge.001, clip/channel equal fitting, original native safe masks/gaps, all neural states frozen. No development query inference or sealed data. This is diagnostic linear fitting, not a new generated model.

Original report SHA f8c0d40d970d5175e95eb0087183546e089f6ff8388d5b7f12b9eee3cec5bf1e.17original files30914551bytes and200preflight/source files4873972bytes SHA-verified;191 source bindings. Full output28096871bytes under32MiB cap. Worker18700 complete, no relaunch.

## Held mouth dynamics

Values below are percentage reduction in clip/channel-equal squared error versus zero dynamic prediction:100*(1-MSE/zeroMSE), not causal explained-variance shares. Negative means worse than the zero baseline. Centered target removes its per-clip mean; displacement uses only actual adjacent25fps observations.

|Split / target|u16|hidden128|audio772|B0 control52|
|---|---:|---:|---:|---:|
|held speaker / centered / expression_difference|-0.262%|-1.774%|-5.885%|2.369%|
|held speaker / centered / neutral_base_error|0.592%|-0.586%|-4.742%|-0.699%|
|held speaker / centered / total_residual|1.017%|-0.143%|-3.357%|4.834%|
|held speaker / displacement / expression_difference|0.326%|-1.063%|-4.971%|1.517%|
|held speaker / displacement / neutral_base_error|0.016%|-0.296%|-1.687%|30.791%|
|held speaker / displacement / total_residual|0.310%|-0.720%|-2.990%|27.862%|
|held sentence / centered / expression_difference|1.122%|0.688%|-1.593%|17.791%|
|held sentence / centered / neutral_base_error|0.693%|-0.149%|-3.583%|0.432%|
|held sentence / centered / total_residual|1.985%|1.600%|-0.045%|19.042%|
|held sentence / displacement / expression_difference|0.008%|-0.381%|-2.850%|15.714%|
|held sentence / displacement / neutral_base_error|0.077%|-0.150%|-1.999%|20.652%|
|held sentence / displacement / total_residual|-0.058%|-0.248%|-2.679%|38.626%|

## Interpretation and limits

Current u has little linearly recoverable paired mouth expression dynamics: centered −0.26%/1.12% and adjacent displacement0.33%/0.01% on speaker/sentence holds. Wider hidden/audio probes fail to generalize better under the fixed probe; raw audio overfits TRAIN. This does not prove nonlinear audio predictability impossible or justify simply widening u.

B0 predicts neutral-base mouth displacement differences much more strongly (30.79%/20.65%) than u, while expression differences vary sharply between holds (1.52%/15.71% for B0). Full residual mouth dynamics are partly associated with a content-conditioned correction task. This is not proof that the current student leaks content: its own prediction of that error component is also weak. Phase50 nonorthogonal decomposition and these predictive associations do not establish causal shares.

Upper expression differences also have weak prediction; for centered brows u gives2.27%/0.49%, eyes0.57%/0.14%. Subtracting two independent performances includes alignment and one-to-many differences, so GT-alignedneutral is an imperfect expression target. Current linear probes omit nonlinear emotion-by-content response and reference interactions. Different feature dimensions imply unequal capacities. The old B0 was trained on original TRAIN safe neutral targets including later internal holds; its control is not an independent causal instrument.

## Fixed next check

Before changing the student, test one explicit conditional-response feature map at the receiving side: frozen B052 plus B0 multiplied by the frozen audio-prior emotion8 and intensity4 probabilities (676D total). No query labels or query GT as input; no content enters the student. Same masks/target/ridge/train and held protocol; compare a replayed B0 control to ensure exact evaluation. This tests whether class-conditioned amplitude response explains what an unconditional linear audio probe misses. Coefficients remain diagnostic only; no automatic model promotion, loss, gain fitting on development or architecture conclusion.

## Validation scope

Verification status: ANALYZED results, with original artifact hashes and frozen-state checks verified; no independent full rerun or statistical significance claim. 11/11 interpretation risks checked: Simpson/group variation (emotion breakdown retained); ecological inference (no individual claims); Berkson/collider selection (approved-pair subset, limited generalization); base rate (per-emotion rows retained); regression to mean (fixed ridge/folds, no claim of guaranteed gain); survivorship (all features/failed directions reported); look-elsewhere/forking paths (no sweep or held-fit, follow-up explicitly new); correlation/causation and reverse causality (associations only, B0-containing target acknowledged).
