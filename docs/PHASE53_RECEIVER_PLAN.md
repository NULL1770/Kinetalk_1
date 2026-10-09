# Phase53: receiver path diagnosis before correction

2026-10-09. Pre-change snapshot c8ff428 is already pushed. User authorizes continued optimization, code snapshots, training/evaluation/rendering. No promised gain.

1. Diagnose Phase51 statistics and Phase52 on 96 emotion-balanced clips each from original TRAIN fit, internal speaker hold, internal sentence hold. Same B0 canonical batch32, fixed independent enrollment references, frozen four probes. Never infer external validation or sealed test during this selection.
2. Intervene static bias, style modulation, global g and local u; report raw and clipped motion, position/adjacent-displacement error, jaw range/correlation and four probes. Zero paths are OOD diagnostics, not trained ablations or selectable models. Confirm exact normal-path replay and immutable checkpoints.
3. Use measured path effects plus already completed Phase50 approved-pair evidence to choose one coherent correction. Preserve neutral B0, 772D affect/prosody-only prior, no motion gradient into student, native masks/clocks and real reference-driven motion style.
4. Before training ensure disk space by SHA-backed archival of completed runs' redundant optimizer last.pt files; retain final checkpoints, all data and parent dependencies. Verify local and remote copies before removal.
5. Run meaningful code/boundary tests, fixed 120-step GPU smoke, then formal budget. Queue unchanged full1367 evaluation, 2026 matched style audit, all8emotion and style videos, complete paper tables. Keep failed outcomes and do not claim SOTA or merge checkpoint metrics.

## Completed diagnostic and fixed next experiment

Original diagnosis completed on 288 TRAIN/internal-held clips per checkpoint; original report SHA is recorded in the download manifest. Phase52 style pre-nonlinearity RMS is about1.77–1.92 vsPhase51 .22–.28; gain saturation47%vs18%. Removing style modulation increases held jaw p95-p05 .23150→.26290 (speaker) and .18497→.22655(sentence); mainF1 .6290→.7045/.5817→.6501. Geometry error worsens. Removing static bias improves many probe scores but is not uniformly beneficial. These are bounded OOD interventions, not causal disentanglement proof or new model results. Different range quantiles/subsets cannot be compared to the full evaluation range.

Implement factorized reference modulation: retain the existing affect gain/shift transform, apply reference gain/shift through a separate bounded transform before each original temporal block. Both reuse the original .1 modulation scale. Reference magnitude cannot move the affect gain inside its tanh saturation. No additional network/loss or anatomical masking. The existing statistical reference offset receives gradients from neutral queries only; forward values are label-independent, and emotional query motion still trains the response path on every observed channel. This prevents that offset's parameters from fitting each actor's average expression. It does not guarantee content disentanglement.

Two fixed8epoch/seed47 controls from the same Phase45-u parent: factorized style-only vsfactorized joint receiver. Both neutral-offset routing, same supports/native position+.5displacement objective. This isolates the effect of allowed receiver updates under the new conditioning. Comparison to Phase52 bundles factorization and offset routing and must be labelled accordingly; neither component is claimed independently proven. No dev-based hyperparameter/epoch sweep. Tests, GPU smoke then formal training; unchanged evaluation/style audit and16videos for each candidate queued. Model not automatically promoted.

13 completed redundant optimizer checkpoints245553785bytes archived locally with SHA checks before remote removal; allfinal checkpoints retained. Root free508366848bytes after archive. Old collection artifacts unchanged.

Diagnostic report SHA256: e0b7ebb457fce9d37ab7d397f56c38f4db0465de5b0a4839eb074e2b28e360f6. 62 targeted checks pass, including neutral-only gradient routing plus skipping Adam momentum/decay on all-emotional batches, exact legacy behavior, strict serialization, content-NaN isolation and affect sensitivity with saturated reference gain. The first test run exposed zero-vs-absent gradient handling; fixed by explicitly clearing posture gradients on batches without neutral queries and tested after nonzero optimizer momentum.

## Dispatch verified

Git implementation0bd8482 pushed. Worker10345; 62remote tests4.78s. Both120step GPU smokes passed:style-only .691847→.468501(ratio.677175), joint .532791→.116755(ratio.219139). B0/prior exact and content-NaN isolation pass; style-only protected receiver exact. Formal style-only command started; actual optimization step still pending at this snapshot (do not mistake cache startup for training progress). Localcollector52028waits finite3h and queues24videos and alltables. Source/smoke archive inprogress. Neither smoke loss nor internal normalized error is MBE/F1.

## Architecture contract for later paper drawing

Statistical reference encoder unchanged: two independent neutral TRAIN supports; masked equal-reference posture descriptors156D and response descriptors208D, each Linear→128→SiLU→32, concatenated64D. No speaker embedding table or query GT inference.

Decoder unchanged inputB0 52→192,4temporal blocks with dilation1/2/4/8,52D residual output and32→52 static offset. Each existing80→384 modulation matrix is split by columns into affect48=(g32,u16) and reference32. No parameter count added by factorization:
- a = W_affect[g,u] + bias; s = W_reference reference_response.
- h_affect = h*(1+.1*tanh(a_gain)) + .1*a_shift.
- h_conditioned = h_affect*(1+.1*tanh(s_gain)) + .1*tanh(s_shift).
- Original temporal block then final B0 + scaled residual + scaled reference offset, on all observed channels.

The static offset has the same forward value with/without the TRAIN neutral gradient mask. Only its learning is routed; there is no label-dependent inference. With no neutral query in a batch, posture encoder/offset gradients are set toNone before AdamW, so stale momentum and decay cannot update them. Mixed batches receive offset reconstruction gradients only from neutral query rows. Emotional samples still supervise the dynamic reference branch and, in the joint control, the full receiver. These boundaries don't prove content timing preservation: original timing/closure and style audits remain necessary.

Original preflight archive complete:208members41556244bytes,189sourcefiles match binding SHA. Effective trainable parameters105972(style-only) /794344(joint), no added factorization parameters. TRAIN fit10903, originalinternalholds743speaker/890sentence, originalexternaldev1367; fixed25actor enrollment50clips and diverse620neutralfit supports. CollectorPID52028alive/noerrors.

## Actual formal progress

Verified firstcandidate epoch2step683 (not cache/smoke), first682update epoch99.33s. Per-epoch frozen prior/B0, exact protected receiver and content-NaN gates passed before epoch1 checkpoint. Internal normalized speakerposition.464624/sentenceposition.641458 are diagnostics, not final MBE/F1; fixed8epochs proceed with failures retained. Joint formal is queued after style-only, then bothfull evaluations/style audits. Estimated remaining combined training25–35minutes; complete metrics/download/render queue50–70minutes, subject to measured joint runtime. Localcollector52028alive/noerror. 208preflight artifacts41556244bytes SHAverified; table schema preflight passed without altering old results.
