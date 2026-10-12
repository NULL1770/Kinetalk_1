# Return to the user's three contributions after v6 direct regression

## Material Passport

2026-10-12. Scope: interpretation and candidate architecture, not a new implementation or novelty claim. Inputs: completed direct seed47 development results, fixed native videos, six-paper primary-source notes from architecture_proposal_20261010_v4. No test read.

## User's intended three contributions

1. Redesign residual generation so it is more than DESTalker's neutral-base plus residual.
2. Redesign the teacher/student architecture to improve emotion and emotional dynamics; the teacher may have a different role than Phase53.
3. Joint generation and disentanglement of articulation, emotion and style. Single-reference style must work; it need not be a separate claimed innovation. MEAD+HDTF supports content/style coverage rather than forming an architectural contribution by itself.

The prior project already recognizes one-to-many motion and uses posterior/prior learning. Adding marginal Gaussian variances or a teacher/student label cannot establish contribution2. A frequency split, factor-named heads, or input-path separation cannot establish contribution3. The full52D structural mouth-freedom test confirms capacity, not emotional mouth competence.

## Current negative evidence and its limits

Direct final768updates: full1367 sample/clip_all lip3.305541mm, emotion-motion probeF1.403719, upperfaceEVE2.615629, closureF1.382824. Both Phase53 and v5control are better in these aggregates. Fixed angry/happy videos show weaker happy smile and persistent angry widening mismatch. Dynamic cosine error is worse across all3draws.

Internal128 student classification accuracy.867188 is much higher than the full1367 generated-motion probeF1.403719. Different examples, tasks and metrics prevent a formal attribution. It motivates inspecting the actual condition-to-motion response instead of presuming audio recognition is the bottleneck.

The redesigned decoder starts from B0 with fresh response heads; the old decoder had much more training.768steps do not settle convergence. However, training budget alone does not excuse regression or justify a long run. It simultaneously changes bottleneck, decoder, default intensity transformation, distribution behavior and several losses, so this run cannot isolate the cause. The24D content code has no phoneme supervision; marginal Gaussian sampling does not learn onset/hold/decay or validate conditional temporal covariance. Neutral-only content anchoring plus full52D output allows expression/style paths to compensate for articulation errors. Algebraic zero anchors alone do not establish semantic roles.

Remaining queue manually stopped after the user questioned the direction. Completed direct evidence preserved; posterior seed47 has resume step384, SHA8d3acad86f825cfcb002b25d4a3aaae4a508236d6f4071db7189cdbfa3adbc9c. No posterior final result or seed91 effect conclusion. Stop receipt records owned supervisor81449/worker82426; no implicit restart.

## Dynamic concept suitable for a defensible paper

Separate semantic affect condition (emotion/ordinal strength), observable expression evolution (mouth-corner/eyebrow/eyelid geometry and activation), and one performance's ambiguous motion realization. Observable dynamics may include attack, sustained expression, relaxation and multi-region coordination. Blinks are facial dynamics and may correlate with affect/context; blink generation alone is not evidence of emotional dynamics. Temporal variance alone cannot establish realism: noise can increase variance.

Automatic mode models p(expression evolution, local events | query audio, affect condition, independent personal reference). It should fit observed event/trajectory likelihoods across examples, rather than require every prior draw to match the one recorded performance frame-by-frame. A privileged TRAIN posterior may reconstruct that observed performance. Audio mean/deterministic predictable components may use reconstruction targets. This is already a known modeling stance, not the innovation by itself.

## Candidate redesign aligned to contribution1

Use an improved audio articulation proposal plus expression-conditioned full52D residual, with selective protection of articulatory geometry that is reliably observable from audio. Protect high-confidence closure timing/central aperture/rounding proxies with soft tolerance, while permitting emotional corner widening and personal amplitude. All channels remain available to emotion/style; no category-specific channel mask. The constraints should depend on phonetic context and uncertainty, not enforce the whole neutral mouth as GT. Existing four geometry coordinates and an unsupervised24D bottleneck do not yet implement this idea.

Distinct candidate claim: retaining phonetic geometry/timing under strong emotional residuals, demonstrated by a lip-quality versus expression-strength tradeoff and comparisons with ordinary lip loss/base+residual. Geometry constraints/projection are established methods; task-specific coupling and actual advantage require broader novelty checking and matched ablations.

## Candidate redesign aligned to contribution2

Train a privileged motion posterior and expression decoder to encode global affect semantics, observable expression strength/trajectory, and local activity structure, with reconstruction and known MEAD labels. Inspect posterior reconstruction before asking the audio prior to learn it. Keep teacher output in the actual decoder's representation; avoid supervising a precursor then re-scaling into unrelated coordinates. Once adequate, freeze or deliberately stabilize this representation and train an audio prior against it. Teacher has a real teaching role: learned expression representation, decoder response and trajectory/event posterior, not merely renaming a threshold parser.

For predictable expressive evolution, train supported state/trajectory distributions and their decoded consequences. For ambiguous transient events, model onset, duration/amplitude and coordinated multi-region context with proper likelihood. Do not convert full observed random event timing into a universal audio regression target. Posterior/prior/CVAE/semi-Markov ingredients are established; a calibrated duration/context-aware teaching scheme with measurable decoder/dynamic gains is the candidate contribution, not a claim of first use.

## Candidate redesign aligned to contribution3

Allow cross-factor interaction through a shared composer, with separate conditioning responsibilities verified by controlled changes. Same-speaker, distinct-sentence/emotion reference episodes teach stable personal response. Use supported cross-reconstruction only when content/emotion alignment and target pairing are valid; HDTF unknown affect labels stay masked. Distinguish same-person reference consistency from making all people identical, and do not demand a false framewise donor-target GT. Demonstrate varying emotion/strength changes expressive mouths while keeping articulation timing, varying reference changes personal response while preserving emotion/content, and varying random seed changes performance realization while retaining the former factors.

A shared decoder may be appropriate: removing every shared layer is not equivalent to disentanglement. Aim for identifiable conditioning and appropriate supervised/intervention evidence, not arbitrary pure-channel outputs. Exchange/cycle/control invariance are established in MEDTalk/Mimic/SubtleTalk; attribute those ingredients and verify the final coupled design's distinct task/benefit.

## Validation before another long training

Use matched TRAIN/internal-dev diagnostic batches to evaluate teacher posterior reconstruction, decoder emotion/intensity response and phonetic interference, initialized/migrated responsibilities and gradient contribution. Then change one architectural hypothesis at a time with matched budget. Do not retune against the fixed two videos or test set. Evidence needed: geometry/LVE/synchrony, generated emotion probes plus independent rendered-emotion/human assessment, appropriate intensity/time proxies, dynamics amplitude and duration/rate/interval/velocity statistics, likelihood/calibration, quality/diversity, reference transfer and leakage. Neither blink-F1 nor clip emotion-F1 alone proves temporal emotional expression.

Bibliographic scope: existing primary-source notes verified MEDTalk uses motion-derived intensity and audio+text, SubtleTalk has deterministic prior/residual flow/region-control invariance, CETalk has continuousVA/fast-slow paths, DESTalker neutral-base residual, Mimic and DiffPoseTalk style encoding/control. Exact SubtleTalk FDD formula remains unavailable in the reviewed supplement scope; do not label a custom formula as faithful reproduction. No guarantee of novelty or superior performance until broader checking and matched measured benefits.
