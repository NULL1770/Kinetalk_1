# v6 bounded factor-path experiment

## Material Passport

User-authorized implementation/optimization. Sources: completed v5 matched256-update pilot, frozen32-clip path interventions, previous six-paper research notes. Status: implementation verified by24 relevant tests; no v6 effect claim. No sealed/test read.

## Fixed hypotheses and roles

- Content-only correction: frozen HuBERT/B0 content front-end -> learned24D bottleneck -> unrestricted52D correction plus4 rig coordinates. Supervised on known-neutral TRAIN motion. It does not receive affect, reference or events. No phoneme labels are available; do not call the latent a validated viseme space.
- Expression posterior: TRAIN motion teacher supplies mean/covariance in the existing10 observable expression coordinates; audio prior predicts the corresponding distribution, emotion category and ordinal intensity. Posterior draws reconstruct observed TRAIN performances; detached-posterior KL and target NLL train the deployable prior, while detached-prior KL regularizes the posterior. A gap-safe9-frame filter produces correlated Gaussian state innovations. Duration/hazard/mark events remain probabilistic. No transcript, GT query motion or event labels at inference. These are model variances, not calibrated psychological uncertainty.
- Anchored composition: independent content response, expression response, multiplicative content-expression interaction, and single-reference posture/bounded response modulation. All responses can affect all52 channels and geometry. Expression response is exactly zero for zero expression inputs, including position/bias cancellation; interactions vanish with expression. Old mixed joint output and mixed geometry heads are deleted. Geometry predictions compose the same factor responses before rig projection.

Default conditioned state is the predicted mean, without the old second intensity scaling. Explicit strength control rescales relative to the predicted ordinal strength. Emotion categories condition the expression path and never select anatomical channels. Emotion, reference and events cannot enter the content bottleneck; this is an input-path guarantee, not empirical semantic disentanglement.

Clarification requested by the user: input isolation does not prevent emotional/style changes to the mouth. Expression, interaction and style responses all predict all52 coefficients plus4 rig coordinates. Emotion can change mouth width, corner lift, lip tension and aperture through its own response and the content-expression product. Style can change mouth posture and bounded content/expression response gains. The content bottleneck stays query-audio-only, while the final mouth combines these paths. There is no hard mouth/non-mouth output partition; preservation of phonetic timing is an empirical goal, not a guaranteed consequence of input isolation.

The frozen raw-direct parent is a TRAIN-only bootstrap target for the first128 updates at weight.25. It is not included in a v6 inference checkpoint. Parent emotion/student/style/event modules migrate strictly; new decoder starts at B0 and must learn replacement responses. This initialization differs from the old parent's joint decoder and must be disclosed in cross-family comparisons.

The KL/NLL terms describe supported marginal Gaussians. Correlated draws add a temporal sampling bias; no exact full-sequence covariance likelihood or ELBO claim is made. The teacher can still encode content-correlated movement in these observable coordinates, so empirical controls must test leakage rather than infer purity from naming.

## Objectives and comparisons

Complete native-clock observed reconstruction, velocity, rig/state/class/event objectives; neutral content anchor.25; posterior reconstruction.5; prior NLL and bidirectional detached KL.1; single-reference relation/latent consistency.05 each and same-speaker alternate-performance output consistency.1; sampled-event execution.05; sampled-state rollout expression mean.5. Random audio-prior trajectories have no pointwise full-face target.

Smoke: posterior arm16 updates/batch12, internal-dev16. Pass finite loss/gradient, new response-head gradients, frozen B0 and strict checkpoint reconstruction. Pilot: direct vs posterior, seeds47/91,768updates each, batch12, lr3e-4; identical TRAIN query/reference stream across arms/seeds. Fixed-final checkpoint, no best checkpoint selection. Other than posterior reconstruction/KL and state sampling, objectives remain matched. Different seeds train independently. This is a bounded feasibility pilot, not a final trained ablation or proof of three innovations.

Queued evaluations: full1367 development; dynamics128 with draws47/91/137; geometry128; controls32; style64. Native fixed angry/happy render inputs saved by the established evaluator. Existing Phase53 and v5 matched control remain descriptive baselines. Need actual lip/geometry improvements, independent dynamic assessment, and donor-style/content-emotion interference measurements before promotion. No automatic promotion or novelty guarantee.

HDTF233 sources/145people/6739 windows remain preserved for subsequent generalization stage. This first redesign pilot uses MEAD alone to isolate architecture effects. It does not establish multi-dataset training benefits.

## Frozen path evidence

The old joint residual RMS is ~.137, low/high adapter RMS ~.000136/.000239. Removing both adapters changes output RMS only~.000272; removing joint changes~.132. Envelope decode-only change is~.00031-.00041. Fixed32-clip interventions do not establish the sole optimization failure cause. JSON records both geometry and proposal sensitivity.
