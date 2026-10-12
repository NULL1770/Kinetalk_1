# v6 direct arm: completed development review

## Material Passport

2026-10-12. Source: exact-SHA collected direct_seed47 final768-update checkpoint and all five completed evaluation stages. Scope: development only. Following review, the remaining queue was stopped and the user requested rollback to v5 matched control; posterior resume384 is preserved and seed91 arms were not started. No test data or three-contribution claim.

| sample / clip_all,1367clips | Phase53 | v5 matched control | v6 direct |
|---|---:|---:|---:|
| mean lip distance mm↓ |3.262565|3.048290|3.305541|
| maximum vertex LVE mm↓ |6.132446|5.853746|6.221172|
| upper-face EVE mm↓ |2.237557|2.435403|2.615629|
| closure proxy F1↑ |.401017|.553729|.382824|
| generated-motion emotion probe F1↑ |.676805|.758772|.403719|
| angry probe F1↑ |.648276|.862974|.503937|
| jaw centered correlation↑ |.477797|.503692|.410575|

Reject promotion of the direct arm. It replaces the old decoder and has768updates of fresh decoder training; v5control continues an already trained decoder for256updates. These cross-family numbers describe current quality, not a matched estimate of a single architectural intervention. A short-budget failure does not prove longer training cannot work. Do not lengthen the budget or select a favorable checkpoint on the basis of this development review.

Dynamics128: expression-direction cosine error .196053/.198366/.198586 for fixed event draws47/91/137, versus Phase53.172515 and v5control~.174-.176. Direction does not improve. These are custom ARKit diagnostics using a weak parser, not an official FDD or framewise psychological emotion metric.

Style64: donor direction partially improves aggregate mean descriptors, but reference swap still changes closure decisions10.47% of frames. Clip-all generated probe-label preservation is60.94%; direct baseline emotion classification is already poor. Lower interference alone cannot establish a better disentangled model when expression is weakened. Three development speakers limit generalization.

## Fixed native videos

Rendered GT/B0/Phase53/v5control/v6direct for predefined mead_M025_angry_L3_005 and happy_L3_005. Original coefficients and audio retained, invalid edges cropped identically, native start.04s,25fps. Angry82frames/happy103frames; full decode and audio checks pass, source blend unchanged and rig coefficient maxerror0. No gain/lag/smoothing chosen for display.

Five uniformly spaced times visually inspected. Angry retains substantial mouth widening mismatch relative to GT and round/open-mouth substitution late in the utterance. Happy smile is weaker than v5control at several reviewed times, particularly the opening and final pose. These two fixed examples do not establish population-wide visual performance.

Artifacts: artifacts/v6_direct_review_20261012/{angry_render,happy_render}/comparison.mp4 and uniform_time_inspection.png; native_review_receipt.json binds input/checkpoint/audio/video SHA. Full tables: full_development.csv, emotion_probes.csv, dynamics.csv, geometry.csv, style.csv, controls.csv with table_provenance.json.

Disposition: reject v6 direct promotion and keep the remaining queue stopped. Restore v5 matched control as the working baseline without discarding the failed prototype or checkpoints. The unmatched multi-change pilot does not establish which individual factor caused the regression. An input-isolated content branch permits emotion/style mouth effects; the structural mouth freedom test passed, but the trained effect is currently inadequate.
