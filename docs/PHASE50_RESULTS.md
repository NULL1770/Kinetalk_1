# Phase50 approved-pair native coordinate diagnostic

2026-10-09. Completed frozen diagnostic, not a new trained model. Source3568e58 was pushed before dispatch. Remote `/root/kinetalk_phase50_paired_coordinates_20261009`, PID48069 finished. Local originals are in `final_experiment/evaluation/diagnostics/phase50_paired_coordinates_20261009`;11 members SHA-verified.13 local and remote tests,24clip GPU smoke and2583clip full diagnostic passed. Validation query inference0; sealed0; B0 state exact; no training.

## Evidence

Approved emotional pairs:2024 original TRAIN-fit,281 internal-held speaker,278 internal-held sentence. Uses the existing safe loader:mouth native_teacher_mask AND event_local_mask AND source observation/channel; upper original teacher/source/channel support. Aligned neutral is imperfect, not causal ground truth.

Physical coefficient squared energies, equal clip then supported-channel weights:

|TRAIN region/kind|GT-B0 total|GT-neutral difference|neutral-B0 error|cross term|
|---|---:|---:|---:|---:|
|Mouth position|.01406337|.01498081|.00112553|-.00204297|
|Mouth centered|.00637067|.00683693|.00095177|-.00141803|
|Mouth adjacent displacement|.00155746|.00146767|.00049129|-.00040149|
|Brows position|.10442695|.06869303|.03102655|.00470737|
|Eyes position|.05154144|.01930612|.03476568|-.00253035|

The terms are nonorthogonal, so they are not additive causal percentages. The difference component can exceed the total because of negative cross terms. Mouth position discrepancy is associated primarily with the expression-difference term; mouth adjacent motion still contains substantial neutral-B0/alignment differences. Upper-face static discrepancy is considerable. These observations support separate treatment of persistent posture and response, but do not prove audio predictability or pure identity.

Original report SHA:f568ecef01019f9c908c1ecd043a093f0c0080a3a87c55f0886d491a81068312. Output15767379bytes, within64MiB declared cap;root390770688bytes free after completion. No prior storage-only failures repeated.

## Decision

Do not change neutral B0 or train the emotion student from full residual reconstruction. Do not use unqualified GT-B0 means as identity labels. TRAIN-fit has620 neutral clips across20 actors (18–34 per actor), beyond the fixed two enrollment clips used repeatedly before. Next test random independent neutral supports and a compact posture/response reference encoder against a temporal-reference control, with frozen prior and unchanged position/adjacent displacement objective. Reference quality and inferred style remain imperfect; generalization must be measured on original internal holds before any external evaluation.

Current best development metrics remain Phase47-latent:F1.719825/MBE.825160/LBE.383860/Lip3.276868mm. This diagnostic has not improved them.
