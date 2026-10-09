# Phase51 results and joint decision

Material Passport: MODE=validate; STATUS=ANALYZED. All1367 development clips, unchanged native clock, raw/clip policies, four frozen probes. No sealed data. No candidate promoted.

|Candidate|MBE↓|LBE↓|Lip mean mm↓|Main clip F1↑|Jaw range|Jaw correlation|
|---|---:|---:|---:|---:|---:|---:|
|Phase47 latent|0.825160|0.383860|3.276868|0.719825|0.151821|0.489966|
|Phase51 temporal|0.859197|0.416775|3.451083|0.734619|0.145985|0.488077|
|Phase51 statistics|0.809449|0.393823|3.386572|0.712325|0.126797|0.468724|

GT jaw range .175279. Main F1 is a whole-clip motion-statistics probe, not framewise emotional truth; GT .660344. Temporal improves clip F1 but worsens geometry. Statistics improves MBE, and style target-direction/stability, but worsens lip error/range/timing versus Phase47. Neither is joint SOTA.

## Style evidence

Full1367 fixed-source tests,2026 directed sentence/emotion/intensity matches. Native separate-performance statistics, not time-aligned counterfactual GT.

|Candidate|Own A/B vs cross output MAE|Own A/B vs cross mouth MAE|Own A/B closure disagreement|Target mouth mean improved|Target mouth range improved|
|---|---:|---:|---:|---:|---:|
|temporal|42.65%|41.88%|9.86%|57.35%|58.09%|
|statistics|29.21%|29.73%|9.87%|83.46%|80.80%|

Phase48 ratio was76.4% overall and closure disagreement22.3%; both new arms are more reference-stable. Statistics mouth mean83.46%, range80.80%, velocity83.07% move toward target statistics; brows remain weaker. Shared rig tests motion style only, not shape identity. Three development people limit generalization.

## Verification and artifacts

13local/remote audit tests,24clip smoke for each arm and full replay gates passed. B0 predictions/probes/metrics and GT/audio probes exact. Candidate replay max temporal/statistics preserved in original report, at unchanged rtol1e-5/atol2e-5. No Phase47 correction. Original36evaluation and45style files SHAverified. All24 videos full-decode/native25fps/audio/rig verified; fixed-middle frames of all24 visually checked. Angry/disgust/fear still show expression/jaw mismatches, not claimed solved.

See PHASE51_VIDEO_GALLERY.md and PHASE51_EXPERIMENT_TABLES.md. Tables include21methods/17trained candidates; the adapted baselines have differing data/budgets and are not official SOTA benchmark scores. Inherited oracle is uncalibrated after style change and excluded from rankings.

## Next controlled intervention

The joint decoder update is a plausible contributor to amplitude drift; Phase51 does not establish it causally. Phase52 will retain the same statistical encoder initialization, independent varied neutral supports and existing reconstruction objective, but train only style encoder/static bias/style modulation columns. Freeze receiver input/temporal trunk/output and g/u conditioning weights to the Phase45-u parent. This isolates trainable scope against Phase51 statistics. It protects weights, NOT a guarantee of unchanged output phoneme timing or amplitude. Those remain measured gates. Do not add arbitrary amplitude gain or claim unseen-emotion personal expression recovery from neutral references.
