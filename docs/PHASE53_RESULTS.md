# Phase53 completed results — no model promotion

2026-10-09. Both seed47 fixed8epoch/5456update runs, full1367 development evaluations and2026 matched directed style pairs per model are complete. Native25fps, original rig, raw+clip, four frozen whole-clip motion probes. No sealed test. The main F1 is NOT framewise emotion classification.

## Motion and semantics (clip[0,1])

|Model|MBE↓|LBE↓|Lip mean mm↓|EVE-max mm↓|main F1↑|jaw range|jaw corr↑|closure F1↑|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|Phase47 latent|.825160|.383860|3.276868|2.518575|.719825|.151821|.489966|.401578|
|Phase51 temporal|.859197|.416775|3.451083|2.581025|.734619|.145985|.488077|.259786|
|Phase51 statistics|.809449|.393823|3.386572|2.357694|.712325|.126797|.468724|.400022|
|Phase52|.762598|.355341|3.081770|2.351000|.620855|.121521|.495885|.477014|
|Phase53 style-only|.763407|.369546|3.262565|2.237557|.676805|.124295|.477797|.401021|
|Phase53 joint|.823704|.406063|3.471205|2.412348|.690150|.128094|.459235|.375772|
|EmoTalk-core adapted|.745003|.331531|2.973232|2.416746|.591884|.154294|.565796|see tables|

GT jaw range .175279; Phase53 reaches about71–73%, with temporal mismatches as well. Main GT F1 .660344 and audio classification .880874 are unchanged; neither is generated-motion F1. Auxiliary probe scores must not substitute for the primary score.

Style-only four clip F1=.676805/.650936/.734088/.681300; raw=.633962/.660825/.691677/.657573. Joint four clip F1=.690150/.700049/.744128/.697743; raw=.685191/.691514/.726047/.686384. Raw MBE/LBE/lip: style-only .764276/.369855/3.275164; joint .825109/.406987/3.497849. Fear main F1=.341880/.310345, happy=.868502/.805112, sad=.857143/.830918 (style-only/joint). Partial semantic recovery is real under these fixed probes, but lips/closures worsen versus Phase52.

## Reference style

Same audio/B0/global/local conditions, independent neutral references. Cross-person targets are native statistics of independent performances matched by sentence/emotion/intensity, not framewise counterfactual GT.

|Metric|Phase52|Phase53 style-only|Phase53 joint|
|---|---:|---:|---:|
|Own A/B MAE /cross MAE, all|37.15%|25.04%|31.00%|
|Own A/B MAE /cross MAE, mouth|31.83%|20.95%|31.89%|
|Own A/B closure disagreement|15.89%|9.60%|7.59%|
|Cross closure disagreement|41.55%|32.45%|22.35%|
|Mouth mean toward target|87.36%|79.32%|81.89%|
|Mouth range toward target|78.97%|75.32%|81.10%|
|Mouth displacement toward target|81.93%|74.93%|83.56%|
|Brow mean toward target|59.87%|55.23%|62.09%|

Own-reference stability improves, but transfer is not uniformly better. Only3 development people; ratios must be interpreted alongside absolute errors in original reports. Closure is a rig diagnostic, not measured phoneme error. Neutral references do not prove personalized expression style for each emotion.

## Verification and visuals

62 local and62 remote checks, both120step GPU smokes, independent exact113 frozen condition tensors per candidate plus36 protected receiver tensors/slices for style-only. Neutral B0 exact. Student uses772D emotion/prosody only; no reconstruction gradient reaches it. TRAIN fit10903/internal speaker743/sentence890;620 neutral fit supports across20 people; supports exclude query clip and sentence.

Checkpoints: style-only SHA80a7ca69b0f868bad6361381e1ca8e4e168ca227a0548221f8efa71b42169c41; joint SHA7d98ddaca1885dc25c926d570a58448f2aae1c57ebb9603e549ca495f63b91e9.
Evaluation reports: b3e14b28378cfe7503b240b98914b737e4cb965b456309696de9c8d582e59b31 /2f448d45fee38ebae5b5b148a5e6d081f4012604597fe3b821e96380f68b562e.
Style reports: dcf75b1ec927e238d06f739c71ee6ba43f28fa80f9feb19b2785df06305b22b0 /c16a13b330aec4c1f65d97fb024ddd3d8a8b1e187ebfb97dbc27e58884f56394.

Original download manifest112members/345162906bytes SHA-verified. Preflight208members/41556244bytes and final training archive15members/16037954bytes independently verified. Worker10345 and collector52028 complete; do not relaunch.

24 videos complete:8main five-column comparisons +8 style comparisons for each candidate. All-frame decode/native clock/audio/rig checks passed. Actual visual inspection: all24 middle frames and fixed10/30/50/70/90% native frames for happy/fear/angry/sad; not complete real-time playback. No evident render corruption; happy large openings/smiles weak, fear temporally too open/too closed, angry shape/tension mismatch, sad corners/upper-face weak. Swapped references visibly change tendencies and own A/B are fairly similar. Separate visual_review_receipt.json records inspection without rewriting original collector receipt.

See PHASE53_VIDEO_GALLERY.md and PHASE53_EXPERIMENT_TABLES.md. Full24method/20candidate raw+clip tables are in final_experiment/paper_tables/phase53_development_20261009. Adapted baselines have unequal budgets; verified MEDTalk/DESTalker trained rows absent. No SOTA claim, no combining best numbers from different models.

## Decision and next diagnostic

Do not promote either candidate. Factorization and neutral-only offset routing were changed together, so separate component attribution is unavailable. Joint receiver retraining worsens geometry/timing; style-only recovers some semantics but loses lip/closure quality. These are useful diagnostic/ablation results, not final success.

Next: fixed TRAIN/internal-held linear predictability audit on approved paired native expression differences GT-alignedneutral, alignedneutral-B0 error, and totalGT-B0. Use per-channel observation/event/gap masks, equal clip fitting, TRAIN-only scaling and fixed ridge. Compare frozen prior local u/hidden,772D audio and a diagnostic B0 control; do not feed content to the student. Existing Phase45 tested latent predictability and Phase50 tested energy decomposition; neither tested predictability of these physical paired dynamics. Target contamination is a hypothesis, not established causal leakage. Choose later model changes from this evidence; no new loss or receiver sweep precommitted.
