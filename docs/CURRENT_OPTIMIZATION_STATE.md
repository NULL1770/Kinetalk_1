# Current optimization state (2026-10-11 v4 reviewed)

## Latest completed recovery — 2026-10-12 dual-timescale pilot

All19stages of `/root/kinetalk_v5_20261012_dual` completed; no remaining training. Three matched256-update arms and full1367/dynamics128×3/control32/geometry128/style64 reports locally SHA-verified. New direct/guided modules give negligible extra mouth gain overcontrol; guided cosine direction error worse in all3samplingseeds. No promotion; Phase53 and all parents retained. Angry82/happy103native frames with audio rendered/decoded at25fps, zero rig coefficient error; fixed-time sheets inspected.30related tests pass. Review `v5_optimization_20261011/dual_timescale_review_20261012.md`, seven tables, `dual_evidence/collection_receipt.json`, videos `artifacts/v5_dual_timescale_review_20261012`. User requested GitHub archive on existing `codex/experiment-snapshot-20260918` branch; archive underway. Earlier active-worker claims below are historical.

## Latest recovery 2026-10-12 18:53 CST — overrides older active process claims

19:13CST update: corrected detached dual-timescale queue17970 is running at `/root/kinetalk_v5_20261012_dual`. Guided16 full-objective smoke PASS (192different TRAINclips, new-head gradients and B0gate), control256updates COMPLETE (222seconds updates/390seconds stage), direct18258 caching, guided queued. Then all three arms full1367/dynamics128×3/control32/geometry128/style64. No modelpromotion; Phase53 retained. Do not restart existing queue. Latest receipts and four-factor explanation: `v5_optimization_20261011/current_four_factor_status_20261012.md`.

Server verified idle (4090,1MiB,0% utilization),9.5GB root free. State-decoder/bottleneck completed; no formal dual-timescale pilot dispatched yet. Old16-step smoke is numeric-only because it selected fit_sids as clip IDs, repeated a batch and omitted complete objective. Receipt preserved. Corrected full-objective runner uses true TRAIN/internal fold, independent references, frozen B0, zero-start adapters and same-budget control/direct/guided arms;12targeted tests passed. No performance or novelty claim. Current gates: `v5_optimization_20261011/dual_timescale_protocol_20261012.md`.

## Latest 2026-10-11 23:45 CST checkpoint

All old data-route workers5630/6827/7017 and new posture worker8413 COMPLETE; do not restart. Full1367 MEAD/all-factor/content-style route tables and matched videos saved. HDTF gradient route is not promoted: lip improves but full angry.5833 and upperface2.5408 are poor. Posture direct/privileged training/eval/dynamics/control exact receipts match256updates/3072MEAD; new pooled head does not improve stateMSE or dynamics. Privileged sample/clip_all full lip3.076654mm/maxvertex5.899234mm/EVE2.427472mm/closure.53208/mainprobe.764505/angry.869565; control3.078506/5.905527/2.426106/.52991/.76133/.85965. Small teacher vs direct probe difference is not sufficient contribution proof, and Phase53 retained. New target-style64matchedpairs and two native six-method videos complete. Review: `docs/v5_optimization_20261011/posture_style_review.md`.

Active worker10122 at `/root/kinetalk_v5_20261011` uses immutable `code_acoustic_geometry`:16step smoke passed and `geometry_coarse_train` started. Three256update arms from SAME MEADc9189be8 checkpoint: same-capacity coarseh0 receiver, HuBERT768+prosody4 geometry receiver, latter+.2sampled geometry clipmeans. Zero-start input/output/state/event invariance and finite new-head gradient/B0 gates pass. Followed by full1367,128×3sampling dynamics,32factor controls,128geometry per arm. Query motion only used for training/score, never inference; no text/test/HDTF in this isolating pilot. Source/update/reference/fit/order/commoninitial/receiverinitial hashes checked, stop-on-failure45minstage timeout. No duplicated run or promotion. SSH bridge83717 still alive.

## Active v5 continuation

Detailed recovery: `docs/v5_optimization_20261011/task_plan.md` and `progress.md`. Phase53 remains the retained model. Three contributions remain hypotheses.

Geometry repair and rollout-mean matched 256-update pilots finished on the fixed 134-clip development diagnostic subset. Mean pilot LVE3.129914mm/F1.720997/angry.608696; geometry repair3.151858/.703241/.545455; continued control3.145602/.724466/.666667. Geometry supervision mask fix is engineering, not a demonstrated contribution. Small improvements do not establish overall success. Angry still has substantial mouth-corner width error in fixed native videos. Local videos/table and verified mean checkpoint under `artifacts/v5_review_20261011` and `docs/v5_optimization_20261011`.

Mixed-data queue PID5015 at `/root/kinetalk_v5_20261011` uses source snapshot `code_mixed_pilot`; do not relaunch. 32-step HDTF smoke passed, actually using128windows/128sources/84people, masked unknown emotion/intensity/neutral targets and unchanged B0. Then256-update matched MEAD12 versus MEAD8+HDTF4 arms from the identical mean-pilot checkpoint, fixed134clip scoring and128clip/three-sampling-seed dynamics. MEAD arm finished; mixed arm running at last verification. MEAD subset sample LVE3.001273/F1.750105/angry.827586, but EVE2.447128 remains worse than Phase53 subset2.297482. These are preliminary subset results, not the full1367 table.

Follow-up PID5630 waits for both matched pilots, then runs full1367 development scoring and32clip single-reference factor sensitivity for each, from snapshot `code_pilot_followup`. It uses separate same-person A/B outputs, not two inputs, and fixed-schedule donor controls. It does not train or promote a model. No sealed/test targets. No new long training queued yet; decide only after the matched results. HDTF233usable sources/145people/6739windows are bound; clock/duration verification does not establish perceptual synchronization.

Later verification: both256-update data arms and128clip/3seed dynamic audits completed. Mixed arm actually used1024HDTFwindows, all233eligible sources/145people. Fixed134 sample: mixed LVE2.917955/F1.737919/angry.769231/EVE2.587907 versus MEAD3.001273/.750105/.827586/2.447128. Dataset mixing helps lips but harms emotion and upper-face relative to matched MEAD. Never compare these subset values with the earlier full1367 table.

Annotation-aware HDTF route candidate PID6827 now runs from separate code_content_style_route snapshot: same checkpoint/seed256steps/data composition as all-factor mixed, but HDTF gradients stop in teacher/student/event/emotion map/decoder-affect; shared decoder/style/content remain trainable. First-step assertion verifies protected MEAD gradients remain exactly unchanged. This is a parameter-gradient route, not a fixed emotional category/channel mask, and shared expression outputs can still change. Follow-up PID7017 waits for route and previous full/control scoring, then evaluates route full1367 and32factor controls. No multi-epoch long run or promotion; results pending. Server processes have PPID1 and survive SSH closure.

Latest matched angry/happy videos are `artifacts/v5_matched_review_20261011/{angry,happy}_pilot_render/comparison.mp4` (GT/B0/Phase53/MEAD-only/MEAD+HDTF), not the earlier mean-pilot videos. Audio/full native frames/rig/complete-decode checks pass; fixed-time sheets were viewed. CSV `docs/v5_optimization_20261011/matched_data_table.csv`. Angry corners remain too narrow even with better subset probe F1; style transfer and better emotional dynamics have not been established.

## Latest v4 outcome

Read `artifacts/v4_review_20261011/v4_readiness_20261011.md`. MEAD TRAIN-only v4 seed47 completed 8 epochs/10,904 updates; 1,367 development clips and 128 balanced dynamics clips with sampling seeds47/91/137 were evaluated. Final checkpoint SHA is `657ef6e61c58568675710baadd3c6e95eec5859133a999c07aabcc5a4b2e31d2`, verified locally against both reports. Do not relaunch completed training/postprocess.

clip_all Phase53/v4_sample: MBE .763406/.845421, LBE .369546/.378105, mean lip LVE3.262565/3.357814mm, EVE2.237557/2.741130, jaw closure .401017/.377687, range .124295/.149294, centered correlation .477797/.507262, main motion-probe F1 .676805/.691575. Angry F1 .648276/.333333. v4 events execute, but the shared-parser schedule execution F1 is not GT emotion-dynamics accuracy. Overall geometry and expression direction regress; Phase53 remains retained, no model promotion.

Full angry/happy videos with matching audio are under `artifacts/v4_review_20261011/{angry,happy}_render/comparison.mp4`, four columns GT/B0/Phase53/v4_sample. Decode, frame counts, zero rig coefficient error, unchanged blend and SHA checks pass; receipt `local_review_receipt.json`. This run excludes HDTF and lacks formal style/factorial validation and matched trained ablations. Three contributions remain unproven. No sealed/test read. Next priority: bounded state/projection/event diagnostic controls before further training, then valid HDTF/style/factorial and matched multi-seed ablations.

## Historical Phase65 state

## Manuscript revision — 2026-10-10

Resume manuscript work from `paper_draft/README_当前稿件.md`, `paper_draft/作者记录_20261010/修订核对与证据.md`, and `docs/paper_rewrite_20261010/修订依据与章节提纲.md`. The current Chinese manuscript is `paper_draft/论文_当前中文版.md`, synchronized to `D:/科研/小论文/论文_当前中文版.md` after validation. Pre-revision archive: pushed commit `0770f17`.

Abstract, introduction, related work, method, experiments, and conclusion have been rewritten against the actual retained Phase53 style-only code and the EmoTalk/MEDTalk papers. Current model is a conditional residual response model, not flow matching. No model code, training, inference, or evaluation was changed in this manuscript revision. Core retraining ablations are still outstanding; see `paper_draft/07_正式实验补全协议.md`. Existing Phase51 reference replacement and Phase53 adaptation-scope contrasts are separate from frozen-model local-condition interventions.

Metric convention: the manuscript LVE/EVE use per-frame regional maximum vertex distance (clip_all: 6.132448/2.237557 mm). Earlier notes below use regional mean distance (lip 3.262565 mm); do not compare the two definitions as a training improvement. Tables are reproducibly generated from existing full-development reports by `docs/paper_rewrite_20261010/build_manuscript_tables.py`, with source hashes in `paper_draft/作者记录_20261010/tables/provenance.json`.

Figure caution: existing 02a/02b compare the Phase53 joint variant, not the retained style-only main model. Current text identifies the variant explicitly. A final style-only main-model figure requires new exports, not relabeling the old image.

## Retained training state

Phase53 style-only remains the retained baseline: MBE .763407, LBE .369546, lip 3.262565 mm, main probe F1 .676805, jaw range .124295 (GT .175279), jaw correlation .477797, closure F1 .401021. Phase64 bounded residual was fully collected but rejected (MBE .762427, LBE .376756, jaw range .122752, correlation .464588, main probe F1 .671319); see docs/PHASE64_RESULTS.md.

Phase65 was an affect-only zero-start additive adapter. It trained only `decoder.affect_adapter` from the frozen Phase53 style-only checkpoint. Inputs were global/local affect latents, reference response code and local RMS envelope; B0/content features were excluded. The existing position + adjacent-displacement reconstruction objective was unchanged. Commit 0ad7e9d is pushed and source-bound.

The first remote launch stopped at the existing 250 MiB disk floor before smoke. Verified redundant remote Phase53/64 artifacts were removed, preserving only the Phase53 style-only final parent. The same Phase65 run was relaunched exactly once. Smoke passed (120 steps, loss ratio .374326, frozen/B0 exact). Formal seed47 completed 8 epochs and 5,456 updates on remote `/root/kinetalk_phase65_affect_adapter_20261010`; development evaluation completed on all 1,367 clips with `test_loaded=false`.

Phase65 failed the development acceptance gate and is rejected. Prior-mean results were MBE .835922, LBE .381512, LVE 3.301901 mm, EVE .807537 mm, jaw range .124278, jaw correlation .476915, and four probe F1 values .566034/.537924/.624539/.571697. Phase53 style-only remains the retained model. No Phase65 metric, checkpoint, or video is promoted as a replacement, and no sealed/test data were read.
