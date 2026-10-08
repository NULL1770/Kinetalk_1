# Current recovery entry: Phase46 training dispatched (2026-10-08)

Read PHASE46_IMPLEMENTATION_AND_TRAINING.md and PHASE46_PLAN.md first. Implementation356cea9, pre-change6e8d5d5 successfully pushed. New push/ref checks failed due network; do not claim latest upload. 38 local+remote tests, both longest432-frame GPU gradient/frozen/NaN/exactresume checks,120-update eight-emotion fits and16clip eval smokes passed.

Remote /root/kinetalk_phase46_receiver_20261008: mixed28213/deploy28214, watcher28215. Two decoder-only8epoch5456update arms from Phase45-u; all conditions/B0/style frozen, two deployment supports, original objective. Local finite collector53184 waits for original1367 evals then16videos/tables. Do not duplicate any launch/collector. No final Phase46 scores yet; actual update rate pending. Parent closure below remains valid. No sealed data/default changes.

# Current recovery entry: Phase45 COMPLETE (2026-10-08)

Read [PHASE45_RESULTS.md](PHASE45_RESULTS.md), [all videos](PHASE45_VIDEO_GALLERY.md), [tables](PHASE45_EXPERIMENT_TABLES.md). Older state retained in archive/CURRENT_OPTIMIZATION_STATE_before_phase45_closure_20261008.md. Do not restart any Phase41-45 pipeline or collector.

## Verified outcome

Phase45-u is the expression/dynamics candidate: primary generated-motion F1 .709447609 vs Phase43-A .695182238, all four probes improve. MBE .827171/LBE .388674/Lip3.320219mm; jaw range .153116 vs parent .142617/GT .175279; jawcorr .490531 vs .473212. Geometry essentially unchanged, not joint SOTA, default unchanged. g and gu lower F1/amplitude; do not adopt. Weak classes neutral .468468/fear .477612; happy .934097. Four probes are clip statistics, not frame-level emotion labels or audio-head accuracy.

Frozen teacher audit: TRAIN10903 fit, held speaker743/sentence890; existing hidden-to-effective-u ridge improves held error1-2%, globalg12-16%. No content leakage proof. One fit-only ridge=.001, per-clip equal; u replaces existing128x16 mean weights only. B0, prior backbone, all variance rows, posterior/style/decoder/semantic heads unchanged. No new modules/losses. AdditionalSGD0 is not training-free: one analytic supervised fit10903; parent24epochs/16368updates. No external validation fit or sealed data.

28 local+remote tests passed; realGPU longest/short TRAIN batch mean mappings, covariance exact, HuBERTNaN isolation, B0 digest/save exact passed; three16clip smoke/full1367 complete. Closure rehashed182 audit+5pipeline+3x192arm members,24videos,12table sources. Mean slices equal saved ridge matrices; other rows/tensors frozen. B0 full predictions/metrics/probes exact; oracle existing float tolerance/probe exact. No original metrics changed.

Remote /root/kinetalk_phase45_target_audit_20261008/head_candidates; local final_experiment/evaluation/diagnostics/phase45_target_audit_20261008/head_candidates. Remote17923 and local136844 complete. 24videos complete, manually viewed u all8 at25/50/75%; angry mouth tension weak, fear/surprise timing discrepancies visible. Three seeds/independent content/target-identity transfer/official benchmark remain unverified. No SOTA promise.

## Next

Archive current results before next code round. Investigate whether a frozen audio prior with receiver trained on deployment mean conditions can lower geometry error without losing expression. Preserve original position/adjacent-displacement objective and use a matched control; do not add losses or alter B0/student inputs. No new Phase46 run yet.

## Persistent boundaries

Every code-change round requires preceding Git push. c16e668 code and earlier0a97b9d diagnosis pushed via existing SSH CONNECT tunnel; direct GitHub HTTPS resets. Credentials private helper only, never print/commit. No child agents. third_party/voca_reference/ unrelated untracked: leave untouched.

Neutral B0 frozen; audio prior ONLY emotion2vec768+prosody4, full mouth output; no queryGT deployment or reconstruction gradient into student; no sealed tuning/oracle ranking/default promotion. Each completed candidate has eight emotions, unchanged raw/clip/fourprobe metrics and honest external-adapted-baseline/trained-ablation tables. MEDTalk/DESTalker lack verified trained rows. Same emotion/content different target-reference correctness is not proven by output changing.
