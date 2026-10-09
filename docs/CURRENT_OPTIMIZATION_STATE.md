# Current: Phase52 formal training ACTIVE (2026-10-09)

Read PHASE51_RESULTS.md and PHASE52_STYLE_SCOPE_PLAN.md first. Git pre-change13351ce and implementationf7adcfb pushed.56local+56remote tests and120step GPU smoke pass (loss first10 .691840 -> last10 .449539, ratio .649773). Frozen B0/prior/receiver tensors exact; HuBERTNaN isolation; style gradients and resume checked.

Remote /root/kinetalk_phase52_style_scope_20261009, workerPID4435. Formal seed47 statistics/style_only eight epochs,5456updates. Latest epoch1step301 at~.142s/update. Do not relaunch. The29second epoch earlier quoted is SMOKE, not formal; approximate training15-20min plus full evaluation/style audit/16videos total25-35min from this snapshot. Real improvement pending.

Pipeline queues unchanged full1367 evaluation, frozen GT/B0/audio replay,24clip style smoke and1367/2026matched style audit. Local finite2h collectorPID34668 waits and collects SHA originals, builds22method/18candidate tables and renders16fixed eight-emotion main/style videos. If worker/collector fails, preserve failure and diagnose; do not restart completed roots. Visual review of Phase52 will remain pending until model returns to inspect. No automatic promotion.

Preflight197original members/19323576bytes incl186bound sources and both smoke checkpoints locally SHAverified under diagnostics/phase52_style_scope_20261009/preflight. root415703040free at epoch1; current/probe/data/rig untouched. Collector state/log available locally and phase52_status.py read-only. User permits leaving after real training starts.

Phase51 already closed:1367x2eval+2026x2style,24videos fulldecode/native/rig and visualchecks,21method/17candidate tables. Statistics improves style but shrinks jaw; temporal F1 .734619/MBE .859197,statistics .712325/.809449. Best joint dev still not SOTA. Phase52 restricts trainable scope without adding architecture/loss/gain; parameter freezing does NOT prove output timing independent.

Earlier records below historical and superseded.

# Current: Phase51 CLOSED; Phase52 scope intervention next (2026-10-09)

Read PHASE51_RESULTS.md and PHASE52_STYLE_SCOPE_PLAN.md. Both complete1367 evals +2026matched style audits;24videos and21method/17candidate tables locally verified. No relaunch. Temporal F1 .734619 / MBE .859197; statistics .712325 / .809449 but jaw range .126797 and corr .468724. Style stability improves, geometry/dynamics mixed; no default promoted. Hypothesis for Phase52: freeze shared receiver and learn only reference-specific parameters, same statistical initialization/supports/loss. Parameter freezing does not guarantee correct output timing.

All original evaluation36files and style45files SHAverified under diagnostics/phase51_evaluation_20261009. CollectorPID138996 complete24videos; all fixed-middle frames reviewed. Source d9a06dc push succeeded after one TLS timeout. root437821440/data93532160 free. No model edits since Phase51 yet.

Historical notes below superseded where conflicting.

# Current recovery: Phase51 complete; full evaluation ACTIVE (2026-10-09)

Both eight-epoch runs completed: temporal SHA b8bde2389ac9dd47bc8b3aa30a34009454dc1ef62f09a9f0ea17a6849af6d03a; statistics SHA a4d735a72f599cb1b2115d8d9ea1340abc02fa69be14f58500c620a8f30bf6ec. All original preflight/formal artifacts locally SHAverified. Do not relaunch training.

Full unchanged evaluator active at /root/kinetalk_phase51_evaluation_20261009, worker PID1752; temporal then statistics, full1367, native/raw+clip/fourprobes and frozen GT/B0 checks. Latest temporal672/1367. No full metric improvement established. Candidate-compatible style audit and all-eight-emotion videos pending; no Phase47 correction may be reused on Phase51 coordinates. Archive258fa40 pushed before audit extension. Next optimization depends on actual geometry, dynamics and same/different-person reference results.

Storage:13 obsolete smoke/restart last.pt originals backed up under final_experiment/artifact_archive/phase51_obsolete_last_20261009 and SHAverified before deletion;419898829bytes reclaimed. Root706MB free before evaluation, data93MB. Never repeat archive deletion. Current finals/data/probes/rig untouched. Private operational helpers remain ignored. Third-party/voca_reference untouched.

Earlier notes below are historical and superseded where conflicting.

# Current recovery: Phase51 formal training ACTIVE (2026-10-09)

Most recent update 2026-10-09 01:46:48 China time: temporal COMPLETE8epochs/5456updates, final SHA b8bde2389ac9dd47bc8b3aa30a34009454dc1ef62f09a9f0ea17a6849af6d03a. All8original formal members/25,969,729bytes locally SHAverified in diagnostics/phase51_reference_response_v2_20261009/formal_temporal. Statistics process has started automatically; process alive=True, state={'status': 'training', 'epoch': 1, 'step': 201, 'batch': 201, 'seconds_per_update': 0.14747205421106138, 'loss': 0.6499613523483276, 'grad_norm': 1.2907689809799194, 'position': 0.619025707244873, 'velocity': 0.0618712417781353}. Allow approximately20-30minutes for remaining training and internal checks, not full evaluation/render. Temporal final TRAINloss .322069,internal-speaker prior_position .471731,sentence .430171: no gain demonstrated on these normalized diagnostics. Full1367 metrics/style swaps/eight-emotion renders NOT queued, require subsequent dispatch/storage. Definition and records were pushed at f2c5ee4,remote independently verified; direct TLS works after relay timeouts. No model code changed or experiment restarted during this recovery.

Read this header first. Implementation d8d544e is pushed. Active remote root /root/kinetalk_phase51_reference_response_v2_20261009, worker PID49242. Read-only process check 2026-10-09 01:43:06 China time: alive; temporal epoch8,5375updates,7complete epochs; statistics follows automatically. Do not relaunch. Both120update GPU smokes passed;52local and52remote checks passed; frozen neutral/expression and HuBERT-NaN isolation exact. Preflight205files/50,179,858bytes including186source files and both smoke checkpoint pairs are locally SHAverified. Training/cache/internal-check ETA approximately20-30minutes from this snapshot, not full evaluation/render. Pipeline stops after8epochs per arm with evaluation_pending: no evaluator/renderer is queued.

Latest USER CORRECTION: user did NOT request static/posture versus dynamic style separation. Wants a literature-consistent, achievable reference-driven speaker motion style with real training/visual/quantitative evidence. Read updated docs/STYLE_DEFINITION_AND_EVALUATION.md first. Phase51 statistics is merely an optional architecture arm, temporal equally eligible; no architectural split or novelty claim is predetermined. Select based on stable same-person references, target-directed inter-person change, preserved source speech/emotion, generalization and full geometry/dynamics metrics. TRAIN-only varied support→query learning is the concrete current optimization. No guarantee of success from architecture or smoke.

Formal temporal evidence is mixed: TRAINloss .339823 to 0.322599 at epoch7, while internal-speaker prior_position .456846 to 0.465108 and sentence .423432 to 0.431558. These normalized diagnostics are NOT MBE/F1 and do not establish improvement. Keep fixed8epoch final; no favorable checkpoint selection. Inherited motion posterior is uncalibrated to changed style, oracle diagnostic only. Current best real dev remains Phase47-latent. Before full evaluation, arrange SHAverified obsolete-artifact cleanup; current free root/data 317927424/93532160bytes. Preserve active/parent/B0/data/probes/rig. Reuse original evaluator and adapt Phase48 swap audit to arbitrary candidates WITHOUT Phase47 correction. Full1367/native/raw-clip/fourprobes/eight-emotion/style renders remain pending.

Historical pre-dispatch note, superseded by the running state above; do not execute it as a new instruction. Read docs/PHASE50_RESULTS.md, docs/PHASE51_REFERENCE_RESPONSE_PLAN.md and docs/STYLE_DEFINITION_AND_EVALUATION.md FIRST. Phase50 source3568e58/plan archived473444 pushed.13 local/remote tests +24GPU smoke +2583 paired diagnostic complete;11original SHA members local. No metric improvement from this diagnostic. TRAIN has620 independent neutral query clips across20fit actors. Phase51 next: temporal encoder control versus compact posture/response encoder, diverse disjoint neutral TRAIN supports, only style+decoder training, frozen B0/772D prior;8epochs/seed47. New code tested locally, not yet dispatched at this update. Do not rerun completed Phase50.

Read docs/PHASE47_RESULTS.md, PHASE47_VIDEO_GALLERY.md, PHASE47_EXPERIMENT_TABLES.md and PAPER_FIGURE_REQUIREMENTS.md first. Do not restart completed Phase41–47 training/evaluation/collectors. Prior recovery history is preserved in docs/archive and .codex-finalizer/planning_archive.

Full1367 development, original audio prior mean / clip[0,1]:

|Model|MBE↓|LBE↓|Lip mean mm↓|Primary clip-motion F1↑|Jaw range|Jaw correlation|
|---|---:|---:|---:|---:|---:|---:|
|Phase45-u|.827171|.388674|3.320219|.709448|.153116|.490531|
|Phase47-latent|.825160|.383860|3.276868|.719825|.151821|.489966|
|Phase47-reference|.857411|.407056|3.476989|.554851|.151904|.489830|

Latent is a modest new development candidate, not automatic default or joint SOTA. Reference is rejected. Latent four clip F1s .719825/.664336/.744721/.677342; raw .659494/.603158/.676337/.628080 not uniformly improved. Raw centered dynamics unchanged; clip jaw range -~.85%, GT .175279; brow mean bias slightly worse. Neutral/fear remain weak. Adapted EmoTalk-core geometry still leads (.745003/.331531/2.973232mm); data/budgets differ. MEDTalk/DESTalker lack verified trained rows.

GT main frozen motion-probe F1 .660344; audio head .880874. Probe is whole-clip generated-motion statistics, not framewise emotional/dynamic truth. F1/t-SNE alone cannot prove realism, disentanglement, generalization or publication readiness.

All Phase45-u parameters/B0 frozen. Fixed ridge=.001, clip-equal TRAIN10903 residual-mean fitting: latent g32+style64→52D constant (5044 affine coefficients); reference adds independent neutral52 and predicted8-class probability interactions (29380). Different-capacity controls. Inherits24epochs/16368updates + Phase45 TRAIN10903 analytic u fit; Phase47 adds one supervised analytic fit per map, zero SGD updates. Parent/correction have separate SHA bindings.

31 relevant checks total (30 existing/module +1 adapter), GPU128/32/32 smoke, two16clip evaluator smokes passed. Frozen parent/B0 exact, HuBERT-NaN isolation passed. Full1367 B0 predictions/metrics/probes exact. Parent oracle probes exact, coefficients max1.4662743e-5 within rtol1e-5/atol2e-5. Raw displacement max numerical difference1.1920929e-7. Original evaluator unchanged; oracle uncorrected diagnostic, never deployment.

16videos complete, native25fps/audio/full decode/GT-B0 clock/rig checked. Fixed-middle-frame visual inspection shows no corruption, but angry/contempt/fear differences remain. No requested paper figures generated.19methods/15supervised candidates in tables, all original raw/clip geometry/coefficients/dynamics/four probes/classes/speakers/interventions/budgets retained.

Local evidence: final_experiment/evaluation/diagnostics/phase47_static_response_20261008. Tables: final_experiment/paper_tables/phase47_development_20261008. Collector local_queue_state.json=complete,16videos,PID130456 finished. Read-only helpers .codex-finalizer/phase47_status.py and phase47_eval_status.py; never dispatch/collect again.

Remote /root/kinetalk_phase47_static_response_v2_20261008 and /root/autodl-tmp/kinetalk_phase47_static_eval_20261008 complete; GPU idle at last check. Root/data free304762880/93532160bytes. Arrange storage before another experiment; preserve historical evidence. Private SSH helpers hold credentials; never print/commit them.

Pushed pre-change474307e,implementation2df2af5,smoke225119d,evaluation754b702. Result closure/push receipt recorded in progress.md and closure_verified.json when complete. Unrelated third_party/voca_reference/ untouched. No child agents.

Next: diagnose teacher target coordinates on manifest-approved aligned TRAIN/internal-held neutral/emotional pairs; distinguish independent neutral baseline, predictable expression and B0/unpredictable residual. Check reference stability/content association before changing global/local targets or claiming target-identity transfer. Do not extend failed Phase46/reference or add arbitrary losses. Phase48 audit code is implemented; model training not started.

Persistent boundaries: neutral B0 frozen; student only emotion2vec768+prosody4; mouth open; no content/queryGT deployment or motion-loss student gradient; native clock/rig/raw-clip/four probes unchanged; no sealed tuning/default promotion. Git push before each model change. Multi-seed, independent content readout, correct target-identity transfer, matched baselines/official benchmark and human evaluation remain paper gaps. Future figure requirements recorded; user explicitly defers production.

Phase48 COMPLETE: read PHASE48_RESULTS.md and PHASE48_VIDEO_GALLERY.md. Frozen full1367,2026directed matched pairs; no training/default change/sealed. Target-stat overall/mouth mean improvement80.3554%/72.3100%, but A/B instability (M025codecos.224; same-personoutputMAE76.4%ofcross,closurediscord22.3%). M025->M037/M039 browtarget and M025->M037mouthvelocity worsen. Positive partial action-style evidence, not face shape or universal identity disentanglement. All8six-panel videos verified+fixed-middle-frame visual review. Closure original SHA rehashed,source gates retained.

Remote /root/kinetalk_phase48_style_audit_v2_20261008; local final_experiment/evaluation/diagnostics/phase48_style_audit_v2_20261008.6tests/24GPUsmoke/full numerical replay passed:B0/latentzero,parent1.2278557e-5. Parent/B0/correction frozen. Original worker pipeline_state.failed is STORAGE-ONLY32765298bytes>20MiB, numerical audit/state.complete. Original failure/binding preserved; storage_acceptance_receipt permits<40MiB after allmanifest checks, no rerun. Failed firstroot and earlylocalcollector records preserved. Do not restart any worker/collector/render.

Implementationc01d984 pushed. Next diagnose independent neutral-reference pose/B0 residual/content association on TRAIN/internal-held and describe dev, before selecting architecture. Any new model training needs SHA-verified remote artifact cleanup first;rootfree263MB/dat93MB. Prior Phase47 metrics and all evidence above remain valid; no fresh metrics improvement from Phase48.

Phase49 COMPLETE:read PHASE49_RESULTS.md/PHASE49_REFERENCE_COORDINATE_PLAN.md.25neutralenrollment actors(20fit/2held/3dev),querycaches0,6tests+stylecode replay/frozen gates pass. Code same/crossRMS ratios .010/.367/.465 support TRAIN-reference overfit concern; GTmouthmean .322/.366/.281, residual .818/1.123/.392 shows GT-B0 not pure style. M025GTbrowposeA/B differs even B0-closed frames; neutraljawGT~.002 vsB0~.04-.06. No causal leakage/poor-fit conclusion from statistics alone. No training/model/default/metric change.

Source16f36d1 pushed. Remote/root/kinetalk_phase49_reference_coordinates_20261008 diagnostic complete; worker failed ONLY JSON2172526bytes>2MiB. Original failure + storage receipt retained;8members SHAverified, no reexecution. Never relaunch. Next style-coordinate/reference-quality and approved TRAIN-pair teacher decomposition before selecting a compact receiver/reference redesign; avoid ad hoc consistency losses or gain increases.

Two SHA-backed obsolete Phase34 smoke last.pt remote copies reclaimed152853632bytes; local originals and .codex-finalizer/phase34_smoke_last_archive_receipt.json retain restore info. Current/B0/parent/data/probe/rig/final models untouched. Root410513408/data93532160freebytes; arrange more for full training. App happy video open request returned queued, not necessarily visible until this thread shown.
