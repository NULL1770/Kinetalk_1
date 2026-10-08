# Current recovery: Phase48 style audit IN PROGRESS (2026-10-08)

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

Phase48: read PHASE48_STYLE_AUDIT_PLAN.md and scripts/audit_reference_style_swap.py. Plan2bb5b76/full audit2445ae7 pushed. Canonical full1367 B0 cache used for24smoke, strict B0 equality retained. No worker/root launched yet. Correct remote latent curves at /root/autodl-tmp/kinetalk_phase47_static_eval_20261008/latent/curves.pt, SHAeffc44326ea6a3ceebcc497b442e8b0dd33b449573eb2ab6f2bfed0eb546a183.
