# Current: Phase64 training (2026-10-10)

Read docs/PHASE64_BOUNDED_RESIDUAL_PLAN.md first after compaction.

Phase53 style-only is the retained baseline: MBE .763407, LBE .369546, lip3.262565mm, F1 .676805, jaw range .124295 (GT .175279), correlation .477797, closure F1 .401021. Phase63 finished and was rejected (MBE .798408, range .093842, closure .227445). Do not resume/relaunch old workers. Historical state is archived in docs/archive/CURRENT_OPTIMIZATION_STATE_through_phase63_20261010.md.

Phase64 adds only 4212 zero-initialized bounded residual-gain parameters, gain [0.5,1.5]. Parent receiver/affect/style/posture/B0 all frozen; original objective and native masks. 46 local tests pass. Git prechange0bc0bad pushed. Formal fixed seed47,8epochs,lr3e-4; GPU smoke/training not yet dispatched at this snapshot. Full1367 eval, original2026style, new25person audit,8+8 standard videos plus8multistyle planned. No gain claimed.

SSH reachable,4090 idle. Four completed Phase63 redundant remote artifacts were SHA-verified against local originals then removed; weights seed47/final.pt retained. Receipt .codex-finalizer/phase64_archive_receipt.json; root free233369600 bytes. Never delete unverified data or third_party/voca_reference.

CREMA-D location pending user reply; MEAD main training continues. See Phase64 plan for dataset evidence and correct usage without neutral pairs. No sealed/test read. Per-change Git upload remains required. Current render/images/figures are actual Phase53 or rejected63, not Phase64.

Latest override: Phase64 worker5447 active, observed epoch1step601, .119sec/update;120GPU smoke passed ratio .891971. Implementation17457cc pushed and independently verified. Local collector6380 queued full download/replay/24videos plus25-person curves; two-hour bound, no remote restart. Check .codex-finalizer/phase64_status.py and local_queue_state before any new dispatch. 46local checks; remote full preflight passed. Source snapshots remain immutable even if local plotting/docs change later.
CREMA-D FOUND: /root/autodl-tmp/kinetalk_data/processed/native_affect_style_v4_refmask (train5797/72people,val723/9people). 3D coefficients exist;83D old audio cache needs compatible772 sidecars. Do not claim absent/unusable solely for lacking paired neutral. This run remains MEAD-only; plan documents next cross-domain use and unknown intensity -1. Test untouched.

Last observed formal progress: epoch2/8, step1283/5456, about0.122s/update. Worker5447 and localcollector6380 active at last checks; no full metrics yet. Estimated remaining training10–15min, combined evaluation/many-person audit/download/render45–75min depending I/O and rendering. Local plotting commit8f70385 is newer than immutable training implementation17457cc; this is intentional, never replace source files in active remote run.
