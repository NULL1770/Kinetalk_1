# Current recovery entry: Phase46 COMPLETE (2026-10-08)

Read [PHASE46_RESULTS.md](PHASE46_RESULTS.md), [all 16 videos](PHASE46_VIDEO_GALLERY.md) and [tables](PHASE46_EXPERIMENT_TABLES.md) first. Older state is retained in archive/CURRENT_OPTIMIZATION_STATE_before_phase46_closure_20261008.md. Do not restart any Phase41–46 launch, pipeline, evaluation or collector.

## Verified outcome

Both Phase46 arms finished 8 added epochs / 5456 updates / full1367 development evaluation. Mixed: MBE .836943, LBE .388290, Lip3.252856mm, primary F1 .702871, jaw range .138563, correlation .490336. Deploy: .842696/.391550/3.261482mm/.705703/.135853/.491785. Lower Lip is offset by worse MBE/F1/amplitude. No joint adoption; default unchanged.

Phase45-u stays expression/dynamics reference: MBE .827171/LBE .388674/Lip3.320219mm/primary F1 .709448/jaw range .153116/correlation .490531. GT range .175279; neutral/fear F1 .468468/.477612, happy .934097. Four probes are clip-level generated-motion statistics, not frame emotion or audio-head accuracy. Other probes .663190/.729681/.668836. These single-seed development results do not establish joint SOTA.

Only existing decoder790056 parameters trained in Phase46; all141 non-decoder tensors and full B0 outputs exact. 38 local and remote checks passed before dispatch. Closure verified two205-member arm manifests,209 launch members,5 pipeline members (overlapping lists),16 video hashes with prior full-decode/native25fps/rig receipts,14 table sources. closure_verified.json exists in diagnostics and paper_tables. The temporary unsupported utf8-sig encoding was replaced by utf-8-sig; no report values changed.

Remote /root/kinetalk_phase46_receiver_20261008 and local final_experiment/evaluation/diagnostics/phase46_receiver_20261008 are complete. Latest read-only SSH confirms GPU1MiB/0%; root free382586880bytes, data disk324894720bytes. No new training active. Original helper .codex-finalizer/phase46_status.py is read-only; phase46_close.py only hashes existing outputs and writes a receipt. Never restart launch/collector to inspect results.

## Next diagnostic priority

TRAIN loss declines while internal held-speaker prior position rises (~1.7% from epoch1 to8); all three external development speakers have worse MBE. Eyebrow mean-bias MSE increases .047571 -> .051168/.051690, while jaw range shrinks9.5%/11.3%. These observations suggest checking cross-speaker/reference-dependent average pose and conditional mean behavior; they do not prove causal identity leakage or identify the reference aggregation alone as the cause.

Fixed Phase45-u reference A/B/two-aggregate comparison is the next control, then neutral/fear/surprise confusion and audio-prior/teacher/receiver error separation. Only after that evidence should a finite architecture/target candidate be chosen. Do not extend Phase46 or add losses based on training loss alone. Multi-seed, independent content readout, correct target-identity transfer, matched baseline budgets/official benchmark and human evaluation remain paper gaps.

## Persistent boundaries

Before every model code-change round, archive and push current Git. Pre-change6e8d5d5, implementation356cea9, launch2c73d2f and updatea0bc941 already have successful pushes recorded; final closure archive status is recorded in the current session. Latest independent remote-ref query timed out; do not invent fresh verification. Credentials stay in private ignored helpers. No child agents. Unrelated third_party/voca_reference/ remains untouched.

Neutral B0 frozen; audio prior ONLY emotion2vec768+prosody4; full mouth output; no queryGT deployment or motion-reconstruction gradient into student; no sealed tuning/oracle ranking/default promotion. Each completed candidate has eight emotions and original raw/clip/four-probe metrics. External baseline rows are adapted shared-rig methods, not official paper numbers. MEDTalk/DESTalker lack verified trained rows. A changed output under another reference is insufficient to prove target-identity transfer.

Root planning files now point to this concise state; old verbose planning history is backed up in .codex-finalizer/planning_archive/phase46_closure_20261008.
