# Current optimization state (2026-10-10 Phase65)

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
