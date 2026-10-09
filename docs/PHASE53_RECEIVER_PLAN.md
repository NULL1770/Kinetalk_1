# Phase53: receiver path diagnosis before correction

2026-10-09. Pre-change snapshot c8ff428 is already pushed. User authorizes continued optimization, code snapshots, training/evaluation/rendering. No promised gain.

1. Diagnose Phase51 statistics and Phase52 on 96 emotion-balanced clips each from original TRAIN fit, internal speaker hold, internal sentence hold. Same B0 canonical batch32, fixed independent enrollment references, frozen four probes. Never infer external validation or sealed test during this selection.
2. Intervene static bias, style modulation, global g and local u; report raw and clipped motion, position/adjacent-displacement error, jaw range/correlation and four probes. Zero paths are OOD diagnostics, not trained ablations or selectable models. Confirm exact normal-path replay and immutable checkpoints.
3. Use measured path effects plus already completed Phase50 approved-pair evidence to choose one coherent correction. Preserve neutral B0, 772D affect/prosody-only prior, no motion gradient into student, native masks/clocks and real reference-driven motion style.
4. Before training ensure disk space by SHA-backed archival of completed runs' redundant optimizer last.pt files; retain final checkpoints, all data and parent dependencies. Verify local and remote copies before removal.
5. Run meaningful code/boundary tests, fixed 120-step GPU smoke, then formal budget. Queue unchanged full1367 evaluation, 2026 matched style audit, all8emotion and style videos, complete paper tables. Keep failed outcomes and do not claim SOTA or merge checkpoint metrics.

Current status: diagnostic implemented; synthetic exact replay/static-offset invariance test passed. Remote diagnostic pending. No model change or training yet.
