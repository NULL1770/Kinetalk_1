# KineTalk final experiment workspace

This repository contains the reproducible implementation for the final KineTalk experiment. The canonical run uses MEAD ARKit52 features, all eight emotion classes, a fixed speaker-disjoint train/validation protocol, and a sealed test split that is loaded only by the final evaluation stage.

The paper claims three methodological contributions: residual factorization of content, emotion, and identity-related execution; a mouth-preserving residual path; and emotion-conditioned teacher–student supervision. The upper-face motion carrier is retained as an implementation aid for natural auxiliary motion. It is not presented as frame-level audio control of eyebrow or eye trajectories.

The final experiment package is under `final_experiment/`. Development diagnostics are stored under `final_experiment/paper_tables/internal/` and are excluded from paper claims. Final paper tables are written only after the eight-emotion KineTalk run, unified baseline runs, and sealed-test evaluation have completed under the same protocol.

Use the preparation, training, and evaluation entry points in `scripts/`. Baseline adaptation is kept in `scripts/train_facediffuser_arkit.py` and the retained releases under `third_party/`. No historical probe, failed dynamic branch, old manifest, or development-only score should be treated as a final result.
