# Paper tables

`internal/development_canonical_20260922/` contains the last development snapshot for debugging only. Its rows are not final results: it uses development data, has `test_loaded=false`, and leaves the formal ablation and baseline rows pending.

The publishable directory remains empty until KineTalk and the unified baselines are evaluated on the sealed test split under the locked ARKit52 protocol. This prevents old checkpoints, smoke runs, or failed dynamic probes from being mistaken for paper evidence.
