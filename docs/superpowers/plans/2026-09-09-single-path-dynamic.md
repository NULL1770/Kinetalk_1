# Single-Path Dynamic Control Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the redundant Stage 3/4 dynamic system with one low-rate probabilistic trajectory that the renderer demonstrably uses.

**Architecture:** Keep only `dynamic_mean` and `dynamic_logvar` at the existing low rate. Remove full-rate mean/variance, event heads, and bridge/calibrator modules. Add one zero-initialized dynamic-to-motion adapter in the renderer path and train Stage 4 with explicit dynamic envelope/velocity supervision plus a condition-use intervention metric.

**Tech Stack:** PyTorch, YAML config, pytest-style standalone regression tests, remote CUDA training.

**Spec:** `ARCHITECTURE.md` and the user's dynamic ablation/evaluation report.

## Global Constraints

- Stage 1 and Stage 2 checkpoints remain fixed teachers.
- Raw 52-D BS and aligned_dtw_v2 timeline remain unchanged.
- AMP/bfloat16 remains enabled unless a fresh numerical test proves otherwise.
- No full-rate dynamic or event output remains in the main model interface.
- Stage 4 must preserve content/style paths while making dynamic intervention measurable.

---

### Task 1: Single-path model interface

**Files:**
- Modify: `kinetalk_b0/models/encoders.py`
- Modify: `kinetalk_b0/models/model.py`
- Modify: `kinetalk_b0/models/dit.py`
- Test: `test_dynamic_single_path.py`

- [ ] Write and run the failing adapter/interface tests.
- [ ] Add `DynamicResidualAdapter(dynamic_dim, motion_dim)` with zero-initialized output projection and one temporal dynamic input.
- [ ] Remove `dynamic_full_*`, full-rate event heads, `BoundedResidualBridge`, and Stage4 full-rate calibrator from active paths.
- [ ] Feed the single upsampled `dynamic_mean` through the adapter at the renderer context and retain global/style conditioning.
- [ ] Run the tests and a CPU compile/import smoke test.

### Task 2: Simplify Stage 3 losses and training

**Files:**
- Modify: `kinetalk_b0/losses.py`
- Modify: `train.py`
- Modify: `configs/train.yaml`
- Test: `test_dynamic_single_path.py`

- [ ] Add a failing test that Stage3 loss accepts only low-rate dynamic outputs and returns finite values.
- [ ] Remove observable full-rate shape/slope/event/NLL terms from `stage3_loss`.
- [ ] Keep global alignment, emotion/intensity CE, Gaussian dynamic distribution, low-rate shape/amplitude/slope losses, and render closure.
- [ ] Make the renderer closure use the one dynamic trajectory only.
- [ ] Remove obsolete config weights and update comments.
- [ ] Run numerical tests in FP32 and bfloat16.

### Task 3: Make Stage 4 train dynamic use

**Files:**
- Modify: `kinetalk_b0/models/model.py`
- Modify: `kinetalk_b0/losses.py`
- Modify: `train.py`
- Modify: `configs/train.yaml`
- Test: `test_dynamic_single_path.py`

- [ ] Add a failing test showing two dynamic trajectories produce different adapter-conditioned renderer inputs.
- [ ] Restrict Stage4 trainable parameters to the single dynamic adapter (and any explicitly necessary dynamic projection), keeping Stage1/2/3/global/style frozen.
- [ ] Add low-rate dynamic envelope and velocity losses against the canonical residual descriptor.
- [ ] Keep flow matching as the primary objective and use a small weighted dynamic-use objective.
- [ ] Add condition dropout only after the dynamic path is active, with a reduced dropout probability.
- [ ] Run one-batch forward/backward checks for finite loss and nonzero dynamic-adapter gradients.

### Task 4: Checkpoint migration, evaluation, and remote retraining

**Files:**
- Modify: `scripts/02_eval_all_stages.py`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `progress.md`
- Modify: `findings.md`

- [ ] Make evaluation consume the single dynamic key and report dynamic condition intervention at low rate.
- [ ] Add checkpoint versioning so old multi-dynamic checkpoints are rejected with a clear message rather than silently loaded.
- [ ] Back up old Stage3/Stage4 checkpoints on the remote host.
- [ ] Retrain Stage3, then Stage4, with AMP enabled.
- [ ] Verify every new checkpoint tensor is finite.
- [ ] Run the full evaluator and compare dynamic correlation/intervention, emotion, style, and final MAE against the user's baseline.
