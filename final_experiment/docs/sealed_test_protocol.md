# Sealed-test metadata protocol

## Implemented update (2026-09-23 21:00; supersedes the historical status below)

The rebuilt canonical eight-emotion manifest is
`/root/autodl-tmp/kinetalk_data/manifests/mead_eight_emotion_sealed_20260923/manifest.json`:
1,622 test queries, six neutral references, the same three held-out identities.
The 537-query/four-emotion list described below is historical and is not the
current evaluation membership.

`scripts/prepare_sealed_inputs.py` now extracts query audio/content inputs
without indexing query motion arrays, reusing the audited frozen emotion2vec
extractor. `scripts/evaluate_sealed_models.py` uses fixed completed checkpoints,
saves and SHA256-seals all predictions, then first opens query motion for
scoring. It binds source hashes, checkpoint, native artifacts, manifest/test
role, input cache, enrollment, rig and independent probe. No test statistics
are fitted and no checkpoint is selected from test scores.

Current reports are under `evaluation/sealed_20260923/<method>` in the remote
workspace; VOCA and EmoTalk completed all 1,622 test queries. Other main models
and ablations are queued. The full runbook is `docs/remote_training_20260923.md`.
Current-thread 20-minute follow-up automation ID: `kinetalk`.

## Historical audit and implementation requirements

Run the metadata-only audit before touching any test artifact:

```powershell
python scripts/audit_sealed_test_manifest.py `
  --manifest <prepared_root>/manifest.json `
  --output final_experiment/evaluation/sealed_test_manifest_audit_YYYYMMDD.json
```

The audit reads exactly one JSON manifest.  It does not open the paths named
by `artifact`, and it does not load ARKit coefficients, audio, labels, target
arrays, checkpoints, or model features.  Its output is a protocol record, not
an evaluation report.  The remote canonical invocation for the current run is:

```bash
python scripts/audit_sealed_test_manifest.py \
  --manifest /root/autodl-tmp/kinetalk_data/prepared_mead_eight_20260922/manifest.json \
  --output /root/autodl-tmp/kinetalk_final_20260922/evaluation/sealed_test_manifest_audit_20260922.json
```

The current manifest passes the identity-disjoint checks: 537 test queries,
six neutral enrollment clips, and three held-out identities (`mead_M022`,
`mead_M028`, `mead_W037`).  Its test role contains emotion IDs 0, 1, 5, and 6
(neutral, angry, happy, and sad), so it is a four-emotion test role.  It must
not be described as an eight-emotion test benchmark.  The train and validation
roles contain all eight emotions and remain the only roles consumed by
`prepare_paper_full_data.py` today.

The formal evaluator is still blocked until a sealed data path is added.  It
must accept the test role without fitting statistics, adapting parameters, or
selecting an epoch; use the frozen train statistics/configuration and the
fixed `stages/audio/final.pt` checkpoint (the dynamics stage currently stopped
with CUDA OOM).  Test inference must write predictions before target arrays are
opened for scoring.  The resulting report must record the manifest canonical
hash, manifest file SHA256, test-role canonical hash, checkpoint SHA256,
feature-statistics SHA256, neutral-reference clip IDs, and an explicit
`test_loaded=true` only after those predictions are sealed.  No test score is
valid until all of these bindings are present.

Required evaluator parameters are:

* the canonical manifest and native artifact root;
* the frozen train/validation prepared-data directory (for feature statistics,
  model config, and extractor provenance);
* the frozen KineTalk checkpoint and its stage/protocol hash;
* the model/extractor directory and device;
* an output directory that is empty at start; and
* the test-role canonical hash copied from the metadata audit.

The evaluator must reject a manifest whose `sealed_test_targets_loaded` is
already true, any train/test or validation/test identity overlap, missing
neutral enrollment, changed role hash, or a checkpoint selected using test
metrics.  It must also reject attempts to claim all-eight-emotion coverage
unless the test role is rebuilt with emotion IDs 0 through 7.
