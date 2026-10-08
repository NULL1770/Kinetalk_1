# Phase46 receiver adaptation: implementation and launch

2026-10-08. COMPLETE: see [PHASE46_RESULTS.md](PHASE46_RESULTS.md). Both arms finished 8 epochs / 5,456 updates and full 1,367 development evaluation. All 16 videos, table sources and original member hashes verified. Mixed/deploy primary F1 .702871/.705703, MBE .836943/.842696, jaw range .138563/.135853; joint acceptance failed. Phase45-u remains the expression reference, not an automatic default. Launch details below are historical.

## Exact change

No architecture or extra loss was added. The existing four-block, 192-wide response decoder is the only trainable module. It receives frozen B0 (52 coefficients), global expression g32, native-interpolated and centered expression dynamics u16, and reference style64. All mouth channels remain open. B0, the 772D emotion2vec768+prosody4 prior, motion posterior, reference encoder, normalization, scales, emotion/intensity heads and variances remain frozen. Motion reconstruction cannot update the student.

Both arms use two independent neutral references aggregated in the same way as inference, without repeating queries into single-reference views. Both continue from identical Phase45-u weights, seed47, TRAIN10903 fit clips, batch16, AdamW1e-4, 8epochs/5456updates. Same original position+.5 adjacent displacement objective; mixed uses q-sampled+.5p-mean, deploy uses1.5p-mean. Mixed has two decoder passes and posterior inference, deploy one. No use of sealed data, queryGT deployment, oracle selection, or default replacement.

Budget includes inherited Phase43-A24epochs/16368updates and Phase45 one analytic fit10903x1 (2048 existing local-mean coefficients), followed by this round8epochs/5456updates. The table generator now exposes inherited analytic fit and decoder pass counts.

## Verified before launch

- 38 targeted local tests and the same38 remote tests passed.
- Real longest16 TRAIN clips, 432 native frames: gradients only in decoder; B0 and every non-decoder tensor exact; HuBERT NaN isolation; optimizer, sampling RNG and order RNG restore exactly. GPU peak allocations mixed554600960bytes / deploy390356480bytes.
- Eight-emotion TRAIN small-data120 updates: deployment reconstruction first10 to last10 means mixed .302415 -> .074458 (ratio.2462), deploy .294270 -> .044933 (ratio.1527). These are training learnability checks, not held-out improvements.
- Both unchanged16-clip evaluator smokes complete; data/fold, parent SHA, code hashes, optimizer settings and inherited budgets verified before dispatch.

## Launch and recovery

Remote `/root/kinetalk_phase46_receiver_20261008`; parent Phase45-u SHA `dfaca4c0dce60cf8380c08788dabf547db7433c098a599522f8ce2286851e4eb`.

Formal workers mixed28213 / deploy28214; finite completion watcher28215. Local collector53184 at `.codex-finalizer/phase46_collect.py` waits for both full1367 evaluations, verifies original hashes and non-decoder/B0 equality, then renders16 native25fps videos and generates tables. Decoder changes legitimately change posterior-oracle predictions, which are not required to equal the parent. B0 still must equal exactly.

Do not duplicate launch or collector. `.codex-finalizer/phase46_status.py` is read-only. Local diagnostics/phase46_receiver_20261008/local_queue_state.json and closure_verified.json record completion. Do not run deploy/launch helpers again; no resume is needed.

Root free at launch659193856bytes. Data and parent checkpoints reused; code1.8MiB, each full evaluation approximately109MiB. Trainer enforces250MiB free before epochs; no historical raw evidence deleted.

## Git and remaining checks

Pre-change `6e8d5d5` successfully pushed before edits. Implementation `356cea9` and launch documentation `2c73d2f` successfully pushed; independent ls-remote confirmed `2c73d2fc38660eda7ccde2ec162701fbdef593b5`. Initial attempts hit DNS/channel/TLS resets; using the server-resolved GitHub address for the SSH tunnel fixed the connection while preserving end-to-end TLS and hostname verification. No secrets committed.

Both formal runs completed 5,456 updates. Identical initial model digest d6a41349b4fa814bbc3d71479d6346896782de25a1fbcdf8b04f0005e5e8909b and initial p reconstruction .2986413538455963 were verified before training. Only 790,056 decoder parameters trained. Closure rehashed 209 launch artifacts, 205 members per arm and 5 root pipeline members, verified 16 video hashes and 14 table sources. Non-decoder 141 tensors and full B0 outputs remain exact. The original geometry/F1/range acceptance did not pass; no default replacement. Do not infer test generalization, causal disentanglement or correct target-identity transfer from this fit.
