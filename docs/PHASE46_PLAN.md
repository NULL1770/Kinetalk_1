# Phase46: frozen-prior deployment receiver adaptation

2026-10-08. Completed; see [PHASE46_RESULTS.md](PHASE46_RESULTS.md). Both arms finished 8 epochs, 5,456 updates, full 1,367 development evaluation, 16 videos and table/hash closure. No joint improvement; retain Phase45-u as expression reference. Do not restart Phase45 or Phase46. The fixed plan below records the original hypothesis.

## Evidence and hypothesis

45-u improves all four generated-motion probes (primary .709447609), jaw range (.153116 vs GT .175279) and correlation (.490531). MBE .827171 remains above the adapted external baselines. Its decoder was trained with a moving posterior/prior and single reference views, whereas deployment uses frozen prior means and two-reference aggregation. Better latent fit alone did not imply better motion (45-g/gu). Test the existing receiver directly, without changing the calibrated student or adding losses.

## Fixed experiment

Parent: Phase45-u final.pt SHA dfaca4c0dce60cf8380c08788dabf547db7433c098a599522f8ce2286851e4eb. Freeze neutral B0, audio prior, posterior, style encoder, semantic heads, feature statistics and scales. Train decoder only. Both arms use the same two independent neutral supports per query, exactly as deployment; no duplicated single-reference views. All 52 output channels remain open.

Same TRAIN internal fold (10903 fit /743 held speaker /890 held sentence), seed47, batch16, AdamW lr1e-4, weight_decay.01, grad clip1, 8 fixed epochs /5456 added updates. Retain original normalized position + .5 adjacent-frame displacement objective. No additional expression/content/identity loss.

- `mixed`: 1.0 frozen posterior-sampled reconstruction + .5 frozen prior-mean reconstruction.
- `deploy`: 1.5 frozen prior-mean reconstruction; do not invoke posterior in this objective.

Total reconstruction coefficient and optimizer budget match. Mixed requires two decoder passes and posterior inference; deploy one. Report that compute difference. These arms isolate the reconstruction conditioning source at fixed deployment references. Parent comparison combines receiver adaptation and reference aggregation; it is not a single-variable reference ablation. No claim that frozen style learning or target-identity transfer is fixed by this round.

## Gates and acceptance

Unit checks: only decoder gradients and state changes; prior and B0 content isolation; deployment prediction has no query-GT inlet; original masking/native-clock objective; sampled mixed branch replay; parent budget includes 24epochs/16368updates plus one inherited analytic fit10903x1.

Remote checks: longest real TRAIN batch, exact optimizer + sampling RNG save/restore, frozen state hashes, HuBERT-NaN independence, finite losses; small eight-emotion TRAIN overfit; unchanged 16-clip evaluator smoke. Formal runs only after both arms pass. No resume without binding/source/parent/protocol agreement. Preserve failed receipts rather than overwrite.

After training: full original1367 development evaluator, original raw+clip metrics/four clip-statistic probes, fixed dynamics/style interventions, all eight native25fps emotion videos per arm and unchanged baseline tables with cumulative budgets. Decoder adaptation legitimately changes posterior-oracle outputs; only B0 predictions must stay exact. Oracle never ranks candidates. Compare full geometry, F1, range, temporal accuracy jointly; loss reduction alone is insufficient and amplitude/F1 collapse rejects adoption. Do not tune from sealed test or promote defaults automatically. More development iterations are exploratory, not independent confirmation/SOTA evidence.

## Execution status

Training/evaluation/collection complete. Original pre-change snapshot 6e8d5d5 and implementation 356cea9 were pushed before dispatch. Latest read-only GPU check: 1 MiB / 0%; root free ~365 MiB, data disk ~310 MiB. Use shared data/code for future work; preserve original evidence. No new training is active.
