# Phase52: restrict reference adaptation to style parameters

Hypothesis: keep the useful statistical reference representation while preventing the shared receiver weights from drifting under style adaptation. Phase51 improved reference stability but compressed mouth amplitude; cause remains unproven.

One fixed seed47,8epoch run from Phase45-u, same statistical candidate initialization and diverse neutral TRAIN supports as Phase51. Only statistical style encoder,52D bias and32style-conditioning columns in each of four modulations train. Frozen: neutralB0,772D emotion/prosody prior,posterior,semantic heads,decoder input/4temporal blocks/output,g/u modulation columns and modulation biases. Original normalized position+.5adjacent displacement objective; no new loss, gains, queryGT input or retiming.

Partial linear weights: mask frozen-column gradients before clipping, use zero AdamW decay on those matrices so masked columns cannot drift, verify frozen decoder digest every epoch and on resume. Other style-only parameters retain original .01weight decay. Record effective trainable count. Meaningful tests must verify exact protected tensors across steps and resumed optimizer, nonzero style gradients and style-dependent output, legacy joint path, contentNaN isolation and strict serialization.

Same120step smoke gate loss last10<.9first10. Formal fixed8epochs if smoke passes; internal-held diagnostics, no favorable epoch selection. Queue unchanged full1367 evaluation and candidate style audit if storage permits; all-eight-emotion renders and tables follow. A parameter freeze does not prove content independence, dynamic accuracy or successful geometry tradeoff. Report failures; no automatic promotion.

Comparison Phase51 statistics vs Phase52 isolates allowed updates, with necessary zero-decay protected matrices explicitly recorded. Previous temporal/statistics comparison confounds encoder and initialization. Continue to preserve fullraw/clip/fourprobes and held-out protocols. Archive/push before implementation; future paper figures deferred.

Implementation ready:56targeted local checks pass, including exact protected receiver tensors across optimizer steps and resumed state, forbidden momentum detection, style-dependent outputs and unchanged joint training. Mixed matrix gradients masked before clipping, weight decay0; all other style parameters .01. No network architecture/loss change. Phase51 table labels corrected from actual immutable support/scope protocol; original numeric reports unchanged.

## Verified formal dispatch

Implementationf7adcfb pushed (Git reset/connect failures recovered before dispatch). RemotePID4435,56remote checks pass.120step smoke .691840 -> .449539, .649773ratio; exact protected receiver/emotion/B0 and contentNaN isolation pass. Formal epoch1step301 at~.142s/update. Eight fixedepochs/5456updates; no selection.197original preflight/source/smoke artifacts19323576bytes SHAverified. Full evaluator/style audit queued in worker, local collectorPID34668 queued finite2h for originals,tables,16videos. Next human/model video review remains pending; no metric success claimed. Smoke29seconds is not formalepoch duration. ETA25-35minutes for entire queue from firstformal301step snapshot.

Latest actual formal check:epoch2,step883; firstformal682updates epoch92.37seconds. Firstepoch frozen receiver/prior/B0 gates passed before checkpoint. Internal normalized diagnostics speaker_position0.438452,sentence_position0.591247; not MBE/F1 or external gains. Collector alive/waiting without error. Read .codex-finalizer/phase52_latest_status.json for original snapshot; do not relaunch.
