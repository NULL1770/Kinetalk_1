# Phase55: conditional receiver interaction probe

2026-10-09. Phase54 showed that a linear `u_a` probe does not generalize to paired mouth expression dynamics, while B0 predicts part of the neutral-base mouth displacement error. This follow-up tests one narrower hypothesis: the receiver may need a content-conditioned amplitude response, represented by frozen B0 coordinates interacting with audio-predicted emotion/intensity probabilities.

This is a bounded diagnostic, not a new model. The feature is 676D: normalized, locally centered B0 52D, B0×8 audio-prior emotion probabilities (416D), and B0×4 audio-prior intensity probabilities (208D). Probabilities come from the frozen prior global mean; query labels, query motion, posterior q, HuBERT content, and GT are never inputs. The interaction is evaluated only at the receiver-side probe and is not injected into a checkpoint.

Fit and held sets, safe native masks, paired targets, fixed ridge=.001, equal clip/channel weighting, centered trajectories, adjacent25fps displacement, TRAIN-only normalization, and no sealed data are identical to Phase54. Compare against the already archived `base52`, `u16`, `hidden128`, and `audio772` probes. No feature sweep, no arbitrary amplitude gain, no decoder loss, no student loss, and no checkpoint promotion.

Interpretation: a held improvement of conditional over B0 and audio probes would support an interaction-capacity bottleneck in the receiver; it would not prove content/emotion causality. No improvement means this simple interaction is insufficient and the target/alignment or nonlinear receiver must be investigated separately. The diagnostic cannot by itself justify adding B0 features to the audio student.
