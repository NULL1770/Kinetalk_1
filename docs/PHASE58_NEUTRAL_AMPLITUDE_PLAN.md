# Phase58: a bounded neutral amplitude candidate

Pre-change snapshot: `eea349a`, pushed before implementation. Phase53 remains
the deployed reference. Phase54–57 are complete and will not be repeated.

## Hypothesis and scope

Test whether a same-frame monotone calibration of frozen neutral B0 can reduce
neutral mouth amplitude error without sacrificing closure/timing. This is a
small trainable receiving-side candidate, not emotional GT fine-tuning of B0,
not a student change, and not a claim that the main residual bottleneck is solved.
The Phase54 displacement association alone does not justify this candidate:
position, amplitude and closure must improve on held data as well.

Fit one piecewise-linear monotone function per mouth/jaw channel (14:41), with
nine fixed uniformly spaced knots on [0,1], endpoints fixed to identity. Values
outside this interval pass through unchanged. Other channels pass through.
There is no reference, label, content feature, temporal convolution or query GT
input at inference. The input is only frozen B0. A monotone map cannot reverse
activation order, but may flatten peaks and alter threshold crossings; this is
not a proof of phonetic timing preservation. The expression receiver still
supports all mouth channels; this is not a hard expression mask.

One convex fit, no sweep: clip/channel-equal physical position MSE plus 0.5
adjacent-displacement MSE, ridge 0.001 toward identity knot values. Native25fps
adjacency, observation/event/channel masks are retained. The endpoints and
monotonicity limit extrapolation; no arbitrary global mouth multiplier is set.

## Data and decision fixed before fitting

Use original TRAIN internal folds, with native neutral queries and approved
emotion-to-neutral pairs only. Equal clip weights, original masks; emotional
native GT is never the target. Fit only the internal train fold; score the
speaker-held and sentence-held folds separately, and native-neutral / paired
subsets separately. Frozen B0 historically saw the wider original TRAIN, so
these are holds for the new calibration only, not full-system unseen-subject
generalization. No development query or sealed test inference or fitting.

Report raw and [0,1]-clipped metrics, per-clip rows and per-emotion groups.
For each of four held subsets (two folds x native neutral / approved pairs),
require mouth position MSE at least 1% lower; mean absolute jaw range error
lower; jaw correlation drop no more than 0.005; closure F1 drop no more than
0.01; displacement MSE no more than 1% worse. Use both raw and clipped views.
These are engineering go/no-go criteria, not significance claims. Closure
threshold is the existing 0.05 rig diagnostic, not measured phoneme error.
Missing metrics or empty subsets fail the gate. A failed candidate is retained
as a negative result and is not stacked with another tweak in this phase.

Only if all gates pass: plan a separately recorded receiver adaptation with
the corrected base, independent reference-style checks, full frozen-probe and
native video evaluation. Do not directly add this map to Phase53 predictions:
the old residual may already compensate for B0 errors. A passing isolated
neutral candidate is not a final generated-motion metric improvement.

## Validation and artifacts

Tests cover masked nonfinite data, channel/gap isolation, equal clip weighting,
identity/monotonicity, endpoint/tail behavior, fit improvement on independent
synthetic examples, provenance and held gates. Remote unit tests precede a
small smoke then one full fit. SHA-bind B0, safe manifest, source and result;
verify B0 state unchanged. Store compact fitted knots, per-clip measurements,
status and summaries; no redundant full tensor exports. Archive completed
optimizer files only after local SHA verification; keep all final models.
