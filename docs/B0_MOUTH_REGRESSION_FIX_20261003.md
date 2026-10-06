# B0 neutral-mouth regression fix

The 2026-10-03 safe-DTW run exposed a deterministic regression in the Stage1
boundary: the decoder was trained only on `art_indices` with affective mouth
channels removed, and its forward pass then zeroed those channels.  This made
the neutral articulation base lose valid mouth shapes before the residual
generator ran.

The corrected protocol trains B0 against the complete neutral coefficient
support.  The residual support still excludes the 15 articulatory mouth
channels, so the affect path can modify only the remaining mouth coefficients
and upper-face channels.  This preserves neutral phonetic mouth motion while
keeping the requested affective mouth residual open.

The aborted run remains archived as `kinetalk_safe_dtw_20261003`; its metrics
must not be mixed with the corrected run.
