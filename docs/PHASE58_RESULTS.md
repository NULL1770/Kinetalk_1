# Phase58 results — neutral amplitude candidate rejected

2026-10-09. One fixed monotone fit is complete. This is a receiving-side
calibration candidate, not a new full expression model. Phase53 is unchanged.

Pre-change Git8716dbe is the latest pushed state before this result/figure
revision; candidate implementation12caf7e. Worker1604 completed in202.44seconds
including remote tests and smoke. Do not relaunch. Local/remote tests11/11;
24clip technical smoke, then3298selected targets: internal fit620native neutral
+2024approved pairs, speaker-held38+281, sentence-held57+278. Original B0 exact;
student not loaded; no development query inference or sealed/test input.

## Held results

Values below use predictions clipped to[0,1] and original targets. These are
physical mouth-coordinate MSE and neutral-target diagnostics, **not** the
full generated-model MBE or emotion F1. Percentages are relative error reduction;
closure changes are percentage points. Native25fps masks/adjacency unchanged.

|Internal hold / target|Clips|Mouth MSE reduction|Jaw range absolute-error reduction|Closure F1 before → after|
|---|---:|---:|---:|---:|
|Speaker / native neutral|38|8.644%|17.325%|0.629202 → 0.614452|
|Speaker / approved pair|281|11.852%|13.027%|0.516788 → 0.508249|
|Sentence / native neutral|57|5.596%|14.423%|0.710724 → 0.686100|
|Sentence / approved pair|278|8.249%|13.646%|0.748204 → 0.733732|

Position, range, correlation and displacement gates pass; closure fails on
three of four held subsets, in both raw and clipped views. The preregistered
maximum closure F1 loss was0.01; observed losses on those subsets were
0.014750/0.024624/0.014472. The fitted jaw map sends0.05 to0.054166, shifting
the input crossing for the0.05 diagnostic threshold to0.046154. This explains
how a same-frame monotone map can alter closure events despite preserving
activation order. It is an explanation of this rig diagnostic, not proof of
phoneme errors measured from audio.

Do not integrate or change the acceptance threshold after seeing the result.
The broader amplitude benefit is useful evidence; a next receiver design must
handle opening amplitude and closure jointly and be checked on independent
held data. Do not simply anchor the map to this metric's threshold, stack a
post-hoc multiplier, or claim a full-model gain. B0 historically saw original
TRAIN, so these holds test the added map only. No new training is queued.

## Verification and recovery

Original212files/9,816,755bytes downloaded and SHA-verified;198source bindings
verified. All840summary metric fields independently replayed from per-clip rows.
Report SHA256:
`12b0c4587b496e578c5cd3f3c92fbe01ed7ab5130b7c4064446d84e841021f4f`.
Artifacts:
`final_experiment/evaluation/diagnostics/phase58_neutral_amplitude_20261009`.
See original_download_manifest.json and local_summary_verification.json.
Remote worker root `/root/kinetalk_phase58_neutral_amplitude_20261009`.

The paper teaser is being revised separately around the actual Phase53
neutral scaffold / audio expression / reference style design. Figure layout
changes do not improve generated-motion metrics.
