# v7 fast feasibility evidence (2026-10-13)

This directory records a bounded real-data pilot for the proposal-aligned
three-factor model. It is an engineering gate, not a paper result.

## Data and protocol

The remote preparation used the existing TRAIN-only packed MEAD cache and the
verified HDTF window manifest. The MEAD subset contains 8 speakers, all 8
emotion IDs, independent same-speaker references with different sentence and
emotion, 256 fit queries, 64 held-sentence queries and 24 unseen-speaker
queries. The pilot also appends 32 HDTF windows. HDTF emotion and intensity
losses are masked (`emotion_id=-1`, `intensity_valid=false`). No sealed/test
file was loaded. The exact provenance is in `data_provenance.json`.

The visual stage trains the motion trajectory teacher and complete population
decoder while fitting a conditional individual-response posterior. The audio
stage freezes the visual teacher, decoder and content observer, then trains the
audio trajectory student and correlated Flow Matching prior. The prediction
path consumes query audio, optional B0-derived `h0` articulation anchor and an
independent reference; query ground-truth motion never enters inference.

## Gate outcome

Three short pilots were run. `pilot_a` exposed signed velocity cancellation and
unstable event trajectories. `pilot_b` fixed those and passed the observer and
roughness checks, but did not produce emotion control or cross-emotion style
separation. `pilot_c` added the articulation anchor and output-level emotion
losses. It still failed the promotion gates:

| held-sentence metric | v7 pilot_c | retained v5 on same crop |
|---|---:|---:|
| mouth MAE | 0.0521 | 0.0373 |
| full motion MAE | 0.0769 | 0.0472 |
| motion-emotion probe accuracy | 0.125 | 0.609 |
| sample acceleration | 0.0149 | 0.00768 |

Pilot v7 does show nonzero seed diversity (0.01485 mean absolute motion
difference) and a smoothness improvement over pilot_a. It does not show a
usable emotional realization or a reliable personal-response advantage:
same-person reference-code distance / different-person distance is 0.993, and
the held-sentence style-swap reconstruction advantage is absent. These are
rejection signals, not evidence that the proposed factorization is impossible.

## Rendering

`render_emotion1c/comparison.mp4` and `preview.png` contain the held-sentence
visual check with six synchronized columns: GT, B0, v7 teacher, v7 audio mean,
v7 sample and v7 controlled. The render is display evidence only; it does not
change or score source coefficients.

## Decision

Do not start a long v7 training run from this prototype. Retain v5 as the
quality baseline. The next implementation must make the visual semantic target
identifiable before adding capacity: train a frozen external emotion observer
on held-out real motion, supervise teacher/output trajectories through that
observer, and replace the current posterior response fit with a support-aware
same-speaker cross-emotion objective. A long run is allowed only after a short
pilot passes emotion accuracy, content/phoneme preservation, style separation,
mouth MAE and dynamic roughness gates simultaneously.

`paper_result=false` is intentional in all JSON reports.
