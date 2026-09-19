# Coarse motion-condition predictability probe

This diagnostic asks a narrower question than the bounded prior experiment:
can native audio predict coarse, time-local motion conditions that could later
control a motion prior? It does not train or replace the generator.

The protocol reuses the bounded experiment's fixed hash split: 613 fitting clips
from 21 sentences and 206 inner-validation clips from five held-out sentences.
The historical 64-clip outer diagnostic is excluded before target extraction and
is not used for normalization, fitting, model selection, or scoring. The source
archive may be memory mapped as a single file; this is recorded as archive
exposure, while no outer clip is retained after the split.

For each native audio-valid run, fixed non-overlapping windows of 10, 25, and 50
frames (0.4, 1, and 2 seconds) are used. No padding, gap crossing, target mask,
phase search, or lag search is allowed. Input features are fit-only normalized
audio: a fixed seeded 1536-to-64 projection plus four prosody channels, the
native-run mean, and local window deviation, standard deviation, and first/last
quarter change. The matched-static control has the same context and native-run
mean but all local blocks set to zero. An own-static intervention and reversed
audio are reported as diagnostics.

Targets are eight coarse conditions: four log RMS speeds (up, down, squint,
wide) and four signed first-to-last quarter group displacements. Motion targets
are read only for fully observed scoring windows and never enter audio inputs.
One fixed ridge model (lambda 1, unpenalized intercept) is fit separately for
audio and matched-static inputs. Clips are equal-weighted during fitting. The
primary comparison is equal-sentence, equal-window-scale, equal-target
normalized MSE: real audio minus the separately fitted matched-static model.
Negative values favor audio. Five-sentence bootstrap intervals are exploratory,
not publication significance claims.

Interpretation is deliberately bounded. A positive result supports a later
conditioning experiment; a null result rejects only this compact linear feature
family. It does not prove that nonlinear audio prediction is impossible. The
probe emits coefficients, predictions, coverage, fit-only statistics, source
hashes, and a manifest for reproducibility.

Run with:

```powershell
D:/Anaconda/python.exe -m scripts.probe_motion_condition_predictability `
  --dataset <continuous_latent_dataset.pt> `
  --reference-protocol <bounded_audio_formal24000/protocol.json> `
  --output <motion_condition_probe>
```

## Completed result (2026-09-19)

The fixed compact linear probe did not beat the independently fitted static control. Primary audio minus matched-static normalized MSE = +0.01200675; exploratory five-sentence CI [-0.00005593, +0.02406943]. Activity difference +0.00482448; direction +0.01918902. Only one of five sentences favored audio overall. No generator integration or default replacement followed. Full artifacts, exact executed source and verified manifest are under artifacts/paper_metrics_20260919/motion_condition_probe_v1. The checked-in source contains a later equivalent coverage-count clarification; the archived executed_source.py is the exact source bound to the run.
