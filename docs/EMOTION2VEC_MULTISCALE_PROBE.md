# Frozen emotion2vec temporal probe

`probe_emotion2vec_multiscale.py` is an isolated test for the missing audio to
dynamic link. It leaves B0, neutral identity, global emotion, motion teacher,
and renderer checkpoints unchanged. The current audited cache contains the
final 768-D emotion2vec frame embedding, so the probe freezes that embedding
and trains only a 96-channel dilated temporal head (dilations 1, 2, 4) to
predict the existing stride-4 upper-face intensity field.

The temporal head receives one feature stream and uses depthwise temporal
filters plus residual pointwise mixing. It therefore supplies local and
phrase-scale context without concatenating current, delta and smoothed copies,
which overfit the 224-clip pilot. The output is passed through the same
masked binning and per-clip centering contract as the existing affect path.

If a future extractor cache stores frozen emotion2vec intermediate layers,
they can replace the 768-D frame stream at the probe boundary; no renderer or
loss redesign is required. Until then this final-layer temporal probe is the
reproducible fallback and must be judged against zero, mean and teacher
baselines on held-out sentences.
