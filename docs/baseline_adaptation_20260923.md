# MEAD / ARKit baseline adaptation contract — 2026-09-23

All named rows below are **shared-audio ARKit adaptations**, not original
end-to-end implementations or reproductions of published dataset numbers.
The purpose is comparison on the same MEAD split and output/metric protocol.
Changing the pretrained audio front end changes the method and is disclosed.

## Common protocol

- Frozen shared 1540-D features: content768, emotion768, prosody4. No raw-audio
  encoder finetuning in these runs; no new large audio cache is required.
- Train 12,536 / validation 1,367. Test 1,622, held-out identities. Native25Hz.
- Identity input is independent neutral enrollment; no query motion or test
  emotion/intensity label enters inference. No test fitting or speaker lookup.
- Same 51 observed ARKit channels; TongueOut unavailable. Identical coefficient
  masks and fixed-rig metric conversion for all methods.
- Fixed final epoch, seed42. VOCA80 epochs/batch8/lr1e-4;
  EmoTalk80 epochs/batch4/lr2e-4. These choices precede corrected test scoring
  and are based on source audit and capacity, not earlier test scores.
- The old simplified models were already evaluated. Their test scores are
  internal results, are not an untouched holdout, and do not select this recipe.
- Table labels and captions must include the adaptation scope. Scores cannot
  be compared directly to original paper tables on other data/topologies.

## FaceFormer — retain completed 100-epoch run

Source: `third_party/faceformer_upstream/faceformer.py`.
Retained: autoregressive residual decoder, one TransformerDecoder layer/four
heads, FF2D, periodic positional encoding, periodic causal attention bias,
one-to-one audio/frame alignment, neutral template addition, zero output head.
Adapted: wav2vec2-base-960h replaced by shared1540 features; vertex output52;
training-speaker one-hot replaced by neutral52->128 identity embedding; native
25Hz clock while the recorded run uses period30 and feature128. Batch mask
expanded for current PyTorch. Training uses teacher forcing; evaluation uses
free autoregression. A generated-sequence teacher-forcing equivalence test
covers neutral-token/prefix alignment. Existing completed checkpoint is valid
for this declared adaptation and does not need relabeling as an official run.

## VOCA-core — new training required

Audited source: TimoBolkart/voca commit
`50bf785a880acd3e55b3e8f28eba2fc1e3c7fbfb`, speech_encoder.py,
expression_layer.py, config_parser.py. Source/license headers are preserved.

Restore16-frame centered local window, input batch normalization, speaker
conditioning before convolutions and at bottleneck, SAME-padding stride2
convolutions32/32/64/64 (kernel3, ReLU), tanh FC128, expression50, linear
expression output plus neutral template. The PyTorch implementation follows
the audited architecture; TensorFlow is not a runtime dependency.

Deviations: shared1540 replaces DeepSpeech29, neutral52 replaces categorical
speaker condition, linear52 replaces FLAME mesh basis. No FLAME PCA basis
initialization. Coefficient MSE + 10*velocity MSE follows the default velocity
weight in coefficient units. Native25Hz and local zero boundary padding.
The old three same-resolution GELU convolutions are not this baseline.

## EmoTalk-core — new training required

Audited source: `third_party/emotalk_release/model.py`, `utils.py`,
`wav2vec.py`, official psyai-net/EmoTalk_release. Hashes are checkpointed.
Retain content512, emotion832->ReLU256, level32/person32, concatenated832
hidden state, one TransformerDecoder layer/four heads/FF832, periodic causal
bias, diagonal cross attention to emotion832, linear52 output, zero head.
Prediction is direct coefficients, as in the released model.

Adaptations: content768 and emotion768 cached streams replace two trainable
wav2vec encoders; this evaluates the decoder/disentanglement architecture with
shared encoders, not official end-to-end performance. Neutral52->32 replaces
24 categorical identities. Four MEAD levels are predicted from audio using
soft probabilities instead of providing a ground-truth level at test time.
The emotion classifier uses masked audio pooling and an8-way head. Period25
preserves a one-second period at native25Hz. No random blinking/smoothing.

Training restores paired cross-content/emotion reconstruction: for target A,
content donor B has the same speaker/sentence/intensity and different emotion;
emotion donor C has the same speaker/emotion/intensity and different sentence.
The cross output targets A; self reconstruction of B is also trained. Native
lengths are resampled individually; padded duration is never a timing signal.
When a donor is unavailable, self fallback is explicit and counted in protocol.
The official release lacks a full trainer/loss recipe, so this is a declared
training adaptation: mean cross/self MSE + .1*mean velocity MSE + .05*mean
emotion CE + .05*mean intensity CE. No claim of exact original training losses.

## Artifacts and monitoring

New outputs: `baselines/{voca,emotalk}_core_arkit_20260923` on SSH.
Old `*_arkit_memmap_20260923` results stay internal and are never overwritten.
Corrected evaluations: `evaluation/sealed_20260923/{voca,emotalk}_core`.
Table builder rejects legacy simplified VOCA/EmoTalk even if test_loaded=true.
Every corrected checkpoint stores architecture version, deviation record,
upstream source hashes, pairing counts, data binding and exact optimizer recipe.

Metrics: MBE/LBE/coefficient FDD, fixed-rig LVE/EVE(mm)/FDD(mm2), independent
motion-probe macro-F1, per-clip scores. These are adaptation baseline metrics.
KineTalk main/ablations and historical FaceDiffuser keep their own disclosed
protocols; the latter also uses KineTalk-derived frozen conditioning, so it is
not an official end-to-end FaceDiffuser reproduction.
