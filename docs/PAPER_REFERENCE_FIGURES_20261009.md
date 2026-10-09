# Paper figure reference delivery — 2026-10-09

These references use actual Phase53 predictions, the original shared ARKit rig,
and verified cached baseline predictions. They are not new model improvements.
The user now requests production of the four previously deferred figure types.
Research-paper-writing guidance was applied to evidence and captions. The
imagegen skill was inspected, but no image-generation service was called:
scientific plots/diagrams use Matplotlib and faces use actual Blender renders.

## Files

All files are under
`D:/实验室项目/新实验/kinetalk_b0_residual_train/final_experiment/evaluation/paper_reference_20261009`.
Each figure has PNG (300 dpi), editable SVG, and PDF versions in `figures/`.

|Figure|Filename stem|Use|
|---|---|---|
|1|01_motion_emotion_tsne|Real motion, Phase53 style-only, Phase53 joint in one shared projection|
|1 supplement|01b_audio_affect_tsne|The same audio-global coordinates coloured by emotion and speaker|
|2|02a_emotion_comparison|Eight emotions, seven rows including GT and adapted baselines|
|2 supplement|02b_native_time_sequence|Four fixed native timestamps of the happy utterance|
|Style|02c_reference_style|Fixed query audio/expression; three speakers' references and own A/B|
|3|03_model_architecture|Actual Phase53 architecture and training/inference boundaries|
|4|04_overview_teaser|Expressive-audio examples and reference-style overview|

`stills_transparent/` and `stills_white/` each contain 90 separate 1024×1024 PNGs.
These are intended for the user's PowerPoint assembly. Filenames map to source
clip, method and timestamp in `stills_manifest.json`. The diagram SVGs are the
editable drawing references; no PowerPoint deck was requested.

For one-file transfer, `final_experiment/evaluation/paper_reference_figures_20261009.zip`
contains the figures, all 180 stills, manifests, projection coordinates and the
render receipt (about 121 MB). `paper_reference_manifest.json` records SHA-256
for every packaged figure/still.

Method indices in filenames: m0 GT; m1 neutral B0; m2 VOCA-core; m3 EmoTalk-core;
m4 FaceFormer; m5 Phase53 joint; m6 FaceDiffuser. Style figure indices have their
own explicit mapping in the manifest and must not be confused with these.

## Caption drafts and evidence limits

**Figure 1.** Emotion structure in a frozen motion-feature space. All 1,367
development clips per domain are embedded using the first Linear+ReLU layer of
the fixed real-TRAIN motion classifier (128D). Ground truth and both generated
domains are concatenated before one unlabelled PCA50 and one t-SNE projection.
Colours denote eight emotion categories. Parameters: seed47, perplexity30,
1,000 iterations, PCA initialization, automatic learning rate. No seed search,
class filtering or separate panel projection. The overlap is retained. This
visualization does not establish temporal accuracy or factor disentanglement.
The audio supplement embeds actual g32 separately; both panels share coordinates.

**Figure 2.** Qualitative comparison on eight existing development utterances.
Within each emotion, every method is shown at the same native timestamp: the
earliest maximum of the observed GT jaw coefficient. The rig, view, material and
lighting are identical. Rows use the project's adapted baseline implementations,
not verified official reproductions. Training budgets/conditions are unequal.
The eight utterances do not all have the same sentence, so this is not a
same-content emotion intervention. The temporal supplement uses fixed 20%, 40%,
60%, 80% native positions. Word/phoneme boundaries are unavailable and no word
labels are invented. FaceDiffuser uses the first prespecified draw (seed42), not
the best of three draws; its historical archive lacks a checkpoint binding.

**Style supplement.** Reference-conditioned motion tendencies on a shared rig.
The query audio, neutral B0 and global/local expression conditions remain fixed;
only independent neutral references change. Own A/B compares separate references
of the query speaker. These are motion-style differences, not geometry identity.
Neutral references alone do not establish person-specific expression for every
emotion. See PHASE53_RESULTS.md for all 2,026 matched directed style comparisons.

**Figure 3.** Implementation diagram of Phase53. Frozen neutral B0 uses content
features; the audio prior uses only emotion2vec768+prosody4. Global g32 and native
local u16 condition the response decoder together with reference style64. Motion
posterior and distribution matching are training-only. Motion reconstruction
does not update the student. Phase53 freezes B0/prior/posterior and adapts the
reference encoder and permitted receiver parameters. All channels remain
available for expression; no anatomical mouth lock is introduced.

**Figure 4.** KineTalk overview using real expressive-audio waveforms and Phase53
joint generated faces. The four emotion rows are distinct utterances, not a
fixed-content intervention. The lower row changes only reference style for the
same happy query. This is a layout reference tied to this model version, not a
claim that every displayed emotion is already reproduced accurately.

## Verification and provenance

- Exported embeddings/predictions were SHA-verified against immutable sources.
- `data/metadata.json`, `data_facediffuser/metadata.json`,
  `figures/tsne_protocol.json` and `data/projection_coordinates.npz` retain
  sources, frozen projection inputs and exact plotting coordinates.
- `stills_transparent/render_receipt.json` binds the rig, source arrays,
  timestamps, camera and each render hash. The original blend was not saved.
- 90 transparent images and 90 white composites, 1024 square; white images are
  alpha composites only. No face retouching or per-method enhancement.
- Seven-method montages and final teaser were visually checked. Teaser images
  keep square physical proportions; the initially stretched layout was fixed.
- Separate stills provide high-resolution reusable evidence; a single still
  cannot establish lip timing or motion quality. See Phase53 videos for motion.

## Current quantitative outcome

|Phase53 model|MBE↓|LBE↓|Lip mean mm↓|Primary emotion F1↑|Jaw range|
|---|---:|---:|---:|---:|---:|
|style-only|0.763407|0.369546|3.262565|0.676805|0.124295|
|joint|0.823704|0.406063|3.471205|0.690150|0.128094|

GT jaw range is 0.175279. Primary F1 is a whole-clip motion-statistics probe,
not a framewise evaluator. Earlier checkpoints reached higher F1 with worse
geometry; their best columns must not be combined into one fictitious model.
Phase54–57 were completed diagnostics, not generator improvements. Overall SOTA,
multiseed robustness, fair official-method comparison and a human study remain
unestablished. No model is promoted on these figures.
