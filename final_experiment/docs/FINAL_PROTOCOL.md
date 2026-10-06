# Final evaluation protocol

The paper run uses MEAD converted to ARKit52 at 25 FPS. All eight emotion labels are included in the training protocol, with speaker-disjoint train and development identities. The sealed test role remains unopened until the final evaluation stage.

KineTalk and every protocol-matched baseline first produce the common 52-D ARKit blendshape sequence. Table 0 reports this common BS space with MBE, LBE, and coefficient FDD. The vertex table is then produced by passing the same BS predictions through one fixed ARKit-to-mesh rig and comparing the resulting vertices with the reference vertices. Table 1 uses the EmoTalk-style max-over-lip/expression-vertices per-frame LVE/EVE definition, averaged over valid frames, plus vertex FDD; mean-region LVE/EVE are retained only as supplementary diagnostics. Table 2 reports the declared ARKit MBE/LBE comparison for EmoTalk, FaceDiffuser, and Ours. Emotion evaluation reports eight-class recognition, macro-F1, per-emotion scores, and a descriptive t-SNE on generated clips. Baselines must use the same split, frame rate, masks, audio clock, fixed rig, and metric definitions.

The auxiliary upper-face carrier is evaluated only as a motion-distribution diagnostic. The manuscript does not claim frame-level audio-to-eyebrow or audio-to-eye trajectory alignment.

No development score is copied into the final paper tables. A table is publishable only when its manifest records `test_loaded=true`, the method checkpoint hash, the split hash, and the metric protocol hash.
