# Final evaluation protocol

The paper run uses MEAD converted to ARKit52 at 25 FPS. All eight emotion labels are included in the training protocol, with speaker-disjoint train and development identities. The sealed test role remains unopened until the final evaluation stage.

KineTalk is evaluated in ARKit coefficient space with mean blendshape error, lip-region error, facial-dynamics distance, and mouth-protection checks. The vertex table is produced only when the same ARKit predictions are passed through the fixed ARKit-to-mesh rig and compared with the corresponding reference vertices using the locked LVE protocol. Emotion evaluation reports eight-class recognition, macro-F1, per-emotion scores, and a descriptive t-SNE on generated clips. Baselines must use the same split, frame rate, masks, audio clock, and metrics.

The auxiliary upper-face carrier is evaluated only as a motion-distribution diagnostic. The manuscript does not claim frame-level audio-to-eyebrow or audio-to-eye trajectory alignment.

No development score is copied into the final paper tables. A table is publishable only when its manifest records `test_loaded=true`, the method checkpoint hash, the split hash, and the metric protocol hash.
