# Current optimization state (2026-10-10 Phase65)

Phase53 style-only remains the retained baseline: MBE .763407, LBE .369546, lip 3.262565 mm, main probe F1 .676805, jaw range .124295 (GT .175279), jaw correlation .477797, closure F1 .401021. Phase64 bounded residual was fully collected but rejected (MBE .762427, LBE .376756, jaw range .122752, correlation .464588, main probe F1 .671319); see docs/PHASE64_RESULTS.md.

Phase65 is an affect-only zero-start additive adapter. It trains only `decoder.affect_adapter` from the frozen Phase53 style-only checkpoint. Inputs are global/local affect latents, reference response code and local RMS envelope; B0/content features are excluded. The existing position + adjacent-displacement reconstruction objective is unchanged. Commit 0ad7e9d is pushed and source-bound.

The first remote launch stopped at the existing 250 MiB disk floor before smoke. Verified redundant remote Phase53/64 artifacts were removed, preserving only the Phase53 style-only final parent. The same Phase65 run was relaunched exactly once. Smoke passed (120 steps, loss ratio .374326, frozen/B0 exact). Formal seed47 is active at epoch 2 step 783/5456 on remote `/root/kinetalk_phase65_affect_adapter_20261010`; last observed speed .226 s/update. No full metrics, promotion, video, SOTA claim, or sealed/test access yet.
