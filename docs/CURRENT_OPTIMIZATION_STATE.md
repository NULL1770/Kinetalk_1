# Current optimization state (2026-10-10 Phase65)

Phase53 style-only remains the retained baseline: MBE .763407, LBE .369546, lip 3.262565 mm, main probe F1 .676805, jaw range .124295 (GT .175279), jaw correlation .477797, closure F1 .401021. Phase64 bounded residual was fully collected but rejected (MBE .762427, LBE .376756, jaw range .122752, correlation .464588, main probe F1 .671319); see docs/PHASE64_RESULTS.md.

Phase65 was an affect-only zero-start additive adapter. It trained only `decoder.affect_adapter` from the frozen Phase53 style-only checkpoint. Inputs were global/local affect latents, reference response code and local RMS envelope; B0/content features were excluded. The existing position + adjacent-displacement reconstruction objective was unchanged. Commit 0ad7e9d is pushed and source-bound.

The first remote launch stopped at the existing 250 MiB disk floor before smoke. Verified redundant remote Phase53/64 artifacts were removed, preserving only the Phase53 style-only final parent. The same Phase65 run was relaunched exactly once. Smoke passed (120 steps, loss ratio .374326, frozen/B0 exact). Formal seed47 completed 8 epochs and 5,456 updates on remote `/root/kinetalk_phase65_affect_adapter_20261010`; development evaluation completed on all 1,367 clips with `test_loaded=false`.

Phase65 failed the development acceptance gate and is rejected. Prior-mean results were MBE .835922, LBE .381512, LVE 3.301901 mm, EVE .807537 mm, jaw range .124278, jaw correlation .476915, and four probe F1 values .566034/.537924/.624539/.571697. Phase53 style-only remains the retained model. No Phase65 metric, checkpoint, or video is promoted as a replacement, and no sealed/test data were read.
