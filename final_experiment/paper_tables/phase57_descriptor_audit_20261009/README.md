# Phase57 descriptor diagnostic tables

Source: docs/PHASE57_RESULTS.md and the original report SHA386deee1259965a00d477d56fe29c60cdfa826149905352fa061db4b5bed51fa.

These are linear recoverability diagnostics on approved TRAIN/internal-held pairs, not generated-motion or SOTA comparisons. `energy_reduction=1-MSE/zero_MSE` after per-clip/channel centering; negative values are retained. `forward` uses all supported targets. Compare `control_forward` and `control_reverse` only to each other because they share an additional support intersection. Counts, absolute errors, TRAIN energy normalization and all emotional subgroups are retained. Mouth coverage is sparse; see the results document. No held-data fitting or new neural generator.
