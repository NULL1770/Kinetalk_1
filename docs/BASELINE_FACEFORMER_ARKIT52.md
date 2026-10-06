# FaceFormer on MEAD-ARKit52

`scripts/faceformer_arkit_adapter.py` is the reproducible adapter used for
the baseline audit.  It retains the upstream FaceFormer components that define
the method—periodic positional encoding, ALiBi-style temporal bias, an
autoregressive Transformer decoder, and the zero-initialized motion residual
head.  The changed interface is explicit:

* the audio input is the locked 1540-D frame cache generated from audio only;
  no transcript or text feature is read;
* the target is a 52-dimensional MEAD-ARKit coefficient vector rather than a
  mesh vertex displacement;
* the upstream subject one-hot style code is replaced by a 52-D neutral
  enrollment anchor.  Anchors are built from independent neutral clips, so a
  held-out validation speaker has a reference condition while its query motion
  is never used to optimize the model.

The adapter accepts a `torch.save` object containing exactly `train` and
`validation` under `splits`, as returned by the prepared-data loaders.  It
rejects a payload with `test_loaded=true`, overlapping train/development
speakers, missing validity masks, or a feature dimension other than 1540.
The smoke mode only checks the interface and must never be reported as a
paper score.

Example commands (after the canonical prepared cache is materialized):

```powershell
python scripts/faceformer_arkit_adapter.py `
  --data artifacts/faceformer/prepared_train_val.pt `
  --output artifacts/faceformer/smoke `
  --device cuda --smoke --steps 4 --batch-size 2

python scripts/faceformer_arkit_adapter.py `
  --data artifacts/faceformer/prepared_train_val.pt `
  --output artifacts/faceformer/formal_train `
  --device cuda --epochs 100 --batch-size 8 --log-every 25

python scripts/faceformer_arkit_adapter.py `
  --data artifacts/faceformer/prepared_train_val.pt `
  --infer-checkpoint artifacts/faceformer/formal_train/checkpoint.pt `
  --output artifacts/faceformer/validation_predictions `
  --device cuda --split validation
```

The prediction archive is coefficient-space output only.  It can be sent to
the existing locked ARKit-to-rig evaluator to obtain LVE, EVE and FDD under
the same masks and units as KineTalk and FaceDiffuser.  This adapter does not
claim official FaceFormer numbers until a sealed test run has been loaded and
the unified evaluator has produced a `test_loaded=true` report.

