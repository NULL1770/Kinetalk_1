# Migrated joint-prior launcher

`scripts/run_joint_prior_background.py` defaults to the verified layout on
`/root/kinetalk_joint_20260918`. It uses `/root/miniconda3/bin/python`, existing
audio/target/native datasets, migrated `full_native_audio_delta`, and
`run12/audio/final.pt`. All input paths can be overridden explicitly.

After migration, check dependencies, disk space and an actual CUDA allocation:

```bash
/root/miniconda3/bin/python /root/kinetalk_joint_20260918/code/scripts/run_joint_prior_background.py --check-only
```

No-GPU mode produces a clear error and launches no job. There is no retry loop
waiting for a GPU. Once a GPU is available, run smoke then the formal experiment:

```bash
/root/miniconda3/bin/python /root/kinetalk_joint_20260918/code/scripts/run_joint_prior_background.py --smoke
/root/miniconda3/bin/python /root/kinetalk_joint_20260918/code/scripts/run_joint_prior_background.py
```

Defaults are `joint_prior_smoke` and `joint_prior_formal30`. Each output must be
absent before launch. Logs are siblings named `<output>.run.log`; detached
supervisor metadata lives under `launches/<output>_<UTC timestamp>/` as
`plan.json`, `launch.json`, `pid`, and `supervisor.log`. Nothing pre-creates the
driver output. Do not delete an old output to rerun casually: use a new explicit
`--output` and preserve confirmation evaluation provenance.

`startup_verified` means a live detached supervisor and driver-written state
were both observed; it does not mean training succeeded. `spawned_not_yet_verified`
means the child has not yet written state and requires inspection of the log.
Only `launch.json` status `complete`, child exit code zero, and driver
`status.json` status `complete` establish a finished run. Motion/audio gate
success must still be read separately from `gate.json`. The formal driver runs
its fixed gated protocol; launching it does not force a failed motion gate into
local-audio training.
