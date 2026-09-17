"""Adapter for arbitrary immutable condition names using the unchanged scorer.

Each named condition is scored as a standalone ``full`` distribution and then
renamed in the report. No condition is treated as a privileged baseline, no
curves are copied or modified, and all original mask/seed checks remain active.
"""
from __future__ import annotations

from scripts.audio_flow_metrics import audit_audio_flow_samples


def audit_named_motion_samples(curves, reference, *, expected_seeds, variogram_power=.5):
    seeds = list(expected_seeds)
    if (len(seeds) < 3 or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int for seed in seeds)):
        raise ValueError("Predeclare at least three distinct integer seeds")
    if (curves.get("noise_seeds") != seeds
            or set(curves.get("motion", {})) != set(map(str, seeds))):
        raise ValueError("Saved seeds/order differ from predeclared seeds")
    names = tuple(curves["motion"][str(seeds[0])])
    if (not names or any(not isinstance(name, str) or not name for name in names)
            or any(set(curves["motion"][str(seed)]) != set(names) for seed in seeds)):
        raise ValueError("Require nonempty matching condition names across seeds")
    result = None
    for name in names:
        single = {"noise_seeds": seeds, "decode_steps": curves.get("decode_steps"),
                  "motion": {str(seed): {"full": curves["motion"][str(seed)][name]}
                             for seed in seeds}}
        scored = audit_audio_flow_samples(single, reference, expected_seeds=seeds,
                                         variogram_power=variogram_power)
        if result is None:
            result = {key: value for key, value in scored.items()
                      if key not in ("schema", "modes", "scores", "paired_condition_response")}
            result.update({"schema": "named_motion_distribution_metrics_v1", "modes": list(names),
                           "scores": {kind: {} for kind in scored["scores"]},
                           "paired_condition_response": {kind: {} for kind in scored["scores"]},
                           "condition_pairing": "No implicit full/zero names; paired comparisons are computed by the caller."})
        elif scored["population_clip_indices"] != result["population_clip_indices"]:
            raise ValueError("Population layouts differ across named conditions")
        for kind, values in scored["scores"].items():
            result["scores"][kind][name] = values["full"]
    return result
