import copy

import pytest
import torch

from scripts.audio_flow_metrics import audit_audio_flow_samples
from scripts.named_motion_distribution_metrics import audit_named_motion_samples
from tests.test_audio_flow_metrics import _bundle


def test_named_conditions_exactly_preserve_standalone_distribution_scores():
    seeds, original, reference = _bundle()
    legacy = audit_audio_flow_samples(original, reference, expected_seeds=seeds)
    names = {"full": "audio_full", "zero": "source_full", "reverse": "audio_reverse"}
    named = {**original, "motion": {seed: {names[key]: tensor for key, tensor in modes.items()}
                                      for seed, modes in original["motion"].items()}}
    before = copy.deepcopy(named)
    actual = audit_named_motion_samples(named, reference, expected_seeds=seeds)
    assert actual["modes"] == list(names.values())
    assert actual["population_clip_indices"] == legacy["population_clip_indices"]
    for kind, modes in legacy["scores"].items():
        for mode, expected in modes.items():
            assert actual["scores"][kind][names[mode]] == expected
    for seed in named["motion"]:
        for name in named["motion"][seed]:
            torch.testing.assert_close(named["motion"][seed][name], before["motion"][seed][name], rtol=0, atol=0)


def test_named_conditions_reject_inconsistent_modes_and_changed_seed_order():
    seeds, curves, reference = _bundle()
    del curves["motion"][str(seeds[-1])]["zero"]
    with pytest.raises(ValueError, match="matching condition"):
        audit_named_motion_samples(curves, reference, expected_seeds=seeds)
    with pytest.raises(ValueError, match="order"):
        audit_named_motion_samples(curves, reference, expected_seeds=seeds[::-1])
