import copy

import pytest
import torch

from scripts.compare_formal_projection_runs import assert_matched_recipes, check_matching_references


def test_recipe_matching_allows_method_and_output_only():
    a = {"args": {"mode": "rrr", "output": "a", "seed": 46, "lr": .0002}, "input_sha256": {"cache": "hash"}}
    b = copy.deepcopy(a)
    b["args"].update(mode="pca", output="b")
    assert_matched_recipes(a, b)
    b["args"]["seed"] = 47
    with pytest.raises(ValueError, match="recipes"):
        assert_matched_recipes(a, b)
    b = copy.deepcopy(a)
    b["input_sha256"]["cache"] = "other"
    with pytest.raises(ValueError, match="recipes"):
        assert_matched_recipes(a, b)


def test_matching_reference_checks_clips_motion_masks_clock_and_identity():
    q = {"clip_id": ["a"], "sentence_id": ["s"], "motion": torch.ones(1, 2, 3),
         "valid": torch.ones(1, 2, dtype=torch.bool), "channel_mask": torch.ones(1, 3, dtype=torch.bool),
         "emotion_id": torch.ones(1, dtype=torch.long), "times": torch.tensor([[0., .04]])}
    a = {"q": q, "base": {"b0": torch.zeros(1, 2, 3)}, "identity": {"baseline": torch.zeros(1, 3)}}
    b = copy.deepcopy(a)
    check_matching_references(a, b)
    b["q"]["times"][0, 1] = .05
    with pytest.raises(ValueError, match="times"):
        check_matching_references(a, b)
