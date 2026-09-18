import copy
import inspect

import pytest
import torch

from scripts import export_semantic_pilot_baseline as export
from scripts.export_clocked_fullface_examples import infer_baseline


def test_selection_requires_exact_fit_metadata_and_safe_unique_ids():
    q = {"clip_id": ["fit_a"], "sentence_id": ["sentence_a"], "speaker": ["person_a"],
         "speaker_id": torch.tensor([1]), "emotion_id": torch.tensor([2])}
    row = {"clip_id": "fit_a", "sentence": "sentence_a", "speaker_name": "person_a",
           "speaker": 1, "emotion": 2}
    assert export.select_fit_queries({"clips": [row]}, q)[1] == [0]
    for changed in ({"clip_id": "dev405"}, {"clip_id": "../fit_a"}, {"emotion": 1},
                    {"speaker": 2}, {"sentence": "different"}):
        with pytest.raises(ValueError):
            export.select_fit_queries({"clips": [{**row, **changed}]}, q)
    with pytest.raises(ValueError, match="Unique"):
        export.select_fit_queries({"clips": [row, row]}, q)


class FrozenSystem:
    def stage1(self, content, valid):
        assert not content.requires_grad
        return {"b0": content[..., :52], "h0": content[..., :8]}

    def generate(self, content, mask, identity, affect, *, initial_noise, steps, base):
        self.last = (content.clone(), mask.clone(), initial_noise.clone())
        result = base["b0"] + identity["baseline"][:, None] + affect["global"][:, None, :1]
        return {"motion": torch.where(mask[..., None], result + .01 * initial_noise, 0.)}


class FrozenAudio:
    def __call__(self, features, valid):
        return {"global": features[:, :, :4].sum(1) / valid.sum(1, keepdim=True),
                "local": features[..., :4], "intensity_value": features.new_ones(1, 1),
                "emotion_logits": features.new_zeros(1, 8), "intensity_logits": features.new_zeros(1, 4)}


def test_inference_has_no_motion_input_and_uses_fixed_noise_and_clean_mask():
    assert "motion" not in inspect.signature(export.infer_acoustic_baseline).parameters
    system, audio = FrozenSystem(), FrozenAudio()
    features = torch.randn(19, 1540)
    valid = torch.ones(19, dtype=torch.bool)
    valid[7] = False
    features[7] = float("nan")
    content = features[:, :768].clone()
    identity = {"code": torch.ones(1, 12), "baseline": torch.ones(1, 52)}
    before = copy.deepcopy(identity)
    a, protocol = export.infer_acoustic_baseline(system, audio, content, features, valid,
                                               identity, "fit_a", 4, "cpu")
    b, again = export.infer_acoustic_baseline(system, audio, content, features, valid,
                                            identity, "fit_a", 4, "cpu")
    assert protocol == again
    assert protocol["query_motion_inference"] is False
    assert a["baseline52"].shape == (19, 52)
    assert a["affect_local"].shape == (19, 4)
    assert torch.equal(a["baseline52"], b["baseline52"])
    assert not a["baseline52"][7].any()
    assert torch.isfinite(system.last[0]).all()
    assert system.last[1].shape == (1, 32)
    assert not system.last[1][0, 19:].any()
    for key in identity:
        assert torch.equal(identity[key], before[key])
    assert all(not value.requires_grad for value in a.values())


def test_mismatched_content_is_rejected_before_model_call():
    features = torch.zeros(2, 1540)
    content = torch.ones(2, 768)
    with pytest.raises(ValueError, match="Matching finite"):
        export.infer_acoustic_baseline(None, None, content, features,
                                      torch.ones(2, dtype=torch.bool), {}, "fit", 4, "cpu")


def test_baseline_matches_previous_frozen_exporter_seed_and_padding():
    features = torch.randn(21, 1540)
    valid = torch.ones(21, dtype=torch.bool)
    identity = {"code": torch.zeros(1, 12), "baseline": torch.randn(1, 52)}
    old, old_protocol = infer_baseline(FrozenSystem(), FrozenAudio(), features[:, :768],
                                      features, valid, identity, "fit_a", 4, "cpu")
    new, new_protocol = export.infer_acoustic_baseline(FrozenSystem(), FrozenAudio(),
        features[:, :768], features, valid, identity, "fit_a", 4, "cpu")
    assert torch.equal(old, new["baseline52"])
    assert old_protocol["clip_seed"] == new_protocol["clip_seed"]
