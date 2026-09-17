import pytest
import torch

from kinetalk_b0.emotion_ray import NUISANCE_CHANNELS_52
from kinetalk_b0.predictable_motion import bin_centered_frames
from scripts.prepare_predictable_renderer_cache import (
    assert_residual_bins, normalize_official_final, validate_enrollment, validate_manifest_order,
)


def test_official_final_requires_both_saved_normalizations_without_mutation():
    raw = torch.arange(2 * 3 * 768, dtype=torch.float32).reshape(2, 3, 768) / 100
    valid = torch.tensor([[True, True, False], [True, True, True]])
    content = torch.randn(2, 3, 768)
    query = {"audio": torch.full((2, 3, 83), -9.), "content": content, "valid": valid}
    cache_mean, cache_std = torch.full((768,), 3.), torch.full((768,), 2.)
    probe_mean, probe_std = torch.full((768,), -1.), torch.full((768,), 4.)
    checkpoint = {"audio_source": "acoustic", "audio_stats": {"mean": cache_mean, "std": cache_std},
                  "feature_stats": {"source": "acoustic", "input_dim": 768,
                                    "mean": probe_mean, "std": probe_std}}
    transformed = normalize_official_final(query, raw, checkpoint)
    expected = ((raw - cache_mean) / cache_std - probe_mean) / probe_std
    torch.testing.assert_close(transformed["audio"][valid], expected[valid])
    assert (transformed["audio"][~valid] == 0).all()
    assert transformed["content"] is content
    assert query["audio"].shape[-1] == 83 and (query["audio"] == -9).all()
    with pytest.raises(ValueError, match="Missing audio_stats"):
        normalize_official_final(query, raw, {k: v for k, v in checkpoint.items() if k != "audio_stats"})


def test_manifest_order_and_identity_enrollment_are_strict():
    rows = [{"clip_id": "a", "sentence": "qa", "speaker_id": 4, "emotion": 1, "speaker": "mead_X"},
            {"clip_id": "b", "sentence": "qb", "speaker_id": 4, "emotion": 5, "speaker": "mead_X"}]
    bundle = {"clip_id": ["a", "b"], "sentence_id": ["qa", "qb"],
              "speaker_id": torch.tensor([4, 4]), "emotion_id": torch.tensor([1, 5])}
    validate_manifest_order(rows, bundle)
    with pytest.raises(ValueError, match="order"):
        validate_manifest_order(list(reversed(rows)), bundle)
    refs = [{"clip_id": f"r{i}", "sentence": f"rs{i}", "speaker_id": 4,
             "speaker": "mead_X", "emotion": 0} for i in range(4)]
    assert len(validate_enrollment(refs, rows)[4]) == 4
    assert len(validate_enrollment(refs[:2], rows, min_references=2)[4]) == 2
    with pytest.raises(ValueError, match=">=4"):
        validate_enrollment(refs[:2], rows)
    with pytest.raises(ValueError, match="two independent"):
        validate_enrollment(refs[:1], rows, min_references=1)
    refs[0]["sentence"] = "qb"
    with pytest.raises(ValueError, match="globally sentence"):
        validate_enrollment(refs, rows)


def test_residual_bin_check_preserves_observed_mask_and_detects_changed_target():
    torch.manual_seed(4)
    motion = torch.randn(2, 9, 52)
    valid = torch.tensor([[True] * 7 + [False] * 2, [True] * 9])
    channel_mask = torch.ones(2, 52, dtype=torch.bool)
    channel_mask[:, 48] = False
    query = {"motion": motion, "valid": valid, "channel_mask": channel_mask}
    base = {"b0": torch.randn_like(motion)}
    identity = {"baseline": torch.randn(2, 52)}
    residual = torch.where(valid[:, :, None] & channel_mask[:, None], motion - base["b0"] - identity["baseline"][:, None], 0)
    residual[:, :, list(NUISANCE_CHANNELS_52)] = 0
    target, weight = bin_centered_frames(residual, valid)
    diagnostic = {"motion_bins": target.float(), "weight": weight.float()}
    assert_residual_bins(query, base, identity, diagnostic, 0, 4)
    diagnostic["motion_bins"][0, 0, 41] += .01
    with pytest.raises(ValueError, match="motion residual bins"):
        assert_residual_bins(query, base, identity, diagnostic, 0, 4)
