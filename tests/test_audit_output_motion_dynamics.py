import copy
import hashlib

import pytest
import torch

from scripts.audit_output_motion_dynamics import (
    COMPARISONS, NOISE_SEEDS, SCHEMA, analyze, compact_report,
    correlation_change, same_recipe, validate_checkpoint,
)
from scripts.train_formal_predictable_projection import canonical_hash
from scripts.train_predictable_renderer import state_hash


def _checkpoint(arm="audio", epoch=0):
    renderer = {"weight": torch.ones(2, 2)}
    encoder = {"output.weight": torch.zeros(2, 2)}
    scales = {"displacement": torch.full((52,), .005), "std": torch.full((52,), .02)}
    recipe = {"schema": SCHEMA, "arm": arm, "protected_sha256": "fixed",
              "initial_encoder_sha256": state_hash(encoder), "motion_scales_sha256": state_hash(scales)}
    payload = {"schema": SCHEMA, "recipe": recipe, "recipe_sha256": canonical_hash(recipe),
               "completed_epochs": epoch, "step": epoch * 145, "protected_sha256": "fixed",
               "renderer": renderer, "renderer_sha256": state_hash(renderer),
               "encoder": encoder, "encoder_sha256": state_hash(encoder), "motion_scales": scales,
               **{key: hashlib.sha256().hexdigest() for key in
                  ("minibatch_sha256", "noise_time_sha256", "choice_draw_sha256")}}
    return recipe, payload


def test_checkpoint_rejects_protected_or_scale_tampering_and_zero_encoder_drift():
    recipe, payload = _checkpoint("zero", 8)
    validate_checkpoint(payload, recipe, 8)
    altered = copy.deepcopy(payload)
    altered["encoder"]["output.weight"][0, 0] = 1
    altered["encoder_sha256"] = state_hash(altered["encoder"])
    with pytest.raises(ValueError, match="zero encoder"):
        validate_checkpoint(altered, recipe, 8)
    altered = copy.deepcopy(payload)
    altered["motion_scales"]["std"][0] = 3
    with pytest.raises(ValueError, match="Motion scales"):
        validate_checkpoint(altered, recipe, 8)
    altered = copy.deepcopy(payload); altered["protected_sha256"] = "changed"
    with pytest.raises(ValueError, match="protocol"):
        validate_checkpoint(altered, recipe, 8)


def test_recipe_comparison_only_ignores_declared_condition_fields():
    base = {"arm": "audio", "training_condition": "direct-audio-local", "loss_weights": {"std": 1}}
    other = {**base, "arm": "zero", "training_condition": "zero-local"}
    assert same_recipe(base) == same_recipe(other)
    other["loss_weights"] = {"std": 2}
    assert same_recipe(base) != same_recipe(other)
    assert correlation_change(None, .9) is None


def _curves():
    target = torch.tensor([0., .1, -.04, .05, -.02, .07])[None, :, None].expand(4, 6, 52).clone()
    target += torch.arange(4)[:, None, None] * .01
    reference = {"q": {"motion": target, "valid": torch.ones(4, 6, dtype=torch.bool),
        "channel_mask": torch.ones(4, 52, dtype=torch.bool), "emotion_id": torch.tensor([0, 1, 0, 1]),
        "speaker_id": torch.tensor([2, 2, 3, 3]), "sentence_id": ["a", "b", "c", "d"],
        "times": torch.arange(6).float()[None].expand(4, -1) * .04},
        "base": {"b0": torch.zeros_like(target)}, "identity": {"baseline": torch.zeros(4, 52)}}
    names = {name for pair in COMPARISONS.values() for name in pair}
    curves = {"noise_seeds": list(NOISE_SEEDS), "decode_steps": 12, "motion": {}}
    for i, seed in enumerate(NOISE_SEEDS):
        curves["motion"][str(seed)] = {name: target + (i + 1) * .0001 for name in names}
    return curves, reference


def test_named_analysis_runs_end_to_end_with_identity_scores_and_protection():
    prior = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        curves, reference = _curves()
        result = analyze(curves, reference, samples=16)
        assert set(result["comparisons"]) == set(COMPARISONS)
        assert "speaker_2/neutral" in result["population_counts"]
        assert "speaker_3/nonneutral" in result["scores"]["audio_full"]["raw_motion"]["brows"]
        row = result["comparisons"]["audio_full_vs_matched_trained_zero"]
        assert row["paired_gt"]["nonneutral"]["brows"]["r2_improvement"] == 0
        assert row["distribution"]["centered_residual"]["nonneutral"]["brows"]["trajectory_energy_score"]["fair"]["improvement"] == 0
        assert all(result["protection_checks"].values())
        assert not result["paired_gt_dynamic_pass"]
        assert not result["generative_distribution_pass"]
        compact = compact_report({"analysis": result, "verification": {}})
        assert "speaker_2/neutral" in compact["per_identity"]["audio_full_vs_source_full"]
    finally:
        torch.set_num_threads(prior)
