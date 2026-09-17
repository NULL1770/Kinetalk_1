from copy import deepcopy
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.emotion_ray import fit_emotion_rays
from scripts.probe_emotion_ray_ridge import (
    fit_fold_rays, nested_select_alpha, run_probe, prepare_supervision,
    UPPER_EXPRESSION_CHANNELS,
)


def _bundle():
    torch.manual_seed(24)
    # Each sentence has neutral/emotional clips for each enrolled speaker.
    sentence = [f"s{s}" for s in range(5) for _ in range(4)]
    speaker = torch.tensor([0, 0, 1, 1] * 5)
    emotion = torch.tensor([0, 1, 0, 1] * 5)
    means = torch.stack((emotion.float() + speaker.float() * .1,
                         emotion.float() * .4, speaker.float() * .2), -1)
    means += torch.randn_like(means) * .01
    valid = torch.ones(20, 1, dtype=torch.bool)
    channel_mask = torch.ones(20, 3, dtype=torch.bool)
    features = torch.randn(20, 7, 4)
    features -= features.mean(1, keepdim=True)
    curve = features[..., :1] * .3
    motion = curve * torch.tensor([1., .4, 0.])[None, None]
    train_ids, hold_ids = torch.arange(12), torch.arange(12, 16)
    state = fit_emotion_rays(means[:, None], valid, channel_mask, speaker, emotion,
                            train_ids, 2)
    direction = state["rays"][emotion]
    scalar = (motion * direction[:, None]).sum(-1, keepdim=True)
    base = {"residual_clip_means": means, "channel_mask": channel_mask,
        "speaker_id": speaker, "emotion_id": emotion, "motion_bins": motion,
        "weight": torch.ones(20, 7), "sentence_id": sentence,
        "features": {"acoustic": features, "content": features * 2},
        "scalar": scalar, "proxy": scalar * direction[:, None],
        "true_direction": direction, "actual_direction": direction * .8,
        "nonneutral": emotion != 0, "groups": {"all_expression": [0, 1, 2]}}
    external = {}
    for key, value in base.items():
        if torch.is_tensor(value):
            external[key] = value[16:]
        elif key == "sentence_id":
            external[key] = value[16:]
        elif key == "features":
            external[key] = {name: x[16:] for name, x in value.items()}
        else:
            external[key] = value
    return base, {"bundles": {"internal": base, "external_dev": external},
                  "rays": state, "train_ids": train_ids, "heldout_ids": hold_ids}


def test_fold_rays_ignore_validation_and_outer_heldout_means():
    bundle, _ = _bundle()
    fit_ids = torch.arange(8)
    expected = fit_fold_rays(bundle, fit_ids, num_emotions=2)
    altered = deepcopy(bundle)
    altered["residual_clip_means"][8:] = float("nan")
    altered["channel_mask"][8:] = False
    altered["speaker_id"][8:] = 999
    altered["emotion_id"][8:] = 999
    actual = fit_fold_rays(altered, fit_ids, num_emotions=2)
    for key in ("rays", "active", "raw_norms", "observed_channels", "neutral_anchors"):
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


def test_nested_selection_never_uses_outer_heldout_values():
    bundle, diagnostic = _bundle()
    args = dict(source="acoustic", train_ids=diagnostic["train_ids"],
                num_emotions=2, alphas=(.01, .1, 1.))
    expected = nested_select_alpha(bundle, **args)
    altered = deepcopy(bundle)
    for key in ("residual_clip_means", "motion_bins", "weight", "scalar"):
        altered[key][12:] = float("nan")
    for values in altered["features"].values():
        values[12:] = float("nan")
    actual = nested_select_alpha(altered, **args)
    assert actual == expected
    _, validation = actual
    for fold in validation["folds"]:
        assert not set(fold["fit_sentences"]) & set(fold["validation_sentences"])
        assert not set(fold["ray_fit_indices"]) & set(fold["all_validation_indices"])


def test_probe_writes_reviewable_artifacts_and_refuses_overwrite(tmp_path):
    _, diagnostic = _bundle()
    output = tmp_path / "probe"
    report = run_probe(diagnostic, output, alphas=(.01, .1), bootstrap_samples=20)
    assert (output / "summary.json").is_file()
    assert (output / "weights.pt").is_file()
    assert (output / "provenance.json").is_file()
    assert (output / "source/probe_emotion_ray_ridge.py").is_file()
    for source in ("acoustic", "content"):
        row = report["arms"][source]["splits"]["internal_heldout"]
        assert row["same_curve_gt_direction"]["scalar_nonneutral"]["r2_against_zero"] > .9
        assert row["interventions"]["zero"]["same_curve_gt_direction"]["scalar_nonneutral"]["r2_against_zero"] == 0
        full = row["same_curve_gt_direction"]["scalar_nonneutral"]["r2_against_zero"]
        reversed_score = row["interventions"]["reverse"]["same_curve_gt_direction"]["scalar_nonneutral"]["r2_against_zero"]
        assert reversed_score < full - .2
    with pytest.raises(FileExistsError):
        run_probe(diagnostic, output)


def _controller_bundle():
    _, diagnostic = _bundle()
    for bundle in diagnostic["bundles"].values():
        # Put two varying components in upper expression and add a dominant
        # mouth mean/motion, which the upper diagnostic must remove entirely.
        for name in ("residual_clip_means", "motion_bins"):
            old = bundle[name]
            expanded = old.new_zeros((*old.shape[:-1], 52))
            expanded[..., 5] = old[..., 0]
            expanded[..., 6] = old[..., 1]
            expanded[..., 17] = old[..., 0] * 100
            bundle[name] = expanded
        bundle["channel_mask"] = torch.ones(len(bundle["scalar"]), 52, dtype=torch.bool)
        bundle["probability"] = torch.nn.functional.one_hot(bundle["emotion_id"], 2).float() * .8
        bundle["probability"][:, 0] += .2
    return diagnostic


def test_upper_supervision_refits_masked_rays_without_mutating_input():
    diagnostic = _controller_bundle()
    original = deepcopy(diagnostic)
    changed = prepare_supervision(diagnostic, "upper_expression")
    assert changed is not diagnostic
    selected = torch.zeros(52, dtype=torch.bool)
    selected[list(UPPER_EXPRESSION_CHANNELS)] = True
    assert not changed["rays"]["rays"][:, ~selected].any()
    assert changed["rays"]["rays"][1, 5] > .9
    for key, bundle in changed["bundles"].items():
        assert not bundle["motion_bins"][..., ~selected].any()
        assert not bundle["residual_clip_means"][..., ~selected].any()
        assert not bundle["channel_mask"][..., ~selected].any()
        assert set(bundle["groups"]) == {"upper_expression"}
        assert bundle["groups"]["upper_expression"] == list(UPPER_EXPRESSION_CHANNELS)
        torch.testing.assert_close(bundle["actual_direction"], bundle["probability"] @ changed["rays"]["rays"])
        for name in ("motion_bins", "residual_clip_means", "channel_mask", "probability"):
            torch.testing.assert_close(diagnostic["bundles"][key][name], original["bundles"][key][name])
    inner = fit_fold_rays(changed["bundles"]["internal"], torch.arange(8), num_emotions=2)
    assert not inner["rays"][:, ~selected].any()


def test_upper_outer_ray_never_reads_heldout_clip_means(tmp_path):
    diagnostic = _controller_bundle()
    expected = prepare_supervision(diagnostic, "upper_expression")
    corrupted = deepcopy(diagnostic)
    corrupted["bundles"]["internal"]["residual_clip_means"][12:] = float("nan")
    corrupted["bundles"]["external_dev"]["residual_clip_means"][:] = float("nan")
    actual = prepare_supervision(corrupted, "upper_expression")
    torch.testing.assert_close(actual["rays"]["rays"], expected["rays"]["rays"], atol=0, rtol=0)
    report = run_probe(diagnostic, tmp_path / "upper", alphas=(.01, .1),
                       bootstrap_samples=10, supervision="upper_expression")
    assert report["supervision"] == "upper_expression"
    assert report["mouth_protection_evaluated"] is False
    for source in ("acoustic", "content"):
        metrics = report["arms"][source]["splits"]["internal_heldout"]["same_curve_audio_direction"]
        assert set(metrics["motion_nonneutral"]) == {"upper_expression"}
