import copy

import numpy as np
import pytest
import torch

from kinetalk_b0.emotion_ray import NUISANCE_CHANNELS_52
from kinetalk_b0.predictable_motion import fit_motion_path, predict_motion, weighted_clip_center
from scripts.probe_scaled_motion_basis import (
    ALPHA, RANK, fit_channel_metric, fit_fold_arm, metric_projection_oracle,
    paired_report, predict_native, run_oof, training_inputs, validate_folds,
)
from scripts.train_predictable_renderer import audio_features, state_hash


def tensors():
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(9, 5, 10, generator=generator, dtype=torch.float64)
    y = x @ torch.randn(10, 12, generator=generator, dtype=torch.float64)
    y *= torch.linspace(.01, 1, 12)
    w = torch.tensor([[4., 4., 4., 2., 0.]]).expand(9, -1).clone()
    x[:, -1], y[:, -1] = float("nan"), float("nan")
    return x, y, w


def saved_folds():
    sentences = [f"sentence{i // 3}" for i in range(9)]
    folds = []
    for number in range(3):
        val = list(range(number * 3, number * 3 + 3))
        fit = [i for i in range(9) if i not in val]
        folds.append({"fold": number, "fit_indices": fit, "validation_indices": val,
                      "fit_sentences": sorted({sentences[i] for i in fit}),
                      "validation_sentences": [f"sentence{number}"]})
    return sentences, folds


def fixture():
    x, y, w = tensors()
    sentences, folds = saved_folds()
    cm = torch.ones(9, 52, dtype=torch.bool)
    cm[:, list(NUISANCE_CHANNELS_52) + [51]] = False
    channels = cm[0].nonzero(as_tuple=True)[0].tolist()
    motion = torch.zeros(9, 5, 52, dtype=torch.float64)
    motion[..., channels] = y[..., torch.arange(41) % 12]
    clips = [f"mead_M00{i % 3 + 1}_clip{i}" for i in range(9)]
    bundle = {"features": {"content": x[..., :4], "middle": x[..., 4:8], "prosody": x[..., 8:]},
              "motion_bins": motion, "weight": w, "channel_mask": cm,
              "clip_id": clips, "sentence_id": sentences,
              "speaker_id": torch.arange(9) % 3,
              "emotion_id": torch.tensor([0, 1, 5, 0, 1, 5, 0, 1, 5]),
              "groups": {"all_expression": channels}}
    class InternalOnly(dict):
        def __getitem__(self, key):
            if key != "internal":
                raise AssertionError("Held-out bundle accessed")
            return super().__getitem__(key)
        def items(self):
            raise AssertionError("Bundle enumeration accessed held-out data")
    source = {"bundles": InternalOnly(internal=bundle), "train_ids": torch.arange(9),
              "heldout_ids": torch.empty(0, dtype=torch.long)}
    fitted = {"states": {"rrr_rank8": {"train_ids": torch.arange(9), "rank": RANK,
               "alpha": ALPHA, "method": "rrr", "motion_channel_indices": channels}},
              "selection": {"folds": folds}}
    return source, fitted


def test_metric_uses_centered_weighted_rms_midpoint_positive_median_and_padding():
    y = torch.tensor([[[-1., -2., 0.], [1., 2., 0.], [float("nan")] * 3]], dtype=torch.float64)
    w = torch.tensor([[1., 1., 0.]])
    result = fit_channel_metric(y, w)
    torch.testing.assert_close(result["rms"], torch.tensor([1., 2., 0.], dtype=torch.float64))
    assert result["floor"] == 1.5
    expected = torch.tensor([1 / 1.5 ** 2, 1 / 2 ** 2, 1 / 1.5 ** 2], dtype=torch.float64)
    torch.testing.assert_close(result["metric"], expected / expected.mean())
    shifted = fit_channel_metric(y + torch.tensor([[[100., -13., 27.]]]), w)
    assert state_hash(result) == state_hash(shifted)
    tiny = fit_channel_metric(y * 1e-100, w)
    torch.testing.assert_close(tiny["metric"], result["metric"])
    with pytest.raises(ValueError, match="positive finite"):
        fit_channel_metric(torch.zeros_like(y), w)
    y[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="observed values"):
        fit_channel_metric(y, w)


@pytest.mark.parametrize("arm", ["native", "scaled"])
def test_fold_fit_does_not_read_validation_values_or_change_rng(arm):
    x, y, w = tensors()
    ids = torch.arange(6)
    rng = torch.random.get_rng_state().clone()
    fitted = fit_fold_arm(x, y, w, ids, arm)
    x[6:], y[6:], w[6:] = float("nan"), float("nan"), float("nan")
    poisoned = fit_fold_arm(x, y, w, ids, arm)
    assert state_hash(fitted["metric"]) == state_hash(poisoned["metric"])
    for key in ("basis", "weights", "std", "spectrum"):
        assert torch.equal(fitted["transformed_model"][key], poisoned["transformed_model"][key])
    assert torch.equal(rng, torch.random.get_rng_state())


def test_native_matches_original_fit_and_scaled_inverse_teacher_duality():
    x, y, w = tensors()
    ids = torch.arange(6)
    native = fit_fold_arm(x, y, w, ids, "native")
    direct = fit_motion_path(x, y, w, ids, [ALPHA], [RANK], methods=("rrr",))[0]
    torch.testing.assert_close(predict_native(x[6:], w[6:], native),
                               predict_motion(x[6:], w[6:], direct), rtol=0, atol=0)
    scaled = fit_fold_arm(x, y, w, ids, "scaled")
    model, d = scaled["transformed_model"], scaled["metric"]["sqrt_metric"]
    target = weighted_clip_center(y[6:], w[6:])
    expected = ((target * d) @ model["basis"]) @ model["basis"].T / d
    torch.testing.assert_close(metric_projection_oracle(y[6:], w[6:], scaled), expected)
    v = scaled["inverse_native_basis"]
    torch.testing.assert_close(v.T @ (d.square()[:, None] * v), torch.eye(RANK, dtype=torch.float64))
    assert not torch.allclose(v.T @ v, torch.eye(RANK, dtype=torch.float64))
    torch.testing.assert_close(scaled["teacher_analysis_basis"], d.square()[:, None] * v)
    # Reverse is purely an audio-input intervention and ignores NaN padding.
    reversed_prediction = predict_native(x[6:], w[6:], scaled, reverse=True)
    assert torch.isfinite(reversed_prediction).all()
    assert torch.equal(reversed_prediction[:, -1], torch.zeros_like(reversed_prediction[:, -1]))


def test_saved_partition_and_train_only_bundle_guard():
    source, fitted = fixture()
    bundle, channels, speakers, folds = training_inputs(source, fitted,
                                                        expected_clips=9, expected_speakers=3)
    assert len(channels) == 41 and len(set(speakers)) == 3 and len(folds) == 3
    bad = copy.deepcopy(fitted["selection"]["folds"])
    bad[0]["fit_indices"].append(0)
    with pytest.raises(ValueError, match="partition"):
        validate_folds(bad, bundle["sentence_id"])
    bad = copy.deepcopy(fitted["selection"]["folds"])
    bad[1] = copy.deepcopy(bad[0])
    bad[1]["fold"] = 1
    with pytest.raises(ValueError, match="exactly once"):
        validate_folds(bad, bundle["sentence_id"])
    bad = copy.deepcopy(fitted["selection"]["folds"])
    bad[0]["validation_sentences"] = ["wrong"]
    with pytest.raises(ValueError, match="do not match"):
        validate_folds(bad, bundle["sentence_id"])


def test_end_to_end_oof_statistics_pairing_and_native_score_reproduction():
    source, fitted = fixture()
    bundle, channels, speakers, folds = training_inputs(source, fitted,
                                                        expected_clips=9, expected_speakers=3)
    saved = fitted["selection"]["folds"]
    x, y, w = audio_features(bundle), bundle["motion_bins"][..., channels], bundle["weight"]
    for row, (fit, val) in zip(saved, folds):
        original = fit_motion_path(x, y, w, fit, [ALPHA], [RANK], methods=("rrr",))[0]
        error = predict_motion(x[val], w[val], original) - weighted_clip_center(y[val], w[val])
        mse = float((error.square() * w[val, :, None]).sum() / (w[val].sum() * len(channels)))
        row["scores"] = [{"method": "rrr", "rank": RANK, "alpha": ALPHA, "native_motion_mse": mse}]
    output = run_oof(bundle, channels, folds, saved)
    assert output["oof_fold"].tolist() == [0] * 3 + [1] * 3 + [2] * 3
    assert len(output["states"]) == 6
    torch.testing.assert_close(output["target"], weighted_clip_center(y, w))
    for arm in ("native", "scaled"):
        for mode, groups in output["statistics"][arm].items():
            for name, stats in groups.items():
                reference = output["statistics"]["native"]["zero"][name]
                assert np.array_equal(stats[:, [1, 3]], reference[:, [1, 3]])
    report = paired_report(output["statistics"], bundle["sentence_id"], bundle["emotion_id"],
                           speakers, samples=20, seed=45)
    row = report["scaled__vs__native_full"]
    assert len(row["by_speaker"]) == 3
    assert row["populations"]["nonneutral"]["brows"]["bootstrap_requested_samples"] == 20
    assert row["populations"]["all"]["all_expression"]["clips"] == 9
    bad = copy.deepcopy(output["statistics"])
    bad["scaled"]["full"]["brows"][0, 1] += 1
    with pytest.raises(ValueError, match="different targets"):
        paired_report(bad, bundle["sentence_id"], bundle["emotion_id"], speakers, samples=20)
    saved[0]["scores"][0]["native_motion_mse"] += 1
    with pytest.raises(ValueError, match="reproduction"):
        run_oof(bundle, channels, folds, saved)
