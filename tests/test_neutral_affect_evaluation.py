import numpy as np
import torch

from scripts.evaluate_neutral_affect_pilot import (
    aggregate_seeds, evaluate_seed, interventions, paired_bootstrap,
)
from scripts.neutral_affect_metrics import motion_metrics


def affects():
    mask = torch.tensor([[True, True, False, True, True], [True, True, True, False, False]])
    teacher = {"local": torch.arange(20.).reshape(2, 5, 2), "global": torch.randn(2, 4),
               "intensity_value": torch.randn(2, 1)}
    audio = {"local": teacher["local"] + 3., "global": torch.randn(2, 4),
             "intensity_value": torch.randn(2, 1)}
    return teacher, audio, mask


def test_mean_reverse_preserve_valid_mean_global_and_intensity():
    teacher, audio, mask = affects()
    conditions = interventions(teacher, audio, mask)
    assert len(conditions) == 11
    for source, original in (("teacher", teacher), ("audio", audio)):
        for name in ("mean", "reverse"):
            changed = conditions[source + "_" + name]
            assert changed["global"] is original["global"]
            assert changed["intensity_value"] is original["intensity_value"]
            assert changed["local"][~mask].count_nonzero() == 0
            for i in range(len(mask)):
                torch.testing.assert_close(changed["local"][i, mask[i]].mean(0), original["local"][i, mask[i]].mean(0))
        for i in range(len(mask)):
            torch.testing.assert_close(conditions[source + "_reverse"]["local"][i, mask[i]],
                                       original["local"][i, mask[i]].flip(0))
    assert conditions["teacher_global_audio_local"]["global"] is teacher["global"]
    assert conditions["teacher_global_audio_local"]["local"] is audio["local"]
    assert conditions["audio_global_teacher_intensity_audio_local"]["intensity_value"] is teacher["intensity_value"]


def test_evaluate_seed_reuses_identical_noise_for_all_interventions():
    teacher, audio, mask = affects()
    class Recorder:
        def __init__(self):
            self.noises = []
        def generate(self, content, valid, identity, affect, *, initial_noise, steps, base):
            self.noises.append(initial_noise.clone())
            motion = initial_noise * .1 + affect["local"].mean(-1, keepdim=True) * .01
            return {"motion": motion}
    model = Recorder()
    query = {"motion": torch.randn(2, 5, 52), "content": torch.randn(2, 5, 3), "valid": mask,
             "channel_mask": torch.ones(2, 52, dtype=torch.bool), "times": torch.arange(5).repeat(2, 1) * .04}
    result, curves = evaluate_seed(model, query, {"b0": torch.zeros(2, 5, 52)}, {}, teacher, audio,
                                  seed=42, steps=2)
    assert len(result["conditions"]) == len(model.noises) == 11
    for noise in model.noises[1:]:
        torch.testing.assert_close(noise, model.noises[0], rtol=0, atol=0)
    assert not curves


def test_seed_aggregate_and_paired_bootstrap_keep_clips_as_units():
    teacher, audio, mask = affects()
    names = interventions(teacher, audio, mask)
    results = []
    for seed in (1, 2):
        conditions = {}
        for name in names:
            error = [1. + seed, 2. + seed] if name.endswith("full") else [2. + seed, 3. + seed]
            conditions[name] = {"masked_mse": np.mean(error),
                                **{region + "per_clip_mse": error for region in ("", "upper_", "brows_", "mouth_", "jaw17_")}}
        results.append({"conditions": conditions})
    averaged = aggregate_seeds(results)
    assert averaged["audio_full"]["mean"]["per_clip_mse"] == [2.5, 3.5]
    boot = paired_bootstrap(averaged, [0, 1], repetitions=100)
    item = boot["comparisons"]["audio_full_vs_audio_mean"]["mse"]
    assert item["clips"] == 2  # Never four independent samples from two seeds.
    assert item["mean_improvement"] == 1.
    assert item["clip_bootstrap_ci95"] == [1., 1.]


def test_brow_metrics_exclude_eyes_and_missing_channels():
    target = torch.zeros(1, 8, 52)
    target[..., 41:46] = torch.arange(8.)[None, :, None] / 8
    prediction = target.clone()
    prediction[..., :14] = 10.
    cm = torch.ones(1, 52, dtype=torch.bool)
    cm[:, 45] = False
    prediction[..., 45] = float("nan")
    metrics = motion_metrics(prediction, target, torch.zeros_like(target), torch.ones(1, 8, dtype=torch.bool), cm,
                             torch.arange(8.)[None] * .04)
    assert metrics["brows_masked_mse"] == 0
    assert metrics["brows_pred_corr"] == 1.
    assert metrics["brows_pred_corr_pairs"] == 4
    assert metrics["upper_masked_mse"] == 100.
