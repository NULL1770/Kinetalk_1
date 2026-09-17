import copy

import numpy as np
import pytest
import torch

from scripts.audit_audio_conditioned_flow_probe import (
    ARMS, NOISE_SEEDS, analyze_curves, paired_score_summary, statistics_for_curves,
    validate_curves, validate_rng,
)
from scripts.train_projection_schedule_ablation import draws


def fixture_curves():
    rng = torch.Generator().manual_seed(16)
    n, frames, channels = 6, 5, 52
    target = torch.randn(n, frames, channels, generator=rng) * .03 + .2
    valid = torch.ones(n, frames, dtype=torch.bool)
    valid[0, -1] = False
    ref = {"q": {"motion": target, "valid": valid, "channel_mask": torch.ones(n, channels, dtype=torch.bool),
        "emotion_id": torch.tensor([0, 1, 1, 0, 1, 1]), "speaker_id": torch.tensor([0, 0, 1, 1, 2, 2]),
        "sentence_id": ["s0", "s1", "s2", "s3", "s4", "s5"],
        "times": torch.arange(frames).double()[None].expand(n, -1) / 25},
        "base": {"b0": torch.zeros_like(target)}, "identity": {"baseline": torch.zeros(n, channels)}}
    curves = {"noise_seeds": list(NOISE_SEEDS), "decode_steps": 12, "motion": {}}
    for seed in NOISE_SEEDS:
        error = torch.randn(target.shape, generator=torch.Generator().manual_seed(seed)) * .015
        pred = target + error
        curves["motion"][str(seed)] = {mode: pred.clone() for mode in ("full", "zero", "reverse", "oracle")}
    return ref, curves


def test_fixed_seed_contract_and_sample_error_aggregation():
    ref, curves = fixture_curves()
    stats, _, _, _ = statistics_for_curves(curves, ref)
    altered = copy.deepcopy(curves); altered["noise_seeds"] = list(reversed(NOISE_SEEDS))
    with pytest.raises(ValueError, match="eight seeds"):
        validate_curves(altered, ref)
    row = stats["full", "raw_motion", "brows"]
    target = ref["q"]["motion"][..., 41:46]
    mask = ref["q"]["valid"][..., None]
    expected = torch.stack([torch.where(mask, (curves["motion"][str(s)]["full"][..., 41:46] - target).double().square(), 0).sum((1, 2)) for s in NOISE_SEEDS]).mean(0)
    np.testing.assert_allclose(row[:, 0], expected.numpy(), rtol=1e-12)
    mean_prediction = torch.stack([curves["motion"][str(s)]["full"] for s in NOISE_SEEDS]).mean(0)
    mean_error = torch.where(mask, (mean_prediction[..., 41:46] - target).double().square(), 0).sum((1, 2))
    assert float(row[:, 0].sum()) > float(mean_error.sum())


def test_paired_score_bootstrap_uses_sentence_clusters_and_positive_improvement():
    row = paired_score_summary([1., 3., 2., 4.], [2., 4., 3., 5.], ["a", "a", "b", "b"], samples=100)
    assert row["sentences"] == 2 and row["clips"] == 4
    assert row["improvement"] == 1.
    assert row["improvement_ci95"] == [1., 1.]
    with pytest.raises(ValueError, match="pairing"):
        paired_score_summary([1.], [1., 2.], ["a"])


def test_mean_raw_protection_and_distribution_gates_are_separate():
    ref, frozen = fixture_curves()
    arms = {"frozen": frozen, "audio_local": copy.deepcopy(frozen), "zero_local": copy.deepcopy(frozen)}
    # Improve centered brow trajectory error but introduce a static brow bias.
    # Raw eye/mouth remain unchanged; the explicit upper-face mean guard must fail.
    for seed in NOISE_SEEDS:
        p = arms["audio_local"]["motion"][str(seed)]["full"]
        p[..., 41:46] = ref["q"]["motion"][..., 41:46] + .3
    result = analyze_curves(arms, ref, samples=30)
    assert all(result["paired_gt_dynamic_checks"].values())
    assert result["protection_checks"]["eyes_mouth_raw_mse_upper90_within_1pct"]
    assert not result["protection_checks"]["nonneutral_upper_mean_mse_upper90_within_1pct"]
    assert not result["paired_gt_dynamic_pass"]
    assert not result["generative_distribution_pass"]
    assert result["protection_reference"].startswith("frozen_full")
    assert set(result["distribution_comparisons"]) == {"frozen_full", "frozen_zero", "own_zero", "own_reverse", "matched_adapted_zero"}


def test_rng_replay_rejects_agreeing_but_wrong_arm_hashes():
    import hashlib
    q = {"motion": torch.zeros(3, 2, 52)}
    loaded = {"cache": {"splits": {"train": {"q": q}}}}
    generator = torch.Generator().manual_seed(46)
    b, n, c = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    records = {}
    for epoch in range(1, 9):
        ids = torch.randperm(3, generator=generator)
        noise, time, choose = draws(generator, 3, q["motion"].shape[1:])
        b.update(ids.numpy().tobytes()); n.update(noise.numpy().tobytes()); n.update(time.numpy().tobytes()); c.update(choose.numpy().tobytes())
        records[epoch] = {"minibatch_sha256": b.hexdigest(), "noise_time_sha256": n.hexdigest(),
            "teacher_choice_draw_sha256": c.hexdigest(), "samples_seen": 3, "step": epoch,
            "epoch": epoch, "t0_fraction": int((time == 0).sum()) / 3}
    arms = {arm: {"records": copy.deepcopy(records), "checkpoints": {8: {"rng": {"training_generator": generator.get_state()}}}} for arm in ARMS}
    validate_rng(arms, loaded)
    for data in arms.values():
        data["records"][3]["noise_time_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Replayed RNG"):
        validate_rng(arms, loaded)
