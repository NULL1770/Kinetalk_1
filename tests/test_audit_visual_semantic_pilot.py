import json
from pathlib import Path

import numpy as np
import torch

from scripts import audit_visual_semantic_pilot as a


def fixture(tmp_path):
    run = tmp_path / "run"; run.mkdir()
    scales = {"metric_scale": torch.ones(9)}
    torch.save(scales, run / "scales.pt")
    clips = {}
    for index, split in enumerate(("holdout", "train", "train", "train")):
        frames = 8; target = np.full((frames, 9), .5 + index * .01)
        samples = {name: np.stack([target, target, target], axis=0) for name in a.ARMS}
        clips[f"clip_{index}"] = {"metadata": {"clip_id": f"clip_{index}", "sentence": f"s{index}", "split": split},
                                   "valid": np.ones(frames, bool), "native_valid": np.ones(frames, bool),
                                   "target": target, "baseline52": np.full((frames, 52), .5), "samples": samples}
    torch.save({"schema": "visual_semantic_condition_pilot_v1", "clips": clips,
                "protocol": {"sample_seeds": [42, 123, 2026, 77]}}, run / "predictions.pt")
    report = {arm: {role: {"summary": None} for role in ("holdout", "fit_examples")} for arm in a.ARMS}
    (run / "reports.json").write_text(json.dumps(report), encoding="utf8")
    return run


def test_score_is_recomputed_from_curve_and_static_baseline_is_separate(tmp_path):
    result = a.audit(fixture(tmp_path), output=tmp_path / "audit.json", bootstrap_draws=40)
    assert result["holdout_count"] == 1
    assert result["fit_example_count"] == 3
    assert result["independent_recompute"]["used_dataset"] is False
    assert result["baseline"]["holdout"]["summary"] is not None
    assert result["paired"]["holdout"]["va_audio_vs_static"]["bootstrap"]["draws"] == 40


def test_sentence_bootstrap_is_fixed_seed_and_clip_equal(tmp_path):
    rows = [{"sentence": "a", "value": 1.}, {"sentence": "a", "value": 3.}, {"sentence": "b", "value": 7.}]
    first = a._sentence_bootstrap(rows, "value", seed=11, draws=50)
    second = a._sentence_bootstrap(rows, "value", seed=11, draws=50)
    assert first == second and first["sentences"] == 2 and first["draws"] == 50


def test_missing_arm_or_shape_rejected(tmp_path):
    run = fixture(tmp_path)
    packed = torch.load(run / "predictions.pt", weights_only=False)
    del packed["clips"]["clip_0"]["samples"]["va_audio"]
    torch.save(packed, run / "predictions.pt")
    try:
        a.audit(run, output=tmp_path / "audit.json")
    except ValueError as exc:
        assert "missing va_audio" in str(exc)
    else:
        raise AssertionError("missing arm was accepted")


def test_reverse_is_explicitly_not_recomputed_from_report(tmp_path):
    result = a.audit(fixture(tmp_path), output=tmp_path / "audit.json", bootstrap_draws=5)
    assert "reverse curves" in result["independent_recompute"]["reported_reverse_arm"]


def test_independent_primary_agrees_for_unequal_gapped_trajectories():
    from scripts.joint_motion_metrics import score_clip, summarize
    rng = np.random.default_rng(729)
    scale = np.linspace(.1, .3, 9); clips = {}; records = []
    for index, frames in enumerate((15, 24)):
        mask = np.ones(frames, bool); mask[6:8] = False
        target = rng.uniform(size=(frames, 9)); samples = rng.uniform(size=(4, frames, 9))
        target[~mask] = np.nan; samples[:, ~mask] = np.nan
        clips[str(index)] = {'valid': mask, 'target': target, 'samples': {'a': samples},
                             'metadata': {'sentence': str(index)}}
        records.append(score_clip(samples, target, mask, scale))
    got = a.independent_primary(clips, list(clips), 'a', scale)
    shared = summarize(records)
    for key in ('raw', 'centered'):
        np.testing.assert_allclose(got['joint_fair_es'][key], shared['joint_fair_es'][key], atol=1e-12)
    np.testing.assert_allclose(got['rms_ratio'], shared['rms_ratio'], atol=1e-12)
    np.testing.assert_allclose(got['speed_rms'], shared['speed']['all']['rms'], atol=1e-12)
