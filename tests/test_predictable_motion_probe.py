from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.probe_predictable_motion import (
    final_feature_agreement, intervene_input, load_aligned_sidecar, nested_select,
)
from kinetalk_b0.predictable_motion import fit_motion_path, predict_motion


def test_nested_choice_ignores_outer_heldout_values_and_refits_each_basis():
    torch.manual_seed(82)
    x = torch.randn(9, 7, 4, dtype=torch.float64)
    y = x[..., :3] + .1 * torch.randn(9, 7, 3, dtype=torch.float64)
    w = torch.ones(9, 7, dtype=torch.float64)
    sentences = ["a", "a", "b", "c", "d", "e", "f", "g", "outer"]
    train = torch.arange(8)
    options = dict(alphas=(.01, 1.), ranks=(1, 2), folds=3, seed=11)
    before = nested_select(x, y, w, sentences, train, **options)
    xx, yy, ww = x.clone(), y.clone(), w.clone()
    xx[-1] = float("nan"); yy[-1] = float("inf"); ww[-1] = float("nan")
    assert nested_select(xx, yy, ww, sentences, train, **options) == before
    validation = []
    for row in before["folds"]:
        assert set(row["fit_sentences"]).isdisjoint(row["validation_sentences"])
        assert 8 not in row["fit_indices"] and 8 not in row["validation_indices"]
        validation.extend(row["validation_indices"])
        fit, val = torch.tensor(row["fit_indices"]), torch.tensor(row["validation_indices"])
        states = fit_motion_path(x, y, w, fit, options["alphas"], options["ranks"])
        state = states[0]
        target = y[val] - y[val].mean(1, keepdim=True)
        expected = (predict_motion(x[val], w[val], state) - target).square().mean()
        assert row["scores"][0]["native_motion_mse"] == pytest.approx(float(expected))
    assert sorted(validation) == train.tolist()


def _sidecar():
    clips = []
    for i, n in enumerate((5, 6)):
        valid = torch.arange(8) < n
        item = {"clip_id": f"c{i}", "valid": valid, "times": torch.arange(8).double() / 25}
        for key, d in (("middle", 768), ("final", 768), ("prosody", 4)):
            item[key] = torch.randn(8, d)
            item[key][~valid] = float("nan")
        clips.append(item)
    return {"clips": clips, "schema": "test"}, {"clip_id": ["c0", "c1"], "weight": torch.tensor([[4., 1.], [4., 2.]])}


def test_sidecar_alignment_uses_ids_and_requires_original_bin_counts(tmp_path):
    payload, bundle = _sidecar()
    path = tmp_path / "train.pt"
    torch.save(payload, path)
    expected, meta = load_aligned_sidecar(bundle, path, 4)
    assert meta == {"schema": "test"}
    payload["clips"].reverse()
    torch.save(payload, path)
    actual, _ = load_aligned_sidecar(bundle, path, 4)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)
    payload["clips"][0]["valid"][5] = False
    torch.save(payload, path)
    with pytest.raises(ValueError, match="bin weights"):
        load_aligned_sidecar(bundle, path, 4)


def test_sidecar_duplicate_ids_and_affine_feature_consistency(tmp_path):
    payload, bundle = _sidecar()
    path = tmp_path / "train.pt"
    torch.save(payload, path)
    raw, _ = load_aligned_sidecar(bundle, path, 4)
    stat = final_feature_agreement(raw["final"], raw["final"] * 3 + 100, bundle["weight"])
    assert stat["mean_axis_correlation"] == pytest.approx(1)
    payload["clips"][1]["clip_id"] = "c0"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="clip IDs"):
        load_aligned_sidecar(bundle, path, 4)


def test_interventions_keep_padding_zero_and_weighted_clip_mean_zero():
    x = torch.randn(3, 5, 4, dtype=torch.float64)
    w = torch.tensor([[4,4,4,1,0], [4,3,0,0,0], [4,4,1,0,0]], dtype=torch.float64)
    x[w == 0] = float("nan")
    for mode in ("reverse", "shuffle", "zero"):
        out = intervene_input(x, w, mode)
        assert torch.isfinite(out).all() and not out[w == 0].any()
        torch.testing.assert_close((out*w[...,None]).sum(1), torch.zeros(3,4,dtype=torch.float64), atol=1e-12, rtol=0)
