import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from kinetalk_b0.multiscale_audio import (
    FrozenEmotion2VecTemporalPredictor, multiscale_feature_summary,
)
from scripts.probe_emotion2vec_multiscale import binned_prediction


def test_temporal_predictor_masks_invalid_frames_and_is_finite():
    torch.manual_seed(2)
    x = torch.randn(3, 17, 768)
    mask = torch.ones(3, 17, dtype=torch.bool)
    mask[1, -4:] = False
    model = FrozenEmotion2VecTemporalPredictor(hidden_dim=12)
    out = model(x, mask)
    assert out.shape == (3, 17, 1)
    assert torch.isfinite(out).all()
    assert torch.equal(out[1, -4:], torch.zeros(4, 1))


def test_temporal_predictor_valid_outputs_ignore_appended_padding():
    torch.manual_seed(45)
    x = torch.randn(2, 24, 768)
    mask = torch.ones(2, 24, dtype=torch.bool)
    mask[1, -5:] = False
    model = FrozenEmotion2VecTemporalPredictor(hidden_dim=12).eval()
    expected = model(x, mask)
    # Neither the number nor the contents of invalid frames may influence
    # temporal normalization at an observed frame.
    padded = torch.nn.functional.pad(x, (0, 0, 0, 16), value=float("nan"))
    padded_mask = torch.nn.functional.pad(mask, (0, 16))
    actual = model(padded, padded_mask)
    torch.testing.assert_close(actual[:, :24], expected, rtol=1e-5, atol=1e-6)
    assert torch.isfinite(actual).all()
    assert torch.equal(actual[~padded_mask], torch.zeros_like(actual[~padded_mask]))


def test_binned_prediction_keeps_existing_centered_contract():
    frame = torch.arange(10, dtype=torch.float32).reshape(1, 10, 1)
    valid = torch.ones(1, 10, dtype=torch.bool)
    pred, weight = binned_prediction(frame, valid, 4)
    assert pred.shape == (1, 3, 1)
    assert weight.tolist() == [[4.0, 4.0, 2.0]]
    torch.testing.assert_close((pred * weight[..., None]).sum(1), torch.zeros(1, 1), atol=1e-6, rtol=0)


def test_feature_summary_reports_temporal_variation():
    x = torch.zeros(2, 5, 4)
    x[:, 1:] = 1
    mask = torch.ones(2, 5, dtype=torch.bool)
    summary = multiscale_feature_summary(x, mask)
    assert summary["frames"] == 10
    assert summary["dimension"] == 4
    assert summary["temporal_std"] > 0
