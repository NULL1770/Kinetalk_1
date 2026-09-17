from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetalk_b0.gated_audio_dynamic import ContentGatedDynamicPredictor
from scripts.probe_audio_content_gating import (
    field_prediction, intervene_content, normalize_inputs, prepare_dynamic_audio,
)


def _query():
    torch.manual_seed(8)
    valid = torch.ones(3, 13, dtype=torch.bool)
    valid[1, 10:] = False
    return {"audio": torch.randn(3, 13, 8),
            "content": torch.randn(3, 13, 8), "valid": valid}


def _model(mode="film"):
    return ContentGatedDynamicPredictor(audio_dim=8, content_dim=8,
        hidden_dim=12, bottleneck_dim=4, mode=mode)


def test_film_initially_matches_audio_baseline_and_receives_gradients():
    query = _query()
    torch.manual_seed(9)
    audio = _model("audio_only")
    torch.manual_seed(9)
    film = _model()
    a, _ = field_prediction(audio, query, 4)
    b, _ = field_prediction(film, query, 4)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    b.square().mean().backward()
    assert film.condition[-1].weight.grad.abs().sum() > 0
    opt = torch.optim.SGD(film.parameters(), lr=.1)
    opt.step()
    p, _ = field_prediction(film, query, 4)
    q, _ = field_prediction(film, intervene_content(query, "reverse"), 4)
    assert (p - q).abs().max() > 1e-6


def test_invalid_nan_padding_cannot_change_observed_prediction():
    query = _query()
    model = _model()
    original = model(**query)
    corrupt = {**query, "audio": query["audio"].clone(), "content": query["content"].clone()}
    corrupt["audio"][~query["valid"]] = float("nan")
    corrupt["content"][~query["valid"]] = float("nan")
    actual = model(**corrupt)
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    assert torch.equal(actual[~query["valid"]], torch.zeros_like(actual[~query["valid"]]))
    # Appending padding must not change per-frame normalization.
    extended = {key: torch.nn.functional.pad(value, (0, 0, 0, 5))
                for key, value in query.items() if key != "valid"}
    extended["valid"] = torch.nn.functional.pad(query["valid"], (0, 5))
    torch.testing.assert_close(model(**extended)[:, :13], original, rtol=1e-5, atol=1e-6)


def test_training_statistics_ignore_heldout_and_padding():
    query = _query()
    train = torch.tensor([0, 1])
    normalized, stats = normalize_inputs(query, train)
    changed = {**query, "audio": query["audio"].clone(), "content": query["content"].clone()}
    for key in ("audio", "content"):
        changed[key][2] = 1e6
        changed[key][~query["valid"]] = float("nan")
    _, other = normalize_inputs(changed, train)
    for key in ("audio", "content"):
        torch.testing.assert_close(stats[key]["mean"], other[key]["mean"])
        torch.testing.assert_close(stats[key]["std"], other[key]["std"])
        selected = normalized[key][train][query["valid"][train]]
        torch.testing.assert_close(selected.mean(0), torch.zeros(8), atol=1e-6, rtol=0)


def test_interventions_preserve_mask_and_clip_centered_contract():
    query = _query()
    for mode in ("zero", "reverse", "shuffle"):
        changed = intervene_content(query, mode)
        assert changed["valid"] is query["valid"]
        assert torch.isfinite(changed["content"]).all()
        assert not changed["content"][~query["valid"]].any()
    for mode in ("audio_only", "film", "content_only"):
        pred, weight = field_prediction(_model(mode), query, 4)
        torch.testing.assert_close((pred * weight[..., None]).sum(1), torch.zeros(3, 1),
                                   atol=1e-5, rtol=0)
        if mode == "audio_only":
            model = _model(mode)
            a, _ = field_prediction(model, query, 4)
            b, _ = field_prediction(model, intervene_content(query, "reverse"), 4)
            torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_linear_readout_keeps_dynamic_gradient_despite_large_static_logit_bias():
    class Logits(torch.nn.Module):
        def __init__(self, offset):
            super().__init__()
            self.amplitude = torch.nn.Parameter(torch.tensor(.1, dtype=torch.float64))
            self.offset = offset

        def forward(self, audio, content, valid):
            return self.offset + self.amplitude * audio

    valid = torch.ones(1, 16, dtype=torch.bool)
    audio = torch.linspace(-1, 1, 16, dtype=torch.float64).reshape(1, 16, 1)
    query = {"audio": audio, "content": audio, "valid": valid}
    target = audio.reshape(1, 4, 4, 1).mean(2)
    predictions, grads = [], []
    for offset in (0., 8.):
        model = Logits(offset)
        pred, _ = field_prediction(model, query, 4, readout="linear")
        (pred - target).square().mean().backward()
        predictions.append(pred)
        grads.append(model.amplitude.grad.clone())
    torch.testing.assert_close(predictions[0], predictions[1], atol=1e-12, rtol=0)
    torch.testing.assert_close(grads[0], grads[1], atol=1e-12, rtol=0)
    assert grads[1].abs() > .1
    saturated = Logits(8.)
    old, _ = field_prediction(saturated, query, 4)
    (old - target).square().mean().backward()
    assert saturated.amplitude.grad.abs() < grads[1].abs() * 1e-4


def test_dynamic_audio_centering_uses_only_own_valid_audio():
    query = _query()
    centered = prepare_dynamic_audio(query, "clip_centered")
    changed = {**query, "audio": query["audio"].clone()}
    changed["audio"] += torch.arange(3)[:, None, None] * 7
    changed["audio"][~query["valid"]] = float("nan")
    changed = prepare_dynamic_audio(changed, "clip_centered")
    torch.testing.assert_close(changed["audio"], centered["audio"], atol=2e-6, rtol=1e-5)
    assert centered["content"] is query["content"]
    torch.testing.assert_close(centered["audio"].sum(1), torch.zeros(3, 8), atol=2e-6, rtol=0)
