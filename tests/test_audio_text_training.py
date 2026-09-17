import copy

import pytest
import torch

from kinetalk_b0.label_guided_intensity import regional_intensity
from kinetalk_b0.models.audio_text_affect import AudioTextAffect
from scripts.train_audio_text_dynamics import (
    GROUPS, LOSS_WEIGHTS, build_affect, generated_losses, training_loss,
)
from scripts.train_direct_audio_dynamics import configure, protected_hash
from scripts.train_projection_schedule_ablation import draws
from scripts.train_predictable_renderer import observed, state_hash
from tests.test_audio_conditioned_flow_probe import fixture


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def training_fixture():
    system, batch, _, _ = fixture()
    batch["q"]["motion"] = torch.full_like(batch["q"]["motion"], .2)
    batch["q"]["emotion_id"] = torch.tensor([1, 5])
    batch["q"]["intensity_id"] = torch.tensor([3, 1])
    scales = torch.full((52,), .2)
    anchors, anchor_valid = torch.zeros(2, 52), torch.ones(2, 52, dtype=torch.bool)
    intensity, valid = regional_intensity(batch["q"]["motion"], observed(batch["q"]), anchors, scales)
    data = {"features": torch.randn(2, 8, 9), "tokens": torch.randn(2, 5, 10),
        "token_valid": torch.tensor([[True, True, True, False, False], [False] * 5]),
        "target_intensity": intensity, "intensity_valid": valid, "anchors": anchors, "anchor_valid": anchor_valid}
    encoder = AudioTextAffect(torch.zeros(9), torch.ones(9), text_dim=10, global_dim=64,
                              hidden=16, local_dim=64, initial_intensity=.8).eval()
    return system, encoder, batch, data, scales


def activate(encoder):
    with torch.no_grad():
        encoder.fusion[-1].weight.normal_(std=.1)
        encoder.intensity_head[-1].weight.normal_(std=.1)


def test_training_conditions_ignore_gt_motion_labels_and_intensity_except_explicit_oracle():
    _, encoder, batch, data, _ = training_fixture(); activate(encoder)
    first, predicted = build_affect(encoder, batch, data, "text")
    altered_batch, altered_data = copy.deepcopy(batch), copy.deepcopy(data)
    altered_batch["q"]["motion"].fill_(999.)
    altered_batch["q"]["emotion_id"].fill_(-99)
    altered_batch["q"]["intensity_id"].fill_(-99)
    altered_data["target_intensity"].fill_(999.)
    altered_data["anchors"].fill_(999.)
    second, again = build_affect(encoder, altered_batch, altered_data, "text")
    for key in first:
        if torch.is_tensor(first[key]): torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    torch.testing.assert_close(predicted["driving_intensity"], predicted["predicted_intensity"], rtol=0, atol=0)
    oracle, output = build_affect(encoder, batch, altered_data, "text", "oracle_intensity")
    assert not torch.allclose(first["local"], oracle["local"])
    assert (output["driving_intensity"][data["intensity_valid"]] == 999).all()
    torch.testing.assert_close(again["predicted_intensity"], predicted["predicted_intensity"], rtol=0, atol=0)


def test_raw_reconstruction_penalizes_persistent_offset_and_equal_weights_regions():
    _, _, batch, data, scales = training_fixture()
    target = batch["q"]["motion"]
    exact = generated_losses(target, batch, data, scales)
    assert all(loss == 0 for loss in exact.values())
    prediction = target.clone()
    prediction[..., GROUPS["brows"]] += .2
    losses = generated_losses(prediction, batch, data, scales)
    # Constant normalized error one has Huber .5. Only one of three equally
    # weighted regions changes, despite their different channel counts.
    assert losses["raw_motion"] == pytest.approx(.5 / 3, abs=1e-7)
    assert losses["output_intensity"] > 0 and losses["domain"] == 0
    torch.testing.assert_close(prediction.std(1), target.std(1), atol=0, rtol=0)


def test_output_losses_mask_nan_padding_missing_channels_and_keep_domain_unclamped():
    _, _, batch, data, scales = training_fixture()
    mask = observed(batch["q"])
    pred = batch["q"]["motion"].clone(); pred[:] = 1.2
    original = generated_losses(pred, batch, data, scales)
    assert original["domain"] == pytest.approx(10., abs=1e-5)
    batch["q"]["motion"][~mask] = float("nan")
    pred[~mask] = float("nan")
    pred.requires_grad_(True)
    actual = generated_losses(pred, batch, data, scales)
    for name in original: torch.testing.assert_close(actual[name], original[name], rtol=0, atol=0)
    sum(actual.values()).backward()
    assert torch.isfinite(pred.grad).all() and not pred.grad[~mask].any()


def test_matched_zero_start_losses_noise_rng_and_frozen_paths_survive_training():
    source, encoder, batch, data, scales = training_fixture()
    initial_states, renderer_gradients, outputs, noise_states = {}, {}, {}, []
    for arm in ("text", "no_text"):
        model, enc = copy.deepcopy(source), copy.deepcopy(encoder)
        parameters = configure(model) + list(enc.parameters())
        initial_states[arm] = (state_hash(model.state_dict()), state_hash(enc.state_dict()))
        frozen = protected_hash(model)
        generator = torch.Generator().manual_seed(46)
        noise, time, choice = draws(generator, 2, (8, 52))
        before_generator = generator.get_state().clone()
        total, losses = training_loss(model, enc, batch, data, scales, noise, time, arm)
        noise_states.append((noise, time, choice, generator.get_state()))
        assert torch.equal(generator.get_state(), before_generator)
        assert set(losses) == {"flow", *LOSS_WEIGHTS}
        outputs[arm] = {name: value.detach() for name, value in losses.items()}
        total.backward()
        assert enc.fusion[-1].weight.grad.abs().sum() > 0
        assert enc.intensity_head[-1].weight.grad.abs().sum() > 0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
        renderer_gradients[arm] = {name: p.grad.detach().clone() for name, p in model.renderer.named_parameters()
                                   if p.grad is not None}
        if arm == "no_text":
            assert all(p.grad is None for name, p in enc.named_parameters()
                       if name.startswith(("cross_attention", "text_")))
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        torch.optim.Adam(parameters, lr=1e-5).step()
        assert protected_hash(model) == frozen
    for name in outputs["text"]:
        torch.testing.assert_close(outputs["text"][name], outputs["no_text"][name], rtol=0, atol=0)
    assert all(torch.equal(a, b) for a, b in zip(*noise_states))
    assert initial_states["text"] == initial_states["no_text"]
    for name in renderer_gradients["text"]:
        torch.testing.assert_close(renderer_gradients["text"][name], renderer_gradients["no_text"][name], rtol=0, atol=0)
    # Text/no-text have different encoder gradients; the shared global norm
    # clip may therefore make their first optimizer updates differ legitimately.


def test_training_rollout_receives_shared_noise_cached_base_and_predicted_conditions(monkeypatch):
    system, encoder, batch, data, scales = training_fixture(); activate(encoder)
    noise, time, _ = draws(torch.Generator().manual_seed(46), 2, (8, 52))
    calls, original_generate = [], system.generate
    def check_generate(content, valid, identity, affect, **kwargs):
        assert kwargs["initial_noise"] is noise and kwargs["base"] is batch["base"]
        assert kwargs["steps"] == 12
        calls.append(affect["local"].detach().clone())
        return original_generate(content, valid, identity, affect, **kwargs)
    monkeypatch.setattr(system, "generate", check_generate)
    first, a = training_loss(system, encoder, batch, data, scales, noise, time, "text")
    changed = {**data, "target_intensity": data["target_intensity"] + 3}
    second, b = training_loss(system, encoder, batch, changed, scales, noise, time, "text")
    torch.testing.assert_close(calls[0], calls[1], rtol=0, atol=0)
    for name in ("flow", "raw_motion", "domain"):
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    assert first != second and a["predicted_intensity"] != b["predicted_intensity"]
