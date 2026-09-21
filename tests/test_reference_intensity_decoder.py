import pytest
import torch

from kinetalk_b0.models.reference_intensity_decoder import (
    ReferenceIntensityDecoder,
    ReferenceIntensityStudent,
    neutral_relative_intensity,
)
from kinetalk_b0.models.slow_state_affect import UPPER_INDICES, compose_upper_face


def example(frames=17, batch=2):
    torch.manual_seed(71)
    model = ReferenceIntensityDecoder(global_dim=4, identity_dim=6, hidden=12).double()
    intensity = torch.rand(batch, frames, 2, dtype=torch.float64)
    valid = torch.ones(batch, frames, dtype=torch.bool)
    global_code = torch.randn(batch, 4, dtype=torch.float64)
    identity_code = torch.randn(batch, 6, dtype=torch.float64)
    anchor = torch.full((batch, 9), .25, dtype=torch.float64)
    return model, intensity, valid, global_code, identity_code, anchor


def test_native_bounded_output_and_exported_config():
    model, intensity, valid, global_code, identity_code, anchor = example()
    result = model(intensity, valid, global_code, identity_code, anchor)
    assert result.shape == (2, 17, 9)
    assert torch.isfinite(result).all()
    assert ((result > 0) & (result < 1)).all()
    assert model.export_config() == {"global_dim": 4, "identity_dim": 6, "hidden": 12}
    restored = ReferenceIntensityDecoder(**model.export_config()).double()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(intensity, valid, global_code, identity_code, anchor), result)


@pytest.mark.parametrize("frames", [1, 2, 17])
def test_static_intensity_produces_constant_output_including_boundaries(frames):
    model, intensity, valid, global_code, identity_code, anchor = example(frames)
    intensity[:] = .75
    result = model(intensity, valid, global_code, identity_code, anchor)
    torch.testing.assert_close(result, result[:, :1].expand_as(result), atol=1e-14, rtol=1e-14)


def test_intensity_pulse_creates_new_event_without_prior_trajectory():
    model, intensity, valid, global_code, identity_code, anchor = example(frames=25, batch=1)
    intensity[:] = .25
    static = model(intensity, valid, global_code, identity_code, anchor)
    pulse = intensity.clone()
    pulse[:, 10:15, 0] = 2.
    generated = model(pulse, valid, global_code, identity_code, anchor)
    assert (generated[:, 10:15] - static[:, 10:15]).abs().max() > 1e-6
    assert generated.std(dim=1).max() > 1e-6
    # Total receptive field is seven frames. There is no old motion carrier.
    torch.testing.assert_close(generated[:, :6], static[:, :6], atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(generated[:, 19:], static[:, 19:], atol=1e-14, rtol=1e-14)


def test_reference_and_static_codes_can_change_learned_execution():
    model, intensity, valid, global_code, identity_code, anchor = example()
    original = model(intensity, valid, global_code, identity_code, anchor)
    for context in (
        (global_code + .7, identity_code, anchor),
        (global_code, identity_code - .7, anchor),
        (global_code, identity_code, anchor + .1),
    ):
        changed = model(intensity, valid, *context)
        assert (changed - original).abs().max() > 1e-6
    full_anchor = torch.randn(2, 52, dtype=torch.float64)
    full_anchor[:, list(UPPER_INDICES)] = anchor
    torch.testing.assert_close(model(intensity, valid, global_code, identity_code, full_anchor), original)


def test_run_isolation_and_poisoned_padding_forward_and_backward():
    model, intensity, valid, global_code, identity_code, anchor = example()
    valid[0, 4:7] = False
    valid[1, 11:] = False
    intensity[~valid] = float("nan")
    intensity.requires_grad_()
    result = model(intensity, valid, global_code, identity_code, anchor)
    for row, left, right in ((0, 0, 4), (0, 7, 17), (1, 0, 11)):
        independent = model(intensity[row:row+1, left:right],
                            torch.ones(1, right-left, dtype=torch.bool),
                            global_code[row:row+1], identity_code[row:row+1], anchor[row:row+1])
        torch.testing.assert_close(result[row, left:right], independent[0], atol=1e-13, rtol=1e-13)
    padded = torch.cat((intensity, torch.full((2, 8, 2), float("inf"), dtype=torch.float64)), 1)
    padded_mask = torch.cat((valid, torch.zeros(2, 8, dtype=torch.bool)), 1)
    padded_result = model(padded, padded_mask, global_code, identity_code, anchor)
    torch.testing.assert_close(result[valid], padded_result[:, :17][valid], atol=1e-13, rtol=1e-13)
    result[valid].square().mean().backward()
    assert torch.isfinite(intensity.grad).all()
    assert intensity.grad[valid].abs().sum() > 1e-8
    assert torch.equal(intensity.grad[~valid], torch.zeros_like(intensity.grad[~valid]))
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.input.weight.grad.abs().sum() > 0


def test_composition_preserves_other43_and_invalid_bits():
    model, intensity, valid, global_code, identity_code, anchor = example()
    valid[0, 4:7] = False
    valid[1, 11:] = False
    baseline = torch.randn(2, 17, 52, dtype=torch.float64)
    baseline[~valid] = float("nan")
    upper = model(intensity, valid, global_code, identity_code, anchor)
    composed = compose_upper_face(baseline, upper, valid)
    others = [i for i in range(52) if i not in UPPER_INDICES]
    assert torch.equal(composed[..., others].view(torch.int64), baseline[..., others].view(torch.int64))
    assert torch.equal(composed[~valid].view(torch.int64), baseline[~valid].view(torch.int64))
    torch.testing.assert_close(composed[..., list(UPPER_INDICES)][valid], upper[valid])


def test_intensity_measures_neutral_relative_amplitude_not_clip_activity():
    anchor = torch.full((1, 9), .2)
    scales = torch.tensor([.1] * 5 + [.2] * 4)
    upper = anchor[:, None].expand(1, 13, 9).clone()
    upper[..., :5] += .2
    upper[..., 5:] -= .1
    valid = torch.ones(1, 13, dtype=torch.bool)
    expected = torch.tensor([2., .5]).expand(1, 13, 2)
    for window in (1, 5):
        result = neutral_relative_intensity(upper, anchor, scales, valid, window)
        torch.testing.assert_close(result, expected)
    cropped = neutral_relative_intensity(upper[:, 3:8], anchor, scales, valid[:, 3:8])
    torch.testing.assert_close(cropped, expected[:, 3:8])
    zeros = neutral_relative_intensity(anchor[:, None].expand_as(upper), anchor, scales, valid)
    assert torch.equal(zeros, torch.zeros_like(zeros))


def test_intensity_signed_ramp_keeps_expression_order_and_local_smoothing():
    anchor = torch.zeros(1, 9)
    upper = torch.arange(11, dtype=torch.float32)[None, :, None].expand(1, 11, 9) / 10
    valid = torch.ones(1, 11, dtype=torch.bool)
    result = neutral_relative_intensity(upper, anchor, torch.ones(9), valid)
    assert (result[:, 1:] > result[:, :-1]).all()
    torch.testing.assert_close(result[0, 0], torch.full((2,), .06))
    torch.testing.assert_close(result[0, -1], torch.full((2,), .94))


def test_intensity_gap_padding_and_output_loss_gradients():
    torch.manual_seed(72)
    upper = torch.rand(2, 13, 9, dtype=torch.float64)
    valid = torch.ones(2, 13, dtype=torch.bool)
    valid[0, 3:6] = False
    valid[1, 10:] = False
    upper[~valid] = float("nan")
    upper.requires_grad_()
    anchor = torch.full((2, 9), .2, dtype=torch.float64)
    scales = torch.linspace(.1, .2, 9, dtype=torch.float64)
    result = neutral_relative_intensity(upper, anchor, scales, valid)
    for row, left, right in ((0, 0, 3), (0, 6, 13), (1, 0, 10)):
        independent = neutral_relative_intensity(upper[row:row+1, left:right], anchor[row:row+1], scales,
                                                  torch.ones(1, right-left, dtype=torch.bool))
        torch.testing.assert_close(result[row, left:right], independent[0], atol=1e-13, rtol=1e-13)
    batch_scales = neutral_relative_intensity(upper, anchor, scales.expand(2, 9), valid)
    torch.testing.assert_close(result, batch_scales)
    assert torch.equal(result[~valid], torch.zeros_like(result[~valid]))
    padded = torch.cat((upper, torch.full((2, 7, 9), float("inf"), dtype=torch.float64)), 1)
    padded_valid = torch.cat((valid, torch.zeros(2, 7, dtype=torch.bool)), 1)
    extended = neutral_relative_intensity(padded, anchor, scales, padded_valid)
    torch.testing.assert_close(result[valid], extended[:, :13][valid], atol=1e-13, rtol=1e-13)
    result[valid].square().mean().backward()
    assert torch.isfinite(upper.grad).all()
    assert upper.grad[valid].abs().sum() > 0
    assert torch.equal(upper.grad[~valid], torch.zeros_like(upper.grad[~valid]))


def test_final_output_intensity_loss_reaches_decoder_input():
    model, intensity, valid, global_code, identity_code, anchor = example()
    intensity.requires_grad_()
    upper = model(intensity, valid, global_code, identity_code, anchor)
    generated_intensity = neutral_relative_intensity(upper, anchor, torch.full((9,), .1, dtype=torch.float64), valid)
    (generated_intensity - .8).square().mean().backward()
    assert torch.isfinite(intensity.grad).all()
    assert intensity.grad.abs().sum() > 1e-8


@pytest.mark.parametrize("failure", ["negative", "nonfinite", "mask", "global", "anchor"])
def test_decoder_rejects_invalid_observed_inputs(failure):
    model, intensity, valid, global_code, identity_code, anchor = example()
    if failure == "negative":
        intensity[0, 0, 0] = -.1
    elif failure == "nonfinite":
        intensity[0, 0, 0] = float("nan")
    elif failure == "mask":
        valid = valid.float()
    elif failure == "global":
        global_code = global_code[:, :3]
    elif failure == "anchor":
        anchor = anchor[:, None]
    with pytest.raises(ValueError):
        model(intensity, valid, global_code, identity_code, anchor)


@pytest.mark.parametrize("failure", ["scales", "window", "nonfinite", "anchor"])
def test_intensity_rejects_invalid_inputs(failure):
    upper = torch.rand(1, 7, 9)
    anchor = torch.zeros(1, 9)
    scales = torch.ones(9)
    valid = torch.ones(1, 7, dtype=torch.bool)
    window = 5
    if failure == "scales":
        scales[0] = 0
    elif failure == "window":
        window = 4
    elif failure == "nonfinite":
        upper[0, 0, 0] = float("inf")
    elif failure == "anchor":
        anchor[0, 0] = float("nan")
    with pytest.raises(ValueError):
        neutral_relative_intensity(upper, anchor, scales, valid, window)


def student_example(frames=19):
    torch.manual_seed(73)
    model = ReferenceIntensityStudent(input_dim=7, global_dim=4, identity_dim=6, hidden=12).double()
    features = torch.randn(2, frames, 7, dtype=torch.float64)
    valid = torch.ones(2, frames, dtype=torch.bool)
    global_code = torch.randn(2, 4, dtype=torch.float64)
    identity_code = torch.randn(2, 6, dtype=torch.float64)
    anchor = torch.full((2, 9), .25, dtype=torch.float64)
    return model, features, valid, global_code, identity_code, anchor


def test_student_native_nonnegative_outputs_and_export_config():
    model, features, valid, global_code, identity_code, anchor = student_example()
    result = model(features, valid, global_code, identity_code, anchor)
    assert result["intensity"].shape == (2, 19, 2)
    assert result["local"].shape == (2, 19, 12)
    assert torch.isfinite(result["intensity"]).all()
    assert (result["intensity"] > 0).all()
    assert model.export_config() == {"input_dim": 7, "global_dim": 4, "identity_dim": 6, "hidden": 12}
    restored = ReferenceIntensityStudent(**model.export_config()).double()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(features, valid, global_code, identity_code, anchor)["intensity"],
                               result["intensity"])
    full_anchor = torch.randn(2, 52, dtype=torch.float64)
    full_anchor[:, list(UPPER_INDICES)] = anchor
    torch.testing.assert_close(model(features, valid, global_code, identity_code, full_anchor)["intensity"],
                               result["intensity"])


@pytest.mark.parametrize("frames", [1, 2, 19])
def test_student_constant_audio_does_not_invent_boundary_motion(frames):
    model, features, valid, global_code, identity_code, anchor = student_example(frames)
    features[:] = .7
    result = model(features, valid, global_code, identity_code, anchor)
    for value in result.values():
        torch.testing.assert_close(value, value[:, :1].expand_as(value), atol=1e-13, rtol=1e-13)


def test_student_static_mode_has_no_time_variation_with_gaps():
    model, features, valid, global_code, identity_code, anchor = student_example()
    valid[0, 5:9] = False
    valid[1, 14:] = False
    features[~valid] = float("nan")
    full = model(features, valid, global_code, identity_code, anchor)
    static = model(features, valid, global_code, identity_code, anchor, temporal_mode="static")
    for row in range(2):
        observed = static["intensity"][row, valid[row]]
        torch.testing.assert_close(observed, observed[:1].expand_as(observed), atol=1e-14, rtol=1e-14)
        expected_local = full["local"][row, valid[row]].mean(0)
        torch.testing.assert_close(static["local"][row, valid[row]], expected_local.expand(valid[row].sum(), 12))
        assert full["intensity"][row, valid[row]].std(0).max() > 1e-5
    for value in static.values():
        assert torch.equal(value[~valid], torch.zeros_like(value[~valid]))


def test_student_temporal_runs_isolated_but_static_context_is_shared():
    model, features, valid, global_code, identity_code, anchor = student_example()
    valid[0, 5:9] = False
    valid[1, 14:] = False
    features[~valid] = float("nan")
    result = model(features, valid, global_code, identity_code, anchor)
    for row, left, right in ((0, 0, 5), (0, 9, 19), (1, 0, 14)):
        independent = model(features[row:row+1, left:right],
                            torch.ones(1, right-left, dtype=torch.bool),
                            global_code[row:row+1], identity_code[row:row+1], anchor[row:row+1])
        # Full intensity may differ: each independent invocation has a new clip mean.
        torch.testing.assert_close(result["local"][row, left:right], independent["local"][0],
                                   atol=1e-13, rtol=1e-13)


@pytest.mark.parametrize("mode", ["full", "static"])
def test_student_padding_and_finite_useful_gradients(mode):
    model, features, valid, global_code, identity_code, anchor = student_example()
    valid[0, 5:9] = False
    valid[1, 14:] = False
    features[~valid] = float("nan")
    features.requires_grad_()
    result = model(features, valid, global_code, identity_code, anchor, temporal_mode=mode)
    padded = torch.cat((features, torch.full((2, 6, 7), float("inf"), dtype=torch.float64)), 1)
    padded_valid = torch.cat((valid, torch.zeros(2, 6, dtype=torch.bool)), 1)
    padded_result = model(padded, padded_valid, global_code, identity_code, anchor, temporal_mode=mode)
    for key in result:
        torch.testing.assert_close(result[key][valid], padded_result[key][:, :19][valid], atol=1e-13, rtol=1e-13)
    result["intensity"][valid].square().mean().backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad[valid].abs().sum() > 1e-8
    assert torch.equal(features.grad[~valid], torch.zeros_like(features.grad[~valid]))
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    assert model.input.weight.grad.abs().sum() > 0
    assert model.context[0].weight.grad.abs().sum() > 0


def test_student_and_decoder_final_output_loss_reaches_audio():
    student, features, valid, global_code, identity_code, anchor = student_example()
    features.requires_grad_()
    decoder = ReferenceIntensityDecoder(global_dim=4, identity_dim=6, hidden=12).double()
    intensity = student(features, valid, global_code, identity_code, anchor)["intensity"]
    upper = decoder(intensity, valid, global_code, identity_code, anchor)
    output_intensity = neutral_relative_intensity(upper, anchor, torch.full((9,), .1, dtype=torch.float64), valid)
    loss = (upper - .4).square().mean() + (output_intensity - .8).square().mean()
    loss.backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 1e-9
    assert student.input.weight.grad.abs().sum() > 0


def test_student_rejects_bad_mode_and_observed_nonfinite_features():
    model, features, valid, global_code, identity_code, anchor = student_example()
    with pytest.raises(ValueError, match="temporal_mode"):
        model(features, valid, global_code, identity_code, anchor, temporal_mode="reverse")
    features[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model(features, valid, global_code, identity_code, anchor)
