import copy

import torch

from scripts.probe_audio_text_intensity_receiver import receiver_affect, training_loss
from scripts.train_projection_schedule_ablation import draws
from tests.test_audio_text_training import training_fixture, activate


def test_frame_static_predicted_and_plus025_are_explicit_masked_drives():
    _, encoder, batch, data, _ = training_fixture(); activate(encoder)
    valid = batch["q"]["valid"]
    data["target_intensity"] = torch.arange(8.).reshape(1, 8, 1).expand(2, -1, -1).clone()
    data["target_intensity"][~data["intensity_valid"]] = float("nan")
    _, frame = receiver_affect(encoder, batch, data, "frame_gt")
    _, static = receiver_affect(encoder, batch, data, "static_gt")
    _, predicted = receiver_affect(encoder, batch, data, "predicted")
    _, higher = receiver_affect(encoder, batch, data, "frame_gt", delta=.25)
    torch.testing.assert_close(frame["driving_intensity"][valid], data["target_intensity"][valid])
    for row in range(2):
        values = static["driving_intensity"][row, valid[row]]
        torch.testing.assert_close(values, torch.full_like(values, data["target_intensity"][row, valid[row]].mean()))
    torch.testing.assert_close(predicted["driving_intensity"], predicted["predicted_intensity"], rtol=0, atol=0)
    torch.testing.assert_close(higher["driving_intensity"][valid] - frame["driving_intensity"][valid], torch.full_like(frame["driving_intensity"][valid], .25))
    assert not higher["driving_intensity"][~valid].any()
    for out in (static, predicted, higher):
        torch.testing.assert_close(frame["predicted_intensity"], out["predicted_intensity"], rtol=0, atol=0)


def test_matched_oracle_arms_start_equal_keep_intensity_head_supervised_and_finite():
    prior = torch.get_num_threads(); torch.set_num_threads(1)
    try:
        source, encoder, batch, data, scales = training_fixture()
        data["target_intensity"] = torch.where(data["intensity_valid"], torch.arange(8.).reshape(1, 8, 1), 0.)
        values, rng = {}, []
        for arm in ("oracle_frame", "oracle_static"):
            model, enc = copy.deepcopy(source), copy.deepcopy(encoder)
            generator = torch.Generator().manual_seed(46)
            noise, time, _ = draws(generator, 2, (8, 52))
            loss, terms = training_loss(model, enc, batch, data, scales, noise, time, arm)
            values[arm] = {key: value.detach() for key, value in terms.items()}
            loss.backward()
            assert enc.intensity_head[-1].weight.grad.abs().sum() > 0
            assert enc.fusion[-1].weight.grad.abs().sum() > 0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in enc.parameters())
            rng.append(generator.get_state())
        assert torch.equal(*rng)
        for key in values["oracle_frame"]:
            torch.testing.assert_close(values["oracle_frame"][key], values["oracle_static"][key], rtol=0, atol=0)
    finally:
        torch.set_num_threads(prior)
