import torch

from kinetalk_b0.dynamic_timing import (
    clip_rms, event_labels, event_loss, event_metrics, field_delta,
    fit_event_threshold, fit_positive_gain, normalize_target_shape,
)
from kinetalk_b0.gated_audio_dynamic import ContentGatedDynamicPredictor
from scripts.probe_audio_content_gating import field_prediction


def test_target_normalization_removes_clip_amplitude_with_training_only_floor():
    values = torch.tensor([[[-2.], [0.], [2.], [99.]],
                           [[-20.], [0.], [20.], [99.]],
                           [[-200.], [0.], [200.], [99.]]])
    weight = torch.tensor([[4., 4., 4., 0.]]).expand(3, -1)
    train = torch.tensor([0, 1])
    normalized, floor = normalize_target_shape(values, weight, train)
    assert torch.allclose(normalized[1], normalized[2])
    assert normalized[:, -1].eq(0).all()
    altered = values.clone()
    altered[2] *= 1000
    again, new_floor = normalize_target_shape(altered, weight, train)
    assert torch.equal(floor, new_floor)
    assert torch.equal(normalized[train], again[train])
    assert torch.allclose(clip_rms(normalized[1:2], weight[1:2]), torch.ones(1, 1, 1))


def test_event_labels_and_loss_exclude_missing_pairs_and_keep_direction():
    values = torch.tensor([[[0.], [2.], [2.], [0.], [999.]]])
    weight = torch.tensor([[4., 4., 4., 4., 0.]])
    threshold = torch.tensor([1.])
    labels, pair_weight = event_labels(values, weight, threshold)
    assert labels[0, :, 0].tolist() == [2, 1, 0, 1]
    assert pair_weight.tolist() == [[4., 4., 4., 0.]]
    assert event_loss(values, values, weight, threshold) < event_loss(-values, values, weight, threshold)
    assert event_metrics(values, values, weight, threshold)["macro_f1"] == 1.0
    missing = torch.tensor([[4., 0., 4., 4., 0.]])
    delta, w = field_delta(values, missing)
    assert w.tolist() == [[0., 0., 4., 0.]]
    assert delta[0, :2].eq(0).all()


def test_event_threshold_and_gain_do_not_use_heldout_targets():
    target = torch.tensor([[[-1.], [0.], [1.]], [[-100.], [0.], [100.]]])
    weight = torch.ones(2, 3)
    train = torch.tensor([0])
    threshold = fit_event_threshold(target, weight, train)
    pred = target * .5
    gain = fit_positive_gain(pred, target, weight, train)
    altered = target.clone()
    altered[1] *= 100
    assert torch.equal(threshold, fit_event_threshold(altered, weight, train))
    assert torch.equal(gain, fit_positive_gain(pred, altered, weight, train))
    assert torch.equal(gain, torch.tensor([2.]))
    assert fit_positive_gain(-pred, target, weight, train).eq(0).all()


def test_single_event_objective_backpropagates_into_film_without_new_channels():
    torch.manual_seed(7)
    model = ContentGatedDynamicPredictor(audio_dim=4, content_dim=5,
        hidden_dim=8, bottleneck_dim=3, output_dim=1, mode="film")
    valid = torch.ones(2, 16, dtype=torch.bool)
    query = {"audio": torch.randn(2, 16, 4), "content": torch.randn(2, 16, 5), "valid": valid}
    pred, weight = field_prediction(model, query, 4)
    target = torch.tensor([[[-1.], [1.], [1.], [-1.]]]).expand(2, -1, -1)
    loss = event_loss(pred, target, weight, torch.tensor([.5]))
    loss.backward()
    assert pred.shape == (2, 4, 1)
    assert torch.isfinite(loss)
    assert model.condition[-1].weight.grad.abs().sum() > 0
    assert model.output.weight.grad.abs().sum() > 0
