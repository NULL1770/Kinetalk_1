import torch

from kinetalk_b0.semantic_losses import (
    cross_style_objective,
    semantic_supervision,
    style_contrastive,
)


def targets(batch=2, frames=4):
    return {
        "valid": torch.ones(batch, frames, dtype=torch.bool),
        "va_valid": torch.ones(batch, frames, dtype=torch.bool),
        "va_confidence": torch.ones(batch, frames),
        "times": torch.arange(frames, dtype=torch.float64)[None].expand(batch, -1) * 0.04,
        "va": torch.zeros(batch, frames, 2),
        "emotion_id": torch.zeros(batch, dtype=torch.long),
        "intensity_id": torch.zeros(batch, dtype=torch.long),
        "intensity_valid": torch.ones(batch, dtype=torch.bool),
    }


def predictions(branch):
    batch = len(branch["emotion_id"])
    return {
        "emotion_logits": torch.zeros(batch, 3, requires_grad=True),
        "intensity_logits": torch.zeros(batch, 3, requires_grad=True),
        "va": torch.zeros_like(branch["va"], requires_grad=True),
    }


def test_unknown_intensity_and_invalid_va_cannot_supervise_or_create_nan():
    branch = targets()
    branch["intensity_id"][:] = -1
    branch["intensity_valid"][:] = False
    branch["va_valid"][:] = False
    branch["va"][:] = float("nan")
    output = predictions(branch)
    result = semantic_supervision(output, branch)
    result["total"].backward()
    assert result["intensity"].item() == 0
    assert result["va"].item() == 0 and result["va_delta"].item() == 0
    assert torch.isfinite(result["total"])
    assert output["intensity_logits"].grad.count_nonzero() == 0
    assert output["va"].grad.count_nonzero() == 0


def test_confidence_weights_va_without_shrinking_the_label_toward_neutral():
    branch = targets(batch=1, frames=2)
    branch["va"][0, 0] = 1.0
    branch["va_confidence"][0] = torch.tensor([0.25, 1.0])
    output = predictions(branch)
    result = semantic_supervision(output, branch)
    # smoothL1(0,1)=0.5, weighted over one noisy and one zero-error frame.
    torch.testing.assert_close(result["va"], torch.tensor(0.1))
    matched = {**output, "va": branch["va"].clone().requires_grad_()}
    assert semantic_supervision(matched, branch)["va"].item() == 0


def test_va_derivative_uses_true_time_and_never_crosses_gaps():
    fine = targets(batch=1, frames=4)
    coarse = targets(batch=1, frames=4)
    fine["times"] = torch.arange(4, dtype=torch.float64)[None] * 0.02
    coarse["times"] = torch.arange(4, dtype=torch.float64)[None] * 0.04
    fine["va"] = fine["times"].float().unsqueeze(-1).expand(-1, -1, 2)
    coarse["va"] = coarse["times"].float().unsqueeze(-1).expand(-1, -1, 2)
    fine_loss = semantic_supervision(predictions(fine), fine)
    coarse_loss = semantic_supervision(predictions(coarse), coarse)
    torch.testing.assert_close(fine_loss["va_delta"], coarse_loss["va_delta"])
    gap = targets(batch=1, frames=4)
    gap["times"] = torch.tensor([[0.0, 0.04, 0.5, 0.54]], dtype=torch.float64)
    gap["va"][:, 2:] = 1
    assert semantic_supervision(predictions(gap), gap)["va_delta"].item() == 0
    gap["times"] = targets(batch=1, frames=4)["times"]
    assert semantic_supervision(predictions(gap), gap)["va_delta"].item() > 0


def test_style_contrastive_uses_all_same_speaker_views_as_positives():
    anchor = torch.tensor([[1.0, 0], [1.0, 0], [0, 1.0]], requires_grad=True)
    positive = anchor.detach().clone().requires_grad_()
    speakers = torch.tensor([1, 1, 2])
    expected = style_contrastive(anchor, positive, speakers)
    false_negatives = style_contrastive(anchor, positive, torch.tensor([1, 2, 3]))
    # Same-person duplicates make the positive likelihood shared across three
    # legitimate views. The analytic value tests the positive mask itself.
    temperature = 0.1
    same_person = torch.log(3.0 + 2.0 * torch.exp(torch.tensor(-1 / temperature)))
    other_person = torch.log(1.0 + 4.0 * torch.exp(torch.tensor(-1 / temperature)))
    analytic = (4 * same_person + 2 * other_person) / 6
    torch.testing.assert_close(expected, analytic)
    assert torch.isfinite(false_negatives)
    expected.backward()
    assert torch.isfinite(anchor.grad).all() and torch.isfinite(positive.grad).all()


def test_style_contrastive_requires_real_negatives_and_ignores_invalid_views():
    anchor = torch.randn(2, 3, requires_grad=True)
    positive = torch.randn(2, 2, 3, requires_grad=True)
    one_person = style_contrastive(anchor, positive, torch.tensor([1, 1]))
    assert one_person.item() == 0
    one_person.backward()
    assert anchor.grad.count_nonzero() == 0
    validity = torch.tensor([[True, True, False], [True, True, False]])
    corrupted = positive.detach().clone()
    corrupted[:, 1] = float("nan")
    actual = style_contrastive(anchor, corrupted, torch.tensor([1, 2]), valid=validity)
    expected = style_contrastive(anchor, positive[:, 0], torch.tensor([1, 2]))
    torch.testing.assert_close(actual, expected)


def cross_inputs():
    branch = targets()
    base = torch.zeros(2, 4, 5)
    base[:, :, 0] = torch.arange(4) * 0.05
    base[:, :, 1] = torch.arange(4) * 0.025
    style = torch.tensor([[1.0, 0], [0, 1.0]], requires_grad=True)
    return {
        "generated_motion": base.clone().requires_grad_(),
        "base_motion": base,
        "semantic_outputs": predictions(branch),
        "target_branch": branch,
        "generated_style": style,
        "donor_style": style.detach().clone().requires_grad_(),
        "style_negatives": style.detach().clone().requires_grad_(),
        "mouth_indices": [0, 1],
        "donor_speaker_ids": torch.tensor([10, 20]),
        "negative_speaker_ids": torch.tensor([10, 20]),
    }


def test_cross_style_excludes_donor_false_negative_and_detaches_target_styles():
    values = cross_inputs()
    loss = cross_style_objective(**values)
    expected = torch.log1p(torch.exp(torch.tensor(-10.0)))
    torch.testing.assert_close(loss["style"], expected, atol=1e-7, rtol=1e-3)
    loss["total"].backward()
    assert values["generated_style"].grad.abs().sum() > 0
    assert values["donor_style"].grad is None
    assert values["style_negatives"].grad is None


def test_mouth_timing_allows_gain_and_ignores_upper_face_but_penalizes_reverse():
    values = cross_inputs()
    unchanged = cross_style_objective(**values)
    changed = values["base_motion"].clone()
    changed[..., :2] *= 3.0
    changed[..., 2:] = torch.randn_like(changed[..., 2:]) * 100
    changed.requires_grad_()
    scaled = cross_style_objective(**{**values, "generated_motion": changed})
    reversed_motion = -values["base_motion"]
    reverse = cross_style_objective(**{**values, "generated_motion": reversed_motion})
    torch.testing.assert_close(scaled["mouth_timing"], unchanged["mouth_timing"], atol=1e-7, rtol=0)
    assert reverse["mouth_timing"] > 1.99
    scaled["mouth_timing"].backward()
    assert changed.grad[..., 2:].count_nonzero() == 0
