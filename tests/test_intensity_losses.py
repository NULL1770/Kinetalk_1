import pytest
import torch

from kinetalk_b0.intensity_losses import (
    build_intensity_triplets,
    intensity_group_audit,
    masked_rms,
    ordinal_intensity_loss,
)


def test_triplets_use_same_content_and_ignore_neutral():
    triplets = build_intensity_triplets(
        ["a"] * 4 + ["b"] * 2,
        ["s"] * 4 + ["s"] * 2,
        [5] * 4 + [5] * 2,
        [0, 1, 2, 3, 1, 2],
    )
    assert triplets.tolist() == [[1, 2, 3]]


def test_duplicate_level_is_rejected_and_audit_counts_missing_groups():
    with pytest.raises(ValueError, match="duplicate intensity"):
        build_intensity_triplets(["a", "a"], ["s", "s"], [1, 1], [1, 1])
    audit = intensity_group_audit(["a", "b"], ["s", "s"], [1, 1], [1, 2])
    assert audit["groups"] == 2
    assert audit["complete_groups"] == 0
    assert audit["incomplete_groups"] == 2


def test_masked_rms_and_quality_aware_ordinal_loss():
    values = torch.tensor([[[3.0, 0.0], [0.0, 4.0]], [[1.0, 0.0], [0.0, 0.0]]])
    valid = torch.tensor([[True, True], [True, False]])
    channels = torch.tensor([[True, True], [True, True]])
    assert torch.allclose(masked_rms(values, valid, channels), torch.tensor([2.5, 2**-0.5]))
    predicted = torch.tensor([1.0, 0.5, 2.0], requires_grad=True)
    target = torch.tensor([1.0, 1.5, 2.0])
    loss, audit = ordinal_intensity_loss(predicted, torch.tensor([[0, 1, 2]]), target_energy=target)
    assert audit == {"candidate_pairs": 2, "used_pairs": 2}
    assert loss.item() == pytest.approx(0.25)
    loss.backward()
    assert torch.isfinite(predicted.grad).all()
