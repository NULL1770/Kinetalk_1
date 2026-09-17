import pytest
import torch

from scripts.diagnose_projection_flow_time import error_sums, select_metadata_ids


def test_metadata_selection_is_balanced_deterministic_and_order_independent():
    q = {"clip_id": [f"mead_M003_{e}_{i}" for e in (0, 1, 5, 6) for i in range(20)],
         "emotion_id": torch.tensor([e for e in (0, 1, 5, 6) for _ in range(20)])}
    ids = select_metadata_ids(q)
    assert len(ids) == 64
    assert torch.bincount(q["emotion_id"][ids], minlength=7)[[0, 1, 5, 6]].tolist() == [16] * 4
    order = torch.arange(79, -1, -1)
    shuffled = {"clip_id": [q["clip_id"][i] for i in order], "emotion_id": q["emotion_id"][order]}
    assert [q["clip_id"][i] for i in ids] == [shuffled["clip_id"][i] for i in select_metadata_ids(shuffled)]


def test_flow_error_sums_ignore_invalid_padding_and_unobserved_channels():
    prediction = torch.tensor([[[2., 9.], [float("nan"), float("nan")]]])
    target = torch.zeros_like(prediction)
    result = error_sums(prediction, target, torch.tensor([[True, False]]), torch.tensor([[True, False]]), [0, 1])
    assert result == (4., 1)
    with pytest.raises(ValueError, match="Nonfinite"):
        error_sums(prediction, target, torch.tensor([[True, True]]), torch.tensor([[True, False]]), [0, 1])
