import pytest
import torch
from scripts.audit_v5_personal_direction import descriptor, select_pairs


def matching_data():
    n = 9
    data = dict(motion=torch.zeros(n, 4, 2), valid=torch.ones(n, 4, dtype=torch.bool),
        dataset_id=torch.zeros(n, dtype=torch.long), speaker_id=torch.arange(n),
        emotion_id=torch.ones(n, dtype=torch.long), emotion_valid=torch.ones(n, dtype=torch.bool),
        intensity=torch.ones(n), intensity_valid=torch.ones(n, dtype=torch.bool))
    records = [dict(row=i, sentence_id='s1') for i in range(n)]
    return records, data


def test_personal_direction_requires_known_matching_content_emotion_intensity_and_different_person():
    records, data = matching_data()
    data['speaker_id'][2] = 0
    records[3]['sentence_id'] = 's2'
    data['emotion_id'][4] = 2
    data['intensity'][5] = 2
    data['intensity_valid'][6] = False
    data['dataset_id'][7] = 1  # HDTF must be excluded even with an erroneous label.
    data['emotion_valid'][8] = False
    pairs = select_pairs(records, data)
    assert (0, 1) in pairs and (1, 0) in pairs
    assert [j for i, j in pairs if i == 0] == [1]
    assert not any(i in (6, 7, 8) or j in (6, 7, 8) for i, j in pairs)


def test_personal_descriptor_uses_only_valid_observations_and_checks_empty_clips():
    y = torch.tensor([[[1., 2.], [3., 4.], [999., 999.]]])
    valid = torch.tensor([[True, True, False]])
    d = descriptor(y, valid, torch.ones(2))
    assert torch.equal(d[:, 0], torch.tensor([[2., 3.]]))
    y[:, 2] = -999.
    assert torch.equal(descriptor(y, valid, torch.ones(2)), d)
    with pytest.raises(ValueError, match='observed frames'):
        descriptor(y, torch.zeros_like(valid), torch.ones(2))
