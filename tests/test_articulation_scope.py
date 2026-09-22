import pytest
import torch

from scripts.train_full_staged import articulation_selection, parser


def query(labels):
    return {'emotion_id': torch.tensor(labels, dtype=torch.long)}


def test_legacy_scope_is_neutral_only_and_long():
    ids, info = articulation_selection(query([0, 1, 0, 7, 2]), 'neutral')
    assert ids.dtype == torch.long
    assert ids.tolist() == [0, 2]
    assert info['actual_scope'] == 'neutral'
    assert info['emotion_counts'] == {'0': 2}
    assert info['all_train_clips_included'] is False


def test_all_emotions_scope_includes_every_train_row_and_reports_counts():
    labels = [0, 1, 0, 7, 2, 1]
    ids, info = articulation_selection(query(labels), 'all-emotions')
    assert ids.tolist() == list(range(len(labels)))
    assert info['all_train_clips_included'] is True
    assert info['emotion_counts'] == {'0': 2, '1': 2, '2': 1, '7': 1}
    assert info['train_emotion_ids'] == [0, 1, 2, 7]


def test_invalid_or_empty_scope_fails_closed():
    with pytest.raises(ValueError): articulation_selection(query([0, 1]), 'all')
    with pytest.raises(ValueError): articulation_selection(query([1, 2]), 'neutral')
    with pytest.raises(ValueError): articulation_selection({'emotion_id': torch.tensor([[0]])}, 'neutral')


def test_cli_default_is_legacy_neutral_and_all_emotions_is_explicit():
    default = parser().parse_args(['--output', 'run'])
    assert default.articulation_scope == 'neutral'
    full = parser().parse_args(['--output', 'run', '--articulation-scope', 'all-emotions'])
    assert full.articulation_scope == 'all-emotions'
