import pytest
from scripts.audit_factor_role_inputs import approved_train_rows, emotion_name


def row(source='query', reference='neutral', **kwargs):
    return {'source_clip_id': source, 'reference_clip_id': reference,
            'teacher_eligible': True, 'teacher_artifact': 'pair.npz',
            'mouth_event_gate': False, 'event_local_mask_path': 'mask.npz', **kwargs}


def test_train_boundary_filters_other_sources_before_opening_artifacts():
    selected, rejected = approved_train_rows([row(), row('sealed', 'sealed_ref')],
                                             {'query'}, {'neutral'})
    assert selected == [row()]
    assert rejected == {'outside_train_source': 1}


def test_train_source_cannot_import_held_out_reference():
    with pytest.raises(ValueError, match='unapproved split'):
        approved_train_rows([row(reference='held_out')], {'query'}, {'neutral'})


def test_partial_gate_never_becomes_full_mouth_supervision():
    with pytest.raises(ValueError, match='local observation mask'):
        approved_train_rows([row(event_local_mask_path=None)], {'query'}, {'neutral'})
    selected, _ = approved_train_rows([row()], {'query'}, {'neutral'})
    assert selected[0]['mouth_event_gate'] is False


def test_duplicate_sources_fail_instead_of_overwriting():
    with pytest.raises(ValueError, match='Duplicate'):
        approved_train_rows([row(), row()], {'query'}, {'neutral'})


def test_numeric_manifest_labels_resolve_neutral_anchors():
    names = ['neutral', 'angry']
    for value in (0, '0', 'neutral'):
        assert emotion_name({'emotion': value}, names) == 'neutral'
    assert emotion_name({'emotion': '1'}, names) == 'angry'
    with pytest.raises(ValueError):
        emotion_name({'emotion': True}, names)
