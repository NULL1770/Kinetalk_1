import numpy as np
import pytest
import torch

from scripts.diagnose_intensity_factorial import MODES, factorial_conditions, paired_intervals
from scripts.audit_matched_motion_curves import MC, RC, REGIONS


def test_factorial_changes_only_specified_inputs_and_keeps_unknown_labels():
    audio = {'global': torch.randn(3, 64), 'intensity_value': torch.tensor([[.4], [1.2], [1.9]]),
             'u_a': torch.randn(3, 10, 64), 'intensity_logits': torch.randn(3, 4)}
    teacher = {'global': torch.randn(3, 64), 'intensity_value': torch.tensor([[.8], [2.2], [2.8]])}
    conditions = factorial_conditions(audio, teacher, torch.tensor([0, 2, -1]), torch.tensor([True, True, False]))
    assert tuple(conditions) == MODES and conditions['audio'] is audio
    changes = {'audio': (), 'teacher_scalar': ('intensity_value',), 'label_scalar': ('intensity_value',),
               'teacher_global': ('global',), 'teacher_both': ('global', 'intensity_value')}
    for mode, condition in conditions.items():
        for key in audio:
            if key not in changes[mode]:
                assert condition[key] is audio[key]
    torch.testing.assert_close(conditions['label_scalar']['intensity_value'], torch.tensor([[0.], [2.], [1.9]]))
    assert conditions['teacher_both']['global'] is teacher['global']
    assert conditions['teacher_scalar']['intensity_value'] is teacher['intensity_value']
    torch.testing.assert_close(audio['intensity_value'], torch.tensor([[.4], [1.2], [1.9]]))


def test_factorial_rejects_misaligned_annotation():
    with pytest.raises(ValueError):
        factorial_conditions({'intensity_value': torch.zeros(2, 1)}, {}, torch.zeros(3), torch.ones(3, dtype=torch.bool))


def test_paired_bootstrap_identity_and_known_scalar_global_interaction():
    labels = np.arange(24) % 8
    predictions = {mode: [labels.copy() for _ in range(4)] for mode in MODES}
    metrics = {mode: np.zeros((24, len(MC))) for mode in MODES}
    regions = {mode: np.zeros((24, len(REGIONS), len(RC))) for mode in MODES}
    metrics['teacher_scalar'] += 1
    metrics['teacher_global'] += 2
    metrics['teacher_both'] += 3
    row = paired_intervals(labels, predictions, metrics, regions, 8, draws=20)
    assert all(v == 0 for v in row['teacher_geometry_interaction_mean'].values())
    for mode in MODES[1:]:
        assert row['branches'][mode]['probe_f1_95ci'] == [[0., 0.]] * 4
    assert row['branches']['teacher_scalar']['metrics_95ci']['MBE'] == [1., 1.]
