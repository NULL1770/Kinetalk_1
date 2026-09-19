import copy

import numpy as np
import pytest
import torch

from scripts import audit_event_receiver_bottleneck as audit
from scripts import evaluate_continuous_motion_latent as native


def fixture():
    torch.manual_seed(51)
    n = 23
    valid = torch.ones(n, dtype=torch.bool); valid[8:11] = False
    target = torch.rand(n, 52)*.6+.2
    baseline = torch.rand(n, 52)*.4+.3
    clip = {'clip_id': 'clip', 'sentence': 's1', 'speaker': 1, 'emotion': 2, 'split': 'train',
            'valid': valid, 'motion_mask': valid[:, None].expand(-1, 9).clone(),
            'motion9': target[:, native.UPPER], 'target52': target, 'baseline52': baseline,
            'channel_mask': torch.ones(n, 52, dtype=torch.bool), 'b9': baseline[valid][:, native.UPPER].mean(0),
            'times': torch.arange(n).double()/25}
    return clip, {'metric_scale': torch.ones(9)}


def test_baseline_rescoring_exports_shared_arkit_without_mutation(tmp_path):
    clip, stats = fixture(); original = copy.deepcopy(clip)
    curves = audit.baseline_curves([clip])
    result, arkit = audit.rescore_arm(curves, [clip], stats, tmp_path/'baseline',
                                      deterministic=True, scope='fixture')
    assert result['summary']['clips'] == 1
    assert arkit['summary']['arkit_mbe']['status'] == 'computed'
    assert not result['naturalness_certified'] and not result['default_replaced']
    assert (tmp_path/'baseline'/'arkit_benchmark.json').is_file()
    for key, value in original.items():
        if torch.is_tensor(value): assert torch.equal(clip[key], value)
    assert result['numerics'][0]['other43_exact']


def test_curve_bindings_reject_seed_reference_and_support_changes():
    clip, _ = fixture()
    curves = audit.baseline_curves([clip])
    curves['clip']['samples'] = np.repeat(curves['clip']['samples'], 4, axis=0)
    curves['clip']['seeds'] = list(audit.SEEDS)
    audit.verify_curve_arm(curves, [clip], stochastic=True)
    for key, changed, message in (
        ('seeds', [42, 123, 2026, 78], 'seeds'),
        ('score_mask', np.zeros(23, dtype=bool), 'mask'),
        ('target', np.zeros((23, 9)), 'reference'),
        ('generated_mask', np.zeros(23, dtype=bool), 'support'),
    ):
        bad = copy.deepcopy(curves); bad['clip'][key] = changed
        with pytest.raises(ValueError, match=message):
            audit.verify_curve_arm(bad, [clip], stochastic=True)
