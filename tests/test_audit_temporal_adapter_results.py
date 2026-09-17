"""Compact audit must not silently discard corrupt unobserved upper payloads."""
from pathlib import Path

import pytest
import torch

from scripts.audit_temporal_adapter_results import DEPLOY_KEYS, reconstruct_compact
from scripts import train_prefix_upper as p


def archives():
    valid = torch.ones(405, 96, dtype=torch.bool)
    valid[:, -1] = False
    baseline = torch.zeros(405, 96, 52)
    baseline[:, -1] = -0.0
    metadata = {'clip_id': [str(i) for i in range(405)], 'target': baseline.clone(),
                'valid': valid, 'channel_mask': torch.ones(405, 52, dtype=torch.bool),
                'times': torch.arange(96, dtype=torch.float64)[None].expand(405, -1)*.04,
                'b0': torch.zeros(405, 52)}
    bases = {**metadata, 'predictions': {str(seed)+'/base': baseline for seed in p.r.SEEDS}}
    upper = baseline[..., p.CC].clone()
    upper[valid] = .25
    compact = {**metadata, 'storage_schema': 'upper9_with_exact_bound_base_v1',
               'upper_indices': list(p.CC), 'baseline_sha256': 'bound', 'baseline_path': 'base.pt',
               'nonupper_invalid_exact_before_storage': True, 'noise_seeds': list(p.r.SEEDS),
               'decode_steps': 12, 'upper_predictions9': {key: upper for key in DEPLOY_KEYS}}
    return compact, bases


def test_restoration_preserves_signed_invalid_baseline_and_original_payload():
    compact, bases = archives()
    restored = reconstruct_compact(compact, bases, Path('base.pt'), 'bound', dc=True)
    prediction = restored['predictions']['42/full']
    assert bool(torch.signbit(prediction[:, -1]).all())
    assert bool((prediction[:, :-1, p.CC] == .25).all())
    assert not bool(prediction[..., list(p.r.NOT_UPPER)].count_nonzero())
    assert not bool(bases['predictions']['42/base'].count_nonzero())


def test_invalid_compact_payload_is_checked_before_it_can_be_ignored_by_restoration():
    compact, bases = archives()
    compact['upper_predictions9']['42/full'] = compact['upper_predictions9']['42/full'].clone()
    compact['upper_predictions9']['42/full'][:, -1] = .5
    with pytest.raises(ValueError, match='Stored invalid upper payload'):
        reconstruct_compact(compact, bases, Path('base.pt'), 'bound', dc=True)
